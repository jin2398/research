"""Stage 2 Ranking: DeepFM (PyTorch) predicting CTR and CVR of Stage 1 candidates.

Impressions : the Two-Tower top-300 of each user is treated as the exposed list.
              Label CTR = the user clicked (view/cart/purchase) the item in the label
              window, CVR = cart/purchase in the label window. Exposed but not clicked
              = implicit negative (mission protocol for Ranking).
Fields      : User    persona, preferred category, history-length bucket
              Item    all Item Tower fields (category hierarchy, brand, color, price,
                      discount, newness, popularity buckets, new/hot) + log-popularity bucket
              Cross   user share of the item's category / brand / bucket, price ratio
                      vs the user's mean price, item already seen by the user
              Context Two-Tower rank bucket, last-click category match, share of the
                      item's category in the last 5 clicks, days since last activity
Model       : shared field embeddings -> FM (1st + 2nd order) + DNN; two heads
              (CTR, CVR); dense features feed the linear part and the DNN.

Data without leakage:
  train rows : context = train split, candidates from the Two-Tower (trained on train
               only), labels = valid-period clicks
  test rows  : context = train + valid, labels = test-period clicks (reported AUC)

    python -m src.recommendation.deepfm train --config config/stage12.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss  # noqa: F401  (import before torch)
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .stage12_data import (SPLIT_TEST, SPLIT_TRAIN, SPLIT_VALID, EventLog, ItemTable,
                           Stage12Config, UserContext, UserStates, item_popularity, set_seed)
from .two_tower import CandidateGenerator

DEEPFM_FILE = "deepfm.pt"

# fixed bucket edges for cross / context features (documented in docs/stage12_report.md)
SHARE_EDGES = [1e-9, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5]
PRICE_RATIO_EDGES = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]
RANK_EDGES = [5, 10, 20, 50, 100, 200]
DAYS_EDGES = [0.5, 1, 3, 7, 14, 30]
NEVENTS_EDGES = [5, 10, 20, 50, 100, 200]
POP_EDGES = [0, 1, 3, 5, 10, 20, 50, 100]

CONTEXT_FIELDS = ["persona", "preferred", "n_events_b", "cat_share_b", "brand_share_b",
                  "bucket_share_b", "price_ratio_b", "seen", "tt_rank_b", "last_cat_match",
                  "session_cat_b", "days_since_b", "pop_b"]
DENSE_FIELDS = ["tt_score", "log_pop", "cat_share", "brand_share", "bucket_share",
                "session_cat_share", "log_price_ratio", "log_n_events"]


def _bucket(values: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    return np.searchsorted(np.asarray(edges, dtype=np.float64), values, side="right") + 1


@dataclass
class FeatureSpec:
    item_fields: list[str]
    cardinalities: list[int]            # item fields then CONTEXT_FIELDS

    @property
    def fields(self) -> list[str]:
        return [f"item_{f}" for f in self.item_fields] + CONTEXT_FIELDS


class FeatureBuilder:
    """(UserContext, candidate rows, Two-Tower scores) -> sparse codes + dense values.

    Used both for offline training/evaluation and per request in the API, so the
    feature definitions cannot drift apart."""

    def __init__(self, items: ItemTable, popularity: np.ndarray, n_persona: int,
                 n_preferred: int) -> None:
        self.items = items
        self.popularity = popularity
        self.log_pop = np.log1p(popularity).astype(np.float32)
        self.pop_b = _bucket(popularity, POP_EDGES)
        self.cat = items.column("category_small")
        self.brand = items.column("brand")
        self.bucket = items.column("category_bucket")
        ctx_card = [n_persona, n_preferred, len(NEVENTS_EDGES) + 2, *([len(SHARE_EDGES) + 2] * 3),
                    len(PRICE_RATIO_EDGES) + 2, 3, len(RANK_EDGES) + 2, 3, len(SHARE_EDGES) + 2,
                    len(DAYS_EDGES) + 3, len(POP_EDGES) + 2]
        self.spec = FeatureSpec(list(items.fields), items.cardinalities() + ctx_card)

    def build(self, ctx: UserContext, rows: np.ndarray, tt_scores: np.ndarray,
              now_ts: float) -> tuple[np.ndarray, np.ndarray]:
        rows = np.asarray(rows, dtype=np.int64)
        n = len(rows)
        cat, brand, bucket = self.cat[rows], self.brand[rows], self.bucket[rows]
        cat_share = ctx.cat_share[cat]
        brand_share = ctx.brand_share[brand]
        bucket_share = ctx.bucket_share[bucket]
        price = self.items.price[rows]
        ratio = price / ctx.mean_price if ctx.mean_price > 0 else np.ones(n, dtype=np.float32)
        seen = np.isin(rows, ctx.seen).astype(np.int64) + 1
        last5 = ctx.recent_items[-5:]
        if len(last5):
            last_cat = self.cat[last5]
            match = (cat == last_cat[-1]).astype(np.int64) + 1
            session_share = (cat[:, None] == last_cat[None, :]).mean(1)
        else:
            match = np.zeros(n, dtype=np.int64)
            session_share = np.zeros(n)
        days = (now_ts - ctx.last_ts) / 86400 if np.isfinite(ctx.last_ts) else np.nan
        days_b = 0 if not np.isfinite(days) else int(_bucket(np.array([days]), DAYS_EDGES)[0])
        ranks = np.arange(n)
        sparse = np.column_stack([
            self.items.codes[rows],
            np.full(n, ctx.persona), np.full(n, ctx.preferred),
            np.full(n, _bucket(np.array([ctx.n_events]), NEVENTS_EDGES)[0]),
            _bucket(cat_share, SHARE_EDGES), _bucket(brand_share, SHARE_EDGES),
            _bucket(bucket_share, SHARE_EDGES), _bucket(ratio, PRICE_RATIO_EDGES), seen,
            _bucket(ranks, RANK_EDGES), match, _bucket(session_share, SHARE_EDGES),
            np.full(n, days_b + 1 if days_b else 0), self.pop_b[rows],
        ]).astype(np.int64)
        dense = np.column_stack([
            tt_scores, self.log_pop[rows] / 5.0, cat_share, brand_share, bucket_share,
            session_share, np.log(np.clip(ratio, 1e-3, None)),
            np.full(n, np.log1p(ctx.n_events) / 5.0),
        ]).astype(np.float32)
        return sparse, dense


# ---------------------------------------------------------------- model
class DeepFM(nn.Module):
    def __init__(self, cardinalities: Sequence[int], n_dense: int, p: dict[str, Any]) -> None:
        super().__init__()
        k = int(p["embedding_dim"])
        offsets = np.concatenate([[0], np.cumsum(cardinalities)[:-1]])
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long), persistent=False)
        total = int(sum(cardinalities))
        self.first = nn.Embedding(total, 1)            # FM 1st order
        self.emb = nn.Embedding(total, k)              # FM 2nd order + DNN, shared
        nn.init.normal_(self.emb.weight, std=0.01)
        nn.init.zeros_(self.first.weight)
        self.dense_linear = nn.Linear(n_dense, 2)
        n_fields = len(cardinalities)
        layers: list[nn.Module] = []
        sizes = [n_fields * k + n_dense, *p["hidden"]]
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(float(p["dropout"]))]
        self.dnn = nn.Sequential(*layers)
        self.heads = nn.Linear(sizes[-1], 2)           # [CTR, CVR]
        self.bias = nn.Parameter(torch.zeros(2))

    def forward(self, sparse: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        idx = sparse + self.offsets
        first = self.first(idx).sum(dim=(1, 2))                         # [B]
        e = self.emb(idx)                                               # [B, F, k]
        fm = 0.5 * ((e.sum(1) ** 2) - (e ** 2).sum(1)).sum(1)           # [B]
        deep = self.heads(self.dnn(torch.cat([e.flatten(1), dense], dim=1)))   # [B, 2]
        shared = (first + fm).unsqueeze(1)
        return shared + deep + self.dense_linear(dense) + self.bias    # logits [B, 2]


def fast_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC by rank statistic, ties averaged (same definition as metrics.auc)."""
    from scipy.stats import rankdata

    labels = np.asarray(labels).astype(bool)
    pos, neg = labels.sum(), (~labels).sum()
    if not pos or not neg:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


# ---------------------------------------------------------------- impressions
@dataclass
class Impressions:
    sparse: np.ndarray
    dense: np.ndarray
    click: np.ndarray
    conv: np.ndarray
    user: np.ndarray            # user index per row
    rows: np.ndarray            # item row per row
    tt_score: np.ndarray
    pop: np.ndarray
    recent_source: np.ndarray   # True = added from the user's recent clicks (not FAISS)


def build_impressions(gen: CandidateGenerator, fb: FeatureBuilder, states: UserStates,
                      log: EventLog, users: Sequence[int], label_split: int, now_ts: float,
                      k: int, n_recent: int = 0, batch: int = 512) -> Impressions:
    clicks = log.truth(label_split)
    convs = log.truth(label_split, ["cart", "purchase"])
    parts: dict[str, list[np.ndarray]] = {key: [] for key in Impressions.__dataclass_fields__}
    for s in range(0, len(users), batch):
        chunk = users[s:s + batch]
        ctxs = [states.context(str(log.user_ids[u])) for u in chunk]
        for u, ctx, (rw, sc, src) in zip(chunk, ctxs, gen.search_pool(ctxs, k, n_recent)):
            sp, de = fb.build(ctx, rw, sc, now_ts)
            parts["sparse"].append(sp)
            parts["dense"].append(de)
            parts["click"].append(np.isin(rw, clicks.get(u, ())).astype(np.float32))
            parts["conv"].append(np.isin(rw, convs.get(u, ())).astype(np.float32))
            parts["user"].append(np.full(len(rw), u))
            parts["rows"].append(rw)
            parts["tt_score"].append(sc)
            parts["pop"].append(fb.popularity[rw])
            parts["recent_source"].append(src)
    return Impressions(**{key: np.concatenate(v) for key, v in parts.items()})


# ---------------------------------------------------------------- ranker (serving)
class Ranker:
    def __init__(self, model: DeepFM, builder: FeatureBuilder, blend: dict[str, float]) -> None:
        self.model, self.builder = model.eval(), builder
        self.blend = blend

    @classmethod
    def load(cls, config: Stage12Config, items: ItemTable, popularity: np.ndarray) -> "Ranker":
        ckpt = torch.load(config.output_dir / DEEPFM_FILE, map_location="cpu", weights_only=False)
        fb = FeatureBuilder(items, popularity, ckpt["n_persona"], ckpt["n_preferred"])
        model = DeepFM(fb.spec.cardinalities, len(DENSE_FIELDS), ckpt["params"])
        model.load_state_dict(ckpt["state"])
        return cls(model, fb, dict(config.deepfm["score_blend"]))

    @torch.no_grad()
    def predict(self, sparse: np.ndarray, dense: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logits = self.model(torch.from_numpy(sparse), torch.from_numpy(dense))
        prob = torch.sigmoid(logits).numpy()
        return prob[:, 0], prob[:, 1]

    def score(self, ctr: np.ndarray, cvr: np.ndarray) -> np.ndarray:
        return self.blend.get("ctr", 1.0) * ctr + self.blend.get("cvr", 0.0) * cvr

    def rank(self, ctx: UserContext, rows: np.ndarray, tt_scores: np.ndarray,
             now_ts: float) -> list[tuple[int, float, float, float]]:
        """-> [(item row, score, p_ctr, p_cvr)] best first."""
        sparse, dense = self.builder.build(ctx, rows, tt_scores, now_ts)
        ctr, cvr = self.predict(sparse, dense)
        score = self.score(ctr, cvr)
        order = np.argsort(-score, kind="stable")
        return [(int(rows[i]), float(score[i]), float(ctr[i]), float(cvr[i])) for i in order]


# ---------------------------------------------------------------- training
def _predict_all(model: DeepFM, imp: Impressions, batch: int = 65536) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(imp.click), batch):
            out.append(torch.sigmoid(model(torch.from_numpy(imp.sparse[s:s + batch]),
                                           torch.from_numpy(imp.dense[s:s + batch]))).numpy())
    return np.vstack(out)


def _subset(imp: Impressions, mask: np.ndarray) -> Impressions:
    return Impressions(**{k: getattr(imp, k)[mask] for k in Impressions.__dataclass_fields__})


def train(config: Stage12Config) -> dict[str, Any]:
    started = time.perf_counter()
    p = config.deepfm
    set_seed(config.seed)
    items = ItemTable.load(config.output_dir / "encoders.json")
    log = EventLog.load(config, items)
    gen = CandidateGenerator.load(config, items)
    weights = config.event_weights
    history = gen.history
    k = int(p["candidates_per_user"])
    n_persona, n_preferred = len(log.persona_vocab) + 1, len(log.preferred_vocab) + 1

    def end_ts(split: int) -> float:
        return float(log.ts[log.split == split].max())

    # ranking train/valid: snapshot at the end of train, labels from the valid period
    train_mask = log.mask(SPLIT_TRAIN)
    states_tr = UserStates.build(log, items, train_mask, history, weights, "train")
    fb_tr = FeatureBuilder(items, item_popularity(log, train_mask, items.n_items, weights),
                           n_persona, n_preferred)
    users_tr = sorted(log.truth(SPLIT_VALID))
    imp = build_impressions(gen, fb_tr, states_tr, log, users_tr, SPLIT_VALID, end_ts(SPLIT_TRAIN), k,
                            int(p.get("recent_candidates", 0)))
    rng = np.random.default_rng(config.seed)
    hold = set(rng.choice(users_tr, int(len(users_tr) * float(p["valid_user_fraction"])), replace=False))
    is_hold = np.isin(imp.user, list(hold))
    tr, va = _subset(imp, ~is_hold), _subset(imp, is_hold)
    print(f"[deepfm] impressions train {len(tr.click)} (CTR {tr.click.mean():.4f}, "
          f"CVR {tr.conv.mean():.5f}), valid {len(va.click)} "
          f"({time.perf_counter() - started:.0f}s)", flush=True)

    model = DeepFM(fb_tr.spec.cardinalities, len(DENSE_FIELDS), p)
    opt = torch.optim.AdamW(model.parameters(), lr=float(p["learning_rate"]),
                            weight_decay=float(p["weight_decay"]))
    bs, n = int(p["batch_size"]), len(tr.click)
    sparse, dense = torch.from_numpy(tr.sparse), torch.from_numpy(tr.dense)
    labels = torch.from_numpy(np.column_stack([tr.click, tr.conv]))
    cvr_w = float(p["cvr_weight"])
    gen_t = torch.Generator().manual_seed(config.seed)
    best = (-1.0, None, -1)
    history_log = []
    for epoch in range(1, int(p["epochs"]) + 1):
        model.train()
        perm = torch.randperm(n, generator=gen_t)
        total, steps, t0 = 0.0, 0, time.perf_counter()
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            logits = model(sparse[idx], dense[idx])
            y = labels[idx]
            loss = (F.binary_cross_entropy_with_logits(logits[:, 0], y[:, 0])
                    + cvr_w * F.binary_cross_entropy_with_logits(logits[:, 1], y[:, 1]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            steps += 1
        prob = _predict_all(model, va)
        auc_ctr = fast_auc(va.click, prob[:, 0])
        auc_cvr = fast_auc(va.conv, prob[:, 1])
        history_log.append({"epoch": epoch, "loss": round(total / steps, 5),
                            "valid_auc_ctr": round(auc_ctr, 4), "valid_auc_cvr": round(auc_cvr, 4),
                            "seconds": round(time.perf_counter() - t0, 1)})
        print(f"[deepfm] epoch {epoch} loss {total / steps:.5f} valid AUC ctr {auc_ctr:.4f} "
              f"cvr {auc_cvr:.4f} ({time.perf_counter() - t0:.0f}s)", flush=True)
        if auc_ctr > best[0]:
            best = (auc_ctr, {key: v.detach().clone() for key, v in model.state_dict().items()}, epoch)
    model.load_state_dict(best[1])
    out = config.output_dir
    torch.save({"state": model.state_dict(), "params": p, "n_persona": n_persona,
                "n_preferred": n_preferred, "fields": fb_tr.spec.fields,
                "dense_fields": DENSE_FIELDS}, out / DEEPFM_FILE)
    report = {
        "params": p, "seed": config.seed,
        "train_data": {"context": "train split", "labels": "valid-period clicks",
                       "impressions": int(len(tr.click)), "ctr": round(float(tr.click.mean()), 5),
                       "cvr": round(float(tr.conv.mean()), 6), "users": len(users_tr) - len(hold)},
        "holdout_users": len(hold), "epochs": history_log, "best_epoch": best[2],
        "fields": fb_tr.spec.fields, "dense_fields": DENSE_FIELDS,
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    (out / "deepfm_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(f"[deepfm] best epoch {best[2]} valid AUC {best[0]:.4f}; saved {out / DEEPFM_FILE}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["train"])
    parser.add_argument("--config", type=Path, default=Path("config/stage12.yaml"))
    args = parser.parse_args()
    train(Stage12Config.from_yaml(args.config))


if __name__ == "__main__":
    main()
