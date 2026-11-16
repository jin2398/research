"""Stage 1 Candidate Generation: PyTorch Two-Tower + FAISS.

User Tower : profile (persona, preferred category, history length) + recent click
             sequence (item feature embeddings, event-type weighted mean pooling,
             full window and last few clicks)            -> 64-d unit vector
Item Tower : category hierarchy, brand, color, product group, price / discount /
             newness / popularity buckets, new/hot flags (+ optional item id)
                                                         -> 64-d unit vector
Score      : dot product. Item vectors are computed offline for the whole catalog
             and stored in a FAISS index; online only the User Tower runs, then
             FAISS returns the top 300 items.

Training   : each first interaction of a user with an item in the train split is a
             positive; the user's previous clicks are the history. Negatives follow
             the mission protocol (random negatives 1:4), optionally plus in-batch
             negatives; sampled softmax with temperature.

    python -m src.recommendation.two_tower train --config config/stage12.yaml
    python -m src.recommendation.two_tower index --config config/stage12.yaml   # rebuild FAISS only
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .stage12_data import (
    SPLIT_TRAIN,
    SPLIT_VALID,
    EventLog,
    ItemTable,
    Stage12Config,
    UserContext,
    UserStates,
    _groups,
    item_popularity,
    load_or_build_items,
    set_seed,
)

TT_MODEL_FILE = "two_tower.pt"
ITEM_EMB_FILE = "item_embeddings.npy"
INDEX_FILE = "items.index"


# ---------------------------------------------------------------- model
def _mlp(sizes: Sequence[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 2):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    return nn.Sequential(*layers)


class FieldEmbedding(nn.Module):
    """Looks up item rows -> concatenated per-field embeddings (row n_items = padding)."""

    def __init__(
        self, item_codes: np.ndarray, cardinalities: Sequence[int], dim: int
    ) -> None:
        super().__init__()
        pad = np.zeros((1, item_codes.shape[1]), dtype=np.int64)
        self.register_buffer(
            "codes", torch.from_numpy(np.vstack([item_codes, pad])), persistent=False
        )
        self.embs = nn.ModuleList(nn.Embedding(c, dim) for c in cardinalities)
        self.out_dim = dim * len(cardinalities)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        codes = self.codes[rows]
        return torch.cat(
            [emb(codes[..., j]) for j, emb in enumerate(self.embs)], dim=-1
        )


class ItemTower(nn.Module):
    def __init__(
        self, item_codes: np.ndarray, cardinalities: Sequence[int], p: dict[str, Any]
    ) -> None:
        super().__init__()
        dim = int(p["embedding_dim"])
        self.fields = FieldEmbedding(item_codes, cardinalities, dim)
        self.n_items = item_codes.shape[0]
        self.item_id = nn.Embedding(self.n_items + 1, dim) if p["use_item_id"] else None
        in_dim = self.fields.out_dim + (dim if self.item_id is not None else 0)
        self.mlp = _mlp(
            [in_dim, *p["hidden"], int(p["output_dim"])], float(p["dropout"])
        )

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        x = self.fields(rows)
        if self.item_id is not None:
            x = torch.cat([x, self.item_id(rows)], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)


class UserTower(nn.Module):
    def __init__(
        self,
        item_codes: np.ndarray,
        cardinalities: Sequence[int],
        n_persona: int,
        n_preferred: int,
        n_users: int,
        n_etypes: int,
        etype_weights: Sequence[float],
        profile_dim: int,
        p: dict[str, Any],
    ) -> None:
        super().__init__()
        dim = int(p["embedding_dim"])
        self.recent_length = int(p["recent_length"])
        self.history = FieldEmbedding(item_codes, cardinalities, dim)  # own params
        self.pad = item_codes.shape[0]
        self.etype = nn.Embedding(n_etypes, self.history.out_dim)
        self.register_buffer(
            "etype_w",
            torch.tensor(list(etype_weights), dtype=torch.float32),
            persistent=False,
        )
        self.persona = nn.Embedding(n_persona, dim)
        self.preferred = nn.Embedding(n_preferred, dim)
        self.n_users = n_users
        self.user_id = (
            nn.Embedding(n_users + 1, dim) if p["use_user_id"] else None
        )  # row n_users = unknown
        self.profile_dim = profile_dim
        in_dim = (
            2 * self.history.out_dim
            + 2 * dim
            + 1
            + profile_dim
            + (dim if self.user_id is not None else 0)
        )
        self.mlp = _mlp(
            [in_dim, *p["hidden"], int(p["output_dim"])], float(p["dropout"])
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        hist, et = batch["hist"], batch["hist_et"]  # [B, H] (pad = n_items)
        mask = (hist != self.pad).float()
        x = self.history(hist) + self.etype(et)  # [B, H, D]
        w = mask * self.etype_w[et]
        pooled = (x * w.unsqueeze(-1)).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-6)
        rw = w.clone()
        rw[:, : -self.recent_length] = 0
        recent = (x * rw.unsqueeze(-1)).sum(1) / rw.sum(1, keepdim=True).clamp_min(1e-6)
        parts = [
            pooled,
            recent,
            self.persona(batch["persona"]),
            self.preferred(batch["preferred"]),
            torch.log1p(batch["n_events"].float()).unsqueeze(-1) / 5.0,
        ]
        if self.profile_dim:
            parts.append(batch["profile"].float())
        if self.user_id is not None:
            uid = batch["user"]
            parts.append(
                self.user_id(
                    torch.where(uid < 0, torch.full_like(uid, self.n_users), uid)
                )
            )
        return F.normalize(self.mlp(torch.cat(parts, dim=-1)), dim=-1)


class TwoTower(nn.Module):
    def __init__(
        self, items: ItemTable, log_meta: dict[str, Any], p: dict[str, Any]
    ) -> None:
        super().__init__()
        self.meta = log_meta
        card = items.cardinalities()
        self.item_tower = ItemTower(items.codes, card, p)
        self.user_tower = UserTower(
            items.codes,
            card,
            len(log_meta["persona_vocab"]) + 1,
            len(log_meta["preferred_vocab"]) + 1,
            log_meta["n_users"],
            len(log_meta["event_types"]),
            log_meta["event_weights"],
            profile_dim(items) if p.get("use_profile", True) else 0,
            p,
        )
        self.temperature = float(p["temperature"])

    @torch.no_grad()
    def item_embeddings(self, batch_size: int = 8192) -> np.ndarray:
        self.eval()
        n = self.item_tower.n_items
        out = [
            self.item_tower(torch.arange(s, min(s + batch_size, n))).numpy()
            for s in range(0, n, batch_size)
        ]
        return np.ascontiguousarray(np.vstack(out), dtype=np.float32)

    @torch.no_grad()
    def user_embeddings(self, batch: dict[str, torch.Tensor]) -> np.ndarray:
        self.eval()
        return np.ascontiguousarray(self.user_tower(batch).numpy(), dtype=np.float32)


PROFILE_FIELDS = ("category_small", "brand", "category_bucket")


def profile_dim(items: ItemTable) -> int:
    """Long-term profile = weighted share of the whole history per category_small /
    brand / category_bucket + log(mean price / catalog median price)."""
    card = dict(zip(items.fields, items.cardinalities()))
    return sum(card[f] for f in PROFILE_FIELDS) + 1


def profile_vectors(contexts: Sequence[UserContext], price_ref: float) -> np.ndarray:
    out = []
    for c in contexts:
        price = np.log(c.mean_price / price_ref) if c.mean_price > 0 else 0.0
        out.append(
            np.concatenate([c.cat_share, c.brand_share, c.bucket_share, [price]])
        )
    return np.asarray(out, dtype=np.float32)


def log_meta(log: EventLog, weights: dict[str, float]) -> dict[str, Any]:
    return {
        "persona_vocab": log.persona_vocab,
        "preferred_vocab": log.preferred_vocab,
        "n_users": log.n_users,
        "event_types": log.event_types,
        "event_weights": [weights.get(e, 1.0) for e in log.event_types],
    }


# ---------------------------------------------------------------- user batches
def contexts_to_batch(
    contexts: Sequence[UserContext], history: int, pad: int, price_ref: float
) -> dict[str, torch.Tensor]:
    """UserContext list -> User Tower input (left-padded history)."""
    n = len(contexts)
    hist = np.full((n, history), pad, dtype=np.int64)
    het = np.zeros((n, history), dtype=np.int64)
    for i, c in enumerate(contexts):
        items, ets = c.recent_items[-history:], c.recent_etypes[-history:]
        if len(items):
            hist[i, history - len(items) :], het[i, history - len(items) :] = items, ets
    return {
        "hist": torch.from_numpy(hist),
        "hist_et": torch.from_numpy(het),
        "persona": torch.tensor([c.persona for c in contexts]),
        "preferred": torch.tensor([c.preferred for c in contexts]),
        "n_events": torch.tensor([c.n_events for c in contexts]),
        "user": torch.tensor([c.user_index for c in contexts]),  # -1 = unknown user
        "profile": torch.from_numpy(profile_vectors(contexts, price_ref)),
    }


def training_examples(
    log: EventLog,
    items: ItemTable,
    mask: np.ndarray,
    history: int,
    first_only: bool,
    weights: dict[str, float],
    price_ref: float,
) -> dict[str, np.ndarray]:
    """One row per target click: the previous `history` clicks of that user (left
    padded) and the long-term profile of everything clicked *before* the target."""
    u, it, et = log.user[mask], log.item[mask], log.etype[mask]
    pad = items.n_items
    w_all = np.array([weights.get(e, 1.0) for e in log.event_types], dtype=np.float64)[
        et
    ]
    card = dict(zip(items.fields, items.cardinalities()))
    cols = [(items.column(f), card[f]) for f in PROFILE_FIELDS]
    order = np.argsort(u, kind="stable")
    hists, hets, targets, users, counts, profiles, anchors, ends = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for user, start, stop in _groups(u[order]):
        rows = order[start:stop]
        seq = np.concatenate([np.full(history, pad), it[rows]])
        eseq = np.concatenate([np.zeros(history, dtype=np.int64), et[rows]])
        win = np.lib.stride_tricks.sliding_window_view(seq, history)[: len(rows)]
        ewin = np.lib.stride_tricks.sliding_window_view(eseq, history)[: len(rows)]
        pos = np.arange(len(rows))
        if first_only:
            _, first = np.unique(it[rows], return_index=True)
            pos = np.sort(first)
        w = w_all[rows]
        blocks = []
        for col, size in cols:
            onehot = np.zeros((len(rows), size))
            onehot[np.arange(len(rows)), col[it[rows]]] = w
            before = np.cumsum(onehot, axis=0)[pos] - onehot[pos]  # exclusive of target
            total = before.sum(1, keepdims=True)
            blocks.append(
                np.divide(before, total, out=np.zeros_like(before), where=total > 0)
            )
        wsum = np.cumsum(w)[pos] - w[pos]
        psum = (
            np.cumsum(w * items.price[it[rows]])[pos] - (w * items.price[it[rows]])[pos]
        )
        price = np.where(
            wsum > 0,
            np.log(np.maximum(psum, 1e-9) / np.maximum(wsum, 1e-9) / price_ref),
            0.0,
        )
        profiles.append(np.hstack(blocks + [price[:, None]]).astype(np.float16))
        hists.append(win[pos])
        hets.append(ewin[pos])
        targets.append(it[rows][pos])
        users.append(np.full(len(pos), user))
        counts.append(pos)
        anchors.append(start + pos)  # index into `seq` (all users, time order)
        ends.append(np.full(len(pos), stop))
    return {
        "hist": np.vstack(hists),
        "hist_et": np.vstack(hets),
        "target": np.concatenate(targets),
        "user": np.concatenate(users),
        "n_events": np.concatenate(counts),
        "profile": np.vstack(profiles),
        "anchor": np.concatenate(anchors),
        "end": np.concatenate(ends),
        "seq": it[order],
    }


# ---------------------------------------------------------------- evaluation helpers
def recall_at_k(top: np.ndarray, truth: Sequence[np.ndarray]) -> float:
    """Mean per-user recall: |top-k ∩ truth| / |truth| (same as metrics.recall_at_k)."""
    return float(np.mean([np.isin(t, row).mean() for row, t in zip(top, truth)]))


def exact_top_k(user_vecs: np.ndarray, item_vecs: np.ndarray, k: int) -> np.ndarray:
    scores = user_vecs @ item_vecs.T
    top = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, top, 1), axis=1)
    return np.take_along_axis(top, order, 1)


# ---------------------------------------------------------------- FAISS
def build_index(vectors: np.ndarray, p: dict[str, Any]) -> faiss.Index:
    d = vectors.shape[1]
    kind = p["index_type"]
    if kind == "flat":
        index = faiss.IndexFlatIP(d)
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(d, int(p["hnsw_m"]), faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = int(p["hnsw_ef_construction"])
        index.hnsw.efSearch = int(p["hnsw_ef_search"])
    elif kind == "ivfpq":
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(
            quantizer,
            d,
            int(p["ivf_nlist"]),
            int(p["pq_m"]),
            8,
            faiss.METRIC_INNER_PRODUCT,
        )
        index.train(vectors)
        index.nprobe = int(p["ivf_nprobe"])
    else:
        raise ValueError(f"unknown faiss.index_type {kind}")
    index.add(vectors)
    return index


def load_index(path: Path, p: dict[str, Any]) -> faiss.Index:
    index = faiss.read_index(str(path))
    if p["index_type"] == "hnsw":
        index.hnsw.efSearch = int(p["hnsw_ef_search"])
    elif p["index_type"] == "ivfpq":
        index.nprobe = int(p["ivf_nprobe"])
    return index


# ---------------------------------------------------------------- serving
class CandidateGenerator:
    """Online Stage 1: User Tower forward + FAISS top-k."""

    def __init__(
        self,
        model: TwoTower,
        index: faiss.Index,
        items: ItemTable,
        history: int,
        top_k: int,
    ) -> None:
        self.model, self.index, self.items = model.eval(), index, items
        self.history, self.top_k = history, top_k
        self.pad = items.n_items
        self.price_ref = float(model.meta["price_ref"])
        self.item_vectors: np.ndarray | None = (
            None  # set by load(); scores extra candidates
        )

    @classmethod
    def load(
        cls, config: Stage12Config, items: ItemTable | None = None
    ) -> CandidateGenerator:
        out = config.output_dir
        items = items or ItemTable.load(out / "encoders.json")
        ckpt = torch.load(out / TT_MODEL_FILE, map_location="cpu", weights_only=False)
        model = TwoTower(items, ckpt["meta"], ckpt["params"])
        model.load_state_dict(ckpt["state"])
        index = load_index(out / INDEX_FILE, config.faiss)
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        gen = cls(
            model,
            index,
            items,
            int(ckpt["params"]["history_length"]),
            int(config.faiss["top_k"]),
        )
        gen.item_vectors = np.load(out / ITEM_EMB_FILE)
        return gen

    def user_vectors(self, contexts: Sequence[UserContext]) -> np.ndarray:
        return self.model.user_embeddings(
            contexts_to_batch(contexts, self.history, self.pad, self.price_ref)
        )

    def search(
        self, contexts: Sequence[UserContext], k: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """-> (scores [n, k], item rows [n, k]), best first."""
        scores, rows = self.index.search(self.user_vectors(contexts), k or self.top_k)
        return scores, rows

    def candidates(
        self, context: UserContext, k: int | None = None
    ) -> list[tuple[int, float]]:
        scores, rows = self.search([context], k)
        return [(int(r), float(s)) for r, s in zip(rows[0], scores[0]) if r >= 0]

    def search_pool(
        self, contexts: Sequence[UserContext], k: int | None = None, n_recent: int = 0
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Ranking candidate pool per user: Two-Tower/FAISS top-k, plus up to `n_recent`
        of the user's most recently clicked items that FAISS did not return (re-views
        are frequent in the log). -> [(rows, two_tower_scores, is_recent_source)]"""
        user_vecs = self.user_vectors(contexts)
        scores, rows = self.index.search(user_vecs, k or self.top_k)
        pools = []
        for vec, ctx, sc, rw in zip(user_vecs, contexts, scores, rows):
            ok = rw >= 0
            sc, rw = sc[ok], rw[ok]
            extra = np.empty(0, dtype=np.int64)
            if n_recent and len(ctx.recent_items):
                newest_first = ctx.recent_items[::-1]
                _, first = np.unique(newest_first, return_index=True)
                distinct = newest_first[np.sort(first)]
                extra = distinct[~np.isin(distinct, rw)][:n_recent]
            if len(extra):
                vectors = (
                    self.item_vectors
                    if self.item_vectors is not None
                    else self.model.item_embeddings()
                )
                rw = np.concatenate([rw, extra])
                sc = np.concatenate([sc, (vectors[extra] @ vec).astype(np.float32)])
            src = np.zeros(len(rw), dtype=bool)
            src[len(rw) - len(extra) :] = True
            pools.append((rw, sc, src))
        return pools


# ---------------------------------------------------------------- training
def train(config: Stage12Config) -> dict[str, Any]:
    started = time.perf_counter()
    p = config.two_tower
    set_seed(config.seed)
    items = load_or_build_items(config)
    log = EventLog.load(config, items)
    weights = config.event_weights
    history, pad = int(p["history_length"]), items.n_items
    print(
        f"[two_tower] events {log.rows}, items {items.n_items}, users {log.n_users} "
        f"({time.perf_counter() - started:.1f}s)",
        flush=True,
    )

    price_ref = float(np.median(items.price))
    ex = training_examples(
        log,
        items,
        log.mask(SPLIT_TRAIN),
        history,
        bool(p["first_interaction_only"]),
        weights,
        price_ref,
    )
    n = len(ex["target"])
    # in-batch negatives are drawn ~ target frequency; logQ correction turns them into
    # (approximately) uniform negatives so item popularity is not pushed down
    freq = np.bincount(ex["target"], minlength=items.n_items) + 1.0
    log_qn = torch.from_numpy(
        np.log(freq / freq.sum() * items.n_items).astype(np.float32)
    )
    print(f"[two_tower] training targets {n}", flush=True)

    # validation: context = train snapshot, truth = items touched in the valid period
    train_states = UserStates.build(
        log, items, log.mask(SPLIT_TRAIN), history, weights, "train"
    )
    valid_truth = log.truth(SPLIT_VALID)
    rng = np.random.default_rng(config.seed)
    vusers = sorted(valid_truth)
    vusers = list(rng.choice(vusers, size=min(3000, len(vusers)), replace=False))
    vctx = [train_states.context(str(log.user_ids[u])) for u in vusers]
    vtruth = [valid_truth[u] for u in vusers]
    k = int(config.faiss["top_k"])

    model = TwoTower(items, {**log_meta(log, weights), "price_ref": price_ref}, p)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(p["learning_rate"]),
        weight_decay=float(p["weight_decay"]),
    )
    n_neg, bs = int(p["num_negatives"]), int(p["batch_size"])
    epochs = int(p["epochs"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=float(p["learning_rate"]), total_steps=epochs * ((n + bs - 1) // bs)
    )
    tensors = {key: torch.from_numpy(val) for key, val in ex.items()}
    horizon = int(p.get("target_horizon", 1))
    gen = torch.Generator().manual_seed(config.seed)
    history_log: list[dict[str, Any]] = []
    best = (-1.0, None, -1)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        total, steps, t0 = 0.0, 0, time.perf_counter()
        for s in range(0, n, bs):
            idx = perm[s : s + bs]
            batch = {
                "hist": tensors["hist"][idx],
                "hist_et": tensors["hist_et"][idx],
                "persona": torch.from_numpy(log.persona)[tensors["user"][idx]],
                "preferred": torch.from_numpy(log.preferred)[tensors["user"][idx]],
                "n_events": tensors["n_events"][idx],
                "user": tensors["user"][idx],
                "profile": tensors["profile"][idx],
            }
            if (
                horizon > 1
            ):  # target = one of the next `horizon` clicks (incl. the next one)
                ahead = tensors["anchor"][idx] + torch.randint(
                    0, horizon, (len(idx),), generator=gen
                )
                pos = tensors["seq"][torch.minimum(ahead, tensors["end"][idx] - 1)]
            else:
                pos = tensors["target"][idx]
            u = model.user_tower(batch)  # [B, d]
            pv = model.item_tower(pos)  # [B, d]
            neg = torch.randint(0, items.n_items, (len(idx), n_neg), generator=gen)
            nv = model.item_tower(neg)  # [B, n_neg, d]
            pos_logit = (u * pv).sum(-1, keepdim=True)
            neg_logit = torch.einsum("bd,bnd->bn", u, nv)
            neg_logit = neg_logit.masked_fill(neg == pos.unsqueeze(1), -1e4)
            parts = [pos_logit, neg_logit]
            if p["in_batch_negatives"]:
                inb = u @ pv.T
                same = pos.unsqueeze(0) == pos.unsqueeze(
                    1
                )  # duplicates of the positive
                inb = inb.masked_fill(same, -1e4)
                parts.append(inb)
            logits = torch.cat(parts, dim=1) / model.temperature
            if p["in_batch_negatives"] and p.get("in_batch_logq", True):
                logits[:, 1 + n_neg :] -= log_qn[pos].unsqueeze(0)
            loss = F.cross_entropy(logits, torch.zeros(len(idx), dtype=torch.long))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach())
            steps += 1
        item_vecs = model.item_embeddings()
        top = exact_top_k(
            model.user_embeddings(contexts_to_batch(vctx, history, pad, price_ref)),
            item_vecs,
            k,
        )
        rec = recall_at_k(top, vtruth)
        history_log.append(
            {
                "epoch": epoch,
                "loss": round(total / steps, 4),
                f"valid_recall_at_{k}": round(rec, 4),
                "seconds": round(time.perf_counter() - t0, 1),
            }
        )
        print(
            f"[two_tower] epoch {epoch} loss {total / steps:.4f} valid Recall@{k} {rec:.4f} "
            f"({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )
        if rec > best[0]:
            best = (
                rec,
                {key: v.detach().clone() for key, v in model.state_dict().items()},
                epoch,
            )
    model.load_state_dict(best[1])

    out = config.output_dir
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state": model.state_dict(), "params": p, "meta": model.meta},
        out / TT_MODEL_FILE,
    )
    vectors = model.item_embeddings()
    np.save(out / ITEM_EMB_FILE, vectors)
    index_info = write_index(config, vectors)

    pop = item_popularity(log, log.mask(SPLIT_TRAIN), items.n_items, weights)
    pop_top = np.argsort(-pop, kind="stable")[:k]
    report = {
        "params": p,
        "seed": config.seed,
        "training_targets": n,
        "split_rows": log.rows,
        "split_times": log.split_times,
        "epochs": history_log,
        "best_epoch": best[2],
        "valid": {
            "users": len(vusers),
            "context": "train split",
            "truth": "valid-period clicks",
            f"two_tower_recall_at_{k}": round(best[0], 4),
            f"popularity_recall_at_{k}": round(
                recall_at_k(np.tile(pop_top, (len(vusers), 1)), vtruth), 4
            ),
        },
        "index": index_info,
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    (out / "two_tower_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[two_tower] best epoch {best[2]} valid Recall@{k} {best[0]:.4f}; saved to {out}",
        flush=True,
    )
    return report


def write_index(
    config: Stage12Config, vectors: np.ndarray | None = None
) -> dict[str, Any]:
    out = config.output_dir
    vectors = vectors if vectors is not None else np.load(out / ITEM_EMB_FILE)
    t0 = time.perf_counter()
    index = build_index(vectors, config.faiss)
    faiss.write_index(index, str(out / INDEX_FILE))
    return {
        "type": config.faiss["index_type"],
        "vectors": int(index.ntotal),
        "dim": int(vectors.shape[1]),
        "build_s": round(time.perf_counter() - t0, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["train", "index"])
    parser.add_argument("--config", type=Path, default=Path("config/stage12.yaml"))
    args = parser.parse_args()
    config = Stage12Config.from_yaml(args.config)
    if args.command == "train":
        train(config)
    else:
        print(write_index(config))


if __name__ == "__main__":
    main()
