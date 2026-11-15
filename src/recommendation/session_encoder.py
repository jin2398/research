"""GRU session encoder: recent clicks -> short-term interest vector.

The encoder reads the CLIP image vectors of the clicked items (oldest -> newest)
and predicts the vector of the next click, in the same CLIP space as the search
index. Stage 3 re-ranking blends cosine(session vector, item vector) with the
Stage 2 score, which combines short-term interest with long-term preference.

Training: next-click prediction on train-split sessions, InfoNCE loss with the
other targets of the batch as negatives. Item vectors stay frozen.

    python -m src.recommendation.session_encoder train    --config config/session_encoder.yaml
    python -m src.recommendation.session_encoder evaluate --config config/session_encoder.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass(frozen=True, slots=True)
class SessionEncoderConfig:
    events_path: Path
    clip_artifacts_dir: Path
    products_path: Path
    session_column: str
    event_type_column: str
    product_id_column: str
    timestamp_column: str
    item_id_column: str
    category_column: str
    click_events: tuple[str, ...]
    train_ratio: float
    valid_ratio: float
    hidden_size: int
    num_layers: int
    dropout: float
    epochs: int
    batch_size: int
    learning_rate: float
    temperature: float
    window: int
    max_sessions: int | None
    eval_sessions: int
    eval_k: int
    model_path: Path
    report_path: Path
    random_seed: int

    @classmethod
    def from_yaml(cls, path: Path) -> "SessionEncoderConfig":
        import yaml

        v = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(v, dict):
            raise ValueError(f"Expected a YAML mapping in {path}")
        data, cols, model, train = v["data"], v["columns"], v["model"], v["train"]
        config = cls(
            events_path=Path(data["events_path"]),
            clip_artifacts_dir=Path(data["clip_artifacts_dir"]),
            products_path=Path(data["products_path"]),
            session_column=str(cols["session_id"]),
            event_type_column=str(cols["event_type"]),
            product_id_column=str(cols["product_id"]),
            timestamp_column=str(cols["timestamp"]),
            item_id_column=str(cols["item_id"]),
            category_column=str(cols["category"]),
            click_events=tuple(str(e) for e in v["click_events"]),
            train_ratio=float(v["split"]["train"]),
            valid_ratio=float(v["split"]["valid"]),
            hidden_size=int(model["hidden_size"]),
            num_layers=int(model["num_layers"]),
            dropout=float(model["dropout"]),
            epochs=int(train["epochs"]),
            batch_size=int(train["batch_size"]),
            learning_rate=float(train["learning_rate"]),
            temperature=float(train["temperature"]),
            window=int(train["window"]),
            max_sessions=int(train["max_sessions"]) if train.get("max_sessions") else None,
            eval_sessions=int(v["evaluation"]["sessions"]),
            eval_k=int(v["evaluation"]["k"]),
            model_path=Path(v["output"]["model_path"]),
            report_path=Path(v["output"]["report_path"]),
            random_seed=int(v["random_seed"]),
        )
        if not 0 < config.train_ratio < config.train_ratio + config.valid_ratio <= 1:
            raise ValueError("split ratios must satisfy 0 < train < train + valid <= 1")
        if config.window < 1 or config.batch_size < 2 or config.temperature <= 0:
            raise ValueError("window >= 1, batch_size >= 2 and temperature > 0 are required")
        return config


# ---------------------------------------------------------------- model
class SessionGRU(nn.Module):
    """CLIP click vectors [B, T, D] -> predicted next-click vectors [B, T, D] (unit norm)."""

    def __init__(self, dim: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(dim, hidden_size), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.out = nn.Linear(hidden_size, dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(self.inp(x), lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        hidden, _ = pad_packed_sequence(self.gru(packed)[0], batch_first=True,
                                        total_length=x.size(1))
        # residual from the latest click: the GRU learns the shift from it
        return nn.functional.normalize(self.out(hidden) + x, dim=-1)


class SessionEncoder:
    """Inference wrapper used by the re-ranker."""

    def __init__(self, model: SessionGRU, vectors: np.ndarray, rows: Mapping[str, int],
                 window: int) -> None:
        self.model = model.eval()
        self.vectors = vectors
        self.rows = rows
        self.window = window

    @classmethod
    def load(cls, path: Path, vectors: np.ndarray, rows: Mapping[str, int]) -> "SessionEncoder":
        state = torch.load(path, map_location="cpu", weights_only=True)
        hp = state["hparams"]
        if int(hp["dim"]) != vectors.shape[1]:
            raise ValueError(f"Encoder dim {hp['dim']} != item vector dim {vectors.shape[1]}")
        model = SessionGRU(int(hp["dim"]), int(hp["hidden_size"]), int(hp["num_layers"]),
                           float(hp["dropout"]))
        model.load_state_dict(state["model"])
        return cls(model, vectors, rows, int(hp["window"]))

    def encode(self, clicks: Sequence[str]) -> np.ndarray | None:
        """clicks: product ids, oldest -> newest. None when no click has a vector."""
        sequence = to_rows(clicks, self.rows)[-self.window:]
        if not sequence:
            return None
        return self.encode_rows([sequence])[0]

    @torch.no_grad()
    def encode_rows(self, sequences: Sequence[Sequence[int]]) -> np.ndarray:
        """Batch encode non-empty row sequences -> [N, D] unit vectors (last step)."""
        x, lengths = _pad(self.vectors, [list(s)[-self.window:] for s in sequences])
        out = self.model(x, lengths)
        last = out[torch.arange(len(sequences)), lengths - 1]
        return last.numpy().astype("float32")


# ---------------------------------------------------------------- data
def to_rows(clicks: Iterable[str], rows: Mapping[str, int]) -> list[int]:
    """Product ids -> vector rows; unknown ids dropped, consecutive repeats merged
    (view -> cart -> purchase of one item is one interest step)."""
    result: list[int] = []
    for pid in clicks:
        row = rows.get(pid)
        if row is not None and (not result or result[-1] != row):
            result.append(row)
    return result


def load_sessions(config: SessionEncoderConfig, rows: Mapping[str, int]) -> tuple[list[list[int]], list[list[int]]]:
    """Train and valid sessions as vector-row sequences, using the time split on all rows."""
    with config.events_path.open(encoding="utf-8-sig", newline="") as handle:
        events = [(r[config.timestamp_column], r[config.session_column],
                   r[config.event_type_column], r.get(config.product_id_column) or "")
                  for r in csv.DictReader(handle)]
    order = sorted(range(len(events)), key=lambda i: events[i][0])   # stable
    a = int(len(events) * config.train_ratio)
    b = int(len(events) * (config.train_ratio + config.valid_ratio))
    clicks = set(config.click_events)

    def sessions(indices: Sequence[int]) -> list[list[int]]:
        grouped: dict[str, list[str]] = {}
        for i in indices:
            _, sid, etype, pid = events[i]
            if etype in clicks and pid:
                grouped.setdefault(sid, []).append(pid)
        return [s for s in (to_rows(p, rows) for p in grouped.values()) if len(s) >= 2]

    return sessions(order[:a]), sessions(order[a:b])


def _pad(vectors: np.ndarray, sequences: Sequence[Sequence[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    x = torch.zeros(len(sequences), int(lengths.max()), vectors.shape[1])
    for i, seq in enumerate(sequences):
        x[i, :len(seq)] = torch.from_numpy(vectors[list(seq)])
    return x, lengths


def info_nce(pred: torch.Tensor, target_rows: torch.Tensor, target_vecs: torch.Tensor,
             temperature: float) -> torch.Tensor:
    """In-batch softmax; other positions holding the same item are not negatives."""
    logits = pred @ target_vecs.T / temperature
    same = target_rows[:, None] == target_rows[None, :]
    eye = torch.eye(len(target_rows), dtype=torch.bool)
    logits = logits.masked_fill(same & ~eye, float("-inf"))
    return nn.functional.cross_entropy(logits, torch.arange(len(target_rows)))


# ---------------------------------------------------------------- train / evaluate
def train(config: SessionEncoderConfig) -> dict[str, Any]:
    from .rerank import load_clip_vectors

    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    vectors, rows = load_clip_vectors(config.clip_artifacts_dir)
    train_sessions, _ = load_sessions(config, rows)
    rng = random.Random(config.random_seed)
    if config.max_sessions and len(train_sessions) > config.max_sessions:
        train_sessions = rng.sample(train_sessions, config.max_sessions)
    # input = clicks[:-1], target = clicks[1:]; long sessions keep their latest window
    samples = [s[-(config.window + 1):] for s in train_sessions]
    model = SessionGRU(vectors.shape[1], config.hidden_size, config.num_layers, config.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    table = torch.from_numpy(vectors)
    history = []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        rng.shuffle(samples)
        total, steps = 0.0, 0
        for start in range(0, len(samples), config.batch_size):
            batch = samples[start:start + config.batch_size]
            x, lengths = _pad(vectors, [s[:-1] for s in batch])
            pred = model(x, lengths)
            mask = torch.arange(x.size(1))[None, :] < lengths[:, None]
            targets = torch.tensor([r for s in batch for r in s[1:]], dtype=torch.long)
            loss = info_nce(pred[mask], targets, table[targets], config.temperature)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total, steps = total + loss.item(), steps + 1
        history.append({"epoch": epoch, "loss": round(total / max(steps, 1), 4)})
        print(f"[session_encoder] epoch {epoch}/{config.epochs} loss={history[-1]['loss']}", flush=True)
    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    hparams = {"dim": vectors.shape[1], "hidden_size": config.hidden_size,
               "num_layers": config.num_layers, "dropout": config.dropout, "window": config.window}
    torch.save({"model": model.state_dict(), "hparams": hparams}, config.model_path)
    return {"train_sessions": len(samples), "epochs": history, "hparams": hparams,
            "train_seconds": round(time.perf_counter() - started, 1),
            "model_path": str(config.model_path)}


def weighted_average(vectors: np.ndarray, sequence: Sequence[int]) -> np.ndarray:
    """Previous session signal (recency-weighted mean of click vectors), as a baseline."""
    weights = np.arange(1, len(sequence) + 1, dtype="float32")
    vector = (weights[:, None] * vectors[list(sequence)]).sum(axis=0)
    return vector / (np.linalg.norm(vector) or 1.0)


def evaluate(config: SessionEncoderConfig) -> dict[str, Any]:
    """Next click on valid sessions: GRU vs weighted average vs last click.

    item level:     HitRate@k / MRR@k of the exact next item among all CLIP items
    category level: share of the top-k items in the next click's category
                    (the signal the re-ranker uses), and HitRate@k of that category
    """
    from ..evaluation.metrics import hit_rate_at_k, reciprocal_rank_at_k
    from .rerank import load_clip_vectors

    vectors, rows = load_clip_vectors(config.clip_artifacts_dir)
    encoder = SessionEncoder.load(config.model_path, vectors, rows)
    _, valid = load_sessions(config, rows)
    rng = random.Random(config.random_seed)
    sample = rng.sample(valid, min(config.eval_sessions, len(valid)))
    prefixes = [s[:-1][-encoder.window:] for s in sample]
    targets = [s[-1] for s in sample]
    with config.products_path.open(encoding="utf-8-sig", newline="") as handle:
        by_id = {r[config.item_id_column]: r[config.category_column] for r in csv.DictReader(handle)}
    row_category = np.array([by_id.get(pid, "") for pid in sorted(rows, key=rows.get)], dtype=object)
    queries = {
        "gru_session_encoder": encoder.encode_rows(prefixes),
        "weighted_average": np.stack([weighted_average(vectors, p) for p in prefixes]),
        "last_click": vectors[[p[-1] for p in prefixes]],
    }
    k = config.eval_k
    results = {}
    for name, q in queries.items():
        hits, rr, cat_share, cat_hit = [], [], [], []
        for start in range(0, len(q), 256):                 # exact search over all items
            scores = q[start:start + 256] @ vectors.T
            top = np.argpartition(-scores, k, axis=1)[:, :k]
            order = np.take_along_axis(scores, top, 1).argsort(axis=1)[:, ::-1]
            top = np.take_along_axis(top, order, 1)
            for ranked, target in zip(top, targets[start:start + 256]):
                ranked_ids = [int(r) for r in ranked]
                hits.append(hit_rate_at_k(ranked_ids, {target}, k))
                rr.append(reciprocal_rank_at_k(ranked_ids, {target}, k))
                same = row_category[ranked] == row_category[target]
                cat_share.append(float(same.mean()))
                cat_hit.append(float(same.any()))
        results[name] = {f"hit_rate_at_{k}": round(float(np.mean(hits)), 4),
                         f"mrr_at_{k}": round(float(np.mean(rr)), 4),
                         f"category_precision_at_{k}": round(float(np.mean(cat_share)), 4),
                         f"category_hit_rate_at_{k}": round(float(np.mean(cat_hit)), 4)}
    started = time.perf_counter()
    for p in prefixes[:200]:
        encoder.encode_rows([p])
    latency = (time.perf_counter() - started) / min(200, len(prefixes)) * 1000
    return {"task": "next click in valid-split sessions (exact search over all CLIP items)",
            "sessions": len(sample), "k": k, "results": results,
            "encode_ms_mean_single": round(latency, 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["train", "evaluate", "all"])
    parser.add_argument("--config", type=Path, default=Path("config/session_encoder.yaml"))
    args = parser.parse_args()
    config = SessionEncoderConfig.from_yaml(args.config)
    report: dict[str, Any] = {"config": {k: str(v) if isinstance(v, Path) else v
                                         for k, v in asdict(config).items()}}
    if config.report_path.is_file():
        report.update({k: v for k, v in json.loads(config.report_path.read_text("utf-8")).items()
                       if k in ("training", "evaluation")})
    if args.action in ("train", "all"):
        report["training"] = train(config)
    if args.action in ("evaluate", "all"):
        report["evaluation"] = evaluate(config)
        print(json.dumps(report["evaluation"], indent=2), flush=True)
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")


if __name__ == "__main__":
    main()
