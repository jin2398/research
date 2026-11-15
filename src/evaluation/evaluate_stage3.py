"""Offline evaluation of Stage 3 re-ranking and the GRU session encoder.

Stage 1/2 models are not in this repository yet, so two stand-in candidate
sources feed the re-ranker (each with and without Stage 3):

- popularity:  top-N popular items (the popularity baseline of the mission)
- session_knn: nearest CLIP items to the GRU session vector of the user's recent clicks

When the Stage 2 ranker is ready, pass its [(product_id, score), ...] instead of
these candidates; everything else stays the same.

    python -m src.evaluation.evaluate_stage3 --config config/evaluate_stage3.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..recommendation.popularity import Event
from ..recommendation.rerank import (Reranker, RerankConfig, RerankResult, load_all_events,
                                     max_category_run, split_events)
from .metrics import evaluate_rankings


@dataclass(frozen=True, slots=True)
class EvalConfig:
    rerank_config: Path
    train_ratio: float
    valid_ratio: float
    truth_events: tuple[str, ...]
    catalog_path: Path
    catalog_id_column: str
    k: int
    candidate_size: int
    max_users: int | None
    latency_sample: int
    report_path: Path

    @classmethod
    def from_yaml(cls, path: Path) -> "EvalConfig":
        import yaml

        v = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        config = cls(Path(v["rerank_config"]), float(v["split"]["train"]),
                     float(v["split"]["valid"]), tuple(v["truth_events"]),
                     Path(v["catalog_path"]), str(v["catalog_id_column"]), int(v["k"]),
                     int(v["candidate_size"]),
                     int(v["max_users"]) if v.get("max_users") else None,
                     int(v["latency_sample"]), Path(v["report_path"]))
        if config.candidate_size < config.k:
            raise ValueError("candidate_size must be >= k")
        return config


@dataclass(frozen=True, slots=True)
class UserContext:
    user_id: str
    history_count: int
    recent_clicks: list[str]


def catalog_size(config: EvalConfig) -> int:
    with config.catalog_path.open(encoding="utf-8-sig", newline="") as handle:
        return len({row[config.catalog_id_column] for row in csv.DictReader(handle)})


def build_contexts(known: Sequence[Event], users: Sequence[str], window: int) -> list[UserContext]:
    history: dict[str, list[str]] = defaultdict(list)
    for event in known:
        if event.product_id:
            history[event.user_id].append(event.product_id)
    return [UserContext(u, len(history[u]), history[u][-window:]) for u in users]


def stage3_summary(reranker: Reranker, results: Sequence[RerankResult]) -> dict[str, Any]:
    reasons: Counter[str] = Counter()
    runs = []
    for result in results:
        reasons.update(r.reason for r in result.recommendations)
        runs.append(max_category_run([reranker.category(r.product_id)
                                      for r in result.recommendations]))
    n = len(results)
    explore = sum(r.is_exploration for res in results for r in res.recommendations)
    return {
        "lists": n,
        "reason_counts": dict(reasons.most_common()),
        "exploration_slots_per_list": round(explore / n, 3),
        "new_arrival_per_list": round(reasons["new_arrival"] / n, 3),
        "max_same_category_run": max(runs),
        "lists_violating_diversity": sum(r > reranker.config.max_same_category_run for r in runs),
        "cold_start_lists": sum(r.cold_start for r in results),
    }


def latency_p95(fn: Callable[[UserContext], Any], contexts: Sequence[UserContext]) -> dict[str, float]:
    """Mission protocol: warm-up 10 calls, then p95 over the sample (single request)."""
    for ctx in contexts[:10]:
        fn(ctx)
    times = []
    for ctx in contexts:
        started = time.perf_counter()
        fn(ctx)
        times.append((time.perf_counter() - started) * 1000)
    return {"calls": len(times), "p50_ms": round(float(np.percentile(times, 50)), 3),
            "p95_ms": round(float(np.percentile(times, 95)), 3)}


@dataclass
class Prepared:
    """Everything an offline Stage 3 run needs, loaded once."""
    rerank_config: RerankConfig
    reranker: Reranker
    now: datetime
    rows: dict[str, int]                        # split sizes
    contexts: list[UserContext]
    truth: dict[str, set[str]]
    test_events: list[Event]
    popular: list[str]
    popular_candidates: list[tuple[str, float]]
    knn: dict[str, list[tuple[str, float]]]
    users_with_session_vector: int


def prepare(config: EvalConfig, with_knn: bool = True) -> Prepared:
    started = time.perf_counter()
    # offline runs never read or write the live bandit state
    rcfg = replace(RerankConfig.from_yaml(config.rerank_config), bandit_store={"backend": "none"})
    events = load_all_events(rcfg)
    train, valid, test = split_events(events, config.train_ratio, config.valid_ratio)
    known = [e for e in train + valid if e.product_id]
    now = (train + valid)[-1].timestamp
    reranker = Reranker.from_config(rcfg, now=now, events=known)
    if with_knn and (reranker.vectors is None or reranker.session_encoder is None):
        raise RuntimeError("Session kNN needs CLIP vectors and a trained session encoder "
                           "(python -m src.recommendation.session_encoder train)")
    truth: dict[str, set[str]] = defaultdict(set)
    for event in test:
        if event.product_id and event.event_type in config.truth_events:
            truth[event.user_id].add(event.product_id)
    users = sorted(truth)[:config.max_users] if config.max_users else sorted(truth)
    truth = {u: truth[u] for u in users}
    contexts = build_contexts(known, users, rcfg.session_window)
    print(f"[eval] split rows train/valid/test = {len(train)}/{len(valid)}/{len(test)}, "
          f"users = {len(users)}, load {time.perf_counter() - started:.1f}s", flush=True)

    n_cand = config.candidate_size
    popular = reranker.popularity.popular[:n_cand]
    popular_candidates = [(pid, reranker.popularity.normalized(pid)) for pid in popular]

    # session kNN candidates: exact inner-product search of GRU vectors over all items
    knn: dict[str, list[tuple[str, float]]] = {}
    with_vec: list[UserContext] = []
    if with_knn:
        encoder, vectors = reranker.session_encoder, reranker.vectors
        row_ids = np.array(sorted(reranker.vector_rows, key=reranker.vector_rows.get), dtype=object)
        with_vec = [c for c in contexts if encoder.encode(c.recent_clicks) is not None]
        for start in range(0, len(with_vec), 512):
            chunk = with_vec[start:start + 512]
            q = np.stack([encoder.encode(c.recent_clicks) for c in chunk])
            scores = q @ vectors.T
            top = np.argpartition(-scores, n_cand, axis=1)[:, :n_cand]
            for ctx, rows, row_scores in zip(chunk, top, np.take_along_axis(scores, top, 1)):
                order = np.argsort(-row_scores)
                knn[ctx.user_id] = [(str(row_ids[rows[i]]), float(row_scores[i])) for i in order]
    return Prepared(rcfg, reranker, now, {"train": len(train), "valid": len(valid), "test": len(test)},
                    contexts, truth, test, popular, popular_candidates, knn, len(with_vec))


def reset_exploration(reranker: Reranker) -> None:
    """Fresh bandit and RNG so every configuration starts from the same state."""
    import random

    from ..recommendation.bandit import CategoryBandit

    cfg = reranker.config
    reranker.bandit = CategoryBandit(reranker.bandit.arms, cfg.bandit_algorithm, cfg.epsilon,
                                     cfg.ucb_c, cfg.random_seed)
    reranker.rng = random.Random(cfg.random_seed)


def run(config: EvalConfig) -> dict[str, Any]:
    started = time.perf_counter()
    prep = prepare(config)
    rcfg, reranker, now = prep.rerank_config, prep.reranker, prep.now
    contexts, truth, users = prep.contexts, prep.truth, list(prep.truth)
    k, n_cand = config.k, config.candidate_size
    popular, popular_candidates, knn = prep.popular, prep.popular_candidates, prep.knn
    encoder = reranker.session_encoder

    def rerank(ctx: UserContext, candidates: list[tuple[str, float]]) -> RerankResult:
        return reranker.rerank(ctx.user_id, candidates, history_count=ctx.history_count,
                               recent_clicks=ctx.recent_clicks, now=now, top_n=k)

    size = catalog_size(config)
    strategies: dict[str, dict[str, Any]] = {}

    def record(name: str, lists: dict[str, list[str]], note: str,
               results: Sequence[RerankResult] | None = None) -> None:
        strategies[name] = {"note": note, **evaluate_rankings(lists, truth, k, size)}
        if results is not None:
            strategies[name]["stage3"] = stage3_summary(reranker, results)
        m = strategies[name]
        print(f"[eval] {name:28s} HitRate@{k}={m[f'hit_rate_at_{k}']:.4f} "
              f"NDCG@{k}={m[f'ndcg_at_{k}']:.4f} Coverage={m['coverage']:.4f}", flush=True)

    record("popularity", {c.user_id: popular[:k] for c in contexts},
           "baseline: same top-k popular items for every user")
    record("session_knn", {u: [p for p, _ in knn.get(u, [])[:k]] for u in users},
           "GRU session vector -> nearest CLIP items, no re-ranking")
    for name, source, note in (
            ("popularity+stage3", lambda c: popular_candidates,
             "popularity candidates -> Stage 3 (GRU session blend, diversity, MAB, new items)"),
            ("session_knn+stage3", lambda c: knn.get(c.user_id, []),
             "session kNN candidates -> Stage 3")):
        results = [rerank(c, source(c)) for c in contexts]
        record(name, {r.user_id: [x.product_id for x in r.recommendations] for r in results},
               note, results)

    # ablation: same pipeline with the old weighted-average session vector instead of the GRU
    reranker.session_encoder = None
    results = [rerank(c, popular_candidates) for c in contexts]
    record("popularity+stage3_no_gru", {r.user_id: [x.product_id for x in r.recommendations]
                                        for r in results},
           "ablation: Stage 3 with weighted-average session vector", results)
    reranker.session_encoder = encoder

    sample = contexts[:config.latency_sample]
    latency = {
        "session_encode": latency_p95(lambda c: encoder.encode(c.recent_clicks), sample),
        "rerank_session_knn": latency_p95(lambda c: rerank(c, knn.get(c.user_id, [])), sample),
    }
    cold = reranker.rerank("COLD_START_CHECK", popular_candidates, history_count=0,
                           recent_clicks=[], now=now, top_n=k)
    base_hit = strategies["popularity"][f"hit_rate_at_{k}"]
    return {
        "protocol": {
            "data": str(rcfg.events_path), "split": "time-based over all rows, train/valid/test "
            f"= {config.train_ratio}/{config.valid_ratio}/{round(1 - config.train_ratio - config.valid_ratio, 2)}",
            "rows": prep.rows,
            "snapshot": str(now), "context": "train+valid events (popularity, history, recent clicks)",
            "truth": f"items with {list(config.truth_events)} events in the test period",
            "users": len(users), "k": k, "candidate_size": n_cand,
            "catalog_size": size, "catalog": str(config.catalog_path),
            "random_seed": rcfg.random_seed,
            "users_with_session_vector": prep.users_with_session_vector,
        },
        "strategies": strategies,
        "lift_vs_popularity_hit_rate": {
            name: (round(m[f"hit_rate_at_{k}"] / base_hit - 1, 4) if base_hit else None)
            for name, m in strategies.items() if name != "popularity"},
        "latency": latency,
        "cold_start_check": {"history_count": 0, "cold_start": cold.cold_start,
                             "reasons": dict(Counter(r.reason for r in cold.recommendations))},
        "targets": {"hit_rate_at_50": 0.20, "ndcg_at_50": 0.08, "coverage": 0.20},
        "note": "Stage 1/2 are stand-ins; final HitRate/NDCG must be re-measured with the "
                "Two-Tower + DeepFM candidates.",
        "elapsed_s": round(time.perf_counter() - started, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("config/evaluate_stage3.yaml"))
    args = parser.parse_args()
    config = EvalConfig.from_yaml(args.config)
    report = run(config)
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(f"[eval] report -> {config.report_path}")


if __name__ == "__main__":
    main()
