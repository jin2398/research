"""Offline evaluation of Stage 1 (Two-Tower + FAISS) and Stage 2 (DeepFM), and of the
full Stage 1 -> 2 -> 3 pipeline, on the test split.

Protocol (same as docs/stage3_report.md): time-based 8/1/1 split over all rows;
user context / popularity = train + valid; truth = items each user viewed, carted or
purchased in the test period; catalog = data/product_features.csv (70,831 items).

Reported
  Stage 1 : Recall@300 (FAISS, and exact search for the ANN loss) vs popularity top-300,
            candidate latency p95 (User Tower + FAISS, 10 warm-up + 100 calls)
  Stage 2 : AUC (CTR / CVR) on the test impressions (= Stage 1 top-300) vs the
            Two-Tower score and popularity, ranking latency p95
  Lists   : HitRate@50 / NDCG@50 / Coverage of popularity, Two-Tower only,
            Two-Tower + DeepFM, Two-Tower + DeepFM + Stage 3
  Ceiling : rule-based and oracle Recall@300 bounds of this log (see `ceiling`)
  API     : end-to-end `RecommendationService` latency is added to the same report by
            `python -m src.recommendation.pipeline bench --save` (kept on re-evaluation)

    python -m src.evaluation.evaluate_stage12 --config config/stage12.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import faiss  # noqa: F401  (import before torch)
import numpy as np

from ..recommendation.deepfm import FeatureBuilder, Ranker, build_impressions, fast_auc
from ..recommendation.stage12_data import (SPLIT_TEST, SPLIT_TRAIN, SPLIT_VALID, EventLog,
                                           ItemTable, Stage12Config, UserStates, item_popularity)
from ..recommendation.two_tower import CandidateGenerator, exact_top_k, recall_at_k
from .metrics import evaluate_rankings

TARGETS = {"recall_at_300": 0.30, "auc": 0.70, "hit_rate_at_50": 0.20, "ndcg_at_50": 0.08,
           "coverage": 0.20, "candidate_p95_ms": 100.0}


def p95(fn: Callable[[int], Any], n_calls: int, warmup: int = 10) -> dict[str, float]:
    """Mission protocol: `warmup` calls, then p50/p95 over `n_calls` single requests."""
    for i in range(warmup):
        fn(i)
    times = []
    for i in range(warmup, warmup + n_calls):
        t0 = time.perf_counter()
        fn(i)
        times.append((time.perf_counter() - t0) * 1000)
    return {"calls": n_calls, "warmup": warmup, "p50_ms": round(float(np.percentile(times, 50)), 3),
            "p95_ms": round(float(np.percentile(times, 95)), 3)}


def run(config: Stage12Config) -> dict[str, Any]:
    started = time.perf_counter()
    ev = config.evaluation
    k, rk = int(ev["k"]), int(ev["recall_k"])
    items = ItemTable.load(config.output_dir / "encoders.json")
    log = EventLog.load(config, items)
    gen = CandidateGenerator.load(config, items)
    weights = config.event_weights
    known = log.mask(SPLIT_TRAIN, SPLIT_VALID)
    now_ts = float(log.ts[log.split == SPLIT_VALID].max())
    states = UserStates.build(log, items, known, gen.history, weights, "train+valid")
    popularity = item_popularity(log, known, items.n_items, weights)
    ranker = Ranker.load(config, items, popularity)
    fb: FeatureBuilder = ranker.builder

    truth_map = log.truth(SPLIT_TEST)
    users = sorted(truth_map)
    if ev.get("max_users"):
        users = users[:int(ev["max_users"])]
    truth = [truth_map[u] for u in users]
    print(f"[eval12] users {len(users)}, load {time.perf_counter() - started:.0f}s", flush=True)

    # ---------------- Stage 1
    contexts = [states.context(str(log.user_ids[u])) for u in users]
    vecs = np.vstack([gen.user_vectors(contexts[s:s + 1024]) for s in range(0, len(users), 1024)])
    _, faiss_rows = gen.index.search(vecs, rk)
    item_vecs = np.load(config.output_dir / "item_embeddings.npy")
    exact_rows = np.vstack([exact_top_k(vecs[s:s + 1024], item_vecs, rk)
                            for s in range(0, len(users), 1024)])
    pop_rows = np.argsort(-popularity, kind="stable")[:rk]
    stage1 = {
        f"two_tower_faiss_recall_at_{rk}": round(recall_at_k(faiss_rows, truth), 4),
        f"two_tower_exact_recall_at_{rk}": round(recall_at_k(exact_rows, truth), 4),
        f"popularity_recall_at_{rk}": round(recall_at_k(np.tile(pop_rows, (len(users), 1)), truth), 4),
        "index_type": config.faiss["index_type"],
    }
    base = stage1[f"popularity_recall_at_{rk}"]
    stage1["lift_vs_popularity"] = round(stage1[f"two_tower_faiss_recall_at_{rk}"] / base - 1, 4)
    # per persona (where the candidate generator is weak or strong)
    personas = np.array([log.persona[u] for u in users])
    per_rec = np.array([np.isin(t, r).mean() for r, t in zip(faiss_rows, truth)])
    stage1["recall_by_persona"] = {log.persona_vocab[c - 1]: round(float(per_rec[personas == c].mean()), 4)
                                   for c in np.unique(personas) if c > 0}
    print(f"[eval12] Stage 1 {stage1}", flush=True)

    # ---------------- Stage 2 (test impressions = Stage 1 top-300 per user)
    n_recent = int(config.deepfm.get("recent_candidates", 0))
    imp = build_impressions(gen, fb, states, log, users, SPLIT_TEST, now_ts, rk, n_recent)
    ctr, cvr = ranker.predict(imp.sparse, imp.dense)
    tt_only = ~imp.recent_source
    bounds_u = np.flatnonzero(np.diff(imp.user)) + 1
    pool_rec = [np.isin(t, r).mean() for r, t in zip(np.split(imp.rows, bounds_u), truth)]
    stage1["ranking_pool_recall"] = round(float(np.mean(pool_rec)), 4)
    stage1["ranking_pool"] = f"FAISS top-{rk} + up to {n_recent} recent clicks not in it"
    stage2 = {
        "recent_candidates_per_user": n_recent,
        "impressions": int(len(imp.click)), "ctr": round(float(imp.click.mean()), 5),
        "cvr": round(float(imp.conv.mean()), 6),
        "auc_ctr_deepfm": round(fast_auc(imp.click, ctr), 4),
        "auc_cvr_deepfm": round(fast_auc(imp.conv, cvr), 4),
        "auc_ctr_two_tower_score": round(fast_auc(imp.click, imp.tt_score), 4),
        "auc_ctr_popularity": round(fast_auc(imp.click, imp.pop), 4),
        "auc_ctr_deepfm_faiss_candidates_only": round(fast_auc(imp.click[tt_only], ctr[tt_only]), 4),
    }
    stage2["auc_lift_vs_two_tower"] = round(stage2["auc_ctr_deepfm"] / stage2["auc_ctr_two_tower_score"] - 1, 4)
    print(f"[eval12] Stage 2 {stage2}", flush=True)

    # ---------------- top-k lists
    score = ranker.score(ctr, cvr)
    ids = items.ids
    lists: dict[str, dict[str, list[str]]] = {"popularity": {}, "two_tower": {}, "two_tower+deepfm": {}}
    stage2_out: dict[str, list[tuple[str, float]]] = {}
    bounds = np.flatnonzero(np.diff(imp.user)) + 1
    for rows, sc, us in zip(np.split(imp.rows, bounds), np.split(score, bounds),
                            np.split(imp.user, bounds)):
        uid = str(log.user_ids[us[0]])
        order = np.argsort(-sc, kind="stable")
        lists["two_tower"][uid] = [ids[r] for r in rows[:k]]
        lists["two_tower+deepfm"][uid] = [ids[rows[i]] for i in order[:k]]
        stage2_out[uid] = [(ids[rows[i]], float(sc[i])) for i in order]
    pop_list = [ids[r] for r in pop_rows[:k]]
    truth_ids = {str(log.user_ids[u]): {ids[i] for i in t} for u, t in zip(users, truth)}
    for uid in truth_ids:
        lists["popularity"][uid] = pop_list
    strategies = {name: evaluate_rankings(l, truth_ids, k, items.n_items) for name, l in lists.items()}

    stage3_info: dict[str, Any] = {}
    latency: dict[str, Any] = {}
    sample = list(range(min(len(users), int(ev["latency_users"]))))

    def cand(i: int) -> Any:
        return gen.search_pool([contexts[i % len(contexts)]], rk, n_recent)[0]

    latency["candidate_generation"] = p95(cand, len(sample) - 10)
    cached = {i: cand(i) for i in sample}

    def rank(i: int) -> Any:
        rows, scs, _ = cached[i % len(sample)]
        return ranker.rank(contexts[i % len(contexts)], rows, scs, now_ts)

    latency["ranking"] = p95(rank, len(sample) - 10)

    if ev.get("run_stage3"):
        lists_s3, stage3_info, rerank_lat = _with_stage3(config, log, users, stage2_out, now_ts, k)
        strategies["two_tower+deepfm+stage3"] = evaluate_rankings(lists_s3, truth_ids, k, items.n_items)
        latency["reranking"] = rerank_lat
    for name, m in strategies.items():
        print(f"[eval12] {name:26s} HitRate@{k}={m[f'hit_rate_at_{k}']:.4f} "
              f"NDCG@{k}={m[f'ndcg_at_{k}']:.4f} Coverage={m['coverage']:.4f}", flush=True)
    latency["sum_p95_ms"] = round(sum(v["p95_ms"] for v in latency.values() if isinstance(v, dict)), 3)

    base_hit = strategies["popularity"][f"hit_rate_at_{k}"]
    final_name = "two_tower+deepfm+stage3" if "two_tower+deepfm+stage3" in strategies else "two_tower+deepfm"
    final = strategies[final_name]
    headline = {
        "recall_at_300": stage1[f"two_tower_faiss_recall_at_{rk}"],
        "auc": stage2["auc_ctr_deepfm"],
        "hit_rate_at_50": final[f"hit_rate_at_{k}"],
        "ndcg_at_50": final[f"ndcg_at_{k}"],
        "coverage": final["coverage"],
        "candidate_p95_ms": latency["candidate_generation"]["p95_ms"],
    }
    return {
        "protocol": {
            "data": config.data["events_path"], "rows": log.rows, "split_times": log.split_times,
            "split": f"time-based over all rows, train/valid/test = {config.train_ratio}/"
                     f"{config.valid_ratio}/{round(1 - config.train_ratio - config.valid_ratio, 2)}",
            "context": "train+valid (user history, popularity)", "two_tower_trained_on": "train",
            "deepfm_trained_on": "train context -> valid labels",
            "truth": "items viewed/carted/purchased in the test period", "users": len(users),
            "catalog_size": items.n_items, "k": k, "recall_k": rk, "random_seed": config.seed,
            "hardware": _hardware(),
        },
        "stage1": stage1, "stage2": stage2, "strategies": strategies,
        "lift_vs_popularity_hit_rate": {n: round(m[f"hit_rate_at_{k}"] / base_hit - 1, 4)
                                        for n, m in strategies.items() if n != "popularity"},
        "stage3": stage3_info, "latency": latency,
        "ceiling": ceiling(log, items, weights, rk) if ev.get("run_ceiling", True) else None,
        "headline": headline, "headline_source": {"lists": final_name},
        "targets": TARGETS,
        "passed": {key: (headline[key] <= v if key.endswith("_ms") else headline[key] >= v)
                   for key, v in TARGETS.items()},
        "elapsed_s": round(time.perf_counter() - started, 1),
    }


def _with_stage3(config: Stage12Config, log: EventLog, users: Sequence[int],
                 stage2_out: dict[str, list[tuple[str, float]]], now_ts: float,
                 k: int) -> tuple[dict[str, list[str]], dict[str, Any], dict[str, float]]:
    """Feed Stage 2 output to the Stage 3 re-ranker exactly as evaluate_stage3 does."""
    from ..recommendation.rerank import Reranker, RerankConfig, load_all_events, split_events
    from .evaluate_stage3 import build_contexts, latency_p95, stage3_summary

    rcfg = replace(RerankConfig.from_yaml(Path(config.evaluation["rerank_config"])),
                   bandit_store={"backend": "none"})
    events = load_all_events(rcfg)
    train, valid, _ = split_events(events, config.train_ratio, config.valid_ratio)
    known = [e for e in train + valid if e.product_id]
    now = (train + valid)[-1].timestamp
    reranker = Reranker.from_config(rcfg, now=now, events=known)
    uids = [str(log.user_ids[u]) for u in users]
    contexts = build_contexts(known, uids, rcfg.session_window)
    results = []
    for ctx in contexts:
        results.append(reranker.rerank(ctx.user_id, stage2_out.get(ctx.user_id, []),
                                       history_count=ctx.history_count,
                                       recent_clicks=ctx.recent_clicks, now=now, top_n=k))
    lists = {r.user_id: [x.product_id for x in r.recommendations] for r in results}
    summary = stage3_summary(reranker, results)
    top10 = [reranker.rerank(c.user_id, stage2_out.get(c.user_id, []), history_count=c.history_count,
                             recent_clicks=c.recent_clicks, now=now, top_n=10) for c in contexts[:500]]
    summary["top10_exploration_slots_per_list"] = round(
        sum(r.is_exploration for res in top10 for r in res.recommendations) / len(top10), 3)
    sample = contexts[:110]
    lat = latency_p95(lambda c: reranker.rerank(c.user_id, stage2_out.get(c.user_id, []),
                                                history_count=c.history_count,
                                                recent_clicks=c.recent_clicks, now=now, top_n=10),
                      sample)
    return lists, summary, lat


# ---------------------------------------------------------------- Recall@300 ceiling
def _share(user: np.ndarray, code: np.ndarray, n_users: int, size: int) -> np.ndarray:
    out = np.zeros((n_users, size))
    np.add.at(out, (user, code), 1.0)
    total = out.sum(1, keepdims=True)
    return np.divide(out, total, out=np.zeros_like(out), where=total > 0)


def ceiling(log: EventLog, items: ItemTable, weights: dict[str, float], k: int) -> dict[str, Any]:
    """How high can Recall@k go on this log? Rule-based and oracle bounds.

    score(u, i) = category_weight(u, i) * persona_lift(persona_u, attr_i) * (1 + pop_i)
                  [* brand share for brand_loyal users]
    attr_i = (is_new, is_hot, discounted); persona_lift = click share of that attribute
    combination for the persona / its catalog share (train+valid). Only category_weight
    differs between variants:
      history_small  : user's category_small share in train+valid (rule)
      history_bucket : user's category_bucket share x global P(small | bucket) (rule)
      oracle_bucket  : user's *test-period* bucket share x P(small | bucket)
      oracle_small   : user's *test-period* category_small share
    oracle_* read the answer and are not models; they bound any history-based model.
    """
    started = time.perf_counter()
    k = min(k, items.n_items - 1)
    known = log.mask(SPLIT_TRAIN, SPLIT_VALID)
    test = log.split == SPLIT_TEST
    card = dict(zip(items.fields, items.cardinalities()))
    cat, bucket, brand = (items.column(f) for f in ("category_small", "category_bucket", "brand"))
    attr = (items.column("is_new") * 9 + items.column("is_hot") * 3
            + (items.column("discount_pct") > 1).astype(np.int64))       # code 1 = no discount
    pop = item_popularity(log, known, items.n_items, weights) + 1.0
    n_users = log.n_users

    base = np.bincount(attr, minlength=27) / items.n_items
    lift = np.zeros((len(log.persona_vocab) + 1, 27))
    for p in range(1, len(log.persona_vocab) + 1):
        m = known & (log.persona[log.user] == p)
        freq = np.bincount(attr[log.item[m]], minlength=27) / max(m.sum(), 1)
        lift[p] = np.divide(freq, base, out=np.zeros(27), where=base > 0)
    sb = np.zeros((card["category_bucket"], card["category_small"]))
    np.add.at(sb, (bucket[log.item[known]], cat[log.item[known]]), 1.0)
    sb /= np.maximum(sb.sum(1, keepdims=True), 1)
    small_given_bucket = sb[bucket, cat]

    ku, ki = log.user[known], log.item[known]
    tu, ti = log.user[test], log.item[test]
    hist_small = _share(ku, cat[ki], n_users, card["category_small"])
    hist_bucket = _share(ku, bucket[ki], n_users, card["category_bucket"])
    hist_brand = _share(ku, brand[ki], n_users, card["brand"])
    test_small = _share(tu, cat[ti], n_users, card["category_small"])
    test_bucket = _share(tu, bucket[ti], n_users, card["category_bucket"])
    loyal = log.persona_vocab.index("brand_loyal") + 1 if "brand_loyal" in log.persona_vocab else -1

    truth = log.truth(SPLIT_TEST)
    variants = {
        "history_small": lambda u: hist_small[u][cat],
        "history_bucket": lambda u: hist_bucket[u][bucket] * small_given_bucket,
        "oracle_bucket": lambda u: test_bucket[u][bucket] * small_given_bucket,
        "oracle_small": lambda u: test_small[u][cat],
    }
    result: dict[str, Any] = {"users": len(truth), "k": k}
    for name, weight in variants.items():
        rec = []
        for u in sorted(truth):
            s = weight(u) * lift[log.persona[u]][attr] * pop
            if log.persona[u] == loyal:
                s = s * (hist_brand[u][brand] + 0.02)
            rec.append(np.isin(truth[u], np.argpartition(-s, k)[:k]).mean())
        result[f"{name}_recall_at_{k}"] = round(float(np.mean(rec)), 4)
    result["note"] = "oracle_* use test-period categories (not achievable by a model)"
    result["elapsed_s"] = round(time.perf_counter() - started, 1)
    print(f"[eval12] ceiling {result}", flush=True)
    return result


def _hardware() -> dict[str, Any]:
    import os
    import platform

    import torch

    return {"platform": platform.platform(), "processor": platform.processor(),
            "logical_cpus": os.cpu_count(), "python": platform.python_version(),
            "torch": torch.__version__, "faiss": faiss.__version__, "gpu": False,
            "torch_threads": torch.get_num_threads(), "measured_at": datetime.now().isoformat(timespec="seconds")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("config/stage12.yaml"))
    parser.add_argument("--no-dashboard", action="store_true",
                        help="do not write data/recommendation_metrics.json")
    args = parser.parse_args()
    config = Stage12Config.from_yaml(args.config)
    write_outputs(config, run(config), dashboard=not args.no_dashboard)


def write_outputs(config: Stage12Config, report: dict[str, Any], dashboard: bool = True) -> None:
    """docs/stage12_eval.json and (optionally) data/recommendation_metrics.json."""
    path = Path(config.evaluation["report_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and "api_bench" not in report:      # keep the last `pipeline bench --save`
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("api_bench"):
            report["api_bench"] = previous["api_bench"]
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval12] report -> {path}")
    if dashboard:
        h = report["headline"]
        metrics = {key: h[key] for key in ("recall_at_300", "auc", "hit_rate_at_50", "ndcg_at_50",
                                           "coverage")}
        metrics["_source"] = (f"{path} (measured {report['protocol']['hardware']['measured_at']}, "
                              f"lists = {report['headline_source']['lists']})")
        dash = Path(config.evaluation["dashboard_metrics_path"])
        dash.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[eval12] dashboard metrics -> {dash}")


if __name__ == "__main__":
    main()
