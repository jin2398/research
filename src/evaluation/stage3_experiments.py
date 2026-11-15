"""Stage 3 experiments on the simulator log.

coverage: grid over exploration slots, exploration pool size and session weight;
          HitRate@k / NDCG@k / Coverage for each candidate source.
stage2:   the same grid (plus Stage 2 score normalization) on the real Stage 2
          (Two-Tower -> DeepFM) output, cached by src.evaluation.stage2_candidates.
          Written to `stage2_grid`; the stand-in `coverage_grid` is left as is.
bandit:   replay test users in time order, give exploration feedback, and compare
          bandit algorithms (random / epsilon-greedy / UCB) by reward rate and regret.

    python -m src.evaluation.stage3_experiments all --config config/evaluate_stage3.yaml
    OMP_NUM_THREADS=1 python -m src.evaluation.stage2_candidates      # once, for `stage2`
    python -m src.evaluation.stage3_experiments stage2
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from .evaluate_stage3 import EvalConfig, Prepared, prepare, reset_exploration
from .metrics import evaluate_rankings
from .evaluate_stage3 import catalog_size


def _candidates(prep: Prepared, source: str,
                stage2: dict[str, list[tuple[str, float]]] | None = None):
    if source == "stage2":
        if stage2 is None:
            raise ValueError("stage2 source needs the cached Stage 2 candidates")
        return lambda ctx: stage2.get(ctx.user_id, [])
    if source == "popularity":
        return lambda ctx: prep.popular_candidates
    if source == "session_knn":
        return lambda ctx: prep.knn.get(ctx.user_id, [])
    raise ValueError(f"unknown candidate source: {source}")


def coverage_grid(config: EvalConfig, prep: Prepared, grid: dict[str, Any],
                  stage2: dict[str, list[tuple[str, float]]] | None = None) -> list[dict[str, Any]]:
    reranker, base = prep.reranker, prep.reranker.config
    size, k = catalog_size(config), config.k
    rows = []
    for source, slots, pool, weight, norm in itertools.product(
            grid["sources"], grid["exploration_slots"], grid["exploration_pool_size"],
            grid["session_weight"], grid.get("score_normalization", [base.score_normalization])):
        started = time.perf_counter()
        reranker.config = replace(base, exploration_slots=int(slots),
                                  exploration_pool_size=int(pool), session_weight=float(weight),
                                  score_normalization=str(norm))
        reset_exploration(reranker)
        get = _candidates(prep, source, stage2)
        lists = {}
        for ctx in prep.contexts:
            result = reranker.rerank(ctx.user_id, get(ctx), history_count=ctx.history_count,
                                     recent_clicks=ctx.recent_clicks, now=prep.now, top_n=k)
            lists[ctx.user_id] = [r.product_id for r in result.recommendations]
        metrics = evaluate_rankings(lists, prep.truth, k, size)
        row = {"source": source, "exploration_slots": slots, "exploration_pool_size": pool,
               "session_weight": weight, "score_normalization": norm,
               **{m: metrics[m] for m in
                                            (f"hit_rate_at_{k}", f"ndcg_at_{k}", "coverage")}}
        rows.append(row)
        print(f"[coverage] {source:12s} slots={slots} pool={pool:<5} w={weight} {norm:6s} "
              f"HR={row[f'hit_rate_at_{k}']:.4f} NDCG={row[f'ndcg_at_{k}']:.4f} "
              f"Cov={row['coverage']:.4f} ({time.perf_counter() - started:.0f}s)", flush=True)
    reranker.config = base
    reset_exploration(reranker)
    return rows


def bandit_simulation(config: EvalConfig, prep: Prepared, settings: dict[str, Any]) -> dict[str, Any]:
    reranker, base = prep.reranker, prep.reranker.config
    get = _candidates(prep, settings["candidate_source"])
    # user order = time of the first test event (online replay)
    first: dict[str, Any] = {}
    for event in prep.test_events:
        if event.user_id in prep.truth and event.user_id not in first:
            first[event.user_id] = event.timestamp
    by_user = {c.user_id: c for c in prep.contexts}
    order = sorted(by_user, key=lambda u: (first.get(u), u))
    user_categories = {u: {reranker.category(p) for p in prep.truth[u]} for u in order}
    # oracle: best single category, ignoring the per-list exclusions
    arm_rate = Counter(c for cats in user_categories.values() for c in cats)
    best_arm, best_count = arm_rate.most_common(1)[0]
    oracle_rate = best_count / len(order)

    results = {}
    for algo in settings["algorithms"]:
        reranker.config = replace(base, bandit_algorithm=algo["algorithm"],
                                  epsilon=float(algo["epsilon"]), ucb_c=float(algo["ucb_c"]))
        reset_exploration(reranker)
        rewards, item_hits, lists = [], 0, {}
        for user in order:
            ctx = by_user[user]
            result = reranker.rerank(user, get(ctx), history_count=ctx.history_count,
                                     recent_clicks=ctx.recent_clicks, now=prep.now,
                                     top_n=config.k)
            lists[user] = [r.product_id for r in result.recommendations]
            for rec in result.recommendations:
                if not rec.is_exploration:
                    continue
                reward = reranker.category(rec.product_id) in user_categories[user]
                item_hits += rec.product_id in prep.truth[user]
                reranker.record_feedback(rec.product_id, reward)
                rewards.append(float(reward))
        n, half = len(rewards), len(rewards) // 2
        pulls = reranker.bandit.pulls
        ranking = evaluate_rankings(lists, prep.truth, config.k, catalog_size(config))
        results[algo["name"]] = {
            f"hit_rate_at_{config.k}": ranking[f"hit_rate_at_{config.k}"],
            "coverage": ranking["coverage"],
            "pulls": n,
            "reward_rate": round(sum(rewards) / n, 4),
            "reward_rate_first_half": round(sum(rewards[:half]) / max(half, 1), 4),
            "reward_rate_second_half": round(sum(rewards[half:]) / max(n - half, 1), 4),
            "regret_vs_best_fixed_arm": round(oracle_rate * n - sum(rewards), 1),
            "item_level_hits": item_hits,
            "arms_tried": sum(1 for v in pulls.values() if v),
            "top_arms": [{"category": a, "pulls": pulls[a],
                          "mean_reward": round(reranker.bandit.mean(a), 3)}
                         for a, _ in Counter(pulls).most_common(5)],
        }
        r = results[algo["name"]]
        print(f"[bandit] {algo['name']:20s} reward={r['reward_rate']:.4f} "
              f"(1st half {r['reward_rate_first_half']:.4f} -> 2nd half "
              f"{r['reward_rate_second_half']:.4f}) regret={r['regret_vs_best_fixed_arm']} "
              f"HR={r[f'hit_rate_at_{config.k}']:.4f} Cov={r['coverage']:.4f}",
              flush=True)
    reranker.config = base
    reset_exploration(reranker)
    return {"rounds_users": len(order), "candidate_source": settings["candidate_source"],
            "reward": "1 if the user touched any item of the explored category in the test period",
            "oracle_best_fixed_arm": {"category": best_arm, "reward_rate": round(oracle_rate, 4)},
            "algorithms": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["coverage", "bandit", "all", "stage2"])
    parser.add_argument("--stage2-candidates", type=Path, default=None,
                        help="cached Stage 2 output (default: stage2_candidates.DEFAULT_PATH)")
    parser.add_argument("--config", type=Path, default=Path("config/evaluate_stage3.yaml"))
    args = parser.parse_args()
    config = EvalConfig.from_yaml(args.config)
    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))["experiments"]
    report_path = Path(settings["report_path"])
    report = json.loads(report_path.read_text("utf-8")) if report_path.is_file() else {}
    needs_knn = (args.action in ("coverage", "all")
                 or (args.action == "bandit"
                     and settings["bandit_simulation"]["candidate_source"] == "session_knn"))
    prep = prepare(config, with_knn=needs_knn)
    report["setup"] = {"users": len(prep.contexts), "k": config.k,
                       "candidate_size": config.candidate_size, "snapshot": str(prep.now),
                       "random_seed": prep.rerank_config.random_seed}
    if args.action in ("coverage", "all"):
        report["coverage_grid"] = coverage_grid(config, prep, settings["coverage_grid"])
    if args.action in ("bandit", "all"):
        report["bandit_simulation"] = bandit_simulation(config, prep, settings["bandit_simulation"])
    if args.action == "stage2":
        from . import stage2_candidates

        path = args.stage2_candidates or stage2_candidates.DEFAULT_PATH
        stage2 = stage2_candidates.load(path)
        missing = sum(1 for c in prep.contexts if c.user_id not in stage2)
        report["stage2_grid"] = {
            "candidates": str(path), "users": len(prep.contexts), "users_without_candidates": missing,
            "rows": coverage_grid(config, prep, settings["stage2_grid"], stage2)}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(f"[experiments] report -> {report_path}")


if __name__ == "__main__":
    main()
