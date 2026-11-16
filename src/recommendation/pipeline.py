"""Online recommendation pipeline: Stage 1 (Two-Tower + FAISS) -> Stage 2 (DeepFM)
-> Stage 3 (re-ranking, MAB, session). Produces the recommendation API response.

For the API owner (FastAPI + Redis): create one ``RecommendationService`` at startup
and call ``recommend(user_id, top_n, recent_clicks=...)`` per request. Pass the
session clicks from the Redis feature store as ``recent_clicks`` (product ids, oldest
-> newest); they are appended to the user's snapshot so the current session is
reflected immediately in both the User Tower and the ranker.

    python -m src.recommendation.pipeline build-state --config config/stage12.yaml
    python -m src.recommendation.pipeline demo --user-id U00001 --top-n 10
    python -m src.recommendation.pipeline demo --user-id NEW_USER          # cold start
    python -m src.recommendation.pipeline bench --calls 100 --save         # p95 -> docs/stage12_eval.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import faiss  # noqa: F401  (import before torch)
import numpy as np

from .deepfm import Ranker
from .stage12_data import (
    EventLog,
    ItemTable,
    Stage12Config,
    UserStates,
    item_popularity,
)
from .strategies import get_strategy, resolve_name
from .two_tower import CandidateGenerator

STATE_FILE = "user_state.pkl"
POPULARITY_FILE = "popularity.npy"


def build_serving_state(
    config: Stage12Config, items: ItemTable
) -> tuple[UserStates, np.ndarray]:
    """Snapshot of every logged event (all splits) = what the live service knows."""
    log = EventLog.load(config, items)
    everything = np.ones(len(log.user), dtype=bool)
    states = UserStates.build(
        log,
        items,
        everything,
        int(config.two_tower["history_length"]),
        config.event_weights,
        "all events",
    )
    popularity = item_popularity(log, everything, items.n_items, config.event_weights)
    out = config.output_dir
    states.save(out / STATE_FILE)
    np.save(out / POPULARITY_FILE, popularity)
    return states, popularity


class RecommendationService:
    def __init__(
        self,
        config_path: Path | str = "config/stage12.yaml",
        rerank_config: Path | str | None = "config/rerank.yaml",
        use_stage3: bool = True,
        ranking_strategy: str | None = None,
    ) -> None:
        t0 = time.perf_counter()
        self.config = Stage12Config.from_yaml(config_path)
        # Stage 2 정렬 기준: config의 serving.ranking_strategy 또는 환경변수 RANKING_STRATEGY
        self.strategy = get_strategy(resolve_name(self.config.raw, ranking_strategy))
        self._newness: np.ndarray | None = None
        out = self.config.output_dir
        self.items = ItemTable.load(out / "encoders.json")
        if (out / STATE_FILE).is_file() and (out / POPULARITY_FILE).is_file():
            self.states = UserStates.load(out / STATE_FILE)
            self.popularity = np.load(out / POPULARITY_FILE)
        else:
            self.states, self.popularity = build_serving_state(self.config, self.items)
        self.generator = CandidateGenerator.load(self.config, self.items)
        self.ranker = Ranker.load(self.config, self.items, self.popularity)
        finite = self.states.last_ts[np.isfinite(self.states.last_ts)]
        self.clock_ts = (
            float(finite.max()) if len(finite) else time.time()
        )  # simulation clock
        self.reranker = None
        if use_stage3 and rerank_config:
            from .rerank import RerankConfig, Reranker

            self.reranker = Reranker.from_config(
                RerankConfig.from_yaml(Path(rerank_config))
            )
        self.load_s = round(time.perf_counter() - t0, 2)

    def item_newness(self) -> np.ndarray:
        """newness_value per item row (0~1, larger = newer), loaded on first use."""
        if self._newness is None:
            import pandas as pd

            col = self.config.columns["item_id"]
            df = pd.read_csv(
                self.config.data["catalog_path"],
                dtype={col: str},
                usecols=[col, "newness_value"],
            )
            values = dict(zip(df[col], df["newness_value"].astype(float)))
            self._newness = np.array(
                [values.get(str(pid), 0.0) for pid in self.items.ids]
            )
        return self._newness

    def has_item(self, product_id: str) -> bool:
        return str(product_id) in self.items.index

    def _rows(self, product_ids: Sequence[str]) -> np.ndarray:
        rows = self.items.lookup(product_ids)
        return rows[rows >= 0]

    def recommend(
        self,
        user_id: str,
        top_n: int = 10,
        recent_clicks: Sequence[str] = (),
        now_ts: float | None = None,
    ) -> dict[str, Any]:
        """Recommendation API response (mission field spec)."""
        if top_n < 1:
            raise ValueError("top_n must be positive")
        t0 = time.perf_counter()
        ctx = self.states.context(user_id, self._rows(recent_clicks))
        rows, tt, _ = self.generator.search_pool(
            [ctx], None, int(self.config.deepfm.get("recent_candidates", 0))
        )[0]
        t1 = time.perf_counter()
        stage2 = (
            self.strategy.rank(self, ctx, rows, tt, now_ts or self.clock_ts)
            if len(rows)
            else []
        )
        t2 = time.perf_counter()
        ids = self.items.ids
        clicks = [str(ids[r]) for r in ctx.recent_items]
        if self.reranker is not None:
            res = self.reranker.rerank(
                str(user_id),
                stage2,
                history_count=ctx.n_events,
                recent_clicks=clicks,
                top_n=top_n,
            )
            recs = [
                {
                    "product_id": r.product_id,
                    "score": round(r.score, 6),
                    "reason": r.reason,
                    "is_exploration": r.is_exploration,
                }
                for r in res.recommendations
            ]
            session = res.session_context
            cold = res.cold_start
        else:
            recs = [
                {
                    "product_id": pid,
                    "score": round(s, 6),
                    "reason": "ranking",
                    "is_exploration": False,
                }
                for pid, s in stage2[:top_n]
            ]
            session = {"recent_clicks": clicks[-5:], "session_interest": None}
            cold = ctx.n_events < 5
        t3 = time.perf_counter()
        ms = lambda a, b: round((b - a) * 1000, 3)
        return {
            "user_id": str(user_id),
            "recommendations": recs,
            "pipeline_latency": {
                "candidate_ms": ms(t0, t1),
                "ranking_ms": ms(t1, t2),
                "reranking_ms": ms(t2, t3),
                "total_ms": ms(t0, t3),
            },
            "session_context": session,
            "cold_start": cold,
            "candidate_count": len(rows),
            "ranking_strategy": self.strategy.name,
        }

    def record_feedback(self, product_id: str, clicked: bool) -> None:
        """Exploration feedback for the Stage 3 bandit (call when an is_exploration item
        was shown and then clicked / ignored)."""
        if self.reranker is not None:
            self.reranker.record_feedback(product_id, clicked)


def retrain(config: Stage12Config) -> None:
    """One command for the CT pipeline (ct_monitor.py --train-command runs a single
    process without a shell): Two-Tower -> FAISS -> DeepFM -> serving state ->
    evaluation (docs/stage12_eval.json + data/recommendation_metrics.json).
    Exit code 0 = every step finished. Item encodings are rebuilt from the catalog."""
    from ..evaluation import evaluate_stage12
    from . import deepfm, two_tower

    (config.output_dir / "encoders.json").unlink(missing_ok=True)
    two_tower.train(config)
    deepfm.train(config)
    build_serving_state(config, ItemTable.load(config.output_dir / "encoders.json"))
    evaluate_stage12.write_outputs(config, evaluate_stage12.run(config))


def _bench(service: RecommendationService, calls: int, top_n: int) -> dict[str, Any]:
    users = [str(u) for u in service.states.user_ids[: calls + 10]]
    for u in users[:10]:
        service.recommend(u, top_n)
    parts: dict[str, list[float]] = {
        "candidate_ms": [],
        "ranking_ms": [],
        "reranking_ms": [],
        "total_ms": [],
    }
    for u in users[10:]:
        lat = service.recommend(u, top_n)["pipeline_latency"]
        for key in parts:
            parts[key].append(lat[key])
    return {
        key: {
            "p50": round(float(np.percentile(v, 50)), 3),
            "p95": round(float(np.percentile(v, 95)), 3),
        }
        for key, v in parts.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["build-state", "demo", "bench", "retrain"])
    parser.add_argument("--config", type=Path, default=Path("config/stage12.yaml"))
    parser.add_argument(
        "--rerank-config", type=Path, default=Path("config/rerank.yaml")
    )
    parser.add_argument("--no-stage3", action="store_true")
    parser.add_argument("--user-id", default="U00001")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--recent", default="", help="comma separated product ids (session clicks)"
    )
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument(
        "--save",
        action="store_true",
        help="bench: store the result as api_bench in evaluation.report_path",
    )
    args = parser.parse_args()
    if args.command == "retrain":
        retrain(Stage12Config.from_yaml(args.config))
        return
    if args.command == "build-state":
        config = Stage12Config.from_yaml(args.config)
        states, _ = build_serving_state(
            config, ItemTable.load(config.output_dir / "encoders.json")
        )
        print(
            f"[pipeline] user state for {len(states.user_ids)} users -> {config.output_dir / STATE_FILE}"
        )
        return
    service = RecommendationService(
        args.config, None if args.no_stage3 else args.rerank_config
    )
    if args.command == "demo":
        recent = [p for p in args.recent.split(",") if p]
        print(
            json.dumps(
                service.recommend(args.user_id, args.top_n, recent),
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        bench = {
            "calls": args.calls,
            "warmup": 10,
            "top_n": args.top_n,
            "load_s": service.load_s,
            "latency": _bench(service, args.calls, args.top_n),
        }
        print(json.dumps(bench, indent=2))
        if args.save:
            path = Path(service.config.evaluation["report_path"])
            report = (
                json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            )
            report["api_bench"] = bench
            path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"[pipeline] api_bench -> {path}")


if __name__ == "__main__":
    main()
