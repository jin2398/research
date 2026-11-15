"""Cache the real Stage 2 (Two-Tower -> DeepFM) output for the test users, so Stage 3
experiments can run on it without loading torch + faiss again.

Candidates are built exactly as in ``evaluate_stage12.run`` (context = train+valid,
pool = FAISS top-``recall_k`` + up to ``recent_candidates`` recent clicks, score =
``Ranker.score``), so Stage 3 metrics here match ``two_tower+deepfm+stage3`` there.

    OMP_NUM_THREADS=1 python -m src.evaluation.stage2_candidates --config config/stage12.yaml

On macOS, OMP_NUM_THREADS=1 avoids the torch/faiss OpenMP crash.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import faiss  # noqa: F401  (import before torch)
import numpy as np

DEFAULT_PATH = Path("artifacts/recommendation/stage12/stage2_test_candidates.npz")


def export(config_path: Path, out_path: Path = DEFAULT_PATH) -> Path:
    from ..recommendation.deepfm import Ranker, build_impressions
    from ..recommendation.stage12_data import (SPLIT_TEST, SPLIT_TRAIN, SPLIT_VALID, EventLog,
                                               ItemTable, Stage12Config, UserStates,
                                               item_popularity)
    from ..recommendation.two_tower import CandidateGenerator

    started = time.perf_counter()
    config = Stage12Config.from_yaml(config_path)
    ev = config.evaluation
    items = ItemTable.load(config.output_dir / "encoders.json")
    log = EventLog.load(config, items)
    gen = CandidateGenerator.load(config, items)
    known = log.mask(SPLIT_TRAIN, SPLIT_VALID)
    now_ts = float(log.ts[log.split == SPLIT_VALID].max())
    states = UserStates.build(log, items, known, gen.history, config.event_weights, "train+valid")
    popularity = item_popularity(log, known, items.n_items, config.event_weights)
    ranker = Ranker.load(config, items, popularity)
    users = sorted(log.truth(SPLIT_TEST))
    if ev.get("max_users"):
        users = users[:int(ev["max_users"])]

    imp = build_impressions(gen, ranker.builder, states, log, users, SPLIT_TEST, now_ts,
                            int(ev["recall_k"]), int(config.deepfm.get("recent_candidates", 0)))
    score = ranker.score(*ranker.predict(imp.sparse, imp.dense))

    # per user, sorted by score (stable), as evaluate_stage12 builds stage2_out
    bounds = np.flatnonzero(np.diff(imp.user)) + 1
    user_ids, offsets, rows_out, scores_out = [], [0], [], []
    for rows, sc, us in zip(np.split(imp.rows, bounds), np.split(score, bounds),
                            np.split(imp.user, bounds)):
        order = np.argsort(-sc, kind="stable")
        user_ids.append(str(log.user_ids[us[0]]))
        rows_out.append(rows[order])
        scores_out.append(sc[order].astype(np.float64))
        offsets.append(offsets[-1] + len(order))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, user_ids=np.array(user_ids), offsets=np.array(offsets),
                        product_ids=np.asarray(items.ids)[np.concatenate(rows_out)].astype(str),
                        scores=np.concatenate(scores_out),
                        meta=np.array(json.dumps({"config": str(config_path), "users": len(user_ids),
                                                  "now_ts": now_ts, "seed": config.seed})))
    print(f"[stage2] {len(user_ids)} users, {offsets[-1]} candidates -> {out_path} "
          f"({time.perf_counter() - started:.0f}s)", flush=True)
    return out_path


def load(path: Path = DEFAULT_PATH) -> dict[str, list[tuple[str, float]]]:
    """{user_id: [(product_id, stage2_score), ...]} sorted by score, best first."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run: OMP_NUM_THREADS=1 python -m "
                                "src.evaluation.stage2_candidates")
    data = np.load(path)
    off, pids, scores = data["offsets"], data["product_ids"], data["scores"]
    return {str(uid): list(zip(pids[off[i]:off[i + 1]].tolist(), scores[off[i]:off[i + 1]].tolist()))
            for i, uid in enumerate(data["user_ids"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("config/stage12.yaml"))
    parser.add_argument("--out", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    export(args.config, args.out)


if __name__ == "__main__":
    main()
