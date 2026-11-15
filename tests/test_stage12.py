"""Stage 1 (Two-Tower) / Stage 2 (DeepFM) on a tiny synthetic catalog and log.

    python -m pytest tests/test_stage12.py -q
"""

from __future__ import annotations

import csv
import random
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import faiss  # noqa: F401  (before torch)
    import numpy as np
    import torch  # noqa: F401
    import yaml
    HAVE_DEPS = True
except ImportError:  # pragma: no cover
    HAVE_DEPS = False

PERSONAS = ["trendsetter", "pragmatist", "value_seeker"]
BUCKETS = ["상의", "하의", "아우터"]


def write_fixture(root: Path, n_items: int = 240, n_users: int = 30, events_per_user: int = 60) -> Path:
    rng = random.Random(0)
    data = root / "data"
    data.mkdir()
    cats = {b: [f"{b}_small{i}" for i in range(3)] for b in BUCKETS}
    with (data / "product_features.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["article_id", "category_bucket", "brand", "price", "discount_pct",
                    "newness_value", "popularity_score", "is_new", "is_hot"])
        items = []
        for i in range(n_items):
            b = BUCKETS[i % 3]
            items.append((f"{i:010d}", b, rng.choice(cats[b])))
            w.writerow([f"{i:010d}", b, f"B{i % 5}", 10000 + 1000 * (i % 40), (i % 4) * 0.1,
                        rng.random(), rng.random(), int(i % 5 == 0), int(i % 7 == 0)])
    with (data / "hm_products.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["article_id", "category_large", "category_middle", "category_small",
                    "product_group", "garment_group", "color"])
        for pid, b, small in items:
            w.writerow([pid, "Ladieswear", f"{b}_mid", small, "Garment", "Jersey", rng.choice(["Black", "White"])])
    start = datetime(2025, 1, 1)
    rows = []
    for u in range(n_users):
        bucket = BUCKETS[u % 3]
        pool = [pid for pid, b, _ in items if b == bucket]
        t = start + timedelta(minutes=u)
        for j in range(events_per_user):
            t += timedelta(hours=12)
            et = rng.choices(["view", "cart", "purchase", "search"], [0.7, 0.15, 0.05, 0.1])[0]
            pid = "" if et == "search" else rng.choice(pool)
            rows.append([f"U{u:05d}", f"U{u:05d}-S{j // 5:05d}", et, pid, t.isoformat(),
                         PERSONAS[u % 3], bucket])
    rows.sort(key=lambda r: r[4])
    with (data / "behavior_logs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "session_id", "event_type", "product_id", "timestamp", "persona",
                    "preferred_category"])
        w.writerows(rows)

    cfg = yaml.safe_load((ROOT / "config/stage12.yaml").read_text(encoding="utf-8"))
    cfg["data"] = {"events_path": str(data / "behavior_logs.csv"),
                   "catalog_path": str(data / "product_features.csv"),
                   "products_path": str(data / "hm_products.csv")}
    cfg["output"]["dir"] = str(root / "out")
    cfg["two_tower"].update(epochs=1, batch_size=64, history_length=8, recent_length=3, output_dim=16,
                            embedding_dim=8, hidden=[32])
    cfg["faiss"].update(top_k=50, hnsw_ef_search=64)
    cfg["deepfm"].update(epochs=1, batch_size=256, candidates_per_user=50, hidden=[32],
                         embedding_dim=4)
    cfg["evaluation"].update(recall_k=50, k=10, latency_users=15, run_stage3=False,
                             report_path=str(root / "eval.json"),
                             dashboard_metrics_path=str(root / "metrics.json"))
    path = root / "stage12.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return path


@unittest.skipUnless(HAVE_DEPS, "torch / faiss / yaml not installed")
class Stage12Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from src.recommendation import deepfm, two_tower
        from src.recommendation.stage12_data import Stage12Config

        cls.tmp = tempfile.TemporaryDirectory()
        cls.config = Stage12Config.from_yaml(write_fixture(Path(cls.tmp.name)))
        cls.tt_report = two_tower.train(cls.config)
        cls.fm_report = deepfm.train(cls.config)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_split_and_items(self) -> None:
        from src.recommendation.stage12_data import EventLog, ItemTable

        items = ItemTable.load(self.config.output_dir / "encoders.json")
        self.assertEqual(items.n_items, 240)
        self.assertTrue((items.codes >= 0).all())
        log = EventLog.load(self.config, items)
        rows = log.rows
        self.assertEqual(rows["train"] + rows["valid"] + rows["test"], rows["total"])
        # time-based: every train event precedes every test event
        self.assertLessEqual(log.ts[log.split == 0].max(), log.ts[log.split == 2].min())

    def test_two_tower_artifacts_and_unit_vectors(self) -> None:
        out = self.config.output_dir
        for name in ("two_tower.pt", "item_embeddings.npy", "items.index", "two_tower_report.json"):
            self.assertTrue((out / name).is_file(), name)
        vecs = np.load(out / "item_embeddings.npy")
        self.assertEqual(vecs.shape[0], 240)
        np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)

    def test_candidates_cover_top_k_and_match_exact_search(self) -> None:
        from src.recommendation.stage12_data import EventLog, ItemTable, UserStates
        from src.recommendation.two_tower import CandidateGenerator, exact_top_k

        items = ItemTable.load(self.config.output_dir / "encoders.json")
        log = EventLog.load(self.config, items)
        gen = CandidateGenerator.load(self.config, items)
        states = UserStates.build(log, items, log.mask(0, 1), gen.history, self.config.event_weights, "t")
        ctx = states.context("U00001")
        cands = gen.candidates(ctx)
        self.assertEqual(len(cands), 50)
        self.assertEqual(len({r for r, _ in cands}), 50)
        scores = [s for _, s in cands]
        self.assertEqual(scores, sorted(scores, reverse=True))
        exact = exact_top_k(gen.user_vectors([ctx]), np.load(self.config.output_dir / "item_embeddings.npy"), 50)
        overlap = len(set(exact[0]) & {r for r, _ in cands}) / 50
        self.assertGreaterEqual(overlap, 0.9)          # HNSW ~ exact on a tiny index
        # unknown user (cold start) still gets candidates from the profile-free tower
        self.assertEqual(len(gen.candidates(states.context("NOBODY"))), 50)

    def test_fresh_session_clicks_change_the_user_vector(self) -> None:
        from src.recommendation.stage12_data import EventLog, ItemTable, UserStates
        from src.recommendation.two_tower import CandidateGenerator

        items = ItemTable.load(self.config.output_dir / "encoders.json")
        log = EventLog.load(self.config, items)
        gen = CandidateGenerator.load(self.config, items)
        states = UserStates.build(log, items, log.mask(0, 1), gen.history, self.config.event_weights, "t")
        before = gen.user_vectors([states.context("U00000")])
        after = gen.user_vectors([states.context("U00000", [1, 4, 7])])
        self.assertGreater(float(np.abs(before - after).sum()), 1e-4)

    def test_deepfm_shapes_and_auc(self) -> None:
        from src.evaluation.metrics import auc
        from src.recommendation.deepfm import DENSE_FIELDS, fast_auc

        self.assertEqual(len(self.fm_report["fields"]), 14 + 13)
        self.assertEqual(len(self.fm_report["dense_fields"]), len(DENSE_FIELDS))
        rng = np.random.default_rng(1)
        labels = rng.integers(0, 2, 500)
        scores = rng.random(500).round(2)                # ties on purpose
        self.assertAlmostEqual(fast_auc(labels, scores), auc(labels.tolist(), scores.tolist()), places=9)

    def test_service_response_fields(self) -> None:
        from src.recommendation.pipeline import RecommendationService

        service = RecommendationService(self.config.path, rerank_config=None, use_stage3=False)
        res = service.recommend("U00002", top_n=10, recent_clicks=["0000000003"])
        self.assertEqual(res["user_id"], "U00002")
        self.assertEqual(len(res["recommendations"]), 10)
        for rec in res["recommendations"]:
            self.assertEqual(set(rec), {"product_id", "score", "reason", "is_exploration"})
        self.assertEqual(set(res["pipeline_latency"]),
                         {"candidate_ms", "ranking_ms", "reranking_ms", "total_ms"})
        self.assertIn("0000000003", res["session_context"]["recent_clicks"])
        self.assertEqual(set(res["session_context"]), {"recent_clicks", "session_interest"})

    def test_retrain_one_command_writes_metrics(self) -> None:
        import json

        from src.recommendation.pipeline import retrain

        retrain(self.config)
        metrics = json.loads(Path(self.config.evaluation["dashboard_metrics_path"]).read_text("utf-8"))
        for key in ("recall_at_300", "auc", "hit_rate_at_50", "ndcg_at_50", "coverage"):
            self.assertTrue(0 <= metrics[key] <= 1, key)
        self.assertTrue((self.config.output_dir / "user_state.pkl").is_file())

    def test_evaluation_report(self) -> None:
        import json

        from src.evaluation import evaluate_stage12

        report = evaluate_stage12.run(self.config)
        self.assertIn("two_tower_faiss_recall_at_50", report["stage1"])
        self.assertGreater(report["stage1"]["two_tower_faiss_recall_at_50"], 0)
        self.assertTrue(0 <= report["stage2"]["auc_ctr_deepfm"] <= 1)
        self.assertIn("two_tower+deepfm", report["strategies"])
        json.dumps(report)                                # serializable


if __name__ == "__main__":
    unittest.main()
