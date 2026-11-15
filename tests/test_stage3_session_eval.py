"""Stage 3 additions: metrics, GRU session encoder, log loading. Synthetic data only.

python -m unittest tests.test_stage3_session_eval -v
"""

from __future__ import annotations

import csv
import math
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from src.evaluation import metrics
from src.recommendation.popularity import Event
from src.recommendation.rerank import (ItemMeta, RerankConfig, load_events, load_items,
                                       split_events, with_first_seen)

ROOT = Path(__file__).resolve().parents[1]


class MetricsTest(unittest.TestCase):
    def test_hit_recall_mrr(self) -> None:
        recs, rel = ["a", "b", "c", "d"], {"c", "z"}
        self.assertEqual(metrics.hit_rate_at_k(recs, rel, 2), 0.0)
        self.assertEqual(metrics.hit_rate_at_k(recs, rel, 3), 1.0)
        self.assertEqual(metrics.recall_at_k(recs, rel, 4), 0.5)
        self.assertAlmostEqual(metrics.reciprocal_rank_at_k(recs, rel, 4), 1 / 3)

    def test_ndcg(self) -> None:
        self.assertAlmostEqual(metrics.ndcg_at_k(["x", "a"], {"a"}, 2), 1 / math.log2(3))
        self.assertEqual(metrics.ndcg_at_k(["a", "b"], {"a", "b"}, 2), 1.0)
        self.assertEqual(metrics.ndcg_at_k(["a", "a"], {"a", "b"}, 2),
                         1 / (1 + 1 / math.log2(3)))           # duplicate counted once

    def test_coverage_and_aggregate(self) -> None:
        self.assertEqual(metrics.coverage([["a", "b"], ["b", "c"]], 10), 0.3)
        result = metrics.evaluate_rankings({"u1": ["a"], "u2": ["b"]},
                                           {"u1": {"a"}, "u2": {"x"}}, k=1, catalog_size=4)
        self.assertEqual(result["hit_rate_at_1"], 0.5)
        self.assertEqual(result["coverage"], 0.5)

    def test_auc(self) -> None:
        self.assertEqual(metrics.auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(metrics.auc([0, 1], [0.5, 0.5]), 0.5)


class LogLoadingTest(unittest.TestCase):
    def test_columns_bom_split_and_first_seen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            products, events = Path(tmp) / "products.csv", Path(tmp) / "events.csv"
            products.write_text("﻿article_id,category_small\n001,Top\n002,Bag\n", "utf-8")
            with events.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["timestamp", "user_id", "event_type", "product_id"])
                for day in range(10):                  # 10 rows: 1 search + 9 product events
                    pid = "" if day == 0 else ("001" if day < 8 else "002")
                    writer.writerow([f"2025-01-{day + 1:02d}T00:00:00", "U1",
                                     "search" if day == 0 else "view", pid])
            config = replace(RerankConfig.from_yaml(ROOT / "config" / "rerank.yaml"),
                             products_path=products, catalog_path=None,
                             events_path=events, train_ratio=0.8)
            items = load_items(config)
            self.assertEqual(set(items), {"001", "002"})       # BOM header read correctly
            train = load_events(config)
            self.assertEqual(len(train), 7)                    # 8 of 10 rows, search dropped
            self.assertTrue(all(e.product_id == "001" for e in train))
            filled = with_first_seen(items, train)
            self.assertEqual(filled["001"].created_at, datetime(2025, 1, 2))
            self.assertIsNone(filled["002"].created_at)         # unseen before the cut

    def test_split_events(self) -> None:
        events = [Event("U", str(i), "view", datetime(2025, 1, 1) + timedelta(hours=i))
                  for i in range(10)]
        train, valid, test = split_events(events, 0.8, 0.1)
        self.assertEqual((len(train), len(valid), len(test)), (8, 1, 1))


try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class SessionEncoderTest(unittest.TestCase):
    def setUp(self) -> None:
        from src.recommendation.session_encoder import SessionEncoder, SessionGRU

        rng = np.random.default_rng(42)
        vectors = rng.normal(size=(30, 16)).astype("float32")
        self.vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        self.rows = {f"P{i}": i for i in range(30)}
        self.encoder = SessionEncoder(SessionGRU(16, 8), self.vectors, self.rows, window=5)

    def test_encode_unit_vector_and_unknown_clicks(self) -> None:
        vector = self.encoder.encode(["P1", "UNKNOWN", "P2", "P2"])
        self.assertEqual(vector.shape, (16,))
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)
        self.assertIsNone(self.encoder.encode(["UNKNOWN"]))

    def test_window_and_batch_agree(self) -> None:
        clicks = [f"P{i}" for i in range(12)]
        single = self.encoder.encode(clicks)
        batch = self.encoder.encode_rows([[self.rows[p] for p in clicks], [1, 2]])[0]
        np.testing.assert_allclose(single, batch, atol=1e-5)
        np.testing.assert_allclose(single, self.encoder.encode(clicks[-5:]), atol=1e-5)

    def test_save_load_roundtrip(self) -> None:
        import torch
        from src.recommendation.session_encoder import SessionEncoder

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gru.pt"
            torch.save({"model": self.encoder.model.state_dict(),
                        "hparams": {"dim": 16, "hidden_size": 8, "num_layers": 1,
                                    "dropout": 0.0, "window": 5}}, path)
            loaded = SessionEncoder.load(path, self.vectors, self.rows)
            np.testing.assert_allclose(loaded.encode(["P3", "P4"]),
                                       self.encoder.encode(["P3", "P4"]), atol=1e-6)

    def test_reranker_uses_encoder(self) -> None:
        from src.recommendation.rerank import make_synthetic

        config = RerankConfig.from_yaml(ROOT / "config" / "rerank.yaml")
        reranker, items = make_synthetic(config, n_items=30, n_events=500)
        reranker.vectors, reranker.vector_rows = self.vectors, {p: i for i, p in enumerate(sorted(items))}
        reranker.session_encoder = replace_rows(self.encoder, reranker.vector_rows)
        clicks = sorted(items)[:3]
        expected = reranker.session_encoder.encode(clicks)
        np.testing.assert_allclose(reranker._session_vector(clicks, None), expected, atol=1e-6)


def replace_rows(encoder, rows):
    from src.recommendation.session_encoder import SessionEncoder
    return SessionEncoder(encoder.model, encoder.vectors, rows, encoder.window)


if __name__ == "__main__":
    unittest.main()
