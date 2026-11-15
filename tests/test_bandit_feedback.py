"""Exploration feedback and bandit persistence (json file and a fake Redis client).

python -m unittest tests.test_bandit_feedback -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.recommendation.bandit import CategoryBandit
from src.recommendation.bandit_store import BanditStore
from src.recommendation.rerank import RerankConfig, make_synthetic

ROOT = Path(__file__).resolve().parents[1]


class FakeRedis:
    """The hash commands BanditStore uses, in memory."""

    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}

    def hgetall(self, key: str) -> dict[str, str]:
        return dict(self.hashes.get(key, {}))

    def hincrby(self, key: str, field: str, amount: int) -> None:
        h = self.hashes.setdefault(key, {})
        h[field] = str(int(h.get(field, "0")) + amount)

    def hincrbyfloat(self, key: str, field: str, amount: float) -> None:
        h = self.hashes.setdefault(key, {})
        h[field] = str(float(h.get(field, "0")) + amount)

    def pipeline(self) -> "FakeRedis":
        return self

    def execute(self) -> None:
        pass


class BanditStoreTest(unittest.TestCase):
    def test_json_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = BanditStore("json", Path(tmp) / "state.json")
            bandit = CategoryBandit(["A", "B"])
            store.record(bandit, "A", 1.0)
            store.record(bandit, "A", 0.0)
            store.record(bandit, "C", 1.0)                 # new arm appears
            restored = CategoryBandit(["A", "B"])
            self.assertTrue(store.load_into(restored))
            self.assertEqual(restored.pulls, {"A": 2, "B": 0, "C": 1})
            self.assertEqual(restored.mean("A"), 0.5)
            self.assertIn("C", restored.arms)

    def test_empty_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(BanditStore("json", Path(tmp) / "none.json")
                             .load_into(CategoryBandit(["A"])))

    def test_redis_workers_share_counts(self) -> None:
        client = FakeRedis()
        worker1, worker2 = (BanditStore("redis", client=client) for _ in range(2))
        b1, b2 = CategoryBandit(["A", "B"]), CategoryBandit(["A", "B"])
        worker1.record(b1, "A", 1.0)
        worker2.record(b2, "A", 0.0)                       # sees worker1's pull too
        self.assertEqual(b2.pulls["A"], 2)
        self.assertEqual(b2.mean("A"), 0.5)

    def test_bad_backend(self) -> None:
        with self.assertRaises(ValueError):
            BanditStore("sqlite", Path("x"))


class FeedbackTest(unittest.TestCase):
    def test_feedback_steers_exploration(self) -> None:
        config = RerankConfig.from_yaml(ROOT / "config" / "rerank.yaml")
        reranker, items = make_synthetic(config)
        with tempfile.TemporaryDirectory() as tmp:
            reranker.bandit_store = BanditStore("json", Path(tmp) / "state.json")
            target = next(p for p in sorted(items) if items[p].category == "Bag")
            other = next(p for p in sorted(items) if items[p].category == "Coat")
            for _ in range(30):
                reranker.record_feedback(target, clicked=True)
                reranker.record_feedback(other, clicked=False)
            self.assertEqual(reranker.bandit.mean("Bag"), 1.0)
            self.assertEqual(reranker.bandit.mean("Coat"), 0.0)
            # greedy view of the learned means (UCB would first try the untried arms)
            reranker.bandit.algorithm, reranker.bandit.epsilon = "epsilon_greedy", 0.0
            self.assertEqual(reranker.bandit.select(1), ["Bag"])
            saved = CategoryBandit(["Bag"])
            reranker.bandit_store.load_into(saved)
            self.assertEqual(saved.pulls["Bag"], 30)


if __name__ == "__main__":
    unittest.main()
