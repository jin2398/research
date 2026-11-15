import json
import tempfile
import unittest
from pathlib import Path
from src.dashboard.integration import (
    load_recommendation_metrics, load_search_reports, load_session_report, search_compare_rows,
)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, name, value):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value), encoding="utf-8")

    def test_no_reports_show_missing_not_fake(self):
        self.assertIsNone(load_session_report(self.root))
        self.assertIsNone(load_recommendation_metrics(self.root))
        self.assertTrue(all(v is None for v in load_search_reports(self.root).values()))

    def test_search_pilot_comparison(self):
        for name, key, mrr in [
            ("docs/search_baseline_pilot.json", "baseline", .6),
            ("docs/search_clip_pilot.json", "clip", .4),
        ]:
            self.write(name, {"metrics": {"mrr": mrr, "ndcg_at_10": .3, "recall_at_10": .1, "queries": 100},
                              "covered_query_metrics": {"mrr": .7, "ndcg_at_10": .5, "recall_at_10": .2, "queries": 50}})
        rows = search_compare_rows(load_search_reports(self.root))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["MRR@10"], .6)
        self.assertEqual(search_compare_rows(load_search_reports(self.root), covered_only=True)[1]["MRR@10"], .7)

    def test_rec_metrics_validation(self):
        self.write("data/recommendation_metrics.json", {"auc": .71, "hit_rate_at_50": .19})
        self.assertEqual(load_recommendation_metrics(self.root), {"auc": .71, "hit_rate_at_50": .19})
        self.write("data/recommendation_metrics.json", {"auc": 1.2})
        with self.assertRaises(ValueError):
            load_recommendation_metrics(self.root)


if __name__ == "__main__":
    unittest.main()
