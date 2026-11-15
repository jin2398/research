import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

from ct_monitor import assess_ct, bump_minor_version, monitor_once, read_hit_rate


class CTMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.logs = self.root / "events.csv"
        self.metrics = self.root / "metrics.json"
        self.state = self.root / "state.json"
        self.metrics.write_text('{"hit_rate_at_50": 0.20}', encoding="utf-8")

    def make_logs(self, count: int) -> None:
        with self.logs.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["event_id"])
            writer.writerows(([n] for n in range(count)))

    def set_baseline(self, count: int = 0) -> None:
        self.make_logs(count)
        status = monitor_once(self.logs, self.metrics, self.state, initialize_baseline=True)
        self.assertTrue(status["baseline_initialized"])

    def test_requires_baseline_before_triggering(self):
        self.make_logs(10_000)
        status = monitor_once(self.logs, self.metrics, self.state, train_command=f'"{sys.executable}" -c "print(123)"')
        self.assertFalse(status["training_due"])
        self.assertFalse(status["trained"])
        self.assertFalse(status["baseline_initialized"])
        self.assertTrue(any("Baseline not initialized" in s for s in status["alerts"]))

    def test_boundary_alert_and_no_training_without_command(self):
        self.set_baseline()
        self.make_logs(10_000)
        status = monitor_once(self.logs, self.metrics, self.state)
        self.assertTrue(status["training_due"])
        self.assertFalse(status["trained"])
        self.assertEqual(status["model_version"], "v1.0")
        self.assertTrue(any("ALERT" in msg for msg in status["alerts"]))
        self.assertEqual([x["event"] for x in status["history"]].count("training_due"), 1)
        again = monitor_once(self.logs, self.metrics, self.state)
        self.assertEqual([x["event"] for x in again["history"]].count("training_due"), 1)

    def test_training_bumps_version_once_and_records_history(self):
        self.set_baseline()
        self.make_logs(10_000)
        command = f'"{sys.executable}" -c "print(123)"'
        first = monitor_once(self.logs, self.metrics, self.state, train_command=command)
        self.assertTrue(first["trained"])
        self.assertEqual(first["model_version"], "v1.1")
        self.assertEqual(first["new_logs_since_training"], 0)
        self.assertIn("training_completed", [x["event"] for x in first["history"]])
        second = monitor_once(self.logs, self.metrics, self.state, train_command=command)
        self.assertFalse(second["trained"])
        self.assertEqual(second["model_version"], "v1.1")

    def test_failed_training_leaves_version(self):
        self.set_baseline()
        self.make_logs(10_000)
        command = f'"{sys.executable}" -c "import sys;sys.exit(4)"'
        status = monitor_once(self.logs, self.metrics, self.state, train_command=command)
        self.assertEqual(status["model_version"], "v1.0")
        self.assertFalse(status["trained"])
        self.assertTrue(status["training_due"])
        self.assertIsNotNone(status["training_error"])
        self.assertIn("training_failed", [x["event"] for x in status["history"]])

    def test_initial_baseline_does_not_later_reset(self):
        self.set_baseline(1000)
        self.make_logs(1200)
        state = monitor_once(self.logs, self.metrics, self.state, initialize_baseline=True)
        self.assertEqual(state["new_logs_since_training"], 200)
        self.assertEqual(state["last_trained_log_count"], 1000)

    def test_hit_rate_from_recommendation_keys_and_0_20_boundary(self):
        self.assertEqual(read_hit_rate(self.metrics), 0.2)
        self.assertTrue(assess_ct(0.20, 10_000)["performance_alert"])
        self.assertTrue(assess_ct(.21, 10_000)["training_due"])
        self.assertFalse(assess_ct(None, 9_999)["training_due"])
        self.metrics.write_text('{"hit_rate": 0.23}', encoding="utf-8")
        self.assertEqual(read_hit_rate(self.metrics), .23)

    def test_missing_metrics_doesnt_fake_hit_rate(self):
        self.metrics.unlink()
        self.set_baseline()
        status = monitor_once(self.logs, self.metrics, self.state)
        self.assertIsNone(status["hit_rate"])
        self.assertFalse(status["performance_alert"])

    def test_version(self):
        self.assertEqual(bump_minor_version("v1.0"), "v1.1")
        self.assertEqual(bump_minor_version("v2.9"), "v2.10")


if __name__ == "__main__":
    unittest.main()
