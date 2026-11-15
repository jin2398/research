import json
import unittest

from src.dashboard.ab_experiment import (
    summarize_experiment_csv, check_normal_approximation,
    make_report, report_markdown, report_json,
)


class ExperimentInputTests(unittest.TestCase):
    def csv(self, body):
        return ("user_id,group,converted\n" + body).encode("utf-8")

    def test_good_csv_and_report(self):
        content = self.csv("u1,control,0\nu2,control,1\nu3,treatment,1\nu4,treatment,0\n")
        info, result = summarize_experiment_csv(content)
        self.assertEqual(info["row_count"], 4)
        self.assertEqual(info["groups"]["control"]["converted"], 1)
        self.assertAlmostEqual(result.p_value, 1)
        report = make_report(info, result, data_source="user upload", experimental_design_verified=False)
        self.assertIn("시뮬레이션", report["status"])
        self.assertFalse(report["randomization_verified_independently"])
        self.assertIn("# RE:SEARCH", report_markdown(report))
        self.assertEqual(json.loads(report_json(report))["results"]["p_value"], 1)

    def test_duplicate_user(self):
        with self.assertRaisesRegex(ValueError, "중복"):
            summarize_experiment_csv(self.csv("u1,control,0\nu1,treatment,1\n"))

    def test_invalid_conversion(self):
        with self.assertRaisesRegex(ValueError, "0 또는 1"):
            summarize_experiment_csv(self.csv("u1,control,2\nu2,treatment,1\n"))

    def test_missing_column(self):
        with self.assertRaisesRegex(ValueError, "필수 컬럼"):
            summarize_experiment_csv(b"user_id,converted\nu1,1")

    def test_only_one_group(self):
        with self.assertRaisesRegex(ValueError, "양쪽"):
            summarize_experiment_csv(self.csv("u1,control,0\nu2,control,1\n"))

    def test_not_event_log(self):
        with self.assertRaisesRegex(ValueError, "필수 컬럼"):
            summarize_experiment_csv(b"user_id,product_id,event_type\nu1,p1,click\n")

    def test_empty_file(self):
        with self.assertRaises(ValueError):
            summarize_experiment_csv(b"")

    def test_utf8_bom(self):
        data = b"\xef\xbb\xbf" + self.csv("u1,control,1\nu2,treatment,1\n")
        info, result = summarize_experiment_csv(data)
        self.assertEqual(info["row_count"], 2)

    def test_normal_approximation_warning(self):
        info, _ = summarize_experiment_csv(self.csv("u1,control,0\nu2,treatment,1\n"))
        self.assertFalse(check_normal_approximation(info))

    def test_row_limit(self):
        with self.assertRaisesRegex(ValueError, "최대"):
            summarize_experiment_csv(self.csv("u1,control,0\nu2,treatment,1\nu3,control,1\n"), max_rows=2)


if __name__ == "__main__":
    unittest.main()
