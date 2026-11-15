import math
import unittest

from src.dashboard.ab_test import analyze_ab_test


class ABTestTests(unittest.TestCase):
    def test_typical_case(self):
        result = analyze_ab_test(1000, 100, 1000, 130)
        self.assertAlmostEqual(result.control_cvr_pct, 10.0)
        self.assertAlmostEqual(result.treatment_cvr_pct, 13.0)
        self.assertAlmostEqual(result.lift_pct, 30.0)
        self.assertTrue(0.01 < result.p_value < 0.1)
        self.assertTrue(result.ci_low_pp < 3.0 < result.ci_high_pp)

    def test_identical_groups(self):
        result = analyze_ab_test(100, 10, 100, 10)
        self.assertAlmostEqual(result.p_value, 1.0)
        self.assertFalse(result.significant_at_005)

    def test_zero_control(self):
        result = analyze_ab_test(100, 0, 100, 5)
        self.assertIsNone(result.lift_pct)
        self.assertTrue(math.isfinite(result.ci_low_pp))
        self.assertTrue(math.isfinite(result.ci_high_pp))

    def test_zero_variance(self):
        result = analyze_ab_test(100, 0, 100, 0)
        self.assertEqual(result.p_value, 1.0)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            analyze_ab_test(10, 11, 10, 1)
        with self.assertRaises(ValueError):
            analyze_ab_test(0, 0, 10, 1)


if __name__ == "__main__":
    unittest.main()
