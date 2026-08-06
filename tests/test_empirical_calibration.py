import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import calibrate_empirical_model as calibration  # noqa: E402


class EmpiricalCalibrationTests(unittest.TestCase):
    def test_targets_are_machine_readable(self):
        targets = calibration.load_targets(ROOT / "data" / "empirical_targets.json")
        required = {
            "major_error_detection_rate",
            "interreviewer_reliability",
            "interreviewer_kappa",
            "positive_result_recommendation_rate",
            "null_result_recommendation_rate",
        }
        self.assertTrue(required.issubset(targets["targets"]))

    def test_kappa_identity(self):
        values = np.array([True, False, True, False])
        self.assertAlmostEqual(calibration.cohen_kappa(values, values), 1.0)

    def test_best_fit_is_within_declared_tolerances(self):
        summary_path = ROOT / "results" / "empirical_calibration" / "calibration_summary.csv"
        if not summary_path.exists():
            self.skipTest("Committed calibration summary not present")
        import pandas as pd

        summary = pd.read_csv(summary_path)
        fitted = summary[summary["fit_role"].isin(["primary", "secondary"])]
        standardized_error = (
            (fitted["best_fit"] - fitted["target"]).abs() / fitted["tolerance"]
        )
        self.assertTrue((standardized_error <= 1.0).all())

    def test_best_parameter_file_is_finite(self):
        path = ROOT / "results" / "empirical_calibration" / "best_fit_parameters.json"
        params = json.loads(path.read_text(encoding="utf-8"))[0]
        self.assertTrue(all(np.isfinite(float(value)) for value in params.values()))


if __name__ == "__main__":
    unittest.main()
