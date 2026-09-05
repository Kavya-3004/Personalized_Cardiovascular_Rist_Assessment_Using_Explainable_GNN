"""
Inference Unit and Validation Test Suite
Tests the predict_cardiovascular_risk and get_patient_summary functions
using real patient records from data/processed/test.csv.
"""

import os
import sys
import unittest
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import (
    predict_cardiovascular_risk,
    get_patient_summary,
    FEATURE_NAMES,
    DEFAULT_MODEL_PATH
)


class TestCardiovascularRiskPrediction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        test_csv_path = PROJECT_ROOT / "data" / "processed" / "test.csv"
        cls.test_df = pd.read_csv(test_csv_path)
        cls.sample_row = cls.test_df.iloc[0].to_dict()

    def test_model_file_exists_and_unmodified(self):
        """Confirms models/best_model.pkl exists and is accessible."""
        self.assertTrue(DEFAULT_MODEL_PATH.exists(), "models/best_model.pkl does not exist!")

    def test_prediction_execution_single_dict(self):
        """Verifies prediction executes successfully for a single patient dictionary."""
        result = predict_cardiovascular_risk(self.sample_row)

        self.assertIsInstance(result, dict)
        self.assertIn("predicted_class", result)
        self.assertIn("prediction_label", result)
        self.assertIn("disease_probability", result)
        self.assertIn("disease_probability_percent", result)
        self.assertIn("risk_level", result)

    def test_probability_range(self):
        """Verifies probability is bounded strictly in [0.0, 1.0]."""
        result = predict_cardiovascular_risk(self.sample_row)
        prob = result["disease_probability"]
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)
        self.assertAlmostEqual(result["disease_probability_percent"], prob * 100.0, places=2)

    def test_risk_level_assignment(self):
        """Verifies presentation layer risk level assignment matches specified thresholds."""
        result = predict_cardiovascular_risk(self.sample_row)
        prob = result["disease_probability"]
        risk = result["risk_level"]

        if prob < 0.30:
            self.assertEqual(risk, "LOW")
        elif prob <= 0.60:
            self.assertEqual(risk, "MEDIUM")
        else:
            self.assertEqual(risk, "HIGH")

    def test_target_leakage_immunity(self):
        """Verifies that extraneous columns (target, target_binary, split) are safely ignored and stripped."""
        leaky_patient = self.sample_row.copy()
        leaky_patient["target"] = 999
        leaky_patient["target_binary"] = 999
        leaky_patient["split"] = "tampered"
        leaky_patient["bogus_metric"] = 12345

        result_clean = predict_cardiovascular_risk(self.sample_row)
        result_with_extra = predict_cardiovascular_risk(leaky_patient)

        self.assertEqual(result_clean["predicted_class"], result_with_extra["predicted_class"])
        self.assertAlmostEqual(result_clean["disease_probability"], result_with_extra["disease_probability"], places=6)

    def test_missing_feature_raises_error(self):
        """Verifies ValueError is raised if any of the 13 clinical features are omitted."""
        incomplete_patient = self.sample_row.copy()
        del incomplete_patient["thalach"]

        with self.assertRaises(ValueError):
            predict_cardiovascular_risk(incomplete_patient)

    def test_deterministic_output(self):
        """Verifies that the inference is 100% deterministic for identical inputs."""
        res1 = predict_cardiovascular_risk(self.sample_row)
        res2 = predict_cardiovascular_risk(self.sample_row)

        self.assertEqual(res1["predicted_class"], res2["predicted_class"])
        self.assertEqual(res1["disease_probability"], res2["disease_probability"])
        self.assertEqual(res1["risk_level"], res2["risk_level"])

    def test_patient_summary_generator(self):
        """Verifies that get_patient_summary formats human-readable labels for all clinical inputs."""
        summary = get_patient_summary(self.sample_row)

        required_keys = [
            "Age", "Sex", "Chest Pain Type", "Resting Blood Pressure",
            "Cholesterol", "Fasting Blood Sugar", "Resting ECG",
            "Maximum Heart Rate", "Exercise-Induced Angina", "ST Depression",
            "ST Segment Slope", "Number of Major Vessels", "Thalassemia Status"
        ]
        for key in required_keys:
            self.assertIn(key, summary)
            self.assertIsInstance(summary[key], str)
            self.assertNotIn("Unknown", summary[key])


def run_demonstration():
    print("=" * 70)
    print("RUNNING CARDIOVASCULAR RISK INFERENCE DEMONSTRATION")
    print("=" * 70)

    test_csv_path = PROJECT_ROOT / "data" / "processed" / "test.csv"
    test_df = pd.read_csv(test_csv_path)

    # Test on a patient with no heart disease (target_binary=0)
    healthy_idx = test_df[test_df["target_binary"] == 0].index[0]
    healthy_row = test_df.loc[healthy_idx].to_dict()

    print(f"\n--- Example Patient A (Actual Status: Healthy / Class 0) ---")
    summary_a = get_patient_summary(healthy_row)
    print("Clinical Profile:")
    for k, v in summary_a.items():
        print(f"  {k}: {v}")

    pred_a = predict_cardiovascular_risk(healthy_row)
    print("\nModel Inference Result:")
    for k, v in pred_a.items():
        print(f"  {k}: {v}")

    # Test on a patient with heart disease (target_binary=1)
    disease_idx = test_df[test_df["target_binary"] == 1].index[0]
    disease_row = test_df.loc[disease_idx].to_dict()

    print(f"\n--- Example Patient B (Actual Status: Cardiac Disease Present / Class 1) ---")
    summary_b = get_patient_summary(disease_row)
    print("Clinical Profile:")
    for k, v in summary_b.items():
        print(f"  {k}: {v}")

    pred_b = predict_cardiovascular_risk(disease_row)
    print("\nModel Inference Result:")
    for k, v in pred_b.items():
        print(f"  {k}: {v}")
    print("=" * 70)


if __name__ == "__main__":
    # Run unittests
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestCardiovascularRiskPrediction)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # If tests pass, print demonstration output
    if result.wasSuccessful():
        run_demonstration()
    else:
        sys.exit(1)
