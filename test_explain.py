"""
SHAP Explanation Unit and Verification Test Suite
Tests local SHAP explainability on real patient records from data/processed/test.csv.
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
    FEATURE_NAMES,
    DEFAULT_MODEL_PATH
)
from src.explain import (
    explain_prediction,
    plot_patient_shap,
    DISPLAY_NAMES
)


class TestShapExplainability(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        test_csv_path = PROJECT_ROOT / "data" / "processed" / "test.csv"
        cls.test_df = pd.read_csv(test_csv_path)
        cls.sample_patient = cls.test_df.iloc[0].to_dict()

    def test_model_exists(self):
        """Confirms models/best_model.pkl is present."""
        self.assertTrue(DEFAULT_MODEL_PATH.exists(), "models/best_model.pkl missing!")

    def test_explanation_returns_top_5(self):
        """Verifies exactly 5 features are returned by default."""
        explanation = explain_prediction(self.sample_patient, top_n=5)
        self.assertIsInstance(explanation, list)
        self.assertEqual(len(explanation), 5)

    def test_features_belong_to_trained_set(self):
        """Verifies all returned features belong to the 13 clinical features."""
        explanation = explain_prediction(self.sample_patient, top_n=5)
        for item in explanation:
            self.assertIn(item["feature"], FEATURE_NAMES)
            self.assertEqual(item["display_name"], DISPLAY_NAMES[item["feature"]])

    def test_shap_values_numeric_and_sorted(self):
        """Verifies SHAP values are numeric, importance is non-negative and sorted descending."""
        explanation = explain_prediction(self.sample_patient, top_n=5)
        importances = []
        for item in explanation:
            self.assertIsInstance(item["shap_value"], float)
            self.assertIsInstance(item["importance"], float)
            self.assertGreaterEqual(item["importance"], 0.0)
            self.assertAlmostEqual(item["importance"], abs(item["shap_value"]), places=4)
            importances.append(item["importance"])

        # Verify sorted descending
        self.assertEqual(importances, sorted(importances, reverse=True))

    def test_direction_matches_sign(self):
        """Verifies direction is 'increases risk' for positive and 'decreases risk' for negative."""
        explanation = explain_prediction(self.sample_patient, top_n=5)
        for item in explanation:
            if item["shap_value"] > 0:
                self.assertEqual(item["direction"], "increases risk")
            else:
                self.assertEqual(item["direction"], "decreases risk")

    def test_target_leakage_immunity(self):
        """Verifies target or metadata columns do not alter SHAP values."""
        tampered_patient = self.sample_patient.copy()
        tampered_patient["target"] = 1
        tampered_patient["target_binary"] = 1
        tampered_patient["split"] = "tampered"

        exp_clean = explain_prediction(self.sample_patient, top_n=5)
        exp_tampered = explain_prediction(tampered_patient, top_n=5)

        for c, t in zip(exp_clean, exp_tampered):
            self.assertEqual(c["feature"], t["feature"])
            self.assertAlmostEqual(c["shap_value"], t["shap_value"], places=4)

    def test_deterministic_output(self):
        """Verifies SHAP explanations are deterministic for identical input."""
        res1 = explain_prediction(self.sample_patient, top_n=5)
        res2 = explain_prediction(self.sample_patient, top_n=5)

        for r1, r2 in zip(res1, res2):
            self.assertEqual(r1["feature"], r2["feature"])
            self.assertEqual(r1["shap_value"], r2["shap_value"])

    def test_visualization_generation(self):
        """Verifies visualization can be generated and saved to outputs/shap_patient_example.png."""
        output_file = PROJECT_ROOT / "outputs" / "shap_patient_example.png"
        generated_path = plot_patient_shap(self.sample_patient, output_path=str(output_file), top_n=5)

        self.assertTrue(output_file.exists())
        self.assertGreater(output_file.stat().st_size, 1000)


def print_formatted_example():
    test_csv_path = PROJECT_ROOT / "data" / "processed" / "test.csv"
    test_df = pd.read_csv(test_csv_path)

    # Use first patient from test set
    patient = test_df.iloc[0].to_dict()

    # Predict risk
    pred = predict_cardiovascular_risk(patient)
    prob_pct = pred["disease_probability_percent"]

    # Explain prediction
    explanation = explain_prediction(patient, top_n=5)

    print("\n" + "=" * 65)
    print("SHAP Explanation Demonstration (Real Test-Set Patient)")
    print("=" * 65)
    print("SHAP Explanation")
    print("----------------")
    print(f"Patient predicted disease probability: {prob_pct:.0f}%\n")
    print("Top contributing factors:\n")

    for i, item in enumerate(explanation, 1):
        name = item["display_name"]
        val = item["shap_value"]
        sign = f"+{val:.4f}" if val >= 0 else f"{val:.4f}"
        direction = item["direction"]
        print(f"{i}. {name:<24} {sign:>8}  {direction}")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestShapExplainability)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print_formatted_example()
    else:
        sys.exit(1)
