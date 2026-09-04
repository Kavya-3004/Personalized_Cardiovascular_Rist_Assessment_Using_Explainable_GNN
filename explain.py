"""
SHAP Explainability Module
Personalized Cardiovascular Risk Assessment

This module computes individualized patient explanations using TreeSHAP on the Champion
Random Forest classifier. It attributes the model's predicted disease probability to the
underlying clinical features, clearly distinguishing between risk-increasing and risk-decreasing
contributions without implying clinical causation.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Union, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.predict import (
    load_model,
    _format_patient_input,
    FEATURE_NAMES,
    DEFAULT_MODEL_PATH
)

# Human-readable feature names
DISPLAY_NAMES = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting Blood Sugar",
    "restecg": "Resting ECG",
    "thalach": "Maximum Heart Rate",
    "exang": "Exercise-Induced Angina",
    "oldpeak": "ST Depression",
    "slope": "ST Segment Slope",
    "ca": "Major Vessels",
    "thal": "Thalassemia Status"
}

# Global explainer cache
_CACHED_EXPLAINER = None
_CACHED_EXPLAINER_MODEL = None


def get_tree_explainer(model=None):
    """
    Initializes and caches a shap.TreeExplainer for the trained Random Forest model.
    """
    global _CACHED_EXPLAINER, _CACHED_EXPLAINER_MODEL
    if model is None:
        model = load_model(DEFAULT_MODEL_PATH)
    if _CACHED_EXPLAINER is None or _CACHED_EXPLAINER_MODEL is not model:
        _CACHED_EXPLAINER = shap.TreeExplainer(model)
        _CACHED_EXPLAINER_MODEL = model
    return _CACHED_EXPLAINER


def _extract_disease_shap_values(raw_shap: Any) -> np.ndarray:
    """
    Extracts SHAP values corresponding to Class 1 (Heart Disease) across
    different versions and formats of the SHAP library.

    Handles:
    - 3D numpy array: shape (N, num_features, 2) -> extracts index 1 on axis 2
    - List of arrays: [shap_class_0, shap_class_1] -> extracts index 1
    - 2D array: shape (N, num_features)
    """
    if isinstance(raw_shap, list):
        if len(raw_shap) > 1:
            return np.array(raw_shap[1])
        return np.array(raw_shap[0])
    elif isinstance(raw_shap, np.ndarray):
        if raw_shap.ndim == 3:
            return raw_shap[:, :, 1]
        elif raw_shap.ndim == 2:
            return raw_shap
        else:
            raise ValueError(f"Unexpected SHAP array dimension: {raw_shap.ndim}")
    else:
        # shap.Explanation object
        if hasattr(raw_shap, "values"):
            vals = raw_shap.values
            if vals.ndim == 3:
                return vals[:, :, 1]
            return vals
        raise TypeError(f"Unsupported SHAP output type: {type(raw_shap)}")


def explain_prediction(
    patient_data: Union[Dict[str, Any], pd.DataFrame, pd.Series],
    top_n: int = 5,
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH
) -> List[Dict[str, Any]]:
    """
    Generates local SHAP explanations for an individual patient's cardiovascular risk prediction.

    Parameters:
        patient_data: Dictionary or DataFrame containing the 13 clinical features.
        top_n: Number of top contributing features to return (default 5).
        model_path: Path to the serialized model file.

    Returns:
        List of dictionaries detailing the top N contributing factors:
        [
            {
                "feature": str,
                "display_name": str,
                "shap_value": float,
                "direction": "increases risk" | "decreases risk",
                "importance": float (absolute value of shap_value)
            },
            ...
        ]
    """
    model = load_model(model_path)
    explainer = get_tree_explainer(model)

    # Format and sanitize inputs
    X = _format_patient_input(patient_data)

    # Compute SHAP values
    raw_shap = explainer.shap_values(X)
    disease_shap = _extract_disease_shap_values(raw_shap)

    # For an individual patient (first row)
    patient_shap = disease_shap[0]

    feature_explanations = []
    for feat_name, s_val in zip(FEATURE_NAMES, patient_shap):
        s_float = float(s_val)
        abs_imp = abs(s_float)
        direction = "increases risk" if s_float > 0 else "decreases risk"

        feature_explanations.append({
            "feature": feat_name,
            "display_name": DISPLAY_NAMES.get(feat_name, feat_name),
            "shap_value": round(s_float, 4),
            "direction": direction,
            "importance": round(abs_imp, 4)
        })

    # Sort descending by absolute contribution
    feature_explanations.sort(key=lambda item: item["importance"], reverse=True)

    # Return top N features
    return feature_explanations[:top_n]


def plot_patient_shap(
    patient_data: Union[Dict[str, Any], pd.DataFrame, pd.Series],
    output_path: str = "outputs/shap_patient_example.png",
    top_n: int = 5,
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH
) -> str:
    """
    Generates a clean horizontal bar chart displaying the top N contributing features
    and their positive / negative impact on model-predicted cardiac risk.

    Saves the visualization to output_path and returns the output path.
    """
    top_features = explain_prediction(patient_data, top_n=top_n, model_path=model_path)

    # Extract display names and values (reversed so largest appears at the top)
    names = [f["display_name"] for f in top_features][::-1]
    values = [f["shap_value"] for f in top_features][::-1]
    directions = [f["direction"] for f in top_features][::-1]

    # Color coding: red/coral for risk increase, blue/teal for risk decrease
    colors = ["#e63946" if d == "increases risk" else "#2a9d8f" for d in directions]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(names, values, color=colors, height=0.55, edgecolor="none", alpha=0.9)

    # Reference line at zero
    ax.axvline(0, color="#6c757d", linestyle="--", linewidth=1.2, alpha=0.8)

    # Annotate value labels on each bar
    for bar, val in zip(bars, values):
        sign = "+" if val > 0 else ""
        text = f" {sign}{val:.3f}" if val >= 0 else f"{val:.3f} "
        ha = "left" if val >= 0 else "right"
        offset = 0.005 if val >= 0 else -0.005
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                text, va="center", ha=ha, fontsize=9.5, fontweight="bold", color="#212529")

    # Styling
    ax.set_title("Top Contributing Factors to Model-Predicted Risk (SHAP)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("SHAP Value (Contribution toward Cardiac Disease Probability)", fontsize=10, labelpad=8)
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    # Legend / Annotation explaining presentation nature
    increase_patch = plt.Rectangle((0, 0), 1, 1, fc="#e63946", label="Increases model risk")
    decrease_patch = plt.Rectangle((0, 0), 1, 1, fc="#2a9d8f", label="Decreases model risk")
    ax.legend(handles=[increase_patch, decrease_patch], loc="lower right", framealpha=0.9, fontsize=9)

    # Explanation note
    fig.text(0.5, -0.04,
             "Note: Features describe model-attributed probability shifts, not biological or medical causation.",
             ha="center", fontsize=8.5, fontstyle="italic", color="#555555")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path
