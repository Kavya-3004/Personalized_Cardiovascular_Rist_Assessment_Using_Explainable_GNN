"""
Prediction and Inference Module
Personalized Cardiovascular Risk Assessment

This module provides reusable inference functions for the Champion Random Forest model
(models/best_model.pkl). It ensures strict feature ordering, extracts predicted probabilities,
and provides a presentation layer risk band (LOW / MEDIUM / HIGH) alongside human-readable
feature summaries for the user interface.
"""

import os
from pathlib import Path
from typing import Dict, Any, Union, List

import joblib
import numpy as np
import pandas as pd

# Path to the trained champion model
DEFAULT_MODEL_PATH = Path("models/best_model.pkl")

# Exact feature ordering required by the trained model
FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal"
]

# Verified categorical mappings based on the UCI Cleveland dataset & preprocessing encoding
CATEGORICAL_MAPPINGS = {
    "sex": {
        0.0: "Female",
        1.0: "Male"
    },
    # Verified project encoding: 1-indexed (1: Typical Angina, 2: Atypical, 3: Non-anginal, 4: Asymptomatic)
    "cp": {
        1.0: "Typical Angina",
        2.0: "Atypical Angina",
        3.0: "Non-anginal Pain",
        4.0: "Asymptomatic"
    },
    "fbs": {
        0.0: "<= 120 mg/dl (Normal)",
        1.0: "> 120 mg/dl (Elevated)"
    },
    # Verified project encoding: 0: Normal, 1: ST-T Wave Abnormality, 2: Left Ventricular Hypertrophy
    "restecg": {
        0.0: "Normal",
        1.0: "ST-T Wave Abnormality",
        2.0: "Left Ventricular Hypertrophy"
    },
    "exang": {
        0.0: "No",
        1.0: "Yes"
    },
    # Verified project encoding: 1: Upsloping, 2: Flat, 3: Downsloping
    "slope": {
        1.0: "Upsloping",
        2.0: "Flat",
        3.0: "Downsloping"
    },
    # Verified project encoding: 3: Normal, 6: Fixed Defect, 7: Reversible Defect
    "thal": {
        3.0: "Normal",
        6.0: "Fixed Defect",
        7.0: "Reversible Defect"
    }
}

# Cache for loaded model to avoid repetitive disk I/O
_CACHED_MODEL = None
_CACHED_MODEL_PATH = None


def load_model(model_path: Union[str, Path] = DEFAULT_MODEL_PATH):
    """
    Loads and caches the trained machine learning model from disk.
    """
    global _CACHED_MODEL, _CACHED_MODEL_PATH
    model_path = Path(model_path)
    if _CACHED_MODEL is None or _CACHED_MODEL_PATH != str(model_path):
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path.resolve()}")
        _CACHED_MODEL = joblib.load(model_path)
        _CACHED_MODEL_PATH = str(model_path)
    return _CACHED_MODEL


def _format_patient_input(patient_data: Union[Dict[str, Any], pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """
    Validates and formats raw patient inputs into a DataFrame with exact feature ordering.
    Strictly removes target or auxiliary columns (e.g., target, target_binary, split).
    """
    if isinstance(patient_data, dict):
        df = pd.DataFrame([patient_data])
    elif isinstance(patient_data, pd.Series):
        df = pd.DataFrame([patient_data.to_dict()])
    elif isinstance(patient_data, pd.DataFrame):
        df = patient_data.copy()
    else:
        raise TypeError(f"Unsupported patient_data type: {type(patient_data)}. Expected dict or DataFrame.")

    # Check for missing required features
    missing = [col for col in FEATURE_NAMES if col not in df.columns]
    if missing:
        raise ValueError(f"Input patient data is missing required features: {missing}")

    # Extract strictly the 13 clinical features in the verified trained order
    # Note: No scaling is applied as the champion Random Forest was trained on unscaled features
    df_formatted = df[FEATURE_NAMES].astype(float)
    return df_formatted


def _assign_risk_level(probability: float) -> str:
    """
    Presentation layer helper: assigns a descriptive risk band based on predicted probability.

    NOTE: These are project presentation thresholds and are NOT medically validated clinical thresholds.
    The actual underlying model prediction remains binary (0: No Heart Disease, 1: Heart Disease).
    """
    if probability < 0.30:
        return "LOW"
    elif probability <= 0.60:
        return "MEDIUM"
    else:
        return "HIGH"


def predict_cardiovascular_risk(
    patient_data: Union[Dict[str, Any], pd.DataFrame, pd.Series],
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Infers cardiovascular disease risk for one or more patient records.

    Parameters:
        patient_data: Dictionary or pandas DataFrame containing the 13 clinical features.
        model_path: Path to the serialized model file (defaults to models/best_model.pkl).

    Returns:
        Structured result containing:
            - predicted_class: int (0 or 1)
            - disease_probability: float (0.0 to 1.0)
            - disease_probability_percent: float (0.0% to 100.0%)
            - prediction_label: str ("No Heart Disease" or "Heart Disease Detected")
            - risk_level: str ("LOW", "MEDIUM", "HIGH") [Presentation layer band]
    """
    model = load_model(model_path)
    X = _format_patient_input(patient_data)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    results = []
    for pred, prob in zip(predictions, probabilities):
        pred_int = int(pred)
        prob_float = float(prob)
        prob_pct = round(prob_float * 100.0, 2)
        label = "Heart Disease Detected" if pred_int == 1 else "No Heart Disease"
        risk = _assign_risk_level(prob_float)

        results.append({
            "predicted_class": pred_int,
            "prediction_label": label,
            "disease_probability": prob_float,
            "disease_probability_percent": prob_pct,
            "risk_level": risk
        })

    # Return single dict if a single patient was passed, else list
    if len(results) == 1 and (isinstance(patient_data, dict) or isinstance(patient_data, pd.Series) or len(patient_data) == 1):
        return results[0]
    return results


def get_patient_summary(
    patient_data: Union[Dict[str, Any], pd.DataFrame, pd.Series]
) -> Dict[str, Any]:
    """
    Extracts and maps raw clinical patient inputs into a human-readable dictionary
    suitable for user interface rendering and clinical reporting.

    Uses actual patient inputs and verified dataset categorical encodings.
    """
    if isinstance(patient_data, pd.DataFrame):
        data = patient_data.iloc[0].to_dict()
    elif isinstance(patient_data, pd.Series):
        data = patient_data.to_dict()
    elif isinstance(patient_data, dict):
        data = patient_data
    else:
        raise TypeError(f"Unsupported patient_data type: {type(patient_data)}")

    # Extract raw values safely
    age = float(data.get("age", 0))
    sex_val = float(data.get("sex", 0))
    cp_val = float(data.get("cp", 1))
    trestbps = float(data.get("trestbps", 0))
    chol = float(data.get("chol", 0))
    fbs_val = float(data.get("fbs", 0))
    restecg_val = float(data.get("restecg", 0))
    thalach = float(data.get("thalach", 0))
    exang_val = float(data.get("exang", 0))
    oldpeak = float(data.get("oldpeak", 0))
    slope_val = float(data.get("slope", 1))
    ca_val = float(data.get("ca", 0))
    thal_val = float(data.get("thal", 3))

    # Construct readable labels
    summary = {
        "Age": f"{int(age)} years",
        "Sex": CATEGORICAL_MAPPINGS["sex"].get(sex_val, f"Unknown ({sex_val})"),
        "Chest Pain Type": CATEGORICAL_MAPPINGS["cp"].get(cp_val, f"Type {cp_val}"),
        "Resting Blood Pressure": f"{int(trestbps)} mm Hg",
        "Cholesterol": f"{int(chol)} mg/dl",
        "Fasting Blood Sugar": CATEGORICAL_MAPPINGS["fbs"].get(fbs_val, f"Unknown ({fbs_val})"),
        "Resting ECG": CATEGORICAL_MAPPINGS["restecg"].get(restecg_val, f"Type {restecg_val}"),
        "Maximum Heart Rate": f"{int(thalach)} bpm",
        "Exercise-Induced Angina": CATEGORICAL_MAPPINGS["exang"].get(exang_val, f"Unknown ({exang_val})"),
        "ST Depression": f"{oldpeak:.1f}",
        "ST Segment Slope": CATEGORICAL_MAPPINGS["slope"].get(slope_val, f"Type {slope_val}"),
        "Number of Major Vessels": f"{int(ca_val)} vessels",
        "Thalassemia Status": CATEGORICAL_MAPPINGS["thal"].get(thal_val, f"Status {thal_val}")
    }
    return summary
