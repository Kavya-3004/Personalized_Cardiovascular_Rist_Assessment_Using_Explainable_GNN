"""
Model Training, Validation, Selection, and Persistence Module
Personalized Cardiovascular Risk Assessment

This module trains baseline tabular classifiers (Logistic Regression, Random Forest, XGBoost),
evaluates their performance on the validation set with a focus on clinical Sensitivity/Recall,
selects the champion model, evaluates it once on the test set, saves the models, and generates
evaluation plots.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    ConfusionMatrixDisplay
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants
FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal"
]
TARGET_COL = "target_binary"
EXCLUDE_COLS = ["target", "target_binary", "split"]


def load_data(
    processed_dir: str = "data/processed"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads unscaled and scaled splits from the data/processed directory.
    """
    path = Path(processed_dir)
    train_df = pd.read_csv(path / "train.csv")
    val_df = pd.read_csv(path / "val.csv")
    test_df = pd.read_csv(path / "test.csv")

    train_scaled_df = pd.read_csv(path / "train_scaled.csv")
    val_scaled_df = pd.read_csv(path / "val_scaled.csv")
    test_scaled_df = pd.read_csv(path / "test_scaled.csv")

    logger.info(f"Loaded datasets successfully:")
    logger.info(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
    return train_df, val_df, test_df, train_scaled_df, val_scaled_df, test_scaled_df


def prepare_features_and_targets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_scaled_df: pd.DataFrame,
    val_scaled_df: pd.DataFrame,
    test_scaled_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Separates feature matrix X and target vector y for both unscaled and scaled splits.
    Ensures target columns are strictly removed from X.
    """
    # Verify features
    for col in FEATURE_NAMES:
        assert col in train_df.columns, f"Feature {col} missing from train dataset."

    # Unscaled
    X_train = train_df[FEATURE_NAMES].copy()
    y_train = train_df[TARGET_COL].values
    X_val = val_df[FEATURE_NAMES].copy()
    y_val = val_df[TARGET_COL].values
    X_test = test_df[FEATURE_NAMES].copy()
    y_test = test_df[TARGET_COL].values

    # Scaled
    X_train_scaled = train_scaled_df[FEATURE_NAMES].copy()
    y_train_scaled = train_scaled_df[TARGET_COL].values
    X_val_scaled = val_scaled_df[FEATURE_NAMES].copy()
    y_val_scaled = val_scaled_df[TARGET_COL].values
    X_test_scaled = test_scaled_df[FEATURE_NAMES].copy()
    y_test_scaled = test_scaled_df[TARGET_COL].values

    return {
        "unscaled": {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test
        },
        "scaled": {
            "X_train": X_train_scaled, "y_train": y_train_scaled,
            "X_val": X_val_scaled, "y_val": y_val_scaled,
            "X_test": X_test_scaled, "y_test": y_test_scaled
        }
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """
    Computes standard binary classification metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist()
    }


def train_and_evaluate_baselines(data: Dict[str, Any], random_state: int = 42) -> Dict[str, Any]:
    """
    Trains Logistic Regression, Random Forest, and XGBoost models,
    and evaluates them on the validation dataset.
    """
    # 1. Logistic Regression (Trained on scaled data)
    logger.info("Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=random_state, max_iter=1000)
    lr_model.fit(data["scaled"]["X_train"], data["scaled"]["y_train"])
    lr_y_pred = lr_model.predict(data["scaled"]["X_val"])
    lr_y_prob = lr_model.predict_proba(data["scaled"]["X_val"])[:, 1]
    lr_metrics = compute_metrics(data["scaled"]["y_val"], lr_y_pred, lr_y_prob)

    # 2. Random Forest (Trained on unscaled features for natural threshold interpretability)
    logger.info("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf_model.fit(data["unscaled"]["X_train"], data["unscaled"]["y_train"])
    rf_y_pred = rf_model.predict(data["unscaled"]["X_val"])
    rf_y_prob = rf_model.predict_proba(data["unscaled"]["X_val"])[:, 1]
    rf_metrics = compute_metrics(data["unscaled"]["y_val"], rf_y_pred, rf_y_prob)

    # 3. XGBoost (Trained on unscaled features)
    logger.info("Training XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=100,
        random_state=random_state,
        eval_metric="logloss",
        learning_rate=0.05,
        max_depth=3
    )
    xgb_model.fit(data["unscaled"]["X_train"], data["unscaled"]["y_train"])
    xgb_y_pred = xgb_model.predict(data["unscaled"]["X_val"])
    xgb_y_prob = xgb_model.predict_proba(data["unscaled"]["X_val"])[:, 1]
    xgb_metrics = compute_metrics(data["unscaled"]["y_val"], xgb_y_pred, xgb_y_prob)

    models = {
        "Logistic Regression": {
            "model": lr_model,
            "use_scaled": True,
            "metrics": lr_metrics,
            "val_pred": lr_y_pred,
            "val_prob": lr_y_prob
        },
        "Random Forest": {
            "model": rf_model,
            "use_scaled": False,
            "metrics": rf_metrics,
            "val_pred": rf_y_pred,
            "val_prob": rf_y_prob
        },
        "XGBoost": {
            "model": xgb_model,
            "use_scaled": False,
            "metrics": xgb_metrics,
            "val_pred": xgb_y_pred,
            "val_prob": xgb_y_prob
        }
    }
    return models


def generate_comparison_table(models: Dict[str, Any]) -> pd.DataFrame:
    """
    Builds a standardized model comparison DataFrame.
    """
    rows = []
    for name, item in models.items():
        m = item["metrics"]
        rows.append({
            "Model": name,
            "Accuracy": round(m["accuracy"], 4),
            "Precision": round(m["precision"], 4),
            "Recall": round(m["recall"], 4),
            "F1": round(m["f1"], 4),
            "ROC-AUC": round(m["roc_auc"], 4)
        })
    df_comp = pd.DataFrame(rows)
    return df_comp


def evaluate_test_set(
    best_model_name: str,
    best_model_info: Dict[str, Any],
    data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluates the selected best model ONCE on the untouched hold-out test set.
    """
    model = best_model_info["model"]
    is_scaled = best_model_info["use_scaled"]
    split_key = "scaled" if is_scaled else "unscaled"

    X_test = data[split_key]["X_test"]
    y_test = data[split_key]["y_test"]

    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_test_pred, y_test_prob)
    report_dict = classification_report(
        y_test,
        y_test_pred,
        target_names=["Healthy (0)", "Disease (1)"],
        output_dict=True
    )
    report_text = classification_report(
        y_test,
        y_test_pred,
        target_names=["Healthy (0)", "Disease (1)"]
    )

    test_results = {
        "model_name": best_model_name,
        "is_scaled": is_scaled,
        "metrics": metrics,
        "report_text": report_text,
        "report_dict": report_dict,
        "y_test_pred": y_test_pred,
        "y_test_prob": y_test_prob,
        "y_test": y_test
    }
    return test_results


def save_models_and_artifacts(
    models: Dict[str, Any],
    best_name: str,
    test_results: Dict[str, Any],
    models_dir: str = "models",
    outputs_dir: str = "outputs"
) -> None:
    """
    Saves trained models, comparison tables, diagnostic plots, and metadata.
    Does NOT overwrite scaler.pkl.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    # 1. Save individual models
    model_filename_map = {
        "Logistic Regression": "logistic_regression.pkl",
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl"
    }

    for name, item in models.items():
        fname = model_filename_map[name]
        save_path = os.path.join(models_dir, fname)
        joblib.dump(item["model"], save_path)
        logger.info(f"Saved {name} to {save_path}")

    # 2. Save best model separately
    best_save_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(models[best_name]["model"], best_save_path)
    logger.info(f"Saved Champion model ({best_name}) to {best_save_path}")

    # 3. Save model metadata
    metadata = {
        "champion_model": best_name,
        "champion_uses_scaled_features": models[best_name]["use_scaled"],
        "feature_names": FEATURE_NAMES,
        "validation_metrics": models[best_name]["metrics"],
        "test_metrics": test_results["metrics"],
        "models_saved": {
            "Logistic Regression": "models/logistic_regression.pkl",
            "Random Forest": "models/random_forest.pkl",
            "XGBoost": "models/xgboost.pkl",
            "Champion": "models/best_model.pkl"
        }
    }
    with open(os.path.join(models_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info("Saved models/model_metadata.json")

    # 4. Save comparison table CSV
    comp_df = generate_comparison_table(models)
    comp_path = os.path.join(outputs_dir, "model_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    logger.info(f"Saved comparison table to {comp_path}")

    # 5. Save Confusion Matrix Plot for Best Model (Validation and Test)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Validation CM
    val_cm = np.array(models[best_name]["metrics"]["confusion_matrix"])
    sns.heatmap(
        val_cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
        xticklabels=["Healthy (0)", "Disease (1)"],
        yticklabels=["Healthy (0)", "Disease (1)"]
    )
    axes[0].set_title(f"Validation CM: {best_name}\n(Recall: {models[best_name]['metrics']['recall']:.2%})", fontsize=12)
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # Test CM
    test_cm = np.array(test_results["metrics"]["confusion_matrix"])
    sns.heatmap(
        test_cm, annot=True, fmt="d", cmap="Greens", cbar=False, ax=axes[1],
        xticklabels=["Healthy (0)", "Disease (1)"],
        yticklabels=["Healthy (0)", "Disease (1)"]
    )
    axes[1].set_title(f"Hold-out Test CM: {best_name}\n(Recall: {test_results['metrics']['recall']:.2%})", fontsize=12)
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    plt.tight_layout()
    cm_plot_path = os.path.join(outputs_dir, "confusion_matrix_best_model.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {cm_plot_path}")

    # 6. Save ROC Curves Plot (Validation Set)
    plt.figure(figsize=(8, 6))
    val_y = pd.read_csv("data/processed/val.csv")[TARGET_COL].values
    for name, item in models.items():
        fpr, tpr, _ = roc_curve(val_y, item["val_prob"])
        auc_val = item["metrics"]["roc_auc"]
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", lw=2)

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance / Random (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    plt.title("ROC Curves Comparison (Validation Set)", fontsize=13)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_plot_path = os.path.join(outputs_dir, "roc_curves.png")
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved ROC curves plot to {roc_plot_path}")

    # 7. Save Feature Importance Plot (Random Forest & XGBoost)
    rf_importances = models["Random Forest"]["model"].feature_importances_
    xgb_importances = models["XGBoost"]["model"].feature_importances_

    fi_df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Random Forest": rf_importances,
        "XGBoost": xgb_importances
    }).set_index("Feature")

    fi_df = fi_df.sort_values(by="Random Forest", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    fi_df.plot(kind="barh", ax=ax, colormap="viridis", width=0.7)
    ax.set_title("Feature Importance Comparison (Random Forest vs XGBoost)", fontsize=13)
    ax.set_xlabel("Relative Importance Score", fontsize=11)
    ax.set_ylabel("Clinical Features", fontsize=11)
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    fi_plot_path = os.path.join(outputs_dir, "feature_importance.png")
    plt.savefig(fi_plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved feature importance plot to {fi_plot_path}")


def run_pipeline() -> Dict[str, Any]:
    """
    Main orchestration routine.
    """
    logger.info("=== Loading Data Splits ===")
    t_df, v_df, te_df, ts_df, vs_df, tes_df = load_data()
    data = prepare_features_and_targets(t_df, v_df, te_df, ts_df, vs_df, tes_df)

    logger.info("=== STEP 1: Training and Evaluating Baseline Models on Validation Set ===")
    models = train_and_evaluate_baselines(data)

    logger.info("=== STEP 2: Generating Model Comparison Table ===")
    comp_df = generate_comparison_table(models)
    print("\n--- VALIDATION MODEL COMPARISON TABLE ---")
    print(comp_df.to_string(index=False))

    # Determine Best Model:
    # High clinical priority: Recall (Sensitivity) for heart disease detection, followed by ROC-AUC and F1
    best_name = None
    best_score = -1.0
    for name, item in models.items():
        m = item["metrics"]
        # Combined medical ranking score: heavily weight Recall (0.5), ROC-AUC (0.3), F1 (0.2)
        score = 0.50 * m["recall"] + 0.30 * m["roc_auc"] + 0.20 * m["f1"]
        if score > best_score:
            best_score = score
            best_name = name

    logger.info(f"Selected Champion Model based on Validation Performance: {best_name}")

    logger.info(f"=== STEP 3: Evaluating Champion Model ({best_name}) ONCE on Test Set ===")
    test_results = evaluate_test_set(best_name, models[best_name], data)
    print(f"\n--- TEST SET CLASSIFICATION REPORT ({best_name}) ---")
    print(test_results["report_text"])

    logger.info("=== STEP 4 & 5: Saving Models and Generating Output Plots ===")
    save_models_and_artifacts(models, best_name, test_results)

    return {
        "comparison_table": comp_df,
        "best_model_name": best_name,
        "models": models,
        "test_results": test_results
    }


if __name__ == "__main__":
    run_pipeline()
