"""
Data Preprocessing, Deduplication, and Segregation Module
Personalized Cardiovascular Risk Assessment
"""

import os
import json
import hashlib
import urllib.request
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants
RAW_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", 
    "restecg", "thalach", "exang", "oldpeak", "slope", 
    "ca", "thal", "target"
]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
TARGET_COL = "target"
BINARY_TARGET_COL = "target_binary"


def fetch_or_load_raw_data(
    raw_dir: str = "data/raw", 
    filename: str = "UCI_Heart_Disease.csv"
) -> pd.DataFrame:
    """
    Downloads or loads the raw UCI Cleveland dataset and saves it to the raw directory.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    
    uci_folder = raw_path / "uci"
    uci_folder.mkdir(parents=True, exist_ok=True)
    raw_data_file = uci_folder / "processed.cleveland.data"
    csv_file = raw_path / filename

    if not raw_data_file.exists():
        logger.info(f"Downloading raw dataset from {RAW_DATA_URL}...")
        try:
            urllib.request.urlretrieve(RAW_DATA_URL, raw_data_file)
            logger.info(f"Saved raw data to {raw_data_file}")
        except Exception as e:
            logger.error(f"Failed to download raw data: {e}")
            raise

    # Read raw data
    df = pd.read_csv(
        raw_data_file,
        header=None,
        names=COLUMN_NAMES,
        na_values=["?", "NA", "null", ""]
    )
    
    # Save a copy as CSV
    df.to_csv(csv_file, index=False)
    logger.info(f"Loaded raw dataset with shape {df.shape} and saved copy to {csv_file}")
    return df


def deduplicate_tabular(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Identifies and removes exact and feature-wise duplicate records in tabular data.
    """
    initial_count = len(df)
    
    # Check exact duplicates
    exact_duplicates = df.duplicated().sum()
    
    # Check duplicates on feature subset (excluding target)
    feature_cols = [c for c in df.columns if c not in [TARGET_COL, BINARY_TARGET_COL]]
    feature_duplicates = df.duplicated(subset=feature_cols).sum()
    
    logger.info(f"Deduplication Check: {exact_duplicates} exact duplicate rows found.")
    logger.info(f"Feature-level duplicate rows: {feature_duplicates}")
    
    # Remove duplicates
    df_dedup = df.drop_duplicates().reset_index(drop=True)
    removed_count = initial_count - len(df_dedup)
    
    dedup_stats = {
        "initial_rows": initial_count,
        "exact_duplicates": int(exact_duplicates),
        "feature_duplicates": int(feature_duplicates),
        "removed_rows": int(removed_count),
        "final_rows": len(df_dedup)
    }
    return df_dedup, dedup_stats


def deduplicate_and_manifest_images(
    raw_dir: str = "data/raw",
    output_dir: str = "data/processed"
) -> pd.DataFrame:
    """
    Scans diagnostic ECG images, checks for duplicate image hashes (MD5),
    and creates a clean image manifest.
    """
    raw_path = Path(raw_dir)
    image_records = []
    seen_hashes = {}
    duplicate_images = []

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    
    for file_path in raw_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            # Compute MD5
            file_bytes = file_path.read_bytes()
            img_hash = hashlib.md5(file_bytes).hexdigest()
            category = file_path.parent.name
            
            if img_hash in seen_hashes:
                duplicate_images.append((str(file_path), seen_hashes[img_hash]))
            else:
                seen_hashes[img_hash] = str(file_path)
                image_records.append({
                    "file_path": str(file_path).replace("\\", "/"),
                    "filename": file_path.name,
                    "category": category,
                    "md5_hash": img_hash,
                    "file_size_bytes": file_path.stat().st_size
                })

    df_images = pd.DataFrame(image_records)
    logger.info(f"Image Check: Found {len(df_images)} unique images across {len(df_images['category'].unique()) if len(df_images) > 0 else 0} categories.")
    logger.info(f"Duplicate images found: {len(duplicate_images)}")
    
    return df_images


def clean_and_impute_missing(
    df: pd.DataFrame, 
    impute_values: Optional[Dict] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Handles missing values and standardizes data types.
    Imputation parameters are derived from the training set if impute_values is None.
    """
    df_clean = df.copy()
    
    # Missing value statistics
    missing_counts = df_clean.isnull().sum().to_dict()
    missing_cols = {k: int(v) for k, v in missing_counts.items() if v > 0}
    if missing_cols:
        logger.info(f"Missing values detected: {missing_cols}")
    else:
        logger.info("No missing values found.")

    computed_imputes = {} if impute_values is None else impute_values.copy()
    
    # Calculate mode for categorical / discrete columns if not provided
    for col in ["ca", "thal"]:
        if col in df_clean.columns:
            if col not in computed_imputes:
                mode_val = df_clean[col].mode(dropna=True)[0]
                computed_imputes[col] = float(mode_val)
            df_clean[col] = df_clean[col].fillna(computed_imputes[col])
            df_clean[col] = pd.to_numeric(df_clean[col])

    # Convert numeric columns
    for col in COLUMN_NAMES:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col])

    # Construct binary target (0: healthy, 1: heart disease present)
    if TARGET_COL in df_clean.columns:
        df_clean[BINARY_TARGET_COL] = (df_clean[TARGET_COL] > 0).astype(int)

    return df_clean, computed_imputes


def split_data_stratified(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    stratify_col: str = BINARY_TARGET_COL
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits data into stratified Train, Validation, and Test sets.
    """
    assert abs((train_size + val_size + test_size) - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    
    # First split: train vs temp (val + test)
    temp_size = val_size + test_size
    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        stratify=df[stratify_col],
        random_state=random_state
    )
    
    # Second split: val vs test
    val_ratio = val_size / temp_size
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio,
        stratify=temp_df[stratify_col],
        random_state=random_state
    )
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    logger.info(f"Dataset Segregation: Train={len(train_df)} ({len(train_df)/len(df):.1%}), "
                f"Val={len(val_df)} ({len(val_df)/len(df):.1%}), "
                f"Test={len(test_df)} ({len(test_df)/len(df):.1%})")
    
    return train_df, val_df, test_df


def scale_features_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    scaler_save_path: Optional[str] = "models/scaler.pkl"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scales continuous and numeric features using StandardScaler fitted STRICTLY on the training set.
    """
    scaler = StandardScaler()
    
    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()
    
    # Fit scaler ONLY on train features
    scaler.fit(train_df[feature_cols])
    
    # Transform all splits
    train_scaled[feature_cols] = scaler.transform(train_df[feature_cols])
    val_scaled[feature_cols] = scaler.transform(val_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])
    
    if scaler_save_path:
        os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
        joblib.dump(scaler, scaler_save_path)
        logger.info(f"StandardScaler saved to {scaler_save_path}")
        
    return train_scaled, val_scaled, test_scaled, scaler


def run_full_preprocessing_pipeline(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    models_dir: str = "models",
    random_state: int = 42
) -> Dict:
    """
    Executes the full preprocessing, deduplication, segregation, and export pipeline.
    """
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    logger.info("=== STEP 1: Load Raw Dataset ===")
    df_raw = fetch_or_load_raw_data(raw_dir=raw_dir)
    
    logger.info("=== STEP 2: Tabular Deduplication ===")
    df_dedup, dedup_stats = deduplicate_tabular(df_raw)
    
    logger.info("=== STEP 3: Image Deduplication & Manifest ===")
    df_images = deduplicate_and_manifest_images(raw_dir=raw_dir, output_dir=processed_dir)
    
    logger.info("=== STEP 4: Initial Train/Val/Test Split (Before Imputation) ===")
    # Split first to prevent data leakage during imputation
    # Temporary binary target for stratified splitting
    df_dedup_temp = df_dedup.copy()
    df_dedup_temp[BINARY_TARGET_COL] = (df_dedup_temp[TARGET_COL] > 0).astype(int)
    
    raw_train, raw_val, raw_test = split_data_stratified(
        df_dedup_temp, 
        train_size=0.70, 
        val_size=0.15, 
        test_size=0.15, 
        random_state=random_state
    )
    
    logger.info("=== STEP 5: Missing Value Imputation (Fit on Train) ===")
    # Compute imputation parameters from train
    train_clean, impute_params = clean_and_impute_missing(raw_train)
    val_clean, _ = clean_and_impute_missing(raw_val, impute_values=impute_params)
    test_clean, _ = clean_and_impute_missing(raw_test, impute_values=impute_params)
    
    # Also prepare full cleaned dataset
    full_clean, _ = clean_and_impute_missing(df_dedup, impute_values=impute_params)
    
    # Add split tag to full dataset
    train_clean_tagged = train_clean.copy()
    train_clean_tagged["split"] = "train"
    val_clean_tagged = val_clean.copy()
    val_clean_tagged["split"] = "val"
    test_clean_tagged = test_clean.copy()
    test_clean_tagged["split"] = "test"
    full_tagged = pd.concat([train_clean_tagged, val_clean_tagged, test_clean_tagged], ignore_index=True)
    
    logger.info("=== STEP 6: Feature Scaling (Zero Leakage) ===")
    feature_cols = [c for c in COLUMN_NAMES if c != TARGET_COL]
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    
    train_scaled, val_scaled, test_scaled, scaler = scale_features_no_leakage(
        train_clean, val_clean, test_clean,
        feature_cols=feature_cols,
        scaler_save_path=scaler_path
    )
    
    logger.info("=== STEP 7: Exporting Datasets ===")
    # Save unscaled splits
    train_clean.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    val_clean.to_csv(os.path.join(processed_dir, "val.csv"), index=False)
    test_clean.to_csv(os.path.join(processed_dir, "test.csv"), index=False)
    
    # Save scaled splits
    train_scaled.to_csv(os.path.join(processed_dir, "train_scaled.csv"), index=False)
    val_scaled.to_csv(os.path.join(processed_dir, "val_scaled.csv"), index=False)
    test_scaled.to_csv(os.path.join(processed_dir, "test_scaled.csv"), index=False)
    
    # Save full dataset
    full_tagged.to_csv(os.path.join(processed_dir, "full_preprocessed.csv"), index=False)
    
    # Image splits manifest
    if len(df_images) > 0:
        # Balanced per-category allocation for images
        split_records = []
        for cat, group in df_images.groupby("category"):
            n = len(group)
            indices = np.arange(n)
            # Shuffle deterministically
            rng = np.random.default_rng(random_state)
            shuffled_idx = rng.permutation(indices)
            
            n_train = max(1, int(round(n * 0.60))) if n >= 3 else max(1, int(n * 0.50))
            n_val = max(1, int(round(n * 0.20))) if n >= 3 else 0
            n_test = n - n_train - n_val
            if n_test <= 0 and n >= 3:
                n_train -= 1
                n_test = 1
                
            group_records = group.iloc[shuffled_idx].copy()
            labels = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
            group_records["split"] = labels
            split_records.append(group_records)
            
        df_images_split = pd.concat(split_records, ignore_index=True)
        df_images_split.to_csv(os.path.join(processed_dir, "image_manifest_splits.csv"), index=False)
        logger.info(f"Saved image manifest splits with {len(df_images_split)} images.")
    
    # Save summary metadata
    summary = {
        "dataset_name": "UCI Cleveland Heart Disease",
        "random_state": random_state,
        "deduplication": dedup_stats,
        "imputation_parameters": impute_params,
        "split_counts": {
            "train": len(train_clean),
            "val": len(val_clean),
            "test": len(test_clean),
            "total": len(full_tagged)
        },
        "split_ratios": {
            "train": round(len(train_clean) / len(full_tagged), 4),
            "val": round(len(val_clean) / len(full_tagged), 4),
            "test": round(len(test_clean) / len(full_tagged), 4)
        },
        "target_binary_distribution": {
            "train": train_clean[BINARY_TARGET_COL].value_counts().to_dict(),
            "val": val_clean[BINARY_TARGET_COL].value_counts().to_dict(),
            "test": test_clean[BINARY_TARGET_COL].value_counts().to_dict()
        },
        "target_multiclass_distribution": {
            "train": train_clean[TARGET_COL].value_counts().to_dict(),
            "val": val_clean[TARGET_COL].value_counts().to_dict(),
            "test": test_clean[TARGET_COL].value_counts().to_dict()
        },
        "features": {
            "all_features": feature_cols,
            "continuous": CONTINUOUS_FEATURES,
            "categorical": CATEGORICAL_FEATURES
        },
        "files_generated": [
            "data/processed/train.csv",
            "data/processed/val.csv",
            "data/processed/test.csv",
            "data/processed/train_scaled.csv",
            "data/processed/val_scaled.csv",
            "data/processed/test_scaled.csv",
            "data/processed/full_preprocessed.csv",
            "data/processed/image_manifest_splits.csv",
            "models/scaler.pkl"
        ]
    }
    
    summary_path = os.path.join(processed_dir, "split_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
        
    logger.info(f"Saved pipeline summary report to {summary_path}")
    logger.info("=== Preprocessing, Deduplication, and Segregation COMPLETE! ===")
    return summary


if __name__ == "__main__":
    summary = run_full_preprocessing_pipeline()
    print("\n--- Summary Report ---")
    print(json.dumps(summary, indent=2))
