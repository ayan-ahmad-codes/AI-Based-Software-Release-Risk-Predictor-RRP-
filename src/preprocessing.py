"""Preprocessing and Feature Engineering Pipeline for Release Risk Prediction.

Handles chronological dataset splitting, feature selection, missing value imputation,
and numerical feature scaling using scikit-learn.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    # 1. Code Changes & Churn
    "files_changed",
    "lines_added",
    "lines_deleted",
    "total_churn",
    # 2. Code Complexity & Quality
    "cyclomatic_complexity",
    "maintainability_index",
    "loc",
    "function_count",
    # 3. Development Process
    "recent_commit_frequency",
    # 4. Historical Defects
    "previous_defects_count",
    # 5. Test Results Evidence
    "test_failure_ratio",
    # 6. Deployment Metadata
    "days_since_last_release",
    "is_main_branch",
]

TARGET_COLUMN = "defect_within_30_days"


def filter_valid_samples(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out samples with unknown or ambiguous defect labels."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing from dataframe.")

    # Only keep confirmed labels (0 or 1)
    clean_df = df[df[TARGET_COLUMN].isin([0, 1])].copy()
    clean_df[TARGET_COLUMN] = clean_df[TARGET_COLUMN].astype(int)
    return clean_df


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically to respect temporal boundaries (no future leakage).

    Args:
        df: DataFrame containing a 'commit_date' column.
        train_ratio: Proportion for training (earliest).
        val_ratio: Proportion for validation.

    Returns:
        (train_df, val_df, test_df)
    """
    sorted_df = df.sort_values("commit_date").reset_index(drop=True)
    n = len(sorted_df)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = sorted_df.iloc[:train_end].copy()
    val_df = sorted_df.iloc[train_end:val_end].copy()
    test_df = sorted_df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def build_preprocessor() -> Pipeline:
    """Construct standard scikit-learn preprocessing pipeline for numerical features."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def prepare_features(
    df: pd.DataFrame,
    fit_pipeline: Pipeline | None = None,
) -> Tuple[np.ndarray, np.ndarray | None, Pipeline]:
    """Extract features and target, applying the preprocessor pipeline.

    If fit_pipeline is provided, transforms using it (preventing test leakage).
    Otherwise fits a new pipeline on the input data.
    """
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            # If missing, initialize with zeros or default
            df[col] = 0.0

    X_raw = df[FEATURE_COLUMNS].copy()

    if fit_pipeline is None:
        pipeline = build_preprocessor()
        X_trans = pipeline.fit_transform(X_raw)
    else:
        pipeline = fit_pipeline
        X_trans = pipeline.transform(X_raw)

    y_val = None
    if TARGET_COLUMN in df.columns and df[TARGET_COLUMN].dropna().count() > 0:
        y_val = df[TARGET_COLUMN].values.astype(int)

    return X_trans, y_val, pipeline
