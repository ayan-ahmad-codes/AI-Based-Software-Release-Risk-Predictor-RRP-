"""Model Training and Verification Pipeline.

Trains baseline models (Logistic Regression, Random Forest), performs chronological
validation, computes feature importances, evaluates on untouched test data,
and persists verified model artifacts.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.defect_labeling import generate_benchmark_dataset
from src.evaluate import evaluate_model
from src.preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    chronological_split,
    filter_valid_samples,
    prepare_features,
)

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
DATA_DIR = Path("data")


def train_and_select_best_model(
    data_df: pd.DataFrame | None = None,
    save_artifacts: bool = True,
    model_version: str = "v1.0.0",
) -> dict[str, Any]:
    """Train candidate models and select the best based on validation PR-AUC and F1.

    Args:
        data_df: Optional custom historical dataset; if None, generates benchmark dataset.
        save_artifacts: Whether to persist model files and reports to disk.
        model_version: Semantic version tag for model tracking.

    Returns:
        dict: Full training summary including metrics for all candidates and selected model.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)

    if data_df is None:
        raw_df = generate_benchmark_dataset(num_samples=1500, random_seed=42)
        if save_artifacts:
            raw_df.to_csv(DATA_DIR / "raw" / "historical_commits.csv", index=False)
    else:
        raw_df = data_df

    clean_df = filter_valid_samples(raw_df)
    if save_artifacts:
        clean_df.to_csv(DATA_DIR / "processed" / "cleaned_dataset.csv", index=False)

    # 1. Chronological split (no leakage)
    train_df, val_df, test_df = chronological_split(clean_df, train_ratio=0.70, val_ratio=0.15)

    # 2. Preprocess features using train set statistics only
    X_train, y_train, preprocessor = prepare_features(train_df, fit_pipeline=None)
    X_val, y_val, _ = prepare_features(val_df, fit_pipeline=preprocessor)
    X_test, y_test, _ = prepare_features(test_df, fit_pipeline=preprocessor)

    assert y_train is not None and y_val is not None and y_test is not None

    # 3. Train Candidate A: Logistic Regression
    lr = LogisticRegression(
        class_weight="balanced",
        C=1.0,
        max_iter=1000,
        random_state=42,
    )
    lr.fit(X_train, y_train)
    lr_val_metrics = evaluate_model(lr, X_val, y_val)

    # 4. Train Candidate B: Random Forest
    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=6,
        min_samples_split=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_val_metrics = evaluate_model(rf, X_val, y_val)

    # 5. Model Selection based on validation F1 and PR-AUC
    if rf_val_metrics["pr_auc"] >= lr_val_metrics["pr_auc"]:
        base_model = rf
        selected_name = "RandomForestClassifier"
        selected_val_metrics = rf_val_metrics
        importances = dict(zip(FEATURE_COLUMNS, [round(float(x), 4) for x in rf.feature_importances_]))
    else:
        base_model = lr
        selected_name = "LogisticRegression"
        selected_val_metrics = lr_val_metrics
        importances = dict(zip(FEATURE_COLUMNS, [round(float(x), 4) for x in lr.coef_[0]]))

    # Calibrate probabilities for V&V calibration compliance
    from sklearn.calibration import CalibratedClassifierCV
    calibrated_model = CalibratedClassifierCV(estimator=base_model, cv=3, method="sigmoid")
    calibrated_model.fit(X_train, y_train)

    # 6. Final Evaluation on untouched Test set
    test_metrics = evaluate_model(calibrated_model, X_test, y_test)

    # 7. Subgroup Slicing Analysis on Test Holdout (V&V requirement)
    from src.evaluate import evaluate_subgroups
    subgroup_metrics = evaluate_subgroups(
        test_df,
        calibrated_model,
        preprocessor,
        FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
    )

    # 8. Metadata package
    metadata = {
        "model_version": model_version,
        "selected_algorithm": selected_name,
        "is_calibrated": True,
        "calibration_method": "sigmoid_platt_scaling",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": FEATURE_COLUMNS,
        "feature_importances": importances,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "validation_metrics": selected_val_metrics,
        "test_metrics": test_metrics,
        "subgroup_analysis": subgroup_metrics,
        "candidates_comparison": {
            "LogisticRegression": lr_val_metrics,
            "RandomForestClassifier": rf_val_metrics,
        },
        "applicable_scope": {
            "language": "Python",
            "supported_features": FEATURE_COLUMNS,
            "max_reliable_loc": 250000,
            "requires_valid_history": True,
        },
    }

    if save_artifacts:
        bundle = {
            "preprocessor": preprocessor,
            "model": calibrated_model,
            "base_model": base_model,
            "metadata": metadata,
        }
        joblib.dump(bundle, MODELS_DIR / "release_risk_model.joblib")
        with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        with open(REPORTS_DIR / "model_evaluation_report.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return metadata


if __name__ == "__main__":
    meta = train_and_select_best_model(save_artifacts=True)
    print("Training finished successfully!")
    print(f"Selected: {meta['selected_algorithm']}")
    print(f"Validation F1: {meta['validation_metrics']['f1']} | PR-AUC: {meta['validation_metrics']['pr_auc']}")
    print(f"Test F1: {meta['test_metrics']['f1']} | PR-AUC: {meta['test_metrics']['pr_auc']}")
