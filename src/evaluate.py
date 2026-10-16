"""Model evaluation and metrics calculation for release risk prediction.

Calculates Precision, Recall, F1, PR-AUC, ROC-AUC, Brier score (calibration),
Expected Calibration Error (ECE), subgroup slicing, and temporal validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_calibration_curve_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 5,
) -> dict[str, Any]:
    """Calculate calibration curve coordinates and Expected Calibration Error (ECE).

    ECE measures the average absolute difference between predicted confidence
    and empirical defect frequency across probability bins.
    """
    prob_true, prob_pred = calibration_curve(y_true, probabilities, n_bins=n_bins, strategy="uniform")

    # Calculate Expected Calibration Error (ECE)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_assignments = np.digitize(probabilities, bin_edges) - 1
    bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)

    total_samples = len(y_true)
    ece = 0.0
    bin_details = []

    for b in range(n_bins):
        mask = bin_assignments == b
        count = int(np.sum(mask))
        if count > 0:
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(probabilities[mask]))
            abs_diff = abs(bin_acc - bin_conf)
            ece += (count / total_samples) * abs_diff
            bin_details.append({
                "bin_index": b,
                "bin_range": f"{bin_edges[b]:.1f}-{bin_edges[b+1]:.1f}",
                "sample_count": count,
                "observed_frequency": round(bin_acc, 4),
                "predicted_probability": round(bin_conf, 4),
                "calibration_gap": round(abs_diff, 4),
            })

    return {
        "expected_calibration_error": round(float(ece), 4),
        "prob_true": [round(float(x), 4) for x in prob_true],
        "prob_pred": [round(float(x), 4) for x in prob_pred],
        "binned_details": bin_details,
    }


def evaluate_model(
    model: Any,
    X: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.50,
) -> dict[str, Any]:
    """Calculate comprehensive evaluation metrics for a binary classifier.

    Args:
        model: Fitted scikit-learn classifier with predict_proba.
        X: Feature matrix.
        y_true: Ground truth binary labels (0 or 1).
        threshold: Classification probability threshold.

    Returns:
        dict: Performance metrics including Precision, Recall, F1, PR-AUC, Brier score, ECE.
    """
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
    else:
        # Fallback for models without predict_proba
        probabilities = model.decision_function(X)

    y_pred = (probabilities >= threshold).astype(int)

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))

    try:
        pr_auc = float(average_precision_score(y_true, probabilities))
    except Exception:
        pr_auc = 0.0

    try:
        roc_auc = float(roc_auc_score(y_true, probabilities))
    except Exception:
        roc_auc = 0.0

    try:
        brier = float(brier_score_loss(y_true, probabilities))
    except Exception:
        brier = 1.0

    calib_metrics = calculate_calibration_curve_metrics(y_true, probabilities)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    return {
        "threshold": threshold,
        "sample_count": len(y_true),
        "positive_cases": int(np.sum(y_true)),
        "negative_cases": int(len(y_true) - np.sum(y_true)),
        "positive_prevalence": round(float(np.mean(y_true)), 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "brier_score": round(brier, 4),
        "expected_calibration_error": calib_metrics["expected_calibration_error"],
        "calibration": calib_metrics,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }


def evaluate_subgroups(
    df: pd.DataFrame,
    model: Any,
    preprocessor: Any,
    feature_columns: list[str],
    target_column: str = "defect_within_30_days",
) -> dict[str, Any]:
    """Perform subgroup slicing analysis to check for model disparities.

    Slices evaluation across:
    1. Repository size tiers (Small < 5K LOC, Medium 5K-15K, Large > 15K)
    2. Churn magnitude (Low Churn < 200 lines, High Churn >= 200 lines)
    3. Defect history depth (Low Defect History < 3, High Defect History >= 3)
    """
    clean_df = df[df[target_column].isin([0, 1])].copy()
    if clean_df.empty:
        return {"subgroups": {}, "note": "No valid labeled data for subgroup analysis"}

    subgroup_definitions = {
        "Repository Size: Small (<5K LOC)": clean_df["loc"] < 5000,
        "Repository Size: Medium (5K-15K LOC)": (clean_df["loc"] >= 5000) & (clean_df["loc"] <= 15000),
        "Repository Size: Large (>15K LOC)": clean_df["loc"] > 15000,
        "Code Churn: Low (<200 lines)": clean_df["total_churn"] < 200,
        "Code Churn: High (>=200 lines)": clean_df["total_churn"] >= 200,
        "Defect History: Low (<3 bugs)": clean_df["previous_defects_count"] < 3,
        "Defect History: High (>=3 bugs)": clean_df["previous_defects_count"] >= 3,
    }

    subgroup_results = {}
    for name, mask in subgroup_definitions.items():
        sub_df = clean_df[mask]
        if len(sub_df) < 5 or sub_df[target_column].nunique() < 1:
            subgroup_results[name] = {
                "sample_count": len(sub_df),
                "positive_count": int(sub_df[target_column].sum()) if len(sub_df) > 0 else 0,
                "status": "INSUFFICIENT_SUBGROUP_DATA",
                "precision": None,
                "recall": None,
                "f1": None,
                "pr_auc": None,
            }
            continue

        X_raw = sub_df[feature_columns]
        y_true = sub_df[target_column].values.astype(int)
        X_trans = preprocessor.transform(X_raw)

        m = evaluate_model(model, X_trans, y_true)
        subgroup_results[name] = {
            "sample_count": len(sub_df),
            "positive_count": int(np.sum(y_true)),
            "positive_prevalence": round(float(np.mean(y_true)), 4),
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "pr_auc": m["pr_auc"],
            "brier_score": m["brier_score"],
            "status": "EVALUATED",
        }

    return subgroup_results


def compare_models(
    results_a: dict[str, Any],
    results_b: dict[str, Any],
    metric: str = "f1",
) -> str:
    """Compare performance of two models on a specified metric."""
    score_a = results_a.get(metric, 0.0)
    score_b = results_b.get(metric, 0.0)
    diff = score_b - score_a

    if abs(diff) < 0.01:
        return f"Performance is equivalent ({metric}: {score_a:.4f} vs {score_b:.4f})"
    if diff > 0:
        return f"Model B outperforms Model A by +{diff:.4f} {metric}"
    return f"Model A outperforms Model B by +{-diff:.4f} {metric}"


def check_regression_gate(
    new_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    max_allowed_drop: float = 0.05,
) -> dict[str, Any]:
    """Verify that a newly trained model does not suffer a regression compared to baseline."""
    critical_metrics = ["f1", "recall", "pr_auc"]
    regressions = []

    for m in critical_metrics:
        new_val = new_metrics.get(m, 0.0)
        base_val = baseline_metrics.get(m, 0.0)
        drop = base_val - new_val
        if drop > max_allowed_drop:
            regressions.append({
                "metric": m,
                "baseline": base_val,
                "new": new_val,
                "drop": round(drop, 4),
            })

    passed = len(regressions) == 0
    return {
        "gate": "Model Performance Regression Gate",
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "max_allowed_drop": max_allowed_drop,
        "regressions": regressions,
        "reason": "Model meets regression criteria." if passed else f"Model regressed in {len(regressions)} metric(s).",
    }
