"""Comprehensive Verification & Validation tests for user's official demands.

Validates:
1. Leakage Prevention (strict pre-release feature boundary)
2. Temporal Validation (chronological holdouts)
3. Model Probability Calibration & Expected Calibration Error (ECE)
4. Subgroup Slicing Analysis (repository size, churn, defect history)
5. Explainability (mathematical local feature attributions)
6. Release Decision Governance Matrix
7. Post-Release Monitoring & PSI Drift Detection
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.defect_labeling import generate_benchmark_dataset
from src.evaluate import calculate_calibration_curve_metrics, evaluate_subgroups
from src.monitoring import calculate_psi, compute_monitoring_health, record_release_prediction, update_post_release_outcome
from src.predict import RiskPredictor
from src.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, chronological_split
from src.quality_gates import determine_release_decision, evaluate_quality_gates


def test_leakage_prevention_guarantee():
    """Verify that post-release fields are strictly excluded from FEATURE_COLUMNS."""
    forbidden_post_release_patterns = [
        "future",
        "fix",
        "after",
        "post",
        "defect_within",
        "label",
        "resolution",
        "closing",
    ]
    for feat in FEATURE_COLUMNS:
        assert feat != TARGET_COLUMN, f"Target column '{TARGET_COLUMN}' leaked into features!"
        for pattern in forbidden_post_release_patterns:
            assert pattern not in feat.lower(), f"Potential leakage feature detected: '{feat}'"


def test_temporal_validation_integrity():
    """Verify temporal validation ensures strict temporal ordering with no lookahead."""
    df = generate_benchmark_dataset(num_samples=120, random_seed=42)
    train_df, val_df, test_df = chronological_split(df, 0.70, 0.15)

    assert train_df["commit_date"].max() <= val_df["commit_date"].min()
    assert val_df["commit_date"].max() <= test_df["commit_date"].min()


def test_calibration_curve_and_ece():
    """Verify calibration curve coordinates and Expected Calibration Error (ECE) bounds."""
    y_true = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
    probs = np.array([0.1, 0.2, 0.15, 0.25, 0.75, 0.3, 0.8, 0.85, 0.9, 0.95])

    metrics = calculate_calibration_curve_metrics(y_true, probs, n_bins=5)
    assert "expected_calibration_error" in metrics
    assert 0.0 <= metrics["expected_calibration_error"] <= 1.0
    assert len(metrics["prob_true"]) > 0
    assert len(metrics["prob_pred"]) > 0


def test_subgroup_analysis_slicing():
    """Verify subgroup analysis slices by repo size, churn, and history."""
    predictor = RiskPredictor()
    df = generate_benchmark_dataset(num_samples=150, random_seed=42)

    subgroups = evaluate_subgroups(
        df=df,
        model=predictor.model,
        preprocessor=predictor.preprocessor,
        feature_columns=FEATURE_COLUMNS,
    )
    assert "Repository Size: Small (<5K LOC)" in subgroups
    assert "Code Churn: High (>=200 lines)" in subgroups
    assert "Defect History: Low (<3 bugs)" in subgroups
    for name, res in subgroups.items():
        assert "sample_count" in res
        assert "status" in res


def test_mathematical_explainability_attributions():
    """Verify local mathematical feature attribution decomposition."""
    predictor = RiskPredictor()
    sample_features = {
        "language": "Python",
        "files_changed": 15,
        "lines_added": 2500,
        "lines_deleted": 800,
        "total_churn": 3300,
        "cyclomatic_complexity": 35.0,
        "maintainability_index": 30.0,
        "loc": 12000,
        "function_count": 80,
        "recent_commit_frequency": 25,
        "previous_defects_count": 5,
        "test_failure_ratio": 0.2,
        "days_since_last_release": 30.0,
        "is_main_branch": 1,
    }
    pred = predictor.predict(sample_features)
    assert "feature_attributions" in pred
    assert len(pred["feature_attributions"]) == len(FEATURE_COLUMNS)
    top_attr = pred["feature_attributions"][0]
    assert "feature" in top_attr
    assert "contribution_score" in top_attr
    assert "direction" in top_attr


def test_release_decision_governance_matrix():
    """Verify release decision mapping for all operational risk tiers."""
    # Tier 1: Automated deploy
    dec1 = determine_release_decision("NO_BLOCKING_FINDINGS", predicted_risk=0.15)
    assert dec1["decision"] == "AUTOMATED_DEPLOY_PERMITTED"

    # Tier 2: Staged rollout on high statistical risk
    dec2 = determine_release_decision("REVIEW_REQUIRED", predicted_risk=0.85, risk_threshold=0.70)
    assert dec2["decision"] == "STAGED_ROLLOUT_AND_QA_SIGNOFF"

    # Tier 3: Hard block on gate failure
    dec3 = determine_release_decision("REVIEW_REQUIRED", predicted_risk=0.20, risk_threshold=0.70)
    assert dec3["decision"] == "HARD_BLOCK"

    # Tier 4: Assessment blocked
    dec4 = determine_release_decision("ASSESSMENT_BLOCKED")
    assert dec4["decision"] == "HARD_BLOCK"


def test_post_release_monitoring_and_psi(tmp_path: Path):
    """Verify post-release event logging, outcome recording, and PSI drift calculation."""
    log_file = tmp_path / "test_monitoring.jsonl"

    # 1. Record release prediction
    rec = record_release_prediction(
        repo_name="psf/requests",
        commit_sha="a1b2c3d4",
        predicted_risk=0.35,
        features={"cyclomatic_complexity": 10.0, "total_churn": 150},
        decision="AUTOMATED_DEPLOY_PERMITTED",
        log_file=log_file,
    )
    assert rec["commit_sha"] == "a1b2c3d4"
    assert rec["follow_up_completed"] is False

    # 2. Update 30-day post-release outcome
    updated = update_post_release_outcome(
        commit_sha="a1b2c3d4",
        defect_observed=False,
        notes="No incidents reported in 30-day window",
        log_file=log_file,
    )
    assert updated is True

    # 3. Test Population Stability Index (PSI) drift calculation
    expected = np.random.normal(50, 10, 1000)
    # Undrifted actual
    actual_stable = np.random.normal(50, 10, 500)
    psi_stable = calculate_psi(expected, actual_stable)
    assert psi_stable < 0.10

    # Heavily drifted actual
    actual_drifted = np.random.normal(90, 15, 500)
    psi_drifted = calculate_psi(expected, actual_drifted)
    assert psi_drifted >= 0.25
