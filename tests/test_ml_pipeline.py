"""Unit and integration tests for ML pipeline and RiskPredictor."""

import numpy as np
import pandas as pd
import pytest
from src.defect_labeling import generate_benchmark_dataset
from src.predict import RiskPredictor
from src.preprocessing import (
    FEATURE_COLUMNS,
    chronological_split,
    filter_valid_samples,
    prepare_features,
)
from src.train import train_and_select_best_model


def test_dataset_generation_and_filtering():
    df = generate_benchmark_dataset(num_samples=100, random_seed=123)
    assert len(df) == 100
    assert "defect_within_30_days" in df.columns
    assert "commit_date" in df.columns

    clean = filter_valid_samples(df)
    assert len(clean) <= 100
    assert set(clean["defect_within_30_days"].unique()).issubset({0, 1})


def test_chronological_split():
    df = generate_benchmark_dataset(num_samples=100, random_seed=42)
    train, val, test = chronological_split(df, 0.7, 0.15)

    assert len(train) + len(val) + len(test) == 100
    # Confirm temporal order
    assert train["commit_date"].max() <= val["commit_date"].min()
    assert val["commit_date"].max() <= test["commit_date"].min()


def test_prepare_features():
    df = generate_benchmark_dataset(num_samples=50, random_seed=42)
    clean = filter_valid_samples(df)
    X, y, pipe = prepare_features(clean)

    assert X.shape[0] == len(clean)
    assert X.shape[1] == len(FEATURE_COLUMNS)
    assert y is not None
    assert len(y) == len(clean)


def test_model_training_and_artifacts(tmp_path):
    meta = train_and_select_best_model(save_artifacts=True)
    assert "model_version" in meta
    assert "validation_metrics" in meta
    assert "test_metrics" in meta
    assert meta["test_metrics"]["precision"] >= 0.0
    assert meta["test_metrics"]["recall"] >= 0.0


def test_predictor_within_scope():
    predictor = RiskPredictor()
    sample_features = {
        "language": "Python",
        "files_changed": 3,
        "lines_added": 120,
        "lines_deleted": 40,
        "total_churn": 160,
        "cyclomatic_complexity": 8.5,
        "maintainability_index": 72.0,
        "loc": 1500,
        "function_count": 25,
        "recent_commit_frequency": 6,
        "previous_defects_count": 1,
        "test_failure_ratio": 0.0,
    }
    pred = predictor.predict(sample_features, risk_threshold=0.70)
    assert pred["is_applicable"] is True
    assert pred["predicted_risk"] is not None
    assert 0.0 <= pred["predicted_risk"] <= 1.0
    assert pred["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert len(pred["explanation_factors"]) > 0


def test_predictor_out_of_scope():
    predictor = RiskPredictor()
    # Non-Python repository
    sample_features = {
        "language": "Rust",
        "loc": 500,
        "cyclomatic_complexity": 5.0,
    }
    pred = predictor.predict(sample_features)
    assert pred["is_applicable"] is False
    assert pred["predicted_risk"] is None
    assert "inapplicability_reason" in pred
