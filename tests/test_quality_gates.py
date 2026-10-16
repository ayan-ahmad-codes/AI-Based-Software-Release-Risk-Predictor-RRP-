"""Verification & Validation tests for src/quality_gates.py."""

from src.quality_gates import evaluate_quality_gates


def test_scenario_all_passed():
    """Tests pass, CI passes, analysis complete -> No mandatory failure."""
    res = evaluate_quality_gates(
        test_status="passed",
        ci_status="passed",
        analysis_complete=True,
        model_validated=True,
        predicted_risk=0.25,
        risk_threshold=0.70,
    )
    assert res["overall_status"] == "NO_BLOCKING_FINDINGS"
    assert not res["has_blocking_findings"]
    # Check V&V requirement: output does not claim SAFE_TO_RELEASE
    assert res["overall_status"] != "SAFE_TO_RELEASE"


def test_scenario_test_fails():
    """A required test fails -> Review required."""
    res = evaluate_quality_gates(
        test_status="failed",
        ci_status="passed",
        analysis_complete=True,
        model_validated=True,
        predicted_risk=0.20,
    )
    assert res["overall_status"] == "REVIEW_REQUIRED"
    assert res["has_blocking_findings"]
    test_gate = next(f for f in res["findings"] if f["gate"] == "Test Verification")
    assert test_gate["status"] == "FAIL"


def test_scenario_ci_unavailable():
    """CI evidence is unavailable -> Unknown or incomplete evidence."""
    res = evaluate_quality_gates(
        test_status="passed",
        ci_status="unknown",
        analysis_complete=True,
        model_validated=True,
        predicted_risk=0.20,
    )
    assert res["overall_status"] == "WARNINGS_DETECTED"
    ci_gate = next(f for f in res["findings"] if f["gate"] == "CI Workflow")
    assert ci_gate["status"] == "UNKNOWN"


def test_scenario_model_not_validated():
    """Model is not validated -> No ML risk claim."""
    res = evaluate_quality_gates(
        test_status="passed",
        ci_status="passed",
        analysis_complete=True,
        model_validated=False,
        predicted_risk=None,
    )
    ml_gate = next(f for f in res["findings"] if f["gate"] == "ML Defect Risk")
    assert ml_gate["status"] == "NOT_APPLICABLE"


def test_scenario_high_risk_prediction():
    """Validated model predicts high risk -> Review required."""
    res = evaluate_quality_gates(
        test_status="passed",
        ci_status="passed",
        analysis_complete=True,
        model_validated=True,
        predicted_risk=0.85,
        risk_threshold=0.70,
    )
    assert res["overall_status"] == "REVIEW_REQUIRED"
    ml_gate = next(f for f in res["findings"] if f["gate"] == "ML Defect Risk")
    assert ml_gate["status"] == "REVIEW_REQUIRED"


def test_scenario_analysis_incomplete():
    """Analysis is incomplete -> Assessment blocked."""
    res = evaluate_quality_gates(
        test_status="passed",
        ci_status="passed",
        analysis_complete=False,
        model_validated=True,
        predicted_risk=0.20,
    )
    assert res["overall_status"] == "ASSESSMENT_BLOCKED"
    assert res["has_blocking_findings"]
    gate = next(f for f in res["findings"] if f["gate"] == "Analysis Completeness")
    assert gate["status"] == "BLOCKED"
