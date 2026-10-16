"""End-to-End integration tests for Software Release Risk Predictor pipeline."""

from pathlib import Path
import pytest
from src.pipeline import run_full_assessment


def test_full_assessment_on_local_repo():
    root = Path(__file__).resolve().parent.parent
    result = run_full_assessment(str(root))

    assert "repo_info" in result
    assert "commit_info" in result
    assert "code_metrics" in result
    assert "ci_evidence" in result
    assert "prediction" in result
    assert "gate_results" in result
    assert "audit_dossier" in result

    # Verify metrics
    metrics = result["code_metrics"]
    assert metrics["files_analyzed"] >= 5
    assert metrics["total_loc"] > 100

    # Verify gates
    gates = result["gate_results"]
    assert gates["overall_status"] in {
        "NO_BLOCKING_FINDINGS",
        "REVIEW_REQUIRED",
        "WARNINGS_DETECTED",
        "ASSESSMENT_BLOCKED",
    }
    # V&V invariant: Never claim SAFE_TO_RELEASE
    assert gates["overall_status"] != "SAFE_TO_RELEASE"

    # Verify audit dossier export
    dossier = result["audit_dossier"]
    assert "report_id" in dossier
    assert "standards_and_limitations" in dossier
