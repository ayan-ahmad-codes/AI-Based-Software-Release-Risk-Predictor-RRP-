"""Software Release Quality Gates Engine.

Implements rule-based Verification & Validation gates separated from ML predictions.
Enforces that missing evidence is not treated as safety, and never declares
a release 'SAFE_TO_RELEASE' (declaring 'NO_BLOCKING_FINDINGS' instead).
"""

from __future__ import annotations

import math
from typing import Any


def evaluate_quality_gates(
    test_status: str,
    ci_status: str,
    analysis_complete: bool,
    model_validated: bool,
    predicted_risk: float | None = None,
    risk_threshold: float = 0.70,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate release readiness across independent quality gates.

    Args:
        test_status: 'passed' | 'failed' | 'unknown'
        ci_status: 'passed' | 'failed' | 'unknown'
        analysis_complete: Whether static code analysis succeeded
        model_validated: Whether the ML model is trained and validated
        predicted_risk: Estimated defect risk probability (0.0 to 1.0) or None
        risk_threshold: Risk threshold triggering a warning/review
        metrics: Optional dictionary of code and churn metrics

    Returns:
        dict: Overall status and individual gate findings.
    """
    findings: list[dict[str, str]] = []

    # 1. Analysis Completeness Gate
    if not analysis_complete:
        findings.append({
            "gate": "Analysis Completeness",
            "status": "BLOCKED",
            "reason": "Required static and repository analysis is incomplete or failed.",
        })
        return {
            "overall_status": "ASSESSMENT_BLOCKED",
            "has_blocking_findings": True,
            "findings": findings,
            "summary": "Assessment is blocked due to incomplete analysis.",
        }

    findings.append({
        "gate": "Analysis Completeness",
        "status": "PASS",
        "reason": "Repository inspection and metric extraction completed.",
    })

    # 2. Test Execution Gate
    normalized_test = (test_status or "unknown").strip().lower()
    if normalized_test == "failed":
        findings.append({
            "gate": "Test Verification",
            "status": "FAIL",
            "reason": "One or more tests failed in the target commit/release.",
        })
    elif normalized_test == "passed":
        findings.append({
            "gate": "Test Verification",
            "status": "PASS",
            "reason": "All observed test suites passed.",
        })
    else:
        findings.append({
            "gate": "Test Verification",
            "status": "UNKNOWN",
            "reason": "Test results are unavailable, pending, or incomplete.",
        })

    # 3. CI Workflow Gate
    normalized_ci = (ci_status or "unknown").strip().lower()
    if normalized_ci == "failed":
        findings.append({
            "gate": "CI Workflow",
            "status": "FAIL",
            "reason": "A required CI workflow run or check-run failed.",
        })
    elif normalized_ci == "passed":
        findings.append({
            "gate": "CI Workflow",
            "status": "PASS",
            "reason": "All required CI workflow runs and check-runs passed.",
        })
    else:
        findings.append({
            "gate": "CI Workflow",
            "status": "UNKNOWN",
            "reason": "CI evidence is unavailable or incomplete.",
        })

    # 4. Optional Code Churn / Complexity Gate
    if metrics:
        total_churn = metrics.get("total_churn", 0)
        max_cc = metrics.get("max_complexity", 0.0)
        if total_churn > 5000:
            findings.append({
                "gate": "Code Churn Gate",
                "status": "WARN",
                "reason": f"Unusually large code churn ({total_churn} lines modified).",
            })
        if max_cc > 40:
            findings.append({
                "gate": "Complexity Gate",
                "status": "WARN",
                "reason": f"Maximum cyclomatic complexity ({max_cc:.1f}) exceeds recommended limit (40).",
            })

    # 5. ML Risk Gate
    if not model_validated or predicted_risk is None or not math.isfinite(predicted_risk):
        findings.append({
            "gate": "ML Defect Risk",
            "status": "NOT_APPLICABLE",
            "reason": "No validated ML risk claim available for this repository.",
        })
    else:
        clamped_risk = max(0.0, min(1.0, float(predicted_risk)))
        if clamped_risk >= risk_threshold:
            findings.append({
                "gate": "ML Defect Risk",
                "status": "REVIEW_REQUIRED",
                "reason": f"Validated model predicts elevated defect risk ({clamped_risk:.1%} >= {risk_threshold:.1%}).",
            })
        else:
            findings.append({
                "gate": "ML Defect Risk",
                "status": "PASS",
                "reason": f"Validated model risk estimate ({clamped_risk:.1%}) is within acceptable threshold.",
            })

    # Determine overall decision
    statuses = [f["status"] for f in findings]

    if "BLOCKED" in statuses:
        overall_status = "ASSESSMENT_BLOCKED"
        has_blocking = True
        summary = "Assessment blocked: prerequisites not met."
    elif "FAIL" in statuses or "REVIEW_REQUIRED" in statuses:
        overall_status = "REVIEW_REQUIRED"
        has_blocking = True
        summary = "Review required: one or more quality gates failed or flagged high risk."
    elif "UNKNOWN" in statuses or "WARN" in statuses:
        overall_status = "WARNINGS_DETECTED"
        has_blocking = False
        summary = "Warnings or unknown evidence detected: release readiness requires verification."
    else:
        # Crucial V&V distinction: Never output SAFE_TO_RELEASE
        overall_status = "NO_BLOCKING_FINDINGS"
        has_blocking = False
        summary = "No blocking findings identified by automated quality gates."

    decision_info = determine_release_decision(
        overall_status=overall_status,
        predicted_risk=predicted_risk,
        risk_threshold=risk_threshold,
    )

    return {
        "overall_status": overall_status,
        "has_blocking_findings": has_blocking,
        "findings": findings,
        "summary": summary,
        "release_decision": decision_info,
        "disclaimer": (
            "Notice: 'NO_BLOCKING_FINDINGS' indicates only that automated checks "
            "did not detect known issues. It does NOT guarantee that the release is error-free."
        ),
    }


def determine_release_decision(
    overall_status: str,
    predicted_risk: float | None = None,
    risk_threshold: float = 0.70,
) -> dict[str, Any]:
    """Map quality gate findings and ML risk to a concrete release governance decision."""
    if overall_status == "ASSESSMENT_BLOCKED":
        return {
            "decision": "HARD_BLOCK",
            "action_level": "CRITICAL",
            "guidance": "Deployment halted. Required static analysis or repository data is unavailable.",
            "required_actions": [
                "Resolve repository access/parsing issues",
                "Re-run full assessment pipeline",
            ],
        }
    if overall_status == "REVIEW_REQUIRED":
        if predicted_risk is not None and predicted_risk >= risk_threshold:
            return {
                "decision": "STAGED_ROLLOUT_AND_QA_SIGNOFF",
                "action_level": "HIGH_CAUTION",
                "guidance": f"High statistical defect risk ({predicted_risk:.1%}). Direct 100% production rollout prohibited.",
                "required_actions": [
                    "Perform canary rollout (10% -> 25% -> 50% -> 100%)",
                    "Verify automated rollback alarms are armed",
                    "Mandate QA Lead and Senior Engineer sign-off",
                ],
            }
        return {
            "decision": "HARD_BLOCK",
            "action_level": "CRITICAL",
            "guidance": "Mandatory quality gate failure detected (e.g. broken tests or failed CI).",
            "required_actions": [
                "Fix failing test suites or CI check runs",
                "Push remediation commit and re-assess",
            ],
        }
    if overall_status == "WARNINGS_DETECTED":
        return {
            "decision": "PEER_REVIEW_AND_SMOKE_VERIFICATION",
            "action_level": "MODERATE_CAUTION",
            "guidance": "Missing verification evidence (e.g. unknown CI) or elevated churn/complexity detected.",
            "required_actions": [
                "Require at least one senior peer review approval",
                "Execute manual or automated smoke tests on staging before promotion",
            ],
        }
    # NO_BLOCKING_FINDINGS
    return {
        "decision": "AUTOMATED_DEPLOY_PERMITTED",
        "action_level": "STANDARD",
        "guidance": "All automated gates passed and ML risk is within normal parameters.",
        "required_actions": [
            "Proceed with standard automated deployment pipeline",
            "Monitor post-release operational telemetry for 30-day window",
        ],
    }

