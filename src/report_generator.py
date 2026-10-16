"""Audit and Verification Report Generator.

Exports structured JSON and formatted Markdown release risk audit reports
containing commit hashes, metrics, quality gate findings, model versions, and limitations.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def generate_audit_report(
    repo_info: dict[str, Any],
    commit_info: dict[str, Any],
    code_metrics: dict[str, Any],
    ci_evidence: dict[str, Any],
    prediction_result: dict[str, Any],
    gate_results: dict[str, Any],
) -> dict[str, Any]:
    """Compile comprehensive release assessment audit dossier."""
    timestamp = datetime.now(timezone.utc).isoformat()

    return {
        "report_id": f"RRP-AUDIT-{commit_info.get('hash', 'UNKNOWN')[:8]}-{int(datetime.now().timestamp())}",
        "generated_at": timestamp,
        "repository": {
            "name": repo_info.get("full_name") or repo_info.get("name"),
            "owner": repo_info.get("owner"),
            "language": repo_info.get("language"),
            "url": repo_info.get("html_url"),
            "default_branch": repo_info.get("default_branch"),
        },
        "target_commit": {
            "hash": commit_info.get("hash"),
            "author": commit_info.get("author_name"),
            "date": commit_info.get("commit_date"),
            "message": commit_info.get("message"),
        },
        "code_metrics_summary": {
            "files_analyzed": code_metrics.get("files_analyzed", 0),
            "total_loc": code_metrics.get("total_loc", 0),
            "avg_complexity": code_metrics.get("avg_complexity", 0.0),
            "max_complexity": code_metrics.get("max_complexity", 0),
            "avg_maintainability_index": code_metrics.get("avg_maintainability_index", 0.0),
            "high_complexity_files_count": code_metrics.get("high_complexity_files_count", 0),
        },
        "verification_evidence": {
            "ci_status": ci_evidence.get("ci_status"),
            "test_status": ci_evidence.get("test_status"),
            "passed_checks": ci_evidence.get("passed_count", 0),
            "failed_checks": ci_evidence.get("failed_count", 0),
            "pending_checks": ci_evidence.get("pending_count", 0),
            "evidence_note": ci_evidence.get("evidence_note"),
        },
        "ml_risk_assessment": {
            "is_applicable": prediction_result.get("is_applicable"),
            "predicted_risk_probability": prediction_result.get("predicted_risk"),
            "risk_level": prediction_result.get("risk_level"),
            "model_version": prediction_result.get("model_version"),
            "model_algorithm": prediction_result.get("model_algorithm"),
            "contributing_factors": prediction_result.get("explanation_factors", []),
        },
        "quality_gates": {
            "overall_status": gate_results.get("overall_status"),
            "has_blocking_findings": gate_results.get("has_blocking_findings"),
            "summary": gate_results.get("summary"),
            "gate_findings": gate_results.get("findings", []),
        },
        "standards_and_limitations": {
            "verdict_meaning": "NO_BLOCKING_FINDINGS indicates absence of automated red flags, not zero defect presence.",
            "ml_limitation": "ML predictions represent statistical risk estimates, not formal proofs.",
            "scope_constraints": "Validated for Python source code; does not cover unobservable post-release operational bugs.",
        },
    }


def format_markdown_report(report_data: dict[str, Any]) -> str:
    """Format audit report into clean, executive-ready Markdown."""
    repo = report_data["repository"]
    commit = report_data["target_commit"]
    metrics = report_data["code_metrics_summary"]
    ci = report_data["verification_evidence"]
    ml = report_data["ml_risk_assessment"]
    gates = report_data["quality_gates"]

    status_icon = {
        "NO_BLOCKING_FINDINGS": "✅",
        "REVIEW_REQUIRED": "⚠️",
        "WARNINGS_DETECTED": "🔍",
        "ASSESSMENT_BLOCKED": "🛑",
    }.get(gates["overall_status"], "ℹ️")

    md = [
        f"# Release Risk Assessment Audit Dossier",
        f"**Audit ID:** `{report_data['report_id']}` | **Generated:** {report_data['generated_at']}",
        "",
        f"## 1. Executive Quality Gate Verdict: {status_icon} **{gates['overall_status']}**",
        f"> {gates['summary']}",
        "",
        "### Quality Gate Findings",
        "| Gate | Status | Reason / Evidence |",
        "|---|---|---|",
    ]

    for f in gates["gate_findings"]:
        icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️", "UNKNOWN": "❓", "BLOCKED": "🛑", "NOT_APPLICABLE": "⚪"}.get(f["status"], "")
        md.append(f"| **{f['gate']}** | {icon} {f['status']} | {f['reason']} |")

    md.extend([
        "",
        "## 2. Target Commit Information",
        f"- **Repository:** [{repo['name']}]({repo.get('url')}) ({repo.get('language')})",
        f"- **Commit SHA:** `{commit.get('hash')}`",
        f"- **Author:** {commit.get('author')}",
        f"- **Date:** {commit.get('date')}",
        f"- **Subject:** `{commit.get('message')}`",
        "",
        "## 3. Code Metrics & Maintainability",
        f"- **Files Analyzed:** {metrics['files_analyzed']}",
        f"- **Total Lines of Code (LOC):** {metrics['total_loc']}",
        f"- **Average Cyclomatic Complexity:** {metrics['avg_complexity']}",
        f"- **Max Cyclomatic Complexity:** {metrics['max_complexity']}",
        f"- **Average Maintainability Index:** {metrics['avg_maintainability_index']}/100",
        f"- **High Complexity Files (>10 CC):** {metrics['high_complexity_files_count']}",
        "",
        "## 4. Verification & CI Evidence",
        f"- **CI Status:** `{ci['ci_status']}` | **Test Status:** `{ci['test_status']}`",
        f"- **Passed Checks:** {ci['passed_checks']} | **Failed Checks:** {ci['failed_checks']} | **Pending:** {ci['pending_checks']}",
        f"- **Evidence Note:** {ci['evidence_note']}",
        "",
        "## 5. Machine Learning Defect Risk Estimate",
        f"- **Applicable Scope:** `{'YES' if ml['is_applicable'] else 'NO'}`",
        f"- **Predicted Defect Probability:** `{f'{ml['predicted_risk_probability']:.1%}' if ml['predicted_risk_probability'] is not None else 'N/A'}`",
        f"- **Risk Level:** `{ml['risk_level']}`",
        f"- **Model Version:** `{ml['model_version']}` ({ml.get('model_algorithm', 'N/A')})",
        "",
        "### Contributing Factors / Explanations:",
    ])

    for factor in ml.get("contributing_factors", []):
        md.append(f"- **{factor.get('factor')}:** {factor.get('detail')}")

    md.extend([
        "",
        "## 6. Verification and Validation Disclaimer",
        f"> [!IMPORTANT]",
        f"> {report_data['standards_and_limitations']['verdict_meaning']}",
        f"> {report_data['standards_and_limitations']['ml_limitation']}",
    ])

    return "\n".join(md)
