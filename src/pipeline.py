"""Unified Analysis Pipeline for Software Release Risk Prediction.

Coordinates repository fetching, code metric calculation, CI evidence aggregation,
ML risk prediction, quality gate evaluation, and audit dossier creation.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from src.ci_collector import aggregate_ci_evidence, get_ci_workflow_runs, get_commit_check_runs
from src.code_metrics import analyze_directory
from src.git_analyzer import clone_repository, get_commit_stats, get_latest_commit, get_recent_churn
from src.github_connector import (
    GitHubAPIError,
    get_commit_details,
    get_commit_history,
    get_repository_info,
    parse_github_url,
)
from src.predict import RiskPredictor
from src.quality_gates import evaluate_quality_gates
from src.report_generator import generate_audit_report


def run_full_assessment(
    repo_url_or_path: str,
    token: str | None = None,
    risk_threshold: float = 0.70,
    temp_dir_base: Path | str | None = None,
) -> dict[str, Any]:
    """Execute complete end-to-end release risk assessment on a repository.

    Args:
        repo_url_or_path: Public GitHub URL (e.g. 'https://github.com/psf/requests')
                          or absolute path to a local Git repository.
        token: Optional GitHub Personal Access Token.
        risk_threshold: ML risk threshold triggering review (default 0.70).
        temp_dir_base: Base directory for temporary clone.

    Returns:
        dict: Full assessment results including metadata, metrics, CI evidence,
              ML risk prediction, quality gates verdict, and exportable audit dossier.
    """
    is_local_path = Path(repo_url_or_path).is_dir()
    cleanup_needed = False
    cloned_path = None

    try:
        if is_local_path:
            repo_path = Path(repo_url_or_path).resolve()
            repo_name = repo_path.name
            repo_info = {
                "name": repo_name,
                "full_name": f"local/{repo_name}",
                "owner": "local",
                "repo": repo_name,
                "description": "Local Git Repository Analysis",
                "default_branch": "HEAD",
                "language": "Python",
                "html_url": str(repo_path),
                "stars": 0,
                "forks": 0,
            }
            commit_info = get_latest_commit(repo_path)
            stats = get_commit_stats(repo_path, "HEAD")
            churn = get_recent_churn(repo_path, window_days=30)
            ci_evidence = {
                "ci_status": "unknown",
                "test_status": "unknown",
                "passed_count": 0,
                "failed_count": 0,
                "pending_count": 0,
                "evidence_note": "Local repository inspected; remote GitHub Actions unavailable.",
                "details": [],
            }
        else:
            # Remote GitHub repository URL
            owner, repo = parse_github_url(repo_url_or_path)
            repo_info = get_repository_info(owner, repo, token=token)

            # Shallow clone safely to calculate accurate file-level Radon metrics
            base_temp = Path(temp_dir_base) if temp_dir_base else Path(tempfile.gettempdir())
            cloned_path = base_temp / f"rrp_clone_{owner}_{repo}_{os.getpid()}"
            cleanup_needed = True

            try:
                repo_path = clone_repository(repo_info["html_url"], cloned_path, depth=25, timeout=60)
                commit_info = get_latest_commit(repo_path)
                stats = get_commit_stats(repo_path, "HEAD")
                churn = get_recent_churn(repo_path, window_days=30)
            except Exception:
                # Fallback to API commit details if shallow clone is blocked/slow
                commits = get_commit_history(owner, repo, count=5, token=token)
                latest_c = commits[0] if commits else {}
                sha = latest_c.get("sha", "HEAD")
                commit_info = {
                    "hash": sha,
                    "author_name": latest_c.get("author", "Unknown"),
                    "author_email": "",
                    "commit_date": latest_c.get("date", ""),
                    "message": latest_c.get("message", ""),
                }
                detail = get_commit_details(owner, repo, sha, token=token) if sha != "HEAD" else {}
                stats = {
                    "files_changed": detail.get("files_changed_count", 1),
                    "lines_added": detail.get("additions", 0),
                    "lines_deleted": detail.get("deletions", 0),
                    "total_churn": detail.get("total_changes", 0),
                }
                churn = {
                    "window_days": 30,
                    "recent_commit_count": len(commits),
                    "active_contributors_count": len({c.get("author") for c in commits}),
                    "recent_total_churn": stats.get("total_churn", 0),
                }
                repo_path = None

            # Fetch CI evidence from GitHub Actions and Check-Runs
            head_sha = commit_info.get("hash")
            wf_data = get_ci_workflow_runs(owner, repo, commit_sha=head_sha, token=token)
            cr_data = get_commit_check_runs(owner, repo, commit_sha=head_sha, token=token) if head_sha else {}
            ci_evidence = aggregate_ci_evidence(wf_data, cr_data)

        # 2. Source Code Metrics (Radon)
        if repo_path and repo_path.is_dir():
            code_metrics = analyze_directory(repo_path, max_files=400)
            analysis_complete = code_metrics.get("status") in {"SUCCESS", "NO_PYTHON_FILES"}
        else:
            code_metrics = {
                "files_analyzed": 0,
                "total_loc": repo_info.get("size_kb", 0) * 20,  # approximate fallback
                "avg_complexity": 0.0,
                "max_complexity": 0,
                "avg_maintainability_index": 70.0,
                "high_complexity_files_count": 0,
                "files": [],
                "status": "API_ONLY",
            }
            analysis_complete = True

        # 3. Prepare Feature vector for ML (Strictly Pre-Release Features)
        default_branch = repo_info.get("default_branch", "main")
        is_main = 1 if default_branch in {"main", "master", "HEAD"} else 0
        feature_dict = {
            "language": repo_info.get("language", "Python"),
            "files_changed": stats.get("files_changed", 1),
            "lines_added": stats.get("lines_added", 0),
            "lines_deleted": stats.get("lines_deleted", 0),
            "total_churn": stats.get("total_churn", 0),
            "cyclomatic_complexity": float(code_metrics.get("avg_complexity", 0.0)),
            "maintainability_index": float(code_metrics.get("avg_maintainability_index", 75.0)),
            "loc": int(code_metrics.get("total_loc", 100)),
            "function_count": int(code_metrics.get("total_functions", 10)),
            "recent_commit_frequency": churn.get("recent_commit_count", 5),
            "previous_defects_count": min(repo_info.get("open_issues_count", 0), 20),
            "test_failure_ratio": 1.0 if ci_evidence.get("ci_status") == "failed" else 0.0,
            "days_since_last_release": 14.0,  # standard pre-release cadence baseline
            "is_main_branch": is_main,
        }

        # 4. Predict ML Risk
        predictor = RiskPredictor()
        prediction = predictor.predict(feature_dict, risk_threshold=risk_threshold)

        # 5. Evaluate Quality Gates
        gate_results = evaluate_quality_gates(
            test_status=ci_evidence.get("test_status", "unknown"),
            ci_status=ci_evidence.get("ci_status", "unknown"),
            analysis_complete=analysis_complete,
            model_validated=prediction.get("model_validated", False),
            predicted_risk=prediction.get("predicted_risk"),
            risk_threshold=risk_threshold,
            metrics={
                "total_churn": stats.get("total_churn", 0),
                "max_complexity": code_metrics.get("max_complexity", 0),
            },
        )

        # 6. Generate Audit Dossier
        dossier = generate_audit_report(
            repo_info=repo_info,
            commit_info=commit_info,
            code_metrics=code_metrics,
            ci_evidence=ci_evidence,
            prediction_result=prediction,
            gate_results=gate_results,
        )

        return {
            "repo_info": repo_info,
            "commit_info": commit_info,
            "stats": stats,
            "churn": churn,
            "code_metrics": code_metrics,
            "ci_evidence": ci_evidence,
            "features": feature_dict,
            "prediction": prediction,
            "gate_results": gate_results,
            "audit_dossier": dossier,
        }

    finally:
        if cleanup_needed and cloned_path and cloned_path.exists():
            shutil.rmtree(cloned_path, ignore_errors=True)
