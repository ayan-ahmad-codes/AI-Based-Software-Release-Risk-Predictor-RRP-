"""CI and Test Evidence Collector.

Fetches GitHub Actions workflow runs and check-runs via GitHub REST API,
and aggregates verification status without assuming missing evidence means safety.
"""

from __future__ import annotations

import os
from typing import Any
import requests
from dotenv import load_dotenv

from src.github_connector import GITHUB_API_BASE, get_headers, GitHubAPIError

load_dotenv()


def get_ci_workflow_runs(
    owner: str,
    repo: str,
    commit_sha: str | None = None,
    branch: str | None = None,
    token: str | None = None,
    timeout: int = 15,
) -> dict[str, Any]:
    """Fetch GitHub Actions workflow runs for a repository or specific commit."""
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/actions/runs"
    params: dict[str, Any] = {"per_page": 20}
    if branch:
        params["branch"] = branch
    if commit_sha:
        params["head_sha"] = commit_sha

    headers = get_headers(token)
    try:
        response = requests.get(url, headers=headers, params=params, timeout=timeout)
    except requests.RequestException as exc:
        return {
            "status": "ERROR",
            "error": f"Network error fetching CI runs: {exc}",
            "runs": [],
            "total_count": 0,
        }

    if response.status_code == 404:
        # Actions might be disabled or repo not found
        return {
            "status": "UNAVAILABLE",
            "reason": "GitHub Actions not enabled or inaccessible for this repository.",
            "runs": [],
            "total_count": 0,
        }
    if response.status_code != 200:
        return {
            "status": "ERROR",
            "error": f"API returned HTTP {response.status_code}: {response.text[:200]}",
            "runs": [],
            "total_count": 0,
        }

    data = response.json()
    raw_runs = data.get("workflow_runs", [])
    parsed_runs = []
    for r in raw_runs:
        parsed_runs.append({
            "id": r.get("id"),
            "name": r.get("name"),
            "head_sha": r.get("head_sha"),
            "event": r.get("event"),
            "status": r.get("status"),  # queued, in_progress, completed
            "conclusion": r.get("conclusion"),  # success, failure, neutral, cancelled, timed_out, None
            "created_at": r.get("created_at"),
            "updated_at": r.get("updated_at"),
            "html_url": r.get("html_url"),
        })

    return {
        "status": "SUCCESS",
        "total_count": data.get("total_count", len(parsed_runs)),
        "runs": parsed_runs,
    }


def get_commit_check_runs(
    owner: str,
    repo: str,
    commit_sha: str,
    token: str | None = None,
    timeout: int = 15,
) -> dict[str, Any]:
    """Fetch GitHub Check Runs for a specific commit SHA."""
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/commits/{commit_sha}/check-runs"
    headers = get_headers(token)
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        return {
            "status": "ERROR",
            "error": f"Network error fetching check runs: {exc}",
            "check_runs": [],
            "total_count": 0,
        }

    if response.status_code != 200:
        return {
            "status": "UNAVAILABLE",
            "reason": f"Check runs unavailable ({response.status_code})",
            "check_runs": [],
            "total_count": 0,
        }

    data = response.json()
    raw_checks = data.get("check_runs", [])
    parsed_checks = []
    for c in raw_checks:
        parsed_checks.append({
            "id": c.get("id"),
            "name": c.get("name"),
            "status": c.get("status"),  # completed, in_progress, queued
            "conclusion": c.get("conclusion"),  # success, failure, neutral, action_required, None
            "html_url": c.get("html_url"),
            "started_at": c.get("started_at"),
            "completed_at": c.get("completed_at"),
        })

    return {
        "status": "SUCCESS",
        "total_count": data.get("total_count", len(parsed_checks)),
        "check_runs": parsed_checks,
    }


def aggregate_ci_evidence(
    workflow_data: dict[str, Any],
    check_run_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Synthesize CI workflow and check runs into normalized verification evidence.

    Status categories:
    - 'passed': At least one completed run/check, and all completed runs/checks succeeded.
    - 'failed': At least one run/check explicitly failed or timed out.
    - 'unknown': No CI or test evidence found, or runs are incomplete / pending.

    Returns:
        dict: CI summary including ci_status ('passed' | 'failed' | 'unknown'),
              passed_count, failed_count, and limitation details.
    """
    runs = workflow_data.get("runs", [])
    checks = (check_run_data or {}).get("check_runs", [])

    if not runs and not checks:
        return {
            "ci_status": "unknown",
            "test_status": "unknown",
            "passed_count": 0,
            "failed_count": 0,
            "pending_count": 0,
            "total_evidence_count": 0,
            "has_failures": False,
            "evidence_note": "No CI workflows or check runs found for this repository/commit. Status marked unknown per V&V standard.",
            "details": [],
        }

    passed_count = 0
    failed_count = 0
    pending_count = 0
    details = []

    for r in runs:
        conc = (r.get("conclusion") or "").lower()
        stat = (r.get("status") or "").lower()
        name = r.get("name") or "Workflow"

        if conc in {"failure", "timed_out", "action_required"}:
            failed_count += 1
            details.append({"type": "Workflow", "name": name, "status": "FAIL", "url": r.get("html_url")})
        elif conc == "success":
            passed_count += 1
            details.append({"type": "Workflow", "name": name, "status": "PASS", "url": r.get("html_url")})
        elif stat in {"in_progress", "queued"}:
            pending_count += 1
            details.append({"type": "Workflow", "name": name, "status": "PENDING", "url": r.get("html_url")})

    for c in checks:
        conc = (c.get("conclusion") or "").lower()
        stat = (c.get("status") or "").lower()
        name = c.get("name") or "Check"

        if conc in {"failure", "timed_out", "action_required"}:
            failed_count += 1
            details.append({"type": "CheckRun", "name": name, "status": "FAIL", "url": c.get("html_url")})
        elif conc == "success":
            passed_count += 1
            details.append({"type": "CheckRun", "name": name, "status": "PASS", "url": c.get("html_url")})
        elif stat in {"in_progress", "queued"}:
            pending_count += 1
            details.append({"type": "CheckRun", "name": name, "status": "PENDING", "url": c.get("html_url")})

    if failed_count > 0:
        ci_status = "failed"
        test_status = "failed"
        note = f"{failed_count} CI/test check(s) failed."
    elif passed_count > 0 and pending_count == 0:
        ci_status = "passed"
        test_status = "passed"
        note = f"All {passed_count} observed CI/test check(s) passed."
    elif pending_count > 0:
        ci_status = "unknown"
        test_status = "unknown"
        note = f"CI checks are in progress ({pending_count} pending)."
    else:
        ci_status = "unknown"
        test_status = "unknown"
        note = "No conclusive pass/fail runs found."

    return {
        "ci_status": ci_status,
        "test_status": test_status,
        "passed_count": passed_count,
        "failed_count": failed_count,
        "pending_count": pending_count,
        "total_evidence_count": len(details),
        "has_failures": failed_count > 0,
        "evidence_note": note,
        "details": details,
    }
