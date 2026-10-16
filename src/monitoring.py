"""Post-Release Monitoring, Operational Feedback, and Distribution Drift Detection.

Tracks releases across the 30-day post-deployment observation window, records actual
observed defects, computes empirical Brier score over time, and calculates
Population Stability Index (PSI) to detect feature distribution drift.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

MONITORING_LOG_FILE = Path("data/monitoring_log.jsonl")


def record_release_prediction(
    repo_name: str,
    commit_sha: str,
    predicted_risk: float | None,
    features: dict[str, Any],
    decision: str,
    log_file: Path | str = MONITORING_LOG_FILE,
) -> dict[str, Any]:
    """Log an assessed release for post-release follow-up and monitoring."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "event_id": f"REL-{commit_sha[:8]}-{int(datetime.now().timestamp())}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "repo_name": repo_name,
        "commit_sha": commit_sha,
        "predicted_risk": predicted_risk,
        "decision": decision,
        "features": features,
        "actual_defect_observed": None,  # Populated after 30-day follow-up
        "follow_up_completed": False,
    }

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return record


def update_post_release_outcome(
    commit_sha: str,
    defect_observed: bool,
    notes: str = "",
    log_file: Path | str = MONITORING_LOG_FILE,
) -> bool:
    """Record observed defect outcome once the 30-day post-release window elapses."""
    log_path = Path(log_file)
    if not log_path.exists():
        return False

    updated = False
    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("commit_sha") == commit_sha:
                rec["actual_defect_observed"] = 1 if defect_observed else 0
                rec["follow_up_completed"] = True
                rec["outcome_recorded_at"] = datetime.now(timezone.utc).isoformat()
                rec["outcome_notes"] = notes
                updated = True
            records.append(rec)

    if updated:
        with open(log_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    return updated


def calculate_psi(
    expected_distribution: np.ndarray,
    actual_distribution: np.ndarray,
    num_bins: int = 5,
) -> float:
    """Calculate Population Stability Index (PSI) to detect feature drift.

    PSI Interpretation:
    - PSI < 0.10: No significant distribution change.
    - 0.10 <= PSI < 0.25: Moderate drift; inspection recommended.
    - PSI >= 0.25: Significant drift; triggers model retraining alert.
    """
    if len(expected_distribution) == 0 or len(actual_distribution) == 0:
        return 0.0

    # Quantile bin edges based on expected (training baseline)
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(expected_distribution, percentiles)
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5

    # Count occurrences
    exp_counts, _ = np.histogram(expected_distribution, bins=bin_edges)
    act_counts, _ = np.histogram(actual_distribution, bins=bin_edges)

    # Convert to proportions with Laplace smoothing
    exp_props = (exp_counts + 1e-4) / (np.sum(exp_counts) + 1e-4 * num_bins)
    act_props = (act_counts + 1e-4) / (np.sum(act_counts) + 1e-4 * num_bins)

    psi_val = np.sum((act_props - exp_props) * np.log(act_props / exp_props))
    return float(round(psi_val, 4))


def compute_monitoring_health(
    log_file: Path | str = MONITORING_LOG_FILE,
    baseline_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compute monitoring summary, calibration health, and distribution drift alerts."""
    log_path = Path(log_file)
    if not log_path.exists():
        return {
            "total_logged_releases": 0,
            "completed_follow_ups": 0,
            "pending_follow_ups": 0,
            "brier_score": None,
            "feature_drift": {},
            "status": "NO_MONITORING_DATA",
        }

    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return {
            "total_logged_releases": 0,
            "completed_follow_ups": 0,
            "status": "NO_MONITORING_DATA",
        }

    df_logs = pd.DataFrame(records)
    total_releases = len(df_logs)
    completed = df_logs[df_logs["follow_up_completed"] == True]

    # Calculate empirical monitoring Brier score if outcomes are available
    brier = None
    calibration_gap = None
    if len(completed) >= 5 and "actual_defect_observed" in completed.columns:
        valid_pairs = completed.dropna(subset=["predicted_risk", "actual_defect_observed"])
        if len(valid_pairs) >= 5:
            preds = valid_pairs["predicted_risk"].values.astype(float)
            actuals = valid_pairs["actual_defect_observed"].values.astype(float)
            brier = float(np.mean((preds - actuals) ** 2))
            calibration_gap = float(abs(np.mean(preds) - np.mean(actuals)))

    # Compute PSI drift on key numerical features against baseline training data
    drift_report = {}
    if baseline_df is not None and not baseline_df.empty and total_releases >= 10:
        key_features = ["cyclomatic_complexity", "total_churn", "loc"]
        for feat in key_features:
            if feat in baseline_df.columns:
                live_values = []
                for _, row in df_logs.iterrows():
                    f_dict = row.get("features", {})
                    if feat in f_dict and f_dict[feat] is not None:
                        live_values.append(float(f_dict[feat]))

                if len(live_values) >= 10:
                    psi_score = calculate_psi(baseline_df[feat].values, np.array(live_values))
                    level = "CRITICAL_DRIFT" if psi_score >= 0.25 else ("MODERATE_DRIFT" if psi_score >= 0.10 else "STABLE")
                    drift_report[feat] = {
                        "psi_score": psi_score,
                        "drift_level": level,
                    }

    return {
        "total_logged_releases": total_releases,
        "completed_follow_ups": len(completed),
        "pending_follow_ups": total_releases - len(completed),
        "brier_score": round(brier, 4) if brier is not None else None,
        "mean_calibration_gap": round(calibration_gap, 4) if calibration_gap is not None else None,
        "feature_drift": drift_report,
        "status": "HEALTHY" if (not drift_report or all(v["drift_level"] == "STABLE" for v in drift_report.values())) else "DRIFT_ALERT",
    }
