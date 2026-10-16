"""Software Release Risk Predictor (RRP) - Streamlit Dashboard.

An evidence-based Verification & Validation platform for inspecting code metrics,
evaluating CI evidence, assessing calibrated ML defect risk, enforcing release quality gates,
conducting subgroup analysis, and monitoring post-release operational telemetry.
"""

from __future__ import annotations

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

from src.pipeline import run_full_assessment
from src.report_generator import format_markdown_report
from src.monitoring import (
    record_release_prediction,
    update_post_release_outcome,
    compute_monitoring_health,
    MONITORING_LOG_FILE,
)

# Page Configuration
st.set_page_config(
    page_title="Software Release Risk Predictor (RRP)",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1e222b;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #30363d;
    }
    .decision-banner {
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("Release Risk Predictor")
    st.caption("AI-Assisted Software Verification & Validation")

    st.markdown("---")
    st.subheader("⚙️ Assessment Settings")
    risk_threshold = st.slider(
        "ML Risk Review Threshold",
        min_value=0.50,
        max_value=0.90,
        value=0.70,
        step=0.05,
        help="Probability threshold at or above which the model flags the release for manual engineering review.",
    )

    github_token = st.text_input(
        "GitHub API Token (Optional)",
        type="password",
        help="Recommended for public GitHub repositories to prevent 60 req/hr rate limits.",
    )

    st.markdown("---")
    st.markdown("### 📚 Key V&V Principles")
    st.info(
        "**V&V Core Invariants:**\n"
        "• **Leakage-Free:** Only pre-release features used.\n"
        "• **Calibration:** Probabilities represent real defect frequencies.\n"
        "• **Subgroup Slicing:** Disparities across repo size & churn tracked.\n"
        "• **Release Action:** We output **NO_BLOCKING_FINDINGS**, never 'SAFE_TO_RELEASE'."
    )

# Header
st.title("🛡️ Software Release Risk Predictor (RRP)")
st.markdown(
    "Estimate software release defect risk from code changes, test results, historical defects, "
    "and deployment metadata with rigorous Verification & Validation assurance."
)

# Repository Input Section
col_input, col_preset = st.columns([3, 2])

with col_input:
    repo_url_input = st.text_input(
        "Repository Location (Public GitHub URL or Local Path)",
        placeholder="https://github.com/owner/repository or . for current repo",
        value=".",
    )

with col_preset:
    preset_choice = st.selectbox(
        "Quick Presets",
        options=[
            "Custom Input Above",
            "Local Workspace (.)",
            "psf/requests (Canonical Python)",
            "pallets/flask (Microframework)",
            "encode/httpx (Async HTTP)",
        ],
    )

# Handle preset selection
target_url = repo_url_input
if preset_choice == "Local Workspace (.)":
    target_url = "."
elif preset_choice == "psf/requests (Canonical Python)":
    target_url = "https://github.com/psf/requests"
elif preset_choice == "pallets/flask (Microframework)":
    target_url = "https://github.com/pallets/flask"
elif preset_choice == "encode/httpx (Async HTTP)":
    target_url = "https://github.com/encode/httpx"

analyze_btn = st.button("🚀 Analyze Repository & Assess Release Risk", type="primary")

if "assessment_data" not in st.session_state:
    st.session_state["assessment_data"] = None

if analyze_btn:
    if not target_url.strip():
        st.error("Please provide a repository URL or local path.")
    else:
        with st.spinner("Inspecting repository, calculating Radon metrics, and evaluating gates..."):
            try:
                res = run_full_assessment(
                    target_url.strip(),
                    token=github_token.strip() if github_token else None,
                    risk_threshold=risk_threshold,
                )
                st.session_state["assessment_data"] = res

                # Automatically record assessment to post-release monitoring log
                record_release_prediction(
                    repo_name=res["repo_info"].get("full_name") or res["repo_info"].get("name"),
                    commit_sha=res["commit_info"].get("hash", "UNKNOWN"),
                    predicted_risk=res["prediction"].get("predicted_risk"),
                    features=res["features"],
                    decision=res["gate_results"].get("release_decision", {}).get("decision", "UNKNOWN"),
                )
                st.success("Analysis complete and recorded for post-release monitoring!")
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

data = st.session_state.get("assessment_data")

if data:
    repo = data["repo_info"]
    commit = data["commit_info"]
    metrics = data["code_metrics"]
    ci = data["ci_evidence"]
    pred = data["prediction"]
    gates = data["gate_results"]
    dossier = data["audit_dossier"]
    decision = gates.get("release_decision", {})

    st.markdown("---")

    # Executive Verdict & Release Governance Decision Banner
    status = gates["overall_status"]
    dec_type = decision.get("decision", "REVIEW_REQUIRED")

    if dec_type == "AUTOMATED_DEPLOY_PERMITTED":
        st.markdown(
            f"""
            <div style="background-color: #132a1c; border-left: 6px solid #2ecc71; padding: 16px; border-radius: 6px;">
                <h3 style="color: #2ecc71; margin: 0;">🚀 RELEASE DECISION: AUTOMATED DEPLOYMENT PERMITTED</h3>
                <p style="margin: 8px 0 4px 0; color: #d0f0d8;"><strong>Gate Verdict:</strong> {status} — {gates['summary']}</p>
                <p style="margin: 0; color: #a3e9b6;"><strong>Governance Guidance:</strong> {decision.get('guidance')}</p>
                <small style="color: #8cd69e;">{gates['disclaimer']}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif dec_type == "STAGED_ROLLOUT_AND_QA_SIGNOFF":
        st.markdown(
            f"""
            <div style="background-color: #3b1115; border-left: 6px solid #e74c3c; padding: 16px; border-radius: 6px;">
                <h3 style="color: #e74c3c; margin: 0;">⚠️ RELEASE DECISION: MANDATORY STAGED ROLLOUT & QA SIGN-OFF</h3>
                <p style="margin: 8px 0 4px 0; color: #fcdede;"><strong>Gate Verdict:</strong> {status} — {gates['summary']}</p>
                <p style="margin: 0; color: #f8b4bb;"><strong>Governance Guidance:</strong> {decision.get('guidance')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif dec_type == "PEER_REVIEW_AND_SMOKE_VERIFICATION":
        st.markdown(
            f"""
            <div style="background-color: #382900; border-left: 6px solid #f39c12; padding: 16px; border-radius: 6px;">
                <h3 style="color: #f39c12; margin: 0;">🔍 RELEASE DECISION: PEER REVIEW & SMOKE VERIFICATION REQUIRED</h3>
                <p style="margin: 8px 0 4px 0; color: #fef5e7;"><strong>Gate Verdict:</strong> {status} — {gates['summary']}</p>
                <p style="margin: 0; color: #fcdba0;"><strong>Governance Guidance:</strong> {decision.get('guidance')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background-color: #2b0c10; border-left: 6px solid #ff4d4f; padding: 16px; border-radius: 6px;">
                <h3 style="color: #ff4d4f; margin: 0;">🛑 RELEASE DECISION: HARD BLOCK</h3>
                <p style="margin: 8px 0 4px 0; color: #ffe6e6;"><strong>Gate Verdict:</strong> {status} — {gates['summary']}</p>
                <p style="margin: 0; color: #ffb3b5;"><strong>Governance Guidance:</strong> {decision.get('guidance')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # Top KPI Metrics row
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Files Analyzed", metrics.get("files_analyzed", 0))
    kpi2.metric("Total LOC", f"{metrics.get('total_loc', 0):,}")
    kpi3.metric("Avg Complexity (CC)", f"{metrics.get('avg_complexity', 0.0):.1f}")
    kpi4.metric("Maintainability (MI)", f"{metrics.get('avg_maintainability_index', 0.0):.1f}/100")
    if pred.get("predicted_risk") is not None:
        kpi5.metric("ML Defect Risk", f"{pred['predicted_risk']:.1%}", delta=pred.get("risk_level"))
    else:
        kpi5.metric("ML Defect Risk", "N/A", help="Model is out of applicable domain")

    # Navigation Tabs
    (
        tab_overview,
        tab_metrics,
        tab_ci,
        tab_ml,
        tab_gates,
        tab_subgroups,
        tab_monitoring,
        tab_export,
    ) = st.tabs([
        "📋 Overview",
        "📊 Code Metrics",
        "🧪 Test & CI Evidence",
        "🧠 ML Risk & Explainability",
        "🛡️ Quality Gates & Decision",
        "🔬 Calibration & Subgroups",
        "📡 Post-Release Monitoring",
        "📥 Export Audit Report",
    ])

    # TAB 1: OVERVIEW
    with tab_overview:
        st.subheader("Repository & Target Commit Specification")
        o_col1, o_col2 = st.columns(2)
        with o_col1:
            st.markdown(f"**Repository Name:** `{repo.get('full_name') or repo.get('name')}`")
            st.markdown(f"**Default Branch:** `{repo.get('default_branch')}`")
            st.markdown(f"**Primary Language:** `{repo.get('language')}`")
            st.markdown(f"**Description:** {repo.get('description')}")
            if repo.get("html_url") and not repo.get("html_url").startswith("/"):
                st.markdown(f"**GitHub URL:** [{repo.get('html_url')}]({repo.get('html_url')})")
        with o_col2:
            st.markdown(f"**Target Commit SHA:** `{commit.get('hash')}`")
            st.markdown(f"**Author:** {commit.get('author_name')}")
            st.markdown(f"**Commit Date:** {commit.get('commit_date')}")
            st.markdown(f"**Commit Message:** *{commit.get('message')}*")
            st.markdown(f"**Commit Churn:** +{data['stats'].get('lines_added', 0)} / -{data['stats'].get('lines_deleted', 0)} ({data['stats'].get('files_changed', 0)} files)")

    # TAB 2: CODE METRICS
    with tab_metrics:
        st.subheader("Radon Static Code & Complexity Analysis")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Max File Complexity", metrics.get("max_complexity", 0))
        m_col2.metric("High Complexity Files (>10 CC)", metrics.get("high_complexity_files_count", 0))
        m_col3.metric("Comment Ratio", f"{metrics.get('comment_ratio', 0.0) * 100:.1f}%")

        files_data = metrics.get("files", [])
        if files_data:
            st.markdown("#### Analyzed Python Source Files")
            df_files = pd.DataFrame(files_data)
            display_cols = [c for c in ["file", "rel_path", "loc", "total_complexity", "max_complexity", "maintainability_index", "maintainability_rank", "function_count", "class_count"] if c in df_files.columns]
            st.dataframe(df_files[display_cols], use_container_width=True)
        else:
            st.info("No individual Python files recorded or analysis was API-only.")

    # TAB 3: CI EVIDENCE
    with tab_ci:
        st.subheader("Verification Evidence & Test Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("CI Workflow Status", ci.get("ci_status", "unknown").upper())
        c2.metric("Test Suite Status", ci.get("test_status", "unknown").upper())
        c3.metric("Passed / Failed Checks", f"{ci.get('passed_count', 0)} Passed / {ci.get('failed_count', 0)} Failed")

        st.info(f"**Verification Note:** {ci.get('evidence_note')}")

        details = ci.get("details", [])
        if details:
            st.markdown("#### Individual Check Runs & Workflows")
            for d in details:
                icon = "✅" if d["status"] == "PASS" else ("❌" if d["status"] == "FAIL" else "⏳")
                st.write(f"{icon} **{d['type']}: {d['name']}** — `{d['status']}`")
        else:
            st.warning("⚠️ No CI workflow runs or check runs are published for this commit.")

    # TAB 4: ML RISK & EXPLAINABILITY
    with tab_ml:
        st.subheader("Calibrated Defect-Risk Prediction & Local Feature Attributions")
        if not pred.get("is_applicable"):
            st.warning(f"⚠️ **ML Prediction Inapplicable:** {pred.get('inapplicability_reason')}")
            st.caption("Per V&V rules, when a repository is outside the model's domain, we refuse prediction rather than hallucinating a risk number.")
        else:
            p_col1, p_col2 = st.columns([1, 2])
            with p_col1:
                st.metric("Predicted Defect Risk", f"{pred['predicted_risk']:.1%}", delta=pred['risk_level'])
                st.write(f"**Risk Level:** `{pred['risk_level']}`")
                st.write(f"**Review Threshold:** `{pred['risk_threshold']:.0%}`")
                st.write(f"**Model Algorithm:** {pred.get('model_algorithm')}")
                st.write(f"**Model Version:** `{pred.get('model_version')}`")

            with p_col2:
                st.markdown("#### Contributing Explanation Factors (Step 22)")
                factors = pred.get("explanation_factors", [])
                for f in factors:
                    icon = "🔍" if f["type"] == "MODEL_CONTRIBUTION" else ("📜" if f["type"] == "HISTORICAL_CONTEXT" else "🧪")
                    st.markdown(f"- {icon} **{f['factor']}:** {f['detail']}")

            st.markdown("#### Local Mathematical Feature Attribution Decomposition")
            st.caption("Decomposition of this specific release's feature contributions relative to model weights:")
            attributions = pred.get("feature_attributions", [])
            if attributions:
                df_attr = pd.DataFrame(attributions)
                st.dataframe(df_attr, use_container_width=True)
                st.bar_chart(df_attr.set_index("feature")["contribution_score"])

    # TAB 5: QUALITY GATES & DECISION
    with tab_gates:
        st.subheader("Release Governance Action Plan")
        st.markdown(f"**Action Level:** `{decision.get('action_level')}`")
        st.markdown(f"**Decision Summary:** {decision.get('guidance')}")

        st.markdown("#### Required Operational Actions:")
        for act in decision.get("required_actions", []):
            st.markdown(f"- [ ] {act}")

        st.markdown("---")
        st.subheader("Independent Quality Gate Statuses")
        for gate in gates.get("findings", []):
            g_status = gate["status"]
            g_name = gate["gate"]
            g_reason = gate["reason"]

            if g_status == "PASS":
                st.success(f"**{g_name}**: PASS — {g_reason}")
            elif g_status == "FAIL":
                st.error(f"**{g_name}**: FAIL — {g_reason}")
            elif g_status == "WARN":
                st.warning(f"**{g_name}**: WARNING — {g_reason}")
            elif g_status == "UNKNOWN":
                st.info(f"**{g_name}**: UNKNOWN — {g_reason}")
            elif g_status == "BLOCKED":
                st.error(f"**{g_name}**: BLOCKED — {g_reason}")
            else:
                st.write(f"**{g_name}**: {g_status} — {g_reason}")

    # TAB 6: CALIBRATION & SUBGROUPS
    with tab_subgroups:
        st.subheader("V&V Deep Dive: Calibration & Subgroup Slicing")
        meta_path = Path("models/model_metadata.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            test_m = meta.get("test_metrics", {})

            st.markdown("#### Probability Calibration Assessment")
            cal1, cal2, cal3 = st.columns(3)
            cal1.metric("Expected Calibration Error (ECE)", f"{test_m.get('expected_calibration_error', 0.0):.4f}")
            cal2.metric("Brier Score Loss", f"{test_m.get('brier_score', 0.0):.4f}")
            cal3.metric("Calibration Status", "Calibrated (Platt/Sigmoid)")

            binned = test_m.get("calibration", {}).get("binned_details", [])
            if binned:
                st.markdown("##### Binned Reliability Coordinates")
                df_binned = pd.DataFrame(binned)
                st.dataframe(df_binned, use_container_width=True)

            st.markdown("#### Subgroup Performance Slicing (Repository Scale & Churn)")
            subgroups = meta.get("subgroup_analysis", {})
            if subgroups:
                sub_rows = []
                for name, s_data in subgroups.items():
                    sub_rows.append({
                        "Subgroup": name,
                        "Sample Count": s_data.get("sample_count"),
                        "Positive Count": s_data.get("positive_count"),
                        "Prevalence": f"{s_data.get('positive_prevalence', 0.0):.1%}" if s_data.get("positive_prevalence") is not None else "N/A",
                        "Precision": s_data.get("precision"),
                        "Recall": s_data.get("recall"),
                        "F1": s_data.get("f1"),
                        "PR-AUC": s_data.get("pr_auc"),
                    })
                st.dataframe(pd.DataFrame(sub_rows), use_container_width=True)
        else:
            st.info("No model metadata file found. Run training pipeline to generate.")

    # TAB 7: POST-RELEASE MONITORING
    with tab_monitoring:
        st.subheader("Post-Release Telemetry & 30-Day Follow-Up")
        st.markdown(
            "Every assessed release is tracked for 30 days post-deployment to observe "
            "empirical defect occurrences and monitor feature distribution drift (PSI)."
        )

        health = compute_monitoring_health()
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Total Logged Releases", health.get("total_logged_releases", 0))
        h2.metric("Completed 30-Day Follow-Ups", health.get("completed_follow_ups", 0))
        h3.metric("Pending 30-Day Follow-Ups", health.get("pending_follow_ups", 0))
        h4.metric("Monitoring Health Status", health.get("status", "HEALTHY"))

        st.markdown("#### Record 30-Day Post-Release Outcome")
        with st.form("outcome_form"):
            form_sha = st.text_input("Commit SHA to Update", value=commit.get("hash", "")[:8])
            form_defect = st.selectbox("Observed Qualifying Defect in 30-day Window?", ["No (0 Defect)", "Yes (Defect Observed)"])
            form_notes = st.text_input("Incident / Issue Reference Notes", placeholder="e.g. Issue #142 hotfixed in prod")
            submit_outcome = st.form_submit_button("Save Post-Release Outcome")

            if submit_outcome:
                has_defect = "Yes" in form_defect
                updated = update_post_release_outcome(form_sha, has_defect, form_notes)
                if updated:
                    st.success(f"Outcome successfully recorded for commit {form_sha}!")
                    st.rerun()
                else:
                    st.warning(f"Commit SHA {form_sha} not found in monitoring log.")

        if MONITORING_LOG_FILE.exists():
            st.markdown("#### Recent Release Telemetry Log")
            with open(MONITORING_LOG_FILE, "r", encoding="utf-8") as f:
                log_recs = [json.loads(line) for line in f if line.strip()]
            if log_recs:
                df_log = pd.DataFrame(log_recs)[["timestamp", "repo_name", "commit_sha", "predicted_risk", "decision", "follow_up_completed", "actual_defect_observed"]]
                st.dataframe(df_log.tail(10), use_container_width=True)

    # TAB 8: REPORT EXPORT
    with tab_export:
        st.subheader("Audit Dossier Export")
        st.write("Download the comprehensive verification and risk assessment report.")

        dossier_json = json.dumps(dossier, indent=2)
        dossier_md = format_markdown_report(dossier)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                "📥 Download JSON Audit Dossier",
                data=dossier_json,
                file_name=f"rrp_audit_{commit.get('hash', 'unknown')[:8]}.json",
                mime="application/json",
            )
        with col_d2:
            st.download_button(
                "📄 Download Markdown Executive Report",
                data=dossier_md,
                file_name=f"rrp_report_{commit.get('hash', 'unknown')[:8]}.md",
                mime="text/markdown",
            )

        with st.expander("Preview Markdown Audit Report"):
            st.markdown(dossier_md)

# Footer
st.markdown("---")
st.caption(
    "Software Release Risk Predictor (RRP) • Verification & Validation Engineering • "
    "Designed for laptop-friendly ML (Ryzen 5 5500U, 16GB RAM) • Compliant with V&V standards."
)
