from __future__ import annotations
import json
import streamlit as st

from src.services.analytics.reporting import Section, build_html_report, build_pdf_from_html
from src.services.analytics.visual_tools import (
    chart_actual_vs_pred, chart_residuals, chart_residuals_qq
)
from src.ui.common import show_last_training_badge, load_active_cleaned_feature_master_from_session


def render():
    st.header("Report Generator")
    run_id = st.session_state.run_id
    # Model artifacts (from last training)
    res = st.session_state.get("last_model")
    if not res:
        st.info("Train a model first in **Analytics Tools → Analytical Tools – Model**.")
        return

    show_last_training_badge()

    # Loading the cleaned feature master for this run
    df, fm_clean = load_active_cleaned_feature_master_from_session()

    # Fetch all relevant data
    # Get y_true / y_pred
    y_true = (res or {}).get("y_true")
    y_pred = (res or {}).get("y_pred")

    # Trained models:
    trained_models = (res or {}).get("trained_models")

    # Performance blocks from what you already save
    perf_payload = {
        "per_model_metrics": (res or {}).get("base", {}).get("per_model_metrics"),
        "ensemble_avg_metrics": (res or {}).get("ensemble_avg", {}).get("metrics"),
        "ensemble_wgt_metrics": (res or {}).get("ensemble_wgt", {}).get("metrics"),
    }

    with st.container(border=True):
        st.markdown("Report Selection Options....")
        include_data = st.checkbox("Include dataset preview (first 20 rows)", value=True, disabled=df is None)
        include_plots = st.checkbox("Include diagnostic plots", value=True)
        include_perf = st.checkbox("Include model metrics & BP test", value=True)
        report_name = st.text_input("Report name (no spaces)", value="pricing_engine_report")

        if st.button("Generate Report", type="primary"):
            sections = []
            meta = {
                "model": ', '.join(trained_models),
                "rows": int(len(df)) if df is not None else None,
                "feature_master_path": fm_clean,
                "run_dir": st.session_state.get("last_model_run_dir"),
            }

            if include_data and df is not None:
                sections.append(Section("Dataset Preview", df.head(20).to_html(index=False)))

            if include_plots:
                imgs = [
                    ("Actual vs Predicted", chart_actual_vs_pred(y_true, y_pred)),
                    ("Residuals vs Predicted", chart_residuals(y_true, y_pred)),
                    ("Residuals Q–Q", chart_residuals_qq(y_true, y_pred))
                ]
                html = "".join([f"<h3>{t}</h3><img src='{src}'/>" for t, src in imgs])
                sections.append(Section("Diagnostics", html))

            if include_perf:
                perf_html = (
                    f"<h3>Performance</h3><pre>{json.dumps(perf_payload, indent=2)}</pre>"
                    # f"<h3>Breusch–Pagan</h3><pre>{json.dumps(res['bp'], indent=2)}</pre>"
                )
                sections.append(Section("Performance & Diagnostics", perf_html))

            html_report = build_html_report("Predictive Pricing Engine – Report", meta, sections)

            # Save report
            out_path = build_pdf_from_html(html_report, report_name, run_id)

            st.success(f"Saved report: {out_path}")
            with open(out_path, "rb") as f:
                st.download_button(
                    "Download PDF",
                    data=f,
                    file_name=f"{report_name}.pdf",
                    mime="application/pdf"
                )
