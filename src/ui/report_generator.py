from __future__ import annotations
import os, json
import pandas as pd
import streamlit as st

from src.config.env_loader import SETTINGS
from src.services.analytics.reporting import Section, build_html_report, save_html_report
from src.services.analytics.visual_tools import (
    chart_actual_vs_pred, chart_residuals, chart_residuals_qq
)


def _load_feature_master_for_current_run() -> tuple[pd.DataFrame | None, str | None]:
    run_dir = st.session_state.get("last_model_run_dir")
    if not run_dir:
        return None, None
    path = os.path.join(run_dir, "feature_master.parquet")
    if not os.path.exists(path):
        return None, None
    return pd.read_parquet(path), path


def _run_badge():
    run_dir = st.session_state.get("last_model_run_dir")
    if run_dir:
        rel = os.path.relpath(run_dir, SETTINGS.PROCESSED_DIR)
        st.caption(f"Using run: `{rel}`")


def render():
    st.header("Report Generator")

    # Model artifacts (from last training)
    res = st.session_state.get("last_model")
    if not res:
        st.info("Train a model first in **Analytics Tools → Analytical Tools – Model**.")
        return

    _run_badge()

    # Try loading the feature master for this run (optional data preview)
    df, fm_path = _load_feature_master_for_current_run()

    with st.expander("Report Selection Options....", expanded=True):
        include_data = st.checkbox("Include dataset preview (first 20 rows)", value=True, disabled=df is None)
        include_plots = st.checkbox("Include diagnostic plots", value=True)
        include_perf = st.checkbox("Include model metrics & BP test", value=True)
        report_name = st.text_input("Report name (no spaces)", value="pricing_engine_report")

        if st.button("Generate Report", type="primary"):
            sections = []
            meta = {
                "model": res["model_name"],
                "rows": int(len(df)) if df is not None else None,
                "feature_master_path": fm_path,
                "run_dir": st.session_state.get("last_model_run_dir"),
            }

            if include_data and df is not None:
                sections.append(Section("Dataset Preview", df.head(20).to_html(index=False)))

            if include_plots:
                imgs = [
                    ("Actual vs Predicted", chart_actual_vs_pred(res["y_true"], res["y_pred"])),
                    ("Residuals vs Predicted", chart_residuals(res["y_true"], res["y_pred"])),
                    ("Residuals Q–Q", chart_residuals_qq(res["y_true"], res["y_pred"]))
                ]
                html = "".join([f"<h3>{t}</h3><img src='{src}'/>" for t, src in imgs])
                sections.append(Section("Diagnostics", html))

            if include_perf:
                perf_html = (
                    f"<h3>Performance</h3><pre>{json.dumps(res['metrics'], indent=2)}</pre>"
                    f"<h3>Breusch–Pagan</h3><pre>{json.dumps(res['bp'], indent=2)}</pre>"
                )
                sections.append(Section("Performance & Diagnostics", perf_html))

            html = build_html_report("Predictive Pricing Engine – Local Report", meta, sections)
            out_path = save_html_report(html, report_name)

            st.success(f"Saved report: {out_path}")
            with open(out_path, "rb") as f:
                st.download_button("Download report", data=f, file_name=f"{report_name}.html")
