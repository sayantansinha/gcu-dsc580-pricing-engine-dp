from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Dict, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.services.analytics.reporting import (
    generate_eda_report,
    generate_visualization_report,
    generate_model_analytics_report,
    generate_full_technical_report,
)
from src.utils.data_io_utils import load_report_for_download, list_reports_for_run
from src.utils.metric_utils import regression_metrics


def _get_active_run_id() -> Optional[str]:
    """Infer the currently active run id from session state."""
    return st.session_state.get("run_id")


def _get_active_dataframe() -> Optional[pd.DataFrame]:
    """Get the cleaned feature master for the active run from session state."""
    return st.session_state.get("cleaned_df")


def _get_model_results() -> Dict[str, Any]:
    """
    Retrieve model results from session state in a defensive way.

    Expected (flexible) structure:
      last_model = {
          "per_model_metrics": [...],
          "ensemble_avg": {"metrics": {...}},
          "ensemble_wgt": {"metrics": {...}},
          "bp": {...},
          "y_true": [...],
          "y_pred": [...],
      }
    """
    last = st.session_state.get("last_model") or {}
    if not isinstance(last, dict):
        return {}

    per_model = last.get("per_model_metrics")
    if per_model is None:
        base = last.get("base") or {}
        per_model = base.get("per_model_metrics")

    ensemble_avg = None
    if "ensemble_avg" in last:
        ensemble_avg = (last.get("ensemble_avg") or {}).get("metrics")
    if ensemble_avg is None:
        ensemble_avg = last.get("ensemble_avg_metrics")

    ensemble_wgt = None
    if "ensemble_wgt" in last:
        ensemble_wgt = (last.get("ensemble_wgt") or {}).get("metrics")
    if ensemble_wgt is None:
        ensemble_wgt = last.get("ensemble_weighted_metrics")

    results = {
        "per_model_metrics": per_model,
        "ensemble_avg_metrics": ensemble_avg,
        "ensemble_wgt_metrics": ensemble_wgt,
        "bp_results": last.get("bp"),
        "y_true": last.get("y_true"),
        "y_pred": last.get("y_pred"),
        "model": last.get("model"),
        "X_valid": last.get("X_valid"),
        "y_valid": last.get("y_valid"),
        "X_sample": last.get("X_sample"),
    }
    return results


def _download_button_for_report(ref: str, label: str) -> None:
    """Given a LOCAL path or s3:// URI, load bytes and expose a download button."""
    if not ref:
        st.error("No report reference returned.")
        return

    try:
        pdf_bytes = load_report_for_download(ref)
    except Exception as ex:
        st.error(f"Unable to load report for download: {ex}")
        return

    if ref.startswith("s3://"):
        filename = ref.rsplit("/", 1)[-1]
    else:
        filename = Path(ref).name

    st.download_button(label, data=pdf_bytes, file_name=filename, mime="application/pdf")


def _create_zip_from_reports(report_map: Dict[str, str]) -> bytes:
    """Zip up multiple reports (by name→ref) into an in-memory ZIP for download."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for logical_name, ref in report_map.items():
            if not ref:
                continue
            try:
                pdf_bytes = load_report_for_download(ref)
            except Exception:
                continue

            if ref.startswith("s3://"):
                filename = ref.rsplit("/", 1)[-1]
            else:
                filename = Path(ref).name

            # Avoid duplicate names inside ZIP
            if filename in zf.namelist():
                stem = Path(filename).stem
                suffix = Path(filename).suffix
                i = 1
                candidate = f"{stem}_{i}{suffix}"
                while candidate in zf.namelist():
                    i += 1
                    candidate = f"{stem}_{i}{suffix}"
                filename = candidate

            zf.writestr(filename, pdf_bytes)

    buf.seek(0)
    return buf.getvalue()


def _show_eda_tab(run_id: str, df: Optional[pd.DataFrame]) -> Optional[str]:
    st.subheader("Quantitative Data Exploration")

    if df is None or df.empty:
        st.info("No dataset is available for the active run. Train a pipeline first.")
        return None

    num_cols = list(df.select_dtypes(include="number").columns)
    if num_cols:
        col = st.selectbox("Numeric column", num_cols, key="eda_num_col")
        fig, ax = plt.subplots()
        df[col].dropna().hist(ax=ax, bins=30)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        st.pyplot(fig)

    cat_cols = list(df.select_dtypes(exclude="number").columns)
    if cat_cols:
        cat = st.selectbox("Categorical column", cat_cols, key="eda_cat_col")
        vc = df[cat].value_counts().head(20)
        fig, ax = plt.subplots()
        vc.plot(kind="bar", ax=ax)
        ax.set_title(f"Top categories in {cat}")
        ax.set_xlabel(cat)
        ax.set_ylabel("Count")
        fig.tight_layout()
        st.pyplot(fig)

    if st.button("Generate PDF (EDA report)", key="btn_gen_eda"):
        ref = generate_eda_report(run_id=run_id, df=df)
        st.success(f"EDA report generated: {ref}")
        _download_button_for_report(ref, "Download EDA report")

        # mark reporting as completed for pipeline flow
        st.session_state["report_generated"] = True

        return ref

    return None


def _show_visual_tab(run_id: str, df: Optional[pd.DataFrame]) -> Optional[str]:
    st.subheader("Visual Exploration")

    if df is None or df.empty:
        st.info("No dataset is available for the active run. Train a pipeline first.")
        return None

    num_cols = list(df.select_dtypes(include="number").columns)
    if len(num_cols) >= 2:
        x_col = st.selectbox("X-axis feature", num_cols, key="vis_x_col")
        y_col = st.selectbox(
            "Y-axis feature",
            num_cols,
            index=1 if len(num_cols) > 1 else 0,
            key="vis_y_col",
        )
        fig, ax = plt.subplots()
        ax.scatter(df[x_col], df[y_col], alpha=0.35)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} vs {x_col}")
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Need at least two numeric columns for scatter plot.")

    if st.button("Generate PDF (Visual Exploration report)", key="btn_gen_visual"):
        ref = generate_visualization_report(run_id=run_id, df=df)
        st.success(f"Visual Exploration report generated: {ref}")
        _download_button_for_report(ref, "Download Visual Exploration report")

        # mark reporting as completed for pipeline flow
        st.session_state["report_generated"] = True

        return ref

    return None


def _show_model_tab(run_id: str, model_results: Dict[str, Any]) -> Optional[str]:
    st.subheader("Model Performance & Analytics")

    per_model = model_results.get("per_model_metrics")
    ensemble_avg = model_results.get("ensemble_avg_metrics")
    ensemble_wgt = model_results.get("ensemble_wgt_metrics")
    bp_results = model_results.get("bp_results")
    y_true = model_results.get("y_true")
    y_pred = model_results.get("y_pred")

    # Per-model metrics
    if per_model:
        st.markdown("**Per-model metrics**")
        st.dataframe(pd.DataFrame(per_model))

    # Fallback: compute quick metrics if only y_true / y_pred exist
    if not per_model and y_true is not None and y_pred is not None:
        metrics = regression_metrics(y_true, y_pred)
        st.markdown("**Regression metrics (computed)**")
        st.json(metrics)

    if ensemble_avg:
        st.markdown("**Ensemble (average) metrics**")
        st.json(ensemble_avg)

    if ensemble_wgt:
        st.markdown("**Ensemble (weighted) metrics**")
        st.json(ensemble_wgt)

    if bp_results:
        st.markdown("**Breusch–Pagan test**")
        st.json(bp_results)

    # Quick in-app residual diagnostics (separate from PDF charts)
    if y_true is not None and y_pred is not None:
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        residuals = y_true_arr - y_pred_arr

        fig, ax = plt.subplots()
        ax.scatter(y_pred_arr, residuals, alpha=0.35)
        ax.axhline(0.0, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual")
        ax.set_title("Residuals vs Predicted")
        fig.tight_layout()
        st.pyplot(fig)

        fig, ax = plt.subplots()
        ax.hist(residuals, bins=30)
        ax.set_xlabel("Residual")
        ax.set_ylabel("Count")
        ax.set_title("Residual distribution")
        fig.tight_layout()
        st.pyplot(fig)

    if st.button("Generate PDF (Model Analytics report)", key="btn_gen_model"):
        ref = generate_model_analytics_report(
            run_id=run_id,
            per_model_metrics=per_model,
            ensemble_avg_metrics=ensemble_avg,
            ensemble_wgt_metrics=ensemble_wgt,
            bp_results=bp_results,
            y_true=y_true,
            y_pred=y_pred,
            model=model_results.get("model"),
            x_valid=model_results.get("X_valid"),
            y_valid=model_results.get("y_valid"),
            x_sample=model_results.get("X_sample"),
        )
        st.success(f"Model Analytics report generated: {ref}")
        _download_button_for_report(ref, "Download Model Analytics report")

        # mark reporting as completed for pipeline flow
        st.session_state["report_generated"] = True

        return ref

    return None


def render_report():
    st.header("Reporting")

    st.session_state.setdefault("report_generated", False)

    run_id = _get_active_run_id()
    if not run_id:
        st.info("No active run selected. Please select or create a run in the pipeline first.")
        return

    df = _get_active_dataframe()
    model_results = _get_model_results()

    # ----------------------------------------
    # Existing reports (NO expander here)
    # ----------------------------------------
    existing_refs = list_reports_for_run(run_id)
    if existing_refs:
        with st.container(border=True):
            st.markdown("#### Existing Reports")
            st.caption("The following reports were found for this run:")

            for ref in existing_refs:
                file_name = Path(ref).name
                cols = st.columns([3, 1])
                with cols[0]:
                    st.write(f"• `{file_name}`")
                with cols[1]:
                    try:
                        pdf_bytes = load_report_for_download(ref)
                        st.download_button(
                            "Download",
                            data=pdf_bytes,
                            file_name=file_name,
                            mime="application/pdf",
                            key=f"btn_download_existing_{file_name}",
                        )
                    except Exception as ex:
                        st.warning(f"Could not load {file_name}: {ex}")

    # ----------------------------------------
    # Tabs for generating new reports
    # ----------------------------------------
    tab_eda, tab_vis, tab_model, tab_all = st.tabs(
        ["Quantitative EDA", "Visual Exploration", "Model Analytics", "All reports"]
    )

    report_refs: Dict[str, str] = {}

    with tab_eda:
        ref = _show_eda_tab(run_id, df)
        if ref:
            report_refs["eda"] = ref

    with tab_vis:
        ref = _show_visual_tab(run_id, df)
        if ref:
            report_refs["visual"] = ref

    with tab_model:
        ref = _show_model_tab(run_id, model_results)
        if ref:
            report_refs["model"] = ref

    with tab_all:
        st.subheader("Download all reports as a ZIP")
        st.markdown(
            "Use this section to generate a single technical summary PDF and/or "
            "download all individual reports bundled into a ZIP archive."
        )

        if st.button("Generate PDF (Technical Summary report)", key="btn_gen_full"):
            ref_full = generate_full_technical_report(
                run_id=run_id,
                df=df,
                per_model_metrics=model_results.get("per_model_metrics"),
                ensemble_avg_metrics=model_results.get("ensemble_avg_metrics"),
                ensemble_wgt_metrics=model_results.get("ensemble_wgt_metrics"),
                bp_results=model_results.get("bp_results"),
                y_true=model_results.get("y_true"),
                y_pred=model_results.get("y_pred"),
            )
            st.success(f"Technical Summary report generated: {ref_full}")
            _download_button_for_report(ref_full, "Download Technical Summary report")
            report_refs["full"] = ref_full
            st.session_state["report_generated"] = True

        if st.button("Download all generated reports as ZIP", key="btn_zip_all"):
            if not report_refs and not existing_refs:
                st.warning(
                    "No reports have been generated yet in this session and none "
                    "were found in storage."
                )
            else:
                # merge in any existing refs so ZIP can include both
                all_refs = dict(report_refs)
                for ref in existing_refs:
                    all_refs.setdefault(Path(ref).name, ref)

                zip_bytes = _create_zip_from_reports(all_refs)
                st.download_button(
                    "Download all reports (.zip)",
                    data=zip_bytes,
                    file_name=f"{run_id}_reports.zip",
                    mime="application/zip",
                )
