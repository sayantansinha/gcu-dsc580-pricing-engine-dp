from __future__ import annotations
import os
import streamlit as st

from src.config.env_loader import SETTINGS
from src.services.analytics.visual_tools import (
    chart_actual_vs_pred, chart_residuals, chart_residuals_qq
)


def _run_badge():
    run_dir = st.session_state.get("last_model_run_dir")
    if run_dir:
        rel = os.path.relpath(run_dir, SETTINGS.PROCESSED_DIR)
        st.caption(f"Using run: `{rel}`")


def render():
    st.header("Visual Tools")

    res = st.session_state.get("last_model")
    if not res:
        st.info("Train a model first in **Analytics Tools → Analytical Tools – Model**.")
        return

    _run_badge()

    st.subheader("Actual vs Predicted")
    st.markdown(f"<img src='{chart_actual_vs_pred(res['y_true'], res['y_pred'])}' width='100%'>",
                unsafe_allow_html=True)

    st.subheader("Residuals vs Predicted")
    st.markdown(f"<img src='{chart_residuals(res['y_true'], res['y_pred'])}' width='100%'>",
                unsafe_allow_html=True)

    st.subheader("Residuals Q–Q Plot")
    st.markdown(f"<img src='{chart_residuals_qq(res['y_true'], res['y_pred'])}' width='100%'>",
                unsafe_allow_html=True)
