import os

import pandas as pd
import streamlit as st

from src.services.source_data.analytics.eda import eda_summary
from src.services.source_data.analytics.visualization import plot_hist, plot_box, plot_bar, plot_scatter, \
    plot_datetime_counts, \
    plot_time_of_day_hist, plot_month_box
from src.ui.common import end_tab_scroll, begin_tab_scroll, section_panel, load_active_feature_master_from_session
from src.utils.data_io_utils import load_figure
from src.utils.log_utils import get_logger

LOGGER = get_logger("ui_exploration")


def _display_figure(path_or_uri: str, *, context: str) -> None:
    """
    Display a figure from LOCAL or S3 and handle errors gracefully.
    """
    caption = os.path.basename(str(path_or_uri))
    try:
        img = load_figure(path_or_uri)
        st.image(img, caption=caption)
    except Exception as e:
        LOGGER.exception(
            "Error displaying visualization (%s): %s", context, path_or_uri, exc_info=e
        )
        st.error(
            f"An error occurred while rendering the {context} visualization. "
            "Please check the application logs for details."
        )


def _viz_numeric_cat(df: pd.DataFrame, run_id: str) -> None:
    try:
        st.subheader("Numeric and Categorical Visualizations")

        num_cols = list(df.select_dtypes(include="number").columns)
        cat_cols = [c for c in df.columns if c not in num_cols]

        if not num_cols:
            st.info("No numeric columns available for visualization.")
            return

        col_left, col_right = st.columns(2)

        # Left column: hist / box
        with col_left:
            nc = st.selectbox("Numeric column", num_cols, key="viz_num_col")
            bins = st.slider("Histogram bins", 10, 100, 30, key="viz_num_bins")

            if st.button("Plot histogram", key="btn_hist"):
                p = plot_hist(df, nc, run_id, bins=bins)
                _display_figure(p, context="histogram")

            if st.button("Plot boxplot", key="btn_box"):
                p = plot_box(df, nc, run_id)
                _display_figure(p, context="boxplot")

        # Right column: bar / scatter
        with col_right:
            if cat_cols:
                cc = st.selectbox("Categorical column for bar chart", cat_cols, key="viz_cat_col")
                if st.button("Plot bar chart", key="btn_bar"):
                    p = plot_bar(df, cc, run_id)
                    _display_figure(p, context="bar chart")
            else:
                st.caption("No non-numeric columns available for bar charts.")

            if len(num_cols) >= 2:
                x_col = st.selectbox("Scatter X (numeric)", num_cols, key="viz_scatter_x")
                y_col = st.selectbox("Scatter Y (numeric)", num_cols, key="viz_scatter_y")
                if st.button("Plot scatter", key="btn_scatter"):
                    p = plot_scatter(df, x_col, y_col, run_id)
                    _display_figure(p, context="scatter")
            else:
                st.caption("Need at least two numeric columns for scatter plot.")
    except Exception as e:
        LOGGER.exception(
            "Error in Visualization – Numeric/Categorical tab for run_id=%s", run_id, exc_info=e
        )
        st.error(
            "An unexpected error occurred while generating numeric/categorical visualizations. "
            "Please check the logs for details."
        )


def _viz_datetime(df: pd.DataFrame, run_id: str) -> None:
    try:
        st.subheader("Datetime Visualizations")

        import pandas as pd

        dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
        if not dt_cols:
            st.info("No datetime columns available for visualization.")
            return

        dc = st.selectbox("Datetime column", dt_cols, key="viz_dt_col")

        if st.button("Plot daily counts", key="btn_dt_counts"):
            p = plot_datetime_counts(df, dc, run_id)
            _display_figure(p, context="daily counts")

        if st.button("Plot hour-of-day histogram", key="btn_dt_hour"):
            p = plot_time_of_day_hist(df, dc, run_id)
            _display_figure(p, context="hour-of-day histogram")

        num_cols = list(df.select_dtypes(include="number").columns)
        if not num_cols:
            st.caption("No numeric columns available for month boxplot.")
            return

        vc = st.selectbox("Numeric value for month boxplot", num_cols, key="viz_dt_month_val")
        if st.button("Plot month boxplot", key="btn_dt_month_box"):
            p = plot_month_box(df, dc, vc, run_id)
            _display_figure(p, context="month boxplot")
    except Exception as e:
        LOGGER.exception(
            "Error in Visualization – Datetime tab for run_id=%s", run_id, exc_info=e
        )
        st.error(
            "An unexpected error occurred while generating datetime visualizations. "
            "Please check the logs for details."
        )


def render_exploration_section():
    run_id = st.session_state.run_id
    LOGGER.info(f"Rendering Exploration (EDA) panel.....run_id: {run_id}")
    st.header("Exploration (EDA)")
    df, label = load_active_feature_master_from_session()
    if df is None:
        st.warning("No feature master found. Build it in Data Staging.")
        return
    st.caption(f"Using: {label} — shape={df.shape}")
    df = st.session_state.df

    with section_panel("Exploration (EDA)", expanded=True):
        tabs = st.tabs(["Describe", "Numeric Correlations", "Visualization"])

        with tabs[0]:
            begin_tab_scroll()
            profile = eda_summary(df)
            st.json(profile.get("describe", {}))
            end_tab_scroll()

        with tabs[1]:
            begin_tab_scroll()
            profile = eda_summary(df)
            st.json(profile.get("corr_numeric", {}))
            end_tab_scroll()

        with tabs[2]:
            subtabs = st.tabs(["Numeric/Categorical", "Datetime"])
            with subtabs[0]:
                begin_tab_scroll()
                _viz_numeric_cat(df, run_id)
                end_tab_scroll()
            with subtabs[1]:
                begin_tab_scroll()
                _viz_datetime(df, run_id)
                end_tab_scroll()

    LOGGER.info(f"Exploration (EDA) panel rendered.....run_id: {run_id}")
