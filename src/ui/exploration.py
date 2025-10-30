import os
import streamlit as st

from src.analytics.eda import eda_summary
from src.analytics.visualization import plot_hist, plot_box, plot_bar, plot_scatter, plot_datetime_counts, \
    plot_time_of_day_hist, plot_month_box
from src.ui.common import end_tab_scroll, begin_tab_scroll, section_panel
from src.utils.log_utils import get_logger

LOGGER = get_logger("ui_exploration")


def _viz_numeric_cat(df, run_id):
    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in all_cols if c not in num_cols]

    chart = st.selectbox("Chart type", ["Histogram", "Boxplot", "Bar", "Scatter"], key="viz_chart")
    if chart == "Histogram":
        if not num_cols:
            st.info("No numeric columns.")
            return
        c = st.selectbox("Column", num_cols, key="hist_col")
        bins = st.slider("Bins", 5, 100, 30, key="hist_bins")
        if st.button("Plot histogram"):
            p = plot_hist(df, c, run_id, bins=bins)
            st.image(p, caption=os.path.basename(p))
    elif chart == "Boxplot":
        if not num_cols:
            st.info("No numeric columns.")
            return
        c = st.selectbox("Column", num_cols, key="box_col")
        if st.button("Plot boxplot"):
            p = plot_box(df, c, run_id)
            st.image(p, caption=os.path.basename(p))
    elif chart == "Bar":
        cat = st.selectbox("Category", cat_cols or all_cols, key="bar_cat")
        val = st.selectbox("Aggregate (optional numeric)", [None] + num_cols, key="bar_val")
        if st.button("Plot bar"):
            p = plot_bar(df, cat, val, run_id)
            st.image(p, caption=os.path.basename(p))
    else:
        if len(num_cols) < 2:
            st.info("Need at least two numeric columns.")
            return
        x = st.selectbox("X", num_cols, key="scatter_x")
        y = st.selectbox("Y", num_cols, key="scatter_y")
        if st.button("Plot scatter"):
            p = plot_scatter(df, x, y, run_id)
            st.image(p, caption=os.path.basename(p))


def _viz_datetime(df, run_id):
    dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
    if not dt_cols:
        st.info("No datetime columns detected.")
        return

    choice = st.selectbox("Datetime chart", ["Counts over time", "Time-of-day", "Value by month"], key="dt_choice")
    if choice == "Counts over time":
        dc = st.selectbox("Datetime column", dt_cols, key="dt_counts_col")
        freq = st.selectbox("Frequency", ["D", "W", "M", "Q", "Y"], index=2, key="dt_counts_freq")
        if st.button("Plot counts over time"):
            p = plot_datetime_counts(df, dc, run_id, freq=freq)
            st.image(p, caption=os.path.basename(p))
    elif choice == "Time-of-day":
        dc = st.selectbox("Datetime column", dt_cols, key="dt_hour_col")
        if st.button("Plot hour histogram"):
            p = plot_time_of_day_hist(df, dc, run_id)
            st.image(p, caption=os.path.basename(p))
    else:
        dc = st.selectbox("Datetime column", dt_cols, key="dt_month_col")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            st.info("Need a numeric value column for boxplot.")
            return
        vc = st.selectbox("Numeric value", num_cols, key="dt_month_val")
        if st.button("Plot month boxplot"):
            p = plot_month_box(df, dc, vc, run_id)
            st.image(p, caption=os.path.basename(p))


def render_exploration_section():
    df = st.session_state.df
    run_id = st.session_state.run_id
    LOGGER.info(f"Rendering Exploration (EDA) panel.....run_id: {run_id}")
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
