import streamlit as st

from src.services.data_io import save_processed, save_profile
from src.services.source_data.preprocessing.cleaning import impute, iqr_filter, winsorize, encode_one_hot, \
    encode_ordinal, scale, \
    deduplicate
from src.ui.common import end_tab_scroll, begin_tab_scroll, section_panel
from src.utils.log_utils import get_logger

LOGGER = get_logger("ui_cleaning")


def _tab_impute(df):
    with st.form("impute_form"):
        strat = st.selectbox("Strategy", ["median", "mean"])
        cols_sel = st.multiselect("Columns (blank = all)", df.columns.tolist())
        submitted = st.form_submit_button("Apply imputation")
    if submitted:
        st.session_state.df = impute(st.session_state.df, strategy=strat, columns=(cols_sel or None))
        st.session_state.steps.append({"impute": {"strategy": strat, "columns": cols_sel or "ALL"}})
        st.success("Imputation applied.")


def _tab_outliers(df):
    num_cols = df.select_dtypes(include="number").columns.tolist()
    with st.form("outlier_form"):
        method = st.selectbox("Method", ["IQR filter", "Winsorize"])
        col = st.selectbox("Numeric column", num_cols)
        if method == "IQR filter":
            k = st.slider("IQR factor (k)", 0.5, 3.0, 1.5, 0.1)
        else:
            lo = st.slider("Lower quantile", 0.0, 0.2, 0.01, 0.01)
            hi = st.slider("Upper quantile", 0.8, 1.0, 0.99, 0.01)
        submitted = st.form_submit_button("Apply outlier handling")
    if submitted:
        if method == "IQR filter":
            st.session_state.df = iqr_filter(st.session_state.df, col, k=k)
            st.session_state.steps.append({"iqr_filter": {"column": col, "k": k}})
        else:
            st.session_state.df = winsorize(st.session_state.df, col, lower_q=lo, upper_q=hi)
            st.session_state.steps.append({"winsorize": {"column": col, "lower_q": lo, "upper_q": hi}})
        st.success(f"Outlier processing applied: {method} on {col}")


def _tab_encode(df):
    cats = [c for c in df.columns if df[c].dtype == "object"]
    st.caption("One-hot encoding")
    ohe_cols = st.multiselect("Columns", cats, key="ohe_cols")
    if st.button("Apply one-hot"):
        st.session_state.df = encode_one_hot(st.session_state.df, ohe_cols)
        st.session_state.steps.append({"one_hot": {"columns": ohe_cols}})
        st.success("One-hot encoding applied.")

    st.markdown("---")
    st.caption("Ordinal encoding")
    ord_col = st.text_input("Ordinal column", key="ord_col")
    ord_order = st.text_input("Order CSV (e.g., AVOD,SVOD,TVOD,PAYTV,BC)", key="ord_order")
    if st.button("Apply ordinal"):
        if ord_col and ord_order:
            mapping = {ord_col: [x.strip() for x in ord_order.split(",")]}
            st.session_state.df = encode_ordinal(st.session_state.df, mapping)
            st.session_state.steps.append({"ordinal": {"mapping": mapping}})
            st.success("Ordinal encoding applied.")
        else:
            st.warning("Provide both column and order CSV.")


def _tab_scale_dedup(df):
    num_cols = df.select_dtypes(include="number").columns.tolist()
    st.caption("Scaling")
    sc_cols = st.multiselect("Scale columns", num_cols, key="scale_cols")
    method = st.selectbox("Method", ["standard", "minmax"], key="scale_method")
    if st.button("Apply scaling"):
        if sc_cols:
            st.session_state.df = scale(st.session_state.df, sc_cols, method=method)
            st.session_state.steps.append({"scale": {"columns": sc_cols, "method": method}})
            st.success("Scaling applied.")
        else:
            st.info("Select at least one numeric column.")

    st.markdown("---")
    st.caption("De-duplication")
    dedup_cols = st.multiselect("Deduplicate by columns", df.columns.tolist(), key="dedup_cols")
    if st.button("Apply de-dup"):
        if dedup_cols:
            st.session_state.df = deduplicate(st.session_state.df, dedup_cols)
            st.session_state.steps.append({"deduplicate": {"subset": dedup_cols}})
            st.success("De-duplication applied.")
        else:
            st.info("Select at least one column.")


def _save_cleaned_dataset(df):
    st.caption("Save the cleaned dataset and transformation manifest.")
    base_out = st.text_input("Base name", value="cleaned", key="save_base")
    if st.button("Save Cleaned Dataset", type="primary"):
        out_path = save_processed(df, st.session_state.run_id, base_name=base_out)
        run_id = st.session_state.run_id
        mf_path = save_profile(
            {"run_id": run_id, "steps": st.session_state.steps},
            f"manifest_{run_id}"
        )
        st.success(f"Saved cleaned dataset: {out_path}")
        st.success(f"Wrote transformation manifest: {mf_path}")


def render_cleaning_section():
    LOGGER.info("Rendering Cleaning panel....")
    df = st.session_state.df

    with section_panel("Cleaning & Preprocessing", expanded=True):
        tabs = st.tabs(["Impute", "Outliers", "Encode", "Scale & Deduplicate", "Save"])
        with tabs[0]:
            begin_tab_scroll()
            _tab_impute(df)
            end_tab_scroll()
        with tabs[1]:
            begin_tab_scroll()
            _tab_outliers(df)
            end_tab_scroll()
        with tabs[2]:
            begin_tab_scroll()
            _tab_encode(df)
            end_tab_scroll()
        with tabs[3]:
            begin_tab_scroll()
            _tab_scale_dedup(df)
            end_tab_scroll()
        with tabs[4]:
            begin_tab_scroll()
            _save_cleaned_dataset(st.session_state.df)
            end_tab_scroll()

    LOGGER.info("Cleaning panel rendered")
