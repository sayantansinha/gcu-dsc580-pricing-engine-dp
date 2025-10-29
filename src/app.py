import io
import os

import pandas as pd
import streamlit as st

from config import SETTINGS
from src.analytics.eda import eda_summary, save_profile
from src.analytics.visualization import plot_hist, plot_box, plot_bar, plot_scatter, plot_datetime_counts, \
    plot_time_of_day_hist, plot_month_box
from src.preprocessing.cleaning import impute, iqr_filter, winsorize, encode_one_hot, encode_ordinal, deduplicate, \
    save_processed, scale, write_manifest
from src.validator.schema_validator import validate_df
from utils.io_utils import load_local_file, save_uploaded_file, read_from_url

st.set_page_config(page_title="Pricing Engine (Local)", layout="wide")
st.title("Predictive Pricing Engine – Local Prototype")

# --- Session state for df and run metadata ---
if "df" not in st.session_state:
    st.session_state.df = None
if "run_id" not in st.session_state:
    st.session_state.run_id = None
if "raw_path" not in st.session_state:
    st.session_state.raw_path = None
if "steps" not in st.session_state:
    st.session_state.steps = []


# ---------------- Sidebar: Ingest ----------------
with st.sidebar:
    st.header("Ingest")
    ingest_opt = st.radio("Select source", ["Upload File", "From URL", "Load Local Path"])

    if ingest_opt == "Upload File":
        up = st.file_uploader("CSV or Parquet", type=["csv", "parquet", "pq"])
        base_name = st.text_input("Base name", value="dataset")
        if up and st.button("Ingest file"):
            if up.name.lower().endswith((".parquet", ".pq")):
                st.session_state.df = pd.read_parquet(io.BytesIO(up.read()))
            else:
                st.session_state.df = pd.read_csv(up)
            st.session_state.run_id, st.session_state.raw_path = save_uploaded_file(st.session_state.df, base_name)
            st.success(f"Ingested: {st.session_state.raw_path}")
            st.session_state.steps = []

    elif ingest_opt == "From URL":
        url = st.text_input("URL to CSV/Parquet")
        base_name = st.text_input("Base name", value="remote")
        if st.button("Ingest URL"):
            if not url:
                st.error("Provide a URL")
            else:
                df, rid, rpath = read_from_url(url, base_name)
                st.session_state.df = df
                st.session_state.run_id = rid
                st.session_state.raw_path = rpath
                st.success(f"Ingested: {st.session_state.raw_path}")
                st.session_state.steps = []

    else:  # Load Local Path
        local_path = st.text_input("Local file path")
        base_name = st.text_input("Base name", value="local")
        if st.button("Load local"):
            st.session_state.df = load_local_file(local_path)
            from utils.io_utils import _new_run_id  # internal helper for run IDs
            st.session_state.run_id = _new_run_id("file")
            st.session_state.raw_path = os.path.join(SETTINGS.RAW_DIR, f"{base_name}_{st.session_state.run_id}.parquet")
            st.session_state.df.to_parquet(st.session_state.raw_path, index=False)
            st.success(f"Loaded and saved copy to: {st.session_state.raw_path}")
            st.session_state.steps = []


# ---------------- Main: if we have data ----------------
df = st.session_state.df
run_id = st.session_state.run_id
raw_path = st.session_state.raw_path

if df is None:
    st.info("Ingest a dataset from the sidebar to begin.")
    st.stop()


# ---------------- Schema & Validation ----------------
st.subheader("Schema & Validation")
try:
    df = validate_df(df)  # ensure any coercions happen on the working df
    st.success("Validation passed")
except Exception as e:
    st.warning(f"Validation warnings/errors: {e}")

# keep session df in sync
st.session_state.df = df


# ---------------- EDA Summary ----------------
st.subheader("Exploration (EDA)")
profile = eda_summary(df)

col1, col2, col3 = st.columns(3)
with col1:
    st.write("**Shape**")
    st.json(profile.get("shape", {}))
with col2:
    st.write("**Dtypes**")
    st.json({"dtypes": profile.get("dtypes", {})})
with col3:
    st.write("**Missing**")
    st.json({"missing": profile.get("missing", {})})

with st.expander("Describe (pandas 2.x)"):
    st.json(profile.get("describe", {}))

with st.expander("Numeric Correlations"):
    st.json(profile.get("corr_numeric", {}))

with st.expander("Sample (head)"):
    st.json(profile.get("sample_head", []))

with st.expander("Datetime Summary"):
    st.json(profile.get("datetime_summary", {}))

with st.expander("Datetime Trends (Monthly)"):
    st.json(profile.get("datetime_trends_monthly", {}))

if st.button("Save EDA Profile"):
    profile_path = save_profile(run_id, profile)
    st.success(f"Profile saved: {profile_path}")


# ---------------- Visualizations ----------------
st.subheader("Visualization")

all_cols = df.columns.tolist()
num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in all_cols if c not in num_cols]
dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()

# Classic numeric/categorical charts
with st.expander("Histogram"):
    if not num_cols:
        st.info("No numeric columns detected.")
    else:
        c = st.selectbox("Numeric column", num_cols, key="hist_col")
        bins = st.slider("Bins", 5, 100, 30, key="hist_bins")
        if st.button("Plot histogram"):
            p = plot_hist(df, c, run_id, bins=bins)
            st.image(p, caption=os.path.basename(p))

with st.expander("Boxplot"):
    if not num_cols:
        st.info("No numeric columns detected.")
    else:
        c = st.selectbox("Numeric column", num_cols, key="box_col")
        if st.button("Plot boxplot"):
            p = plot_box(df, c, run_id)
            st.image(p, caption=os.path.basename(p))

with st.expander("Bar"):
    if not cat_cols and not all_cols:
        st.info("No categorical columns detected.")
    else:
        cat = st.selectbox("Category", cat_cols or all_cols, key="bar_cat")
        val_opt = st.selectbox("Aggregate (optional numeric)", [None] + num_cols, key="bar_val")
        if st.button("Plot bar"):
            p = plot_bar(df, cat, val_opt, run_id)
            st.image(p, caption=os.path.basename(p))

with st.expander("Scatter"):
    if len(num_cols) < 2:
        st.info("Need at least two numeric columns.")
    else:
        x = st.selectbox("X", num_cols, key="scatter_x")
        y = st.selectbox("Y", num_cols, key="scatter_y")
        if st.button("Plot scatter"):
            p = plot_scatter(df, x, y, run_id)
            st.image(p, caption=os.path.basename(p))

# Datetime visualizations
st.subheader("Time Series Analysis")

with st.expander("Counts over time"):
    if not dt_cols:
        st.info("No datetime columns detected.")
    else:
        dtc = st.selectbox("Datetime column", dt_cols, key="dt_counts_col")
        freq = st.selectbox("Frequency", ["D", "W", "M", "Q", "Y"], index=2, key="dt_counts_freq")
        if st.button("Plot counts over time"):
            p = plot_datetime_counts(df, dtc, run_id, freq=freq)
            st.image(p, caption=os.path.basename(p))

with st.expander("Time-of-day distribution"):
    if not dt_cols:
        st.info("No datetime columns detected.")
    else:
        dtc2 = st.selectbox("Datetime column", dt_cols, key="dt_hour_col")
        if st.button("Plot hour histogram"):
            p = plot_time_of_day_hist(df, dtc2, run_id)
            st.image(p, caption=os.path.basename(p))

with st.expander("Value by month (boxplot)"):
    if not dt_cols or not num_cols:
        st.info("Need a datetime column and a numeric value column.")
    else:
        dtc3 = st.selectbox("Datetime column", dt_cols, key="dt_month_col")
        valc = st.selectbox("Numeric value", num_cols, key="dt_month_val")
        if st.button("Plot month boxplot"):
            p = plot_month_box(df, dtc3, valc, run_id)
            st.image(p, caption=os.path.basename(p))


# ---------------- Cleaning & Preprocessing ----------------
st.subheader("Cleaning & Preprocessing")

# Reset steps per ingest; steps are appended as you apply transforms
if "steps" not in st.session_state:
    st.session_state.steps = []

with st.expander("Impute"):
    strat = st.selectbox("Strategy", ["median", "mean"], key="imp_strategy")
    cols_sel = st.multiselect("Columns (blank = all)", df.columns.tolist(), key="imp_cols")
    if st.button("Apply impute"):
        st.session_state.df = impute(st.session_state.df, strategy=strat, columns=(cols_sel or None))
        st.session_state.steps.append({"impute": {"strategy": strat, "columns": cols_sel or "ALL"}})
        st.success("Imputed.")

with st.expander("Outliers"):
    if not num_cols:
        st.info("No numeric columns detected.")
    else:
        col_iqr = st.selectbox("Column (IQR filter)", num_cols, key="iqr_col")
        k = st.slider("IQR factor", 0.5, 3.0, 1.5, 0.1, key="iqr_k")
        if st.button("Apply IQR filter"):
            st.session_state.df = iqr_filter(st.session_state.df, col_iqr, k=k)
            st.session_state.steps.append({"iqr_filter": {"column": col_iqr, "k": k}})
            st.success("Filtered.")

        col_w = st.selectbox("Column (Winsorize)", num_cols, key="win_col")
        lo = st.slider("Lower quantile", 0.0, 0.2, 0.01, 0.01, key="win_lo")
        hi = st.slider("Upper quantile", 0.8, 1.0, 0.99, 0.01, key="win_hi")
        if st.button("Apply winsorize"):
            st.session_state.df = winsorize(st.session_state.df, col_w, lower_q=lo, upper_q=hi)
            st.session_state.steps.append({"winsorize": {"column": col_w, "lower_q": lo, "upper_q": hi}})
            st.success("Winsorized.")

with st.expander("Encode"):
    cats = [c for c in st.session_state.df.columns if st.session_state.df[c].dtype == "object"]
    cats_sel = st.multiselect("One-hot encode columns", cats, key="oh_cols")
    if st.button("Apply one-hot"):
        st.session_state.df = encode_one_hot(st.session_state.df, cats_sel)
        st.session_state.steps.append({"one_hot": {"columns": cats_sel}})
        st.success("Encoded (one-hot).")

    st.caption("Ordinal encode (provide column & order CSV, e.g., AVOD,SVOD,TVOD,PAYTV,BC).")
    ord_col = st.text_input("Ordinal column", key="ord_col")
    ord_order = st.text_input("Order CSV", key="ord_order")
    if st.button("Apply ordinal"):
        if ord_col and ord_order:
            mapping = {ord_col: [x.strip() for x in ord_order.split(",")]}
            st.session_state.df = encode_ordinal(st.session_state.df, mapping)
            st.session_state.steps.append({"ordinal": {"mapping": mapping}})
            st.success("Encoded (ordinal).")
        else:
            st.warning("Provide both column and order CSV.")

with st.expander("Scale & Deduplicate"):
    num_cols_after = st.session_state.df.select_dtypes(include="number").columns.tolist()
    sc_cols = st.multiselect("Scale columns", num_cols_after, key="scale_cols")
    method = st.selectbox("Method", ["standard", "minmax"], key="scale_method")
    if st.button("Apply scaling"):
        if sc_cols:
            st.session_state.df = scale(st.session_state.df, sc_cols, method=method)
            st.session_state.steps.append({"scale": {"columns": sc_cols, "method": method}})
            st.success("Scaled.")
        else:
            st.info("Select at least one numeric column to scale.")

    dedup_cols = st.multiselect("Deduplicate by columns", st.session_state.df.columns.tolist(), key="dedup_cols")
    if st.button("Deduplicate"):
        if dedup_cols:
            st.session_state.df = deduplicate(st.session_state.df, dedup_cols)
            st.session_state.steps.append({"deduplicate": {"subset": dedup_cols}})
            st.success("Deduplicated.")
        else:
            st.info("Select at least one column for de-duplication.")

# reflect working df
df = st.session_state.df

# ---------------- Save Cleaned Dataset ----------------
st.subheader("Save Cleaned Dataset")
base_out = st.text_input("Base name", value="cleaned", key="save_base")
if st.button("Save"):
    out_path = save_processed(df, run_id, base_name=base_out)
    mf_path = write_manifest(run_id, st.session_state.steps)
    st.success(f"Saved cleaned dataset: {out_path}")
    st.success(f"Wrote transformation manifest: {mf_path}")

# ---------------- Footer: Debug / Paths ----------------
with st.expander("Run metadata & paths"):
    st.json({
        "run_id": run_id,
        "raw_path": raw_path,
        "figures_dir": SETTINGS.FIGURES_DIR,
        "profiles_dir": SETTINGS.PROFILES_DIR,
        "processed_dir": SETTINGS.PROCESSED_DIR
    })
