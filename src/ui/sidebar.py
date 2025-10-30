# src/ui/sidebar.py

import os
import io
import pandas as pd
import streamlit as st

from src.config import SETTINGS
from src.utils.io_utils import save_uploaded_file, read_from_url, load_local_file, _new_run_id


# ---------- tiny utilities ----------

def _reset_mode_state():
    for k in ("upload_widget", "url_widget", "local_widget"):
        st.session_state.pop(k, None)


def _get_mode() -> str:
    return st.radio("Source", ["Upload File", "From URL", "Load Local Path"], key="ingest_mode")


def _ensure_mode_fresh(mode: str):
    if st.session_state.get("_prev_ingest_mode") != mode:
        _reset_mode_state()
        st.session_state["_prev_ingest_mode"] = mode


def _render_input(mode: str):
    """
    Render exactly one input widget based on mode and return a tuple
    (uploaded_file, url, local_path) with two Nones and one populated value.
    """
    if mode == "Upload File":
        uploaded = st.file_uploader(
            "Upload CSV or Parquet",
            type=["csv", "parquet", "pq"],
            help="Drag and drop or browse a file (max 200 MB).",
            key="upload_widget",
        )
        return uploaded, None, None

    if mode == "From URL":
        url = st.text_input("Enter URL (CSV/Parquet/TSV/TSV.GZ)", key="url_widget")
        return None, url, None

    # Load Local Path
    path = st.text_input("Local file path (CSV/Parquet)", key="local_widget")
    return None, None, path


# ---------- ingestion handlers (no UI branching here) ----------

def _ingest_upload(uploaded_file, base_name):
    if not uploaded_file:
        raise ValueError("Please select a file to upload.")
    if uploaded_file.name.lower().endswith((".parquet", ".pq")):
        df = pd.read_parquet(io.BytesIO(uploaded_file.read()))
    else:
        df = pd.read_csv(uploaded_file)
    rid, rpath = save_uploaded_file(df, base_name)
    st.session_state.update(df=df, run_id=rid, raw_path=rpath, steps=[], ingested=True)
    return f"Ingested: {rpath}"


def _ingest_url(url, base_name):
    if not url:
        raise ValueError("Please enter a URL.")
    df, rid, rpath = read_from_url(url, base_name)
    st.session_state.update(df=df, run_id=rid, raw_path=rpath, steps=[], ingested=True)
    return f"Ingested: {rpath}"


def _ingest_local(path, base_name):
    if not path:
        raise ValueError("Please provide a local file path.")
    df = load_local_file(path)
    rid = _new_run_id("file")
    os.makedirs(SETTINGS.RAW_DIR, exist_ok=True)
    rpath = os.path.join(SETTINGS.RAW_DIR, f"{base_name}_{rid}.parquet")
    df.to_parquet(rpath, index=False)
    st.session_state.update(df=df, run_id=rid, raw_path=rpath, steps=[], ingested=True)
    return f"Loaded and saved copy to: {rpath}"


# ---------- public entry (low complexity) ----------

def render_sidebar_and_handle_ingest():
    with st.sidebar:
        st.header("Select source")

        mode = _get_mode()
        _ensure_mode_fresh(mode)

        st.divider()
        st.subheader("File Selector")

        uploaded, url, local_path = _render_input(mode)

        st.subheader("Base name")
        base_name = st.text_input("", value="dataset", label_visibility="collapsed")

        st.markdown("---")
        if st.button("Load Data", use_container_width=True):
            try:
                handlers = {
                    "Upload File": lambda: _ingest_upload(uploaded, base_name),
                    "From URL": lambda: _ingest_url(url, base_name),
                    "Load Local Path": lambda: _ingest_local(local_path, base_name),
                }
                msg = handlers[mode]()  # dispatch without if/elif chains
                st.success(msg)
            except Exception as e:
                st.error(f"Ingestion failed: {e}")
