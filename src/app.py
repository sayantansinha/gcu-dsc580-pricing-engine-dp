from __future__ import annotations

import sys

import streamlit as st

from ui.common import inject_css
from ui.menu import get_nav
from ui.pipeline_hub import render as render_pipeline_hub
from utils.log_utils import handle_streamlit_exception, get_logger

sys.excepthook = handle_streamlit_exception
LOGGER = get_logger("app")

st.set_page_config(page_title="Predictive Pricing Engine", layout="wide")
inject_css()

# --- session defaults: DO NOT set run_id here ---
for k, v in {
    "df": None,
    "run_id": None,
    "raw_path": None,
    "steps": [],
    "ingested": False,
    "_show_viz_panel": False,
}.items():
    st.session_state.setdefault(k, v)


# --- inline blank-slate renderer ---
def _render_home():
    st.title("Welcome")
    st.subheader("Start a pipeline")
    st.write(
        "Use **New Pipeline** at the bottom-right of the sidebar to create a fresh run, "
        "or click an existing run under **Pipeline Runs** to continue."
    )
    st.caption(
        "After creating/selecting a run, the main area will guide you through staging, "
        "feature building, EDA, cleaning, and modeling."
    )


# --- routes ---
ROUTES = {
    "home": _render_home,
    "pipeline_hub": render_pipeline_hub,
}


def _dispatch(page_key: str):
    ROUTES.get(page_key, _render_home)()


# --- sidebar + route ---
section, page_key = get_nav()  # sidebar always renders
_dispatch(page_key)
