from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

import streamlit as st

from config.env_loader import SETTINGS
from ui.common import get_run_id_from_session_state
from utils.log_utils import handle_streamlit_exception, get_logger
# Your existing globals
from ui.common import inject_css
from ui.cleaning import render_cleaning_section
from ui.display_data import render_display_section
from ui.exploration import render_exploration_section
from ui.pipeline_hub import render as render_pipeline_hub

# New sidebar menu
from ui.menu import get_nav

# Loading Data pages
from ui import source_loader
from ui import features

# Analytics Tools pages
from ui import analytical_tools, visual_tools, report_generator

sys.excepthook = handle_streamlit_exception

LOGGER = get_logger("app")
LOGGER.info("Streamlit exception hook active.")

st.set_page_config(page_title="Predictive Pricing Engine", layout="wide")
inject_css()


def _new_run_id() -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}"


# Keep your existing session defaults
LOGGER.info("Setting session states")
_defaults = {
    "df": None,
    "run_id": None,
    "raw_path": None,
    "steps": [],
    "ingested": False,
    "_show_viz_panel": False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        if k == "run_id":
            st.session_state[k] = _new_run_id()
        else:
            st.session_state[k] = v

# Make run directory under data directories
run_id = get_run_id_from_session_state()
LOGGER.info(f"Creating session directory using run_id [{run_id}]")
os.makedirs(Path(SETTINGS.RAW_DIR) / run_id, exist_ok=True)
os.makedirs(Path(SETTINGS.PROCESSED_DIR) / run_id, exist_ok=True)
os.makedirs(Path(SETTINGS.FIGURES_DIR) / run_id, exist_ok=True)
os.makedirs(Path(SETTINGS.PROFILES_DIR) / run_id, exist_ok=True)
os.makedirs(Path(SETTINGS.REPORTS_DIR) / run_id, exist_ok=True)

# Router
ROUTES = {
    # Hub orchestrates the whole pipeline in the main area
    "pipeline_hub": render_pipeline_hub,  # <-- NEW

    # (still available for direct access, Hub calls them internally)
    # "source_selector": source_loader.render,
    # "display_data": render_display_section,
    # "exploration": render_exploration_section,
    # "cleaning": render_cleaning_section,
    # "feature_builder": features.render,

    # Analytics pages (Hub calls these too)
    # "analytical_tools": render_analytical_tools,
    # "visual_tools": render_visual_tools,
    # "report_generator": render_report_generator,
}

# Ensure a sane default page (Hub)
st.session_state.setdefault("nav_page", "pipeline_hub")


# Pages that require an ingested dataset in session (unchanged behavior)
# _PAGES_REQUIRE_DATA = {
#     "display_data",
#     "exploration",
#     "cleaning",
# }


def _dispatch(page_key: str):
    fn = ROUTES.get(page_key) or ROUTES["pipeline_hub"]
    fn()


try:
    section, page_key = get_nav()
except Exception:
    LOGGER.exception("Error in get_nav()")
    st.error("Error rendering menu. See logs for details.")
else:
    try:
        _dispatch(page_key)
    except Exception:
        LOGGER.exception("Error dispatching page '%s'", page_key)
        st.error("Error rendering page. See logs for details.")
