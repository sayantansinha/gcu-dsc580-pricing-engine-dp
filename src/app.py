import streamlit as st

from ui.common import inject_css
from ui.cleaning import render_cleaning_section
from ui.display_data import render_display_section
from ui.exploration import render_exploration_section
from ui.sidebar import render_sidebar_and_handle_ingest

st.set_page_config(page_title="Predictive Pricing Engine", layout="wide")
st.title("Predictive Pricing Engine")
inject_css()

# ---- Session defaults ----
defaults = {
    "df": None, "run_id": None, "raw_path": None, "steps": [], "ingested": False,
    "_show_viz_panel": False
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---- Sidebar (ingest) ----
render_sidebar_and_handle_ingest()

# Gate until a dataset is loaded
if not st.session_state.ingested or st.session_state.df is None:
    st.info("Use the sidebar to ingest a dataset (Upload, URL, or Local Path).")
    st.stop()

# ---- Main sections (match your mockups) ----
render_display_section()
render_exploration_section()
render_cleaning_section()
