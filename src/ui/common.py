import contextlib

import streamlit as st

from src.utils.log_utils import get_logger

LOGGER = get_logger("ui_common")


# Inject CSS
def inject_css():
    st.markdown("""
    <style>
    /* Use theme tokens so dark/light both work */
    section[data-testid="stSidebar"] * {
      color: var(--text-color) !important;
    }
    /* Expander header text */
    section[data-testid="stSidebar"] details > summary {
      color: var(--text-color) !important;
    }
    /* Expander body background follows secondary panel */
    section[data-testid="stSidebar"] .stExpanderContent {
      background: var(--secondary-background-color) !important;
    }
    /* Buttons in sidebar: ensure readable text */
    section[data-testid="stSidebar"] .stButton > button {
      color: var(--background-color) !important;           /* text */
      background: var(--primary-color) !important;         /* bg */
      border: 1px solid var(--primary-color) !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
      filter: brightness(0.92);
    }
    </style>
    """, unsafe_allow_html=True)
    # if "_css_injected" not in st.session_state:
    #     LOGGER.info("Injecting CSS")
    #     st.markdown("""
    #     <style>
    #       /* Expander (collapsible panel) — target multiple Streamlit versions */
    #       div[data-testid="stExpander"] {
    #         background: #f7faff;
    #         border: 1px solid #cfd8eb;
    #         border-radius: 10px;
    #         margin: 1rem 0;
    #         box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    #       }
    #       div[data-testid="stExpander"] .st-expander__header,
    #       details.st-expander > summary {
    #         background: linear-gradient(180deg,#e4edff 0%, #dbe7ff 100%);
    #         border-bottom: 1px solid #c7d7f2;
    #         border-radius: 10px 10px 0 0;
    #         padding: 0.75rem 1rem !important;
    #         font-weight: 700;
    #         font-size: 1.1rem;   /* larger than tab labels */
    #         color: #143d85;
    #         letter-spacing: 0.2px;
    #         list-style: none;
    #       }
    #       div[data-testid="stExpander"] .st-expander__icon,
    #       details.st-expander summary svg {
    #         color: #143d85 !important;
    #         transform: scale(1.1);
    #       }
    #
    #       /* Tabs content: fixed height + scroll */
    #       .tab-scroll {
    #         max-height: 440px;
    #         overflow-y: auto;
    #         padding-right: 8px;
    #         border: 1px dashed #d9e2f1;
    #         border-radius: 6px;
    #         background: #ffffff;
    #       }
    #     </style>
    #     """, unsafe_allow_html=True)
    #     st.session_state["_css_injected"] = True


@contextlib.contextmanager
def noop_container():
    """A no-op context manager that behaves like a layout container."""
    yield


def section_panel(title: str, expanded: bool = False):
    """
    Standard section wrapper. By default, renders an expander.
    If st.session_state['_suppress_section_panel'] is True, render a simple container instead.
    This allows parent pages (like the Pipeline Hub) to avoid nested expanders.
    """
    if st.session_state.get("_suppress_section_panel", False):
        # render without an expander to avoid nesting
        return st.container()
    # default behavior (your original pattern)
    return st.expander(title, expanded=expanded)


def begin_tab_scroll():
    """Start a fixed-height, scrollable area inside a tab."""
    st.markdown("<div class='tab-scroll'>", unsafe_allow_html=True)


def end_tab_scroll():
    """Close the scrollable area."""
    st.markdown("</div>", unsafe_allow_html=True)


def get_run_id_from_session_state() -> str:
    """Get Rub ID from the session"""
    return st.session_state["run_id"]
