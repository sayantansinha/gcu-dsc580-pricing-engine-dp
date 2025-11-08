import contextlib
import os
from pathlib import Path

import streamlit as st

from src.utils.log_utils import get_logger

LOGGER = get_logger("ui_common")


def inject_css_from_file(css_path: str, rerun_on_first_load: bool = True):
    """
    Inject CSS globally and guarantee it applies even on the first render.

    Parameters
    ----------
    css_path : str
        Path to the CSS file.
    rerun_on_first_load : bool, default True
        Whether to perform a one-time rerun after first CSS injection
        (ensures proper styling on the very first load).
    autorefresh : bool, default False
        If True, watches the CSS file modification time and auto-reruns
        once when the file changes (useful during development).
    state_key_prefix : str, default "_menu_css"
        Prefix for session state keys used internally.
    """
    state_key_prefix: str = "_css_injected"
    injected_key = f"{state_key_prefix}_injected"
    rerun_key = f"{state_key_prefix}_rerun_done"
    mtime_key = f"{state_key_prefix}_mtime"

    # If already injected, optionally check for file change
    # if st.session_state.get(injected_key, False):
    #     if autorefresh:
    #         p = Path(css_path).expanduser()
    #         if p.exists():
    #             mtime = p.stat().st_mtime
    #             if st.session_state.get(mtime_key) != mtime:
    #                 st.session_state[mtime_key] = mtime
    #                 st.session_state[rerun_key] = True
    #                 st.rerun()
    #     return

    # Load and inject CSS
    p = Path(css_path).expanduser()
    if not p.exists():
        st.warning(f"CSS file not found: {p}")
        st.session_state[injected_key] = True
        return

    css_text = p.read_text(encoding="utf-8")
    st.markdown(f"<style>{css_text}</style>", unsafe_allow_html=True)

    # Track mtime for dev autorefresh
    st.session_state[mtime_key] = os.path.getmtime(p)

    # Mark injected and optionally rerun once
    st.session_state[injected_key] = True
    if rerun_on_first_load and not st.session_state.get(rerun_key, False):
        st.session_state[rerun_key] = True
        st.rerun()


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
