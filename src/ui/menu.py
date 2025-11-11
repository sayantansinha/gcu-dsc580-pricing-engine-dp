from __future__ import annotations

import base64
import os
import time
import uuid
from pathlib import Path
from typing import List, Optional

import streamlit as st

from src.config.env_loader import SETTINGS

# -------------------------------------------------------------------
# App identity (logo + name)
# -------------------------------------------------------------------
APP_NAME = "Predictive Pricing Engine"
_LOGO_PATHS = [
    "src/ui/assets/logo.svg",
    "ui/assets/logo.svg",
    "assets/logo.svg",
    "logo.svg"
]


def _logo_path() -> Path | None:
    for p in _LOGO_PATHS:
        if os.path.exists(p):
            return Path(p)
    return None


def _section_header():
    path = _logo_path()
    if path:
        svg_text = path.read_text(encoding="utf-8")
        b64 = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
        st.sidebar.markdown(
            f"<div class='logo-container'>"
            f"<img src='data:image/svg+xml;base64,{b64}' />"
            f"<span class='app-name-text'>{APP_NAME}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(f"### {APP_NAME}")
    st.sidebar.markdown("---")


# -------------------------------------------------------------------
# Run management helpers
# -------------------------------------------------------------------
def _new_run_id() -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _list_run_dirs() -> List[str]:
    raw_root = Path(SETTINGS.RAW_DIR)
    proc_root = Path(SETTINGS.PROCESSED_DIR)
    seen = set()
    for root in (raw_root, proc_root):
        if root.exists():
            for p in root.iterdir():
                if p.is_dir():
                    seen.add(p.name)
    return sorted(seen, reverse=True)


def _infer_stage(run_id: str) -> str:
    raw_dir = Path(SETTINGS.RAW_DIR) / run_id
    proc_dir = Path(SETTINGS.PROCESSED_DIR) / run_id
    has_raw = raw_dir.exists() and any(raw_dir.iterdir())
    fm_clean = list(proc_dir.glob("feature_master_cleaned_*.parquet"))
    fm_raw = list(proc_dir.glob("feature_master_*.parquet"))
    model_dir = Path(SETTINGS.MODELS_DIR) / run_id
    has_model = model_dir.exists() and any(model_dir.iterdir())
    if has_model: return "MODELED"
    if fm_clean:  return "CLEANED"
    if fm_raw:    return "FEATURE_BUILT"
    if has_raw:   return "STAGED"
    return "EMPTY"


_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _format_created_from_run_id(run_id: str) -> Optional[str]:
    try:
        head = run_id.split("_")[0]
        date_part, time_part = head.split("-") if "-" in head else head.split("_")
    except Exception:
        compact = run_id.replace("_", "").replace("-", "")
        if len(compact) >= 14:
            date_part, time_part = compact[:8], compact[8:14]
        else:
            return None
    try:
        yyyy, mm, dd = int(date_part[:4]), int(date_part[4:6]), int(date_part[6:8])
        HH, MM, SS = int(time_part[:2]), int(time_part[2:4]), int(time_part[4:6])
        ampm = "AM" if HH < 12 else "PM"
        hh12 = HH % 12 or 12
        return f"{_MONTHS[mm - 1]} {dd:02d}, {yyyy} {hh12:02d}:{MM:02d} {ampm}"
    except Exception:
        return None


def _ensure_run_dirs(run_id: str):
    for root in (
            Path(SETTINGS.RAW_DIR),
            Path(SETTINGS.PROCESSED_DIR),
            Path(SETTINGS.FIGURES_DIR),
            Path(SETTINGS.PROFILES_DIR),
            Path(SETTINGS.MODELS_DIR),
            Path(SETTINGS.REPORTS_DIR)
    ):
        (root / run_id).mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# Sidebar menu builder
# -------------------------------------------------------------------
def get_nav() -> tuple[str, str]:
    with st.sidebar:
        # Logo + app header
        _section_header()

        st.markdown('<div class="sidebar-title">Pipeline Runs</div>', unsafe_allow_html=True)
        st.markdown('<div class="run-list-container">', unsafe_allow_html=True)

        run_ids = _list_run_dirs()
        current = st.session_state.get("run_id", "")

        if run_ids:
            for rid in run_ids:
                stage = _infer_stage(rid)
                created = _format_created_from_run_id(rid)
                run_label = f"{rid} ({stage}) - {created if created else ''}"
                row = st.container()
                with row:
                    # Note: the button is NOT a child of this div in Streamlit DOM,
                    # so we style via sidebar-wide selectors above.
                    row.markdown(
                        f'<div class="menu-row {"active" if rid == current else ""}">',
                        unsafe_allow_html=True
                    )
                    clicked = st.button(run_label, key=f"run_{rid}", use_container_width=True)
                    # st.markdown(f'<div class="menu-status">{meta}</div></div>', unsafe_allow_html=True)
                    if clicked:
                        st.session_state["run_id"] = rid
                        st.session_state.pop("staged_raw", None)
                        st.session_state.pop("_raw_preview", None)
                        _ensure_run_dirs(rid)
                        st.rerun()
        else:
            st.caption("No runs yet. Create a new pipeline to get started.")

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="menu-separator"></div>', unsafe_allow_html=True)

        # New Button
        # st.markdown('<div class="sticky-footer"><div class="footer-row">', unsafe_allow_html=True)
        if st.button("New Pipeline", key="btn_new_run", use_container_width=True):
            new_id = _new_run_id()
            st.session_state["run_id"] = new_id
            st.session_state.pop("staged_raw", None)
            st.session_state.pop("_raw_preview", None)
            _ensure_run_dirs(new_id)
            st.rerun()
        # st.markdown('</div></div>', unsafe_allow_html=True)

        if st.session_state.get("run_id"):
            return ("Pipeline Runs", "pipeline_hub")
        else:
            return ("Pipeline Runs", "home")
