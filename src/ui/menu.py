from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import streamlit as st
from src.config.env_loader import SETTINGS

APP_NAME = "Predictive Pricing Engine"


def _list_runs() -> List[str]:
    proc = Path(SETTINGS.PROCESSED_DIR)
    if proc.exists():
        return sorted([p.name for p in proc.iterdir() if p.is_dir()], reverse=True)
    raw = Path(SETTINGS.RAW_DIR)
    if raw.exists():
        return sorted([p.name for p in raw.iterdir() if p.is_dir()], reverse=True)
    return []


def _latest_under(prefix: str, under: Path) -> Optional[Path]:
    if not under.exists():
        return None
    files = [p for p in under.iterdir()
             if p.is_file() and p.name.startswith(prefix) and p.suffix == ".parquet"]
    if not files:
        return None
    files.sort(key=lambda p: p.name, reverse=True)
    return files[0]


def _artifact_state_for_run(run_id: str) -> Dict[str, Optional[Path]]:
    run_proc = Path(SETTINGS.PROCESSED_DIR) / run_id
    fm_clean = _latest_under("feature_master_clean_", run_proc)
    fm_raw = _latest_under("feature_master_", run_proc)
    model = None
    models_dir = run_proc / "models"
    if models_dir.exists():
        any_files = [p for p in models_dir.iterdir() if p.is_file()]
        if any_files:
            model = any_files[0]
    if model is None and "last_model" in st.session_state:
        model = Path("__in_session__")
    return {"fm_clean": fm_clean, "fm_raw": fm_raw, "model": model}


def _header():
    st.sidebar.markdown(f"### {APP_NAME}")
    st.sidebar.markdown("---")


def get_nav() -> tuple[str, str]:
    """Sidebar lists runs + Resume. The main area is handled by the Pipeline Hub page."""
    st.session_state.setdefault("nav_page", "pipeline_hub")
    _header()

    runs = _list_runs()
    st.sidebar.markdown("#### Existing Runs")
    if not runs:
        st.sidebar.caption("No previous runs found.")
        st.sidebar.markdown("---")
        return "New Pipeline", st.session_state["nav_page"]

    for rid in runs:
        artifacts = _artifact_state_for_run(rid)
        if artifacts["model"]:
            stage = "MODELED";
            badge = "🤖"
        elif artifacts["fm_clean"]:
            stage = "CLEANED";
            badge = "🧹"
        elif artifacts["fm_raw"]:
            stage = "FEATURE_BUILT";
            badge = "🧱"
        else:
            stage = "STAGED";
            badge = "📥"

        cols = st.sidebar.columns([0.75, 0.25])
        with cols[0]:
            st.caption(f"{badge} {rid}")
            st.caption(f"• Stage: **{stage}**")
        with cols[1]:
            if st.sidebar.button("Resume", key=f"resume:{rid}"):
                st.session_state["run_id"] = rid
                # remember the best FM for downstream pages
                best_fm = artifacts["fm_clean"] or artifacts["fm_raw"]
                if best_fm and best_fm.exists():
                    st.session_state["last_feature_master_path"] = str(best_fm)
                # jump to the hub and rerun
                st.session_state["nav_page"] = "pipeline_hub"
                st.rerun()

    st.sidebar.markdown("---")
    # Always return the hub page; it renders the whole pipeline in the main area
    return "New Pipeline", "pipeline_hub"
