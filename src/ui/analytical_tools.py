# src/ui/analytical_tools.py
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from src.config.env_loader import SETTINGS
from src.services.analytics.modeling import (
    train_models_parallel,
    combine_average,
    combine_weighted_inverse_rmse,
)
from src.utils.log_utils import streamlit_safe


# ----------------------------------------------------------------------
# Helper: find all feature master files
# ----------------------------------------------------------------------
def _find_feature_masters() -> list[tuple[str, str]]:
    """List all feature_master_*.parquet files under PROCESSED_DIR."""
    processed = SETTINGS.PROCESSED_DIR
    if not os.path.isdir(processed):
        return []
    items: list[tuple[str, str]] = []
    for fname in sorted(os.listdir(processed), reverse=True):
        if fname.startswith("feature_master_") and fname.endswith(".parquet"):
            full_path = os.path.join(processed, fname)
            items.append((fname, full_path))
    return items


def _load_feature_master(path: str) -> pd.DataFrame | None:
    """Safely load the chosen feature master file."""
    try:
        df = pd.read_parquet(path)
        if df.empty:
            st.warning(f"{os.path.basename(path)} is empty.")
            return None
        return df
    except Exception as e:
        st.error(f"Failed to read {path}: {e}")
        return None


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _model_param_ui(algo: str) -> dict:
    """Param UI for allowed models (from your excerpt)."""
    params: dict = {}
    if algo in ("Ridge", "Lasso"):
        params["alpha"] = st.number_input(f"{algo} alpha", 0.0001, 10.0, 1.0, key=f"{algo}_alpha")
    elif algo == "XGBoost":
        params["n_estimators"] = st.slider("XGBoost n_estimators", 100, 1500, 600, 50, key="xgb_ne")
        params["max_depth"] = st.slider("XGBoost max_depth", 2, 12, 6, key="xgb_md")
        params["learning_rate"] = st.number_input("XGBoost learning_rate", 0.01, 0.5, 0.05, key="xgb_lr")
        params["subsample"] = 0.9
        params["colsample_bytree"] = 0.9
        params["random_state"] = 42
        params["tree_method"] = "hist"
    elif algo == "LightGBM":
        params["n_estimators"] = st.slider("LightGBM n_estimators", 100, 2000, 800, 50, key="lgb_ne")
        params["learning_rate"] = st.number_input("LightGBM learning_rate", 0.005, 0.5, 0.05, key="lgb_lr")
        params["max_depth"] = st.number_input("LightGBM max_depth (-1=none)", -1, 64, -1, 1, key="lgb_md")
        params["random_state"] = 42
    elif algo == "MLP":
        hl = st.text_input("MLP hidden_layer_sizes (e.g., 256,128)", value="256,128", key="mlp_hl")
        params["hidden_layer_sizes"] = tuple(int(x.strip()) for x in hl.split(",") if x.strip())
        params["max_iter"] = st.number_input("MLP epochs (max_iter)", 50, 2000, 300, 50, key="mlp_ep")
        params["random_state"] = 42
    # LinearRegression has no key params here
    return params


def _multimodel_param_ui(selected_models: list[str]) -> dict[str, dict]:
    blocks: dict[str, dict] = {}
    with st.expander("Hyperparameters (per selected model)", expanded=False):
        for m in selected_models:
            st.markdown(f"**{m}**")
            blocks[m] = _model_param_ui(m)
            st.markdown("---")
    return blocks


# ----------------------------------------------------------------------
# Page render (MENU-TRIGGERED ENTRY POINT) — PARALLEL ONLY
# ----------------------------------------------------------------------
@streamlit_safe
def render():
    st.header("Analytical Tools - Model")

    # Discover feature masters
    fm_files = _find_feature_masters()
    if not fm_files:
        st.warning(
            "No feature master files found in the processed directory.\n\n"
            "Go to **Build Feature Master** to create one (named like `feature_master_<suffix>.parquet`)."
        )
        return

    # Default selection from last run (if any)
    default_file = None
    if "last_model_run_dir" in st.session_state:
        path = st.session_state["last_model_run_dir"]
        fname = os.path.basename(path)
        for name, _full in fm_files:
            if name == fname:
                default_file = name
                break

    labels = [name for name, _ in fm_files]
    default_idx = labels.index(default_file) if default_file in labels else 0
    selected_label = st.selectbox("Select Feature Master File", labels, index=default_idx)
    selected_path = dict(fm_files)[selected_label]

    df = _load_feature_master(selected_path)
    if df is None:
        return

    num_cols = _numeric_columns(df)
    if not num_cols:
        st.error("No numeric columns found in the selected feature master.")
        return

    # Target + features
    target = st.selectbox("Target (dependent variable)", num_cols)
    features = st.multiselect(
        "Feature columns (independent variables)",
        [c for c in df.columns if c != target],
        default=[c for c in num_cols if c != target],
    )
    if not features:
        st.info("Please select at least one feature.")
        return

    # Allowed models only (from your excerpt)
    allowed = ["LinearRegression", "Ridge", "Lasso", "XGBoost", "LightGBM", "MLP"]
    with st.expander("Model Selection", expanded=True):
        selected_models = st.multiselect(
            "Select models to train (runs in parallel; ensembles computed automatically)",
            allowed,
            default=["LinearRegression", "LightGBM", "MLP"],
        )
    if not selected_models:
        st.info("Select at least one model.")
        return

    params_map = _multimodel_param_ui(selected_models)

    # Single unified action: Train & Evaluate (parallel only)
    if st.button("Train & Evaluate", type="primary"):
        with st.spinner(f"Training in parallel on {selected_label}..."):
            base_out = train_models_parallel(
                df[features + [target]],
                target,
                selected_models,
                params_map,
            )

        st.subheader("Per-model Validation Metrics")
        st.dataframe(pd.DataFrame(base_out["per_model_metrics"]).set_index("model"))

        # Ensembles (computed automatically behind the scenes)
        avg = combine_average(base_out)
        wgt = combine_weighted_inverse_rmse(base_out)

        st.subheader("Ensemble (automatic)")
        st.markdown("**Simple Average**")
        st.json(avg["metrics"])
        st.markdown("**Weighted Average (by inverse RMSE)**")
        st.json(wgt["metrics"])

        # Persist context
        st.session_state["last_model"] = {"base": base_out, "ensemble_avg": avg, "ensemble_wgt": wgt}
        st.session_state["last_model_run_dir"] = selected_path
        st.success("Training and evaluation completed.")
