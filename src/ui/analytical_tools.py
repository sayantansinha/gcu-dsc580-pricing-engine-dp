from __future__ import annotations
import os
import streamlit as st
import pandas as pd

from src.config import SETTINGS
from src.services.analytics.modeling import AVAILABLE_MODELS, train_model


def _discover_runs() -> list[str]:
    runs_root = os.path.join(SETTINGS.PROCESSED_DIR, "runs")
    if not os.path.isdir(runs_root):
        return []
    runs = [d for d in os.listdir(runs_root) if os.path.isdir(os.path.join(runs_root, d))]
    runs.sort(reverse=True)  # newest first
    return runs


def _load_feature_master_from(run_dir: str) -> pd.DataFrame | None:
    path = os.path.join(run_dir, "feature_master.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def _runs_root() -> str:
    return os.path.join(SETTINGS.PROCESSED_DIR, "runs")


def _ensure_runs_ready() -> bool:
    runs = _discover_runs()
    if not runs and "last_run_dir" not in st.session_state:
        st.warning("No feature master found. Go to **Build Feature Master** first.")
        return False
    return True


def _default_run_index(runs: list[str], default_run_dir: str | None) -> int | None:
    if not runs:
        return None
    if not default_run_dir:
        return 0
    run_name = os.path.basename(default_run_dir)
    return runs.index(run_name) if run_name in runs else 0


def _select_run(runs: list[str]) -> str | None:
    idx = _default_run_index(runs, st.session_state.get("last_run_dir"))
    return st.selectbox("Select Run", runs, index=idx if runs else None,
                        placeholder="Pick a run with a feature master")


def _load_master_df(run_name: str) -> pd.DataFrame | None:
    path = os.path.join(_runs_root(), run_name, "feature_master.parquet")
    if not os.path.exists(path):
        st.error(f"No feature_master.parquet found in: {os.path.dirname(path)}")
        return None
    df = pd.read_parquet(path)
    if df.empty:
        st.error("Feature master is empty.")
        return None
    return df


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _pick_target_features(df: pd.DataFrame, num_cols: list[str]) -> tuple[str | None, list[str]]:
    target = st.selectbox("Target", num_cols)
    features = st.multiselect("Features", [c for c in df.columns if c != target],
                              default=[c for c in num_cols if c != target])
    return target, features


def _pick_algo_and_params() -> tuple[str, dict]:
    algo = st.selectbox("Algorithm", list(AVAILABLE_MODELS.keys()))
    params = {}
    if algo in ("Ridge", "Lasso", "ElasticNet"):
        params["alpha"] = st.number_input("alpha", 0.0001, 10.0, 1.0)
        if algo == "ElasticNet":
            params["l1_ratio"] = st.slider("l1_ratio", 0.0, 1.0, 0.5)
    elif algo == "RandomForest":
        params["n_estimators"] = st.slider("n_estimators", 100, 2000, 500, 50)
        params["random_state"] = 42
    elif algo == "XGBoost":
        params["n_estimators"] = st.slider("n_estimators", 100, 1500, 600, 50)
        params["max_depth"] = st.slider("max_depth", 2, 12, 6)
        params["learning_rate"] = st.number_input("learning_rate", 0.01, 0.5, 0.05)
    return algo, params


def _train_and_show(df: pd.DataFrame, target: str, features: list[str], algo: str, params: dict, run_dir: str):
    with st.spinner("Training…"):
        res = train_model(df[features + [target]], target, algo, params)
    st.session_state["last_model"] = res
    st.session_state["last_model_run_dir"] = run_dir
    st.success(f"{algo} trained. R²={res['metrics']['r2']:.3f} | RMSE={res['metrics']['rmse']:.3f}")
    st.subheader("Metrics")
    st.json(res["metrics"])
    st.subheader("Breusch–Pagan")
    st.json(res["bp"])


def render():
    st.header("Analytical Tools - Model")
    if not _ensure_runs_ready():
        return

    runs = _discover_runs()
    selected_run = _select_run(runs)
    if not selected_run:
        st.warning("Pick a run to continue.")
        return

    run_dir = os.path.join(_runs_root(), selected_run)
    df = _load_master_df(selected_run)
    if df is None:
        return

    num_cols = _numeric_cols(df)
    if not num_cols:
        st.error("No numeric columns found in feature master. Check your mapping/build step.")
        return

    with st.expander("Model Picker", expanded=True):
        target, features = _pick_target_features(df, num_cols)
        if not features:
            st.info("Select at least one feature to train.")
            return
        algo, params = _pick_algo_and_params()

        if st.button("Train model"):
            _train_and_show(df, target, features, algo, params, run_dir)
