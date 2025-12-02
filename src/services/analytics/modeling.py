from __future__ import annotations
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# BP test (your preference)
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

# ---------------------------------------------------------------------
# Allowed models (ONLY those from your excerpt)
# ---------------------------------------------------------------------
AVAILABLE_MODELS: Dict[str, Any] = {
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "XGBoost": XGBRegressor,
    "LightGBM": LGBMRegressor,
    "MLP": MLPRegressor,
}


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def _split(df: pd.DataFrame, target: str, test_size: float = 0.2, rs: int = 42):
    X = df.drop(columns=[target])
    y = df[target].values
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=test_size, random_state=rs)
    return Xtr, Xva, ytr, yva, X.columns.tolist()


def _build_model(name: str, params: Dict[str, Any]):
    cls = AVAILABLE_MODELS.get(name)
    if cls is None:
        raise ValueError(f"Model {name} is not supported.")
    if cls is XGBRegressor and XGBRegressor is None:
        raise ImportError("XGBoost not installed.")
    if cls is LGBMRegressor and LGBMRegressor is None:
        raise ImportError("LightGBM not installed.")
    return cls(**(params or {}))


def _breusch_pagan_for_linear(X_va: pd.DataFrame, y_va: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    try:
        residuals = y_va - y_pred
        X_sm = sm.add_constant(X_va, has_constant="add")
        bp = het_breuschpagan(residuals, X_sm)
        return {"LM": float(bp[0]), "LM_pvalue": float(bp[1]), "F": float(bp[2]), "F_pvalue": float(bp[3])}
    except Exception:
        return {}


# ---------------------------------------------------------------------
# Single model (kept for your existing flow)
# ---------------------------------------------------------------------
def train_model(df: pd.DataFrame, target: str, algo: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    Xtr, Xva, ytr, yva, feat_names = _split(df, target)
    model = _build_model(algo, params or {})
    model.fit(Xtr, ytr)
    preds = model.predict(Xva)
    out = {
        "algo": algo,
        "params": params or {},
        "metrics": _metrics(yva, preds),
        "features": feat_names,
    }
    if algo in ("LinearRegression", "Ridge", "Lasso"):
        out["bp"] = _breusch_pagan_for_linear(pd.DataFrame(Xva, columns=feat_names), yva, preds)
    return out


# ---------------------------------------------------------------------
# Parallel training (evaluate individually)
# ---------------------------------------------------------------------
def train_models_parallel(
        df: pd.DataFrame,
        target: str,
        model_names: List[str],
        params_map: Dict[str, Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    Train and evaluate multiple models on a consistent train/validation split.

    Returns:
        {
            "features": [str],              # feature names
            "X_valid": pd.DataFrame,        # NEW: validation feature matrix (for explainability)
            "y_valid": np.ndarray,          # validation target
            "valid_preds": {name: np.ndarray},
            "per_model_metrics": [ {...}, ... ],
            "models": {name: fitted_model},
        }
    """
    Xtr, Xva, ytr, yva, feat_names = _split(df, target)
    params_map = params_map or {}
    per_model_metrics: List[Dict[str, Any]] = []
    valid_preds: Dict[str, np.ndarray] = {}
    fitted: Dict[str, Any] = {}

    # Keep validation features as a DataFrame so downstream code (explainability, BP test, etc.)
    # has easy access to column names.
    Xva_df = pd.DataFrame(Xva, columns=feat_names)

    for name in model_names:
        mdl = _build_model(name, params_map.get(name, {}))
        mdl.fit(Xtr, ytr)
        pr = mdl.predict(Xva)
        valid_preds[name] = pr
        fitted[name] = mdl

        row = {"model": name, **_metrics(yva, pr)}
        # Compute BP for linear-family models
        if name in ("LinearRegression", "Ridge", "Lasso"):
            row["bp"] = _breusch_pagan_for_linear(Xva_df, yva, pr)
        per_model_metrics.append(row)

    return {
        "features": feat_names,
        "X_valid": Xva_df,  # NEW: used by explainability + reports
        "y_valid": yva,
        "valid_preds": valid_preds,
        "per_model_metrics": per_model_metrics,
        "models": fitted,
    }


# ---------------------------------------------------------------------
# Ensembles (computed automatically; no UI selection)
# ---------------------------------------------------------------------
def combine_average(base_out: Dict[str, Any]) -> Dict[str, Any]:
    yv = base_out["y_valid"]
    M = np.column_stack(list(base_out["valid_preds"].values()))
    pred = M.mean(axis=1)
    return {"kind": "average", "pred": pred, "metrics": _metrics(yv, pred)}


def combine_weighted_inverse_rmse(base_out: Dict[str, Any]) -> Dict[str, Any]:
    yv = base_out["y_valid"]
    metrics = {r["model"]: r for r in base_out["per_model_metrics"]}
    names = list(base_out["valid_preds"].keys())
    M = np.column_stack([base_out["valid_preds"][n] for n in names])
    rmses = np.array([metrics[n]["RMSE"] for n in names])
    inv = 1.0 / (rmses + 1e-8)
    w = inv / inv.sum()
    pred = (M * w).sum(axis=1)
    return {
        "kind": "weighted_inverse_rmse",
        "pred": pred,
        "weights": {n: float(w[i]) for i, n in enumerate(names)},
        "metrics": _metrics(yv, pred)
    }
