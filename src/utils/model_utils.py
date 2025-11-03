from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor

from src.utils.metric_utils import regression_metrics


def make_model(name: str, params: Dict[str, Any]):
    name = name.lower()
    if name == "linear regression":
        return LinearRegression(**params)
    if name == "ridge regression":
        return Ridge(**params)
    if name == "lasso regression":
        return Lasso(**params)
    if name == "neural network (mlp)":
        return MLPRegressor(**params)
    if name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(**params)
    if name == "lightgbm":
        return LGBMRegressor(**params)
    raise ValueError(f"Unknown model: {name}")


def train_base_models(
        X_train: pd.DataFrame, y_train: pd.Series,
        X_valid: pd.DataFrame, y_valid: pd.Series,
        selections: List[Tuple[str, dict]],
):
    """
    selections: list of (model_name, params)
    returns: dict with fitted models, valid preds, and metrics
    """
    results = {}
    for model_name, params in selections:
        mdl = make_model(model_name, params)
        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_valid)
        results[model_name] = {
            "model": mdl,
            "valid_preds": preds,
            "metrics": regression_metrics(y_valid, preds),
        }
    return results


def ensemble_simple_average(valid_pred_list: List[np.ndarray]) -> np.ndarray:
    return np.mean(np.column_stack(valid_pred_list), axis=1)


def ensemble_weighted_average(valid_pred_list: List[np.ndarray], rmses: List[float]) -> np.ndarray:
    # weight by inverse RMSE (smaller rmse => larger weight)
    eps = 1e-8
    inv = np.array([1.0 / (r + eps) for r in rmses])
    w = inv / inv.sum()
    M = np.column_stack(valid_pred_list)
    return (M * w).sum(axis=1)


def stacking_with_oof(
        X: pd.DataFrame, y: pd.Series,
        base_selections: List[Tuple[str, dict]],
        meta_params: dict = None,
        n_splits: int = 5, random_state: int = 42
):
    """
    Produces Level-2 Ridge on OOF predictions from base models.
    Returns fitted base models (full), meta model, and valid inference helper.
    """
    meta_params = meta_params or {"alpha": 1.0}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    base_models = {name: [] for name, _ in base_selections}
    oof_stack = np.zeros((len(X), len(base_selections)))

    # OOF generation
    for fold, (tr, va) in enumerate(kf.split(X, y)):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        for j, (name, params) in enumerate(base_selections):
            mdl = make_model(name, params)
            mdl.fit(X_tr, y_tr)
            oof_stack[va, j] = mdl.predict(X_va)
            base_models[name].append(mdl)

    # Fit meta-learner on OOF predictions
    meta = Ridge(**meta_params).fit(oof_stack, y)

    def predict_stacked(X_new: pd.DataFrame) -> np.ndarray:
        # average predictions from base models per type to create meta features
        meta_feats = []
        for name, _ in base_selections:
            fold_preds = [mdl.predict(X_new) for mdl in base_models[name]]
            meta_feats.append(np.mean(np.column_stack(fold_preds), axis=1))
        meta_X = np.column_stack(meta_feats)
        return meta.predict(meta_X)

    return {"base_models": base_models, "meta_model": meta, "predict": predict_stacked}


def residual_chain(
        X_train: pd.DataFrame, y_train: pd.Series,
        X_valid: pd.DataFrame,
        chain: List[Tuple[str, dict]]
):
    """
    chain: ordered list like [("Linear Regression", {...}), ("LightGBM", {...}), ("Neural Network (MLP)", {...})]
    Returns models and cumulative predictions on valid set.
    """
    preds_valid = np.zeros(len(X_valid))
    residual = y_train.copy()
    fitted = []

    # Fit first model on y, subsequent models on residuals of previous stage
    for stage, (name, params) in enumerate(chain, start=1):
        mdl = make_model(name, params)
        mdl.fit(X_train, residual)
        fitted.append((name, mdl))
        # update residuals on train
        train_pred = mdl.predict(X_train)
        residual = residual - train_pred
        # update cumulative predictions on valid
        preds_valid = preds_valid + mdl.predict(X_valid)

    return {"chain_models": fitted, "valid_preds": preds_valid}
