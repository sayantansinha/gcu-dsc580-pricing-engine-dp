from __future__ import annotations
from typing import Dict, Any
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

# Catalog (your preference: LR baseline + OLS/BP)
AVAILABLE_MODELS: Dict[str, Any] = {
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "ElasticNet": ElasticNet,
    "RandomForest": RandomForestRegressor,
    "XGBoost": XGBRegressor,
}


def _split_xy(df: pd.DataFrame, target: str):
    y = df[target].astype(float)
    X = pd.get_dummies(df.drop(columns=[target]), drop_first=True, dtype=float)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(df: pd.DataFrame, target: str, algo: str, params: Dict[str, Any]):
    """Fit a model and return predictions, metrics, and Breusch–Pagan diagnostics."""
    xtr, xte, ytr, yte = _split_xy(df, target)
    model = AVAILABLE_MODELS[algo]
    # sensible defaults
    if algo == "RandomForest":
        params = {"n_estimators": 500, "random_state": 42, **params}
    if algo == "XGBoost":
        params = {
            "n_estimators": 600, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.9, "colsample_bytree": 0.9, "objective": "reg:squarederror",
            "n_jobs": -1, **params
        }

    model = model(**params)
    model.fit(xtr, ytr)
    yhat = model.predict(xte)

    rmse = mean_squared_error(yte, yhat, squared=False)
    r2 = r2_score(yte, yhat)

    # OLS + Breusch–Pagan
    xte_sm = sm.add_constant(xte)
    ols = sm.OLS(yte, xte_sm).fit()
    bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(ols.resid, ols.model.exog)

    return {
        "model_name": algo,
        "model": model,
        "X_test": xte.reset_index(drop=True),
        "y_true": yte.reset_index(drop=True),
        "y_pred": pd.Series(yhat, index=yte.index).reset_index(drop=True),
        "metrics": {"rmse": float(rmse), "r2": float(r2)},
        "bp": {
            "lm_stat": float(bp_lm), "lm_pvalue": float(bp_lm_p),
            "f_stat": float(bp_f), "f_pvalue": float(bp_f_p)
        },
        "pipeline_repr": repr(model),
    }
