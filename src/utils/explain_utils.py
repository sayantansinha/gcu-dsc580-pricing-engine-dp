import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_importance_scores(model, X_valid: pd.DataFrame, y_valid: pd.Series, n_repeats: int = 5) -> pd.DataFrame:
    r = permutation_importance(model, X_valid, y_valid, n_repeats=n_repeats, random_state=42)
    return pd.DataFrame({"feature": X_valid.columns, "importance": r.importances_mean}).sort_values("importance",
                                                                                                    ascending=False)


def shap_summary_df(model, X_sample: pd.DataFrame):
    """
    Returns mean |SHAP| per feature for tree models (XGBoost/LightGBM) and linear/MLP via KernelExplainer fallback.
    Avoids plotting so Streamlit can chart the table or barplot.
    """
    try:
        import shap
        if hasattr(model, "get_booster") or model.__class__.__name__.startswith("LGBM"):
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_sample)
        else:
            explainer = shap.KernelExplainer(model.predict, shap.sample(X_sample, 200, random_state=42))
            shap_vals = explainer.shap_values(shap.sample(X_sample, 200, random_state=42), nsamples=200)
        sv = np.abs(shap_vals).mean(axis=0)
        return pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": sv}).sort_values("mean_abs_shap",
                                                                                            ascending=False)
    except Exception as e:
        return pd.DataFrame({"feature": [], "mean_abs_shap": []})
