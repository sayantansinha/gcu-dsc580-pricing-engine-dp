"""
feature_utils: small helpers for feature engineering & column ordering.
"""

from typing import List

import pandas as pd


def add_popularity_blend(df: pd.DataFrame, pv_col: str = "pv_30d") -> pd.DataFrame:
    if pv_col in df.columns and "numVotes" in df.columns:
        df["popularity_blend"] = df[pv_col].fillna(0).rank(pct=True) * 0.6 + \
                                 df["numVotes"].fillna(0).rank(pct=True) * 0.4
    return df


def preferred_column_order(df: pd.DataFrame) -> List[str]:
    base = [
        "license_id", "title", "title_final", "title_norm", "release_year",
        "tconst", "method", "match_score",
        "titleType", "startYear", "genres", "runtimeMinutes",
        "averageRating", "numVotes", "log1p_numVotes",
    ]
    pvs = sorted([c for c in df.columns if c.startswith("pv_")])
    tail = ["has_entity"]
    order = [c for c in base if c in df.columns] + pvs + [c for c in tail if c in df.columns]
    order += [c for c in df.columns if c not in order]
    return order
