"""
title_utils: normalization & string similarity helpers.
"""

import pandas as pd
from difflib import SequenceMatcher

def normalize_title(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.lower().str.strip()
    s = s.str.replace(r"[_]+", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True)
    s = s.str.replace(r"[^\w\s]", "", regex=True)
    return s

def str_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a or "", b or "").ratio()
