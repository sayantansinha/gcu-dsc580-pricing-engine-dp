"""
imdb_utils: loaders + candidate universe assembly for IMDb.
"""

import numpy as np
import pandas as pd

from src.services.data_io import load_processed
from src.utils.log_utils import get_logger
from tools.lib.io_utils import ensure_columns
from tools.lib.title_utils import normalize_title

LOGGER = get_logger("imdb_utils")


def load_imdb_core(basics_file_name: str, ratings_file_name: str) -> pd.DataFrame:
    basics = load_processed(basics_file_name)
    ratings = load_processed(ratings_file_name)
    if basics is not None and ratings is not None:
        core = basics.merge(ratings, on="tconst", how="left")
    elif basics is not None:
        core = basics
    elif ratings is not None:
        core = ratings
    else:
        raise ValueError("IMDB core cannot be formulated as neither IMDB basics nor ratings file exists")

    if "primaryTitle" not in core.columns and "primary_title" in core.columns:
        core = core.rename(columns={"primary_title": "primaryTitle"})
    if "releaseYear" not in core.columns and "release_year" in core.columns:
        core = core.rename(columns={"release_year": "releaseYear"})
    for c in ["numVotes", "averageRating", "runtimeMinutes", "titleType", "genres"]:
        if c not in core.columns:
            core[c] = np.nan
    core["primary_title_norm"] = normalize_title(core["primaryTitle"])
    LOGGER.info(f"IMDB core loaded from basics file [{basics_file_name}] and ratings file [{ratings_file_name}]")
    return core


def load_imdb_akas(file_name: str) -> pd.DataFrame:
    akas = load_processed(file_name)
    ensure_columns(akas, ["tconst", "title"], "IMDb AKAs")
    akas["aka_title_norm"] = normalize_title(akas["title"])
    akas = akas[["tconst", "aka_title_norm", "region", "language"]].drop_duplicates()
    LOGGER.info(f"IMDB AKAs loaded from file [{file_name}]")
    return akas


def imdb_candidate_universe(core: pd.DataFrame, akas: pd.DataFrame) -> pd.DataFrame:
    cand_core = core[["tconst", "primary_title_norm", "releaseYear", "numVotes"]].rename(
        columns={"primary_title_norm": "title_norm"}
    )
    cand_akas = akas.rename(columns={"aka_title_norm": "title_norm"}) \
        .merge(core[["tconst", "releaseYear", "numVotes"]], on="tconst", how="left")
    cand = pd.concat([cand_core, cand_akas], ignore_index=True) \
        .dropna(subset=["title_norm"]).drop_duplicates()
    return cand
