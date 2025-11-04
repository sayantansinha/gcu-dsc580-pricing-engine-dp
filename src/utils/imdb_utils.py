"""
imdb_utils: loaders + candidate universe assembly for IMDb.
"""

import numpy as np
import pandas as pd

from src.utils.data_io_utils import load_processed, load_raw
from src.utils.log_utils import get_logger
from src.utils.title_utils import normalize_title

LOGGER = get_logger("imdb_utils")


def load_imdb_core(basics_file_path: str, ratings_file_path: str) -> pd.DataFrame:
    LOGGER.info("Loading IMDB Core....")
    basics = load_raw(basics_file_path)
    ratings = load_raw(ratings_file_path)

    try:
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
        if "startYear" in core.columns and "releaseYear" not in core.columns:
            core["releaseYear"] = pd.to_numeric(core["startYear"].fillna(0), errors="coerce").astype("int64")
            core = core.drop(columns=["startYear"])
        for c in ["numVotes", "averageRating", "runtimeMinutes", "titleType", "genres"]:
            if c not in core.columns:
                core[c] = np.nan
        core["primary_title_norm"] = normalize_title(core["primaryTitle"])
        core.rename(columns={"tconst": "title_id"}, inplace=True)
        LOGGER.info(
            f"IMDB core loaded successfully from basics file [{basics_file_path}] and ratings file [{ratings_file_path}]")
        return core
    except Exception as ex:
        LOGGER.error("Error loading IMDB core", exc_info=ex)
        raise ValueError("Error loading IMDB core") from ex


def load_imdb_akas(file_path: str) -> pd.DataFrame:
    LOGGER.info("Loading IMDB AKAs....")
    akas = load_raw(file_path)

    try:
        akas["aka_title_norm"] = normalize_title(akas["title"])
        akas = akas[["titleId", "aka_title_norm", "region", "language"]].drop_duplicates()
        akas.rename(columns={"titleId": "title_id"}, inplace=True)
        LOGGER.info(f"IMDB AKAs loaded successfully from file [{file_path}]")
        return akas
    except Exception as ex:
        LOGGER.error("Error loading IMDB AKAs", exc_info=ex)
        raise ValueError("Error loading IMDB AKAs") from ex


def imdb_candidate_universe(core: pd.DataFrame, akas: pd.DataFrame) -> pd.DataFrame:
    LOGGER.info("Formulating IMDB Candidate Universe....")
    try:
        cand_core = core[["title_id", "primary_title_norm", "releaseYear", "numVotes"]].rename(
            columns={"primary_title_norm": "title_norm"}
        )
        cand_akas = akas.rename(columns={"aka_title_norm": "title_norm"}) \
            .merge(core[["title_id", "releaseYear", "numVotes"]], on="title_id", how="left")
        cand = pd.concat([cand_core, cand_akas], ignore_index=True) \
            .dropna(subset=["title_norm"]).drop_duplicates()
        LOGGER.info("IMDB Candidate Universe populated successfully")
        return cand
    except Exception as ex:
        LOGGER.error("Error loading IMDB Candidate Universe", exc_info=ex)
        raise ValueError("Error loading IMDB Candidate Universe") from ex
