import numpy as np
import pandas as pd

from config.config_loader import load_config_from_file
from lib.imdb_utils import load_imdb_core
from lib.io_utils import ensure_columns
from lib.title_utils import normalize_title
from src.services.data_io import load_processed, save_processed
from src.utils.log_utils import get_logger
from lib.feature_utils import preferred_column_order
from src.services.analytics.entity_resolution import create_entity_mapping

LOGGER = get_logger("feature_formulation")
FEATURE_MASTER_FILENAME = "feature_master_" + datetime.now().strftime("%Y%m%d_%H%M")


def _rank_entity_map(emap: pd.DataFrame) -> pd.DataFrame:
    """
    If multiple mapping rows exist per license_id, prefer stronger methods and higher scores.
    """
    # tuning from config
    er_config = load_config_from_file("config/er_config.toml")
    method_rank = er_config["entity_resolution"]["method_rank"]
    emap = emap.assign(
        _mrank=emap["method"]
        .replace("+", "_")  # method has "+", replace with "_"
        .map(method_rank)
        .fillna(1)
        .astype(int)
    )
    emap = (
        emap.sort_values(["license_id", "_mrank", "match_score"], ascending=[True, False, False])
        .drop_duplicates("license_id", keep="first")
        .drop(columns=["_mrank"])
    )
    return emap


def build_features(base_filename: str, imdb_basics_filename: str, imdb_ratings_filename: str, imdb_akas_filename: str):
    # ---- load base (synthetic)
    base = load_processed(base_filename)
    ensure_columns(base, ["license_id", "title", "release_year"], "Synthetic base")
    LOGGER.info("Synthetic base file read and validated")

    # Normalized title
    base["title_norm"] = normalize_title(base["title"])
    LOGGER.info("Synthetic base file: title normalized")

    # ---- load mapping (from your current ER pipeline)
    emap, emap_filename = create_entity_mapping(
        base_filename,
        imdb_basics_filename,
        imdb_ratings_filename,
        imdb_akas_filename
    )
    LOGGER.info(f"Entity Mapping file name {emap_filename}")
    ensure_columns(emap, ["license_id", "tconst", "method", "match_score"], "Entity mapping")
    # rank entity map
    emap = _rank_entity_map(emap)
    LOGGER.info("Entity mapping file read and ranked")

    # ---- load IMDb core
    imdb = load_imdb_core(imdb_basics_filename, imdb_ratings_filename)
    imdb = imdb[
        [
            "tconst",
            "primaryTitle",
            "primary_title_norm",
            "releaseYear",
            "titleType",
            "averageRating",
            "numVotes",
        ]
    ].drop_duplicates("tconst")
    LOGGER.info("IMDB core file loaded")

    ## Joins
    # base and mapping
    final_df = base.merge(
        emap[["license_id", "tconst", "method", "match_score"]],
        on="license_id",
        how="left",
    )

    # add IMDB
    final_df = final_df.merge(imdb, on="tconst", how="left")

    # ---- features
    # Prefer IMDb title when available; otherwise retain synthetic display name
    final_df["title_final"] = final_df["primaryTitle"].fillna(final_df["title"])

    # IMDb transforms
    final_df["log1p_numVotes"] = np.log1p(final_df["numVotes"].fillna(0))

    # Coverage flags
    final_df["has_entity"] = final_df["tconst"].notna().astype(int)

    # order and write
    final_df = final_df[preferred_column_order(final_df)]

    LOGGER.info(
        f"Rows={len(final_df)} "
        f"| ER hit={(100 * final_df['has_entity'].mean()): .1f} "
        f"| writing → {FEATURE_MASTER_FILENAME}")
    save_processed(final_df, FEATURE_MASTER_FILENAME)
