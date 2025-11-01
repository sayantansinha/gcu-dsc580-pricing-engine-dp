from datetime import datetime

import pandas as pd
from pandas import DataFrame

from config.config_loader import load_config_from_file
from lib.ent_res_utils import resolve_entities, id_entity_map
from lib.imdb_utils import load_imdb_core, load_imdb_akas, imdb_candidate_universe
from lib.io_utils import ensure_columns
from lib.title_utils import normalize_title
from src.services.data_io import load_processed, save_processed
from src.utils.log_utils import get_logger

# Logger setup
LOGGER = get_logger("entity_resolution")

# SYNTHETIC_BASE_FILENAME = "cleaned_synthetic_file_20251029-232604_aa85f67a"
# IMDB_BASICS_FILENAME = "cleaned_imdb_basics_file_20251029-234402_e26346aa"
# IMDB_RATINGS_FILENAME = "cleaned_imdb_rating_file_20251030-000026_3650dc61"
# IMDB_AKAS_FILENAME = "cleaned_imdb_akas_file_20251030-000457_6ce2d0f2"
ENTITY_MAPPING_FILENAME = "feature_dataset_entity_mapping_" + datetime.now().strftime("%Y%m%d_%H%M")


def create_entity_mapping(
        base_filename: str,
        imdb_basics_filename: str,
        imdb_ratings_filename: str,
        imdb_akas_filename: str
) -> tuple[DataFrame, str]:
    # tuning from config
    er_config = load_config_from_file("config/er_config.toml")
    contingency_year = er_config["entity_resolution"]["contingency_year"]
    min_similarity_score = er_config["entity_resolution"]["min_similarity_score"]
    country_iso2_code_map = er_config["entity_resolution"]["iso3_to_iso2_map"]

    # load base
    base = load_processed(base_filename)
    ensure_columns(base, ["license_id", "title", "release_year"], "Synthetic base")
    base["title_norm"] = normalize_title(base["title"])

    # IMDb candidates
    LOGGER.info("Loading IMDB Core....")
    core = load_imdb_core(imdb_basics_filename, imdb_ratings_filename)
    LOGGER.info("IMDB Core loaded successfully")
    LOGGER.info("Loading IMDB AKAs....")
    akas = load_imdb_akas(imdb_akas_filename)
    LOGGER.info("IMDB AKAs loaded successfully")
    LOGGER.info("Formulating IMDB Candidate Universe....")
    cand = imdb_candidate_universe(core, akas)
    LOGGER.info("IMDB Candidate Universe populated successfully")

    # First: ID-first deterministic mapping
    LOGGER.info("ID Mapping in-progress....")
    mapping = id_entity_map(base, core)
    LOGGER.info(f"ID Mapping completed successfully, mapped {len(mapping)} records")

    # Second: For rows still unmatched - year and territory mapping
    matched_ids = set(mapping["license_id"])
    base_left = base[~base["license_id"].isin(matched_ids)]

    if not base_left.empty:
        LOGGER.info(f"{len(base_left)} un-matched rows exists, Year and Territory mapping in-progress....")
        er_map = resolve_entities(
            base_left,
            cand,
            contingency_year,
            min_similarity_score,
            country_iso2_code_map
        )
        mapping = pd.concat([mapping, er_map], ignore_index=True)

    LOGGER.info("Mapping created successfully")
    hit = 100 * mapping["tconst"].notna().mean()
    LOGGER.info(
        f"Mapping stats: Total Count = {len(mapping)} -> ID Match Count = {(mapping['method'] == 'id_exact').sum()}, "
        f"Year and Territory Match Count = {(mapping['method'] != 'id_exact').sum()}, "
        f"Total Hit = {hit: .1f}")

    save_processed(mapping, ENTITY_MAPPING_FILENAME)

    return mapping, ENTITY_MAPPING_FILENAME
