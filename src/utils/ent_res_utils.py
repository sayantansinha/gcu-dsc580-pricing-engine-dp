"""
er_utils: entity resolution (record linkage) logic.
"""
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from src.utils.log_utils import get_logger

# Logger setup
LOGGER = get_logger("ent_res_utils")


def id_entity_map(base: pd.DataFrame, core: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic mapping using tconst in base['title_id']
    :param base: synthetic dataset
    :param core: IMDB core dataset
    :return ID matched dataframe
    """
    # exact ID join
    id_matched = base.merge(core[["tconst"]], left_on="title_id", right_on="tconst", how="left")
    id_matched = id_matched.dropna(subset=["tconst"])  # keep hits
    if id_matched.empty:
        return pd.DataFrame(columns=["license_id", "tconst", "method", "match_score"])

    out = id_matched[["license_id", "tconst"]].drop_duplicates()
    out["method"] = "id_exact"
    out["match_score"] = 1.0
    return out


def _metadata_block(
        cand: pd.DataFrame,
        year: int,
        iso3_terr: str,
        contingency: int,
        country_iso2_code_map: Dict[str, str]
) -> pd.DataFrame:
    blk = cand

    # year block
    if pd.notna(year):
        sy = pd.to_numeric(blk["releaseYear"], errors="coerce")
        blk = blk[sy.sub(int(year)).abs() <= contingency]

    # region block
    r = country_iso2_code_map.get(str(iso3_terr).upper())
    if r:
        blk = blk[(blk["region"].isna()) | (blk["region"] == r)]
    return blk


def _score_metadata(blk: pd.DataFrame, year: int) -> pd.Series | None:
    if blk.empty:
        return None
    b = blk.copy()
    # features
    b["yr_pen"] = 0.0
    if pd.notna(year):
        sy = pd.to_numeric(b["releaseYear"], errors="coerce")
        b["yr_pen"] = sy.sub(int(year)).abs().fillna(5)  # smaller is better

    # region bonus: exact region gets a small bump
    b["region_bonus"] = (~b["region"].isna()).astype(float) * 0.1

    # popularity prior (rank to 0..1)
    b["r_votes"] = b["numVotes"].fillna(0).rank(pct=True)

    # overall score: higher is better (invert year penalty)
    b["meta_score"] = (1.0 - (b["yr_pen"] / (b["yr_pen"].max() or 1.0)).clip(0, 1)) * 0.6 \
                      + b["r_votes"] * 0.3 \
                      + b["region_bonus"] * 0.1

    # pick highest meta_score, then numVotes
    return b.sort_values(["meta_score", "numVotes"], ascending=[False, False]).iloc[0]


def resolve_entities(
        base: pd.DataFrame,
        cand: pd.DataFrame,
        contingency_year: int,
        min_similarity_score: float,
        country_iso2_code_map: Dict[str, str]
) -> pd.DataFrame:
    """
    Resolve using release year and region (territory)
    Returns DataFrame[license_id, tconst, method, match_score, snapshot_date]
    """
    rows: List[Tuple[str, object, str, float]] = []

    # defensively ensure required columns exist (helps static analyzers, too)
    for df, name, cols in [
        (base, "base", ["license_id", "title_norm", "release_year"]),
        (cand, "cand", ["tconst", "title_norm", "releaseYear", "numVotes"]),
    ]:
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")

    LOGGER.debug("No missing columns found")

    for idx, (lic, terr, yr) in enumerate(base[["license_id", "territory", "release_year"]].itertuples(index=False)):
        block = _metadata_block(cand, yr, terr, contingency_year, country_iso2_code_map)
        best = _score_metadata(block, yr)

        if best is not None and best["meta_score"] >= min_similarity_score:
            mapping_rec = (lic, best["tconst"], "meta_year+region+votes", float(best["meta_score"]))
        else:
            mapping_rec = (lic, np.nan, "no_match", 0.0)

        rows.append(mapping_rec)
        LOGGER.debug(f"Row # {idx + 1} => Mapping Record [{mapping_rec}], Block [{block}]")
        if (idx + 1) % 100 == 0 or idx + 1 == len(base):
            LOGGER.info(
                f"Processed {idx + 1} records from base dataframe, running result rows size [{len(rows)}] "
                f"with {sum(1 for r in rows if r[2] == 'no_match')} records having no matches")

    out = pd.DataFrame(rows, columns=["license_id", "tconst", "method", "match_score"])
    out["snapshot_date"] = pd.Timestamp.utcnow().date().isoformat()
    return out
