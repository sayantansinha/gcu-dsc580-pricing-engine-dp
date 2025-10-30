import os
from typing import Dict, Optional, Tuple, List
import pandas as pd

OUT_DIR = "data/public/imdb/processed"
os.makedirs(OUT_DIR, exist_ok=True)

def _read_tsv_like(url: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    return pd.read_csv(
        url,
        sep="\t",
        compression="infer",
        na_values="\\N",
        usecols=usecols,
        low_memory=False
    )

def _normalize_title(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.normalize("NFKC")
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )

def load_imdb_from_urls(urls: List[str]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Accepts arbitrary list of IMDb TSV/TSV.GZ URLs.
    Detects basics/ratings/akas by filename and merges what’s available.
    Returns:
      merged_core_df,
      paths_written: dict of {name: local_parquet_path}
    """
    if not urls:
        raise ValueError("No URLs provided.")

    urls_by_kind: Dict[str, str] = {}
    for u in urls:
        lu = u.lower()
        if "title.basics" in lu:
            urls_by_kind["basics"] = u
        elif "title.ratings" in lu:
            urls_by_kind["ratings"] = u
        elif "title.akas" in lu:
            urls_by_kind["akas"] = u

    # Fallback: if user gives a single non-standard file, just read it raw
    if not urls_by_kind and len(urls) == 1:
        df = _read_tsv_like(urls[0])
        path = os.path.join(OUT_DIR, "single_input.parquet")
        df.to_parquet(path, index=False)
        return df, {"single_input": path}

    basics = ratings = akas = None

    if "basics" in urls_by_kind:
        basics = _read_tsv_like(
            urls_by_kind["basics"],
            usecols=["tconst", "primaryTitle", "startYear", "titleType"]
        )
        basics["primaryTitle_norm"] = _normalize_title(basics["primaryTitle"])
        basics["release_year"] = pd.to_numeric(basics["startYear"], errors="coerce").astype("Int64")
        basics.drop(columns=["startYear"], inplace=True)

    if "ratings" in urls_by_kind:
        ratings = _read_tsv_like(
            urls_by_kind["ratings"],
            usecols=["tconst", "averageRating", "numVotes"]
        )

    if "akas" in urls_by_kind:
        akas = _read_tsv_like(
            urls_by_kind["akas"],
            usecols=["titleId", "title", "region", "language", "isOriginalTitle"]
        )
        akas["title_norm"] = _normalize_title(akas["title"])
        # prefer original titles
        akas = akas.sort_values(["titleId", "isOriginalTitle"], ascending=[True, False])
        akas = akas.groupby(["titleId", "region"], as_index=False).first()
        akas.rename(columns={"titleId": "tconst"}, inplace=True)

    # Merge what we have
    core = None
    if basics is not None and ratings is not None:
        core = basics.merge(ratings, on="tconst", how="left")
    elif basics is not None:
        core = basics
    elif ratings is not None:
        core = ratings
    # (If only akas was supplied, we treat it as core for the UI)
    elif akas is not None:
        core = akas

    if core is None:
        raise ValueError("Provided URLs did not match recognized IMDb files, and no single file fallback was possible.")

    # Write outputs
    paths_written: Dict[str, str] = {}
    core_path = os.path.join(OUT_DIR, "title_core.parquet")
    core.to_parquet(core_path, index=False)
    paths_written["title_core"] = core_path

    if akas is not None:
        akas_path = os.path.join(OUT_DIR, "title_akas.parquet")
        akas.to_parquet(akas_path, index=False)
        paths_written["title_akas"] = akas_path

    return core, paths_written
