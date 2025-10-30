import logging
from datetime import datetime
from typing import Dict, Tuple

import pandas as pd
import pandera as pa
from pandera import DataFrameSchema, Column, Check

LOGGER = logging.getLogger("schema_validator")

# ---- 1) Column name normalization (accept tconst or title_id) ----
_COLUMN_SYNONYMS: Dict[str, str] = {
    "tconst": "title_id",
    "imdb_id": "title_id",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {src: dst for src, dst in _COLUMN_SYNONYMS.items() if src in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)
        LOGGER.info(f"Normalized columns: {rename_map}")
    return df


# ---- 2) Canonical schema (use only pandera DataFrameSchema/Column API) ----
# Note: use pandera-native dtypes via Python types; add Checks as needed.
_BASE_SCHEMA = DataFrameSchema(
    {
        "title_id": Column(str, nullable=True, required=False),
        "title": Column(str, nullable=True, required=False),
        "territory": Column(str, nullable=True, required=False),
        "media": Column(str, nullable=True, required=False),
        "platform": Column(str, nullable=True, required=False),
        "release_year": Column(int, nullable=True, required=False),
        "window_start":Column(datetime, nullable=True, required=False),
        "window_end": Column(datetime, nullable=True, required=False),
        "price": Column(float, nullable=True, required=False, checks=Check.gt(0)),
    },
    coerce=True,  # attempt to cast values to declared types
    strict=False,  # allow extra columns beyond those listed
)


def _partial_schema_for(df: pd.DataFrame) -> DataFrameSchema:
    """Use only columns that actually exist in df to avoid over-strict failures early on."""
    cols_present = {c: col for c, col in _BASE_SCHEMA.columns.items() if c in df.columns}
    return DataFrameSchema(cols_present, coerce=True, strict=False)


# ---- 3) Validation entrypoint returning (validated_df, report) ----
def validate_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Validate df against the canonical schema (after normalizing synonyms).
    Returns:
        validated_df: DataFrame (possibly coerced) if validation passes; original df if it fails
        report: {"status": "passed"|"failed", "errors": [ {column, check, failure_case}, ... ]}
    """
    report: Dict = {"status": "passed", "errors": []}

    # Normalize synonyms like tconst -> title_id
    df_norm = normalize_columns(df)

    # Build schema only with columns present
    schema = _partial_schema_for(df_norm)

    try:
        validated = schema.validate(df_norm, lazy=True)
        LOGGER.info("Schema validation passed.")
        return validated, report

    except pa.errors.SchemaErrors as e:
        # Produce a friendly, structured report for the UI
        try:
            failures = e.failure_cases[["column", "check", "failure_case"]].to_dict(orient="records")
        except Exception:
            failures = [{"error": str(e)}]
        report["status"] = "failed"
        report["errors"] = failures
        LOGGER.warning(f"Schema validation failed with {len(failures)} issue(s).")
        # Return original df (not partially coerced) so downstream can decide what to do
        return df, report
