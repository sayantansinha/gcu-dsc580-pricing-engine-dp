import pandas as pd
from pandera import Column, DataFrameSchema

# Generic minimal schema (adapt as you finalize IMDb/synthetic fields)
generic_schema = DataFrameSchema({
    # Example core fields you often rely on
    "title_id": Column(str, nullable=True),        # IMDb tconst or your mapping
    "territory": Column(str, nullable=True),
    "license_type": Column(str, nullable=True),
    "platform": Column(str, nullable=True),
    "price": Column(float, nullable=True),
}, coerce=True)

def validate_df(df: pd.DataFrame) -> pd.DataFrame:
    # Apply only columns present to avoid over-strict failures in early iteration
    cols = [c for c in generic_schema.columns.keys() if c in df.columns]
    if not cols:
        return df
    partial_schema = DataFrameSchema({c: generic_schema.columns[c] for c in cols}, coerce=True)
    return partial_schema.validate(df, lazy=True)
