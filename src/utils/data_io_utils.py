import _io
import json
import os
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Dict
from urllib.parse import urlparse

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import requests

from src.config.env_loader import SETTINGS
from src.utils.log_utils import get_logger
from src.utils.s3_utils import (
    ensure_bucket,
    list_bucket_objects,
    load_bucket_object,
    write_dataframe_parquet,
    write_bucket_object,
    formulate_s3_uri,
)

LOGGER = get_logger("data_io_utils")


@dataclass
class RunInfo:
    run_id: str
    has_raw: bool = False
    has_feature_master: bool = False
    has_feature_master_cleaned: bool = False
    has_model: bool = False


# -----------------------------
# Helpers: resolve where to write/read
# -----------------------------
def _is_s3() -> bool:
    return SETTINGS.IO_BACKEND == "S3"


def _ensure_buckets():
    # idempotent
    for b in filter(None, [
        SETTINGS.RAW_BUCKET, SETTINGS.PROCESSED_BUCKET, SETTINGS.PROFILES_BUCKET,
        SETTINGS.FIGURES_BUCKET, SETTINGS.MODELS_BUCKET, SETTINGS.REPORTS_BUCKET
    ]):
        ensure_bucket(b)


# -----------------------------
# Raw data
# -----------------------------
def save_raw(df: pd.DataFrame, base_dir: str, name: str) -> str:
    """
    Save raw dataset either to local or S3.
    base_dir acts as a subdirectory (LOCAL) or a prefix (S3).
    """
    if _is_s3():
        _ensure_buckets()
        key = f"{base_dir.strip('/')}/{name}.parquet"
        write_dataframe_parquet(df, SETTINGS.RAW_BUCKET, key, index=False)
        uri = formulate_s3_uri(SETTINGS.RAW_BUCKET, key)
        LOGGER.info(f"Saved raw (S3) → {uri}")
        return uri
    else:
        path = os.path.join(Path(SETTINGS.RAW_DIR) / base_dir, f"{name}.parquet")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        LOGGER.info(f"Saved raw (LOCAL) → {path}")
        return path


def load_raw(path_or_name: str, base_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw dataset. If S3, pass name without suffix + base_dir; if LOCAL, pass a full path.
    """
    if _is_s3():
        name = path_or_name if path_or_name.endswith(".parquet") else f"{path_or_name}.parquet"
        key = f"{base_dir.strip('/')}/{name}" if base_dir else name
        LOGGER.info(f"Loading raw (S3) ← s3://{SETTINGS.RAW_BUCKET}/{key}")
        return load_bucket_object(SETTINGS.RAW_BUCKET, key)
    else:
        path = path_or_name
        LOGGER.info(f"Loading raw (LOCAL) ← {path}")
        return pd.read_parquet(path)


def list_raw_files(base_dir: str) -> List[str]:
    if _is_s3():
        prefix = f"{base_dir.strip('/')}/"
        keys = [k for k in list_bucket_objects(SETTINGS.RAW_BUCKET, prefix) if k.endswith(".parquet")]
        keys.sort()
        LOGGER.info(f"Listing raw (S3) {SETTINGS.RAW_BUCKET}/{prefix} => {keys}")
        return keys
    else:
        raw = Path(SETTINGS.RAW_DIR) / base_dir
        if not os.path.isdir(raw):
            return []
        files = [f for f in os.listdir(raw) if f.endswith(".parquet")]
        files.sort()
        LOGGER.info(f"Listing raw (LOCAL) {raw} => {files}")
        return files


def _read_parquet_head(path_or_file, n_rows: int = 50000, columns=None) -> pd.DataFrame:
    """
    Read only the first n_rows rows from a Parquet file efficiently.
    Accepts a local path, S3 URL, or a file-like object.
    """
    pf = pq.ParquetFile(path_or_file)

    tables = []
    rows_read = 0

    for rg_idx in range(pf.num_row_groups):
        table = pf.read_row_group(rg_idx, columns=columns)
        tables.append(table)
        rows_read += table.num_rows

        if rows_read >= n_rows:
            break

    if not tables:
        return pd.DataFrame()

    combined = pa.concat_tables(tables)
    combined = combined.slice(0, min(n_rows, combined.num_rows))
    return combined.to_pandas()


def _read_parquet_head_from_url(url: str, n_rows: int = 50000, columns=None) -> pd.DataFrame:
    with fsspec.open(url, "rb") as f:
        return _read_parquet_head(f, n_rows, columns)


def save_from_url(url: str, base_dir: str, cap_n_rows: int = 50000) -> str:
    """
    Read from a remote URL (parquet, tsv/tsv.gz, or generic CSV),
    optionally sampling only the first `sample_rows` rows, and save as raw parquet.
    """
    try:
        LOGGER.info(f"Reading data from URL: {url}")

        # Strip query params for suffix detection and basename
        parsed = urlparse(url)
        path_only = parsed.path  # e.g. '/datasets/title.basics.tsv.gz'

        lower_path = path_only.lower()
        base_name = Path(path_only).stem.replace(".tsv", "").replace(".gz", "") or "remote"

        if lower_path.endswith((".parquet", ".pq")):
            df = _read_parquet_head_from_url(url, n_rows=cap_n_rows)
        elif lower_path.endswith((".tsv.gz", ".tsv")):
            LOGGER.info("Parsing TSV data from URL")
            df = pd.read_csv(
                url,
                sep="\t",
                compression="infer",
                na_values="\\N",
                low_memory=False,
                nrows=cap_n_rows
            )
        else:
            LOGGER.info("Parsing text data from URL")
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(
                _io.StringIO(r.text),
                low_memory=False,
                nrows=cap_n_rows
            )

        return save_raw(df, base_dir, base_name)
    except Exception as ex:
        error_text = f"Error reading file from {url}"
        LOGGER.exception(error_text)
        raise ValueError(error_text) from ex


# -----------------------------
# Processed data
# -----------------------------
def save_processed(df: pd.DataFrame, base_dir: str, name: str) -> str:
    if _is_s3():
        _ensure_buckets()
        key = f"{base_dir.strip('/')}/{name}.parquet"
        write_dataframe_parquet(df, SETTINGS.PROCESSED_BUCKET, key, index=False)
        uri = formulate_s3_uri(SETTINGS.PROCESSED_BUCKET, key)
        LOGGER.info(f"Saved processed (S3) → {uri}")
        return uri
    else:
        path = os.path.join(Path(SETTINGS.PROCESSED_DIR) / base_dir, f"{name}.parquet")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        LOGGER.info(f"Saved processed (LOCAL) → {path}")
        return path


def load_processed(name: str, base_dir: str = None) -> pd.DataFrame:
    if _is_s3():
        name = name if name.endswith(".parquet") else f"{name}.parquet"
        key = f"{base_dir.strip('/')}/{name}" if base_dir else name
        LOGGER.info(f"Loading processed (S3) ← s3://{SETTINGS.PROCESSED_BUCKET}/{key}")
        return load_bucket_object(SETTINGS.PROCESSED_BUCKET, key)
    else:
        name = name if name.endswith(".parquet") else f"{name}.parquet"
        path = os.path.join(Path(SETTINGS.PROCESSED_DIR) / base_dir, name)
        LOGGER.info(f"Loading processed (LOCAL) ← {path}")
        return pd.read_parquet(path)


# -----------------------------
# Figures
# -----------------------------
def save_figure(fig, base_dir: str, name: str) -> str:
    """
    Save a matplotlib figure either locally or to S3.

    Returns:
        LOCAL → filesystem path as str
        S3    → s3://bucket/key URI
    """
    # normalise name so we don't end up with .png.png
    if name.lower().endswith(".png"):
        base_name = name
    else:
        base_name = f"{name}.png"

    if _is_s3():
        _ensure_buckets()

        buf = BytesIO()
        fig.savefig(buf, bbox_inches="tight")
        buf.seek(0)

        key = f"{base_dir.strip('/')}/{base_name}"
        write_bucket_object(
            SETTINGS.FIGURES_BUCKET,
            key,
            buf.read(),
            content_type="image/png",
        )
        uri = formulate_s3_uri(SETTINGS.FIGURES_BUCKET, key)
        LOGGER.info(f"Saved figure (S3) → {uri}")
        return uri
    else:
        path = Path(SETTINGS.FIGURES_DIR) / base_dir / base_name
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        LOGGER.info(f"Saved figure (LOCAL) → {path}")
        return str(path)


def load_figure(path_or_uri: str):
    """
    Load a figure for display in Streamlit.

    LOCAL:
        returns a filesystem path (Streamlit will open it)
    S3:
        returns raw bytes that st.image can render
    """
    if isinstance(path_or_uri, str) and path_or_uri.startswith("s3://"):
        try:
            with fsspec.open(path_or_uri, "rb") as f:
                data = f.read()
            return data
        except Exception as ex:
            LOGGER.exception(f"Error loading figure from {path_or_uri}: {ex}")
            raise
    # for local paths, just return the string
    return path_or_uri


# -----------------------------
# Profiles / validation summaries
# -----------------------------
def save_profile(profile_obj: dict, base_dir: str, name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{name}_{timestamp}.json"
    if _is_s3():
        _ensure_buckets()
        key = f"{base_dir.strip('/')}/{filename}"
        write_bucket_object(
            SETTINGS.PROFILES_BUCKET,
            key,
            json.dumps(profile_obj, indent=2),
            content_type="application/json"
        )
        uri = formulate_s3_uri(SETTINGS.PROFILES_BUCKET, key)
        LOGGER.info(f"Saved profile (S3) → {uri}")
        return uri
    else:
        path = os.path.join(Path(SETTINGS.PROFILES_DIR) / base_dir, filename)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(profile_obj, f, indent=2)
        LOGGER.info(f"Saved profile (LOCAL) → {path}")
        return path


# -----------------------------
# Latest file under (LOCAL or S3)
# -----------------------------
def latest_file_under_directory(
        prefix: str,
        under_dir: Path | None = None,
        suffix: str = ".parquet",
        exclusion: str = None
) -> Optional[str]:
    if _is_s3():
        # Default to processed bucket / root if no base provided
        base_prefix = "" if under_dir is None else str(under_dir).strip("/")
        search = f"{base_prefix}/{prefix}".strip("/")
        keys = [k for k in list_bucket_objects(SETTINGS.PROCESSED_BUCKET, search)
                if k.startswith(search) and k.endswith(suffix) and (exclusion is None or exclusion not in k)]
        if not keys:
            LOGGER.warning(f"No keys under s3://{SETTINGS.PROCESSED_BUCKET}/{search}")
            return None
        keys.sort(reverse=True)
        latest = keys[0]
        return formulate_s3_uri(SETTINGS.PROCESSED_BUCKET, latest)
    else:
        if under_dir is None:
            LOGGER.warning("under_dir is required for LOCAL backend")
            return None
        if not under_dir.exists():
            LOGGER.warning(f"Directory {under_dir.name} doesn't exist")
            return None
        files = [p for p in under_dir.iterdir()
                 if p.is_file()
                 and p.name.startswith(prefix)
                 and p.suffix == suffix
                 and (exclusion is None or exclusion not in p.name)]
        if not files:
            LOGGER.warning(f"No files found under {under_dir.name} directory")
            return None
        files.sort(key=lambda p: p.name, reverse=True)
        return str(files[0])


# -----------------------------
# List run
# -----------------------------
def list_runs() -> List[RunInfo]:
    """
    Collect run-level metadata for all runs, abstracting over LOCAL vs S3.

    - In LOCAL mode:
        RAW_DIR/<run_id>/...
        PROCESSED_DIR/<run_id>/feature_master*.parquet
        MODELS_DIR/<run_id>/...

    - In S3 mode:
        RAW_BUCKET:        <run_id>/...
        PROCESSED_BUCKET:  <run_id>/feature_master*.parquet
        MODELS_BUCKET:     <run_id>/...
    """
    info_by_id: Dict[str, RunInfo] = {}

    def _get(run_id: str) -> RunInfo:
        if run_id not in info_by_id:
            info_by_id[run_id] = RunInfo(run_id=run_id)
        return info_by_id[run_id]

    if _is_s3():
        # ---------- RAW ----------
        if SETTINGS.RAW_BUCKET:
            for key in list_bucket_objects(SETTINGS.RAW_BUCKET, prefix=""):
                parts = key.split("/")
                if len(parts) < 2:
                    continue
                run_id = parts[0]
                _get(run_id).has_raw = True

        # ---------- PROCESSED (feature master) ----------
        if SETTINGS.PROCESSED_BUCKET:
            for key in list_bucket_objects(SETTINGS.PROCESSED_BUCKET, prefix=""):
                parts = key.split("/")
                if len(parts) < 2:
                    continue
                run_id, filename = parts[0], parts[-1]
                info = _get(run_id)
                if filename.startswith("feature_master_cleaned_") and filename.endswith(".parquet"):
                    info.has_feature_master_cleaned = True
                elif filename.startswith("feature_master_") and filename.endswith(".parquet"):
                    # Avoid double-counting cleaned as raw
                    if "cleaned" not in filename:
                        info.has_feature_master = True

        # ---------- MODELS ----------
        if getattr(SETTINGS, "MODELS_BUCKET", None):
            for key in list_bucket_objects(SETTINGS.MODELS_BUCKET, prefix=""):
                parts = key.split("/")
                if len(parts) < 2:
                    continue
                run_id = parts[0]
                _get(run_id).has_model = True

    else:
        raw_root = Path(SETTINGS.RAW_DIR)
        proc_root = Path(SETTINGS.PROCESSED_DIR)
        models_root = Path(getattr(SETTINGS, "MODELS_DIR", "")) if getattr(SETTINGS, "MODELS_DIR", None) else None

        # ---------- RAW ----------
        if raw_root.exists():
            for run_dir in raw_root.iterdir():
                if run_dir.is_dir() and any(run_dir.iterdir()):
                    _get(run_dir.name).has_raw = True

        # ---------- PROCESSED (feature master) ----------
        if proc_root.exists():
            for run_dir in proc_root.iterdir():
                if not run_dir.is_dir():
                    continue
                run_id = run_dir.name
                info = _get(run_id)

                fm_clean = list(run_dir.glob("feature_master_cleaned_*.parquet"))
                fm_raw = [
                    p for p in run_dir.glob("feature_master_*.parquet")
                    if "cleaned" not in p.name
                ]

                if fm_clean:
                    info.has_feature_master_cleaned = True
                if fm_raw:
                    info.has_feature_master = True

        # ---------- MODELS ----------
        if models_root and models_root.exists():
            for run_dir in models_root.iterdir():
                if run_dir.is_dir() and any(run_dir.iterdir()):
                    _get(run_dir.name).has_model = True

    infos = list(info_by_id.values())
    # Keep same ordering semantics as your old _list_runs
    infos.sort(key=lambda ri: ri.run_id, reverse=True)
    LOGGER.info(
        "Run infos (%s backend) → %s",
        "S3" if _is_s3() else "LOCAL",
        [(ri.run_id, ri.has_raw, ri.has_feature_master, ri.has_feature_master_cleaned, ri.has_model) for ri in infos],
    )
    return infos


# -----------------------------
# Model artifacts
# -----------------------------
def model_run_exists(run_id: str) -> bool:
    """
    True if there is *any* model artifact for this run_id,
    abstracting over LOCAL vs S3.
    """
    if _is_s3():
        bucket = getattr(SETTINGS, "MODELS_BUCKET", None)
        if not bucket:
            LOGGER.warning("MODELS_BUCKET is not configured; model_run_exists → False")
            return False
        prefix = f"{run_id.strip('/')}/"
        keys = list_bucket_objects(bucket, prefix=prefix)
        exists = len(keys) > 0
        LOGGER.info(f"model_run_exists(S3) run_id={run_id} → {exists} (keys={len(keys)})")
        return exists
    else:
        models_dir = Path(SETTINGS.MODELS_DIR) / run_id
        exists = models_dir.exists() and any(models_dir.iterdir())
        LOGGER.info(f"model_run_exists(LOCAL) run_id={run_id} → {exists}")
        return exists


def load_model_csv(run_id: str, filename: str) -> Optional[pd.DataFrame]:
    """
    Load a CSV model artifact (predictions, per-model metrics) for a run_id.
    Returns None if the file doesn't exist.
    """
    if _is_s3():
        bucket = getattr(SETTINGS, "MODELS_BUCKET", None)
        if not bucket:
            LOGGER.warning("MODELS_BUCKET not configured; load_model_csv → None")
            return None
        key = f"{run_id.strip('/')}/{filename}"
        try:
            text = load_bucket_object(bucket, key, as_text=True)
            return pd.read_csv(_io.StringIO(text))
        except FileNotFoundError:
            LOGGER.info(f"load_model_csv(S3) missing: s3://{bucket}/{key}")
            return None
        except Exception as ex:
            LOGGER.exception(f"Error loading model CSV s3://{bucket}/{key}: {ex}")
            raise
    else:
        path = Path(SETTINGS.MODELS_DIR) / run_id / filename
        if not path.exists():
            LOGGER.info(f"load_model_csv(LOCAL) missing: {path}")
            return None
        return pd.read_csv(path)


def load_model_json(run_id: str, filename: str) -> Optional[dict]:
    """
    Load a JSON model artifact (ensemble summaries, params map) for a run_id.
    Returns None if the file doesn't exist.
    """
    if _is_s3():
        bucket = getattr(SETTINGS, "MODELS_BUCKET", None)
        if not bucket:
            LOGGER.warning("MODELS_BUCKET not configured; load_model_json → None")
            return None
        key = f"{run_id.strip('/')}/{filename}"
        try:
            text = load_bucket_object(bucket, key, as_text=True)
            return json.loads(text)
        except FileNotFoundError:
            LOGGER.info(f"load_model_json(S3) missing: s3://{bucket}/{key}")
            return None
        except Exception as ex:
            LOGGER.exception(f"Error loading model JSON s3://{bucket}/{key}: {ex}")
            raise
    else:
        path = Path(SETTINGS.MODELS_DIR) / run_id / filename
        if not path.exists():
            LOGGER.info(f"load_model_json(LOCAL) missing: {path}")
            return None
        with open(path, "r") as f:
            return json.load(f)
