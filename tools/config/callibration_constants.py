from typing import Final
import numpy as np
from numpy.random import Generator

from tools.config.config_loader import load_config_from_file

# Config from TOML file
_CAL_CFG = load_config_from_file("config/sim_config.toml")

# Simulation config
SEED: Final[int] = _CAL_CFG["general"]["seed"]
GEN: Final[Generator] = np.random.default_rng(SEED)
N_LICENSES: Final[int] = _CAL_CFG["general"]["n_licenses"]
BASE_CURRENCY = _CAL_CFG["general"]["base_currency"]


# normalize function
def _normalize(v: np.ndarray) -> np.ndarray:
    s = float(v.sum())
    if s <= 0:
        raise ValueError("weights sum must be > 0")
    v = v / s
    return v


# Territory and weights
TERRITORIES: Final[list[str]] = list(_CAL_CFG.get("territory_weights").keys())
terr_weights = np.array(list(_CAL_CFG["territory_weights"].values()), dtype=float)
TERRITORY_WEIGHTS: Final[np.ndarray] = _normalize(terr_weights)

# Medias and Platforms
MEDIAS: Final[list[str]] = list(_CAL_CFG.get("media_weights").keys())
media_wts = np.array(list(_CAL_CFG["media_weights"].values()), dtype=float)
MEDIA_WEIGHTS: Final[np.ndarray] = _normalize(media_wts)


# Platforms
def load_platform_by_weight_config():
    pbt = _CAL_CFG.get("platform_weights_by_media") or {}
    plat_keys = {}
    plat_w = {}
    for media in MEDIAS:
        block = pbt.get(media)
        if not block:
            raise KeyError(f"platform_weights_by_media.{media} missing")
        ks = list(block.keys())
        ws = _normalize(np.array([float(block[k]) for k in ks], dtype=float))
        plat_keys[media] = ks
        plat_w[media] = ws

    return plat_keys, plat_w


platform_keys, platform_weights = load_platform_by_weight_config()
PLATFORMS: Final[dict] = platform_keys
PLATFORM_WEIGHTS: Final[dict] = platform_weights

# Media window days (min or max)
MEDIA_WINDOW_DAYS_MIN_MAX: Final[dict] = _CAL_CFG["window_days_by_media"]
seasonality_month_weights = _CAL_CFG["seasonality_month_weight"]
norm_seasonality_month_weights = {}
if seasonality_month_weights and any(isinstance(k, str) for k in seasonality_month_weights.keys()):
    norm_seasonality_month_weights = {int(k): v for k, v in seasonality_month_weights.items()}
SEASONALITY_MONTH_WEIGHTS: Final[dict] = norm_seasonality_month_weights
PRICE_BY_TERR: Final[dict] = _CAL_CFG["price_by_territory"]

# Log normalization units
MU = _CAL_CFG["units_lognorm"]["mu"]
SIGMA = _CAL_CFG["units_lognorm"]["sigma"]

# New Release uplift config
HALF_LIFE = _CAL_CFG["new_release_uplift"]["half_life_days"]
MAX_U = _CAL_CFG["new_release_uplift"]["max_uplift"]

# Validation config
UNITS_MIN: Final[int] = _CAL_CFG["validation_config"]["units_min"]
PRICE_MIN: Final[int] = _CAL_CFG["validation_config"]["price_min"]
WIN_DAYS_MIN: Final[int] = _CAL_CFG["validation_config"]["window_days_min"]
WIN_DAYS_MAX: Final[int] = _CAL_CFG["validation_config"]["window_days_max"]
