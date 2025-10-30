from __future__ import annotations

import csv
import math
import uuid
from datetime import date, timedelta
from pathlib import Path
from numpy import ndarray

from config.callibration_constants import MEDIA_WINDOW_DAYS_MIN_MAX, SEASONALITY_MONTH_WEIGHTS, TERRITORIES, \
    TERRITORY_WEIGHTS, MEDIA_WEIGHTS, PLATFORM_WEIGHTS, N_LICENSES, MEDIAS, PLATFORMS, MAX_U, HALF_LIFE, \
    BASE_CURRENCY, PRICE_BY_TERR, GEN, MU, SIGMA
from src.utils.log_utils import get_logger

# Logger setup
LOGGER = get_logger("simulate_licensing_data")


def _pick(arr, probs):
    return GEN.choice(arr, p=probs)


def _randint(a: int, b: int) -> int:
    return int(GEN.integers(low=a, high=b + 1))  # inclusive


def _price_sample_usd(territory: str) -> float:
    median, iqr = PRICE_BY_TERR[territory]
    mu = math.log(median)
    sigma = 0.5 if iqr <= 0 else min(1.0, math.log(1 + iqr / median))
    return float(math.exp(GEN.normal(mu, sigma)))


def _units_sample() -> int:
    return int(max(0, round(GEN.lognormal(mean=MU, sigma=SIGMA))))


def _simulate(
        n_size: int,
        territories: list[str],
        t_weights: ndarray,
        medias: list[str],
        media_weights: ndarray,
        platforms: dict,
        p_weights: dict
) -> list[dict]:
    today = date.today()
    rows = []
    for _ in range(n_size):
        territory = _pick(territories, t_weights)
        media = _pick(medias, media_weights)
        platform = _pick(platforms[media], p_weights[media])

        wmin, wmax = MEDIA_WINDOW_DAYS_MIN_MAX[media]
        win_days = _randint(wmin, wmax)
        end = today - timedelta(days=_randint(0, 365))
        start = end - timedelta(days=win_days)

        title_id = f"tt{_randint(1_000_000, 9_999_999)}"
        release_year = _randint(2000, today.year)
        title = f"Title {release_year}-{_randint(1, 9999)}"

        u = _units_sample()
        base_price = _price_sample_usd(territory)
        season_mult = SEASONALITY_MONTH_WEIGHTS[start.month]
        recency_days = (today - date(release_year, 1, 1)).days
        recency_mult = 1 + MAX_U * math.exp(-math.log(2) * recency_days / max(1, HALF_LIFE))

        price = max(0.0, base_price * season_mult * recency_mult)
        price *= (1.0 - 0.12 * min(1.0, math.log1p(u) / 10))  # simple elasticity

        rows.append({
            "license_id": str(uuid.uuid4()),
            "title_id": title_id,
            "title": title,
            "release_year": release_year,
            "territory": territory,  # ISO alpha-3
            "media": media,
            "platform": platform,
            "window_start": start.isoformat(),
            "window_end": end.isoformat(),
            "units": u,
            "price": round(price, 2),
            "currency": BASE_CURRENCY,
        })

    return rows


def main():
    # Set output directory and file
    out_dir = Path("../data/source")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "licensing_deals.csv"

    # Generate synthetic dataset
    synthetic_data_rows = _simulate(
        N_LICENSES,
        TERRITORIES,
        TERRITORY_WEIGHTS,
        MEDIAS,
        MEDIA_WEIGHTS,
        PLATFORMS,
        PLATFORM_WEIGHTS
    )

    # Write to file
    with out_file.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(synthetic_data_rows[0].keys()))
        w.writeheader();
        w.writerows(synthetic_data_rows)
    LOGGER.info(f"Wrote {out_file}, rows={len(synthetic_data_rows)}")


if __name__ == "__main__":
    main()
