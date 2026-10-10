"""Loading, input repairs (applied to every row) and the label filter (training rows only).

Repairs never drop rows, so every load that needs a prediction keeps one. The label filter
only ever touches labelled training data inside a fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def load_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[C.DATE] = pd.to_datetime(df[C.DATE])
    return df


def load_train() -> pd.DataFrame:
    return load_csv(C.TRAIN_FILE)


def load_validation() -> pd.DataFrame:
    return load_csv(C.VALIDATION_FILE)


def repair_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level fixes that need no learned statistics.

    - weight < 0: a sign error (the magnitudes match the normal weight distribution), so take |w|.
    - weight == cap: censored at 47,500 lb; keep the value and add a flag.
    - missing weight / market_index: add a flag; the value itself is imputed later from
      training-fold statistics (weight) or the same-day market level (market_index).
    """
    out = df.copy()
    out["weight_was_negative"] = (out["weight"] < 0).astype(int)
    out["weight"] = out["weight"].abs()
    out["weight_missing"] = out["weight"].isna().astype(int)
    out["weight_capped"] = np.isclose(out["weight"], C.WEIGHT_CAP).astype(int)
    out["market_index_missing"] = out["market_index"].isna().astype(int)
    return out


def segment_ratio(df: pd.DataFrame, reference: pd.DataFrame | None = None) -> pd.Series:
    """Rate-per-mile divided by the median rate-per-mile of the (equipment x distance band)
    segment, with medians taken from `reference` (defaults to df itself)."""
    reference = df if reference is None else reference

    def band(d):
        return pd.cut(d["distance"], C.DISTANCE_BANDS, labels=False)

    rpm_ref = reference[C.TARGET] / reference["distance"]
    med = rpm_ref.groupby([reference["equipment"], band(reference)]).median()
    key = pd.MultiIndex.from_arrays([df["equipment"], band(df)])
    seg_median = pd.Series(med.reindex(key).to_numpy(), index=df.index)
    return (df[C.TARGET] / df["distance"]) / seg_median


def clean_label_mask(df: pd.DataFrame, reference: pd.DataFrame | None = None) -> pd.Series:
    """True for rows whose label looks valid (positive and within the outlier limits)."""
    ratio = segment_ratio(df, reference)
    return (df[C.TARGET] > 0) & ratio.between(C.OUTLIER_LOW, C.OUTLIER_HIGH)


def city_table(*frames: pd.DataFrame) -> pd.DataFrame:
    """One row per city with its coordinates (they are constant per city in both files)."""
    parts = []
    for f in frames:
        for name, lat, lon in C.CITY_COLS:
            parts.append(f[[name, lat, lon]].set_axis(["city", "lat", "lon"], axis=1))
    return pd.concat(parts).drop_duplicates("city").set_index("city")


def daily_market_level(*frames: pd.DataFrame) -> pd.Series:
    """Mean market_index per date. market_index is an input feature (no target information),
    and each load's value is the day's market level plus ~0.025 noise, so the daily mean is the
    better signal and also fills dates where a row's value is missing."""
    allrows = pd.concat([f[[C.DATE, "market_index"]] for f in frames])
    return allrows.groupby(C.DATE)["market_index"].mean()
