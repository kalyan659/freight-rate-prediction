"""Feature construction. `FeatureBuilder.fit` learns statistics from the training fold only;
`transform` applies them to any frame (training fold, test fold, validation.csv, December)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from . import config as C
from .data import repair_inputs

EQUIPMENT = ["Dry Van", "Flatbed", "Reefer"]


def log_rpm(df: pd.DataFrame) -> pd.Series:
    """Model target: log(rate per mile). Rates scale ~linearly with distance, so per-mile in
    log space gives a well-behaved, roughly symmetric target and multiplicative errors."""
    return np.log(df[C.TARGET] / df["distance"])


def to_rate(pred_log_rpm, distance) -> np.ndarray:
    return np.exp(np.asarray(pred_log_rpm)) * np.asarray(distance, dtype=float)


def haversine_miles(lat1, lon1, lat2, lon2):
    r = 3958.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlmb = p2 - p1, np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


class FeatureBuilder:
    """Learned pieces (all fit on the training fold):
    - median weight per equipment (imputation)
    - smoothed city effects for pickup and delivery: mean of log-rpm residual after an
      equipment x distance-band baseline. Training rows get out-of-fold values, so a row's own
      label never feeds its own feature. Unseen cities get 0 (the overall mean).
    The daily market level is passed in (it is built from input features, not labels).
    """

    def __init__(self, market_level: pd.Series, smoothing: float = 30.0, n_folds: int = 5):
        self.market_level = market_level
        self.smoothing = smoothing
        self.n_folds = n_folds

    # ---- city effects -------------------------------------------------------------
    @staticmethod
    def _band(d):
        return pd.cut(d["distance"], C.DISTANCE_BANDS, labels=False)

    def _base_residual(self, d: pd.DataFrame) -> pd.Series:
        y = log_rpm(d)
        return y - y.groupby([d["equipment"], self._band(d)]).transform("mean")

    def _encode(self, fit_df, res, col):
        g = res.groupby(fit_df[col]).agg(["sum", "count"])
        return g["sum"] / (g["count"] + self.smoothing)

    def fit(self, train: pd.DataFrame) -> "FeatureBuilder":
        train = repair_inputs(train)
        self.weight_median = train.groupby("equipment")["weight"].median()
        self.weight_median_all = float(train["weight"].median())
        res = self._base_residual(train)
        self.city_maps = {c: self._encode(train, res, c) for c in ("pickup", "delivery")}
        # out-of-fold encodings for the training rows themselves
        oof = {c: pd.Series(0.0, index=train.index) for c in ("pickup", "delivery")}
        kf = KFold(self.n_folds, shuffle=True, random_state=C.SEED)
        for fit_idx, enc_idx in kf.split(train):
            f, e = train.iloc[fit_idx], train.iloc[enc_idx]
            r = self._base_residual(f)
            for c in oof:
                oof[c].iloc[enc_idx] = e[c].map(self._encode(f, r, c)).fillna(0.0).to_numpy()
        self._train_index = train.index
        self._train_oof = oof
        return self

    # ---- transform ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame, is_training_fold: bool = False) -> pd.DataFrame:
        d = repair_inputs(df)
        X = pd.DataFrame(index=d.index)

        w = d["weight"].fillna(d["equipment"].map(self.weight_median)).fillna(self.weight_median_all)
        X["weight_k"] = w / 1000.0
        X["weight_missing"] = d["weight_missing"]
        X["weight_capped"] = d["weight_capped"]

        X["log_distance"] = np.log(d["distance"])
        X["log_distance_sq"] = X["log_distance"] ** 2
        X["inv_distance"] = 100.0 / d["distance"]
        for e in EQUIPMENT:
            X[f"eq_{e.replace(' ', '_').lower()}"] = (d["equipment"] == e).astype(int)

        day_level = d[C.DATE].map(self.market_level)
        X["market_day"] = day_level
        X["market_load"] = d["market_index"].fillna(day_level)
        X["market_index_missing"] = d["market_index_missing"]
        X["quote_signal"] = d["quote_signal"]

        X["trend_days"] = (d[C.DATE] - pd.Timestamp(C.TREND_ORIGIN)).dt.days
        dow = d[C.DATE].dt.dayofweek
        for k in range(7):
            X[f"dow_{k}"] = (dow == k).astype(int)
        X["day_of_month"] = d[C.DATE].dt.day

        X["pickup_lat"], X["pickup_lon"] = d["pickup_lat"], d["pickup_lon"]
        X["delivery_lat"], X["delivery_lon"] = d["delivery_lat"], d["delivery_lon"]
        hav = haversine_miles(d["pickup_lat"], d["pickup_lon"], d["delivery_lat"], d["delivery_lon"])
        X["circuity"] = d["distance"] / np.maximum(hav, 1.0)

        for c in ("pickup", "delivery"):
            if is_training_fold:
                X[f"{c}_effect"] = self._train_oof[c].reindex(d.index).to_numpy()
            else:
                X[f"{c}_effect"] = d[c].map(self.city_maps[c]).fillna(0.0).astype(float)
        return X


# Column groups used by the models
LINEAR_COLS = [
    "log_distance", "log_distance_sq", "inv_distance", "eq_flatbed", "eq_reefer",
    "weight_k", "weight_missing", "weight_capped", "market_day", "trend_days",
    "dow_0", "dow_1", "dow_2", "dow_3", "dow_4", "dow_5",
    "pickup_effect", "delivery_effect",
]
TREND_COLS = ["market_day", "trend_days"]
