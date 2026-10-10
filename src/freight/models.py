"""Model candidates and the end-to-end estimator (raw loads -> dollars)."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .data import clean_label_mask, daily_market_level
from .features import LINEAR_COLS, FeatureBuilder, log_rpm, to_rate


class MedianBaseline:
    """Median log rate-per-mile per equipment x distance band. No time, no market."""
    name = "Median baseline (equipment x distance band)"

    def fit(self, X, y, raw):
        self.keys = ["equipment", "band"]
        k = self._key(raw)
        self.table = y.groupby([k[c] for c in self.keys]).median()
        self.eq_table = y.groupby(raw["equipment"]).median()
        self.global_median = float(y.median())
        return self

    def _key(self, raw):
        return pd.DataFrame({"equipment": raw["equipment"],
                             "band": pd.cut(raw["distance"], C.DISTANCE_BANDS, labels=False)},
                            index=raw.index)

    def predict(self, X, raw):
        k = self._key(raw)
        idx = pd.MultiIndex.from_frame(k)
        out = pd.Series(self.table.reindex(idx).to_numpy(), index=raw.index)
        out = out.fillna(raw["equipment"].map(self.eq_table)).fillna(self.global_median)
        return out.to_numpy()


class RidgeModel:
    name = "Ridge (linear, log rate-per-mile)"

    def __init__(self, alpha=1.0, cols=LINEAR_COLS):
        self.cols = cols
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))

    def fit(self, X, y, raw):
        self.model.fit(X[self.cols], y)
        return self

    def predict(self, X, raw):
        return self.model.predict(X[self.cols])


LGB_PARAMS = dict(
    n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=40,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=C.SEED, verbose=-1, n_jobs=4,
)


class LGBMModel:
    """Tree model on all features, including trend_days. Cannot extrapolate time."""
    name = "LightGBM (all features)"

    def __init__(self, **params):
        self.params = {**LGB_PARAMS, **params}

    def fit(self, X, y, raw):
        self.cols = list(X.columns)
        self.model = lgb.LGBMRegressor(**self.params).fit(X[self.cols], y)
        return self

    def predict(self, X, raw):
        return self.model.predict(X[self.cols])


class RandomForestModel:
    name = "Random forest (all features)"

    def fit(self, X, y, raw):
        self.cols = list(X.columns)
        self.model = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.5,
                                           n_jobs=4, random_state=C.SEED).fit(X[self.cols], y)
        return self

    def predict(self, X, raw):
        return self.model.predict(X[self.cols])


class RidgePlusLGBM:
    """Stage 1: ridge on the linear features (carries the market level and the time trend, which
    extrapolate past the last training date). Stage 2: LightGBM on the stage-1 residual using
    every feature except trend_days, so the trees model non-linear structure (weight, distance,
    lanes, geography) without having to extrapolate time."""
    name = "Ridge + LightGBM on residual (final)"

    def __init__(self, alpha=1.0, **params):
        self.alpha = alpha
        self.params = {**LGB_PARAMS, **params}

    def fit(self, X, y, raw):
        self.stage1 = RidgeModel(self.alpha).fit(X, y, raw)
        resid = y - self.stage1.predict(X, raw)
        self.cols = [c for c in X.columns if c != "trend_days"]
        self.stage2 = lgb.LGBMRegressor(**self.params).fit(X[self.cols], resid)
        return self

    def predict(self, X, raw):
        return self.stage1.predict(X, raw) + self.stage2.predict(X[self.cols])


CANDIDATES = {
    "median_baseline": MedianBaseline,
    "ridge": RidgeModel,
    "random_forest": RandomForestModel,
    "lightgbm": LGBMModel,
    "ridge_plus_lightgbm": RidgePlusLGBM,
}
FINAL_MODEL = "ridge_plus_lightgbm"


class FreightRateEstimator:
    """Raw labelled loads -> label filter -> features -> model -> predicted dollars.
    Everything learned (label-filter medians, imputation values, city effects, model) is fit
    on the rows passed to `fit` only."""

    def __init__(self, model_name: str = FINAL_MODEL, market_level: pd.Series | None = None,
                 filter_labels: bool = True):
        self.model_name = model_name
        self.market_level = market_level
        self.filter_labels = filter_labels

    def fit(self, train: pd.DataFrame) -> "FreightRateEstimator":
        if self.market_level is None:
            self.market_level = daily_market_level(train)
        if self.filter_labels:
            keep = clean_label_mask(train)
            self.n_dropped = int((~keep).sum())
            train = train[keep]
        train = train.reset_index(drop=True)
        self.features = FeatureBuilder(self.market_level).fit(train)
        X = self.features.transform(train, is_training_fold=True)
        y = log_rpm(train)
        self.model = CANDIDATES[self.model_name]().fit(X, y, train)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self.features.transform(df)
        return to_rate(self.model.predict(X, df), df["distance"])
