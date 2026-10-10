"""EDA helpers behind the feature and validation decisions (used by the notebook and report)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from . import config as C
from .data import clean_label_mask
from .features import FeatureBuilder, log_rpm

STRUCTURAL = ["log_distance", "log_distance_sq", "inv_distance", "eq_flatbed", "eq_reefer",
              "pickup_effect", "delivery_effect"]


def level_residuals(train: pd.DataFrame, market: pd.Series, include_weight: bool = True):
    """Log rate-per-mile residual after the structural load features (distance, equipment,
    city effects, optionally weight). What is left is the date-level signal + noise."""
    clean = train[clean_label_mask(train)].reset_index(drop=True)
    X = FeatureBuilder(market).fit(clean).transform(clean, is_training_fold=True)
    cols = STRUCTURAL + (["weight_k", "weight_capped"] if include_weight else [])
    y = log_rpm(clean)
    res = y - LinearRegression().fit(X[cols], y).predict(X[cols])
    return clean, X, res


def daily_level_table(clean, X, res) -> pd.DataFrame:
    d = pd.DataFrame({C.DATE: clean[C.DATE], "residual": res, "market": X["market_day"],
                      "quote_signal": clean["quote_signal"]}).groupby(C.DATE).mean()
    d["t"] = (d.index - pd.Timestamp(C.TREND_ORIGIN)).days
    return d


def trend_stability(daily: pd.DataFrame) -> pd.DataFrame:
    """Fit residual ~ market + t on expanding windows. Shows when trend and market can be told
    apart (low corr(market, t)) and how stable the trend estimate is."""
    rows = []
    for end in ["2025-07-01", "2025-08-01", "2025-09-01", "2025-10-01", "2025-11-01"]:
        a = daily[daily.index < end]
        lr = LinearRegression().fit(a[["market", "t"]], a["residual"])
        rows.append({"training window": f"Jan – {(pd.Timestamp(end) - pd.Timedelta(days=1)):%b}",
                     "corr(market, time)": round(a["market"].corr(a["t"]), 2),
                     "market coef": round(lr.coef_[0], 3),
                     "trend %/month": round(lr.coef_[1] * 30.4 * 100, 2)})
    return pd.DataFrame(rows)


def data_quality_table(train: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    def counts(d):
        return {
            "missing weight": int(d["weight"].isna().sum()),
            "negative weight": int((d["weight"] < 0).sum()),
            "weight at 47,500 cap": int(np.isclose(d["weight"].abs(), C.WEIGHT_CAP).sum()),
            "missing market_index": int(d["market_index"].isna().sum()),
            "duplicate load_id": int(d[C.ID].duplicated().sum()),
            "corrupted rate label": int((~clean_label_mask(d)).sum()) if C.TARGET in d else None,
        }
    t, v = counts(train), counts(val)
    treat = {
        "missing weight": "flag + equipment median (fit on training fold)",
        "negative weight": "sign error: absolute value + flag",
        "weight at 47,500 cap": "censored: kept, flagged",
        "missing market_index": "flag + that date's market level",
        "duplicate load_id": "none found",
        "corrupted rate label": "dropped from training only (ratio outside 0.6–1.6× segment median)",
    }
    return pd.DataFrame({"issue": list(t), "train": list(t.values()),
                         "validation": ["–" if v[k] is None else v[k] for k in t],
                         "treatment": [treat[k] for k in t]})
