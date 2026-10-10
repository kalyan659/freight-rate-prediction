"""Fast unit tests on small synthetic frames (no assessment data needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from freight import config as C  # noqa: E402
from freight.data import clean_label_mask, daily_market_level, repair_inputs  # noqa: E402
from freight.features import FeatureBuilder, log_rpm, to_rate  # noqa: E402
from freight.models import FreightRateEstimator  # noqa: E402
from freight.validation import metrics, rolling_origin, time_holdout, unseen_cities  # noqa: E402

CITIES = {"A": (35.0, -90.0), "B": (40.0, -85.0), "C": (32.0, -97.0), "D": (42.0, -75.0),
          "E": (30.0, -82.0), "F": (38.0, -100.0), "G": (44.0, -88.0), "H": (34.0, -118.0),
          "I": (36.0, -80.0), "J": (39.0, -95.0)}


def make_frame(n=3000, seed=0, with_target=True):
    rng = np.random.default_rng(seed)
    names = list(CITIES)
    pu = rng.choice(names, n)
    de = np.array([rng.choice([c for c in names if c != p]) for p in pu])
    dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 300, n), unit="D")
    day_level = 1 + 0.2 * np.sin(np.arange(400) / 40)
    d = pd.DataFrame({
        "load_id": [f"TR-{i:06d}" for i in range(n)],
        "pickup": pu, "delivery": de,
        "pickup_lat": [CITIES[c][0] for c in pu], "pickup_lon": [CITIES[c][1] for c in pu],
        "delivery_lat": [CITIES[c][0] for c in de], "delivery_lon": [CITIES[c][1] for c in de],
        "distance": rng.uniform(80, 2500, n).round(1),
        "equipment": rng.choice(["Dry Van", "Reefer", "Flatbed"], n),
        "weight": rng.uniform(10_000, 47_500, n).round(),
        "date": dates,
    })
    t = (d.date - pd.Timestamp("2025-01-01")).dt.days.to_numpy()
    d["market_index"] = day_level[t] + rng.normal(0, 0.02, n)
    d["quote_signal"] = rng.normal(2, 0.2, n)
    if with_target:
        rpm = 2.0 * (d.distance / 1000) ** -0.15 * np.exp(0.15 * day_level[t] + 0.0002 * t) \
            * np.exp(rng.normal(0, 0.02, n))
        d["posted_rate"] = (rpm * d.distance).round(2)
    return d


def test_repairs_keep_every_row_and_fix_sign():
    d = make_frame(50)
    d.loc[0, "weight"] = -20_000
    d.loc[1, "weight"] = np.nan
    d.loc[2, "weight"] = C.WEIGHT_CAP
    r = repair_inputs(d)
    assert len(r) == len(d)
    assert r.loc[0, "weight"] == 20_000 and r.loc[0, "weight_was_negative"] == 1
    assert r.loc[1, "weight_missing"] == 1 and r.loc[2, "weight_capped"] == 1


def test_label_filter_flags_corrupted_rows_only():
    d = make_frame(2000)
    d.loc[:9, "posted_rate"] *= 4
    d.loc[10:19, "posted_rate"] *= 0.25
    mask = clean_label_mask(d)
    assert not mask.iloc[:20].any()
    assert mask.iloc[20:].mean() > 0.995


def test_target_round_trip():
    d = make_frame(100)
    assert np.allclose(to_rate(log_rpm(d), d.distance), d.posted_rate)


def test_city_effects_out_of_fold_and_unseen_city_is_zero():
    d = make_frame(2000)
    market = daily_market_level(d)
    fb = FeatureBuilder(market).fit(d)
    X_train = fb.transform(d, is_training_fold=True)
    X_full = fb.transform(d)
    assert not np.allclose(X_train.pickup_effect, X_full.pickup_effect)  # OOF differs from full-fit
    new = d.head(5).copy()
    new["pickup"] = "Nowhere"
    assert (fb.transform(new).pickup_effect == 0).all()


def test_splits_are_forward_in_time_and_disjoint():
    d = make_frame(3000)
    d["date"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(np.arange(3000) % 304, unit="D")
    for gen in (time_holdout, rolling_origin, unseen_cities):
        for _, tr, te in gen(d):
            assert len(set(tr) & set(te)) == 0
            assert d.loc[tr, "date"].max() < d.loc[te, "date"].min()


def test_unseen_city_split_removes_held_cities_from_training():
    d = make_frame(3000)
    d["date"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(np.arange(3000) % 304, unit="D")
    _, tr, te = next(unseen_cities(d, n_cities=3))
    held = set(d.loc[te, "pickup"]) | set(d.loc[te, "delivery"])
    train_cities = set(d.loc[tr, "pickup"]) | set(d.loc[tr, "delivery"])
    test_only = {c for c in held if c not in train_cities}
    assert len(test_only) == 3


@pytest.mark.parametrize("model", ["median_baseline", "ridge", "ridge_plus_lightgbm"])
def test_estimator_predicts_positive_finite_rates(model):
    d = make_frame(2500)
    train, test = d[d.date < "2025-09-01"], d[d.date >= "2025-09-01"]
    est = FreightRateEstimator(model, daily_market_level(d)).fit(train)
    p = est.predict(test.drop(columns="posted_rate"))
    assert len(p) == len(test) and np.isfinite(p).all() and (p > 0).all()


def test_trend_model_extrapolates_better_than_trees_on_trending_data():
    d = make_frame(4000, seed=3)
    train, test = d[d.date < "2025-08-01"], d[d.date >= "2025-08-01"]
    market = daily_market_level(d)
    m_trend = metrics(test.posted_rate, FreightRateEstimator("ridge_plus_lightgbm", market).fit(train).predict(test))
    m_base = metrics(test.posted_rate, FreightRateEstimator("median_baseline", market).fit(train).predict(test))
    assert m_trend["MAPE_%"] < m_base["MAPE_%"]


def test_metrics_values():
    m = metrics([100, 200], [110, 180])
    assert m["MAPE_%"] == pytest.approx(10.0)
    assert m["MAE"] == pytest.approx(15.0)
