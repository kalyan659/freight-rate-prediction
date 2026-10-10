"""Predict every load in validation.csv and every day of the December chart file, then run strict
format checks. Writes outputs/validation_predictions.csv and outputs/december_chart_predictions.csv.

Usage:  python src/predict.py          (trains first if artifacts/final_model.joblib is missing)
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from freight import config as C  # noqa: E402
from freight.data import city_table, load_train, load_validation  # noqa: E402

DEC_COLUMNS = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]


def december_features(dec_raw: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    """The December file only has pickup, delivery, distance, equipment, weight and date.
    Fill the remaining model inputs:
    - coordinates: looked up from the city table (Lexington and Fort Wayne both appear in the data)
    - market_index: left missing, so the model uses that date's market level from validation.csv
    - quote_signal: training median (it carries almost no signal; see report)."""
    cities = city_table(train, val)
    d = dec_raw.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["distance"] = d["distance"].astype(float)
    d["weight"] = d["weight"].astype(float)
    for name, lat, lon in C.CITY_COLS:
        d[lat] = d[name].map(cities["lat"])
        d[lon] = d[name].map(cities["lon"])
        if d[lat].isna().any():
            raise ValueError(f"no coordinates for {name} city in December file")
    d["market_index"] = np.nan
    d["quote_signal"] = float(train["quote_signal"].median())
    return d


def check_predictions(out: pd.DataFrame, template: pd.DataFrame) -> None:
    assert list(out.columns) == ["load_id", "predicted_rate"], out.columns
    assert len(out) == 12_000, len(out)
    assert out["load_id"].is_unique
    assert (out["load_id"].to_numpy() == template["load_id"].to_numpy()).all(), "ID order changed"
    p = out["predicted_rate"].to_numpy(float)
    assert np.isfinite(p).all() and (p > 0).all()


def main():
    if not C.MODEL_FILE.exists():
        from train import main as train_main
        train_main()
    est = joblib.load(C.MODEL_FILE)
    train, val = load_train(), load_validation()
    template = pd.read_csv(C.TEMPLATE_FILE)

    # --- 12,000 validation loads, in template order --------------------------------
    pred = pd.Series(est.predict(val), index=val["load_id"])
    out = template.copy()
    out["predicted_rate"] = out["load_id"].map(pred).round(2)
    check_predictions(out, template)
    C.OUTPUT_DIR.mkdir(exist_ok=True)
    out.to_csv(C.PREDICTIONS_FILE, index=False)
    reread = pd.read_csv(C.PREDICTIONS_FILE)
    check_predictions(reread, template)
    print(f"Wrote {C.PREDICTIONS_FILE} ({len(reread):,} rows, checks passed)")

    # --- December chart file: keep the raw file untouched, write a filled copy -------
    dec_raw = pd.read_csv(C.DECEMBER_INPUT_FILE)
    dec_pred = est.predict(december_features(dec_raw, train, val))
    dec_out = dec_raw.copy()
    dec_out["predicted_rate"] = np.round(dec_pred, 2)
    assert list(dec_out.columns) == DEC_COLUMNS and len(dec_out) == 31
    dec_out.to_csv(C.DECEMBER_PREDICTIONS_FILE, index=False)
    print(f"Wrote {C.DECEMBER_PREDICTIONS_FILE} "
          f"(${dec_out.predicted_rate.min():.2f} - ${dec_out.predicted_rate.max():.2f})")


if __name__ == "__main__":
    main()
