"""Fit the final model on all labelled data (Jan-Oct 2025) and save it.

Usage:  python src/train.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))

from freight import config as C  # noqa: E402
from freight.data import daily_market_level, load_train, load_validation  # noqa: E402
from freight.models import FINAL_MODEL, FreightRateEstimator  # noqa: E402


def train_final() -> FreightRateEstimator:
    train, val = load_train(), load_validation()
    # Daily market level is built from the market_index *input* of both files (no labels), so
    # Nov-Dec dates have a level available at prediction time.
    market = daily_market_level(train, val)
    est = FreightRateEstimator(FINAL_MODEL, market).fit(train)
    return est


def main():
    est = train_final()
    C.ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(est, C.MODEL_FILE)
    ridge = est.model.stage1.model[-1]
    cols = est.model.stage1.cols
    scale = est.model.stage1.model[0].scale_
    coef = dict(zip(cols, ridge.coef_ / scale))
    print(f"Trained {FINAL_MODEL} on {len(load_train()) - est.n_dropped:,} rows "
          f"({est.n_dropped} corrupted labels removed).")
    print(f"Trend: {coef['trend_days'] * 30.4 * 100:.2f}% per month; "
          f"market level: {coef['market_day']:.3f} log-units per index point.")
    print(f"Saved -> {C.MODEL_FILE}")


if __name__ == "__main__":
    main()
