"""Small hyper-parameter grid for the final model (ridge alpha, stage-2 LightGBM size), scored on
the three rolling windows whose training data separates the market from the trend
(corr(market, time) <= 0.73). Writes artifacts/tuning.csv. ~3 minutes.

Usage:  python src/tune.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from freight import config as C  # noqa: E402
from freight import models as M  # noqa: E402
from freight.data import clean_label_mask, daily_market_level, load_train, load_validation  # noqa: E402
from freight.validation import ROLLING_WINDOWS, metrics  # noqa: E402

GRID = {"alpha": [1.0, 10.0], "num_leaves": [15, 31, 63], "n_estimators": [300, 600]}
WINDOWS = [w for w in ROLLING_WINDOWS if w[0] >= "2025-08-01"]


def main():
    train, val = load_train(), load_validation()
    market = daily_market_level(train, val)
    rows = []
    for alpha, leaves, n_est in itertools.product(*GRID.values()):
        M.CANDIDATES["tuned"] = lambda a=alpha, l=leaves, n=n_est: M.RidgePlusLGBM(
            alpha=a, num_leaves=l, n_estimators=n)
        for tr_end, te_end, label in WINDOWS:
            tr = train[train[C.DATE] < tr_end]
            te = train[(train[C.DATE] >= tr_end) & (train[C.DATE] < te_end)]
            mask = clean_label_mask(te, reference=tr).to_numpy()
            p = M.FreightRateEstimator("tuned", market).fit(tr).predict(te)
            m = metrics(te[C.TARGET].to_numpy()[mask], p[mask])
            rows.append({"alpha": alpha, "num_leaves": leaves, "n_estimators": n_est,
                         "window": label, "MAPE_%": m["MAPE_%"]})
    res = pd.DataFrame(rows)
    C.ARTIFACT_DIR.mkdir(exist_ok=True)
    res.to_csv(C.ARTIFACT_DIR / "tuning.csv", index=False)
    print(res.groupby(["alpha", "num_leaves", "n_estimators"])["MAPE_%"].mean()
          .sort_values().round(3).to_string())


if __name__ == "__main__":
    main()
