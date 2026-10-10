"""Split strategies, metrics and the evaluation loop. Each fold refits the full estimator
(label filter, imputation, city effects, model) on the training part only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from . import config as C
from .data import clean_label_mask

TS = pd.Timestamp

# (train end, test end, label): expanding training window, forward test window
ROLLING_WINDOWS = [
    ("2025-05-01", "2025-07-01", "Jan-Apr -> May-Jun"),
    ("2025-07-01", "2025-09-01", "Jan-Jun -> Jul-Aug"),
    ("2025-08-01", "2025-10-01", "Jan-Jul -> Aug-Sep"),
    ("2025-09-01", "2025-11-01", "Jan-Aug -> Sep-Oct"),
    ("2025-10-01", "2025-11-01", "Jan-Sep -> Oct"),
]


def time_holdout(df):
    """Model selection split: train Jan-Aug, test Sep-Oct (a 2-month forward window, like
    the real task: train to Oct, predict Nov-Dec)."""
    tr = df[C.DATE] < TS("2025-09-01")
    yield "holdout Jan-Aug -> Sep-Oct", df.index[tr], df.index[~tr]


def rolling_origin(df):
    """Expanding-window backtest with 2-month forward test windows."""
    for train_end, test_end, label in ROLLING_WINDOWS:
        tr = df[C.DATE] < TS(train_end)
        te = (df[C.DATE] >= TS(train_end)) & (df[C.DATE] < TS(test_end))
        yield f"rolling {label}", df.index[tr], df.index[te]


def unseen_cities(df, n_cities: int = 8, seed: int = C.SEED):
    """Same time holdout, but 8 random cities are removed from training entirely and the test
    set is restricted to loads touching them (validation.csv has 8 cities never seen in
    training)."""
    rng = np.random.default_rng(seed)
    cities = sorted(set(df["pickup"]) | set(df["delivery"]))
    held = set(rng.choice(cities, n_cities, replace=False))
    touches = df["pickup"].isin(held) | df["delivery"].isin(held)
    tr = (df[C.DATE] < TS("2025-09-01")) & ~touches
    te = (df[C.DATE] >= TS("2025-09-01")) & touches
    yield f"unseen cities ({', '.join(sorted(held))})", df.index[tr], df.index[te]


def random_kfold(df, k: int = 5):
    """Contrast only: random split leaks future market/trend information into training."""
    for i, (a, b) in enumerate(KFold(k, shuffle=True, random_state=C.SEED).split(df)):
        yield f"random 5-fold #{i + 1}", df.index[a], df.index[b]


SPLITS = {"holdout": time_holdout, "rolling": rolling_origin,
          "unseen_city": unseen_cities, "random_kfold": random_kfold}


def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    ss_res, ss_tot = (err ** 2).sum(), ((y_true - y_true.mean()) ** 2).sum()
    return {
        "MAPE_%": 100 * np.mean(np.abs(err) / y_true),
        "MAE": np.mean(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err ** 2)),
        "R2": 1 - ss_res / ss_tot,
        "bias_%": 100 * np.mean(err / y_true),
    }


def evaluate(df, model_names, split_names, estimator_factory, keep_predictions=False):
    """Returns (scores, predictions). Scores are computed twice per fold: on every test row,
    and on test rows whose label passes the outlier filter (the filter's medians come from the
    training part, so test labels never set their own limits)."""
    rows, preds = [], []
    for split in split_names:
        for fold, tr_idx, te_idx in SPLITS[split](df):
            train, test = df.loc[tr_idx], df.loc[te_idx]
            clean = clean_label_mask(test, reference=train).to_numpy()
            for name in model_names:
                p = estimator_factory(name).fit(train).predict(test)
                y = test[C.TARGET].to_numpy()
                for subset, m in (("all rows", np.ones(len(y), bool)), ("clean labels", clean)):
                    rows.append({"split": split, "fold": fold, "model": name, "subset": subset,
                                 "n_train": len(train), "n_test": int(m.sum()),
                                 **metrics(y[m], p[m])})
                if keep_predictions:
                    preds.append(pd.DataFrame({"split": split, "fold": fold, "model": name,
                                               C.ID: test[C.ID].to_numpy(), "y": y, "pred": p,
                                               "clean": clean}))
    return pd.DataFrame(rows), (pd.concat(preds) if preds else None)
