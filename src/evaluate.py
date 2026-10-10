"""Compare every model on every validation split. Writes artifacts/cv_scores.csv (per fold),
artifacts/cv_summary.csv (mean per split/model/subset) and artifacts/holdout_predictions.csv
(Sep-Oct holdout predictions, used for residual analysis).

Usage:  python src/evaluate.py [--splits holdout rolling unseen_city random_kfold]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from freight import config as C  # noqa: E402
from freight.data import daily_market_level, load_train, load_validation  # noqa: E402
from freight.models import CANDIDATES, FreightRateEstimator  # noqa: E402
from freight.validation import SPLITS, evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--models", nargs="+", default=list(CANDIDATES))
    args = ap.parse_args()

    C.ARTIFACT_DIR.mkdir(exist_ok=True)
    train, val = load_train(), load_validation()
    market = daily_market_level(train, val)

    t0 = time.time()
    scores, preds = evaluate(train, args.models, args.splits,
                             lambda n: FreightRateEstimator(n, market), keep_predictions=True)
    scores.to_csv(C.ARTIFACT_DIR / "cv_scores.csv", index=False)
    summary = (scores.groupby(["split", "subset", "model"])
               [["MAPE_%", "MAE", "RMSE", "R2", "bias_%", "n_test"]].mean().round(4).reset_index())
    summary.to_csv(C.ARTIFACT_DIR / "cv_summary.csv", index=False)
    preds[preds["split"] == "holdout"].to_csv(C.ARTIFACT_DIR / "holdout_predictions.csv", index=False)

    view = summary[summary["subset"] == "clean labels"].pivot(index="model", columns="split",
                                                              values="MAPE_%")
    print("MAPE % on clean-label test rows (mean over folds):")
    print(view.round(3).to_string())
    print(f"\nDone in {time.time() - t0:.0f}s -> {C.ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
