# Freight Rate Prediction

Predicts `posted_rate` for 12,000 freight loads (Nov–Dec 2025) from 48,000 labelled loads (Jan–Oct 2025).
Full write-up: [`reports/assessment_report.pdf`](reports/assessment_report.pdf).

| Deliverable | File |
|---|---|
| Final predictions (`load_id,predicted_rate`, 12,000 rows) | [`outputs/validation_predictions.csv`](outputs/validation_predictions.csv) |
| December chart inputs with `predicted_rate` filled | [`outputs/december_chart_predictions.csv`](outputs/december_chart_predictions.csv) |
| December chart from the provided `score.py` | [`outputs/scorer_results/candidate_december.png`](outputs/scorer_results/candidate_december.png) |
| Report (validation and split approach, results, chart) | [`reports/assessment_report.pdf`](reports/assessment_report.pdf) |
| Analysis notebook (Colab-ready, executed) | [`notebooks/freight_rate_analysis.ipynb`](notebooks/freight_rate_analysis.ipynb) |

## Results

MAPE on test rows whose label passes the outlier filter. `score.py` computes no metric locally (Spotter scores
Nov–Dec after submission), so these are the forward-in-time validation numbers only.

| Validation split | Final model | LightGBM alone | Median baseline |
|---|---|---|---|
| Time holdout: train Jan–Aug, test Sep–Oct (model selection) | **2.05%** | 2.12% | 3.82% |
| Rolling backtest, 5 forward windows (median / mean) | **1.76% / 3.00%** | 2.74% / 2.97% | 3.83% / 4.32% |
| Unseen cities: 8 cities removed from training | 2.41% | 2.33% | 3.82% |

All-rows numbers, MAE, RMSE, R² and bias are in [`artifacts/cv_summary.csv`](artifacts/cv_summary.csv) and the report.

**Model:** ridge regression on the daily market level and a linear time trend, plus LightGBM on the ridge residual
(without the day number). Target: `log(rate / distance)`. Rates drift up about 0.64%/month on top of the market
level, and tree models cannot carry that past the last training date. The final model wins 4 of 5 rolling windows.
It loses only the window trained on Jan–Jun, where market and time are 0.86 correlated and the trend cannot be
identified; on the final Jan–Oct training window that correlation is −0.11.

**Main limitation:** on the Sep–Oct holdout the model under-predicts by about 1.7% on average, in every segment
(market-level uncertainty). No December or holiday history exists to learn from.

## Setup

The four assessment CSVs are not committed. Put them in `data/` with these names:

```
data/train_test.csv
data/validation.csv
data/validation_predictions_template.csv
data/december_chart_inputs.csv
```

(The downloaded files may use hyphens, e.g. `train-test.csv`; rename them, or let the notebook do it.)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python run_all.py                    # tests, validation, final fit, predictions, scorer, report (~9 min)
python run_all.py --skip-evaluation  # reuse saved validation results (~1 min)
python run_all.py --tune             # also run the small tuning grid (+~4 min)
```

Or step by step:

```bash
python -m pytest -q tests            # 11 unit tests on synthetic data
python src/evaluate.py               # all models x all splits -> artifacts/cv_scores.csv, cv_summary.csv
python src/tune.py                   # optional: ridge alpha / tree size grid -> artifacts/tuning.csv
python src/train.py                  # final fit on Jan–Oct -> artifacts/final_model.joblib
python src/predict.py                # -> outputs/validation_predictions.csv, outputs/december_chart_predictions.csv
python score.py --predictions outputs/validation_predictions.csv \
                --december-predictions outputs/december_chart_predictions.csv \
                --output-dir outputs/scorer_results
python src/report.py                 # -> reports/assessment_report.pdf
```

**Google Colab:** open `notebooks/freight_rate_analysis.ipynb`, set `REPO_URL` in the first cell, and run all
cells. It clones the repo, installs requirements, prompts for the four CSVs, and runs EDA, validation, the final
fit, predictions and the scorer.

The raw `december_chart_inputs.csv` is never modified; the filled copy goes to `outputs/`. Missing December inputs
are filled as follows: coordinates from the city table, market level from that date in `validation.csv`,
`quote_signal` at its training median.

## Code layout

```
src/freight/
  config.py       paths, column names, seed, outlier limits (single source)
  data.py         loading; input repairs (all rows); label filter (training rows only); daily market level
  features.py     FeatureBuilder: fit on the training fold, transform anywhere; out-of-fold city effects
  models.py       median baseline, ridge, random forest, LightGBM, Ridge+LightGBM; FreightRateEstimator
  validation.py   time holdout, rolling origin, unseen-city and random splits; metrics; evaluation loop
  analysis.py     EDA helpers (level residuals, trend identification, data-quality table)
  plots.py        figures shared by the notebook and report
src/evaluate.py, tune.py, train.py, predict.py, report.py    entry points
score.py          provided scorer, unmodified
tests/            unit tests (repairs, label filter, OOF encoding, split ordering, estimators, metrics)
```

`FreightRateEstimator` wraps the full chain (label filter → features → model → dollars). Each validation fold
refits it from scratch on its training part, so outlier limits, imputation values and city encodings never see
test labels.

## Validation approach (summary)

- **Forward in time.** The task is a 2-month forecast past the last label, so selection uses train Jan–Aug / test
  Sep–Oct, backed by an expanding-window backtest over 5 windows.
- **Unseen cities.** 8 validation cities never appear in training, so the holdout is re-run with 8 random cities
  removed from training.
- **No leakage.** Every fold refits the label filter, imputation, out-of-fold city effects and model on its
  training part only. The daily market level uses the `market_index` input column, never labels.
- **Why not a random split.** Random 5-fold is reported as a contrast. It ranks LightGBM and random forest best
  (~1.1%) because they memorise each day's level, and both trail the final model forward in time.

## Data-quality issues handled

| Issue | Train | Validation | Treatment |
|---|---|---|---|
| Missing `weight` | 300 | 165 | flag + equipment median (training fold) |
| Negative `weight` (sign error) | 292 | 145 | absolute value + flag |
| `weight` censored at 47,500 lb | 1,204 | 301 | kept, flagged |
| Missing `market_index` | 374 | 249 | flag + that date's market level |
| Corrupted rate labels (0.17–0.47× or 2.1–5.4× segment median) | 677 | – | removed from training only |

## Environment

Tested with Python 3.13 on Linux, using both pandas 2.3 (the pinned requirements) and pandas 3.0, with
scikit-learn 1.9 and LightGBM 4.7. Seed 42 throughout.
