# Loom script (target 2:45)

Screen order: report page 1 → notebook EDA figures → data-quality table → split table + trend table →
model comparison chart → `src/freight/models.py` → `src/freight/validation.py` → December chart.

---

**0:00 – Intro (15 s)** · *report page 1*

The task is to predict posted freight rates for 12,000 loads in November and December 2025, from 48,000 labelled
loads running January to October. That makes it a forecast one to two months past the last label, and that fact
drove most of my decisions.

**0:15 – Key findings (35 s)** · *notebook section 2 figures*

Distance is the main driver, so I model log rate per mile. Three findings mattered:
- `market_index` varies by load, but almost all the signal is in the daily average; a load's deviation from its
  day adds nothing. So I use the daily market level.
- After the market, there is a steady upward trend of about 0.64% per month.
- Heavier loads cost more per mile, there's a small midweek peak, and pickup and delivery cities each shift rates
  by a few percent. `quote_signal` showed no real relationship with the rate.

**0:50 – Data quality (25 s)** · *data-quality table, outlier histogram*

677 rate labels, about 1.4%, sit two to five times above or below their segment, with a clean gap from the normal
loads. They look like unit errors, so I drop them from training only; every validation load still gets a
prediction. Negative weights are sign errors, so I take the absolute value. Weights capped at 47,500 are flagged, and
missing weight and market index get flags plus training-fold imputation.

**1:15 – Validation and split (35 s)** · *split table, trend-identification table*

Because the job is forecasting, I validate forward in time only. Model selection trains on January to August and
tests on September to October, the same two-month horizon. A five-window rolling backtest and an unseen-city test
back that up, since 8 validation cities never appear in training. Every fold refits the whole pipeline, including
outlier limits and city encodings, on its training part only. Random K-fold is reported only as a contrast: it
ranks the tree models best because they memorise each day's level, which is useless for November.

**1:50 – Model choice (30 s)** · *model comparison chart, rolling-window chart*

The final model is ridge regression on the market level and the time trend, with LightGBM fitted on the ridge
residual without the day number. Trees can't extrapolate a trend past the last date; the linear stage can. On the
holdout it scores 2.05% MAPE on clean labels against 3.82% for the median baseline and 2.12% for LightGBM. It wins
4 of 5 rolling windows. It loses the one window trained only on January to June, where market and time move
together and the trend can't be identified. The final fit uses January to October, where that correlation is
near zero.

**2:20 – Code walkthrough (20 s)** · *models.py → validation.py*

`FreightRateEstimator` takes raw loads, filters labels, builds features and fits the model, so the same object is
refit inside every fold. `RidgePlusLGBM` is the two-stage model. `validation.py` holds the split strategies and
scores each fold on all rows and on clean labels.

**2:40 – Output and limits (15 s)** · *December chart*

`score.py` validates both files and draws this December chart: a weekly cycle on a slight upward trend. The main
risk is the level: on the holdout the model under-predicted by about 1.7%, and there is no December history to
learn holidays from.
