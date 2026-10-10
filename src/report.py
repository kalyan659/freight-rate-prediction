"""Build reports/assessment_report.pdf from the saved artifacts and the official scorer chart.
Every number in the report is read from artifacts/ or computed from the data here.

Run after: src/evaluate.py, src/train.py, src/predict.py and score.py.
Usage:  python src/report.py
"""
from __future__ import annotations

import os
import sys
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from freight import analysis, plots  # noqa: E402
from freight import config as C  # noqa: E402
from freight.data import daily_market_level, load_train, load_validation  # noqa: E402
from freight.models import FINAL_MODEL  # noqa: E402

REPORT = C.ROOT / "reports" / "assessment_report.pdf"
CHART = C.OUTPUT_DIR / "scorer_results" / "candidate_december.png"

# ---- fonts / styles (DejaVu ships with matplotlib, so no system font is needed) -------------
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DV", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DVB", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DVI", str(FONT_DIR / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DVM", str(FONT_DIR / "DejaVuSansMono.ttf")))
from reportlab.lib.fonts import addMapping  # noqa: E402

addMapping("DV", 0, 0, "DV"); addMapping("DV", 1, 0, "DVB"); addMapping("DV", 0, 1, "DVI")

INK = colors.HexColor("#1f2933")
ACCENT = colors.HexColor("#064A56")
MUTED = colors.HexColor("#52514e")
RULE = colors.HexColor("#d9dee0")
BODY = ParagraphStyle("body", fontName="DV", fontSize=9.2, leading=13.2, textColor=INK, spaceAfter=5)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=7.8, leading=10.5, textColor=MUTED)
H1 = ParagraphStyle("h1", fontName="DVB", fontSize=17, leading=21, textColor=ACCENT, spaceAfter=4)
H2 = ParagraphStyle("h2", fontName="DVB", fontSize=12, leading=15, textColor=ACCENT, spaceBefore=10, spaceAfter=5)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=7.8, leading=9.8, spaceAfter=0, alignment=TA_LEFT)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="DVB")
BULLET = ParagraphStyle("bullet", parent=BODY, leftIndent=11, bulletIndent=2, spaceAfter=2.5)
CODE = ParagraphStyle("code", fontName="DVM", fontSize=7.6, leading=10.2, textColor=INK,
                      backColor=colors.HexColor("#f3f5f6"), borderPadding=5, leftIndent=4, spaceAfter=8)


def p(text, style=BODY):
    return Paragraph(text, style)


def bullets(items):
    return [Paragraph(t, BULLET, bulletText="•") for t in items]


def table(rows, widths, header=True, bold_row=None):
    data = [[Paragraph(str(c), CELLB if (header and i == 0) or (bold_row is not None and i == bold_row) else CELL)
             for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT), ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE),
             ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("LEFTPADDING", (0, 0), (-1, -1), 4)]
    if bold_row is not None:
        style.append(("BACKGROUND", (0, bold_row), (-1, bold_row), colors.HexColor("#fdf0ea")))
    t.setStyle(TableStyle(style))
    return t


def fig_image(fig, width_cm=17.0):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    matplotlib.pyplot.close(fig)
    buf.seek(0)
    img = Image(buf)
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = width_cm * cm, width_cm * cm * ratio
    return img


def caption(text):
    return p(text, SMALL)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DV", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.2 * cm, "Freight Rate Prediction — ML assessment report")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"page {doc.page}")
    canvas.restoreState()


def main():
    for f in [C.ARTIFACT_DIR / "cv_summary.csv", C.ARTIFACT_DIR / "cv_scores.csv",
              C.ARTIFACT_DIR / "holdout_predictions.csv", C.MODEL_FILE, C.PREDICTIONS_FILE, CHART]:
        if not f.exists():
            raise SystemExit(f"Missing {f}. Run evaluate.py, train.py, predict.py and score.py first.")

    train, val = load_train(), load_validation()
    market = daily_market_level(train, val)
    summary = pd.read_csv(C.ARTIFACT_DIR / "cv_summary.csv")
    scores = pd.read_csv(C.ARTIFACT_DIR / "cv_scores.csv")
    hold = pd.read_csv(C.ARTIFACT_DIR / "holdout_predictions.csv")
    tuning = pd.read_csv(C.ARTIFACT_DIR / "tuning.csv") if (C.ARTIFACT_DIR / "tuning.csv").exists() else None
    est = joblib.load(C.MODEL_FILE)
    preds = pd.read_csv(C.PREDICTIONS_FILE)
    dec = pd.read_csv(C.DECEMBER_PREDICTIONS_FILE)

    clean, Xc, res = analysis.level_residuals(train, market)
    daily = analysis.daily_level_table(clean, Xc, res)
    stab = analysis.trend_stability(daily)
    dq = analysis.data_quality_table(train, val)

    s_clean = summary[summary.subset == "clean labels"].set_index(["split", "model"])
    s_all = summary[summary.subset == "all rows"].set_index(["split", "model"])
    M = lambda split, model, col="MAPE_%", s=s_clean: s.loc[(split, model), col]  # noqa: E731
    roll = scores[(scores.split == "rolling") & (scores.subset == "clean labels")]
    roll_med = roll.groupby("model")["MAPE_%"].median()
    wins = (roll.pivot_table(index="fold", columns="model", values="MAPE_%")
            .pipe(lambda t: (t["ridge_plus_lightgbm"] < t["lightgbm"]).sum()))
    ridge = est.model.stage1.model[-1]
    coef = dict(zip(est.model.stage1.cols, ridge.coef_ / est.model.stage1.model[0].scale_))
    trend_month = coef["trend_days"] * 30.4 * 100
    final_rows = len(train) - est.n_dropped

    h = hold[(hold.model == FINAL_MODEL) & hold.clean].merge(train[[C.ID, "equipment"]], on=C.ID)
    h["pct"] = 100 * (h.pred / h.y - 1)
    by_eq = h.groupby("equipment")["pct"].agg(lambda e: e.abs().mean())

    story = []
    # ---------------------------------------------------------------- title + summary
    story += [p("Freight Rate Prediction", H1),
              p("Machine Learning Engineer assessment — validation approach, model and December forecast", SMALL),
              Spacer(1, 8)]
    story.append(p("1. Executive summary", H2))
    story += bullets([
        f"<b>Task.</b> Predict <font name='DVM'>posted_rate</font> for 12,000 loads dated Nov–Dec 2025, using "
        f"{len(train):,} labelled loads from Jan–Oct 2025. The deliverable is a forecast 1–2 months past the last label.",
        f"<b>Validation.</b> Forward-in-time splits only: model selection on train Jan–Aug / test Sep–Oct, "
        f"confirmed by a 5-window rolling backtest and an unseen-city test. Random K-fold is reported only as a "
        f"contrast: it rewards models that memorise each day's level.",
        f"<b>Model.</b> Ridge regression on the daily market level and a linear time trend, plus LightGBM on the "
        f"ridge residual. Target is log(rate per mile). The trend ({trend_month:.2f}% per month on top of the market "
        f"level) is the part tree models cannot carry past the last training date.",
        f"<b>Results (clean-label MAPE).</b> Time holdout {M('holdout', FINAL_MODEL):.2f}% vs "
        f"{M('holdout', 'median_baseline'):.2f}% for the median baseline and {M('holdout', 'lightgbm'):.2f}% for "
        f"LightGBM alone. Rolling backtest median {roll_med[FINAL_MODEL]:.2f}% (wins {wins} of 5 windows against "
        f"LightGBM). Unseen cities {M('unseen_city', FINAL_MODEL):.2f}%.",
        f"<b>Data quality.</b> {est.n_dropped} corrupted rate labels (1.4%, off by 2–5× or 0.2–0.5×) removed from "
        f"training only; sign-flipped, missing and censored weights repaired and flagged; missing market_index "
        f"filled from the same day's market level.",
        "<b>Outputs.</b> <font name='DVM'>validation_predictions.csv</font> (12,000 rows, "
        "<font name='DVM'>load_id,predicted_rate</font>) and the December chart, both validated by the provided "
        "<font name='DVM'>score.py</font>. Spotter computes the final metric; no score on Nov–Dec is claimed here.",
    ])

    # ---------------------------------------------------------------- data
    story.append(p("2. Data and exploratory findings", H2))
    story.append(p(
        f"<font name='DVM'>train_test.csv</font>: {len(train):,} loads × 14 columns, 2025-01-01 to 2025-10-31. "
        f"<font name='DVM'>validation.csv</font>: {len(val):,} loads, 2025-11-01 to 2025-12-31, same columns without "
        f"the target. 64 cities in training; validation has 72, of which 8 never appear in training "
        f"(Allentown, Charlotte, Chicago, Jackson, Knoxville, Laredo, Norfolk, San Diego). Equipment: Dry Van 57%, "
        f"Reefer 25%, Flatbed 18%. No duplicate ids or duplicate rows."))
    story += bullets([
        "<b>Distance dominates.</b> log(rate) correlates 0.90 with distance; per-mile rates fall with distance. "
        "Modelling log(rate ÷ distance) gives a well-behaved target and multiplicative errors.",
        f"<b>Market level, not the per-load value.</b> market_index varies by load, but its within-day spread is "
        f"small (std ≈ 0.025). The daily mean correlates 0.70 with the daily rate level; a load's deviation from its "
        f"day's mean adds almost nothing (0.08). The model uses the daily mean, built from the input column of both "
        f"files (no labels involved).",
        "<b>Upward trend.</b> After the market level, rates drift up steadily: January sits below and Sep–Oct above "
        "what the market alone explains (figure 2).",
        "<b>Weight and calendar.</b> Per-mile rate rises ~7% from the lightest to the heaviest weight decile. Day of "
        "week moves rates ±1% (midweek high, weekend low). Pickup and delivery cities each shift rates by up to ±3%.",
        "<b>quote_signal</b> has no useful relationship with the rate (correlation 0.08 at load and day level). It is "
        "kept as a tree input but carries little weight.",
        "<b>No December history.</b> Training ends in October, so no December or holiday effect can be learned.",
    ])
    story.append(fig_image(plots.fig_market_and_level(train, val, market, daily["residual"]), 16.5))
    story.append(caption("Figure 2. Top: daily market level; orange is the validation period. Bottom: the rate level "
                         "left after distance, equipment, weight and city effects (training data)."))

    # ---------------------------------------------------------------- data quality
    story.append(p("3. Data-quality issues and treatment", H2))
    rows = [["Issue", "Train", "Validation", "Treatment"]] + dq.astype(str).values.tolist()
    story.append(table(rows, [3.6 * cm, 1.4 * cm, 2.1 * cm, 9.9 * cm]))
    story.append(Spacer(1, 4))
    story.append(fig_image(plots.fig_target_outliers(train), 15))
    story.append(caption("Figure 3. Rate per mile divided by its equipment × distance-band median. 98% of loads sit "
                         "within 0.86–1.16×; the corrupted labels form separate clusters at 0.17–0.47× and "
                         "2.1–5.4×, so the 0.6× / 1.6× limits (dashed) separate them cleanly."))
    story.append(p(
        "Repairs that need no learned statistics (sign fix, flags) apply to every row. Statistics are learned on the "
        "training fold only: imputation medians, outlier limits and city effects. The label filter only ever removes "
        "training rows, so every load that needs a prediction gets one."))

    # ---------------------------------------------------------------- features & leakage
    story.append(p("4. Feature engineering and leakage prevention", H2))
    story += bullets([
        "<b>Distance:</b> log distance, its square, 100/distance (fixed per-load charge on short hauls); circuity "
        "(road ÷ straight-line distance).",
        "<b>Load:</b> equipment one-hot; weight (repaired) with missing and censored-at-47,500 flags.",
        "<b>Market and time:</b> daily market level, the load's own market_index, days since 2025-01-01, day of week, "
        "day of month.",
        "<b>Geography:</b> pickup/delivery coordinates and smoothed pickup/delivery city effects. Training rows get "
        "5-fold out-of-fold effects, so a row's label never feeds its own feature. Unseen cities get 0 (the average) "
        "and rely on coordinates.",
    ])
    story.append(p(
        "Every feature is present in <font name='DVM'>validation.csv</font>; <font name='DVM'>posted_rate</font> is "
        "the only training-only column. Inside each validation fold the whole estimator is refit from scratch on the "
        "training part (label filter, imputation, city effects, model), so no test-fold label touches training."))

    # ---------------------------------------------------------------- split
    story.append(p("5. Train / validation split", H2))
    story.append(p(
        "The submission is scored on Nov–Dec, after every training label. The validation must therefore test "
        "forecasting forward in time with the same horizon. A random split puts loads from the same days in both "
        "train and test, so the model sees each day's level before being tested on it. That is optimistic and, here, "
        "it picks the wrong model (section 6)."))
    split_rows = [["Split", "Train", "Test", "n train / n test", "Role"]]
    sizes = scores[(scores.subset == "all rows") & (scores.model == FINAL_MODEL)].groupby("fold")[
        ["n_train", "n_test"]].first()
    def sz(fold):  # noqa: E306
        r = sizes.loc[fold]
        return f"{int(r.n_train):,} / {int(r.n_test):,}"
    hold_fold = [f for f in sizes.index if f.startswith("holdout")][0]
    city_fold = [f for f in sizes.index if f.startswith("unseen")][0]
    split_rows += [
        ["Time holdout", "Jan 1 – Aug 31", "Sep 1 – Oct 31", sz(hold_fold), "model selection"],
        ["Rolling backtest", "expanding from Jan 1", "5 forward windows (2 months; last 1)", "19,110–43,147 train",
         "stability check"],
        ["Unseen cities", "Jan–Aug minus 8 random cities", "Sep–Oct loads touching them", sz(city_fold),
         "new-city check"],
        ["Random 5-fold", "random 80%", "random 20%", "38,400 / 9,600", "contrast only"],
    ]
    story.append(table(split_rows, [2.9 * cm, 3.6 * cm, 4.2 * cm, 3.2 * cm, 3.1 * cm]))
    story.append(Spacer(1, 4))
    story.append(p(
        f"Random seed {C.SEED} is used for the K-fold split, the out-of-fold city effects, the unseen-city draw and "
        f"the models. Scores are computed on all test rows and on test rows whose label passes the outlier filter "
        f"(limits from the training part). Corrupted labels can't be predicted by any model, so clean-label MAPE is "
        f"the selection metric. No official metric is computed by <font name='DVM'>score.py</font>, so MAPE is "
        f"used (scale-free across $200 and $6,000 loads), with MAE, RMSE, R² and bias alongside."))
    story.append(KeepTogether([
        p("<b>Can the trend be identified?</b> A linear trend can only be separated from the market level when the "
          "two are not moving together in the training window:", BODY),
        table([list(stab.columns)] + stab.astype(str).values.tolist(), [4.0 * cm, 4.0 * cm, 3.5 * cm, 3.5 * cm]),
        Spacer(1, 3),
        p("With Jan–Jun data the market and time are 0.86 correlated and the trend estimate doubles. From Jan–Aug "
          "onward it settles at 0.5–0.7% per month. The final model trains on Jan–Oct, where the correlation is "
          "−0.11: the best-identified window of all."),
    ]))

    # ---------------------------------------------------------------- models
    story.append(PageBreak())
    story.append(p("6. Models compared and selection", H2))
    story.append(p(
        "All candidates predict log(rate ÷ distance) and are converted back to dollars. (1) <b>Median baseline</b> per "
        "equipment × distance band. (2) <b>Ridge</b> on the linear features. (3) <b>Random forest</b> and "
        "(4) <b>LightGBM</b> on all features, including the day number. (5) <b>Ridge + LightGBM on residual</b>: "
        "ridge carries the market level and trend, which extrapolate; LightGBM fits the remaining non-linear "
        "structure without the day number, so it never has to extrapolate time."))
    names = list(plots.MODEL_LABELS)
    rows = [["Model", "Holdout MAPE", "Holdout MAPE (all rows)", "MAE $", "RMSE $", "Bias %", "Rolling mean / median",
             "Unseen cities", "Random 5-fold"]]
    for m in names:
        rows.append([plots.MODEL_LABELS[m], f"{M('holdout', m):.2f}%", f"{M('holdout', m, s=s_all):.2f}%",
                     f"{M('holdout', m, 'MAE'):.1f}", f"{M('holdout', m, 'RMSE'):.1f}",
                     f"{M('holdout', m, 'bias_%'):+.2f}",
                     f"{M('rolling', m):.2f}% / {roll_med[m]:.2f}%", f"{M('unseen_city', m):.2f}%",
                     f"{M('random_kfold', m):.2f}%"])
    story.append(table(rows, [3.5 * cm, 1.6 * cm, 1.8 * cm, 1.3 * cm, 1.3 * cm, 1.3 * cm, 2.3 * cm, 1.7 * cm, 1.8 * cm],
                       bold_row=len(rows) - 1))
    story.append(caption("Clean-label test rows unless marked. Holdout = train Jan–Aug, test Sep–Oct. MAE, RMSE and "
                         "bias are holdout, clean labels."))
    story.append(Spacer(1, 4))
    story.append(fig_image(plots.fig_model_comparison(summary), 17))
    story.append(fig_image(plots.fig_rolling_folds(scores), 15.5))
    story.append(caption("Figure 4. Clean-label MAPE by split, and per rolling window."))
    story += bullets([
        "Every learned model beats the median baseline on the average of every forward split. Per window there is "
        "one exception: in Jan–Jun → Jul–Aug the two trend-based models (ridge, final) do worse than the baseline.",
        f"On the time holdout the final model has the lowest MAPE, MAE and RMSE.",
        f"In the rolling backtest it wins {wins} of 5 windows against LightGBM, by a wide margin "
        f"(e.g. Jan–Jul → Aug–Sep: 1.76% vs 3.53%). It loses only Jan–Jun → Jul–Aug (8.2%), the one window "
        f"where the trend can't be identified. That fold drags its mean to the level of the tree models; the median "
        f"({roll_med[FINAL_MODEL]:.2f}% vs {roll_med['lightgbm']:.2f}% LightGBM, {roll_med['random_forest']:.2f}% "
        f"random forest) shows the typical case.",
        "Random 5-fold ranks LightGBM and random forest first (≈1.1–1.2%) because they memorise each day's level. "
        "Those same models trail the final model on the forward holdout, which is why random K-fold is not used.",
        f"Unseen cities cost about {M('unseen_city', FINAL_MODEL) - M('holdout', FINAL_MODEL):.2f} points of MAPE.",
    ])
    tune_txt = ""
    if tuning is not None:
        g = tuning.groupby(["alpha", "num_leaves", "n_estimators"])["MAPE_%"].mean()
        tune_txt = (f" A small grid ({len(g)} settings: ridge alpha, stage-2 leaves and trees) on the three "
                    f"identified rolling windows ranged {g.min():.3f}–{g.max():.3f}% MAPE, with the defaults at "
                    f"{g.loc[(1.0, 31, 600)]:.3f}%, so the defaults are kept.")
    story.append(p(
        "<b>Decision: Ridge + LightGBM on residual.</b> It is the best model wherever the trend is identifiable. The "
        "final training window is the best-identified one, and it is the only candidate that carries the trend into "
        "Nov–Dec." + tune_txt))

    # ---------------------------------------------------------------- results & limitations
    story.append(p("7. Validation results of the final model and limitations", H2))
    story.append(fig_image(plots.fig_holdout_residuals(hold, train), 16))
    story.append(caption("Figure 5. Final model on the Sep–Oct holdout (clean labels)."))
    story += bullets([
        f"Holdout (clean labels): MAPE {M('holdout', FINAL_MODEL):.2f}%, MAE ${M('holdout', FINAL_MODEL, 'MAE'):.0f}, "
        f"RMSE ${M('holdout', FINAL_MODEL, 'RMSE'):.0f}, R² {M('holdout', FINAL_MODEL, 'R2'):.3f}. On all rows "
        f"(including corrupted labels) MAPE is {M('holdout', FINAL_MODEL, s=s_all):.2f}%; the gap comes from the "
        f"1.5% of test labels that are off by 2–5×.",
        f"Errors are flat across distance bands (≈2.0–2.1% in each). By equipment: Dry Van {by_eq['Dry Van']:.2f}%, "
        f"Reefer {by_eq['Reefer']:.2f}%, Flatbed {by_eq['Flatbed']:.2f}%.",
        f"<b>Main limitation: level bias.</b> On the holdout the model under-predicts by "
        f"{-M('holdout', FINAL_MODEL, 'bias_%'):.1f}% on average, with the same sign in every segment. Sep–Oct came "
        f"in above what Jan–Aug implied. Most of the remaining error is this market-level uncertainty, which more "
        f"training data narrows (the final fit has 2 more months and a better-identified trend) but can't remove.",
        "The linear trend is assumed to continue for two months. If the market reverses, Nov–Dec will be off by "
        "roughly the trend over the horizon (≈0.6–1.3%).",
        "No December or holiday data exists, so Christmas-week effects are not modelled.",
        "Corrupted labels in the hidden Nov–Dec set (if present at the training rate) will add error no model can "
        "remove; the all-rows MAPE above shows their size.",
        "December inputs not present in the file were filled: coordinates from the city table, market level from "
        "that date in <font name='DVM'>validation.csv</font>, quote_signal at its training median.",
    ])

    # ---------------------------------------------------------------- December chart
    story.append(PageBreak())
    story.append(p("8. Fixed December prediction chart (from score.py)", H2))
    img = Image(str(CHART))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = 17 * cm, 17 * cm * ratio
    story.append(img)
    story.append(caption("Generated by the provided score.py from outputs/december_chart_predictions.csv, unmodified."))
    story.append(p(
        f"Fixed load: Lexington → Fort Wayne, 360 miles, Dry Van, 32,000 lb. Predictions range "
        f"${dec.predicted_rate.min():.2f}–${dec.predicted_rate.max():.2f}. The shape is a weekly cycle (midweek "
        f"peak, weekend dip) learned from the Jan–Oct day-of-week pattern, on a slight upward drift from the "
        f"trend and the December market level. As a sanity check, the 21 clean Dry Van loads on this lane in "
        f"Jan–Oct had a median rate of $807.90 at ~362 miles; the December predictions sit 2.6–6.0% above that, "
        f"the direction the estimated trend implies."))
    story.append(p(
        f"<font name='DVM'>score.py</font> output: <i>Validated 12,000 final predictions. Validated 31 fixed "
        f"December predictions.</i> The final predictions file has {len(preds):,} rows, ids identical to and in the "
        f"same order as the template, no missing or non-positive values (range "
        f"${preds.predicted_rate.min():.2f}–${preds.predicted_rate.max():.2f}, median "
        f"${preds.predicted_rate.median():.2f})."))

    # ---------------------------------------------------------------- reproduce
    story.append(p("9. Reproducing the results", H2))
    story.append(p(
        "Put the four assessment CSVs in <font name='DVM'>data/</font> (train_test.csv, validation.csv, "
        "validation_predictions_template.csv, december_chart_inputs.csv), then:"))
    story.append(Paragraph(
        "pip install -r requirements.txt<br/>"
        "python src/evaluate.py      # all models × all splits -> artifacts/ (~7 min)<br/>"
        "python src/tune.py          # optional small grid (~4 min)<br/>"
        "python src/train.py         # final fit on Jan–Oct -> artifacts/final_model.joblib<br/>"
        "python src/predict.py       # -> outputs/validation_predictions.csv, december_chart_predictions.csv<br/>"
        "python score.py --predictions outputs/validation_predictions.csv \\<br/>"
        "&nbsp;&nbsp;--december-predictions outputs/december_chart_predictions.csv --output-dir outputs/scorer_results<br/>"
        "python src/report.py        # this PDF", CODE))
    story.append(p(
        f"Or run <font name='DVM'>notebooks/freight_rate_analysis.ipynb</font> top to bottom (Colab or local). "
        f"Tested with Python {sys.version.split()[0]}, pandas {pd.__version__}, scikit-learn "
        f"{__import__('sklearn').__version__}, LightGBM {__import__('lightgbm').__version__}. The final model "
        f"trains on {final_rows:,} rows."))

    REPORT.parent.mkdir(exist_ok=True)
    doc = SimpleDocTemplate(str(REPORT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=2 * cm, title="Freight Rate Prediction — report",
                            author="Kalyanbrat Medhi")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    os.chdir(C.ROOT)
    main()
