"""Figures shared by the notebook and the PDF report. Every function returns a matplotlib Figure
built from real data/artifacts; nothing is hard-coded."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from . import config as C
from .data import clean_label_mask, segment_ratio

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#a3a29c"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 170, "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.edgecolor": NEUTRAL,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.7, "lines.linewidth": 2,
    "legend.frameon": False,
})

MODEL_LABELS = {
    "median_baseline": "Median baseline",
    "ridge": "Ridge",
    "random_forest": "Random forest",
    "lightgbm": "LightGBM",
    "ridge_plus_lightgbm": "Ridge + LightGBM (final)",
}


def fig_target_outliers(train: pd.DataFrame):
    """Rate per mile relative to its segment median: the clean mass vs the corrupted clusters."""
    ratio = segment_ratio(train)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    bins = np.logspace(np.log10(0.15), np.log10(6), 90)
    ax.hist(ratio, bins=bins, color=BLUE, edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    for x in (C.OUTLIER_LOW, C.OUTLIER_HIGH):
        ax.axvline(x, color=ORANGE, lw=1.5, ls="--")
    ax.set_xticks([0.2, 0.3, 0.5, 0.6, 1, 1.6, 2, 3, 5])
    ax.set_xticklabels(["0.2", "0.3", "0.5", "0.6", "1", "1.6", "2", "3", "5"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    n_bad = int((~clean_label_mask(train)).sum())
    ax.set_title(f"Rate per mile ÷ segment median — {n_bad} labels (1.4%) fall outside the dashed limits")
    ax.set_xlabel("ratio to (equipment × distance band) median, log scale")
    ax.set_ylabel("loads (log scale)")
    fig.tight_layout()
    return fig


def fig_market_and_level(train, val, market_level, daily_residual):
    """Top: daily market level across train and validation dates.
    Bottom: daily mean log rate-per-mile residual (after distance/equipment/weight/city) in train."""
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    ml = market_level.sort_index()
    cut = pd.Timestamp("2025-11-01")
    axes[0].plot(ml[ml.index < cut], color=BLUE, lw=1.6, label="train dates")
    axes[0].plot(ml[ml.index >= cut], color=ORANGE, lw=1.6, label="validation dates (Nov–Dec)")
    axes[0].set_title("Daily market level (mean market_index per date)")
    axes[0].legend(loc="upper right")
    r = daily_residual.sort_index()
    axes[1].plot(r.index, 100 * r.values, color=BLUE, lw=1.0, alpha=0.5, label="daily")
    axes[1].plot(r.rolling(14, center=True).mean() * 100, color=BLUE, lw=2.2, label="14-day mean")
    axes[1].set_title("Rate level after distance, equipment, weight and city effects (train)")
    axes[1].set_ylabel("% vs average")
    axes[1].legend(loc="upper left")
    for ax in axes:
        ax.axvline(cut, color=NEUTRAL, lw=1, ls=":")
    fig.tight_layout()
    return fig


def fig_effects(clean: pd.DataFrame, residual: pd.Series):
    """Residual effect of weight decile and day of week (percent)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    w = clean["weight"].abs()
    by_w = residual.groupby(pd.qcut(w, 10, labels=False)).mean() * 100
    mids = w.groupby(pd.qcut(w, 10, labels=False)).median() / 1000
    axes[0].plot(mids, by_w, color=BLUE, marker="o", ms=5)
    axes[0].set_title("Heavier loads cost more per mile")
    axes[0].set_xlabel("weight decile median (thousand lb)")
    axes[0].set_ylabel("% vs average")
    dow = residual.groupby(clean[C.DATE].dt.dayofweek).mean() * 100
    axes[1].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], dow, color=BLUE, width=0.6)
    axes[1].axhline(0, color=NEUTRAL, lw=0.8)
    axes[1].set_title("Midweek peak, weekend dip")
    axes[1].set_ylabel("% vs average")
    fig.tight_layout()
    return fig


def fig_model_comparison(summary: pd.DataFrame):
    """Clean-label MAPE per model for each split (one panel per split, one hue, final highlighted)."""
    s = summary[summary["subset"] == "clean labels"]
    splits = [("holdout", "Time holdout\nJan–Aug → Sep–Oct"), ("rolling", "Rolling backtest\n5 windows (mean)"),
              ("unseen_city", "Unseen cities\n8 removed"), ("random_kfold", "Random 5-fold\n(contrast only)")]
    order = list(MODEL_LABELS)
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.9), sharey=True)
    for ax, (key, title) in zip(axes, splits):
        v = s[s["split"] == key].set_index("model").reindex(order)["MAPE_%"]
        colors = [ORANGE if m == "ridge_plus_lightgbm" else BLUE for m in order]
        ax.barh(range(len(order)), v.values, color=colors, height=0.62)
        for i, x in enumerate(v.values):
            ax.text(x + 0.08, i, f"{x:.2f}", va="center", fontsize=8, color=INK)
        ax.set_title(title, fontsize=8.5)
        ax.grid(axis="x", color=GRID)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(v.max() * 1.35, 1))
        ax.set_xlabel("MAPE %")
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([MODEL_LABELS[m] for m in order])
    axes[0].invert_yaxis()
    fig.tight_layout()
    return fig


def fig_rolling_folds(scores: pd.DataFrame):
    """MAPE per rolling window for LightGBM vs the final model (shows the one collinear window)."""
    s = scores[(scores["split"] == "rolling") & (scores["subset"] == "clean labels")]
    s = s[s["model"].isin(["lightgbm", "ridge_plus_lightgbm", "median_baseline"])]
    p = s.pivot_table(index="fold", columns="model", values="MAPE_%")
    order = ["Jan-Apr -> May-Jun", "Jan-Jun -> Jul-Aug", "Jan-Jul -> Aug-Sep", "Jan-Aug -> Sep-Oct", "Jan-Sep -> Oct"]
    p = p.reindex([f"rolling {o}" for o in order])
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    w = 0.26
    for i, (m, col) in enumerate([("median_baseline", NEUTRAL), ("lightgbm", BLUE), ("ridge_plus_lightgbm", ORANGE)]):
        ax.bar(x + (i - 1) * w, p[m], width=w - 0.03, color=col, label=MODEL_LABELS[m])
    ax.set_xticks(x)
    ax.set_xticklabels([o.replace(" -> ", " → ") for o in order])
    ax.set_ylabel("MAPE % (clean labels)")
    ax.set_title("Rolling backtest by window")
    ax.set_ylim(0, float(np.nanmax(p.values)) * 1.3)
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    fig.tight_layout()
    return fig


def fig_holdout_residuals(hold: pd.DataFrame, raw: pd.DataFrame):
    """Final-model holdout: weekly bias over time and absolute % error by distance band."""
    h = hold[(hold["model"] == "ridge_plus_lightgbm") & hold["clean"]].merge(
        raw[[C.ID, C.DATE, "distance", "equipment"]], on=C.ID)
    h["pct"] = 100 * (h["pred"] / h["y"] - 1)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    wk = h.groupby(pd.Grouper(key=C.DATE, freq="W"))["pct"].mean()
    axes[0].plot(wk.index, wk.values, color=ORANGE, marker="o", ms=5)
    axes[0].axhline(0, color=NEUTRAL, lw=0.8)
    axes[0].set_title("Weekly bias, Sep–Oct holdout")
    axes[0].set_ylabel("mean % error (pred vs actual)")
    axes[0].tick_params(axis="x", rotation=30)
    band = pd.cut(h["distance"], [0, 250, 500, 1000, 1500, 2000, 4000],
                  labels=["<250", "250–500", "500–1k", "1k–1.5k", "1.5k–2k", "2k+"])
    eq = h.groupby([band, "equipment"], observed=True)["pct"].apply(lambda e: e.abs().mean()).unstack()
    xs = np.arange(len(eq))
    for i, (e, col) in enumerate(zip(["Dry Van", "Flatbed", "Reefer"], [BLUE, ORANGE, AQUA])):
        axes[1].plot(xs, eq[e].values, color=col, marker="o", ms=5, label=e)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(eq.index, fontsize=8)
    axes[1].set_title("MAPE by distance band (miles)")
    axes[1].set_ylabel("MAPE %")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return fig
