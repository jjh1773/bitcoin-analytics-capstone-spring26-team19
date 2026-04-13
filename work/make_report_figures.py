"""
Generate 5 missing figures for the Layer 1 report.

Saves to: work/output/report_figures/
  fig1_cycle_returns.png      — 사이클 구간별 BTC 30일 수익률 분포
  fig2_mvrv_returns.png       — MVRV 구간별 미래 수익률
  fig3_exchange_returns.png   — 거래소 유출 vs 미래 수익률
  fig4_ablation_bar.png       — 7개 신호 ablation 막대 차트
  fig5_stage_progression.png  — V0→V3 단계별 누적수익률 비교

Run from project root:
    python -m work.make_report_figures
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.WARNING)

from template.prelude_template import load_data
from work.model_development import (
    precompute_features,
    compute_cycle_position,
    HALVING_DATES,
    FLOW_IN_COL,
    FLOW_OUT_COL,
)

OUT = Path(__file__).parent / "output" / "report_figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 180,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

C_BUY    = "#059669"
C_SELL   = "#dc2626"
C_NEUT   = "#6b7280"
C_BLUE   = "#2563eb"
C_ORANGE = "#d97706"
C_PURPLE = "#7c3aed"

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
btc      = load_data()
features = precompute_features(btc)
price    = btc["PriceUSD_coinmetrics"].dropna()
print(f"  Price: {len(price)} rows | Features: {len(features)} rows\n")


# ==============================================================================
# FIG 1 — 사이클 구간별 BTC 30일 수익률 분포
# ==============================================================================
def make_fig1():
    print("Fig 1: Cycle phase vs 30-day returns...")

    dates      = price.index
    cycle_pos  = compute_cycle_position(dates)
    fwd_30d    = price.pct_change(30).shift(-30) * 100   # look-ahead (EDA only)

    df = pd.DataFrame({
        "cycle_pos": cycle_pos,
        "fwd_30d":   fwd_30d,
        "price":     price,
    }).dropna()

    # 5 phases matching our signal design
    bins   = [0, 0.15, 0.50, 0.80, 0.92, 1.01]
    labels = [
        "Post-Halving\n(0–15%)\nNeutral",
        "Bull Market\n(15–50%)\nReduce",
        "Bear Market\n(50–80%)\nBUY",
        "Recovery\n(80–92%)\nMild Buy",
        "Pre-Halving\n(92–100%)\nNeutral",
    ]
    colors = [C_NEUT, C_SELL, C_BUY, C_BLUE, C_NEUT]

    df["phase"] = pd.cut(df["cycle_pos"], bins=bins, labels=labels, right=False)
    df = df.dropna(subset=["phase"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: box plot of 30-day returns by phase
    ax = axes[0]
    phase_order = labels
    data_by_phase = [df[df["phase"] == p]["fwd_30d"].dropna().values for p in phase_order]

    bp = ax.boxplot(
        data_by_phase,
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2.5),
        whiskerprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=2, alpha=0.3),
        widths=0.55,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(phase_order, fontsize=8.5)
    ax.set_ylabel("30-Day Forward Return (%)", fontsize=11)
    ax.set_title("BTC 30-Day Forward Returns by\nHalving Cycle Phase", fontweight="bold")

    # Annotate medians
    for i, (data, color) in enumerate(zip(data_by_phase, colors), 1):
        med = np.median(data)
        ax.text(i, med + 4, f"{med:+.1f}%", ha="center", va="bottom",
                fontsize=8.5, color=color, fontweight="bold")

    # Right: mean return + 95% CI per phase (bar)
    ax = axes[1]
    means, cis, counts = [], [], []
    for data in data_by_phase:
        boot = [np.mean(np.random.choice(data, len(data), replace=True))
                for _ in range(2000)]
        means.append(np.mean(data))
        cis.append((np.mean(data) - np.percentile(boot, 2.5),
                    np.percentile(boot, 97.5) - np.mean(data)))
        counts.append(len(data))

    x      = np.arange(len(labels))
    bar_c  = [C_BUY if m > 0 else C_SELL for m in means]
    bars   = ax.bar(x, means, color=bar_c, alpha=0.80, width=0.6)
    ax.errorbar(x, means,
                yerr=np.array(cis).T,
                fmt="none", color="black", capsize=5, linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Mean 30-Day Forward Return (%)", fontsize=11)
    ax.set_title("Mean Return + 95% Bootstrap CI\nby Cycle Phase", fontweight="bold")

    for bar, mean, n in zip(bars, means, counts):
        ax.text(bar.get_x() + bar.get_width()/2,
                mean + (1.5 if mean >= 0 else -3.5),
                f"n={n}", ha="center", fontsize=8, color="black")

    plt.suptitle(
        "Hypothesis Validation: Bear Market Phase (50–80%) Yields Highest Forward Returns",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    path = OUT / "fig1_cycle_returns.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # Print stats
    print("  Phase stats (mean 30d fwd return):")
    for label, data in zip(labels, data_by_phase):
        print(f"    {label[:20]:20s}: {np.mean(data):+.2f}%  (n={len(data)})")


# ==============================================================================
# FIG 2 — MVRV 구간별 미래 수익률
# ==============================================================================
def make_fig2():
    print("Fig 2: MVRV zones vs forward returns...")

    feat    = features.copy()
    fwd_30d = price.pct_change(30).shift(-30) * 100
    fwd_90d = price.pct_change(90).shift(-90) * 100

    df = pd.DataFrame({
        "mvrv_z":  feat["mvrv_zscore"],
        "fwd_30d": fwd_30d.reindex(feat.index),
        "fwd_90d": fwd_90d.reindex(feat.index),
    }).dropna()

    # MVRV zones
    zone_bins   = [-np.inf, -2, -1, 0, 1, 1.5, 2.5, np.inf]
    zone_labels = [
        "Deep Value\n(<-2)",
        "Value\n(-2 to -1)",
        "Mild Value\n(-1 to 0)",
        "Neutral\n(0 to 1)",
        "Caution\n(1 to 1.5)",
        "Danger\n(1.5 to 2.5)",
        "Extreme\n(>2.5)",
    ]
    zone_colors = [
        "#065f46", C_BUY, "#6ee7b7",
        C_NEUT,
        "#fca5a5", C_SELL, "#7f1d1d"
    ]

    df["zone"] = pd.cut(df["mvrv_z"], bins=zone_bins, labels=zone_labels)
    df = df.dropna(subset=["zone"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, col, horizon in zip(axes, ["fwd_30d", "fwd_90d"], ["30-Day", "90-Day"]):
        means, cis = [], []
        counts = []
        for zone in zone_labels:
            data = df[df["zone"] == zone][col].dropna().values
            if len(data) < 5:
                means.append(0); cis.append((0, 0)); counts.append(0)
                continue
            boot = [np.mean(np.random.choice(data, len(data), replace=True))
                    for _ in range(1500)]
            means.append(np.mean(data))
            cis.append((np.mean(data) - np.percentile(boot, 2.5),
                        np.percentile(boot, 97.5) - np.mean(data)))
            counts.append(len(data))

        x    = np.arange(len(zone_labels))
        bars = ax.bar(x, means, color=zone_colors, alpha=0.82, width=0.65)
        ax.errorbar(x, means, yerr=np.array(cis).T,
                    fmt="none", color="black", capsize=4, linewidth=1.5)
        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")

        ax.set_xticks(x)
        ax.set_xticklabels(zone_labels, fontsize=8.5)
        ax.set_ylabel(f"Mean {horizon} Forward Return (%)")
        ax.set_title(f"MVRV Z-Score Zone → {horizon} Forward Return\n(+95% Bootstrap CI)",
                     fontweight="bold")

        for bar, mean, n in zip(bars, means, counts):
            if n > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        mean + (1 if mean >= 0 else -3),
                        f"n={n}", ha="center", fontsize=7.5)

    plt.suptitle(
        "MVRV Z-Score as Valuation Signal: Deep Value Zones Predict Strong Forward Returns",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    path = OUT / "fig2_mvrv_returns.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ==============================================================================
# FIG 3 — 거래소 유출 vs 미래 수익률
# ==============================================================================
def make_fig3():
    print("Fig 3: Exchange flow vs forward returns...")

    feat    = features.copy()
    fwd_30d = price.pct_change(30).shift(-30) * 100

    df = pd.DataFrame({
        "exchange_signal": feat["exchange_signal"],
        "fwd_30d":         fwd_30d.reindex(feat.index),
        "price":           price.reindex(feat.index),
    }).dropna()
    df = df.loc["2018-01-01":]   # exchange flow signal reliable from 2018

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Left: scatter with rolling average
    ax = axes[0]
    ax.scatter(df["exchange_signal"], df["fwd_30d"],
               alpha=0.08, color=C_BLUE, s=8)
    # Rolling mean line
    df_s  = df.sort_values("exchange_signal")
    bins  = pd.qcut(df_s["exchange_signal"], q=20, duplicates="drop")
    means = df_s.groupby(bins, observed=True)["fwd_30d"].mean()
    mids  = df_s.groupby(bins, observed=True)["exchange_signal"].mean()
    ax.plot(mids.values, means.values, color=C_ORANGE, linewidth=2.5, label="Quintile mean")
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
    corr = df["exchange_signal"].corr(df["fwd_30d"])
    ax.set_xlabel("Exchange Flow Signal (–1=inflow, +1=outflow)", fontsize=10)
    ax.set_ylabel("30-Day Forward Return (%)", fontsize=10)
    ax.set_title(f"Scatter: Exchange Signal vs\n30d Forward Return  (r={corr:+.3f})",
                 fontweight="bold")
    ax.legend(fontsize=9)

    # Middle: quintile bar chart
    ax = axes[1]
    df["quintile"] = pd.qcut(df["exchange_signal"], q=5,
                              labels=["Q1\nStrong\nInflow", "Q2", "Q3\nNeutral",
                                      "Q4", "Q5\nStrong\nOutflow"])
    q_mean = df.groupby("quintile", observed=True)["fwd_30d"].mean()
    q_n    = df.groupby("quintile", observed=True)["fwd_30d"].count()
    bar_c  = [C_BUY if v > 0 else C_SELL for v in q_mean.values]
    bars   = ax.bar(range(5), q_mean.values, color=bar_c, alpha=0.82, width=0.6)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xticks(range(5))
    ax.set_xticklabels(q_mean.index, fontsize=8.5)
    ax.set_ylabel("Mean 30-Day Forward Return (%)", fontsize=10)
    ax.set_title("Exchange Signal Quintile →\nMean 30-Day Return", fontweight="bold")
    for bar, mean, n in zip(bars, q_mean.values, q_n.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                mean + (0.5 if mean >= 0 else -2),
                f"n={n}", ha="center", fontsize=8)

    # Right: time series of signal with BTC price
    ax = axes[2]
    ax2 = ax.twinx()
    date_idx = df.index
    ax.fill_between(date_idx, df["exchange_signal"], 0,
                    where=(df["exchange_signal"] > 0),
                    alpha=0.35, color=C_BUY, label="Net Outflow (buy)")
    ax.fill_between(date_idx, df["exchange_signal"], 0,
                    where=(df["exchange_signal"] < 0),
                    alpha=0.35, color=C_SELL, label="Net Inflow (sell)")
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_ylabel("Exchange Flow Signal", fontsize=10, color=C_BLUE)
    ax.set_ylim(-1.2, 1.2)
    ax2.semilogy(date_idx, price.reindex(date_idx), color=C_ORANGE,
                 linewidth=1.2, alpha=0.8, label="BTC Price")
    ax2.set_ylabel("BTC Price (log)", fontsize=10, color=C_ORANGE)
    ax.set_title("Exchange Flow Signal\nvs BTC Price (2018–2025)", fontweight="bold")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.suptitle(
        "Net Exchange Outflow Signal: HODLer Accumulation Predicts Positive Forward Returns",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    path = OUT / "fig3_exchange_returns.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ==============================================================================
# FIG 4 — 7개 신호 ablation 막대 차트
# ==============================================================================
def make_fig4():
    print("Fig 4: Signal ablation bar chart...")

    # Results from signal_ablation.py (extra_weight=0.10, baseline=1.05558)
    baseline_ratio = 1.05558
    signals = [
        ("HashRate\nPuell Multiple",   "CoinMetrics",  +0.300,  +1.24, C_BUY),
        ("Funding Rate",               "Binance",       +0.064,  +0.54, C_BUY),
        ("Google Trends\n'bitcoin'",   "pytrends",      -0.029,  +0.24, C_SELL),
        ("NVT Ratio",                  "CoinMetrics",   -0.253,  +0.14, C_SELL),
        ("Panic Volume",               "CoinMetrics",   -0.519,  +0.14, C_SELL),
        ("Stablecoin\nSupply",         "DeFiLlama",     -0.424,  -0.66, C_SELL),
        ("Exchange\nSupply Level",     "CoinMetrics",   -0.778,  -0.36, C_SELL),
    ]
    # Sort by delta (descending)
    signals.sort(key=lambda x: x[2], reverse=True)

    labels  = [s[0] for s in signals]
    sources = [s[1] for s in signals]
    deltas  = [s[2] for s in signals]
    wr_chg  = [s[3] for s in signals]
    colors  = [s[4] for s in signals]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: cumulative return delta
    ax = axes[0]
    x    = np.arange(len(labels))
    bars = ax.barh(x, deltas, color=colors, alpha=0.82, height=0.6)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(x)
    ax.set_yticklabels([f"{l}\n({s})" for l, s in zip(labels, sources)], fontsize=9)
    ax.set_xlabel("Δ Mean Ratio vs Baseline (%)", fontsize=11)
    ax.set_title("Signal Ablation: Δ Cumulative Return\n(vs Baseline, extra_weight=0.10)",
                 fontweight="bold")
    for bar, val in zip(bars, deltas):
        ax.text(val + (0.01 if val >= 0 else -0.01),
                bar.get_y() + bar.get_height()/2,
                f"{val:+.3f}%",
                va="center", ha="left" if val >= 0 else "right",
                fontsize=9, fontweight="bold")

    # Baseline annotation
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlim(-0.95, 0.50)

    # Right: win rate change
    ax = axes[1]
    wr_colors = [C_BUY if v > 0 else C_SELL for v in wr_chg]
    bars2 = ax.barh(x, wr_chg, color=wr_colors, alpha=0.82, height=0.6)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(x)
    ax.set_yticklabels([f"{l}" for l in labels], fontsize=9)
    ax.set_xlabel("Δ Win Rate vs Baseline (%p)", fontsize=11)
    ax.set_title("Signal Ablation: Δ Win Rate\n(vs Baseline, extra_weight=0.10)",
                 fontweight="bold")
    for bar, val in zip(bars2, wr_chg):
        ax.text(val + (0.02 if val >= 0 else -0.02),
                bar.get_y() + bar.get_height()/2,
                f"{val:+.2f}%p",
                va="center", ha="left" if val >= 0 else "right",
                fontsize=9, fontweight="bold")

    accepted  = mpatches.Patch(color=C_BUY, alpha=0.82, label="Accepted (positive impact)")
    rejected  = mpatches.Patch(color=C_SELL, alpha=0.82, label="Rejected (negative impact)")
    fig.legend(handles=[accepted, rejected], loc="lower center",
               ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.04))

    plt.suptitle(
        "External Signal Ablation: 7 Candidates Tested — Only Puell Multiple Improves Performance",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    path = OUT / "fig4_ablation_bar.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ==============================================================================
# FIG 5 — V0→V3 단계별 누적수익률 비교
# ==============================================================================
def make_fig5():
    print("Fig 5: Stage progression chart...")

    # Stage results from compare_all_stages.py
    stages = {
        "V0\nMVRV + MA\n(Baseline)": {
            "mean": 1.038844, "wr": 60.86,
            "by_year": {2018:1.21176, 2019:1.28547, 2020:0.77362,
                        2021:1.08595, 2022:0.82090, 2023:1.01248,
                        2024:1.08059, 2025:1.08777},
        },
        "V1\n+ Cycle + Exchange\n(On-chain only)": {
            "mean": 1.053847, "wr": 62.84,
            "by_year": {2018:1.24333, 2019:1.29578, 2020:0.79319,
                        2021:1.10703, 2022:0.84790, 2023:1.01112,
                        2024:1.07789, 2025:1.09154},
        },
        "V2\n+ Monetary + Fear\n(+ External data)": {
            "mean": 1.055577, "wr": 62.96,
            "by_year": {2018:1.24283, 2019:1.28469, 2020:0.80611,
                        2021:1.10674, 2022:0.87581, 2023:0.99313,
                        2024:1.07888, 2025:1.09660},
        },
        "V3\n+ Puell Multiple\n(Final model)": {
            "mean": 1.058583, "wr": 64.20,
            "by_year": {2018:1.24609, 2019:1.26947, 2020:0.81526,
                        2021:1.12555, 2022:0.89924, 2023:0.97774,
                        2024:1.07612, 2025:1.09291},
        },
    }

    stage_labels = list(stages.keys())
    stage_colors = ["#94a3b8", "#60a5fa", C_ORANGE, C_BUY]
    means  = [stages[s]["mean"] for s in stage_labels]
    wrs    = [stages[s]["wr"]   for s in stage_labels]
    years  = sorted(stages["V0\nMVRV + MA\n(Baseline)"]["by_year"].keys())

    fig = plt.figure(figsize=(16, 10))
    gs  = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.35)

    # ── Top-left: overall mean ratio bar ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    x   = np.arange(len(stage_labels))
    bars = ax1.bar(x, [(m-1)*100 for m in means],
                   color=stage_colors, alpha=0.85, width=0.55)
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.split("\n")[0] for s in stage_labels], fontsize=10)
    ax1.set_ylabel("Cumulative Return vs Uniform DCA (%)", fontsize=10)
    ax1.set_title("Overall Cumulative Return by Stage", fontweight="bold")
    ax1.axhline(0, color="black", linewidth=0.8)
    for bar, m in zip(bars, means):
        val = (m - 1) * 100
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                 f"+{val:.3f}%", ha="center", va="bottom",
                 fontsize=9.5, fontweight="bold")
    # Delta arrows
    for i in range(1, len(means)):
        delta = (means[i] - means[i-1]) * 100
        mid_x = (x[i-1] + x[i]) / 2
        ax1.annotate(f"+{delta:.3f}%",
                     xy=(x[i], (means[i]-1)*100), xytext=(mid_x, (means[i]-1)*100 + 0.15),
                     fontsize=8, color=C_BUY, ha="center",
                     arrowprops=dict(arrowstyle="-", color=C_BUY, lw=0.8))

    # ── Top-right: win rate bar ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(x, wrs, color=stage_colors, alpha=0.85, width=0.55)
    ax2.axhline(50, color=C_SELL, linewidth=1, linestyle="--", label="50% threshold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.split("\n")[0] for s in stage_labels], fontsize=10)
    ax2.set_ylabel("Win Rate (%)", fontsize=10)
    ax2.set_ylim(55, 68)
    ax2.set_title("Win Rate by Stage", fontweight="bold")
    ax2.legend(fontsize=9)
    for bar, wr in zip(bars2, wrs):
        ax2.text(bar.get_x() + bar.get_width()/2, wr + 0.1,
                 f"{wr:.2f}%", ha="center", va="bottom",
                 fontsize=9.5, fontweight="bold")

    # ── Bottom: year-by-year mean ratio ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :])
    yr_x = np.arange(len(years))
    width = 0.20
    offsets = [-1.5, -0.5, 0.5, 1.5]

    for i, (stage, color, offset) in enumerate(zip(stage_labels, stage_colors, offsets)):
        vals = [(stages[stage]["by_year"].get(yr, 1.0) - 1) * 100 for yr in years]
        bars3 = ax3.bar(yr_x + offset * width, vals,
                        width=width, color=color, alpha=0.82,
                        label=stage.replace("\n", " ").strip())

    ax3.axhline(0, color="black", linewidth=1)
    ax3.set_xticks(yr_x)
    ax3.set_xticklabels(years, fontsize=10)
    ax3.set_ylabel("Cumulative Return vs Uniform DCA (%)", fontsize=10)
    ax3.set_title("Year-by-Year Cumulative Return: Stage Progression", fontweight="bold")
    ax3.legend(fontsize=8.5, loc="upper right", ncol=2)

    # Annotate 2022 improvement
    ax3.annotate(
        "2022: +7.8% improvement\nV0→V3 (Fed rate hike cycle\n+ Puell capitulation)",
        xy=(yr_x[years.index(2022)] + 1.5*width, (0.89924-1)*100),
        xytext=(yr_x[years.index(2022)] - 0.5, -28),
        fontsize=8, color=C_BUY,
        arrowprops=dict(arrowstyle="->", color=C_BUY),
    )

    plt.suptitle(
        "Model Evolution: Incremental Contribution of Each Signal Group\n"
        "V0 (MVRV+MA) → V1 (+Cycle+Exchange) → V2 (+Macro+Fear) → V3 (+Puell)",
        fontsize=13, fontweight="bold"
    )

    path = OUT / "fig5_stage_progression.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()

    print(f"\n{'='*55}")
    print(f"All 5 figures saved to: {OUT}")
    print(f"{'='*55}")
    for p in sorted(OUT.glob("*.png")):
        size_kb = p.stat().st_size // 1024
        print(f"  {p.name:<35} {size_kb:>5} KB")
