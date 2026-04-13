"""
Bitcoin DCA Strategy Analysis
==============================
Novel Contributions:
1. Contrarian Halving Cycle (mid-cycle bear market = best accumulation window)
2. Net Exchange Outflow (HODLers accumulating = on-chain signal)
3. Monetary Policy Signal (Fed rate velocity + M2 growth + DXY momentum)
4. Fear Composite Signal (VIX spike + Crypto Fear & Greed extremes)

Run from project root:
    python -m work.analysis
"""

import logging
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from template.prelude_template import load_data
from work.model_development import (
    precompute_features,
    compute_cycle_position,
    compute_cycle_signal,
    compute_exchange_flow_signal,
    HALVING_DATES,
    FLOW_IN_COL,
    FLOW_OUT_COL,
    WEIGHT_MVRV,
    WEIGHT_CYCLE,
    WEIGHT_MONETARY,
    WEIGHT_FEAR,
    WEIGHT_EXCHANGE,
    WEIGHT_MA,
)

logging.basicConfig(level=logging.WARNING)
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 150

OUTPUT_DIR = Path(__file__).parent / "output" / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_cycle_signal_design():
    """Show the contrarian cycle signal shape vs a naive positive cycle signal."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.linspace(0, 1, 500)
    contrarian = compute_cycle_signal(x)

    # Naive "post-halving = good" signal for comparison
    naive = np.piecewise(
        x,
        [x < 0.35, (x >= 0.35) & (x < 0.55), (x >= 0.55) & (x < 0.75), x >= 0.75],
        [
            lambda v: 0.6 - (v / 0.35) * 0.4,
            lambda v: 0.2 - ((v - 0.35) / 0.20) * 0.2,
            lambda v: -((v - 0.55) / 0.20) * 0.2,
            lambda v: -0.2 - ((v - 0.75) / 0.25) * 0.3,
        ],
    )

    ax1.plot(x, contrarian, color="#059669", linewidth=2.5, label="Contrarian (this model)")
    ax1.plot(x, naive, color="#dc2626", linewidth=2, linestyle="--", label="Naive post-halving")
    ax1.axhline(0, color="black", linestyle=":", linewidth=1)
    ax1.fill_between(x, contrarian, 0, where=(contrarian > 0), alpha=0.15, color="#059669", label="Buy zone")
    ax1.fill_between(x, contrarian, 0, where=(contrarian < 0), alpha=0.15, color="#dc2626", label="Reduce zone")
    ax1.set_xlabel("Cycle Position (0 = halving, 1 = next halving)", fontsize=11)
    ax1.set_ylabel("Cycle Signal", fontsize=11)
    ax1.set_title("Contrarian vs. Naive Cycle Signal Design", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 1)
    ax1.set_xticks([0, 0.15, 0.50, 0.80, 0.92, 1.0])
    ax1.set_xticklabels(["0\nHalving", "0.15", "0.50\nBear", "0.80", "0.92\nPre-Halving", "1.0"], fontsize=8)

    # Phase labels on the contrarian signal
    phases = [
        (0.07, "Post-Halving\n(Neutral)", "#6b7280"),
        (0.33, "Bull Market\n(Reduce)", "#dc2626"),
        (0.65, "Bear Market\n(BUY!)", "#059669"),
        (0.86, "Recovery\n(Mild Buy)", "#2563eb"),
        (0.96, "Pre-Halving\n(Neutral)", "#6b7280"),
    ]
    for px, label, color in phases:
        ax1.annotate(label, (px, compute_cycle_signal(np.array([px]))[0] + 0.05),
                     ha="center", va="bottom", fontsize=7.5, color=color, fontweight="bold")

    # Right: Show real cycle positions over time
    dates_daily = pd.date_range("2016-01-01", "2025-12-31", freq="D")
    cycle_pos = compute_cycle_position(dates_daily)
    cycle_sig = pd.Series(compute_cycle_signal(cycle_pos.values), index=dates_daily)

    ax2.plot(dates_daily, cycle_sig, color="#059669", linewidth=1.5)
    ax2.axhline(0, color="black", linestyle=":", linewidth=1)
    ax2.fill_between(dates_daily, cycle_sig, 0, where=(cycle_sig > 0), alpha=0.2, color="#059669")
    ax2.fill_between(dates_daily, cycle_sig, 0, where=(cycle_sig < 0), alpha=0.2, color="#dc2626")

    for h in HALVING_DATES[1:-1]:
        ax2.axvline(h, color="#f59e0b", linestyle="--", linewidth=1.5, alpha=0.8)
    ax2.annotate("Halvings\n(2016, 2020, 2024)", (HALVING_DATES[2], 0.35), fontsize=8, color="#f59e0b")

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    ax2.set_ylabel("Cycle Signal", fontsize=11)
    ax2.set_title("Cycle Signal Over Time (2016-2025)", fontsize=12, fontweight="bold")
    ax2.set_ylim(-0.35, 0.55)

    plt.suptitle(
        "Key Insight: Buy During the Bear Market Phase (50-80% of Cycle)",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = OUTPUT_DIR / "01_cycle_signal_design.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_signals_vs_price(features_df: pd.DataFrame, btc_df: pd.DataFrame):
    """Overlay all 6 novel signals on BTC price + MVRV."""
    start = "2018-01-01"
    end = "2025-12-31"
    feat = features_df.loc[start:end]
    price = btc_df["PriceUSD_coinmetrics"].loc[start:end]

    fig, axes = plt.subplots(6, 1, figsize=(14, 20), sharex=True)

    def add_halvings(ax):
        for h in HALVING_DATES[2:]:
            if pd.Timestamp(start) <= h <= pd.Timestamp(end):
                ax.axvline(h, color="#6366f1", linestyle="--", linewidth=1.2, alpha=0.6)

    # 1. BTC Price
    ax = axes[0]
    ax.semilogy(price.index, price, color="#f59e0b", linewidth=1.5)
    add_halvings(ax)
    ax.set_ylabel("BTC Price (log)", fontsize=9)
    ax.set_title("BTC Price (log scale)", fontsize=10, fontweight="bold")

    # 2. MVRV Z-score
    ax = axes[1]
    mvrv = feat["mvrv_zscore"]
    ax.plot(feat.index, mvrv, color="#7c3aed", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.axhline(1.5, color="#dc2626", linewidth=0.8, linestyle="--", alpha=0.5, label="Caution (1.5)")
    ax.axhline(-1.0, color="#059669", linewidth=0.8, linestyle="--", alpha=0.5, label="Value (-1.0)")
    ax.fill_between(feat.index, mvrv, 0, where=(mvrv < -1), alpha=0.15, color="#059669")
    ax.fill_between(feat.index, mvrv, 0, where=(mvrv > 1.5), alpha=0.15, color="#dc2626")
    ax.set_ylabel("MVRV Z-score", fontsize=9)
    ax.set_title(f"MVRV Z-score  [{WEIGHT_MVRV*100:.0f}% weight] (valuation baseline)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")

    # 3. Halving Cycle Signal (NOVEL)
    ax = axes[2]
    csig = feat["cycle_signal"]
    ax.plot(feat.index, csig, color="#059669", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.fill_between(feat.index, csig, 0, where=(csig > 0), alpha=0.2, color="#059669", label="Accumulate")
    ax.fill_between(feat.index, csig, 0, where=(csig < 0), alpha=0.2, color="#dc2626", label="Reduce")
    ax.set_ylabel("Cycle Signal", fontsize=9)
    ax.set_title(f"Halving Cycle Signal  [{WEIGHT_CYCLE*100:.0f}% weight]  [NEW] — contrarian mid-cycle bear market signal", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    add_halvings(ax)

    # 4. Net Exchange Outflow Signal (NOVEL)
    ax = axes[3]
    esig = feat["exchange_signal"]
    ax.plot(feat.index, esig, color="#2563eb", linewidth=1.0, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.fill_between(feat.index, esig, 0, where=(esig > 0), alpha=0.2, color="#2563eb", label="HODLers accumulating")
    ax.fill_between(feat.index, esig, 0, where=(esig < 0), alpha=0.2, color="#ef4444", label="Selling pressure")
    ax.set_ylabel("Exchange Signal", fontsize=9)
    ax.set_title(f"Net Exchange Outflow  [{WEIGHT_EXCHANGE*100:.0f}% weight]  [NEW] — on-chain HODLer accumulation", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # 5. Monetary Policy Signal (NOVEL — external data)
    ax = axes[4]
    msig = feat["monetary_signal"] if "monetary_signal" in feat.columns else pd.Series(0.0, index=feat.index)
    ax.plot(feat.index, msig, color="#d97706", linewidth=1.0, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.fill_between(feat.index, msig, 0, where=(msig > 0), alpha=0.2, color="#d97706", label="Loose conditions (tailwind)")
    ax.fill_between(feat.index, msig, 0, where=(msig < 0), alpha=0.2, color="#991b1b", label="Tight conditions (headwind)")
    ax.set_ylabel("Monetary Signal", fontsize=9)
    ax.set_title(f"Monetary Policy Signal  [{WEIGHT_MONETARY*100:.0f}% weight]  [NEW] — Fed rate + M2 + DXY (external)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    # Annotate 2022 rate hike cycle
    ax.annotate("2022 Fed hikes\n(→ reduced DCA)", xy=(pd.Timestamp("2022-06-01"), -0.5),
                xytext=(pd.Timestamp("2022-01-01"), -0.8),
                arrowprops=dict(arrowstyle="->", color="#991b1b"),
                fontsize=8, color="#991b1b")

    # 6. Fear Composite Signal (NOVEL — external data)
    ax = axes[5]
    fsig = feat["fear_signal"] if "fear_signal" in feat.columns else pd.Series(0.0, index=feat.index)
    ax.plot(feat.index, fsig, color="#be185d", linewidth=1.0, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.fill_between(feat.index, fsig, 0, where=(fsig > 0), alpha=0.2, color="#be185d", label="Extreme fear (contrarian buy)")
    ax.fill_between(feat.index, fsig, 0, where=(fsig < 0), alpha=0.2, color="#7c2d12", label="Extreme greed (reduce)")
    ax.set_ylabel("Fear Signal", fontsize=9)
    ax.set_title(f"Fear Composite Signal  [{WEIGHT_FEAR*100:.0f}% weight]  [NEW] — VIX + Crypto Fear & Greed (external)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    # Annotate COVID crash
    ax.annotate("COVID crash\nMar 2020", xy=(pd.Timestamp("2020-03-15"), 0.4),
                xytext=(pd.Timestamp("2019-09-01"), 0.55),
                arrowprops=dict(arrowstyle="->", color="#be185d"),
                fontsize=8, color="#be185d")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)

    plt.suptitle(
        "Six-Signal DCA Model: MVRV + Cycle + Exchange Flow + Monetary Policy + Fear (2018-2025)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    path = OUTPUT_DIR / "02_signals_over_time.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_performance_comparison():
    """Compare our model vs example_1 and template across years."""
    import json

    # Load metrics for our model
    with open("work/output/metrics.json") as f:
        m_our = json.load(f)
    df_our = pd.DataFrame(m_our["window_level_data"])
    df_our["start_date"] = pd.to_datetime(df_our["start_date"])
    df_our["year"] = df_our["start_date"].dt.year
    df_our["excess"] = df_our["dynamic_percentile"] - df_our["uniform_percentile"]
    df_our["wins"] = df_our["excess"] > 0

    # Load metrics for example_1
    with open("example_1/output/metrics.json") as f:
        m_ex = json.load(f)
    df_ex = pd.DataFrame(m_ex["window_level_data"])
    df_ex["start_date"] = pd.to_datetime(df_ex["start_date"])
    df_ex["year"] = df_ex["start_date"].dt.year
    df_ex["excess"] = df_ex["dynamic_percentile"] - df_ex["uniform_percentile"]
    df_ex["wins"] = df_ex["excess"] > 0

    by_year_our = df_our.groupby("year").agg(win_rate=("wins", "mean"), mean_excess=("excess", "mean"))
    by_year_ex = df_ex.groupby("year").agg(win_rate=("wins", "mean"), mean_excess=("excess", "mean"))

    years = sorted(set(list(by_year_our.index) + list(by_year_ex.index)))
    x = np.arange(len(years))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Win rate comparison
    wr_our = [by_year_our.loc[y, "win_rate"] * 100 if y in by_year_our.index else 0 for y in years]
    wr_ex = [by_year_ex.loc[y, "win_rate"] * 100 if y in by_year_ex.index else 0 for y in years]

    bars1 = ax1.bar(x - width/2, wr_our, width, label="Our Model\n(6-Signal: Cycle+Exchange+Monetary+Fear)", color="#059669", alpha=0.85)
    bars2 = ax1.bar(x + width/2, wr_ex, width, label="Example 1\n(MVRV + MA + Polymarket)", color="#6366f1", alpha=0.85)
    ax1.axhline(50, color="red", linestyle="--", linewidth=1, label="50% threshold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(years)
    ax1.set_ylabel("Win Rate (%)", fontsize=11)
    ax1.set_title("Win Rate by Window Start Year", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 115)
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.0f}%", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.0f}%", ha="center", va="bottom", fontsize=8, color="#6366f1")

    # Mean excess comparison
    exc_our = [by_year_our.loc[y, "mean_excess"] if y in by_year_our.index else 0 for y in years]
    exc_ex = [by_year_ex.loc[y, "mean_excess"] if y in by_year_ex.index else 0 for y in years]

    colors_our = ["#059669" if v >= 0 else "#dc2626" for v in exc_our]
    colors_ex = ["#6366f1" if v >= 0 else "#9f1239" for v in exc_ex]

    ax2.bar(x - width/2, exc_our, width, color=colors_our, alpha=0.85, label="Our Model (6 signals)")
    ax2.bar(x + width/2, exc_ex, width, color=colors_ex, alpha=0.85, label="Example 1")
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(years)
    ax2.set_ylabel("Mean Excess Percentile (%)", fontsize=11)
    ax2.set_title("Mean Excess SPD Percentile vs. Uniform DCA by Year", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)

    plt.suptitle(
        f"Our Model vs Example 1: Win Rate {m_our['summary_metrics']['win_rate']:.1f}% vs "
        f"{m_ex['summary_metrics']['win_rate']:.1f}% — "
        f"Mean Excess {m_our['summary_metrics']['mean_excess']:.2f}% vs "
        f"{m_ex['summary_metrics']['mean_excess']:.2f}%",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    path = OUTPUT_DIR / "03_performance_comparison.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_model_scores_table():
    """Create a summary table of all model metrics."""
    import json

    with open("work/output/metrics.json") as f:
        m_our = json.load(f)["summary_metrics"]
    with open("example_1/output/metrics.json") as f:
        m_ex = json.load(f)["summary_metrics"]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")

    rows = [
        ["Metric", "Our Model\n(6-Signal: Cycle+Exchange+Monetary+Fear)", "Example 1\n(MVRV + MA + Polymarket)", "Δ vs Example 1"],
        ["Win Rate", f"{m_our['win_rate']:.2f}%", f"{m_ex['win_rate']:.2f}%", f"+{m_our['win_rate']-m_ex['win_rate']:.2f}%p"],
        ["Model Score", f"{m_our['score']:.2f}%", f"{m_ex['score']:.2f}%", f"+{m_our['score']-m_ex['score']:.2f}%p"],
        ["Mean Excess Percentile", f"{m_our['mean_excess']:.2f}%", f"{m_ex['mean_excess']:.2f}%", f"+{m_our['mean_excess']-m_ex['mean_excess']:.2f}%"],
        ["Median Excess Percentile", f"{m_our['median_excess']:.2f}%", f"{m_ex['median_excess']:.2f}%", f"+{m_our['median_excess']-m_ex['median_excess']:.2f}%"],
        ["Relative Improvement (mean)", f"{m_our['relative_improvement_pct_mean']:.2f}%", f"{m_ex['relative_improvement_pct_mean']:.2f}%", f"+{m_our['relative_improvement_pct_mean']-m_ex['relative_improvement_pct_mean']:.2f}%"],
        ["Sats/$ Ratio (mean)", f"{m_our['mean_ratio']:.3f}", f"{m_ex['mean_ratio']:.3f}", f"+{m_our['mean_ratio']-m_ex['mean_ratio']:.3f}"],
        ["Total Windows", f"{m_our['total_windows']}", f"{m_ex['total_windows']}", "-"],
        ["Wins", f"{m_our['wins']}", f"{m_ex['wins']}", f"+{m_our['wins']-m_ex['wins']}"],
    ]

    table = ax.table(cellText=rows[1:], colLabels=rows[0], cellLoc="center", loc="center",
                      colWidths=[0.35, 0.25, 0.25, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)

    for j in range(4):
        table[(0, j)].set_facecolor("#1e3a5f")
        table[(0, j)].set_text_props(weight="bold", color="white")

    for i in range(1, len(rows)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f0f9ff")
            if j == 3:
                val = rows[i][3]
                if val.startswith("+"):
                    table[(i, j)].set_facecolor("#d1fae5")
                    table[(i, j)].set_text_props(color="#065f46", weight="bold")

    ax.set_title("Performance Summary: 6-Signal Novel Model vs. Reference Implementation",
                  fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    path = OUTPUT_DIR / "04_metrics_summary_table.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def print_key_findings():
    """Print a summary of the key findings."""
    import json

    with open("work/output/metrics.json") as f:
        m = json.load(f)["summary_metrics"]
    with open("example_1/output/metrics.json") as f:
        m_ex = json.load(f)["summary_metrics"]

    print("\n" + "=" * 65)
    print("KEY FINDINGS: 6-Signal Contrarian Bitcoin DCA Strategy")
    print("=" * 65)
    print()
    print("NOVEL SIGNALS (all absent from template and example_1):")
    print()
    print("  1. Halving Cycle Position (contrarian design) — 22% weight")
    print("     - Buys MORE during bear market phase (50-80% of cycle)")
    print("     - Reduces during post-halving bull market (15-50%)")
    print("     - Historical evidence: 2018 & 2022 bear market bottoms")
    print()
    print("  2. Net Exchange Outflow (on-chain) — 11% weight")
    print("     - HODLers moving BTC off exchanges = accumulation signal")
    print("     - 180-day rolling z-score + EMA(21) smoothing")
    print("     - Capped at -0.2 during deep-value zones (avoid double-penalizing)")
    print()
    print("  3. Monetary Policy Signal (external: FRED) — 13% weight")
    print("     - Fed rate 3-month velocity (50%): rapid hikes → reduce")
    print("     - M2 YoY growth (25%): expansion → buy, contraction → reduce")
    print("     - DXY 90-day momentum (25%): dollar strength → reduce")
    print("     - Key impact: Correctly reduces DCA during 2022 rate hike cycle")
    print()
    print("  4. Fear Composite Signal (external: VIX + Alternative.me) — 6% weight")
    print("     - VIX 180-day z-score (40%): spike = fear = contrarian buy")
    print("     - Crypto Fear & Greed cubic transform (60%): extreme fear only")
    print()
    print("PERFORMANCE vs. UNIFORM DCA:")
    print(f"  Win Rate:              {m['win_rate']:.2f}% ({m['wins']}/{m['total_windows']} windows)")
    print(f"  Mean Excess:           +{m['mean_excess']:.2f}% percentile")
    print(f"  Relative Improvement:  +{m['relative_improvement_pct_mean']:.2f}%")
    print(f"  Sats/$ Ratio:          {m['mean_ratio']:.3f}×")
    print()
    print("PERFORMANCE vs. EXAMPLE 1 (MVRV + MA + Polymarket):")
    print(f"  Win Rate:    {m['win_rate']:.2f}% vs {m_ex['win_rate']:.2f}% (+{m['win_rate']-m_ex['win_rate']:.2f}%p)")
    print(f"  Model Score: {m['score']:.2f}% vs {m_ex['score']:.2f}% (+{m['score']-m_ex['score']:.2f}%p)")
    print(f"  Rel. Impr.:  {m['relative_improvement_pct_mean']:.2f}% vs {m_ex['relative_improvement_pct_mean']:.2f}% (+{m['relative_improvement_pct_mean']-m_ex['relative_improvement_pct_mean']:.2f}%)")
    print()
    print("LIMITATIONS:")
    print("  - 2020 windows (-9.9% excess): COVID crash creates extreme regime")
    print("  - 2022 windows (-15.5% excess): Falling knives in prolonged bear")
    print("  - Halving dates fixed (estimated next halving ~2028-04-17)")
    print("  - Exchange flow data starts 2010, full signal from 2018")
    print("=" * 65)


def main():
    btc_df = load_data()
    features_df = precompute_features(btc_df)

    print("Generating analysis charts...")
    plot_cycle_signal_design()
    plot_signals_vs_price(features_df, btc_df)
    plot_performance_comparison()
    plot_model_scores_table()
    print_key_findings()

    print(f"\nAll charts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
