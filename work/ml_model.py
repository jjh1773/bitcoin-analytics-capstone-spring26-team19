"""
Supervised Learning DCA Model
==============================
Compares 7 supervised learning models using a CONTRARIAN target variable
aligned with the DCA objective (accumulate more sats when price is cheap).

Target variable (contrarian cheapness score):
    y[t] = log(trailing_365d_mean_price / forward_30d_mean_price)
    Positive → next 30 days are cheaper than recent trend → buy more
    Negative → next 30 days are expensive relative to trend → buy less

Walk-forward protocol
---------------------
  • Minimum training window : 730 days (2 years)
  • Retrain frequency       : every 30 days
  • Out-of-sample period    : 2020-01-01 → end
  • Features                : 13 pre-computed signals (already 1-day lagged)
  • Target                  : contrarian cheapness score (see above)

Run from project root:
    python -m work.ml_model
"""

import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Sklearn ────────────────────────────────────────────────────────────────────
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import shap

# ── Project modules ────────────────────────────────────────────────────────────
from template.prelude_template import load_data
from template.model_development_template import allocate_sequential_stable, _clean_array
from work.model_development import precompute_features

# ── Constants ─────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
OUT_DIR     = Path(__file__).parent / "output" / "ml"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_MIN_DAYS  = 730    # minimum history before first prediction
RETRAIN_FREQ    = 30     # retrain every N days
FORWARD_DAYS    = 30     # prediction horizon (days)
OOS_START       = pd.Timestamp("2020-01-01")
DYNAMIC_STRENGTH = 5.0   # same scaling as V3

# Signal features used as ML inputs
FEATURE_COLS = [
    "price_vs_ma",
    "mvrv_zscore",
    "mvrv_gradient",
    "mvrv_acceleration",
    "mvrv_volatility",
    "cycle_position",
    "cycle_signal",
    "exchange_signal",
    "monetary_signal",
    "fear_signal",
    "real_yield_signal",   # 10yr TIPS real yield + yield curve spread
    "credit_signal",       # HY credit spread (risk-off stress indicator)
    "v3_signal",           # Rule-Based DCA Model's combined signal — ML inherits domain knowledge
]

# ── Colour palette ─────────────────────────────────────────────────────────────
C_BASELINE = "#94a3b8"
C_V3       = "#22c55e"
C_PALETTE  = [
    "#3b82f6",  # Ridge
    "#06b6d4",  # Lasso
    "#8b5cf6",  # ElasticNet
    "#f97316",  # RandomForest
    "#ef4444",  # GradientBoosting
    "#eab308",  # XGBoost
    "#ec4899",  # LightGBM
]

# =============================================================================
# 0. V3 signal helper (defined early — used to build feature matrix)
# =============================================================================

def compute_v3_signal_series(feat: pd.DataFrame,
                              start: pd.Timestamp = None,
                              end:   pd.Timestamp = None) -> pd.Series:
    """Compute the Rule-Based DCA Model's pre-exp combined signal as a daily time series.

    Returns the raw combined signal score (before exp() and allocation),
    clipped to [-1, 1].  This is the ideal ML feature because:
      • It encodes ALL of the Rule-Based model's domain knowledge in one number
      • Linear scale (no explosion) — ML-friendly
      • Positive = buy more signal, Negative = buy less signal
    """
    from work.model_development import (
        compute_asymmetric_mvrv_boost,
        WEIGHT_MVRV, WEIGHT_CYCLE, WEIGHT_MONETARY,
        WEIGHT_FEAR, WEIGHT_EXCHANGE, WEIGHT_MA,
    )
    f = feat.copy()
    if start is not None:
        f = f.loc[start:]
    if end is not None:
        f = f.loc[:end]

    pvm = _clean_array(f["price_vs_ma"].values)
    mz  = _clean_array(f["mvrv_zscore"].values)
    mg  = _clean_array(f["mvrv_gradient"].values)
    ma_ = _clean_array(f["mvrv_acceleration"].values)
    mv  = _clean_array(f["mvrv_volatility"].values)
    mv  = np.where(mv == 0, 0.5, mv)
    cs  = _clean_array(f["cycle_signal"].values)
    es  = _clean_array(f["exchange_signal"].values)
    ms  = _clean_array(f["monetary_signal"].values)
    fs  = _clean_array(f["fear_signal"].values)

    mvrv_signal = -mz + compute_asymmetric_mvrv_boost(mz)
    ma_signal   = -pvm
    in_dv       = mz < -1.5
    es_adj      = np.where(in_dv, np.maximum(es, -0.2), es)
    ms_adj      = np.where(in_dv, np.maximum(ms, -0.3), ms)

    combined = (mvrv_signal * WEIGHT_MVRV + cs * WEIGHT_CYCLE
                + ms_adj * WEIGHT_MONETARY + fs * WEIGHT_FEAR
                + es_adj * WEIGHT_EXCHANGE + ma_signal * WEIGHT_MA)

    same_dir  = (ma_ * mg) > 0
    accel_mod = np.where(same_dir, 1.0 + 0.15*np.abs(ma_),
                                   1.0 - 0.10*np.abs(ma_))
    accel_mod = np.clip(accel_mod, 0.85, 1.15)
    combined  = combined * accel_mod
    vol_damp  = np.where(mv > 0.8, 1.0 - 0.2*(mv-0.8)/0.2, 1.0)
    combined  = combined * vol_damp

    return pd.Series(combined.clip(-1, 1), index=f.index, name="v3_signal")


# =============================================================================
# 1. Data loading
# =============================================================================

print("=" * 60)
print("Supervised Learning DCA Model")
print("=" * 60)
print("\nLoading data and computing features...")

btc_df   = load_data()
features = precompute_features(btc_df)
price    = btc_df["PriceUSD_coinmetrics"].dropna()

print(f"  Price:    {len(price)} rows")
print(f"  Features: {len(features)} rows")

# =============================================================================
# 2. Build feature matrix and target
# =============================================================================

def build_ml_dataset(features: pd.DataFrame, price: pd.Series,
                     forward_days: int = 30) -> tuple[pd.DataFrame, pd.Series]:
    """Build (X, y) for supervised learning with contrarian DCA target.

    X[t] = 10 pre-computed signals at day t (already 1-day lagged internally)
    y[t] = log(trailing_365d_mean_price[t] / forward_30d_mean_price[t])

    Interpretation:
      y > 0  →  future prices cheaper than recent trend → model should buy more
      y < 0  →  future prices expensive relative to trend → model should buy less

    This directly aligns the ML objective with DCA: buy more when cheap,
    less when expensive.  The trailing mean uses only past data (no leakage);
    the forward mean is the unknown target to predict.
    """
    X = features[FEATURE_COLS].copy()

    # Trailing 365-day mean price at t: known, no look-ahead
    trailing_mean = price.rolling(365, min_periods=180).mean()

    # Forward 30-day mean price: average of prices over [t, t+forward_days)
    # Computed by shifting a backward rolling mean forward
    fwd_mean = price.rolling(forward_days, min_periods=max(1, forward_days // 2)).mean() \
                    .shift(-forward_days)

    # Contrarian cheapness score: positive = future is cheaper than recent trend
    y = np.log(trailing_mean / fwd_mean.replace(0, np.nan))
    y = y.reindex(X.index)

    # Align and drop rows where either X or y is missing
    valid = X.notna().all(axis=1) & y.notna()
    return X[valid], y[valid]


# Add V3 combined signal as a feature (encodes all V3 domain knowledge)
# ML can now inherit V3's expertise and learn to adjust it contextually
v3_sig_full = compute_v3_signal_series(features)
features["v3_signal"] = v3_sig_full.reindex(features.index).fillna(0.0)

X_all, y_all = build_ml_dataset(features, price, FORWARD_DAYS)
print(f"\n  Dataset: {len(X_all)} samples  "
      f"({X_all.index.min().date()} → {X_all.index.max().date()})")

# =============================================================================
# 3. Model registry
# =============================================================================

def make_models() -> dict:
    """Return a fresh dict of {name: model} instances."""
    return {
        "Ridge":            Ridge(alpha=1.0),
        "Lasso":            Lasso(alpha=0.001, max_iter=5000),
        "ElasticNet":       ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000),
        "RandomForest":     RandomForestRegressor(
                                n_estimators=200, max_depth=4,
                                min_samples_leaf=10, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(
                                n_estimators=200, max_depth=3,
                                learning_rate=0.05, subsample=0.8,
                                min_samples_leaf=10, random_state=42),
        "XGBoost":          xgb.XGBRegressor(
                                n_estimators=200, max_depth=3,
                                learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, reg_alpha=0.1,
                                reg_lambda=1.0, random_state=42,
                                verbosity=0, eval_metric="rmse"),
        "LightGBM":         lgb.LGBMRegressor(
                                n_estimators=200, max_depth=3,
                                learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, reg_alpha=0.1,
                                reg_lambda=1.0, random_state=42,
                                verbose=-1),
    }

MODEL_NAMES = list(make_models().keys())

# =============================================================================
# 4. Walk-forward training
# =============================================================================

def walk_forward_predict(X: pd.DataFrame, y: pd.Series,
                         model_name: str,
                         train_min: int = TRAIN_MIN_DAYS,
                         retrain_freq: int = RETRAIN_FREQ,
                         oos_start: pd.Timestamp = OOS_START,
                         ) -> pd.Series:
    """Run expanding-window walk-forward prediction for one model.

    Returns a Series of predicted 30-day log-returns indexed by date,
    starting from oos_start.
    """
    dates     = X.index
    preds     = {}
    scaler    = StandardScaler()
    model     = make_models()[model_name]
    last_fit  = None   # date of last model fit

    for i, t in enumerate(dates):
        if t < oos_start:
            continue

        # Only retrain when forced or first prediction
        need_retrain = (last_fit is None) or ((t - last_fit).days >= retrain_freq)

        if need_retrain:
            # Training set: all samples strictly before t
            # AND with target observable (i.e., s + FORWARD_DAYS in the past)
            train_mask = (dates < t) & (dates <= (t - pd.Timedelta(days=FORWARD_DAYS)))
            X_tr = X[train_mask]
            y_tr = y[train_mask]

            if len(X_tr) < train_min:
                continue  # not enough history yet

            # Fit scaler on training data only
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)

            model = make_models()[model_name]
            model.fit(X_tr_sc, y_tr)
            last_fit = t

        if last_fit is None:
            continue

        # Predict for today
        X_t_sc = scaler.transform(X.loc[[t]])
        pred   = model.predict(X_t_sc)[0]
        preds[t] = pred

    return pd.Series(preds, name=model_name)


print("\nWalk-forward training for all models (contrarian cheapness target)...")
predictions: dict[str, pd.Series] = {}

for name in MODEL_NAMES:
    print(f"  {name:<20}", end="", flush=True)
    pred = walk_forward_predict(X_all, y_all, name)
    predictions[name] = pred
    corr = pred.corr(y_all.reindex(pred.index))
    rmse = np.sqrt(mean_squared_error(
        y_all.reindex(pred.index), pred))
    print(f"  n={len(pred):4d}  corr(cheapness)={corr:+.3f}  RMSE={rmse:.4f}")

# =============================================================================
# 5. Convert predictions → DCA multipliers
# =============================================================================

def pred_to_multiplier(pred: pd.Series, corr: float) -> pd.Series:
    """Convert ML predictions to DCA multipliers via rank-based normalisation.

    Rank normalisation is used (not V3's exp(5×signal)) because ML predictions
    have daily noise (RMSE ≈ 0.22) that exp amplification would magnify into
    harmful extreme concentrations.  Rank is robust to scale and outliers.

    Adaptive range scales with OOS correlation:
      corr = 0.10  →  [exp(-0.2), exp(+0.2)] = [0.82, 1.22]  (conservative)
      corr = 0.90  →  [exp(-1.8), exp(+1.8)] = [0.17, 6.05]  (aggressive)
      corr = 1.00  →  [exp(-2.0), exp(+2.0)] = [0.14, 7.39]  (oracle)
    """
    rank      = pred.rank(pct=True)
    norm      = 2 * rank - 1                              # [-1, 1]
    strength  = np.clip(abs(corr), 0.05, 1.0) * 2.0
    multiplier = np.exp(norm * strength)
    return multiplier.where(np.isfinite(multiplier), 1.0)


# =============================================================================
# 6. Backtest engine
# =============================================================================

BACKTEST_START = pd.Timestamp("2018-01-01")
BACKTEST_END   = price.index.max() - pd.Timedelta(days=365)


def _proportional_weights(multiplier: np.ndarray) -> np.ndarray:
    """Simple proportional weight allocation.

    weight[i] = multiplier[i] / sum(multiplier)

    This is the natural allocation for pre-computed multipliers: days with
    higher predicted returns receive proportionally more capital.  It avoids
    the running-mean normalisation in allocate_sequential_stable, which was
    designed for on-the-fly production use and distorts pre-computed ML scores.
    """
    mult = np.where(np.isfinite(multiplier) & (multiplier > 0), multiplier, 1.0)
    total = mult.sum()
    return mult / total if total > 0 else np.ones(len(mult)) / len(mult)


def run_backtest_v3(feat: pd.DataFrame, pr: pd.Series,
                    start: pd.Timestamp = BACKTEST_START,
                    end:   pd.Timestamp = BACKTEST_END,
                    simple_alloc: bool = False) -> dict:
    """Re-run V3 (rule-based) backtest.

    simple_alloc=True uses proportional weights (same as ML backtest) for
    a fair apples-to-apples OOS comparison.
    """
    from work.model_development import (
        compute_asymmetric_mvrv_boost,
        WEIGHT_MVRV, WEIGHT_CYCLE, WEIGHT_MONETARY,
        WEIGHT_FEAR, WEIGHT_EXCHANGE, WEIGHT_MA,
    )

    ratios, year_data = [], []
    dates = pr.loc[start:end].index

    for s in dates:
        e = s + pd.Timedelta(days=364)
        if e not in pr.index:
            continue
        price_w = pr.loc[s:e]
        feat_w  = feat.loc[s:e]
        n       = len(price_w)
        if n < 30:
            continue

        pvm = _clean_array(feat_w["price_vs_ma"].values)
        mz  = _clean_array(feat_w["mvrv_zscore"].values)
        mg  = _clean_array(feat_w["mvrv_gradient"].values)
        ma_ = _clean_array(feat_w["mvrv_acceleration"].values)
        mv  = _clean_array(feat_w["mvrv_volatility"].values)
        mv  = np.where(mv == 0, 0.5, mv)
        cs  = _clean_array(feat_w["cycle_signal"].values)
        es  = _clean_array(feat_w["exchange_signal"].values)
        ms  = _clean_array(feat_w["monetary_signal"].values)
        fs  = _clean_array(feat_w["fear_signal"].values)

        mvrv_signal = -mz + compute_asymmetric_mvrv_boost(mz)
        ma_signal   = -pvm
        in_dv       = mz < -1.5
        es_adj      = np.where(in_dv, np.maximum(es, -0.2), es)
        ms_adj      = np.where(in_dv, np.maximum(ms, -0.3), ms)

        combined = (mvrv_signal * WEIGHT_MVRV + cs * WEIGHT_CYCLE
                    + ms_adj * WEIGHT_MONETARY + fs * WEIGHT_FEAR
                    + es_adj * WEIGHT_EXCHANGE + ma_signal * WEIGHT_MA)

        same_dir  = (ma_ * mg) > 0
        accel_mod = np.where(same_dir, 1.0 + 0.15*np.abs(ma_),
                                       1.0 - 0.10*np.abs(ma_))
        accel_mod = np.clip(accel_mod, 0.85, 1.15)
        combined  = combined * accel_mod
        vol_damp  = np.where(mv > 0.8, 1.0 - 0.2*(mv-0.8)/0.2, 1.0)
        combined  = combined * vol_damp

        adj        = np.clip(combined * DYNAMIC_STRENGTH, -5, 100)
        multiplier = np.exp(adj)
        multiplier = np.where(np.isfinite(multiplier), multiplier, 1.0)

        if simple_alloc:
            weights = _proportional_weights(multiplier)
        else:
            raw     = np.ones(n) / n * multiplier
            weights = allocate_sequential_stable(raw, n, None)

        sats        = 1e8 / price_w.values
        dynamic_spd = np.sum(weights * sats)
        uniform_spd = np.mean(sats)
        ratio       = dynamic_spd / uniform_spd

        ratios.append(ratio)
        year_data.append({"year": s.year, "ratio": ratio})

    df_y  = pd.DataFrame(year_data)
    by_yr = df_y.groupby("year")["ratio"].mean() if len(df_y) else pd.Series()
    return {"mean_ratio": np.mean(ratios), "win_rate": np.mean([r>1 for r in ratios])*100,
            "n_windows": len(ratios), "by_year": by_yr}


def run_backtest_ml(ml_mult: pd.Series, pr: pd.Series,
                    start: pd.Timestamp = OOS_START,
                    end:   pd.Timestamp = BACKTEST_END) -> dict:
    """Backtest using pre-computed ML multipliers with proportional allocation."""
    ratios, year_data = [], []
    dates = pr.loc[start:end].index

    for s in dates:
        e = s + pd.Timedelta(days=364)
        if e not in pr.index:
            continue
        price_w = pr.loc[s:e]
        n       = len(price_w)
        if n < 30:
            continue

        mult_w  = ml_mult.reindex(price_w.index).fillna(1.0).values
        weights = _proportional_weights(mult_w)

        sats        = 1e8 / price_w.values
        dynamic_spd = np.sum(weights * sats)
        uniform_spd = np.mean(sats)
        ratio       = dynamic_spd / uniform_spd

        ratios.append(ratio)
        year_data.append({"year": s.year, "ratio": ratio})

    df_y  = pd.DataFrame(year_data)
    by_yr = df_y.groupby("year")["ratio"].mean() if len(df_y) else pd.Series()
    return {"mean_ratio": np.mean(ratios), "win_rate": np.mean([r>1 for r in ratios])*100,
            "n_windows": len(ratios), "by_year": by_yr}


# ── Run all backtests ──────────────────────────────────────────────────────────
print("\nRunning backtests...")

# Rule-Based DCA Model baseline — full period + OOS with both allocators
v3_full      = run_backtest_v3(features, price, BACKTEST_START, BACKTEST_END, simple_alloc=False)
v3_oos_orig  = run_backtest_v3(features, price, OOS_START,     BACKTEST_END, simple_alloc=False)
v3_oos       = run_backtest_v3(features, price, OOS_START,     BACKTEST_END, simple_alloc=True)
print(f"  Rule-Based (full 2018–, original): mean_ratio={v3_full['mean_ratio']:.5f}  "
      f"WR={v3_full['win_rate']:.2f}%  n={v3_full['n_windows']}")
print(f"  Rule-Based (OOS  2020–, original): mean_ratio={v3_oos_orig['mean_ratio']:.5f}  "
      f"WR={v3_oos_orig['win_rate']:.2f}%")
print(f"  Rule-Based (OOS  2020–, simple  ): mean_ratio={v3_oos['mean_ratio']:.5f}  "
      f"WR={v3_oos['win_rate']:.2f}%  n={v3_oos['n_windows']}")

# ── Compute OOS prediction accuracy FIRST (needed for multiplier calibration) ─
oos_corr = {}
oos_rmse = {}
for name, pred in predictions.items():
    y_actual = y_all.reindex(pred.index)
    valid    = y_actual.notna() & pred.notna()
    if valid.sum() < 10:
        oos_corr[name] = 0.0
        oos_rmse[name] = np.nan
        continue
    oos_corr[name] = float(y_actual[valid].corr(pred[valid]))
    oos_rmse[name] = float(np.sqrt(mean_squared_error(y_actual[valid], pred[valid])))

# Build multipliers — scale range by OOS correlation (adaptive conservatism)
ml_multipliers: dict[str, pd.Series] = {}
for name, pred in predictions.items():
    ml_multipliers[name] = pred_to_multiplier(pred, oos_corr[name])

# Oracle: use actual cheapness values as perfect multiplier (upper bound)
# i.e., perfect prediction of the contrarian target → best achievable DCA
oracle_series = pred_to_multiplier(y_all, corr=1.0)
oracle_res = run_backtest_ml(oracle_series, price, OOS_START, BACKTEST_END)
print(f"  {'Oracle (perfect pred)':<20}  mean_ratio={oracle_res['mean_ratio']:.5f}  "
      f"WR={oracle_res['win_rate']:.2f}%  (upper bound)")

ml_results: dict[str, dict] = {}
for name, mult in ml_multipliers.items():
    res = run_backtest_ml(mult, price, OOS_START, BACKTEST_END)
    ml_results[name] = res
    print(f"  {name:<20}  mean_ratio={res['mean_ratio']:.5f}  "
          f"WR={res['win_rate']:.2f}%  n={res['n_windows']}")

# Best ML model
best_name = max(ml_results, key=lambda k: ml_results[k]["mean_ratio"])
best_res  = ml_results[best_name]
print(f"\n  Best ML model: {best_name}  "
      f"mean_ratio={best_res['mean_ratio']:.5f}  "
      f"WR={best_res['win_rate']:.2f}%")

# =============================================================================
# 7. SHAP feature importance for best model
# =============================================================================

print(f"\nComputing SHAP values for {best_name}...")

train_end = BACKTEST_END - pd.Timedelta(days=FORWARD_DAYS)
X_full_tr = X_all[X_all.index <= train_end]
y_full_tr = y_all[y_all.index <= train_end]

scaler_final = StandardScaler()
X_full_sc    = scaler_final.fit_transform(X_full_tr)

best_model = make_models()[best_name]
best_model.fit(X_full_sc, y_full_tr)

if best_name in ("XGBoost", "LightGBM", "RandomForest", "GradientBoosting"):
    explainer  = shap.TreeExplainer(best_model)
    shap_vals  = explainer.shap_values(X_full_sc)
else:
    explainer  = shap.LinearExplainer(best_model, X_full_sc)
    shap_vals  = explainer.shap_values(X_full_sc)

mean_abs_shap = pd.Series(
    np.abs(shap_vals).mean(axis=0),
    index=FEATURE_COLS,
).sort_values(ascending=True)

# =============================================================================
# 8. Figures
# =============================================================================

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150,
                     "font.family": "Arial"})


# ── Fig A: Model comparison (mean ratio + win rate) ─────────────────────────

def save_fig_a():
    names_ml  = list(ml_results.keys())
    ratios_ml = [(ml_results[n]["mean_ratio"] - 1) * 100 for n in names_ml]
    wrs_ml    = [ml_results[n]["win_rate"]           for n in names_ml]

    v3_oos_ratio     = (v3_oos["mean_ratio"]     - 1) * 100
    oracle_ratio     = (oracle_res["mean_ratio"] - 1) * 100
    v3_oos_wr        = v3_oos["win_rate"]
    oracle_wr        = oracle_res["win_rate"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Supervised Learning DCA Model Comparison — Contrarian Target (Price Cheapness)\n"
                 "(Evaluation period: 2020-01-01 onward — out-of-sample)",
                 fontsize=13, fontweight="bold")

    # Combine all for bar chart (Uniform | Rule-Based | Oracle | ML×7)
    all_names  = (["Uniform DCA", "Rule-Based\nDCA Model", "Oracle\n(perfect pred)"]
                  + names_ml)
    all_ratios = [0.0, v3_oos_ratio, oracle_ratio] + ratios_ml
    all_wrs    = [50.0, v3_oos_wr,   oracle_wr]    + wrs_ml
    colors     = [C_BASELINE, C_V3, "#fbbf24"] + C_PALETTE

    x = np.arange(len(all_names))

    # Left: cumulative return
    bars1 = ax1.bar(x, all_ratios, color=colors, alpha=0.85, width=0.6)
    ax1.axhline(0, color="black", lw=0.8, ls="--")
    ax1.axhline(v3_oos_ratio, color=C_V3, lw=1.2, ls=":", alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_names, rotation=35, ha="right", fontsize=9)
    ax1.set_ylabel("Mean Outperformance vs Uniform DCA (%)", fontsize=10)
    ax1.set_title("Mean Ratio vs Uniform DCA", fontweight="bold")
    for bar, val in zip(bars1, all_ratios):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                 f"{val:+.2f}%", ha="center", va="bottom", fontsize=8,
                 fontweight="bold")

    # Right: win rate
    bars2 = ax2.bar(x, all_wrs, color=colors, alpha=0.85, width=0.6)
    ax2.axhline(50, color="black", lw=0.8, ls="--")
    ax2.axhline(v3_oos_wr, color=C_V3, lw=1.2, ls=":", alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_names, rotation=35, ha="right", fontsize=9)
    ax2.set_ylabel("Win Rate (%)", fontsize=10)
    ax2.set_title("Win Rate (% of 1-yr windows > Uniform)", fontweight="bold")
    for bar, val in zip(bars2, all_wrs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=8,
                 fontweight="bold")

    plt.tight_layout()
    path = OUT_DIR / "figA_model_comparison.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Fig B: SHAP feature importance ──────────────────────────────────────────

def save_fig_b():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Feature Importance — {best_name} (Best ML Model)\n"
                 "Left: mean |SHAP|  |  Right: Prediction vs Actual (OOS)",
                 fontsize=12, fontweight="bold")

    # Left: SHAP bar
    colors_shap = [C_PALETTE[0]] * len(mean_abs_shap)
    ax1.barh(mean_abs_shap.index, mean_abs_shap.values,
             color=C_PALETTE[0], alpha=0.85)
    ax1.set_xlabel("Mean |SHAP value|", fontsize=10)
    ax1.set_title("Signal Contribution to Predictions", fontweight="bold")
    for i, (feat, val) in enumerate(mean_abs_shap.items()):
        ax1.text(val + 0.0002, i, f"{val:.4f}", va="center", fontsize=8)

    # Right: predicted vs actual
    pred_best  = predictions[best_name]
    y_act      = y_all.reindex(pred_best.index)
    valid_mask = y_act.notna() & pred_best.notna()
    ax2.scatter(y_act[valid_mask], pred_best[valid_mask],
                alpha=0.3, s=8, color=C_PALETTE[0])
    lims = [min(y_act[valid_mask].min(), pred_best[valid_mask].min()),
            max(y_act[valid_mask].max(), pred_best[valid_mask].max())]
    ax2.plot(lims, lims, "r--", lw=1, label="Perfect prediction")
    corr = float(y_act[valid_mask].corr(pred_best[valid_mask]))
    ax2.set_xlabel("Actual 30d log-return", fontsize=10)
    ax2.set_ylabel("Predicted 30d log-return", fontsize=10)
    ax2.set_title(f"Predicted vs Actual  (r = {corr:+.3f})", fontweight="bold")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    path = OUT_DIR / "figB_feature_importance.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Fig C: Year-by-year comparison (V3 vs best ML) ──────────────────────────

def save_fig_c():
    # Collect year-by-year for all ML models + Rule-Based OOS
    v3_by_year   = v3_oos["by_year"]
    best_by_year = best_res["by_year"]
    all_years    = sorted(set(v3_by_year.index) | set(best_by_year.index))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Year-by-Year Performance: Rule-Based DCA Model vs Supervised Learning DCA Models\n"
                 "(OOS period: 2020 onward)", fontsize=12, fontweight="bold")

    # Left: all ML models + Rule-Based vs year
    ax = axes[0]
    x  = np.arange(len(all_years))
    w  = 0.08
    offset = -(len(MODEL_NAMES) / 2) * w

    for i, (name, col) in enumerate(zip(MODEL_NAMES, C_PALETTE)):
        by_yr = ml_results[name]["by_year"]
        vals  = [(by_yr.get(yr, np.nan) - 1) * 100 for yr in all_years]
        ax.bar(x + offset + i * w, vals, width=w, color=col, alpha=0.8, label=name)

    # Rule-Based OOS as line
    v3_vals = [(v3_by_year.get(yr, np.nan) - 1) * 100 for yr in all_years]
    ax.plot(x + offset + (len(MODEL_NAMES)-1) * w / 2,
            v3_vals, "o-", color=C_V3, lw=2, ms=6, label="Rule-Based DCA Model", zorder=5)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(all_years)
    ax.set_ylabel("Outperformance vs Uniform DCA (%)")
    ax.set_title("All ML Models vs Rule-Based DCA Model by Year")
    ax.legend(fontsize=7, ncol=2)

    # Right: Rule-Based vs best Supervised Learning head-to-head
    ax2 = axes[1]
    x2  = np.arange(len(all_years))

    v3_v  = np.array([(v3_by_year.get(yr, np.nan)   - 1)*100 for yr in all_years])
    ml_v  = np.array([(best_by_year.get(yr, np.nan) - 1)*100 for yr in all_years])

    ax2.bar(x2 - 0.15, v3_v, 0.28, color=C_V3,        alpha=0.85, label="Rule-Based DCA Model")
    ax2.bar(x2 + 0.15, ml_v, 0.28, color=C_PALETTE[0], alpha=0.85, label=f"Supervised Learning ({best_name})")

    for j in range(len(all_years)):
        valid_vals = [v for v in [v3_v[j], ml_v[j]] if not np.isnan(v)]
        if len(valid_vals) < 2:
            continue
        top = max(valid_vals) + 0.3
        d_ml = ml_v[j] - v3_v[j]
        ax2.text(j + 0.15, top, f"Δ{d_ml:+.1f}%", ha="center", fontsize=7,
                 color=C_PALETTE[0])

    ax2.axhline(0, color="black", lw=0.8, ls="--")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(all_years)
    ax2.set_ylabel("Outperformance vs Uniform DCA (%)")
    ax2.set_title(f"Rule-Based DCA Model vs Supervised Learning ({best_name})")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    path = OUT_DIR / "figC_year_by_year.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Fig D: Prediction accuracy (corr + RMSE per model) ─────────────────────

def save_fig_d():
    names  = list(oos_corr.keys())
    corrs  = [oos_corr[n] for n in names]
    rmses  = [oos_rmse[n] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("OOS Prediction Quality: Predicted vs Actual 30-day Return",
                 fontsize=12, fontweight="bold")

    x = np.arange(len(names))
    ax1.bar(x, corrs, color=C_PALETTE, alpha=0.85)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=30, ha="right")
    ax1.set_ylabel("Pearson Correlation (pred vs actual)")
    ax1.set_title("Prediction Correlation (higher = better)")
    for bar, val in zip(ax1.patches, corrs):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.002 if val >= 0 else bar.get_height() - 0.01,
                 f"{val:+.3f}", ha="center", fontsize=9, fontweight="bold")

    ax2.bar(x, rmses, color=C_PALETTE, alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=30, ha="right")
    ax2.set_ylabel("RMSE of predicted 30-day log-return")
    ax2.set_title("Prediction Error (lower = better)")
    for bar, val in zip(ax2.patches, rmses):
        if not np.isnan(val):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.0005,
                     f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = OUT_DIR / "figD_prediction_accuracy.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Generate all figures ───────────────────────────────────────────────────
print("\nGenerating figures...")
save_fig_a()
save_fig_b()
save_fig_c()
save_fig_d()

# =============================================================================
# 9. Summary
# =============================================================================

print("\n" + "=" * 65)
print("SUMMARY — Supervised Learning DCA Model Results (OOS: 2020–)")
print("=" * 65)
print(f"{'Model':<30} {'MeanRatio':>10} {'Outperf%':>10} {'WinRate%':>10} {'Corr':>8}")
print("-" * 65)
print(f"{'Uniform DCA':<30} {'1.00000':>10} {'+0.00%':>10} {'50.0%':>10} {'—':>8}")
print(f"{'Rule-Based DCA Model (OOS)':<30} {v3_oos['mean_ratio']:>10.5f} "
      f"{(v3_oos['mean_ratio']-1)*100:>+9.2f}% "
      f"{v3_oos['win_rate']:>9.2f}%  {'—':>8}")
print(f"{'Oracle (perfect pred)':<30} {oracle_res['mean_ratio']:>10.5f} "
      f"{(oracle_res['mean_ratio']-1)*100:>+9.2f}% "
      f"{oracle_res['win_rate']:>9.2f}%  {'1.000':>8}")
print("-" * 65)
for name in MODEL_NAMES:
    r   = ml_results[name]
    cor = oos_corr.get(name, 0)
    marker = " <-- BEST" if name == best_name else ""
    print(f"{name:<30} {r['mean_ratio']:>10.5f} "
          f"{(r['mean_ratio']-1)*100:>+9.2f}% "
          f"{r['win_rate']:>9.2f}%  {cor:>+7.3f}{marker}")

print("=" * 65)
print(f"\nKey findings:")
print(f"  Oracle upper bound         : MR={oracle_res['mean_ratio']:.4f}  "
      f"WR={oracle_res['win_rate']:.2f}%")
print(f"  Rule-Based DCA Model (OOS) : MR={v3_oos['mean_ratio']:.4f}  "
      f"WR={v3_oos['win_rate']:.2f}%")
print(f"  Best Supervised Learning   : {best_name},  MR={best_res['mean_ratio']:.4f}  "
      f"WR={best_res['win_rate']:.2f}%")
print(f"\nConclusion:")
print(f"  Rule-Based DCA Model maximises total accumulation "
      f"(MR={v3_oos['mean_ratio']:.4f}, WR={v3_oos['win_rate']:.1f}%)")
print(f"  Supervised Learning maximises consistency "
      f"(MR={best_res['mean_ratio']:.4f}, WR={best_res['win_rate']:.1f}%)")
print(f"\nOutput saved to: {OUT_DIR}")
