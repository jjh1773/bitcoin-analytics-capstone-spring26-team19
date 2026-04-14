"""Dynamic DCA weight computation: Halving Cycle + Exchange Flow + Macro Signals.

Novel contributions (all absent from template and example_1):
1. Contrarian Halving Cycle: Bear market phase (50-80% of cycle) = best accumulation window
2. Net Exchange Outflow: HODLers removing BTC from exchanges = accumulation signal
3. Monetary Policy Signal: Fed rate hike velocity + M2 growth + DXY momentum
   → Tight financial conditions reduce DCA weight (fixes 2022 rate-hike problem)
4. Fear Composite Signal: VIX spike + Crypto Fear & Greed extremes
   → Extreme market fear = contrarian buy signal (fixes 2020 COVID problem)

Signal weights:
   - MVRV Z-score:        40%  (core on-chain valuation)
   - Halving Cycle:       20%  (contrarian macro timing)
   - Monetary Policy:     15%  (Fed + M2 + DXY — NEW external data)
   - Fear Composite:      10%  (VIX + Fear&Greed — NEW external data)
   - Net Exchange Flow:   10%  (on-chain accumulation)
   - 200-day MA:           5%  (trend confirmation)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Import base functionality from template
from template.model_development_template import (
    _compute_stable_signal,
    allocate_sequential_stable,
    _clean_array,
)

# =============================================================================
# Constants
# =============================================================================

PRICE_COL = "PriceUSD_coinmetrics"
MVRV_COL = "CapMVRVCur"
FLOW_IN_COL = "FlowInExNtv"
FLOW_OUT_COL = "FlowOutExNtv"

# Strategy parameters
MIN_W = 1e-6
MA_WINDOW = 200
MVRV_ROLLING_WINDOW = 365
MVRV_GRADIENT_WINDOW = 30
MVRV_ACCEL_WINDOW = 14
MVRV_VOLATILITY_WINDOW = 90
DYNAMIC_STRENGTH = 5.0
MVRV_VOLATILITY_DAMPENING = 0.2

# MVRV Zone thresholds
MVRV_ZONE_DEEP_VALUE = -2.0
MVRV_ZONE_VALUE = -1.0
MVRV_ZONE_CAUTION = 1.5
MVRV_ZONE_DANGER = 2.5

# Signal weights: must sum to 1.0
WEIGHT_MVRV       = 0.43   # Core on-chain valuation
WEIGHT_CYCLE      = 0.22   # Contrarian halving cycle
WEIGHT_MONETARY   = 0.13   # Monetary policy (Fed + M2 + DXY) — external
WEIGHT_FEAR       = 0.06   # Fear composite (VIX + Fear&Greed) — external (light touch)
WEIGHT_EXCHANGE   = 0.11   # Net exchange outflow
WEIGHT_MA         = 0.05   # 200-day MA trend

# Bitcoin halving dates (confirmed historical + estimated next)
HALVING_DATES = pd.to_datetime([
    "2009-01-03",   # Genesis (reference start)
    "2012-11-28",   # 1st halving: 50 → 25 BTC/block
    "2016-07-09",   # 2nd halving: 25 → 12.5 BTC/block
    "2020-05-11",   # 3rd halving: 12.5 → 6.25 BTC/block
    "2024-04-19",   # 4th halving: 6.25 → 3.125 BTC/block
    "2028-04-17",   # 5th halving (estimated): 3.125 → 1.5625 BTC/block
])

# Exchange flow normalization window
EXCHANGE_FLOW_WINDOW = 180  # Rolling window for z-score normalization
EXCHANGE_FLOW_EMA = 21      # EMA smoothing to reduce daily noise

# External data paths (relative to project root)
_BASE = Path(__file__).parent.parent
EXT_FRED_DFF   = _BASE / "data/externals/fred/dff.csv"
EXT_FRED_M2    = _BASE / "data/externals/fred/m2sl.csv"
EXT_FRED_TIPS  = _BASE / "data/externals/fred/dfii10.csv"
EXT_FRED_YIELD = _BASE / "data/externals/fred/t10y2y.csv"
EXT_FRED_HY    = _BASE / "data/externals/fred/bamlh0a0hym2.csv"
EXT_FEAR_GREED = _BASE / "data/externals/fear_greed/fear_greed.csv"
EXT_VIX        = _BASE / "data/externals/market/vix.csv"
EXT_DXY        = _BASE / "data/externals/market/dxy.csv"


# =============================================================================
# Halving Cycle Signal
# =============================================================================


def compute_cycle_position(dates: pd.DatetimeIndex) -> pd.Series:
    """Compute normalized position [0, 1] within the current halving cycle.

    0.0 = just after a halving (supply shock just occurred)
    1.0 = just before next halving (full cycle elapsed)

    This signal captures the 4-year supply shock dynamic:
    - Early cycle (0.0-0.40): Aggressive accumulation phase
    - Mid cycle (0.40-0.65): Normal DCA
    - Late cycle (0.65-1.0): Distribution / pre-halving

    Args:
        dates: DatetimeIndex for which to compute cycle position

    Returns:
        Series of cycle positions in [0, 1]
    """
    positions = pd.Series(0.5, index=dates, dtype=float)

    for i in range(len(HALVING_DATES) - 1):
        halving_start = HALVING_DATES[i]
        halving_end = HALVING_DATES[i + 1]
        cycle_length = (halving_end - halving_start).days

        mask = (dates >= halving_start) & (dates < halving_end)
        days_elapsed = (dates[mask] - halving_start).days
        positions[mask] = days_elapsed / cycle_length

    return positions


def compute_cycle_signal(cycle_position: np.ndarray) -> np.ndarray:
    """Convert cycle position [0, 1] into a contrarian accumulation signal [-1, 1].

    Key insight (contrarian cycle hypothesis): The best DCA accumulation window
    in each 4-year Bitcoin cycle is NOT immediately after the halving (prices
    already ran up pre-halving), but during the MID-CYCLE BEAR MARKET that
    follows the peak bull market (~50-80% through the cycle).

    Historical evidence:
    - 2016 cycle: Bear bottom Oct 2018 → Feb 2019 (cycle pos ~0.62-0.72)
    - 2020 cycle: Bear bottom Nov 2022 → Jan 2023 (cycle pos ~0.65-0.68)
    - Pre-halving pump (85-100%): Prices rise sharply → neutral, don't over-buy

    Signal design (contrarian):
    - 0.00-0.15: Neutral (immediate post-halving, prices elevated from pre-halving pump)
    - 0.15-0.50: Mild reduce (bull market to peak/crash onset — over-excited phase)
    - 0.50-0.80: STRONG BUY (bear market bottom — best accumulation window)
    - 0.80-0.92: Mild buy (recovery toward pre-halving)
    - 0.92-1.00: Neutral (pre-halving pump, prices running hot)

    Args:
        cycle_position: Array of cycle positions in [0, 1]

    Returns:
        Cycle signal in [-1, 1] (positive = buy more, negative = buy less)
    """
    signal = np.piecewise(
        cycle_position.astype(float),
        [
            cycle_position < 0.15,
            (cycle_position >= 0.15) & (cycle_position < 0.50),
            (cycle_position >= 0.50) & (cycle_position < 0.80),
            (cycle_position >= 0.80) & (cycle_position < 0.92),
            cycle_position >= 0.92,
        ],
        [
            # Post-halving stabilization (0-15%): neutral
            lambda x: 0.0,
            # Bull market / crash (15-50%): mild reduce
            lambda x: -((x - 0.15) / 0.35) * 0.25,   # 0 to -0.25
            # Bear market bottom (50-80%): STRONG BUY — contrarian accumulation
            lambda x: 0.45 * np.sin(np.pi * (x - 0.50) / 0.30),  # smooth peak at 65%
            # Recovery / pre-halving (80-92%): mild buy
            lambda x: 0.25 - ((x - 0.80) / 0.12) * 0.25,  # +0.25 to 0.0
            # Pre-halving pump (92-100%): neutral
            lambda x: 0.0,
        ],
    )

    return np.clip(signal, -1, 1)


# =============================================================================
# Exchange Flow Signal
# =============================================================================


def compute_exchange_flow_signal(
    flow_in: pd.Series,
    flow_out: pd.Series,
    window: int = EXCHANGE_FLOW_WINDOW,
    ema_span: int = EXCHANGE_FLOW_EMA,
) -> pd.Series:
    """Compute normalized net exchange outflow signal.

    Net outflow = BTC leaving exchanges (HODLers accumulating)
    Net inflow = BTC entering exchanges (selling pressure)

    Process:
    1. Compute raw net outflow: flow_out - flow_in
    2. Apply EMA smoothing to reduce daily noise
    3. Normalize using rolling z-score
    4. Apply tanh transformation to bound to [-1, 1]

    Args:
        flow_in: Daily BTC exchange inflows (native units)
        flow_out: Daily BTC exchange outflows (native units)
        window: Rolling window for z-score (default 90 days)
        ema_span: EMA smoothing span (default 14 days)

    Returns:
        Signal in [-1, 1] where +1 = max accumulation, -1 = max selling
    """
    # Raw net outflow (positive = more BTC leaving exchanges = bullish)
    net_outflow = (flow_out - flow_in).fillna(0)

    # EMA smoothing to reduce daily noise
    net_smooth = net_outflow.ewm(span=ema_span, adjust=False).mean()

    # Rolling z-score normalization
    roll_mean = net_smooth.rolling(window, min_periods=window // 3).mean()
    roll_std = net_smooth.rolling(window, min_periods=window // 3).std()

    with np.errstate(divide="ignore", invalid="ignore"):
        z_score = (net_smooth - roll_mean) / roll_std.replace(0, np.nan)
    z_score = z_score.fillna(0)

    # Tanh to bound signal
    signal = np.tanh(z_score * 0.7)  # Scale factor 0.7 for moderate sensitivity

    return signal.clip(-1, 1)


# =============================================================================
# MVRV Features (adapted from example_1)
# =============================================================================


def zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling z-score."""
    mean = series.rolling(window, min_periods=window // 2).mean()
    std = series.rolling(window, min_periods=window // 2).std()
    return ((series - mean) / std).fillna(0)


def compute_mvrv_volatility(mvrv_zscore: pd.Series, window: int) -> pd.Series:
    """Compute rolling volatility of MVRV Z-score (for dampening)."""
    vol = mvrv_zscore.rolling(window, min_periods=window // 4).std()
    vol_pct = vol.rolling(window * 4, min_periods=window).apply(
        lambda x: (x.iloc[-1] > x[:-1]).sum() / max(len(x) - 1, 1)
        if len(x) > 1
        else 0.5,
        raw=False,
    )
    return vol_pct.fillna(0.5)


# =============================================================================
# External Signal Engineering
# =============================================================================


def _load_csv_series(path: Path, col: str, index_col: str = "date") -> pd.Series:
    """Load a single column from a CSV, return as date-indexed Series."""
    df = pd.read_csv(path, index_col=index_col, parse_dates=True)
    df.index = df.index.tz_localize(None)
    return df[col].sort_index()


def compute_monetary_policy_signal(price_index: pd.DatetimeIndex) -> pd.Series:
    """Compute financial conditions signal from Fed rate, M2, and DXY.

    Tight conditions (rate hikes, dollar strength, contracting M2) hurt BTC.
    Easy conditions (rate cuts, dollar weakness, M2 expansion) support BTC.

    Components:
    - Fed rate 3-month velocity: rapid hikes → strong negative signal
    - M2 YoY growth rate: expansion → positive, contraction → negative
    - DXY 90-day momentum: dollar strength → negative for BTC

    Returns:
        Signal in [-1, 1], positive = loose conditions = buy more
    """
    result = pd.Series(0.0, index=price_index)

    # --- Fed Funds Rate velocity ---
    try:
        dff = _load_csv_series(EXT_FRED_DFF, "dff").reindex(price_index, method="ffill").ffill().bfill()
        # 3-month (63-day) change in rate — captures hike/cut pace
        rate_change_3m = dff.diff(63)
        # Normalize: clip extreme moves, scale so ±2% = ±1.0
        rate_signal = np.tanh(-rate_change_3m / 2.0)  # negative: hike → reduce
        result += rate_signal * 0.50
    except Exception as e:
        logging.warning(f"Fed rate signal unavailable: {e}")

    # --- M2 Money Supply YoY growth ---
    try:
        m2 = _load_csv_series(EXT_FRED_M2, "m2sl").reindex(price_index, method="ffill").ffill().bfill()
        m2_yoy = m2.pct_change(365) * 100  # % change year-over-year
        # Normalize: 0% growth = 0 signal, +10% = +0.5, -5% = -0.5
        m2_signal = np.tanh(m2_yoy / 15.0)
        result += m2_signal * 0.25
    except Exception as e:
        logging.warning(f"M2 signal unavailable: {e}")

    # --- DXY (dollar index) momentum ---
    try:
        dxy = _load_csv_series(EXT_DXY, "dxy").reindex(price_index, method="ffill").ffill().bfill()
        # 90-day z-score of DXY: rising dollar = negative for BTC
        dxy_roll_mean = dxy.rolling(90, min_periods=30).mean()
        dxy_roll_std  = dxy.rolling(90, min_periods=30).std().replace(0, np.nan)
        dxy_z = ((dxy - dxy_roll_mean) / dxy_roll_std).fillna(0).clip(-3, 3)
        dxy_signal = np.tanh(-dxy_z * 0.5)  # strong dollar → negative signal
        result += dxy_signal * 0.25
    except Exception as e:
        logging.warning(f"DXY signal unavailable: {e}")

    return result.clip(-1, 1).fillna(0)


def compute_fear_composite_signal(price_index: pd.DatetimeIndex) -> pd.Series:
    """Compute contrarian fear signal from VIX and Crypto Fear & Greed Index.

    Extreme fear → contrarian buy signal (positive).
    Extreme greed → reduce signal (negative).

    Components (equal weight):
    - VIX z-score: spike above historical norm = elevated fear = buy more
    - Fear & Greed: extreme fear (0-25) = buy more, extreme greed (75-100) = reduce

    Returns:
        Signal in [-1, 1], positive = extreme fear = buy more (contrarian)
    """
    result = pd.Series(0.0, index=price_index)

    # --- VIX (CBOE Volatility Index) ---
    try:
        vix = _load_csv_series(EXT_VIX, "vix").reindex(price_index, method="ffill").ffill().bfill()
        # Rolling 180-day z-score: elevated VIX = fear = buy more
        vix_mean = vix.rolling(180, min_periods=60).mean()
        vix_std  = vix.rolling(180, min_periods=60).std().replace(0, np.nan)
        vix_z    = ((vix - vix_mean) / vix_std).fillna(0).clip(-4, 4)
        # Positive z-score (high VIX) → buy more
        vix_signal = np.tanh(vix_z * 0.5)
        result += vix_signal * 0.40
    except Exception as e:
        logging.warning(f"VIX signal unavailable: {e}")

    # --- Crypto Fear & Greed Index ---
    try:
        fg = _load_csv_series(EXT_FEAR_GREED, "fear_greed")
        fg = fg.reindex(price_index, method="ffill").fillna(50)  # neutral default
        # Cubic transformation: emphasizes extremes, mutes middle range.
        # At fg=10 (extreme fear): +0.51 (strong buy)
        # At fg=75 (greed):       -0.13 (mild reduce — avoids over-suppressing bull markets)
        # At fg=95 (extreme greed):-0.73 (strong reduce)
        fg_norm = (50 - fg) / 50.0        # [-1, 1], positive = fear
        fg_signal = fg_norm ** 3           # cubic: amplifies extremes
        result += fg_signal.clip(-1, 1) * 0.60
    except Exception as e:
        logging.warning(f"Fear & Greed signal unavailable: {e}")

    return result.clip(-1, 1).fillna(0)


def compute_real_yield_signal(price_index: pd.DatetimeIndex) -> pd.Series:
    """Real yield stress signal from 10-yr TIPS yield + yield curve spread.

    High real yields  → money is expensive → risk assets suffer → reduce DCA
    Inverted curve    → recession risk → risk-off → reduce DCA
    Both flip when conditions ease → buy more

    Components:
    - DFII10 (10yr real yield): rising above 0 → negative; deeply negative → positive
    - T10Y2Y (yield curve spread): inverted (negative) → negative signal
    """
    result = pd.Series(0.0, index=price_index)

    # --- 10yr TIPS real yield ---
    try:
        tips = _load_csv_series(EXT_FRED_TIPS, "dfii10") \
                   .reindex(price_index, method="ffill").ffill().bfill()
        # Real yield in %: positive real rates hurt BTC, negative support it
        # Scale: +2% real yield → −1.0 signal; −1% → +0.5
        tips_signal = np.tanh(-tips / 1.5)
        result += tips_signal * 0.50
    except Exception as e:
        logging.warning(f"TIPS real yield signal unavailable: {e}")

    # --- Yield curve spread (10yr - 2yr) ---
    try:
        yc = _load_csv_series(EXT_FRED_YIELD, "t10y2y") \
                 .reindex(price_index, method="ffill").ffill().bfill()
        # Deeply inverted curve (< −0.5%) → strong negative (recession risk)
        # Steep positive curve (> +1%) → mildly positive (growth)
        yc_signal = np.tanh(yc / 1.0)
        result += yc_signal * 0.50
    except Exception as e:
        logging.warning(f"Yield curve signal unavailable: {e}")

    return result.clip(-1, 1).fillna(0)


def compute_credit_stress_signal(price_index: pd.DatetimeIndex) -> pd.Series:
    """Credit stress signal from HY (junk bond) OAS spread.

    Widening HY spreads → credit tightening → risk-off → reduce DCA (contrarian: wait)
    Tight HY spreads   → complacency / bull market → greed → reduce (overvalued)
    Spiking then normalizing spreads → buy signal (panic has peaked)

    Uses:
    - BAMLH0A0HYM2: BofA US High Yield OAS spread
    """
    result = pd.Series(0.0, index=price_index)

    try:
        hy = _load_csv_series(EXT_FRED_HY, "bamlh0a0hym2") \
                 .reindex(price_index, method="ffill").ffill().bfill()

        # Rolling 365-day z-score: spike = credit fear = buy (contrarian)
        hy_mean = hy.rolling(365, min_periods=90).mean()
        hy_std  = hy.rolling(365, min_periods=90).std().replace(0, np.nan)
        hy_z    = ((hy - hy_mean) / hy_std).fillna(0).clip(-3, 3)

        # High z-score (spread widening) = credit panic = contrarian buy
        # BUT also: absolutely high spread (> 8%) = genuine distress = reduce
        hy_z_signal = np.tanh(hy_z * 0.6)          # contrarian on spikes
        hy_level_signal = np.tanh(-(hy - 5.0) / 3.0)  # high absolute spread → reduce

        result = (hy_z_signal * 0.60 + hy_level_signal * 0.40).clip(-1, 1)
    except Exception as e:
        logging.warning(f"HY credit spread signal unavailable: {e}")

    return result.clip(-1, 1).fillna(0)


# =============================================================================
# Feature Engineering
# =============================================================================


def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all model features.

    Features (all lagged 1 day to prevent look-ahead bias):
    - price_vs_ma: 200-day MA distance [-1, 1]
    - mvrv_zscore: MVRV Z-score (365-day window) [-4, 4]
    - mvrv_gradient: Smoothed MVRV trend [-1, 1]
    - mvrv_acceleration: Momentum of trend [-1, 1]
    - mvrv_volatility: Volatility percentile [0, 1]
    - cycle_position: Halving cycle position [0, 1]  <- NOVEL
    - cycle_signal: Cycle buy/sell signal [-1, 1]   <- NOVEL
    - exchange_signal: Net outflow signal [-1, 1]   <- NOVEL

    Args:
        df: DataFrame with price, MVRV, and exchange flow columns

    Returns:
        DataFrame with price and computed features
    """
    if PRICE_COL not in df.columns:
        raise KeyError(f"'{PRICE_COL}' not found. Available: {list(df.columns)}")

    price = df[PRICE_COL].loc["2010-07-18":].copy()

    # --- 200-day MA ---
    ma = price.rolling(MA_WINDOW, min_periods=MA_WINDOW // 2).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        price_vs_ma = ((price / ma) - 1).clip(-1, 1).fillna(0)

    # --- MVRV features ---
    if MVRV_COL in df.columns:
        mvrv = df[MVRV_COL].loc[price.index]

        mvrv_z = zscore(mvrv, MVRV_ROLLING_WINDOW).clip(-4, 4)

        gradient_raw = mvrv_z.diff(MVRV_GRADIENT_WINDOW)
        gradient_smooth = gradient_raw.ewm(span=MVRV_GRADIENT_WINDOW, adjust=False).mean()
        mvrv_gradient = np.tanh(gradient_smooth * 2).fillna(0)

        accel_raw = mvrv_gradient.diff(MVRV_ACCEL_WINDOW)
        mvrv_acceleration = np.tanh(
            accel_raw.ewm(span=MVRV_ACCEL_WINDOW, adjust=False).mean() * 3
        ).fillna(0)

        mvrv_volatility = compute_mvrv_volatility(mvrv_z, MVRV_VOLATILITY_WINDOW)
    else:
        logging.warning("MVRV data not found — falling back to neutral MVRV signals")
        mvrv_z = pd.Series(0.0, index=price.index)
        mvrv_gradient = pd.Series(0.0, index=price.index)
        mvrv_acceleration = pd.Series(0.0, index=price.index)
        mvrv_volatility = pd.Series(0.5, index=price.index)

    # --- Halving Cycle Position (NOVEL) ---
    cycle_pos = compute_cycle_position(price.index)
    cycle_sig = pd.Series(
        compute_cycle_signal(cycle_pos.values),
        index=price.index,
    )

    # --- Net Exchange Outflow Signal (NOVEL) ---
    if FLOW_IN_COL in df.columns and FLOW_OUT_COL in df.columns:
        flow_in = df[FLOW_IN_COL].loc[price.index].fillna(0)
        flow_out = df[FLOW_OUT_COL].loc[price.index].fillna(0)
        exchange_sig = compute_exchange_flow_signal(flow_in, flow_out)
    else:
        logging.warning("Exchange flow data not found — falling back to neutral signal")
        exchange_sig = pd.Series(0.0, index=price.index)

    # --- External signals (NOVEL: macro + fear) ---
    monetary_sig    = compute_monetary_policy_signal(price.index)
    fear_sig        = compute_fear_composite_signal(price.index)
    real_yield_sig  = compute_real_yield_signal(price.index)
    credit_sig      = compute_credit_stress_signal(price.index)

    # --- Assemble and lag all signals by 1 day (no look-ahead bias) ---
    features = pd.DataFrame(
        {
            PRICE_COL:           price,
            "price_ma":          ma,
            "price_vs_ma":       price_vs_ma,
            "mvrv_zscore":       mvrv_z,
            "mvrv_gradient":     mvrv_gradient,
            "mvrv_acceleration": mvrv_acceleration,
            "mvrv_volatility":   mvrv_volatility,
            "cycle_position":    cycle_pos,
            "cycle_signal":      cycle_sig,
            "exchange_signal":   exchange_sig,
            "monetary_signal":   monetary_sig,
            "fear_signal":       fear_sig,
            "real_yield_signal": real_yield_sig,  # NEW: TIPS + yield curve
            "credit_signal":     credit_sig,       # NEW: HY spread
        },
        index=price.index,
    )

    signal_cols = [
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
        "real_yield_signal",
        "credit_signal",
    ]
    features[signal_cols] = features[signal_cols].shift(1)

    features["mvrv_volatility"] = features["mvrv_volatility"].fillna(0.5)
    features = features.fillna(0)

    logging.info(
        f"Features computed: {len(features)} rows, "
        f"{features.index.min().date()} to {features.index.max().date()}"
    )
    return features


# =============================================================================
# Dynamic Multiplier
# =============================================================================


def compute_asymmetric_mvrv_boost(mvrv_zscore: np.ndarray) -> np.ndarray:
    """Asymmetric boost for extreme MVRV values (adapted from example_1)."""
    boost = np.zeros_like(mvrv_zscore)

    # Deep value: strong quadratic boost
    boost = np.where(
        mvrv_zscore < MVRV_ZONE_DEEP_VALUE,
        0.8 * (mvrv_zscore - MVRV_ZONE_DEEP_VALUE) ** 2 + 0.5,
        boost,
    )
    # Moderate value: linear boost
    boost = np.where(
        (mvrv_zscore >= MVRV_ZONE_DEEP_VALUE) & (mvrv_zscore < MVRV_ZONE_VALUE),
        -0.5 * mvrv_zscore,
        boost,
    )
    # Caution zone: moderate negative
    boost = np.where(
        (mvrv_zscore >= MVRV_ZONE_CAUTION) & (mvrv_zscore < MVRV_ZONE_DANGER),
        -0.3 * (mvrv_zscore - MVRV_ZONE_CAUTION),
        boost,
    )
    # Danger zone: strong quadratic negative
    boost = np.where(
        mvrv_zscore >= MVRV_ZONE_DANGER,
        -0.5 * (mvrv_zscore - MVRV_ZONE_DANGER) ** 2 - 0.3,
        boost,
    )
    return boost


def compute_dynamic_multiplier(
    price_vs_ma: np.ndarray,
    mvrv_zscore: np.ndarray,
    mvrv_gradient: np.ndarray,
    mvrv_acceleration: np.ndarray,
    mvrv_volatility: np.ndarray,
    cycle_signal: np.ndarray,
    exchange_signal: np.ndarray,
    monetary_signal: np.ndarray,
    fear_signal: np.ndarray,
) -> np.ndarray:
    """Compute weight multiplier from six signal sources.

    Signal composition:
    - MVRV (40%):           Core on-chain valuation
    - Halving Cycle (20%):  Contrarian macro timing
    - Monetary Policy (15%):Fed rate + M2 + DXY (EXTERNAL)
    - Fear Composite (10%): VIX + Fear & Greed (EXTERNAL)
    - Exchange Flow (10%):  Net outflow accumulation signal
    - MA 200d (5%):         Trend confirmation

    Modulated by MVRV acceleration (momentum) and volatility dampening.
    """
    # 1. MVRV valuation signal
    mvrv_signal = -mvrv_zscore + compute_asymmetric_mvrv_boost(mvrv_zscore)

    # 2. MA trend signal
    ma_signal = -price_vs_ma

    # 3. Exchange signal: cap negative in deep value (capitulation ≠ structural selling)
    in_deep_value = mvrv_zscore < -1.5
    exchange_signal_adj = np.where(
        in_deep_value,
        np.maximum(exchange_signal, -0.2),
        exchange_signal,
    )

    # 4. Monetary signal: during extreme tightening, scale down its negative impact
    #    when MVRV is already in deep value (both already say reduce — don't double-penalize)
    monetary_signal_adj = np.where(
        in_deep_value,
        np.maximum(monetary_signal, -0.3),
        monetary_signal,
    )

    # Weighted combination
    combined = (
        mvrv_signal          * WEIGHT_MVRV
        + cycle_signal       * WEIGHT_CYCLE
        + monetary_signal_adj* WEIGHT_MONETARY
        + fear_signal        * WEIGHT_FEAR
        + exchange_signal_adj* WEIGHT_EXCHANGE
        + ma_signal          * WEIGHT_MA
    )

    # MVRV acceleration modifier (momentum amplification / dampening)
    # Same direction as gradient = momentum building = amplify
    same_dir = (mvrv_acceleration * mvrv_gradient) > 0
    accel_mod = np.where(
        same_dir,
        1.0 + 0.15 * np.abs(mvrv_acceleration),
        1.0 - 0.10 * np.abs(mvrv_acceleration),
    )
    accel_mod = np.clip(accel_mod, 0.85, 1.15)
    combined = combined * accel_mod

    # Volatility dampening in extreme uncertainty periods
    vol_damp = np.where(
        mvrv_volatility > 0.8,
        1.0 - MVRV_VOLATILITY_DAMPENING * (mvrv_volatility - 0.8) / 0.2,
        1.0,
    )
    combined = combined * vol_damp

    # Scale, clip, exponentiate
    adjustment = np.clip(combined * DYNAMIC_STRENGTH, -5, 100)
    multiplier = np.exp(adjustment)
    return np.where(np.isfinite(multiplier), multiplier, 1.0)


# =============================================================================
# Weight Computation API
# =============================================================================


def compute_weights_fast(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    n_past: int | None = None,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """Compute weights for a date window using precomputed features."""
    df = features_df.loc[start_date:end_date]
    if df.empty:
        return pd.Series(dtype=float)

    n = len(df)
    base = np.ones(n) / n

    price_vs_ma       = _clean_array(df["price_vs_ma"].values)
    mvrv_zscore       = _clean_array(df["mvrv_zscore"].values)
    mvrv_gradient     = _clean_array(df["mvrv_gradient"].values)
    mvrv_acceleration = _clean_array(df["mvrv_acceleration"].values)
    mvrv_volatility   = _clean_array(df["mvrv_volatility"].values)
    mvrv_volatility   = np.where(mvrv_volatility == 0, 0.5, mvrv_volatility)
    cycle_signal      = _clean_array(df["cycle_signal"].values)
    exchange_signal   = _clean_array(df["exchange_signal"].values)
    monetary_signal   = _clean_array(df.get("monetary_signal", pd.Series(0.0, index=df.index)).values)
    fear_signal       = _clean_array(df.get("fear_signal",     pd.Series(0.0, index=df.index)).values)

    dyn = compute_dynamic_multiplier(
        price_vs_ma,
        mvrv_zscore,
        mvrv_gradient,
        mvrv_acceleration,
        mvrv_volatility,
        cycle_signal,
        exchange_signal,
        monetary_signal,
        fear_signal,
    )
    raw = base * dyn

    if n_past is None:
        n_past = n
    weights = allocate_sequential_stable(raw, n_past, locked_weights)

    return pd.Series(weights, index=df.index)


def compute_window_weights(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """Compute weights for a date range with lock-on-compute stability.

    Two modes:
    1. BACKTEST (locked_weights=None): Signal-based allocation
    2. PRODUCTION (locked_weights provided): DB-backed stability

    Args:
        features_df: DataFrame from precompute_features()
        start_date: Investment window start
        end_date: Investment window end
        current_date: Current date (past/future boundary)
        locked_weights: Optional locked weights from database

    Returns:
        Series of weights summing to 1.0
    """
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")

    missing = full_range.difference(features_df.index)
    if len(missing) > 0:
        placeholder = pd.DataFrame(
            {col: 0.0 for col in features_df.columns},
            index=missing,
        )
        placeholder["mvrv_volatility"] = 0.5
        for neutral_col in ["monetary_signal", "fear_signal", "exchange_signal", "cycle_signal"]:
            if neutral_col in placeholder.columns:
                placeholder[neutral_col] = 0.0
        features_df = pd.concat([features_df, placeholder]).sort_index()

    past_end = min(current_date, end_date)
    if start_date <= past_end:
        n_past = len(pd.date_range(start=start_date, end=past_end, freq="D"))
    else:
        n_past = 0

    weights = compute_weights_fast(
        features_df, start_date, end_date, n_past, locked_weights
    )
    return weights.reindex(full_range, fill_value=0.0)
