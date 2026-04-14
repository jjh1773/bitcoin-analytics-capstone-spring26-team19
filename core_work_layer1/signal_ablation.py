"""
Signal Ablation Study — 누적 수익률 기준 최적 외부 데이터 통합 전략
=================================================================
평가 기준: mean_ratio = mean(dynamic_spd / uniform_spd) across all 1-year windows
           = "균등 DCA 대비 평균 누적 수익률 배수"

테스트 신호 (9개):
  [기존 CoinMetrics 활용]
  A. MVRV Z-score          — 기준선 신호
  B. Halving Cycle         — 기존 novel
  C. Exchange Flow (daily) — 기존 novel
  D. HashRate Puell        — 채굴자 수익성 / 항복 지표 (신규)
  E. Exchange Supply Level — 거래소 BTC 잔량 추세 (신규)
  F. NVT Ratio             — 온체인 거래량 대비 가치평가 (신규)
  G. Panic Volume          — 패닉 셀링 볼륨 급등 포착 (신규)

  [외부 다운로드]
  H. Google Trends         — 소매 투자자 FOMO 감지
  I. Funding Rate          — 선물 시장 포지션 편향
  J. Stablecoin Supply     — 유동성 / 드라이 파우더

Run from project root:
    python -m work.signal_ablation
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.WARNING)

from template.prelude_template import load_data
from template.model_development_template import allocate_sequential_stable, _clean_array
from work.model_development import (
    precompute_features,
    compute_dynamic_multiplier,
    WEIGHT_MVRV, WEIGHT_CYCLE, WEIGHT_EXCHANGE, WEIGHT_MA,
)

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
EXT  = ROOT / "data" / "externals"

# ─── Load base data ───────────────────────────────────────────────────────────
print("Loading data...")
BTC_DF   = load_data()
FEATURES = precompute_features(BTC_DF)
PRICE    = BTC_DF["PriceUSD_coinmetrics"].dropna()

BACKTEST_START = pd.Timestamp("2018-01-01")
BACKTEST_END   = PRICE.index.max() - pd.Timedelta(days=365)


# =============================================================================
# New Signal Functions
# =============================================================================

def _zscore_rolling(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window, min_periods=window // 3).mean()
    sd = s.rolling(window, min_periods=window // 3).std().replace(0, np.nan)
    return ((s - m) / sd).fillna(0)


def compute_puell_multiple_signal(price_index: pd.DatetimeIndex) -> pd.Series:
    """HashRate Puell Multiple: current / 365d MA.
    낮은 Puell = 채굴자 항복 = 저점 신호 → 매수 증가
    높은 Puell = 채굴자 초과 수익 = 고점 신호 → 매수 감소
    """
    hr = BTC_DF["HashRate"].reindex(price_index, method="ffill").ffill().bfill()
    hr_ma365 = hr.rolling(365, min_periods=90).mean()
    puell = (hr / hr_ma365.replace(0, np.nan)).fillna(1.0)
    # 낮은 Puell → 강한 매수 신호
    signal = np.tanh(-(np.log(puell + 1e-9)) * 2.0)
    return signal.clip(-1, 1).fillna(0)


def compute_exchange_supply_signal(price_index: pd.DatetimeIndex) -> pd.Series:
    """거래소 BTC 잔량 추세 (SplyExNtv).
    잔량 감소 추세 = HODLer 인출 = 강세 신호
    잔량 증가 추세 = 매도 압력 증가 = 약세 신호
    """
    supply = BTC_DF["SplyExNtv"].reindex(price_index, method="ffill").ffill().bfill()
    # 90일 변화율 (YoY 너무 느림, 장기 추세 포착)
    change_90d = supply.pct_change(90)
    signal = np.tanh(-change_90d * 15.0)  # 감소 → 양수
    return signal.clip(-1, 1).fillna(0)


def compute_nvt_signal(price_index: pd.DatetimeIndex) -> pd.Series:
    """NVT Ratio: Market Cap / On-chain Transaction Count.
    고 NVT = 가격 대비 온체인 활동 부족 = 과대평가 → 매수 감소
    저 NVT = 온체인 활동이 가격 지지 = 과소평가 → 매수 증가
    """
    mcap   = BTC_DF["CapMrktCurUSD"].reindex(price_index, method="ffill").ffill().bfill()
    tx_cnt = BTC_DF["TxCnt"].reindex(price_index, method="ffill").ffill().bfill()
    nvt_raw = (mcap / tx_cnt.replace(0, np.nan)).fillna(method="ffill")
    nvt_z   = _zscore_rolling(np.log(nvt_raw + 1), 365)
    signal  = np.tanh(-nvt_z * 0.5)  # 고 NVT → 음수
    return signal.clip(-1, 1).fillna(0)


def compute_panic_volume_signal(price_index: pd.DatetimeIndex) -> pd.Series:
    """패닉 볼륨: 거래량 급등 + 가격 하락 = 항복 매도 = 역방향 매수.
    볼륨 z-score가 높고 동시에 가격이 떨어지면 강한 매수 신호.
    """
    vol   = BTC_DF["volume_reported_spot_usd_1d"].reindex(price_index, method="ffill").ffill().bfill()
    price = PRICE.reindex(price_index, method="ffill").ffill().bfill()

    vol_z   = _zscore_rolling(vol, 90).clip(-4, 4)
    ret_5d  = price.pct_change(5)  # 5일 수익률

    # 볼륨 급등 + 가격 하락 = 패닉 매도 = 매수 신호
    panic = vol_z * (-ret_5d.clip(-0.5, 0) * 10)
    signal = np.tanh(panic * 0.5)
    return signal.clip(-1, 1).fillna(0)


def compute_google_trends_signal(price_index: pd.DatetimeIndex) -> pd.Series:
    """Google Trends 'bitcoin' 검색량.
    극단적 고점 = 소매 FOMO = 매수 감소
    극단적 저점 = 무관심 = 매수 증가
    """
    path = EXT / "google_trends" / "bitcoin_trends.csv"
    gt = pd.read_csv(path, index_col="date", parse_dates=True)["bitcoin_trends"]
    gt = gt.reindex(price_index, method="ffill").ffill().bfill()
    # 180일 z-score: 상대적 관심도
    gt_z = _zscore_rolling(gt, 180).clip(-3, 3)
    signal = np.tanh(-gt_z * 0.6)  # 높은 검색량 → 음수 신호
    return signal.clip(-1, 1).fillna(0)


def compute_funding_rate_signal(price_index: pd.DatetimeIndex) -> pd.Series:
    """선물 펀딩 레이트.
    음수 펀딩 = 숏 우세 = 과도한 비관 = 역방향 매수
    양수 펀딩 = 롱 우세 = 과도한 낙관 = 매수 감소
    데이터: 2019-09~ (이전 기간은 0으로 패딩)
    """
    path = EXT / "funding_rate" / "funding_rate.csv"
    fr = pd.read_csv(path, index_col="date", parse_dates=True)["funding_rate"]
    fr = fr.reindex(price_index).fillna(0)  # 데이터 없는 구간은 중립
    # 펀딩레이트 자체가 매우 작음 (0.01%대), 90일 z-score
    fr_z = _zscore_rolling(fr, 90).clip(-3, 3)
    signal = np.tanh(-fr_z * 0.7)  # 음수 펀딩 → 양수 신호
    return signal.clip(-1, 1).fillna(0)


def compute_stablecoin_signal(price_index: pd.DatetimeIndex) -> pd.Series:
    """스테이블코인 공급 추세.
    공급 증가 = 드라이 파우더 축적 = 강세 신호
    공급 감소 = 크립토에 이미 투입 = 약세 신호
    """
    path = EXT / "stablecoin" / "stablecoin_mcap.csv"
    sc = pd.read_csv(path, index_col="date", parse_dates=True)["stablecoin_mcap"]
    sc = sc.reindex(price_index, method="ffill").ffill().bfill()
    # 90일 변화율
    sc_chg = sc.pct_change(90)
    signal = np.tanh(sc_chg * 8.0)
    return signal.clip(-1, 1).fillna(0)


# =============================================================================
# Ablation Engine
# =============================================================================

def run_fast_backtest(extra_signals: dict[str, pd.Series],
                      extra_weight: float = 0.0,
                      base_weights: dict | None = None) -> dict:
    """
    fast backtest — 1년 롤링 창.
    extra_signals: {name: series} — 추가 신호들 (등가중 결합)
    extra_weight: 추가 신호 전체 합산 가중치 (MVRV 등에서 차감)
    """
    if base_weights is None:
        base_weights = {
            "mvrv":     WEIGHT_MVRV,
            "cycle":    WEIGHT_CYCLE,
            "exchange": WEIGHT_EXCHANGE,
            "ma":       WEIGHT_MA,
            "monetary": 0.13,
            "fear":     0.06,
        }

    windows = pd.date_range(BACKTEST_START, BACKTEST_END, freq="D")

    ratios    = []
    year_data = []

    for start in windows:
        end = start + pd.Timedelta(days=364)
        feat_w = FEATURES.loc[start:end]
        price_w = PRICE.loc[start:end]
        if len(price_w) < 300 or feat_w.empty:
            continue

        n    = len(price_w)
        base = np.ones(n) / n

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

        # extra 신호 결합 (등가중)
        if extra_signals:
            combined_extra = np.zeros(n)
            for sig_series in extra_signals.values():
                sig_slice = _clean_array(sig_series.reindex(feat_w.index).fillna(0).values)
                combined_extra += sig_slice
            combined_extra /= len(extra_signals)
        else:
            combined_extra = np.zeros(n)

        # 가중치 재조정: extra_weight를 MVRV에서 차감
        w_mvrv = max(0.10, base_weights["mvrv"] - extra_weight)
        w_extra = extra_weight if extra_signals else 0.0

        # MVRV signal
        from work.model_development import compute_asymmetric_mvrv_boost
        mvrv_signal = -mz + compute_asymmetric_mvrv_boost(mz)
        ma_signal   = -pvm

        in_deep_value = mz < -1.5
        es_adj = np.where(in_deep_value, np.maximum(es, -0.2), es)
        ms_adj = np.where(in_deep_value, np.maximum(ms, -0.3), ms)

        combined = (
            mvrv_signal  * w_mvrv
            + cs         * base_weights["cycle"]
            + ms_adj     * base_weights["monetary"]
            + fs         * base_weights["fear"]
            + es_adj     * base_weights["exchange"]
            + ma_signal  * base_weights["ma"]
            + combined_extra * w_extra
        )

        # accel modifier
        same_dir = (ma_ * mg) > 0
        accel_mod = np.where(same_dir, 1.0 + 0.15 * np.abs(ma_), 1.0 - 0.10 * np.abs(ma_))
        accel_mod = np.clip(accel_mod, 0.85, 1.15)
        combined  = combined * accel_mod

        # vol dampening
        vol_damp = np.where(mv > 0.8, 1.0 - 0.2 * (mv - 0.8) / 0.2, 1.0)
        combined  = combined * vol_damp

        adjustment = np.clip(combined * 5.0, -5, 100)
        multiplier = np.exp(adjustment)
        multiplier = np.where(np.isfinite(multiplier), multiplier, 1.0)

        raw     = base * multiplier
        weights = allocate_sequential_stable(raw, n, None)

        sats          = 1e8 / price_w.values
        dynamic_spd   = np.sum(weights * sats)
        uniform_spd   = np.mean(sats)
        ratio         = dynamic_spd / uniform_spd

        ratios.append(ratio)
        year_data.append({"year": start.year, "ratio": ratio, "excess_spd": dynamic_spd - uniform_spd})

    df_y = pd.DataFrame(year_data)
    by_year = df_y.groupby("year")["ratio"].mean()

    return {
        "mean_ratio":   np.mean(ratios),
        "median_ratio": np.median(ratios),
        "win_rate":     np.mean([r > 1.0 for r in ratios]) * 100,
        "n_windows":    len(ratios),
        "by_year":      by_year,
    }


# =============================================================================
# Precompute all new signals
# =============================================================================

print("Precomputing all signals...")
price_idx = PRICE.loc[BACKTEST_START:].index

SIGNALS = {
    "D_puell":      compute_puell_multiple_signal(price_idx),
    "E_ex_supply":  compute_exchange_supply_signal(price_idx),
    "F_nvt":        compute_nvt_signal(price_idx),
    "G_panic_vol":  compute_panic_volume_signal(price_idx),
    "H_gtrends":    compute_google_trends_signal(price_idx),
    "I_funding":    compute_funding_rate_signal(price_idx),
    "J_stablecoin": compute_stablecoin_signal(price_idx),
}

SIGNAL_LABELS = {
    "D_puell":      "HashRate Puell Multiple     (CoinMetrics)",
    "E_ex_supply":  "Exchange Supply Level       (CoinMetrics)",
    "F_nvt":        "NVT Ratio                  (CoinMetrics)",
    "G_panic_vol":  "Panic Volume               (CoinMetrics)",
    "H_gtrends":    "Google Trends 'bitcoin'    (External)",
    "I_funding":    "Funding Rate               (Binance)",
    "J_stablecoin": "Stablecoin Supply          (DeFiLlama)",
}

print("Done.\n")


# =============================================================================
# Phase 1: Baseline
# =============================================================================

print("=" * 70)
print("PHASE 1: BASELINE (기존 모델)")
print("=" * 70)
baseline = run_fast_backtest({}, extra_weight=0.0)
print(f"  Mean Ratio:   {baseline['mean_ratio']:.5f}  (+{(baseline['mean_ratio']-1)*100:.3f}%)")
print(f"  Win Rate:     {baseline['win_rate']:.2f}%")
print(f"  Windows:      {baseline['n_windows']}")


# =============================================================================
# Phase 2: Individual Signal Ablation
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 2: 신호별 단독 추가 효과 (extra_weight=0.10, MVRV에서 차감)")
print("=" * 70)

EXTRA_WEIGHT = 0.10  # 각 신호에 부여할 가중치

individual_results = {}
for key, sig in SIGNALS.items():
    result = run_fast_backtest({key: sig}, extra_weight=EXTRA_WEIGHT)
    delta  = (result["mean_ratio"] - baseline["mean_ratio"]) * 100
    individual_results[key] = result
    marker = "▲" if delta > 0 else "▼"
    print(f"  {marker} {SIGNAL_LABELS[key]:<44} "
          f"ratio={result['mean_ratio']:.5f}  "
          f"delta={delta:+.4f}%  "
          f"WR={result['win_rate']:.1f}%")


# =============================================================================
# Phase 3: Weight Sensitivity (winning signals)
# =============================================================================

# Sort by delta
sorted_signals = sorted(individual_results.items(),
                        key=lambda x: x[1]["mean_ratio"], reverse=True)
winning = [(k, SIGNALS[k]) for k, r in sorted_signals if r["mean_ratio"] > baseline["mean_ratio"]]
losing  = [(k, SIGNALS[k]) for k, r in sorted_signals if r["mean_ratio"] <= baseline["mean_ratio"]]

print(f"\n  → 유효 신호: {len(winning)}개 | 무효: {len(losing)}개")
print(f"  → 유효: {[SIGNAL_LABELS[k][:20] for k,_ in winning]}")

print("\n" + "=" * 70)
print("PHASE 3: 유효 신호들의 최적 가중치 탐색")
print("=" * 70)

best_combo = None
best_ratio = baseline["mean_ratio"]

if winning:
    # Grid search over weight for each winning signal individually
    weight_candidates = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
    weight_results = {}
    for key, sig in winning:
        results_by_w = {}
        for w in weight_candidates:
            r = run_fast_backtest({key: sig}, extra_weight=w)
            results_by_w[w] = r["mean_ratio"]
        best_w = max(results_by_w, key=results_by_w.get)
        weight_results[key] = best_w
        print(f"  {SIGNAL_LABELS[key][:44]}  최적 weight={best_w:.2f}  "
              f"ratio={results_by_w[best_w]:.5f}")


# =============================================================================
# Phase 4: Greedy Forward Selection (combination)
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 4: Greedy Forward Selection — 최적 신호 조합 탐색")
print("=" * 70)

selected  = {}
remaining = dict(winning)
current_ratio = baseline["mean_ratio"]
combo_history = [("(baseline)", baseline["mean_ratio"])]

while remaining:
    best_key, best_r, best_w_val = None, current_ratio, 0.10

    for key, sig in remaining.items():
        test_signals = {**selected, key: sig}
        # Try multiple weight distributions
        for total_w in [0.08, 0.12, 0.15, 0.18, 0.20]:
            r = run_fast_backtest(test_signals, extra_weight=total_w)
            if r["mean_ratio"] > best_r:
                best_r, best_key, best_w_val = r["mean_ratio"], key, total_w

    if best_key is None:
        break  # no improvement

    selected[best_key] = remaining.pop(best_key)
    current_ratio = best_r
    combo_history.append((SIGNAL_LABELS[best_key][:35], best_r))
    delta = (best_r - baseline["mean_ratio"]) * 100
    print(f"  + {SIGNAL_LABELS[best_key]:<44} ratio={best_r:.5f}  "
          f"cumδ={delta:+.4f}%  w={best_w_val:.2f}")


# =============================================================================
# Phase 5: Final Optimal vs Baseline
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 5: 최적 조합 최종 성과 (연도별)")
print("=" * 70)

if selected:
    # Fine-tune final weight
    best_final_ratio = 0
    best_final_w     = 0.10
    best_final_result = None
    for w in [0.08, 0.10, 0.12, 0.15, 0.18]:
        r = run_fast_backtest(selected, extra_weight=w)
        if r["mean_ratio"] > best_final_ratio:
            best_final_ratio, best_final_w, best_final_result = r["mean_ratio"], w, r
    final = best_final_result
else:
    final = baseline
    best_final_w = 0

print(f"  선택된 신호: {[SIGNAL_LABELS[k][:25] for k in selected]}")
print(f"  최적 통합 가중치: {best_final_w:.2f}")
print()
print(f"  {'연도':<6}  {'기준선':>10}  {'최적조합':>10}  {'개선':>8}")
print(f"  {'-'*40}")
for yr in sorted(set(baseline["by_year"].index) | set(final["by_year"].index)):
    b = baseline["by_year"].get(yr, 1.0)
    f = final["by_year"].get(yr, 1.0) if final != baseline else b
    diff = (f - b) * 100
    marker = "▲" if diff > 0.01 else ("▼" if diff < -0.01 else "─")
    print(f"  {yr}   {b:.5f}     {f:.5f}    {marker} {diff:+.4f}%")

print()
print("=" * 70)
print("최종 요약")
print("=" * 70)
print(f"  기준선 mean_ratio:       {baseline['mean_ratio']:.6f}  (+{(baseline['mean_ratio']-1)*100:.3f}%)")
print(f"  최적 조합 mean_ratio:    {final['mean_ratio']:.6f}  (+{(final['mean_ratio']-1)*100:.3f}%)")
print(f"  추가 개선:               {(final['mean_ratio']-baseline['mean_ratio'])*100:+.4f}%")
print(f"  기준선 win rate:         {baseline['win_rate']:.2f}%")
print(f"  최적 조합 win rate:      {final['win_rate']:.2f}%")
print()
print("통합 전략 권고:")
for k in selected:
    print(f"  ✓ {SIGNAL_LABELS[k]}")
for k, _ in losing:
    print(f"  ✗ {SIGNAL_LABELS[k]} (성능 개선 없음)")
