"""Backtest runner for the Halving Cycle + On-Chain Accumulation DCA model.

Compares our novel model against:
1. Uniform DCA (baseline)
2. Template model (200-day MA only)
3. Example 1 (MVRV + MA + Polymarket)

Run from the project root:
    python -m work.run_backtest
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Import template infrastructure
from template.prelude_template import load_data
from template.backtest_template import run_full_analysis

# Import our novel model
from work.model_development import precompute_features, compute_window_weights

# Global precomputed features
_FEATURES_DF = None


def compute_weights_wrapper(df_window: pd.DataFrame) -> pd.Series:
    """Adapter from backtest engine interface to our model's compute_window_weights."""
    global _FEATURES_DF

    if _FEATURES_DF is None:
        raise ValueError("Features not precomputed. Call precompute_features() first.")

    if df_window.empty:
        return pd.Series(dtype=float)

    start_date = df_window.index.min()
    end_date = df_window.index.max()
    current_date = end_date  # Backtesting: all dates are in the past

    return compute_window_weights(_FEATURES_DF, start_date, end_date, current_date)


def main():
    global _FEATURES_DF

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("=" * 60)
    logging.info("Bitcoin DCA - Halving Cycle + On-Chain Accumulation Model")
    logging.info("Novel signals: Halving Cycle Position + Net Exchange Outflow")
    logging.info("=" * 60)

    # 1. Load data
    btc_df = load_data()

    # 2. Precompute features (including novel signals)
    logging.info("Precomputing features (MVRV + Halving Cycle + Exchange Flow)...")
    _FEATURES_DF = precompute_features(btc_df)

    # Log feature coverage summary
    logging.info(f"Feature summary:")
    for col in ["mvrv_zscore", "cycle_signal", "exchange_signal", "price_vs_ma"]:
        if col in _FEATURES_DF.columns:
            non_zero = (_FEATURES_DF[col] != 0).sum()
            logging.info(f"  {col}: {non_zero} non-zero rows")

    # 3. Define output directory
    base_dir = Path(__file__).parent
    output_dir = base_dir / "output"

    # 4. Run backtest
    run_full_analysis(
        btc_df=btc_df,
        features_df=_FEATURES_DF,
        compute_weights_fn=compute_weights_wrapper,
        output_dir=output_dir,
        strategy_label="Halving Cycle + On-Chain Accumulation DCA",
    )


if __name__ == "__main__":
    main()
