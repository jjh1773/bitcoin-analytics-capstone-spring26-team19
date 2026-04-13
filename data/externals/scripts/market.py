"""Download market data (VIX, DXY) via yfinance.

No API key required. Uses Yahoo Finance.

Saved files (data/externals/market/):
  vix.csv   - CBOE Volatility Index (^VIX) — fear gauge → Fear composite signal
  dxy.csv   - US Dollar Index via UUP ETF  — macro headwind → Monetary policy signal

Optionally via Alpaca (requires API keys in environment):
  Set ALPACA_API_KEY and ALPACA_API_SECRET env vars to use Alpaca.
  Falls back to yfinance if keys not set.

Run from project root:
    python data/externals/scripts/market.py
"""

import os
from pathlib import Path
import pandas as pd
import yfinance as yf
from datetime import datetime

START = "2017-01-01"
END   = datetime.today().strftime("%Y-%m-%d")
OUT   = Path(__file__).parent.parent / "market"   # data/externals/market/

TICKERS = {
    "vix": "^VIX",   # CBOE Volatility Index
    "dxy": "UUP",    # Invesco DB US Dollar Index Bullish Fund (DXY proxy)
}

def download_yfinance(key: str, ticker: str) -> pd.DataFrame:
    print(f"  Downloading {ticker} via yfinance ...")
    raw = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=True)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Flatten multi-level columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Close"]].copy()
    df.columns = [key]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    df = df.dropna()
    return df


def try_alpaca(key: str, ticker: str) -> pd.DataFrame | None:
    """Try to download via Alpaca if API keys are configured."""
    api_key    = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        return None

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        client = StockHistoricalDataClient(api_key, api_secret)
        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=START,
            end=END,
        )
        bars = client.get_stock_bars(request).df
        bars = bars.reset_index()
        bars["date"] = pd.to_datetime(bars["timestamp"]).dt.tz_localize(None).dt.normalize()
        result = bars.set_index("date")[["close"]].rename(columns={"close": key})
        print(f"  Alpaca OK: {ticker}")
        return result
    except Exception as e:
        print(f"  Alpaca failed for {ticker} ({e}), falling back to yfinance")
        return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for key, ticker in TICKERS.items():
        try:
            # Try Alpaca first (only for non-index tickers like UUP)
            df = None
            if not ticker.startswith("^"):
                df = try_alpaca(key, ticker)

            if df is None:
                df = download_yfinance(key, ticker)

            out_path = OUT / f"{key}.csv"
            df.to_csv(out_path)
            print(f"  Saved: {out_path} ({len(df)} rows, "
                  f"{df.index.min().date()} ~ {df.index.max().date()})")
        except Exception as e:
            print(f"  ERROR downloading {ticker}: {e}")


if __name__ == "__main__":
    print("=== Market Data Download (yfinance / Alpaca) ===")
    main()
    print("Done.")
