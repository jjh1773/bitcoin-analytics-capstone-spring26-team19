"""Download macroeconomic data from FRED (Federal Reserve Economic Data).

No API key required — uses pandas_datareader web reader.

Saved files (data/externals/fred/):
  dff.csv   - Daily Fed Funds Rate (%)          → Monetary policy velocity signal
  m2sl.csv  - M2 Money Supply (monthly→daily)   → Liquidity signal

Run from project root:
    python data/externals/scripts/fred.py
"""

from pathlib import Path
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime

START = "2017-01-01"
END   = datetime.today().strftime("%Y-%m-%d")
OUT   = Path(__file__).parent.parent / "fred"   # data/externals/fred/

SERIES = {
    "dff":            "DFF",            # Fed Funds Effective Rate (daily)
    "m2sl":           "M2SL",           # M2 Money Supply (monthly → daily)
    "dfii10":         "DFII10",         # 10-yr TIPS real yield (daily)
    "t10y2y":         "T10Y2Y",         # 10yr-2yr yield spread (daily)
    "bamlh0a0hym2":   "BAMLH0A0HYM2",  # HY credit spread OAS (daily)
}


def download_fred(key: str, series_id: str) -> pd.DataFrame:
    print(f"  Downloading FRED: {series_id} ...")
    df = web.DataReader(series_id, "fred", START, END)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    df.columns = [key]
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for key, series_id in SERIES.items():
        try:
            df = download_fred(key, series_id)

            if key in ("m2sl", "dfii10", "t10y2y", "bamlh0a0hym2"):
                daily_idx = pd.date_range(df.index.min(), END, freq="D")
                df = df.reindex(daily_idx).ffill()
                df.index.name = "date"

            out_path = OUT / f"{key}.csv"
            df.to_csv(out_path)
            print(f"  Saved: {out_path} ({len(df)} rows, "
                  f"{df.index.min().date()} ~ {df.index.max().date()})")
        except Exception as e:
            print(f"  ERROR downloading {series_id}: {e}")


if __name__ == "__main__":
    print("=== FRED Data Download ===")
    main()
    print("Done.")
