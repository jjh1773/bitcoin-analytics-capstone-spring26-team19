"""Download Crypto Fear & Greed Index from Alternative.me API.

No API key required. Free public API.
Data available: 2018-02-01 ~ today

Saved files (data/externals/fear_greed/):
  fear_greed.csv  - Daily Fear & Greed index (0=Extreme Fear, 100=Extreme Greed)
                    → Fear composite signal (cubic transformation)

Run from project root:
    python data/externals/scripts/fear_greed.py
"""

from pathlib import Path
import requests
import pandas as pd

OUT     = Path(__file__).parent.parent / "fear_greed"   # data/externals/fear_greed/
API_URL = "https://api.alternative.me/fng/"
LIMIT   = 3000  # max history available (~2018-02-01 to today)


def download_fear_greed() -> pd.DataFrame:
    print("  Downloading Fear & Greed from alternative.me ...")
    resp = requests.get(API_URL, params={"limit": LIMIT, "format": "json"}, timeout=30)
    resp.raise_for_status()
    data = resp.json()["data"]

    rows = []
    for item in data:
        rows.append({
            "date":           pd.to_datetime(int(item["timestamp"]), unit="s").normalize(),
            "fear_greed":     int(item["value"]),
            "classification": item["value_classification"],
        })

    df = pd.DataFrame(rows).set_index("date").sort_index()
    df.index = df.index.tz_localize(None)
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        df = download_fear_greed()
        out_path = OUT / "fear_greed.csv"
        df.to_csv(out_path)
        print(f"  Saved: {out_path} ({len(df)} rows, "
              f"{df.index.min().date()} ~ {df.index.max().date()})")
        print(f"  Recent: score={df['fear_greed'].iloc[-1]} "
              f"({df['classification'].iloc[-1]})")
    except Exception as e:
        print(f"  ERROR: {e}")


if __name__ == "__main__":
    print("=== Fear & Greed Index Download ===")
    main()
    print("Done.")
