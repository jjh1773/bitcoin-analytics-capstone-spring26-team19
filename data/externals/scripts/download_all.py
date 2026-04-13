"""Master download script — refreshes all external data used in the model.

Run from project root:
    python data/externals/scripts/download_all.py

Data saved to:
    data/externals/
        fred/
            dff.csv        - Fed Funds Rate (daily)           → monetary_policy_signal
            m2sl.csv       - M2 Money Supply (monthly→daily)  → monetary_policy_signal
        fear_greed/
            fear_greed.csv - Crypto Fear & Greed Index        → fear_composite_signal
        market/
            vix.csv        - VIX volatility index             → fear_composite_signal
            dxy.csv        - US Dollar Index proxy (UUP)      → monetary_policy_signal
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
PYTHON = sys.executable

SCRIPTS = [
    (SCRIPTS_DIR / "fred.py",        "FRED (Fed Funds + M2)"),
    (SCRIPTS_DIR / "fear_greed.py",  "Fear & Greed Index"),
    (SCRIPTS_DIR / "market.py",      "Market (VIX + DXY)"),
]

if __name__ == "__main__":
    print("=" * 55)
    print("External Data Download — All Sources")
    print("=" * 55)
    for script, label in SCRIPTS:
        print(f"\n[{label}]")
        result = subprocess.run([PYTHON, str(script)], capture_output=False)
        if result.returncode != 0:
            print(f"  WARNING: {script.name} exited with code {result.returncode}")
    print("\n" + "=" * 55)
    print("All downloads complete.")
    print("=" * 55)
