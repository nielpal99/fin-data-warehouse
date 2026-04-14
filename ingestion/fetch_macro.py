import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from fredapi import Fred

load_dotenv()

SERIES = {
    "FEDFUNDS": "fed_funds_rate",
    "CPIAUCSL": "cpi",
    "T10Y2Y": "yield_curve_spread",
    "UNRATE": "unemployment_rate",
}

START = "2015-01-01"
END = pd.Timestamp.today().strftime("%Y-%m-%d")
OUTPUT = Path(__file__).parent.parent / "staging" / "raw_macro.parquet"


def fetch_macro() -> pd.DataFrame:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError("FRED_API_KEY not set. Copy .env.example to .env and fill in your key.")

    fred = Fred(api_key=api_key)
    frames = {}

    for series_id, col_name in SERIES.items():
        print(f"  Fetching {series_id} → {col_name}...")
        s = fred.get_series(series_id, observation_start=START, observation_end=END)
        frames[col_name] = s

    df = pd.DataFrame(frames)
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])

    # Forward-fill missing values (e.g. monthly series on daily date range)
    full_dates = pd.date_range(start=START, end=END, freq="D")
    df = df.set_index("date").reindex(full_dates).ffill().reset_index()
    df.rename(columns={"index": "date"}, inplace=True)
    df = df.dropna(how="all", subset=list(SERIES.values()))

    return df


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fetching FRED macro series ({START} to {END})...")
    df = fetch_macro()

    df.to_parquet(OUTPUT, index=False)

    print(f"\nOutput: {OUTPUT}")
    print(f"Total rows : {len(df):,}")
    print(f"\nLatest values per series:")
    latest = df.dropna().tail(1).squeeze()
    for col in SERIES.values():
        if col in df.columns:
            print(f"  {col:<25}: {latest[col]}")


if __name__ == "__main__":
    main()
