import yfinance as yf
import pandas as pd
from pathlib import Path

TICKERS = {
    "VTI": "Equity",
    "VOO": "Equity",
    "QQQ": "Equity",
    "SPY": "Equity",
    "IWM": "Equity",
    "BND": "Fixed Income",
    "TLT": "Fixed Income",
    "GLD": "Commodity",
    "BTC-USD": "Crypto",
    "ETH-USD": "Crypto",
    "DX-Y.NYB": "Macro",
}

START = "2015-01-01"
END = pd.Timestamp.today().strftime("%Y-%m-%d")
OUTPUT = Path(__file__).parent.parent / "staging" / "raw_prices.parquet"


def fetch_prices() -> pd.DataFrame:
    frames = []
    for ticker, asset_class in TICKERS.items():
        print(f"  Fetching {ticker} ({asset_class})...")
        df = yf.download(ticker, start=START, end=END, auto_adjust=False, progress=False)
        if df.empty:
            print(f"  WARNING: no data for {ticker}")
            continue
        df = df.reset_index()
        df.columns = [c[0].lower().replace(" ", "_") if isinstance(c, tuple) else c.lower().replace(" ", "_") for c in df.columns]
        # yfinance multi-level columns when downloading single ticker
        if "adj_close" not in df.columns and "adj close" in df.columns:
            df.rename(columns={"adj close": "adj_close"}, inplace=True)
        df["ticker"] = ticker
        df["asset_class"] = asset_class
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    # Normalize column names
    combined.columns = [c.lower().replace(" ", "_") for c in combined.columns]
    return combined


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fetching prices for {len(TICKERS)} tickers ({START} to {END})...")
    df = fetch_prices()

    df.to_parquet(OUTPUT, index=False)

    print(f"\nOutput: {OUTPUT}")
    print(f"Total rows    : {len(df):,}")
    print(f"Date range    : {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"\nRows per ticker:")
    print(df.groupby("ticker")["date"].count().to_string())


if __name__ == "__main__":
    main()
