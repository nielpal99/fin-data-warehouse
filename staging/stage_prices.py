import duckdb
from pathlib import Path

BASE = Path(__file__).parent
RAW_PRICES = BASE / "raw_prices.parquet"
RAW_MACRO = BASE / "raw_macro.parquet"
DB_PATH = BASE / "local.duckdb"


def validate_table(con: duckdb.DuckDBPyConnection, table: str) -> None:
    print(f"\n{'='*50}")
    print(f"TABLE: {table}")
    print(f"{'='*50}")

    row_count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"Row count : {row_count:,}")

    print("\nNull counts per column:")
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()]
    for col in cols:
        nulls = con.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} IS NULL").fetchone()[0]
        print(f"  {col:<30}: {nulls:,} nulls")

    if table == "stg_daily_prices":
        print("\nRows per ticker:")
        rows = con.execute(
            "SELECT ticker, COUNT(*) as cnt FROM stg_daily_prices GROUP BY ticker ORDER BY ticker"
        ).fetchall()
        for ticker, cnt in rows:
            print(f"  {ticker:<12}: {cnt:,}")

    if table == "stg_macro":
        print("\nLatest macro values:")
        row = con.execute(
            "SELECT * FROM stg_macro ORDER BY date DESC LIMIT 1"
        ).fetchdf().squeeze()
        print(row.to_string())

    print("\n10 sample rows:")
    sample = con.execute(f"SELECT * FROM {table} LIMIT 10").fetchdf()
    print(sample.to_string(index=False))


def main():
    if not RAW_PRICES.exists():
        raise FileNotFoundError(f"Missing {RAW_PRICES}. Run ingestion/fetch_prices.py first.")
    if not RAW_MACRO.exists():
        raise FileNotFoundError(f"Missing {RAW_MACRO}. Run ingestion/fetch_macro.py first.")

    if DB_PATH.exists():
        DB_PATH.unlink()

    con = duckdb.connect(str(DB_PATH))

    print("Creating stg_daily_prices...")
    con.execute(f"""
        CREATE TABLE stg_daily_prices AS
        SELECT
            CAST(date AS DATE)   AS date,
            ticker,
            asset_class,
            CAST(open  AS DOUBLE) AS open,
            CAST(high  AS DOUBLE) AS high,
            CAST(low   AS DOUBLE) AS low,
            CAST(close AS DOUBLE) AS close,
            CAST(volume AS BIGINT) AS volume,
            CAST("adj_close" AS DOUBLE) AS adj_close
        FROM read_parquet('{RAW_PRICES}')
        WHERE close IS NOT NULL
    """)

    print("Creating stg_macro...")
    con.execute(f"""
        CREATE TABLE stg_macro AS
        SELECT
            CAST(date AS DATE)              AS date,
            CAST(fed_funds_rate AS DOUBLE)  AS fed_funds_rate,
            CAST(cpi AS DOUBLE)             AS cpi,
            CAST(yield_curve_spread AS DOUBLE) AS yield_curve_spread,
            CAST(unemployment_rate AS DOUBLE)  AS unemployment_rate
        FROM read_parquet('{RAW_MACRO}')
    """)

    validate_table(con, "stg_daily_prices")
    validate_table(con, "stg_macro")

    con.close()
    print(f"\nDuckDB written to: {DB_PATH}")
    print("\nStep 4 complete. Await Snowflake credentials before proceeding.")


if __name__ == "__main__":
    main()
