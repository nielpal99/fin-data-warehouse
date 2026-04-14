import os
import pandas as pd
import snowflake.connector
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE = Path(__file__).parent.parent / "staging"
RAW_PRICES = BASE / "raw_prices.parquet"
RAW_MACRO = BASE / "raw_macro.parquet"

DB = "FIN_DATA_WAREHOUSE"
SCHEMA = "RAW"


def get_conn(database=None, schema=None):
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        role=os.environ["SNOWFLAKE_ROLE"],
        database=database,
        schema=schema,
    )


def setup_database():
    print(f"Connecting to Snowflake...")
    con = get_conn()
    cur = con.cursor()

    print(f"Creating database {DB}...")
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {DB}")
    cur.execute(f"USE DATABASE {DB}")

    print(f"Creating schema {DB}.{SCHEMA}...")
    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    cur.execute(f"USE SCHEMA {SCHEMA}")

    con.close()
    print("Database and schema ready.")


def upload_table(df: pd.DataFrame, table: str):
    from snowflake.connector.pandas_tools import write_pandas

    print(f"Uploading {len(df):,} rows to {DB}.{SCHEMA}.{table}...")
    df.columns = [c.upper() for c in df.columns]

    # Snowflake connector needs string dates
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%d")

    con = get_conn(database=DB, schema=SCHEMA)
    success, nchunks, nrows, _ = write_pandas(
        con, df, table, auto_create_table=True, overwrite=True
    )
    con.close()

    if success:
        print(f"  Uploaded {nrows:,} rows in {nchunks} chunk(s).")
    else:
        raise RuntimeError(f"Upload failed for {table}")


def main():
    setup_database()

    print("\nLoading raw_prices.parquet...")
    prices = pd.read_parquet(RAW_PRICES)
    upload_table(prices, "RAW_DAILY_PRICES")

    print("\nLoading raw_macro.parquet...")
    macro = pd.read_parquet(RAW_MACRO)
    upload_table(macro, "STG_MACRO")

    print("\nSnowflake setup complete.")
    print(f"  Database : {DB}")
    print(f"  Schema   : {SCHEMA}")
    print(f"  Tables   : RAW_DAILY_PRICES, STG_MACRO")
    print("\nReady to run: dbt run --profiles-dir profiles/")


if __name__ == "__main__":
    main()
