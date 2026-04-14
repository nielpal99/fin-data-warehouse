{{ config(materialized='view') }}

SELECT
    CAST(date AS DATE)          AS date,
    ticker,
    asset_class,
    CAST(open AS FLOAT)         AS open,
    CAST(high AS FLOAT)         AS high,
    CAST(low AS FLOAT)          AS low,
    CAST(close AS FLOAT)        AS close,
    CAST(volume AS BIGINT)      AS volume,
    CAST(adj_close AS FLOAT)    AS adjusted_close
FROM {{ source('raw', 'RAW_DAILY_PRICES') }}
WHERE close IS NOT NULL
