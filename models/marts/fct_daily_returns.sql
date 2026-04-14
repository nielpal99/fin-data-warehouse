{{ config(materialized='table') }}

WITH base AS (
    SELECT
        p.date,
        p.ticker,
        t.asset_class,
        p.adjusted_close,
        LAG(p.adjusted_close) OVER (
            PARTITION BY p.ticker ORDER BY p.date
        ) AS prev_adjusted_close
    FROM {{ ref('stg_daily_prices') }} p
    LEFT JOIN {{ ref('stg_tickers') }} t USING (ticker)
),

with_returns AS (
    SELECT
        date,
        ticker,
        asset_class,
        adjusted_close,
        CASE
            WHEN prev_adjusted_close IS NOT NULL AND prev_adjusted_close != 0
            THEN (adjusted_close - prev_adjusted_close) / prev_adjusted_close
        END AS daily_return
    FROM base
),

with_stddev AS (
    SELECT
        *,
        STDDEV(daily_return) OVER (PARTITION BY ticker) AS stddev_return
    FROM with_returns
)

SELECT
    date,
    ticker,
    asset_class,
    adjusted_close,
    daily_return,
    ABS(daily_return) > 2 * stddev_return AS is_anomaly
FROM with_stddev
WHERE date IS NOT NULL
  AND daily_return IS NOT NULL
