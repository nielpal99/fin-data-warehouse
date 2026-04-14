{{ config(materialized='table') }}

WITH base AS (
    SELECT
        date,
        ticker,
        asset_class,
        adjusted_close,
        daily_return
    FROM {{ ref('fct_daily_returns') }}
),

rolling AS (
    SELECT
        date,
        ticker,
        asset_class,
        adjusted_close,
        daily_return,

        -- Rolling returns
        (adjusted_close / NULLIF(LAG(adjusted_close, 21)  OVER (PARTITION BY ticker ORDER BY date), 0)) - 1 AS return_1m,
        (adjusted_close / NULLIF(LAG(adjusted_close, 63)  OVER (PARTITION BY ticker ORDER BY date), 0)) - 1 AS return_3m,
        (adjusted_close / NULLIF(LAG(adjusted_close, 252) OVER (PARTITION BY ticker ORDER BY date), 0)) - 1 AS return_1y,
        (adjusted_close / NULLIF(LAG(adjusted_close, 756) OVER (PARTITION BY ticker ORDER BY date), 0)) - 1 AS return_3y,

        -- 30-day rolling volatility
        STDDEV(daily_return) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS volatility_30d

    FROM base
),

spy AS (
    SELECT date, return_1m AS spy_return_1m
    FROM rolling
    WHERE ticker = 'SPY'
)

SELECT
    r.date,
    r.ticker,
    r.asset_class,
    r.adjusted_close,
    r.daily_return,
    r.return_1m,
    r.return_3m,
    r.return_1y,
    r.return_3y,
    r.volatility_30d,
    r.return_1m - s.spy_return_1m AS relative_return_1m
FROM rolling r
LEFT JOIN spy s USING (date)
