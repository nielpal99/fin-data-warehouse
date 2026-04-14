{{ config(materialized='table') }}

WITH last_252 AS (
    SELECT
        ticker,
        date,
        daily_return
    FROM {{ ref('fct_daily_returns') }}
    QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) <= 252
)

SELECT
    a.ticker                               AS ticker_a,
    b.ticker                               AS ticker_b,
    CORR(a.daily_return, b.daily_return)   AS correlation,
    252                                    AS period_days
FROM last_252 a
JOIN last_252 b ON a.date = b.date
GROUP BY a.ticker, b.ticker
