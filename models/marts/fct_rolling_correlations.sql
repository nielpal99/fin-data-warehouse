{{ config(materialized='table') }}

-- Snowflake does not support CORR() with a sliding window frame or named
-- WINDOW clauses, so we derive rolling 90-day Pearson correlation from
-- first-principles with inlined windowed SUM/COUNT aggregates.
--
-- Pearson r = (n·Σxy − Σx·Σy) / sqrt((n·Σx² − (Σx)²)·(n·Σy² − (Σy)²))

WITH base AS (
    SELECT ticker, date, daily_return
    FROM {{ ref('fct_daily_returns') }}
),

pairs AS (
    SELECT
        a.date,
        a.ticker        AS ticker_a,
        b.ticker        AS ticker_b,
        a.daily_return  AS return_a,
        b.daily_return  AS return_b
    FROM base a
    JOIN base b ON a.date = b.date
    WHERE (a.ticker, b.ticker) IN (
        ('BTC-USD', 'QQQ'),
        ('BTC-USD', 'GLD'),
        ('BTC-USD', 'ETH-USD'),
        ('BTC-USD', 'VTI'),
        ('QQQ',     'GLD'),
        ('QQQ',     'TLT'),
        ('GLD',     'TLT'),
        ('ETH-USD', 'QQQ')
    )
),

windowed AS (
    SELECT
        date,
        ticker_a,
        ticker_b,
        COUNT(*) OVER (
            PARTITION BY ticker_a, ticker_b ORDER BY date
            ROWS BETWEEN 89 PRECEDING AND CURRENT ROW
        ) AS n,
        SUM(return_a) OVER (
            PARTITION BY ticker_a, ticker_b ORDER BY date
            ROWS BETWEEN 89 PRECEDING AND CURRENT ROW
        ) AS sum_a,
        SUM(return_b) OVER (
            PARTITION BY ticker_a, ticker_b ORDER BY date
            ROWS BETWEEN 89 PRECEDING AND CURRENT ROW
        ) AS sum_b,
        SUM(return_a * return_b) OVER (
            PARTITION BY ticker_a, ticker_b ORDER BY date
            ROWS BETWEEN 89 PRECEDING AND CURRENT ROW
        ) AS sum_ab,
        SUM(return_a * return_a) OVER (
            PARTITION BY ticker_a, ticker_b ORDER BY date
            ROWS BETWEEN 89 PRECEDING AND CURRENT ROW
        ) AS sum_a2,
        SUM(return_b * return_b) OVER (
            PARTITION BY ticker_a, ticker_b ORDER BY date
            ROWS BETWEEN 89 PRECEDING AND CURRENT ROW
        ) AS sum_b2
    FROM pairs
)

SELECT
    date,
    ticker_a,
    ticker_b,
    (n * sum_ab - sum_a * sum_b)
        / NULLIF(
            SQRT(
                NULLIF((n * sum_a2 - sum_a * sum_a), 0) *
                NULLIF((n * sum_b2 - sum_b * sum_b), 0)
            ), 0
          ) AS correlation_90d
FROM windowed
WHERE n >= 2
  AND (
      ticker_a != 'BTC-USD' OR ticker_b != 'ETH-USD'
      OR date IN (SELECT date FROM {{ ref('fct_daily_returns') }} WHERE ticker = 'QQQ')
  )
