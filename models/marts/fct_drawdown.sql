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

with_drawdown AS (
    SELECT
        date,
        ticker,
        asset_class,
        adjusted_close,
        daily_return,
        MAX(adjusted_close) OVER (
            PARTITION BY ticker ORDER BY date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS running_max
    FROM base
)

SELECT
    date,
    ticker,
    asset_class,
    adjusted_close,
    daily_return,
    running_max,
    (adjusted_close - running_max) / NULLIF(running_max, 0)          AS drawdown,
    (adjusted_close - running_max) / NULLIF(running_max, 0) < -0.10  AS is_major_drawdown,
    (adjusted_close - running_max) / NULLIF(running_max, 0)
        BETWEEN -0.05 AND 0                                           AS is_in_recovery
FROM with_drawdown
