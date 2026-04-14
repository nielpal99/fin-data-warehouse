{{ config(materialized='table') }}

WITH rolling AS (
    SELECT * FROM {{ ref('fct_rolling_returns') }}
),

drawdown AS (
    SELECT date, ticker, drawdown, is_major_drawdown, is_in_recovery
    FROM {{ ref('fct_drawdown') }}
),

macro AS (
    SELECT * FROM {{ source('raw', 'STG_MACRO') }}
),

joined AS (
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
        r.relative_return_1m,
        d.drawdown,
        d.is_major_drawdown,
        d.is_in_recovery,
        m.fed_funds_rate,
        m.cpi,
        m.yield_curve_spread,
        m.unemployment_rate,
        LAG(m.fed_funds_rate, 63) OVER (PARTITION BY r.ticker ORDER BY r.date) AS fed_funds_rate_63d_ago
    FROM rolling r
    LEFT JOIN drawdown d USING (date, ticker)
    LEFT JOIN macro m USING (date)
)

SELECT
    *,

    CASE
        WHEN fed_funds_rate > fed_funds_rate_63d_ago THEN 'hiking'
        WHEN fed_funds_rate < fed_funds_rate_63d_ago THEN 'cutting'
        ELSE 'neutral'
    END AS rate_trend,

    CASE
        WHEN cpi > 4 THEN 'high'
        WHEN cpi > 2 THEN 'moderate'
        ELSE 'low'
    END AS inflation_regime,

    CASE
        WHEN yield_curve_spread < 0    THEN 'inverted'
        WHEN yield_curve_spread < 0.5  THEN 'flat'
        ELSE 'normal'
    END AS yield_curve_regime

FROM joined
