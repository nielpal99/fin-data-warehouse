{{ config(materialized='table') }}

WITH today AS (
    SELECT * FROM {{ ref('fct_macro_overlay') }}
    WHERE date = (SELECT MAX(date) FROM {{ ref('fct_macro_overlay') }})
),

spy_today AS (
    SELECT return_1m AS spy_30d
    FROM today
    WHERE ticker = 'SPY'
    LIMIT 1
),

btc_today AS (
    SELECT return_1m AS btc_30d
    FROM today
    WHERE ticker = 'BTC-USD'
    LIMIT 1
)

SELECT
    t.date,
    t.ticker,
    t.asset_class,

    SNOWFLAKE.CORTEX.COMPLETE(
        'llama3.1-70b',
        '2 sentence analytical portfolio note for ' || t.ticker || ' (' || t.asset_class || '). ' ||
        '30d return:' || ROUND(t.return_1m * 100, 2) || '%, ' ||
        '1y return:' || ROUND(t.return_1y * 100, 2) || '%, ' ||
        'drawdown:' || ROUND(t.drawdown * 100, 2) || '%, ' ||
        'vol:' || ROUND(t.volatility_30d * 100, 2) || '%, ' ||
        'vs SPY:' || ROUND(t.relative_return_1m * 100, 2) || '%. Concise, analytical, no disclaimers.'
    ) AS ticker_daily_note,

    SNOWFLAKE.CORTEX.COMPLETE(
        'claude-3-5-sonnet',
        '3 sentence macro regime classification for cross-asset portfolio. ' ||
        'rate_trend:' || t.rate_trend || ', ' ||
        'inflation:' || t.inflation_regime || ', ' ||
        'yield_curve:' || t.yield_curve_regime || ', ' ||
        'cpi:' || ROUND(t.cpi, 2) || ', ' ||
        'unemployment:' || ROUND(t.unemployment_rate, 2) || '%, ' ||
        'SPY 30d:' || ROUND(s.spy_30d, 2) || '%, ' ||
        'BTC 30d:' || ROUND(b.btc_30d, 2) || '%. Direct, analytical, no disclaimers.'
    ) AS macro_regime_summary

FROM today t
CROSS JOIN spy_today s
CROSS JOIN btc_today b
