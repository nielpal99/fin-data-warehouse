{{ config(materialized='table') }}

WITH base AS (
    SELECT
        ticker,
        asset_class,
        date,
        daily_return,
        drawdown,
        rate_trend,
        inflation_regime,
        yield_curve_regime
    FROM {{ ref('fct_macro_overlay') }}
    WHERE daily_return IS NOT NULL
      AND rate_trend IS NOT NULL
      AND inflation_regime IS NOT NULL
      AND yield_curve_regime IS NOT NULL
),

grouped AS (
    SELECT
        ticker,
        asset_class,
        rate_trend,
        inflation_regime,
        yield_curve_regime,
        rate_trend || '_' || inflation_regime || '_' || yield_curve_regime  AS composite_regime,

        AVG(daily_return)                                                    AS avg_daily_return,
        AVG(daily_return) * 252                                              AS annualized_return,
        STDDEV(daily_return) * SQRT(252)                                     AS annualized_vol,
        STDDEV(CASE WHEN daily_return < 0 THEN daily_return END) * SQRT(252) AS downside_vol,
        MIN(drawdown)                                                         AS max_drawdown,
        COUNT(*)                                                              AS days_in_regime,
        SUM(CASE WHEN daily_return > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS positive_days_pct
    FROM base
    GROUP BY ticker, asset_class, rate_trend, inflation_regime, yield_curve_regime
),

with_ratios AS (
    SELECT
        *,
        (annualized_return - 0.05) / NULLIF(annualized_vol, 0)   AS sharpe,
        (annualized_return - 0.05) / NULLIF(downside_vol, 0)      AS sortino
    FROM grouped
)

SELECT
    ticker,
    asset_class,
    rate_trend,
    inflation_regime,
    yield_curve_regime,
    composite_regime,
    avg_daily_return,
    annualized_return,
    annualized_vol,
    sharpe,
    sortino,
    max_drawdown,
    days_in_regime,
    positive_days_pct,
    sharpe = MAX(sharpe) OVER (PARTITION BY ticker)  AS best_regime
FROM with_ratios
