{{ config(materialized='table') }}

WITH base_returns AS (
    SELECT ticker, asset_class, date, daily_return
    FROM {{ ref('fct_daily_returns') }}
),

base_drawdown AS (
    SELECT ticker, date, drawdown
    FROM {{ ref('fct_drawdown') }}
),

w_1y AS (
    SELECT r.ticker, r.asset_class, r.daily_return, d.drawdown, '1Y' AS period_label
    FROM base_returns r
    LEFT JOIN base_drawdown d USING (ticker, date)
    QUALIFY ROW_NUMBER() OVER (PARTITION BY r.ticker ORDER BY r.date DESC) <= 252
),

w_3y AS (
    SELECT r.ticker, r.asset_class, r.daily_return, d.drawdown, '3Y' AS period_label
    FROM base_returns r
    LEFT JOIN base_drawdown d USING (ticker, date)
    QUALIFY ROW_NUMBER() OVER (PARTITION BY r.ticker ORDER BY r.date DESC) <= 756
),

w_5y AS (
    SELECT r.ticker, r.asset_class, r.daily_return, d.drawdown, '5Y' AS period_label
    FROM base_returns r
    LEFT JOIN base_drawdown d USING (ticker, date)
    QUALIFY ROW_NUMBER() OVER (PARTITION BY r.ticker ORDER BY r.date DESC) <= 1260
),

w_all AS (
    SELECT r.ticker, r.asset_class, r.daily_return, d.drawdown, 'all' AS period_label
    FROM base_returns r
    LEFT JOIN base_drawdown d USING (ticker, date)
),

unioned AS (
    SELECT * FROM w_1y
    UNION ALL
    SELECT * FROM w_3y
    UNION ALL
    SELECT * FROM w_5y
    UNION ALL
    SELECT * FROM w_all
),

metrics AS (
    SELECT
        ticker,
        asset_class,
        period_label,
        AVG(daily_return) * 252                                                   AS annualized_return,
        STDDEV(daily_return) * SQRT(252)                                          AS annualized_vol,
        STDDEV(CASE WHEN daily_return < 0 THEN daily_return END) * SQRT(252)     AS downside_vol,
        MIN(drawdown)                                                              AS max_drawdown_1y,
        COUNT(*)                                                                   AS period_days,
        SUM(CASE WHEN daily_return > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)    AS win_rate
    FROM unioned
    GROUP BY ticker, asset_class, period_label
)

SELECT
    ticker,
    asset_class,
    period_label,
    annualized_return,
    annualized_vol,
    (annualized_return - 0.05) / NULLIF(annualized_vol, 0)  AS sharpe_ratio,
    (annualized_return - 0.05) / NULLIF(downside_vol, 0)    AS sortino_ratio,
    max_drawdown_1y,
    win_rate,
    period_days
FROM metrics
