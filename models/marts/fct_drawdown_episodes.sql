{{ config(materialized='table') }}

WITH base AS (
    SELECT ticker, asset_class, date, drawdown, is_major_drawdown
    FROM {{ ref('fct_drawdown') }}
    WHERE drawdown IS NOT NULL
),

-- Detect transitions into major drawdown territory
with_lag AS (
    SELECT *,
        LAG(is_major_drawdown) OVER (PARTITION BY ticker ORDER BY date) AS prev_major
    FROM base
),

-- Flag each entry into a new episode
episode_flags AS (
    SELECT *,
        CASE
            WHEN is_major_drawdown = TRUE
             AND (prev_major = FALSE OR prev_major IS NULL)
            THEN 1 ELSE 0
        END AS is_episode_start
    FROM with_lag
),

-- Assign episode IDs — only carries through rows inside major drawdown
major_dd_rows AS (
    SELECT *,
        SUM(is_episode_start) OVER (PARTITION BY ticker ORDER BY date
                                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                                   ) AS episode_id
    FROM episode_flags
    WHERE is_major_drawdown = TRUE
),

-- Episode-level aggregates: start date + trough depth
episode_stats AS (
    SELECT
        ticker,
        asset_class,
        episode_id,
        MIN(date)     AS episode_start,
        MIN(drawdown) AS episode_trough_depth
    FROM major_dd_rows
    GROUP BY ticker, asset_class, episode_id
),

-- Trough date: first date of the minimum drawdown within each episode
trough_dates AS (
    SELECT m.ticker, m.episode_id, m.date AS episode_trough_date
    FROM major_dd_rows   m
    JOIN episode_stats   s
      ON m.ticker = s.ticker AND m.episode_id = s.episode_id
     AND m.drawdown = s.episode_trough_depth
    QUALIFY ROW_NUMBER() OVER (PARTITION BY m.ticker, m.episode_id ORDER BY m.date) = 1
),

-- Combine into one episode row per ticker+episode
episodes AS (
    SELECT
        s.ticker,
        s.asset_class,
        s.episode_id,
        s.episode_start,
        t.episode_trough_date,
        s.episode_trough_depth
    FROM episode_stats s
    JOIN trough_dates  t ON s.ticker = t.ticker AND s.episode_id = t.episode_id
),

-- Recovery: first date after the trough where drawdown recovers to >= -0.01
recovery AS (
    SELECT
        e.ticker,
        e.episode_id,
        MIN(b.date) AS episode_end
    FROM episodes e
    JOIN base     b ON  b.ticker = e.ticker
                    AND b.date   > e.episode_trough_date
                    AND b.drawdown >= -0.01
    GROUP BY e.ticker, e.episode_id
)

SELECT
    e.ticker,
    e.asset_class,
    e.episode_start,
    e.episode_trough_date,
    e.episode_trough_depth,
    r.episode_end,
    DATEDIFF('day', e.episode_start,        e.episode_trough_date) AS days_to_trough,
    DATEDIFF('day', e.episode_trough_date,  r.episode_end)         AS recovery_days,
    DATEDIFF('day', e.episode_start,        r.episode_end)         AS total_episode_days,
    r.episode_end IS NULL                                           AS still_in_drawdown
FROM episodes  e
LEFT JOIN recovery r ON e.ticker = r.ticker AND e.episode_id = r.episode_id
