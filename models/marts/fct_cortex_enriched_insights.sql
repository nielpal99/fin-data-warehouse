{{ config(materialized='table') }}

SELECT *
FROM {{ source('raw', 'CORTEX_ENRICHED_INSIGHTS') }}
WHERE run_date = (
    SELECT MAX(run_date)
    FROM {{ source('raw', 'CORTEX_ENRICHED_INSIGHTS') }}
)
