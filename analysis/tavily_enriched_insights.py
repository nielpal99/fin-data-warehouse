import os
import warnings
import datetime
import textwrap
import snowflake.connector
from pathlib import Path
from dotenv import load_dotenv
from tavily import TavilyClient

warnings.filterwarnings("ignore")
load_dotenv(Path(__file__).parent.parent / ".env")

TAVILY_KEY = os.environ.get("TAVILY_API_KEY")
if not TAVILY_KEY:
    raise EnvironmentError("TAVILY_API_KEY not set in .env")

TODAY = datetime.date.today()
SNIPPET_MAX = 400   # chars per Tavily snippet kept in the prompt


# ── helpers ────────────────────────────────────────────────────────────────────

def get_conn():
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        role=os.environ["SNOWFLAKE_ROLE"],
        database="FIN_DATA_WAREHOUSE",
        schema="RAW",
    )


def escape(s: str) -> str:
    """Escape single quotes and backslashes for inline Snowflake SQL strings."""
    return str(s).replace("\\", "\\\\").replace("'", "\\'")


def cortex_complete(cur, model: str, prompt: str) -> str:
    prompt_esc = escape(prompt)
    cur.execute(f"SELECT SNOWFLAKE.CORTEX.COMPLETE('{model}', '{prompt_esc}')")
    row = cur.fetchone()
    return (row[0] or "").strip() if row else ""


def tavily_snippets(client: TavilyClient, query: str) -> str:
    """Return top-3 content snippets joined as a single string."""
    try:
        resp = client.search(query, search_depth="basic", max_results=3)
        snippets = [
            r.get("content", "")[:SNIPPET_MAX]
            for r in resp.get("results", [])
            if r.get("content")
        ]
        return " | ".join(snippets) if snippets else "No recent news found."
    except Exception as e:
        return f"Search unavailable: {e}"


def section(title: str):
    print(f"\n{'='*65}")
    print(f" {title}")
    print(f"{'='*65}")


# ── STEP 1: pull data from Snowflake ─────────────────────────────────────────

section("STEP 1 — Loading data from Snowflake")

con = get_conn()
cur = con.cursor()

# Anomalies on the latest trading date
cur.execute("""
    SELECT ticker, asset_class, date, daily_return, adjusted_close
    FROM FIN_DATA_WAREHOUSE.RAW_MARTS.fct_daily_returns
    WHERE is_anomaly = TRUE
      AND date = (SELECT MAX(date) FROM FIN_DATA_WAREHOUSE.RAW_MARTS.fct_daily_returns)
""")
anomaly_rows = cur.fetchall()
anomaly_date = anomaly_rows[0][2] if anomaly_rows else None
print(f"  Anomalies today : {len(anomaly_rows)} row(s)  (date={anomaly_date})")

# Active major drawdowns on latest date
cur.execute("""
    SELECT ticker, asset_class, ROUND(drawdown * 100, 2) AS dd_pct
    FROM FIN_DATA_WAREHOUSE.RAW_MARTS.fct_drawdown
    WHERE is_major_drawdown = TRUE
      AND date = (SELECT MAX(date) FROM FIN_DATA_WAREHOUSE.RAW_MARTS.fct_drawdown)
""")
drawdown_rows = cur.fetchall()
print(f"  Major drawdowns : {len(drawdown_rows)} ticker(s)")

# Current macro regime
cur.execute("""
    SELECT DISTINCT rate_trend, inflation_regime, yield_curve_regime,
        rate_trend || '_' || inflation_regime || '_' || yield_curve_regime AS composite_regime,
        cpi, fed_funds_rate, yield_curve_spread
    FROM FIN_DATA_WAREHOUSE.RAW_MARTS.fct_macro_overlay
    WHERE date = (SELECT MAX(date) FROM FIN_DATA_WAREHOUSE.RAW_MARTS.fct_macro_overlay)
    QUALIFY ROW_NUMBER() OVER (ORDER BY date) = 1
""")
macro_row = cur.fetchone()
rate_trend, inflation_regime, yield_curve_regime, composite_regime, \
    cpi, fed_funds_rate, yield_curve_spread = macro_row
print(f"  Macro regime    : {composite_regime}  "
      f"(CPI={cpi:.2f}, FFR={fed_funds_rate:.2f}, spread={yield_curve_spread:.2f})")


# ── STEP 2: Tavily searches ───────────────────────────────────────────────────

section("STEP 2 — Tavily searches")

tavily = TavilyClient(api_key=TAVILY_KEY)
context_map: dict[str, str] = {}

for ticker, asset_class, date, daily_return, adj_close in anomaly_rows:
    query = f"{ticker} price movement today {asset_class.lower()} market"
    print(f"  Searching: {query[:60]}")
    context_map[ticker] = tavily_snippets(tavily, query)

for ticker, asset_class, dd_pct in drawdown_rows:
    if ticker not in context_map:
        query = f"{ticker} drawdown market news latest"
        print(f"  Searching: {query[:60]}")
        context_map[ticker] = tavily_snippets(tavily, query)

macro_query = "Fed reserve rate decision CPI inflation latest macro outlook"
print(f"  Searching: {macro_query[:60]}")
context_map["MACRO"] = tavily_snippets(tavily, macro_query)

print(f"  Searches completed: {len(context_map)}")


# ── STEP 3: Cortex enriched calls ────────────────────────────────────────────

section("STEP 3 — Cortex COMPLETE calls")

# Switch to RAW_MARTS schema for Cortex
cur.execute("USE SCHEMA FIN_DATA_WAREHOUSE.RAW_MARTS")

insights: list[dict] = []

# Anomaly notes
for ticker, asset_class, date, daily_return, adj_close in anomaly_rows:
    snippets = context_map.get(ticker, "No context available.")
    prompt = (
        f"2 sentence explanation for anomalous move in {ticker} ({asset_class}). "
        f"Daily return: {daily_return:.2%}. "
        f"News context: {snippets[:800]}. "
        f"Be direct, cite the news if relevant, no disclaimers."
    )
    print(f"  Cortex anomaly note: {ticker} ({daily_return:+.2%})")
    note = cortex_complete(cur, "llama3.1-70b", prompt)
    insights.append({
        "ticker":        ticker,
        "insight_type":  "anomaly",
        "tavily_context": snippets[:1000],
        "enriched_note": note,
    })

# Drawdown notes
for ticker, asset_class, dd_pct in drawdown_rows:
    snippets = context_map.get(ticker, "No context available.")
    prompt = (
        f"2 sentence drawdown analysis for {ticker}. "
        f"Current drawdown: {dd_pct}%. "
        f"News context: {snippets[:800]}. "
        f"Explain whether macro-driven, asset-specific, or technical. No disclaimers."
    )
    print(f"  Cortex drawdown note: {ticker} ({dd_pct}%)")
    note = cortex_complete(cur, "llama3.1-70b", prompt)
    insights.append({
        "ticker":        ticker,
        "insight_type":  "drawdown",
        "tavily_context": snippets[:1000],
        "enriched_note": note,
    })

# Macro regime summary
macro_snippets = context_map.get("MACRO", "No context available.")
macro_prompt = (
    f"3 sentence macro regime analysis grounded in both quantitative data and current news. "
    f"Rate trend: {rate_trend}, inflation: {inflation_regime}, "
    f"yield curve: {yield_curve_regime}, "
    f"CPI: {cpi:.2f}, Fed funds: {fed_funds_rate:.2f}, spread: {yield_curve_spread:.2f}. "
    f"News context: {macro_snippets[:800]}. "
    f"Be direct and analytical. No disclaimers."
)
print(f"  Cortex macro regime summary")
macro_note = cortex_complete(cur, "llama3.1-70b", macro_prompt)
insights.append({
    "ticker":        "ALL",
    "insight_type":  "macro",
    "tavily_context": macro_snippets[:1000],
    "enriched_note": macro_note,
})

print(f"  Generated {len(insights)} enriched insight(s)")


# ── STEP 4: write back to Snowflake ──────────────────────────────────────────

section("STEP 4 — Writing results to Snowflake")

cur.execute("USE SCHEMA FIN_DATA_WAREHOUSE.RAW")

cur.execute("""
    CREATE TABLE IF NOT EXISTS CORTEX_ENRICHED_INSIGHTS (
        run_date        DATE,
        ticker          VARCHAR,
        insight_type    VARCHAR,
        tavily_context  VARCHAR,
        enriched_note   VARCHAR
    )
""")

# Remove today's rows before re-inserting (idempotent re-runs)
cur.execute("DELETE FROM CORTEX_ENRICHED_INSIGHTS WHERE run_date = CURRENT_DATE")
deleted = cur.rowcount
print(f"  Deleted {deleted} existing row(s) for today")

for row in insights:
    cur.execute(
        """INSERT INTO CORTEX_ENRICHED_INSIGHTS
           (run_date, ticker, insight_type, tavily_context, enriched_note)
           VALUES (CURRENT_DATE, %s, %s, %s, %s)""",
        (row["ticker"], row["insight_type"],
         row["tavily_context"], row["enriched_note"])
    )

con.commit()
con.close()
print(f"  Inserted {len(insights)} row(s) into CORTEX_ENRICHED_INSIGHTS")


# ── STEP 5: print sample output ───────────────────────────────────────────────

section("STEP 5 — Sample enriched insights")

for row in insights:
    print(f"\n  [{row['insight_type'].upper()}] {row['ticker']}")
    for line in textwrap.wrap(row["enriched_note"], width=70):
        print(f"    {line}")

print(f"\n  run_date : {TODAY}")
print(f"  rows     : {len(insights)}")
print("\n  Done. Run: dbt run --select fct_cortex_enriched_insights")
