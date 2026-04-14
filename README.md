# fin-data-warehouse

A full-stack financial data warehouse built with Python, Snowflake, dbt, and Snowflake Cortex — covering ingestion, transformation, portfolio analytics, and LLM-enriched market insights across 11 cross-asset tickers from 2015 to present.

## Stack

| Layer | Tool |
|-------|------|
| Price ingestion | yfinance |
| Macro ingestion | FRED API |
| Local staging / validation | DuckDB |
| Cloud warehouse | Snowflake |
| SQL transformation | dbt-snowflake |
| LLM insights | Snowflake Cortex (`llama3.1-70b`) |
| Live news enrichment | Tavily Search API |
| Quantitative analysis | Python (numpy, scipy, pandas, matplotlib) |

## Asset universe

| Ticker | Name | Class |
|--------|------|-------|
| VTI | Vanguard Total Stock Market ETF | Equity |
| VOO | Vanguard S&P 500 ETF | Equity |
| QQQ | Invesco Nasdaq-100 ETF | Equity |
| SPY | SPDR S&P 500 ETF | Equity |
| IWM | iShares Russell 2000 ETF | Equity |
| BND | Vanguard Total Bond Market ETF | Fixed Income |
| TLT | iShares 20+ Year Treasury ETF | Fixed Income |
| GLD | SPDR Gold Shares | Commodity |
| BTC-USD | Bitcoin | Crypto |
| ETH-USD | Ethereum | Crypto |
| DX-Y.NYB | US Dollar Index | Macro |

---

## Project structure

```
fin-data-warehouse/
├── ingestion/
│   ├── fetch_prices.py          # yfinance → Snowflake RAW_DAILY_PRICES
│   └── fetch_macro.py           # FRED API → Snowflake STG_MACRO
├── staging/
│   └── stage_prices.py          # local DuckDB validation before Snowflake load
├── models/
│   ├── sources.yml              # dbt source definitions (RAW schema)
│   ├── staging/
│   │   ├── stg_daily_prices.sql
│   │   ├── stg_tickers.sql
│   │   └── schema.yml
│   └── marts/
│       ├── fct_daily_returns.sql
│       ├── fct_rolling_returns.sql
│       ├── fct_drawdown.sql
│       ├── fct_drawdown_episodes.sql
│       ├── fct_rolling_correlations.sql
│       ├── fct_correlation_matrix.sql
│       ├── fct_macro_overlay.sql
│       ├── fct_portfolio_analytics.sql
│       ├── fct_regime_performance.sql
│       ├── fct_cortex_insights.sql
│       ├── fct_cortex_enriched_insights.sql
│       └── schema.yml
├── analysis/
│   ├── portfolio_optimizer.py        # efficient frontier + constrained max Sharpe
│   ├── dca_simulator.py              # DCA vs lump sum vs HYSA comparison
│   ├── correlation_lead_lag.py       # BTC/QQQ correlation as leading/lagging indicator
│   ├── regime_entry_backtest.py      # 1Y forward returns by macro regime entry
│   └── tavily_enriched_insights.py  # Tavily + Cortex daily enrichment pipeline
├── scripts/
│   └── setup_snowflake.py
├── profiles/
│   └── profiles.yml
├── dbt_project.yml
├── requirements.txt
└── .env.example
```

---

## dbt model DAG

```
RAW_DAILY_PRICES ──► stg_daily_prices ──► fct_daily_returns ──► fct_rolling_returns
                                                             ├──► fct_macro_overlay ──► fct_cortex_insights
                                                             ├──► fct_drawdown ──► fct_drawdown_episodes
                                                             ├──► fct_rolling_correlations
                                                             ├──► fct_correlation_matrix
                                                             ├──► fct_portfolio_analytics
                                                             └──► fct_regime_performance
STG_MACRO ────────────────────────────► stg_tickers
CORTEX_ENRICHED_INSIGHTS (Python) ───► fct_cortex_enriched_insights
```

**13 mart models · 53 schema tests · staging=view, marts=table**

---

## Mart models

### `fct_daily_returns`
Daily adjusted close, daily return, 252-day rolling stddev, and anomaly flag (`|return| > 2σ`). 11 tickers × ~2,800 trading days.

### `fct_rolling_returns`
Cumulative returns over 1M, 3M, 6M, 1Y, 3Y, 5Y windows per ticker.

### `fct_drawdown`
Rolling drawdown from all-time high. `is_major_drawdown = drawdown < -10%`.

### `fct_drawdown_episodes`
382 discrete drawdown episodes with start date, trough depth, recovery date, days to recovery, and `still_in_drawdown` flag. BTC has the fastest average recovery (96 days); BND the slowest (837 days).

### `fct_rolling_correlations`
90-day rolling Pearson correlation across 8 pairs: BTC/QQQ, BTC/GLD, BTC/ETH, BTC/VTI, QQQ/GLD, QQQ/TLT, GLD/TLT, ETH/QQQ. Implemented as a manual Pearson formula — Snowflake does not support `CORR()` in sliding window frames.

### `fct_correlation_matrix`
Full pairwise correlation matrix across all tickers at 90d, 180d, 1Y, and all-time lookback periods.

### `fct_macro_overlay`
Fed funds rate, CPI, and 10Y-2Y yield curve spread joined to daily prices. Classifies each day as `rate_trend` (hiking/cutting/neutral), `inflation_regime` (high/moderate/low), and `yield_curve_regime` (inverted/flat/normal).

### `fct_portfolio_analytics`
Annualized return, volatility, Sharpe ratio, max drawdown, and Calmar ratio over 1Y, 3Y, 5Y, and full-history windows. 44 rows (11 tickers × 4 windows).

### `fct_regime_performance`
Per-ticker Sharpe ratio, mean return, and volatility broken down by composite macro regime. 99 rows. Flags `best_regime` per ticker.

### `fct_cortex_insights`
Snowflake Cortex (`llama3.1-70b`) generates a daily narrative note per ticker and a macro regime summary from quantitative inputs.

### `fct_cortex_enriched_insights`
Cortex insights enriched with live Tavily news snippets — anomaly explanations, drawdown analysis, and macro commentary grounded in current events.

---

## Analysis scripts

### `portfolio_optimizer.py`
10,000 Monte Carlo portfolios + constrained SLSQP optimization to find the max Sharpe allocation subject to position limits (BTC ≤ 25%, GLD ≤ 25%, any single equity ≤ 30%, VTI ≥ 10%).

**Result:** QQQ 30% · GLD 25% · BTC 25% · VTI 10% · VOO 10% — Sharpe **0.86**

Output: `analysis/efficient_frontier_v2.png`

### `dca_simulator.py`
Compares three strategies over 1Y/3Y/5Y horizons using 10,000 Monte Carlo simulations drawn from historical return distributions:

| Strategy | 1Y Median | 3Y Median | 5Y Median |
|----------|-----------|-----------|-----------|
| DCA + HYSA ($500/mo, 4.5% on cash) | $6,875 | $26,821 | $58,954 |
| Lump Sum (full capital day 1) | $7,620 | $36,419 | $99,451 |
| HYSA Only (4.5%, no investment) | $6,270 | $20,541 | $37,385 |

Lump sum dominates above a ~3% annual portfolio return threshold — well below this portfolio's blended ~30%. HYSA meaningfully closes the gap but doesn't flip the result.

Output: `analysis/dca_hysa_comparison.png`

### `correlation_lead_lag.py`
Cross-correlates the BTC/QQQ 90-day rolling correlation against the SPY drawdown at lags −30 to +30 trading days.

**Result:** BTC/QQQ correlation is a **lagging indicator** — it rises ~5 trading days *after* SPY drawdowns deepen (r = −0.509, p < 0.0001). The sign is negative: deeper drawdowns pull crypto closer to equities (risk-off contagion), not the reverse.

Output: `analysis/correlation_lead_lag.png`

### `regime_entry_backtest.py`
Tags every historical date with its macro regime at entry, then measures the 1-year forward return. Aggregates median return, hit rate, and Sharpe by ticker × regime (min. 30 observations).

**Current regime: `neutral_high_normal`**

| Ticker | Rank | Median 1Y Fwd Return | Hit Rate |
|--------|------|----------------------|----------|
| BTC-USD | **#1 of 9** | +9.7% | 68% |
| VTI | #3 of 9 | +2.7% | 79% |
| QQQ | #5 of 9 | +2.5% | 72% |
| GLD | #7 of 9 | -0.0% | 50% |
| TLT | **#7 of 9** | -1.7% | 36% |

Output: `analysis/regime_entry_backtest.png`

### `tavily_enriched_insights.py`
Daily pipeline: pulls anomaly and drawdown flags from Snowflake, searches Tavily for current news per ticker, and passes quantitative context + snippets to Snowflake Cortex for 2-sentence LLM explanations. Writes back to `CORTEX_ENRICHED_INSIGHTS` (idempotent — delete + re-insert for today's date).

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/fin-data-warehouse.git
cd fin-data-warehouse
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
# Fill in: SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD,
#          SNOWFLAKE_WAREHOUSE, SNOWFLAKE_ROLE, FRED_API_KEY, TAVILY_API_KEY
```

The Snowflake account identifier uses the format `orgname-accountname`. Find yours at `https://app.snowflake.com/<org>/<account>/`.

### 3. Ingest data

```bash
python ingestion/fetch_prices.py   # equity, crypto, macro prices via yfinance
python ingestion/fetch_macro.py    # FRED macro series (CPI, Fed funds, yield curve)
python staging/stage_prices.py     # validate locally in DuckDB before upload
```

### 4. Run dbt

```bash
set -a && source .env && set +a
dbt run --profiles-dir profiles/
dbt test --profiles-dir profiles/
```

### 5. Run analysis

```bash
python analysis/portfolio_optimizer.py
python analysis/dca_simulator.py
python analysis/correlation_lead_lag.py
python analysis/regime_entry_backtest.py
python analysis/tavily_enriched_insights.py   # requires TAVILY_API_KEY
```

---

## Key findings

- **BTC/QQQ correlation lags SPY stress by ~5 days** — it confirms equity contagion after the fact, not before. Useful as a regime monitor, not an early warning signal.
- **Lump sum beats DCA+HYSA above a ~3% annual return threshold** — at this portfolio's ~30% blended return, lump sum produces ~70% more terminal wealth at 5Y. HYSA at 4.5% meaningfully reduces the DCA penalty but doesn't eliminate it.
- **`neutral_high_normal` is BTC's best historical entry regime** — +9.7% median 1Y forward return across 3,664 observations. The same regime is TLT's worst (-1.7%, 36% hit rate).
- **Timing the macro regime matters far more for fixed income and crypto than equities** — the worst QQQ regime still returns +1.1% median. TLT's range spans -2.7% to +1.5%.
- **BTC recovers from drawdowns fastest** (96-day average) despite having the deepest troughs. BND is the slowest to recover (837-day average).

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `SNOWFLAKE_ACCOUNT` | Account identifier (`orgname-accountname`) |
| `SNOWFLAKE_USER` | Snowflake username |
| `SNOWFLAKE_PASSWORD` | Snowflake password |
| `SNOWFLAKE_WAREHOUSE` | Compute warehouse (e.g. `COMPUTE_WH`) |
| `SNOWFLAKE_ROLE` | Role (e.g. `ACCOUNTADMIN`) |
| `SNOWFLAKE_DATABASE` | Target database (e.g. `FIN_DATA_WAREHOUSE`) |
| `SNOWFLAKE_SCHEMA` | Default schema (e.g. `RAW`) |
| `FRED_API_KEY` | [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) |
| `TAVILY_API_KEY` | [Tavily API key](https://tavily.com) |
