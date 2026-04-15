import os
import re
import datetime
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
import snowflake.connector

warnings.filterwarnings("ignore")
load_dotenv(Path(__file__).parent.parent / ".env")

st.set_page_config(
    page_title="Fin Data Warehouse",
    page_icon="📈",
    layout="wide",
)

ANALYSIS = Path(__file__).parent.parent / "analysis"


@st.cache_resource
def get_conn():
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        role=os.environ["SNOWFLAKE_ROLE"],
        database="FIN_DATA_WAREHOUSE",
        schema="RAW_MARTS",
    )


@st.cache_data(ttl=3600)
def query(sql):
    df = pd.read_sql(sql, get_conn())
    df.columns = df.columns.str.lower()
    return df


# ── Shared helpers ────────────────────────────────────────────────────────────

PREAMBLE_PATTERNS = [
    r"^Here is a \d+ sentence macro regime analysis[:\s]+",
    r"^Here is a \d+-sentence macro regime analysis[:\s]+",
    r"^Here is a \d+ sentence drawdown analysis for [^:]+[:\s]+",
    r"^Here is a \d+-sentence drawdown analysis for [^:]+[:\s]+",
    r"^Here is a \d+ sentence explanation for[^:]+[:\s]+",
    r"^Here is a \d+-sentence explanation for[^:]+[:\s]+",
]

def clean_note(text):
    """Strip Cortex preamble phrases and surrounding whitespace."""
    if not isinstance(text, str):
        return ""
    t = text.strip()
    for pattern in PREAMBLE_PATTERNS:
        t = re.sub(pattern, "", t, flags=re.IGNORECASE).strip()
    return t


def insight_label(row):
    """Human-readable expander label for a Cortex insight row."""
    if row["insight_type"] == "macro":
        return "📊 Today's Macro Regime Summary"
    dd_pct = row.get("dd_pct", "")
    pct_str = f" ({dd_pct}%)" if dd_pct else ""
    return f"⚠️ {row['ticker']} — Active Drawdown{pct_str}"


def _tavily_key():
    """Resolve Tavily API key from Streamlit secrets or env."""
    try:
        return st.secrets["TAVILY_API_KEY"]
    except Exception:
        return os.environ.get("TAVILY_API_KEY")


@st.cache_data(ttl=3600)
def get_tavily_insights(drawdown_tickers: tuple, api_key: str):
    """
    Primary insight source. Searches Tavily for macro + each active drawdown
    ticker. Returns dict: {label -> {"snippets": [...], "dd_pct": str}}.
    drawdown_tickers is a tuple of (ticker, dd_pct) pairs (hashable for cache).
    api_key passed explicitly so cache key includes it.
    """
    key = api_key
    if not key:
        return None
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=key)
        results = {}

        # Macro search
        resp = client.search(
            "Fed reserve CPI inflation macro outlook latest",
            search_depth="basic", max_results=3,
        )
        results["macro"] = {
            "label":    "📊 Today's Macro Context",
            "snippets": [r.get("content", "")[:350]
                         for r in resp.get("results", [])
                         if r.get("content")][:3],
            "dd_pct":   "",
        }

        # One search per active drawdown ticker
        for ticker, dd_pct in drawdown_tickers:
            name = ticker.replace("-USD", "").replace("-", " ")
            resp = client.search(
                f"{name} {ticker} price market news today",
                search_depth="basic", max_results=3,
            )
            results[ticker] = {
                "label":    f"⚠️ {ticker} — Active Drawdown ({dd_pct}%)",
                "snippets": [r.get("content", "")[:350]
                             for r in resp.get("results", [])
                             if r.get("content")][:2],
                "dd_pct":   str(dd_pct),
            }

        return results
    except Exception:
        return None


def render_overview_insights():
    """
    Overview page insight renderer.
    PRIMARY: Tavily live news (dynamic drawdown tickers).
    FALLBACK: fct_cortex_enriched_insights with preamble stripping.
    """
    # Fetch active major drawdowns to drive dynamic Tavily searches
    try:
        dd_df = query("""
            SELECT ticker, ROUND(drawdown * 100, 1) AS dd_pct
            FROM fct_drawdown
            WHERE is_major_drawdown = TRUE
              AND date = (SELECT MAX(date) FROM fct_drawdown)
            ORDER BY drawdown
        """)
        drawdown_tickers = tuple(
            (row["ticker"], row["dd_pct"]) for _, row in dd_df.iterrows()
        )
    except Exception:
        drawdown_tickers = ()

    # ── Try Tavily first ──────────────────────────────────────────────────────
    news = get_tavily_insights(drawdown_tickers, _tavily_key() or "")
    if news:
        ts = datetime.datetime.now().strftime("%b %d %Y %H:%M")
        for key, item in news.items():
            snippets = item["snippets"]
            label    = item["label"]
            if key == "macro":
                with st.expander(label, expanded=True):
                    for s in snippets:
                        st.markdown(f"- {s}")
                    st.caption(f"Updated hourly · {ts}")
            else:
                with st.expander(label):
                    for s in snippets:
                        st.markdown(f"- {s}")
        return

    # ── Tavily failed — fall back to Cortex ──────────────────────────────────
    st.caption("Live news unavailable — showing Cortex quantitative analysis")
    try:
        insights_df = query("""
            SELECT ticker, insight_type, enriched_note
            FROM fct_cortex_enriched_insights
            ORDER BY ticker
        """)
        dd_today = {}
        try:
            dd_today = query("""
                SELECT ticker, ROUND(drawdown * 100, 1) AS dd_pct
                FROM fct_drawdown
                WHERE date = (SELECT MAX(date) FROM fct_drawdown)
            """).set_index("ticker")["dd_pct"].to_dict()
        except Exception:
            pass

        valid = insights_df[
            insights_df["enriched_note"].notna() &
            (insights_df["enriched_note"].str.strip() != "")
        ]
        if len(valid) > 0:
            for _, row in valid.iterrows():
                row = row.copy()
                row["dd_pct"] = dd_today.get(row["ticker"], "")
                label = insight_label(row)
                note  = clean_note(row["enriched_note"])
                with st.expander(label):
                    st.write(note)
            return
    except Exception:
        pass

    st.caption("Insights unavailable — check API credentials")


def render_insights(insights_df):
    """
    Used by The Research page (Cortex primary, Tavily fallback).
    """
    try:
        dd_today = query("""
            SELECT ticker, ROUND(drawdown * 100, 1) AS dd_pct
            FROM fct_drawdown
            WHERE date = (SELECT MAX(date) FROM fct_drawdown)
        """).set_index("ticker")["dd_pct"].to_dict()
    except Exception:
        dd_today = {}

    valid = insights_df[
        insights_df["enriched_note"].notna() &
        (insights_df["enriched_note"].str.strip() != "")
    ] if len(insights_df) > 0 else insights_df

    if len(valid) > 0:
        for _, row in valid.iterrows():
            row = row.copy()
            row["dd_pct"] = dd_today.get(row["ticker"], "")
            label = insight_label(row)
            note  = clean_note(row["enriched_note"])
            with st.expander(label):
                st.write(note)
        return

    # Cortex empty — try Tavily
    st.caption("Cortex insights unavailable — fetching live news via Tavily")
    try:
        dd_df = query("""
            SELECT ticker, ROUND(drawdown * 100, 1) AS dd_pct
            FROM fct_drawdown
            WHERE is_major_drawdown = TRUE
              AND date = (SELECT MAX(date) FROM fct_drawdown)
        """)
        drawdown_tickers = tuple(
            (row["ticker"], row["dd_pct"]) for _, row in dd_df.iterrows()
        )
    except Exception:
        drawdown_tickers = ()

    news = get_tavily_insights(drawdown_tickers, _tavily_key() or "")
    if news:
        for key, item in news.items():
            with st.expander(f"{item['label']} — via Tavily"):
                for s in item["snippets"]:
                    st.markdown(f"- {s}")
        return

    st.caption("Insights unavailable — check API credentials")


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("", [
    "The Research",
    "Overview",
    "Portfolio Analytics",
    "Drawdown & Episodes",
    "Correlations",
    "Macro Regime",
    "Analysis Charts",
])

# ── The Research ──────────────────────────────────────────────────────────────
if page == "The Research":

    def prose(text):
        """Render narrative text centred in a narrow column."""
        _, col, _ = st.columns([1, 6, 1])
        with col:
            st.markdown(text)

    # ── Opening ───────────────────────────────────────────────────────────────
    prose("""
I've been following markets seriously for years. I have opinions about macro,
I invest my own money, and I've spent more time than I'd like reading Fed
statements and earnings transcripts.

But there were three things I kept believing without ever actually checking:
That crypto diversifies an equity portfolio. That bonds are the safe asset.
That sophisticated portfolio optimization beats simple approaches.

So I built a framework to find out. Here's what 10 years of daily data
across 11 assets actually said.
""")

    st.divider()

    # ── Finding 1 ─────────────────────────────────────────────────────────────
    prose("""
## Finding 1 — Your crypto hedge disappears after the damage starts

Everyone knows BTC and QQQ correlate during market stress. The question
nobody asks is: does the correlation spike before equity drawdowns begin,
during, or after?

I cross-correlated rolling 90-day BTC/QQQ correlation against SPY drawdowns
at every lag from −30 to +30 trading days.
""")

    # BTC rolling correlations over time
    btc_corr = query("""
        SELECT date,
               ticker_a || ' / ' || ticker_b AS pair,
               correlation_90d
        FROM fct_rolling_correlations
        WHERE ticker_a = 'BTC-USD'
        ORDER BY date, pair
    """)
    btc_corr["date"] = pd.to_datetime(btc_corr["date"])

    fig1, ax1 = plt.subplots(figsize=(12, 4))
    for pair, grp in btc_corr.groupby("pair"):
        lw = 2.2 if pair == "BTC-USD / QQQ" else 1.0
        alpha = 0.95 if pair == "BTC-USD / QQQ" else 0.55
        ax1.plot(grp["date"], grp["correlation_90d"], label=pair, linewidth=lw, alpha=alpha)
    ax1.axhline(0, color="black", linewidth=0.7, alpha=0.4)
    ax1.set_ylabel("90-Day Rolling Correlation")
    ax1.set_title("BTC Rolling Correlations vs All Pairs")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.25)
    import matplotlib.dates as mdates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    prose("""
The result: BTC/QQQ correlation is a **lagging indicator** of equity stress
by approximately 5 trading days. By the time crypto and tech are moving in
lockstep, the SPY drawdown is already a week old.

The genuine crisis hedge in the data is gold — not crypto. In June 2022,
BTC/GLD correlation went negative while BTC/QQQ spiked to 0.53. Gold held
its bid while everything else fell.
""")

    # Correlation heatmap (all-time)
    matrix = query("""
        SELECT ticker_a, ticker_b, correlation
        FROM fct_correlation_matrix
        WHERE period_days = (SELECT MAX(period_days) FROM fct_correlation_matrix)
    """)
    pivot = matrix.pivot(index="ticker_a", columns="ticker_b", values="correlation").round(2)
    fig2, ax2 = plt.subplots(figsize=(9, 7))
    im = ax2.imshow(pivot.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax2.set_xticks(range(len(pivot.columns)))
    ax2.set_yticks(range(len(pivot.index)))
    ax2.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax2.set_yticklabels(pivot.index, fontsize=8)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax2.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                         color="black" if abs(val) < 0.7 else "white")
    plt.colorbar(im, ax=ax2)
    ax2.set_title("All-Time Pairwise Correlation Matrix")
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    st.divider()

    # ── Finding 2 ─────────────────────────────────────────────────────────────
    prose("""
## Finding 2 — The asset most people think is safe had the worst recovery profile

When most people think about risk, they think about how far an asset can fall.
They rarely ask: if it does fall, how long until I get my money back?
""")

    # Recovery bar chart
    recovery = query("""
        SELECT ticker,
               COUNT(*) AS n_episodes,
               ROUND(AVG(recovery_days), 0) AS avg_recovery_days
        FROM fct_drawdown_episodes
        WHERE recovery_days IS NOT NULL
        GROUP BY ticker
        ORDER BY avg_recovery_days DESC
    """)

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    colors3 = ["#e74c3c" if t in ("BND", "TLT") else
               "#2ecc71" if t == "BTC-USD" else "#5b9bd5"
               for t in recovery["ticker"]]
    bars = ax3.barh(recovery["ticker"], recovery["avg_recovery_days"],
                    color=colors3, alpha=0.85)
    for bar, n in zip(bars, recovery["n_episodes"]):
        ax3.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                 f"{int(bar.get_width())}d  (n={int(n)})",
                 va="center", fontsize=8)
    ax3.set_xlabel("Average Days to Recovery")
    ax3.set_title("Average Drawdown Recovery Time by Asset")
    ax3.set_xlim(0, recovery["avg_recovery_days"].max() * 1.25)
    ax3.grid(alpha=0.25, axis="x")
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    prose("""
Bitcoin recovers from major drawdowns in **96 days** on average across 73 episodes
since 2015. BND takes **837 days** across 6 episodes — dominated by the 2022 rate
shock which created a bond drawdown that lasted years.

The asset class most people hold for safety had the worst recovery profile in
the entire dataset. Bitcoin crashes harder. It bounces faster.
""")

    st.divider()

    # ── Finding 3 ─────────────────────────────────────────────────────────────
    prose("""
## Finding 3 — Asset selection matters. Optimization doesn't.

I optimized a portfolio on 2015–2019 data and tested it blind on 2020–2026.
The question: does a sophisticated optimizer actually add value over naive
equal weighting across the same assets?
""")

    oos_df = pd.DataFrame({
        "Portfolio":        ["Optimized (train weights)", "Equal Weight", "SPY", "60/40 (VTI/BND)"],
        "Sharpe (OOS)":     ["0.863", "0.867", "0.486", "0.313"],
        "Cumulative Return (2020–2026)": ["+293%", "+275%", "+123%", "+68%"],
    })
    _, col_mid, _ = st.columns([1, 6, 1])
    with col_mid:
        st.dataframe(oos_df, use_container_width=True, hide_index=True)

    prose("""
The optimized portfolio returned **+293%** vs SPY's **+123%** — genuinely strong.
But the naive equal-weight portfolio did almost identically well on a
risk-adjusted basis (0.867 vs 0.863 Sharpe).

All that optimization machinery produced the same result as putting equal
amounts in everything. Once you've chosen the right assets, the weights
are almost noise.
""")

    st.divider()

    # ── Section 5 ─────────────────────────────────────────────────────────────
    prose("""
## This updates daily

The three findings above reflect 10 years of historical data. But the more
valuable thing is that the framework updates every day.

The correlation tracker shows in real time whether BTC is drifting toward
or away from equities. The regime classifier flags when the macro environment
shifts. The anomaly detector fires when any asset moves beyond 2 standard
deviations and Cortex explains why in plain English grounded in that day's news.

Most investing decisions are made on recent headlines and narrative.
This framework doesn't eliminate uncertainty — no framework does.
But it ensures the starting point is evidence, not vibes.
""")

    insights = query("""
        SELECT ticker, insight_type, enriched_note
        FROM fct_cortex_enriched_insights
        ORDER BY ticker
    """)
    _, col_ins, _ = st.columns([1, 6, 1])
    with col_ins:
        render_insights(insights)

    st.divider()
    st.caption(
        "Built with Python · DuckDB · Snowflake · dbt · Snowflake Cortex · Tavily  \n"
        "Data: yfinance + FRED API · Not financial advice · "
        "[GitHub](https://github.com/nielpal99/fin-data-warehouse)"
    )


# ── Overview ──────────────────────────────────────────────────────────────────
elif page == "Overview":
    st.title("Financial Data Warehouse")
    st.caption("2015–2026 · 11 tickers · yfinance + FRED · Snowflake + dbt")

    col1, col2, col3, col4 = st.columns(4)

    latest = query("""
        SELECT date, ticker, daily_return, adjusted_close, is_anomaly
        FROM fct_daily_returns
        WHERE date = (SELECT MAX(date) FROM fct_daily_returns)
        ORDER BY ticker
    """)

    macro = query("""
        SELECT rate_trend, inflation_regime, yield_curve_regime,
               cpi, fed_funds_rate, yield_curve_spread
        FROM fct_macro_overlay
        WHERE date = (SELECT MAX(date) FROM fct_macro_overlay)
        QUALIFY ROW_NUMBER() OVER (ORDER BY date) = 1
    """).iloc[0]

    drawdowns = query("""
        SELECT ticker, ROUND(drawdown * 100, 1) AS dd_pct, is_major_drawdown
        FROM fct_drawdown
        WHERE date = (SELECT MAX(date) FROM fct_drawdown)
        ORDER BY drawdown
    """)

    col1.metric("Rate Trend", macro["rate_trend"].upper())
    col2.metric("Inflation Regime", macro["inflation_regime"].upper())
    col3.metric("Yield Curve", macro["yield_curve_regime"].upper())
    col4.metric("Fed Funds Rate", f"{float(macro['fed_funds_rate']):.2f}%")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Latest Daily Returns")
        disp = latest[["ticker", "daily_return", "adjusted_close", "is_anomaly"]].copy()
        disp["daily_return"] = (disp["daily_return"] * 100).round(2).astype(str) + "%"
        disp["adjusted_close"] = disp["adjusted_close"].round(2)
        disp["is_anomaly"] = disp["is_anomaly"].map({True: "⚠️", False: ""})
        disp.columns = ["Ticker", "Return", "Close", "Anomaly"]
        st.dataframe(disp, use_container_width=True, hide_index=True)
        last_date = latest["date"].max() if "date" in latest.columns else query(
            "SELECT MAX(date) AS d FROM fct_daily_returns").iloc[0]["d"]
        st.caption(
            f"Prices update on market trading days. "
            f"Last ingestion: {pd.to_datetime(last_date).strftime('%b %d, %Y')}"
        )
        if disp["Anomaly"].eq("").all():
            st.caption("No anomalies detected today")

    with col_b:
        st.subheader("Current Drawdowns")
        disp_dd = drawdowns[["ticker", "dd_pct", "is_major_drawdown"]].copy()
        disp_dd["is_major_drawdown"] = disp_dd["is_major_drawdown"].map({True: "🔴", False: ""})
        disp_dd.columns = ["Ticker", "Drawdown %", "Major"]
        st.dataframe(disp_dd, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Market Insights")
    render_overview_insights()


# ── Portfolio Analytics ───────────────────────────────────────────────────────
elif page == "Portfolio Analytics":
    st.title("Portfolio Analytics")
    st.caption(
        "⚠️ All metrics use the full dataset (2015–2026). The optimizer weights shown "
        "(QQQ 30% / GLD 25% / BTC 25% / VTI 10% / VOO 10%) were derived from 2015–2019 "
        "training data only. Out-of-sample testing on 2020–2026 showed naive equal weighting "
        "matched the optimizer on risk-adjusted returns (Sharpe 0.867 vs 0.863) — asset "
        "selection matters more than weight optimization."
    )

    df = query("""
        SELECT ticker, period_label, annualized_return, annualized_vol,
               sharpe_ratio, max_drawdown_1y, sortino_ratio
        FROM fct_portfolio_analytics
        ORDER BY period_label, sharpe_ratio DESC
    """)

    period = st.selectbox("Period", ["1Y", "3Y", "5Y", "ALL"])
    sub = df[df["period_label"] == period].copy()

    sub["annualized_return"] = (sub["annualized_return"] * 100).round(1).astype(str) + "%"
    sub["annualized_vol"]    = (sub["annualized_vol"]    * 100).round(1).astype(str) + "%"
    sub["sharpe_ratio"]      = sub["sharpe_ratio"].round(3)
    sub["max_drawdown_1y"]   = (sub["max_drawdown_1y"]   * 100).round(1).astype(str) + "%"
    sub["sortino_ratio"]     = sub["sortino_ratio"].round(3)
    sub.columns = ["Ticker", "Period", "Ann Return", "Ann Vol", "Sharpe", "Max DD (1Y)", "Sortino"]

    st.dataframe(sub.drop(columns="Period"), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Out-of-Sample Validation (2020–2026)")
    st.caption("Weights trained on 2015–2019 only, evaluated on unseen 2020–2026 data.")
    oos_data = {
        "Portfolio":    ["Optimized (train weights)", "Equal Weight", "SPY", "60/40 (VTI/BND)"],
        "Ann Return":   ["+24.6%", "+23.5%", "+15.0%", "+9.2%"],
        "Ann Vol":      ["22.7%",  "21.3%",  "20.5%",  "13.4%"],
        "Sharpe":       ["0.863",  "0.867",  "0.486",  "0.313"],
        "Max Drawdown": ["-38.9%", "-36.0%", "-33.7%", "-22.7%"],
        "Cum Return":   ["+293%",  "+275%",  "+123%",  "+68%"],
    }
    st.dataframe(pd.DataFrame(oos_data), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Sharpe Ratio by Ticker")
    fig, ax = plt.subplots(figsize=(10, 4))
    raw = df[df["period_label"] == period].sort_values("sharpe_ratio", ascending=True)
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in raw["sharpe_ratio"]]
    ax.barh(raw["ticker"], raw["sharpe_ratio"], color=colors, alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Sharpe Ratio")
    ax.set_title(f"Sharpe Ratio — {period} Period (historical)")
    ax.grid(alpha=0.3, axis="x")
    st.pyplot(fig)
    plt.close(fig)


# ── Drawdown & Episodes ───────────────────────────────────────────────────────
elif page == "Drawdown & Episodes":
    st.title("Drawdown & Episodes")

    tickers = query("SELECT DISTINCT ticker FROM fct_drawdown ORDER BY ticker")["ticker"].tolist()
    selected = st.multiselect("Tickers", tickers, default=["SPY", "BTC-USD", "QQQ"])

    if selected:
        ticker_sql = ", ".join(f"'{t}'" for t in selected)
        dd = query(f"""
            SELECT date, ticker, drawdown
            FROM fct_drawdown
            WHERE ticker IN ({ticker_sql})
            ORDER BY date
        """)
        dd["date"] = pd.to_datetime(dd["date"])

        fig, ax = plt.subplots(figsize=(12, 5))
        for t in selected:
            sub = dd[dd["ticker"] == t]
            ax.plot(sub["date"], sub["drawdown"] * 100, label=t, linewidth=1.2)
        ax.axhline(-10, color="red", linewidth=0.8, linestyle="--", alpha=0.5, label="-10% threshold")
        ax.set_ylabel("Drawdown (%)")
        ax.set_title("Rolling Drawdown from ATH")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.subheader("Drawdown Episodes")
    episodes = query("""
        SELECT ticker, episode_start, episode_trough_date, episode_end,
               ROUND(episode_trough_depth * 100, 1) AS trough_pct,
               days_to_trough, recovery_days, still_in_drawdown
        FROM fct_drawdown_episodes
        ORDER BY episode_trough_depth
        LIMIT 50
    """)
    episodes.columns = ["Ticker", "Start", "Trough Date", "Recovery Date",
                        "Trough %", "Days to Trough", "Days to Recovery", "Active"]
    st.dataframe(episodes, use_container_width=True, hide_index=True)


# ── Correlations ──────────────────────────────────────────────────────────────
elif page == "Correlations":
    st.title("Rolling Correlations")

    pairs = query("""
        SELECT DISTINCT ticker_a || ' / ' || ticker_b AS pair
        FROM fct_rolling_correlations
        ORDER BY 1
    """)["pair"].tolist()

    selected_pairs = st.multiselect("Pairs", pairs, default=["BTC-USD / QQQ", "QQQ / TLT"])

    if selected_pairs:
        conditions = " OR ".join(
            f"(ticker_a = '{p.split(' / ')[0]}' AND ticker_b = '{p.split(' / ')[1]}')"
            for p in selected_pairs
        )
        corr = query(f"""
            SELECT date, ticker_a || ' / ' || ticker_b AS pair, correlation_90d
            FROM fct_rolling_correlations
            WHERE {conditions}
            ORDER BY date
        """)
        corr["date"] = pd.to_datetime(corr["date"])

        fig, ax = plt.subplots(figsize=(12, 5))
        for pair in selected_pairs:
            sub = corr[corr["pair"] == pair]
            ax.plot(sub["date"], sub["correlation_90d"], label=pair, linewidth=1.2)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_ylabel("90-Day Rolling Correlation")
        ax.set_title("Rolling Correlation")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.subheader("Correlation Matrix (All-Time)")
    matrix = query("""
        SELECT ticker_a, ticker_b, correlation
        FROM fct_correlation_matrix
        WHERE period_days = (SELECT MAX(period_days) FROM fct_correlation_matrix)
    """)
    pivot = matrix.pivot(index="ticker_a", columns="ticker_b", values="correlation").round(2)
    fig2, ax2 = plt.subplots(figsize=(9, 7))
    im = ax2.imshow(pivot.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax2.set_xticks(range(len(pivot.columns)))
    ax2.set_yticks(range(len(pivot.index)))
    ax2.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax2.set_yticklabels(pivot.index, fontsize=8)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax2.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                         color="black" if abs(val) < 0.7 else "white")
    plt.colorbar(im, ax=ax2)
    ax2.set_title("All-Time Correlation Matrix")
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)


# ── Macro Regime ──────────────────────────────────────────────────────────────
elif page == "Macro Regime":
    st.title("Macro Regime Performance")

    regime_df = query("""
        SELECT ticker, composite_regime,
               ROUND(annualized_return * 100, 2) AS mean_return_pct,
               ROUND(sharpe, 3) AS sharpe,
               ROUND(annualized_vol * 100, 2) AS vol_pct,
               best_regime
        FROM fct_regime_performance
        ORDER BY ticker, sharpe DESC
    """)

    tickers = regime_df["ticker"].unique().tolist()
    selected_ticker = st.selectbox("Ticker", tickers)

    sub = regime_df[regime_df["ticker"] == selected_ticker].copy()
    sub["best_regime"] = sub["best_regime"].map({True: "⭐", False: ""})
    sub = sub[["composite_regime", "mean_return_pct", "sharpe", "vol_pct", "best_regime"]]
    sub.columns = ["Regime", "Mean Return %", "Sharpe", "Vol %", "Best"]
    st.dataframe(sub, use_container_width=True, hide_index=True)

    st.divider()
    raw = regime_df[regime_df["ticker"] == selected_ticker].sort_values("sharpe")
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in raw["sharpe"]]
    ax.barh(raw["composite_regime"], raw["sharpe"], color=colors, alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Sharpe Ratio")
    ax.set_title(f"{selected_ticker} — Sharpe by Macro Regime")
    ax.grid(alpha=0.3, axis="x")
    st.pyplot(fig)
    plt.close(fig)


# ── Analysis Charts ───────────────────────────────────────────────────────────
elif page == "Analysis Charts":
    st.title("Analysis Charts")

    charts = {
        "Correlation Lead-Lag (BTC/QQQ vs SPY)": ANALYSIS / "correlation_lead_lag.png",
        "Regime Entry Backtest":                 ANALYSIS / "regime_entry_backtest.png",
        "Walk-Forward Validation":               ANALYSIS / "walk_forward_results.png",
    }

    captions = {
        "Correlation Lead-Lag (BTC/QQQ vs SPY)": None,
        "Regime Entry Backtest": (
            "⚠️ All regime data uses full history 2015–2026. Observations within a regime are "
            "time-clustered, not independent — treat rankings as directional context, not a buy signal."
        ),
        "Walk-Forward Validation": None,
    }

    selected_chart = st.selectbox("Chart", list(charts.keys()))
    caption = captions[selected_chart]
    if caption:
        st.caption(caption)

    path = charts[selected_chart]
    if path.exists():
        st.image(str(path), use_container_width=True)
    else:
        st.warning(f"Chart not found at {path}. Run the corresponding analysis script first.")
