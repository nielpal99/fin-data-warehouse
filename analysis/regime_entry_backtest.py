import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import snowflake.connector
from pathlib import Path
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv(Path(__file__).parent.parent / ".env")

OUT = Path(__file__).parent
TICKERS = ["BTC-USD", "QQQ", "GLD", "VTI", "TLT"]
CURRENT_REGIME = "neutral_high_normal"
MIN_OBS = 30


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


def print_table(title, df):
    print(f"\n{'='*72}")
    print(f" {title}")
    print(f"{'='*72}")
    col_w = [max(len(c), df[c].astype(str).str.len().max()) for c in df.columns]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    print(fmt.format(*df.columns))
    print("  ".join("-" * w for w in col_w))
    for _, row in df.iterrows():
        print(fmt.format(*row.astype(str)))


# ── STEP 1: pull data ──────────────────────────────────────────────────────────

print("\n" + "="*72)
print(" STEP 1 — Loading data from Snowflake")
print("="*72)

con = get_conn()

raw = pd.read_sql("""
    SELECT
        r.date,
        r.ticker,
        r.asset_class,
        r.daily_return,
        r.adjusted_close,
        m.rate_trend,
        m.inflation_regime,
        m.yield_curve_regime,
        m.rate_trend || '_' || m.inflation_regime || '_' || m.yield_curve_regime
            AS composite_regime
    FROM FIN_DATA_WAREHOUSE.RAW_MARTS.fct_daily_returns r
    LEFT JOIN FIN_DATA_WAREHOUSE.RAW_MARTS.fct_macro_overlay m
        ON r.date = m.date
    WHERE r.ticker IN ('BTC-USD', 'QQQ', 'GLD', 'VTI', 'TLT')
      AND r.daily_return IS NOT NULL
    ORDER BY r.ticker, r.date
""", con)
raw.columns = raw.columns.str.lower()
con.close()

raw["date"] = pd.to_datetime(raw["date"])
raw = raw.sort_values(["ticker", "date"]).reset_index(drop=True)
raw["composite_regime"] = raw["composite_regime"].ffill()

print(f"  Rows loaded       : {len(raw):,}")
print(f"  Tickers           : {sorted(raw['ticker'].unique())}")
print(f"  Date range        : {raw['date'].min().date()} → {raw['date'].max().date()}")
print(f"  Regimes seen      : {sorted(raw['composite_regime'].dropna().unique())}")


# ── STEP 2: compute 252-day forward returns ────────────────────────────────────

print("\n" + "="*72)
print(" STEP 2 — Computing 252-day forward returns")
print("="*72)

frames = []
for ticker, grp in raw.groupby("ticker"):
    g = grp.sort_values("date").reset_index(drop=True)
    g["future_close"] = g["adjusted_close"].shift(-252)
    g["forward_return_1y"] = (g["future_close"] / g["adjusted_close"]) - 1
    frames.append(g)

df = pd.concat(frames, ignore_index=True)
df_valid = df.dropna(subset=["forward_return_1y", "composite_regime"]).copy()

print(f"  Rows before drop  : {len(df):,}")
print(f"  Rows after drop   : {len(df_valid):,}  (last 252 trading days removed)")


# ── STEP 3: aggregate by ticker + entry_regime ────────────────────────────────

print("\n" + "="*72)
print(" STEP 3 — Aggregating by ticker × entry_regime")
print("="*72)

def agg_regime(g):
    r = g["forward_return_1y"]
    return pd.Series({
        "median_1y_return": r.median(),
        "mean_1y_return":   r.mean(),
        "pct_positive":     (r > 0).mean(),
        "pct_double":       (r > 1.0).mean(),
        "n_observations":   len(r),
        "sharpe_entry":     r.mean() / r.std() if r.std() > 0 else np.nan,
    })

agg = (
    df_valid
    .groupby(["ticker", "composite_regime"])
    .apply(agg_regime)
    .reset_index()
)

agg = agg[agg["n_observations"] >= MIN_OBS].copy()
agg["rank"] = (
    agg.groupby("ticker")["median_1y_return"]
    .rank(ascending=False, method="min")
    .astype(int)
)
agg["n_regimes"] = agg.groupby("ticker")["composite_regime"].transform("count")

print(f"  Ticker×regime rows (n≥{MIN_OBS}): {len(agg)}")
print(f"  Regimes per ticker:")
for t in TICKERS:
    n = len(agg[agg["ticker"] == t])
    print(f"    {t:<12} {n} regimes")


# ── STEP 4: print top/bottom 3 per ticker + current regime ranking ────────────

print("\n" + "="*72)
print(" STEP 4 — Top 3 / Bottom 3 entry regimes per ticker")
print("="*72)

current_rank_rows = []

for ticker in TICKERS:
    sub = agg[agg["ticker"] == ticker].sort_values("median_1y_return", ascending=False).reset_index(drop=True)
    n_total = len(sub)

    top3    = sub.head(3).copy()
    bottom3 = sub.tail(3).sort_values("median_1y_return").reset_index(drop=True)

    def fmt_sub(s):
        d = s[["ticker", "composite_regime", "median_1y_return", "pct_positive", "n_observations"]].copy()
        d["median_1y_pct"]  = (d["median_1y_return"] * 100).round(1).astype(str) + "%"
        d["pct_positive"]   = (d["pct_positive"] * 100).round(1).astype(str) + "%"
        d["n_observations"] = d["n_observations"].astype(int).astype(str)
        return d[["ticker", "composite_regime", "median_1y_pct", "pct_positive", "n_observations"]]

    print_table(f"{ticker} — Top 3 Entry Regimes", fmt_sub(top3))
    print_table(f"{ticker} — Bottom 3 Entry Regimes", fmt_sub(bottom3))

    # Current regime rank
    cur_row = sub[sub["composite_regime"] == CURRENT_REGIME]
    if len(cur_row):
        cr = cur_row.iloc[0]
        rank_str = f"#{int(cr['rank'])} of {n_total}"
        med_str  = f"{cr['median_1y_return']*100:+.1f}%"
        pct_str  = f"{cr['pct_positive']*100:.0f}%"
        current_rank_rows.append({
            "ticker":          ticker,
            "rank":            rank_str,
            "median_1y_pct":   med_str,
            "pct_positive":    pct_str,
            "n_observations":  int(cr["n_observations"]),
        })
    else:
        current_rank_rows.append({
            "ticker":         ticker,
            "rank":           "N/A (n<30)",
            "median_1y_pct":  "—",
            "pct_positive":   "—",
            "n_observations": 0,
        })

print("\n" + "="*72)
print(f" Current Regime ({CURRENT_REGIME}) — Historical Ranking per Ticker")
print("="*72)
cur_df = pd.DataFrame(current_rank_rows)
col_w = [max(len(c), cur_df[c].astype(str).str.len().max()) for c in cur_df.columns]
fmt   = "  ".join(f"{{:<{w}}}" for w in col_w)
print(fmt.format(*cur_df.columns))
print("  ".join("-" * w for w in col_w))
for _, row in cur_df.iterrows():
    print(fmt.format(*row.astype(str)))


# ── STEP 5: chart ─────────────────────────────────────────────────────────────

print("\n" + "="*72)
print(" STEP 5 — Generating chart")
print("="*72)

fig, axes = plt.subplots(len(TICKERS), 1, figsize=(14, 4.5 * len(TICKERS)))

for ax, ticker in zip(axes, TICKERS):
    sub = (
        agg[agg["ticker"] == ticker]
        .sort_values("median_1y_return")
        .reset_index(drop=True)
    )

    regimes  = sub["composite_regime"].tolist()
    medians  = sub["median_1y_return"].tolist()
    colors   = ["#2ecc71" if v >= 0 else "#e74c3c" for v in medians]

    bars = ax.barh(regimes, medians, color=colors, alpha=0.80, height=0.6)

    # Highlight current regime bar
    if CURRENT_REGIME in regimes:
        idx = regimes.index(CURRENT_REGIME)
        bars[idx].set_edgecolor("black")
        bars[idx].set_linewidth(2.0)
        bars[idx].set_alpha(1.0)

    ax.axvline(0, color="black", linewidth=0.8, alpha=0.6)

    # Annotate each bar with value
    for bar, val in zip(bars, medians):
        xpos = val + (0.005 if val >= 0 else -0.005)
        ha   = "left" if val >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{val*100:+.1f}%", va="center", ha=ha, fontsize=7.5)

    # Build title with current regime rank
    cur_row = sub[sub["composite_regime"] == CURRENT_REGIME]
    n_total = len(sub)
    if len(cur_row):
        cr       = cur_row.iloc[0]
        rank_lbl = f"#{int(cr['rank'])} of {n_total}"
        med_lbl  = f"{cr['median_1y_return']*100:+.1f}%"
        title    = (
            f"{ticker}  —  Median 1Y Forward Return by Entry Regime\n"
            f"Current regime: {CURRENT_REGIME}  →  ranks {rank_lbl}  (median {med_lbl})"
        )
    else:
        title = (
            f"{ticker}  —  Median 1Y Forward Return by Entry Regime\n"
            f"Current regime: {CURRENT_REGIME}  →  N/A (insufficient history)"
        )

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Median 1Y Forward Return", fontsize=9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    ax.grid(alpha=0.3, axis="x")
    ax.tick_params(axis="y", labelsize=8)

    # Current regime label on y-axis in bold
    for label in ax.get_yticklabels():
        if label.get_text() == CURRENT_REGIME:
            label.set_fontweight("bold")
            label.set_color("#2255cc")

fig.suptitle(
    f"Regime Entry Backtest — 1Y Forward Returns by Macro Regime\n"
    f"Current regime: {CURRENT_REGIME}  |  Min observations: {MIN_OBS}",
    fontsize=13, fontweight="bold",
)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / "regime_entry_backtest.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved → analysis/regime_entry_backtest.png")

print("\n" + "="*72)
print(" Done.")
print("="*72)
