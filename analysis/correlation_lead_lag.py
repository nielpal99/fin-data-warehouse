import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import snowflake.connector
from pathlib import Path
from dotenv import load_dotenv
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")
load_dotenv(Path(__file__).parent.parent / ".env")

OUT = Path(__file__).parent

# ── Convention note ────────────────────────────────────────────────────────────
# lag > 0 : BTC/QQQ correlation at time t correlates with SPY drawdown at t+lag
#           → correlation changes BEFORE drawdown deepens → LEADING indicator
# lag < 0 : BTC/QQQ correlation at time t correlates with SPY drawdown at t+lag
#           → drawdown deepens BEFORE correlation changes  → LAGGING indicator
# lag = 0 : COINCIDENT
#
# Implementation: pearsonr(corr.shift(lag), drawdown)
#   corr.shift(lag)[t] = corr[t - lag]
#   ≡ pearsonr(corr[t], drawdown[t + lag])
# ──────────────────────────────────────────────────────────────────────────────


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
    print(f"\n{'='*65}")
    print(f" {title}")
    print(f"{'='*65}")
    col_w = [max(len(c), df[c].astype(str).str.len().max()) for c in df.columns]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    print(fmt.format(*df.columns))
    print("  ".join("-" * w for w in col_w))
    for _, row in df.iterrows():
        print(fmt.format(*row.astype(str)))


# ── STEP 1: pull data ──────────────────────────────────────────────────────────

print("\n" + "="*65)
print(" STEP 1 — Loading data from Snowflake")
print("="*65)

con = get_conn()

corr_df = pd.read_sql("""
    SELECT date, correlation_90d
    FROM fct_rolling_correlations
    WHERE ticker_a = 'BTC-USD' AND ticker_b = 'QQQ'
    ORDER BY date
""", con)
corr_df.columns = corr_df.columns.str.lower()

dd_df = pd.read_sql("""
    SELECT date, drawdown
    FROM fct_drawdown
    WHERE ticker = 'SPY'
    ORDER BY date
""", con)
dd_df.columns = dd_df.columns.str.lower()

con.close()

corr_df["date"] = pd.to_datetime(corr_df["date"])
dd_df["date"]   = pd.to_datetime(dd_df["date"])

# Join on date, forward fill nulls (BTC/QQQ has weekend gaps; SPY does not)
merged = pd.merge(dd_df, corr_df, on="date", how="left")
merged = merged.sort_values("date").reset_index(drop=True)
merged["correlation_90d"] = merged["correlation_90d"].ffill()
merged = merged.dropna()

print(f"  BTC/QQQ correlation rows : {len(corr_df):,}")
print(f"  SPY drawdown rows        : {len(dd_df):,}")
print(f"  Merged (after ffill+drop): {len(merged):,}")
print(f"  Date range               : {merged['date'].min().date()} → {merged['date'].max().date()}")

spy_drawdown = merged["drawdown"]
btc_qqq_corr = merged["correlation_90d"]


# ── STEP 2: cross-correlation at lags -30 to +30 ─────────────────────────────

print("\n" + "="*65)
print(" STEP 2 — Cross-correlation (-30 to +30 trading days)")
print("="*65)

LAG_RANGE = range(-30, 31)
records = []

for lag in LAG_RANGE:
    shifted = btc_qqq_corr.shift(lag)
    pair = pd.DataFrame({"corr": shifted, "dd": spy_drawdown}).dropna()
    r, p = pearsonr(pair["corr"], pair["dd"])
    direction = (
        "LEADS drawdown" if lag > 0
        else "LAGS drawdown" if lag < 0
        else "COINCIDENT"
    )
    records.append({
        "lag_days":  lag,
        "pearson_r": round(r, 4),
        "p_value":   round(p, 4),
        "direction": direction,
    })

results = pd.DataFrame(records)

# Full table
display_df = results.copy()
display_df["pearson_r"] = display_df["pearson_r"].map(lambda x: f"{x:+.4f}")
display_df["p_value"]   = display_df["p_value"].map(lambda x: f"{x:.4f}")
print_table("Cross-Correlation: BTC/QQQ Correlation vs SPY Drawdown", display_df)

# Top 5 strongest (by |r|)
top5 = results.reindex(results["pearson_r"].abs().nlargest(5).index).sort_values(
    "pearson_r", key=abs, ascending=False
).reset_index(drop=True)
top5_disp = top5.copy()
top5_disp["pearson_r"] = top5_disp["pearson_r"].map(lambda x: f"{x:+.4f}")
top5_disp["p_value"]   = top5_disp["p_value"].map(lambda x: f"{x:.4f}")
print_table("Top 5 Strongest Cross-Correlations", top5_disp)


# ── STEP 3: classify and print conclusion ─────────────────────────────────────

print("\n" + "="*65)
print(" STEP 3 — Classification")
print("="*65)

best_idx    = results["pearson_r"].abs().idxmax()
best        = results.loc[best_idx]
best_lag    = int(best["lag_days"])
best_r      = best["pearson_r"]
best_p      = best["p_value"]

if best_lag > 0:
    indicator_type = "LEADING"
    timing_desc    = f"BTC/QQQ correlation rises {best_lag} trading days BEFORE SPY drawdowns deepen"
elif best_lag < 0:
    indicator_type = "LAGGING"
    timing_desc    = f"BTC/QQQ correlation rises {abs(best_lag)} trading days AFTER SPY drawdowns deepen"
else:
    indicator_type = "COINCIDENT"
    timing_desc    = "BTC/QQQ correlation moves simultaneously with SPY drawdowns"

sig = "statistically significant (p < 0.05)" if best_p < 0.05 else "not statistically significant (p ≥ 0.05)"

print(f"\n  Max |r| lag   : {best_lag:+d} days")
print(f"  Pearson r     : {best_r:+.4f}")
print(f"  p-value       : {best_p:.4f}  ({sig})")
print(f"  Classification: {indicator_type}")
print(f"\n  CONCLUSION: BTC/QQQ correlation is a {indicator_type} indicator of SPY stress")
print(f"              by {abs(best_lag)} trading days (r={best_r:+.4f}, p={best_p:.4f})")
print(f"              {timing_desc}.")

# Lag-0 baseline for comparison
lag0 = results.loc[results["lag_days"] == 0].iloc[0]
print(f"\n  Lag-0 (coincident) baseline: r={lag0['pearson_r']:+.4f}, p={lag0['p_value']:.4f}")
print(f"  Improvement over coincident: {abs(best_r) - abs(lag0['pearson_r']):+.4f} in |r|")


# ── STEP 4: chart ─────────────────────────────────────────────────────────────

print("\n" + "="*65)
print(" STEP 4 — Generating chart")
print("="*65)

fig, axes = plt.subplots(3, 1, figsize=(14, 12))
dates = merged["date"]

# ── Subplot 1: BTC/QQQ rolling 90d correlation ───────────────────────────────
ax1 = axes[0]
ax1.plot(dates, btc_qqq_corr, color="steelblue", linewidth=0.9, alpha=0.9)
ax1.axhline(btc_qqq_corr.mean(), color="steelblue", linewidth=1.2,
            linestyle="--", alpha=0.5, label=f"Mean ({btc_qqq_corr.mean():.2f})")
ax1.fill_between(dates, btc_qqq_corr, btc_qqq_corr.mean(),
                 where=btc_qqq_corr > btc_qqq_corr.mean(),
                 alpha=0.15, color="steelblue")
ax1.set_ylabel("Correlation (90d rolling)", fontsize=10)
ax1.set_title("BTC/QQQ 90-Day Rolling Correlation", fontsize=11, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_major_locator(mdates.YearLocator())

# ── Subplot 2: SPY drawdown ───────────────────────────────────────────────────
ax2 = axes[1]
ax2.fill_between(dates, spy_drawdown * 100, 0,
                 color="crimson", alpha=0.4)
ax2.plot(dates, spy_drawdown * 100, color="crimson", linewidth=0.8)
ax2.axhline(-10, color="darkred", linewidth=1.0, linestyle="--",
            alpha=0.7, label="-10% threshold")
ax2.set_ylabel("Drawdown (%)", fontsize=10)
ax2.set_title("SPY Drawdown from Peak", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator())

# ── Subplot 3: cross-correlation function ────────────────────────────────────
ax3 = axes[2]
lags_arr = results["lag_days"].values
rs_arr   = results["pearson_r"].values

# Colour bars by lead/lag/coincident
colors = [
    "steelblue" if l > 0 else
    "darkorange" if l < 0 else
    "gray"
    for l in lags_arr
]
bars = ax3.bar(lags_arr, rs_arr, color=colors, alpha=0.75, width=0.8)

# Highlight the maximum |r| bar
bars[best_idx].set_edgecolor("black")
bars[best_idx].set_linewidth(2.0)
bars[best_idx].set_alpha(1.0)

ax3.axvline(0, color="black", linewidth=1.2, linestyle="-", alpha=0.6, label="Lag = 0")
ax3.axvline(best_lag, color="gold", linewidth=2.0, linestyle="--",
            label=f"Max |r| at lag={best_lag:+d} ({indicator_type})")
ax3.axhline(0, color="black", linewidth=0.6, alpha=0.4)

# Significance bands (approximate, ±2/√n)
n_obs = len(merged) - 30
sig_band = 2 / np.sqrt(n_obs)
ax3.axhline( sig_band, color="gray", linewidth=1.0, linestyle=":", alpha=0.6, label="±95% CI")
ax3.axhline(-sig_band, color="gray", linewidth=1.0, linestyle=":", alpha=0.6)

# Annotate best point
ax3.annotate(
    f"r={best_r:+.3f}\nlag={best_lag:+d}d\n{indicator_type}",
    xy=(best_lag, best_r),
    xytext=(best_lag + (4 if best_lag < 15 else -12), best_r - 0.015),
    fontsize=8, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
)

# Legend patches for colour coding
from matplotlib.patches import Patch
legend_handles = [
    Patch(color="steelblue",  alpha=0.75, label="Corr leads drawdown (+lag)"),
    Patch(color="darkorange", alpha=0.75, label="Corr lags drawdown (−lag)"),
    Patch(color="gray",       alpha=0.75, label="Coincident (lag=0)"),
]
ax3.legend(handles=legend_handles + [
    plt.Line2D([0], [0], color="gold",  linewidth=2, linestyle="--",
               label=f"Max |r| lag={best_lag:+d} ({indicator_type})"),
    plt.Line2D([0], [0], color="black", linewidth=1.2, linestyle="-",
               label="Lag = 0"),
    plt.Line2D([0], [0], color="gray",  linewidth=1.0, linestyle=":",
               label="±95% CI"),
], fontsize=8, loc="lower left")

ax3.set_xlabel("Lag (trading days)", fontsize=10)
ax3.set_ylabel("Pearson r", fontsize=10)
ax3.set_title(
    f"Cross-Correlation Function — BTC/QQQ Correlation vs SPY Drawdown\n"
    f"Peak |r|={abs(best_r):.3f} at lag={best_lag:+d}d → {indicator_type} indicator",
    fontsize=11, fontweight="bold"
)
ax3.set_xticks(range(-30, 31, 5))
ax3.grid(alpha=0.3, axis="y")

fig.suptitle(
    "Does BTC/QQQ Correlation Lead or Lag SPY Stress?",
    fontsize=14, fontweight="bold",
)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / "correlation_lead_lag.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved → analysis/correlation_lead_lag.png")

print("\n" + "="*65)
print(" Done.")
print("="*65)
