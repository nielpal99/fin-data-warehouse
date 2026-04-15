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
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
load_dotenv(Path(__file__).parent.parent / ".env")

OUT = Path(__file__).parent

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_END   = "2019-12-31"
TEST_START  = "2020-01-01"
TEST_END    = "2026-04-01"
RISK_FREE   = 0.05
N_SIM       = 10_000
INITIAL_INV = 10_000
HORIZONS      = [252,  756, 1260]
HORIZON_LABELS = ["1Y", "3Y",  "5Y"]

OPT_TICKERS = ["BTC-USD", "GLD", "QQQ", "VOO", "VTI"]
ALL_TICKERS = sorted(["BTC-USD", "GLD", "QQQ", "VOO", "VTI",
                       "BND", "SPY", "TLT", "ETH-USD", "IWM", "DX-Y.NYB"])


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


def portfolio_metrics(daily_rets, rf=RISK_FREE):
    """Annualized return, vol, Sharpe, max drawdown from a daily return series."""
    ann_ret = daily_rets.mean() * 252
    ann_vol = daily_rets.std()  * np.sqrt(252)
    sharpe  = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
    cum     = (1 + daily_rets).cumprod()
    drawdown = (cum / cum.cummax()) - 1
    max_dd  = drawdown.min()
    cum_ret = cum.iloc[-1] - 1
    return ann_ret, ann_vol, sharpe, max_dd, cum_ret


def constrained_max_sharpe(mean_d, cov_d, tickers, rf=RISK_FREE):
    """SLSQP max Sharpe with position constraints."""
    n = len(tickers)
    ticker_idx = {t: i for i, t in enumerate(tickers)}

    def neg_sharpe(w):
        r = w @ mean_d * 252
        v = np.sqrt(w @ cov_d @ w * 252)
        return -(r - rf) / v if v > 0 else 0

    # Bounds
    bounds = []
    for t in tickers:
        if t == "BTC-USD":
            bounds.append((0.0, 0.25))
        elif t == "GLD":
            bounds.append((0.0, 0.25))
        elif t == "QQQ":
            bounds.append((0.0, 0.30))
        elif t == "VTI":
            bounds.append((0.10, 1.0))
        else:
            bounds.append((0.0, 1.0))

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    w0 = np.ones(n) / n
    res = minimize(neg_sharpe, w0, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-12, "maxiter": 1000})
    return res.x


# ── STEP 1: pull data ──────────────────────────────────────────────────────────

print("\n" + "="*72)
print(" STEP 1 — Loading data from Snowflake")
print("="*72)

tickers_sql = ", ".join(f"'{t}'" for t in OPT_TICKERS + ["BND", "SPY"])
con = get_conn()
df_raw = pd.read_sql(f"""
    SELECT date, ticker, daily_return
    FROM fct_daily_returns
    WHERE ticker IN ({tickers_sql})
    ORDER BY date, ticker
""", con)
con.close()

df_raw.columns = df_raw.columns.str.lower()
df_raw["date"] = pd.to_datetime(df_raw["date"])

wide = df_raw.pivot(index="date", columns="ticker", values="daily_return")

# Train / test split
train = wide.loc[wide.index <= TRAIN_END, OPT_TICKERS].dropna()
test  = wide.loc[(wide.index >= TEST_START) & (wide.index <= TEST_END)].dropna()

# Benchmark tickers available in test
bench_tickers = ["BND", "SPY"]
test_opt  = test[OPT_TICKERS]
test_full = test[OPT_TICKERS + bench_tickers]

print(f"  Train period : {train.index.min().date()} → {train.index.max().date()}  ({len(train):,} days)")
print(f"  Test period  : {test.index.min().date()}  → {test.index.max().date()}   ({len(test):,} days)")
print(f"  Opt tickers  : {OPT_TICKERS}")
print(f"  Bench tickers: {bench_tickers}")


# ── STEP 2: optimize on TRAIN only ────────────────────────────────────────────

print("\n" + "="*72)
print(" STEP 2 — Optimizing on train period (2015–2019)")
print("="*72)

train_mean = train.mean().values
train_cov  = train.cov().values

# Fix PSD
min_eig = np.linalg.eigvalsh(train_cov).min()
if min_eig < 0:
    train_cov += (-min_eig + 1e-10) * np.eye(len(OPT_TICKERS))

train_weights = constrained_max_sharpe(train_mean, train_cov, OPT_TICKERS)
train_port_d  = train.values @ train_weights
train_ann_ret = train_port_d.mean() * 252
train_ann_vol = train_port_d.std()  * np.sqrt(252)
train_sharpe  = (train_ann_ret - RISK_FREE) / train_ann_vol

# Train benchmarks
n_opt    = len(OPT_TICKERS)
ew_w     = np.ones(n_opt) / n_opt

train_ew_d  = train.values @ ew_w
train_ew_r  = train_ew_d.mean() * 252
train_ew_v  = train_ew_d.std()  * np.sqrt(252)
train_ew_sh = (train_ew_r - RISK_FREE) / train_ew_v

# SPY and 60/40 on train — need those tickers
train_spy = wide[(wide.index <= TRAIN_END)][["SPY"]].dropna()
train_bnd = wide[(wide.index <= TRAIN_END)][["BND"]].dropna()
train_spy_bnd = pd.concat([train_spy, train_bnd], axis=1).dropna()

train_spy_d  = train_spy_bnd["SPY"].values
train_spy_r  = train_spy_d.mean() * 252
train_spy_v  = train_spy_d.std()  * np.sqrt(252)
train_spy_sh = (train_spy_r - RISK_FREE) / train_spy_v

train_6040_d  = 0.6 * train_spy_bnd["SPY"].values + 0.4 * train_spy_bnd["BND"].values
train_6040_r  = train_6040_d.mean() * 252
train_6040_v  = train_6040_d.std()  * np.sqrt(252)
train_6040_sh = (train_6040_r - RISK_FREE) / train_6040_v

print(f"\n  Optimized weights (train):")
for t, w in zip(OPT_TICKERS, train_weights):
    print(f"    {t:<12} {w*100:5.1f}%")
print(f"\n  Train Sharpe: {train_sharpe:.3f}  "
      f"Return: {train_ann_ret*100:.1f}%  Vol: {train_ann_vol*100:.1f}%")


# ── STEP 3: evaluate on TEST period ──────────────────────────────────────────

print("\n" + "="*72)
print(" STEP 3 — Out-of-sample evaluation (2020–2026)")
print("="*72)

# Optimized portfolio on test
test_port_d = pd.Series(test_opt.values @ train_weights, index=test_opt.index)

# Equal weight on test (same 5 assets)
test_ew_d = pd.Series(test_opt.values @ ew_w, index=test_opt.index)

# SPY only
test_spy_d = test_full["SPY"]

# 60/40 VTI/BND — use VTI from test_opt, BND from test_full
test_vti_idx = OPT_TICKERS.index("VTI")
test_6040_d  = pd.Series(
    0.6 * test_opt["VTI"].values + 0.4 * test_full["BND"].values,
    index=test_opt.index,
)

# Metrics
def row(name, series):
    ann_r, ann_v, sh, mdd, cum = portfolio_metrics(series)
    return {
        "Portfolio":   name,
        "Ann Return":  f"{ann_r*100:+.1f}%",
        "Ann Vol":     f"{ann_v*100:.1f}%",
        "Sharpe":      f"{sh:.3f}",
        "Max Drawdown":f"{mdd*100:.1f}%",
        "Cum Return":  f"{cum*100:+.1f}%",
    }

train_table = pd.DataFrame([
    row("Optimized",  pd.Series(train_port_d)),
    row("Equal Weight", pd.Series(train_ew_d)),
    row("SPY",        train_spy_bnd["SPY"]),
    row("60/40",      pd.Series(train_6040_d)),
])
test_table = pd.DataFrame([
    row("Optimized",  test_port_d),
    row("Equal Weight", test_ew_d),
    row("SPY",        test_spy_d),
    row("60/40",      test_6040_d),
])

print_table("TRAIN PERIOD (2015–2019) Performance", train_table)
print_table("OUT-OF-SAMPLE TEST PERIOD (2020–2026) Performance", test_table)

# Sharpe degradation
test_opt_metrics = portfolio_metrics(test_port_d)
test_sharpe = test_opt_metrics[2]
print(f"\n  Sharpe degradation (optimized): "
      f"train={train_sharpe:.3f} → test={test_sharpe:.3f}  "
      f"(Δ={test_sharpe - train_sharpe:+.3f})")

# Beat benchmarks?
test_spy_sh   = portfolio_metrics(test_spy_d)[2]
test_ew_sh    = portfolio_metrics(test_ew_d)[2]
test_6040_sh  = portfolio_metrics(test_6040_d)[2]
print(f"\n  Out-of-sample Sharpe vs benchmarks:")
print(f"    Optimized   : {test_sharpe:.3f}")
print(f"    Equal Weight: {test_ew_sh:.3f}  → optimized {'✓ wins' if test_sharpe > test_ew_sh else '✗ loses'}")
print(f"    SPY         : {test_spy_sh:.3f}  → optimized {'✓ wins' if test_sharpe > test_spy_sh else '✗ loses'}")
print(f"    60/40       : {test_6040_sh:.3f}  → optimized {'✓ wins' if test_sharpe > test_6040_sh else '✗ loses'}")


# ── STEP 4: Monte Carlo using OUT-OF-SAMPLE parameters ───────────────────────

print("\n" + "="*72)
print(" STEP 4 — Monte Carlo (out-of-sample return distribution)")
print("="*72)

# Fit distribution from test period (not train)
test_mean_d = test_opt.mean().values
test_cov_d  = test_opt.cov().values

min_eig = np.linalg.eigvalsh(test_cov_d).min()
if min_eig < 0:
    test_cov_d += (-min_eig + 1e-10) * np.eye(len(OPT_TICKERS))

np.random.seed(42)

mc_rows = []
for horizon, label in zip(HORIZONS, HORIZON_LABELS):
    sim_assets = np.random.multivariate_normal(
        test_mean_d, test_cov_d, size=(N_SIM, horizon)
    )
    sim_port   = sim_assets @ train_weights        # apply train weights
    factors    = 1 + sim_port
    terminal   = INITIAL_INV * np.cumprod(factors, axis=1)[:, -1]

    p5, p50, p95 = np.percentile(terminal, [5, 50, 95])
    prob_loss = (terminal < INITIAL_INV).mean()
    prob_2x   = (terminal > INITIAL_INV * 2).mean()

    mc_rows.append({
        "Horizon":    label,
        "P5":         f"${p5:,.0f}",
        "Median":     f"${p50:,.0f}",
        "P95":        f"${p95:,.0f}",
        "Prob Loss":  f"{prob_loss*100:.1f}%",
        "Prob 2x":    f"{prob_2x*100:.1f}%",
    })
    print(f"  {label}: median=${p50:,.0f}  P5=${p5:,.0f}  P95=${p95:,.0f}  "
          f"P(loss)={prob_loss*100:.1f}%  P(2x)={prob_2x*100:.1f}%")

print_table(
    f"Monte Carlo — Out-of-Sample Parameters  (${INITIAL_INV:,} initial, 10k sims)",
    pd.DataFrame(mc_rows),
)


# ── STEP 5: chart ─────────────────────────────────────────────────────────────

print("\n" + "="*72)
print(" STEP 5 — Generating chart")
print("="*72)

fig, axes = plt.subplots(2, 1, figsize=(14, 11))

# ── Subplot 1: cumulative returns 2020–2026 ───────────────────────────────────
ax1 = axes[0]

series_map = {
    "Optimized (train weights)": test_port_d,
    "Equal Weight":               test_ew_d,
    "SPY":                        test_spy_d,
    "60/40 (VTI/BND)":           test_6040_d,
}
colors = ["steelblue", "darkorange", "seagreen", "crimson"]

for (label, s), color in zip(series_map.items(), colors):
    cum = (1 + s).cumprod()
    lw  = 2.5 if label.startswith("Optimized") else 1.5
    ax1.plot(s.index, cum, label=label, color=color, linewidth=lw)

ax1.axhline(1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
ax1.axvline(pd.Timestamp("2020-03-23"), color="gray", linewidth=1.0,
            linestyle="--", alpha=0.6, label="COVID trough (Mar 2020)")

ax1.set_ylabel("Cumulative Return (1 = initial)", fontsize=10)
ax1.set_title(
    "Out-of-Sample Cumulative Returns (2020–2026)\n"
    "Weights optimized on 2015–2019 train data only",
    fontsize=11, fontweight="bold",
)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}x"))
ax1.legend(fontsize=9, loc="upper left")
ax1.grid(alpha=0.3)

# ── Subplot 2: rolling 12-month Sharpe ────────────────────────────────────────
ax2 = axes[1]

window = 252
rf_daily = RISK_FREE / 252

for (label, s), color in zip(
    [("Optimized", test_port_d), ("SPY", test_spy_d)],
    ["steelblue", "seagreen"],
):
    excess = s - rf_daily
    roll_sh = (excess.rolling(window).mean() * 252) / \
              (s.rolling(window).std() * np.sqrt(252))
    lw = 2.2 if label == "Optimized" else 1.5
    ax2.plot(roll_sh.index, roll_sh, label=f"{label} (12-mo rolling Sharpe)",
             color=color, linewidth=lw)

ax2.axhline(0, color="black", linewidth=0.8, alpha=0.5)
ax2.axhline(1, color="gray",  linewidth=0.8, linestyle="--", alpha=0.4, label="Sharpe = 1")

ax2.fill_between(test_port_d.index,
    (test_port_d - rf_daily).rolling(window).mean() * 252 /
    (test_port_d.rolling(window).std() * np.sqrt(252)),
    0,
    where=(((test_port_d - rf_daily).rolling(window).mean() * 252 /
            (test_port_d.rolling(window).std() * np.sqrt(252))) > 0),
    alpha=0.08, color="steelblue",
)

ax2.set_ylabel("Rolling 12-Month Sharpe Ratio", fontsize=10)
ax2.set_title(
    "Rolling 12-Month Sharpe — Optimized vs SPY (2020–2026)",
    fontsize=11, fontweight="bold",
)
ax2.legend(fontsize=9, loc="lower left")
ax2.grid(alpha=0.3)

# Shared x formatting
import matplotlib.dates as mdates
for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

fig.suptitle(
    "Walk-Forward Validation — Constrained Max Sharpe Portfolio\n"
    f"Train: 2015–2019  |  Test (out-of-sample): 2020–2026  |  "
    f"Train Sharpe: {train_sharpe:.2f}  →  Test Sharpe: {test_sharpe:.2f}",
    fontsize=12, fontweight="bold",
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "walk_forward_results.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved → analysis/walk_forward_results.png")

print("\n" + "="*72)
print(" Done.")
print("="*72)
