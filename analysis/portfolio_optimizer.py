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
RISK_FREE   = 0.05
N_PORTFOLIOS = 10_000
N_SIMULATIONS = 10_000
HORIZONS      = [252, 756, 1260]
HORIZON_LABELS = ["1Y", "3Y", "5Y"]
INITIAL = 10_000


# ── helpers ────────────────────────────────────────────────────────────────────

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


def print_table(title, df, floatfmt=".4f"):
    print(f"\n{'='*65}")
    print(f" {title}")
    print(f"{'='*65}")
    str_df = df.copy()
    for col in str_df.select_dtypes(include="float").columns:
        str_df[col] = str_df[col].map(lambda x: f"{x:{floatfmt}}")
    col_w = [max(len(c), str_df[c].astype(str).str.len().max()) for c in str_df.columns]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    print(fmt.format(*str_df.columns))
    print("  ".join("-" * w for w in col_w))
    for _, row in str_df.iterrows():
        print(fmt.format(*row.astype(str)))
    print(f"  ({len(df)} rows)")


def portfolio_stats(w, returns_vec, cov_matrix):
    ret = w @ returns_vec
    vol = np.sqrt(w @ cov_matrix @ w)
    sharpe = (ret - RISK_FREE) / vol
    return ret, vol, sharpe


def monte_carlo(weights, returns_vec, cov_matrix, label_suffix=""):
    """Run Monte Carlo simulation and return paths dict keyed by horizon label."""
    daily_means = returns_vec / 252
    daily_cov   = cov_matrix / 252
    paths = {}
    for horizon, label in zip(HORIZONS, HORIZON_LABELS):
        sim_returns = np.random.multivariate_normal(
            daily_means, daily_cov, size=(N_SIMULATIONS, horizon)
        )
        port_daily  = sim_returns @ weights
        cum_returns = np.cumprod(1 + port_daily, axis=1)
        terminal    = INITIAL * cum_returns[:, -1]
        paths[label] = {"terminal": terminal, "paths": INITIAL * cum_returns}
    return paths


def print_mc_table(label, paths_dict):
    for hl in HORIZON_LABELS:
        terminal  = paths_dict[hl]["terminal"]
        p5, p25, p50, p75, p95 = np.percentile(terminal, [5, 25, 50, 75, 95])
        prob_loss = (terminal < INITIAL).mean() * 100
        prob_2x   = (terminal > INITIAL * 2).mean() * 100
        pct_df = pd.DataFrame({
            "Metric": ["P5", "P25", "Median (P50)", "P75", "P95", "Prob Loss %", "Prob 2x %"],
            "Value":  [f"${p5:,.0f}", f"${p25:,.0f}", f"${p50:,.0f}",
                       f"${p75:,.0f}", f"${p95:,.0f}",
                       f"{prob_loss:.1f}%", f"{prob_2x:.1f}%"],
        })
        print_table(f"{label} — {hl} Horizon (initial ${INITIAL:,.0f})", pct_df)


# ── STEP 1: load data + build matrices ────────────────────────────────────────

print("\n" + "="*65)
print(" STEP 1 — Loading data from Snowflake + Constrained Optimisation")
print("="*65)

con = get_conn()
analytics = pd.read_sql("""
    SELECT ticker, annualized_return, annualized_vol
    FROM fct_portfolio_analytics
    WHERE period_label = 'all'
    ORDER BY ticker
""", con)
analytics.columns = analytics.columns.str.lower()

corr_df = pd.read_sql("""
    SELECT ticker_a, ticker_b, correlation
    FROM fct_correlation_matrix
    ORDER BY ticker_a, ticker_b
""", con)
corr_df.columns = corr_df.columns.str.lower()
con.close()

tickers = sorted(analytics["ticker"].tolist())
n   = len(tickers)
idx = {t: i for i, t in enumerate(tickers)}

returns_vec = analytics.set_index("ticker").loc[tickers, "annualized_return"].values
vol_vec     = analytics.set_index("ticker").loc[tickers, "annualized_vol"].values

corr_matrix = np.eye(n)
for _, row in corr_df.iterrows():
    if row["ticker_a"] in idx and row["ticker_b"] in idx:
        i, j = idx[row["ticker_a"]], idx[row["ticker_b"]]
        corr_matrix[i, j] = row["correlation"]
        corr_matrix[j, i] = row["correlation"]

cov_matrix = corr_matrix * np.outer(vol_vec, vol_vec)
min_eig = np.linalg.eigvalsh(cov_matrix).min()
if min_eig < 0:
    cov_matrix += (-min_eig + 1e-8) * np.eye(n)

print(f"  Tickers : {tickers}")
print(f"  Cov matrix min eigenvalue: {np.linalg.eigvalsh(cov_matrix).min():.6f}")

# ── bounds definition ──────────────────────────────────────────────────────────
EQUITY_TICKERS = {"VTI", "VOO", "SPY", "QQQ", "IWM"}

bounds_lower = np.zeros(n)
bounds_upper = np.ones(n)

SPECIFIC_BOUNDS = {
    "BTC-USD":  (0.00, 0.25),
    "ETH-USD":  (0.00, 0.15),
    "GLD":      (0.00, 0.25),
    "TLT":      (0.00, 0.10),
    "BND":      (0.00, 0.10),
    "VTI":      (0.10, 0.30),
}
EQUITY_MAX = 0.30

for t in tickers:
    i = idx[t]
    if t in SPECIFIC_BOUNDS:
        bounds_lower[i], bounds_upper[i] = SPECIFIC_BOUNDS[t]
    elif t in EQUITY_TICKERS:
        bounds_upper[i] = EQUITY_MAX


def get_constrained_max_sharpe(returns_vec, cov_matrix, tickers):
    """SLSQP minimisation of negative Sharpe with per-asset bounds."""
    def neg_sharpe(w):
        ret = w @ returns_vec
        vol = np.sqrt(w @ cov_matrix @ w)
        return -(ret - RISK_FREE) / (vol + 1e-10)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    scipy_bounds = [(bounds_lower[i], bounds_upper[i]) for i in range(n)]
    x0 = np.clip(np.ones(n) / n, bounds_lower, bounds_upper)
    x0 /= x0.sum()

    result = minimize(
        neg_sharpe, x0,
        method="SLSQP",
        bounds=scipy_bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    return result.x


con_ms_weights = get_constrained_max_sharpe(returns_vec, cov_matrix, tickers)
con_ret, con_vol, con_sharpe = portfolio_stats(con_ms_weights, returns_vec, cov_matrix)
print(f"  Constrained Max Sharpe solved: return={con_ret:.4f}, vol={con_vol:.4f}, sharpe={con_sharpe:.4f}")


# ── STEP 2: unconstrained efficient frontier ───────────────────────────────────

print("\n" + "="*65)
print(" STEP 2 — Unconstrained Efficient Frontier (10,000 portfolios)")
print("="*65)

np.random.seed(42)
w_unc = np.random.dirichlet(np.ones(n), size=N_PORTFOLIOS)

unc_returns  = w_unc @ returns_vec
unc_vols     = np.sqrt(np.einsum("ij,jk,ik->i", w_unc, cov_matrix, w_unc))
unc_sharpes  = (unc_returns - RISK_FREE) / unc_vols

results_unc = pd.DataFrame(w_unc, columns=tickers)
results_unc["return"]     = unc_returns
results_unc["volatility"] = unc_vols
results_unc["sharpe"]     = unc_sharpes

ms_port  = results_unc.loc[results_unc["sharpe"].idxmax()]
mv_port  = results_unc.loc[results_unc["volatility"].idxmin()]
mr_port  = results_unc.loc[results_unc["return"].idxmax()]

eq_w     = np.ones(n) / n
eq_ret, eq_vol, eq_sharpe = portfolio_stats(eq_w, returns_vec, cov_matrix)

results_unc.to_csv(OUT / "efficient_frontier.csv", index=False)
print(f"  Saved {len(results_unc):,} portfolios → analysis/efficient_frontier.csv")


# ── STEP 3: constrained efficient frontier ────────────────────────────────────

print("\n" + "="*65)
print(" STEP 3 — Constrained Frontier (10,000 portfolios, clip+renorm)")
print("="*65)

def sample_constrained(n_samples):
    """Dirichlet sample → clip to [lower, upper] → renorm → reject if bounds violated."""
    collected, attempts = [], 0
    rng = np.random.default_rng(seed=99)
    while len(collected) < n_samples and attempts < n_samples * 50:
        w = rng.dirichlet(np.ones(n))
        w = np.clip(w, bounds_lower, bounds_upper)
        total = w.sum()
        if total == 0:
            attempts += 1
            continue
        w /= total
        # Validate all bounds hold after renorm (within tolerance)
        if np.all(w >= bounds_lower - 1e-6) and np.all(w <= bounds_upper + 1e-6):
            collected.append(w)
        attempts += 1
    print(f"  Collected {len(collected):,} valid portfolios from {attempts:,} attempts")
    return np.array(collected)

w_con = sample_constrained(N_PORTFOLIOS)

con_returns = w_con @ returns_vec
con_vols    = np.sqrt(np.einsum("ij,jk,ik->i", w_con, cov_matrix, w_con))
con_sharpes = (con_returns - RISK_FREE) / con_vols

results_con = pd.DataFrame(w_con, columns=tickers)
results_con["return"]     = con_returns
results_con["volatility"] = con_vols
results_con["sharpe"]     = con_sharpes

cms_port = results_con.loc[results_con["sharpe"].idxmax()]  # simulated best
cmv_port = results_con.loc[results_con["volatility"].idxmin()]
cmr_port = results_con.loc[results_con["return"].idxmax()]


# ── STEP 4: comparison table + weights ────────────────────────────────────────

print("\n" + "="*65)
print(" STEP 4 — Portfolio Comparison")
print("="*65)

ms_ret, ms_vol, ms_sh = portfolio_stats(ms_port[tickers].values, returns_vec, cov_matrix)

comparison = pd.DataFrame({
    "Portfolio": [
        "Unconstrained Max Sharpe",
        "Constrained Max Sharpe (SLSQP)",
        "Min Vol (unconstrained)",
        "Equal Weight",
    ],
    "Return":     [ms_ret,    con_ret,    mv_port["return"],     eq_ret],
    "Volatility": [ms_vol,    con_vol,    mv_port["volatility"], eq_vol],
    "Sharpe":     [ms_sh,     con_sharpe, mv_port["sharpe"],     eq_sharpe],
})
print_table("Portfolio Comparison", comparison, floatfmt=".4f")

# Constrained weights
cw_df = pd.DataFrame({
    "Ticker":  tickers,
    "Weight":  con_ms_weights,
    "Bound":   [f"[{bounds_lower[i]:.0%}, {bounds_upper[i]:.0%}]" for i in range(n)],
}).sort_values("Weight", ascending=False).reset_index(drop=True)
cw_df["Weight_pct"] = (cw_df["Weight"] * 100).map(lambda x: f"{x:.2f}%")
print_table("Constrained Max Sharpe Weights (SLSQP)", cw_df[["Ticker", "Weight_pct", "Bound"]])


# ── STEP 5: Monte Carlo — unconstrained and constrained ───────────────────────

print("\n" + "="*65)
print(" STEP 5 — Monte Carlo Simulation")
print("="*65)

np.random.seed(77)
unc_paths = monte_carlo(ms_port[tickers].values, returns_vec, cov_matrix)
con_paths = monte_carlo(con_ms_weights,           returns_vec, cov_matrix)

print("\n--- Unconstrained Max Sharpe ---")
print_mc_table("Unconstrained Max Sharpe", unc_paths)

print("\n--- Constrained Max Sharpe (SLSQP) ---")
print_mc_table("Constrained Max Sharpe", con_paths)

# Side-by-side summary
print("\n" + "="*65)
print(" Monte Carlo Side-by-Side Summary")
print("="*65)
for hl in HORIZON_LABELS:
    u_t = unc_paths[hl]["terminal"]
    c_t = con_paths[hl]["terminal"]
    rows = []
    for label, terminal in [("Unconstrained", u_t), ("Constrained", c_t)]:
        p5, _, p50, _, p95 = np.percentile(terminal, [5, 25, 50, 75, 95])
        rows.append({
            "Portfolio":   label,
            "Horizon":     hl,
            "P5":          f"${p5:,.0f}",
            "Median":      f"${p50:,.0f}",
            "P95":         f"${p95:,.0f}",
            "Prob Loss":   f"{(terminal < INITIAL).mean()*100:.1f}%",
            "Prob 2x":     f"{(terminal > INITIAL*2).mean()*100:.1f}%",
        })
    print_table(f"Side-by-Side — {hl}", pd.DataFrame(rows))

# Save 500 sampled paths (constrained)
mc_rows = []
sample_idx = np.random.choice(N_SIMULATIONS, 500, replace=False)
for label in HORIZON_LABELS:
    for sim_i, path in enumerate(con_paths[label]["paths"][sample_idx]):
        for day_i, val in enumerate(path):
            mc_rows.append({"horizon": label, "simulation": sim_i,
                            "day": day_i + 1, "value": round(val, 2)})
pd.DataFrame(mc_rows).to_csv(OUT / "monte_carlo_paths.csv", index=False)
print(f"\n  Saved constrained MC paths → analysis/monte_carlo_paths.csv")


# ── STEP 6: charts ────────────────────────────────────────────────────────────

print("\n" + "="*65)
print(" STEP 6 — Charts")
print("="*65)

# Chart 1 (original): unchanged
fig, ax = plt.subplots(figsize=(12, 7))
sc = ax.scatter(results_unc["volatility"], results_unc["return"],
                c=results_unc["sharpe"], cmap="RdYlGn",
                alpha=0.4, s=8, linewidths=0)
plt.colorbar(sc, ax=ax, label="Sharpe Ratio")
ax.scatter(ms_port["volatility"], ms_port["return"],
           marker="*", color="gold", s=400, zorder=5,
           edgecolors="black", linewidths=0.8, label="Max Sharpe")
ax.scatter(mv_port["volatility"], mv_port["return"],
           marker="D", color="royalblue", s=150, zorder=5,
           edgecolors="black", linewidths=0.8, label="Min Volatility")
ax.scatter(eq_vol, eq_ret,
           marker="o", color="darkorange", s=150, zorder=5,
           edgecolors="black", linewidths=0.8, label="Equal Weight")
ax.set_xlabel("Annualized Volatility", fontsize=12)
ax.set_ylabel("Annualized Return", fontsize=12)
ax.set_title("Efficient Frontier — 11 Asset Portfolio (10,000 Simulations)",
             fontsize=14, fontweight="bold")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "efficient_frontier.png", dpi=150)
plt.close(fig)
print("  Saved → analysis/efficient_frontier.png")

# Chart 2 (v2): unconstrained + constrained overlaid
fig, ax = plt.subplots(figsize=(13, 7))

# Unconstrained cloud (blue-green)
sc1 = ax.scatter(results_unc["volatility"], results_unc["return"],
                 c=results_unc["sharpe"], cmap="RdYlGn",
                 alpha=0.25, s=8, linewidths=0, label="_nolegend_")

# Constrained cloud (purple)
sc2 = ax.scatter(results_con["volatility"], results_con["return"],
                 c=results_con["sharpe"], cmap="PuRd",
                 alpha=0.35, s=8, linewidths=0, label="_nolegend_")

cb1 = plt.colorbar(sc1, ax=ax, fraction=0.03, pad=0.01)
cb1.set_label("Sharpe (Unconstrained)", fontsize=9)
cb2 = plt.colorbar(sc2, ax=ax, fraction=0.03, pad=0.05)
cb2.set_label("Sharpe (Constrained)", fontsize=9)

# Unconstrained max sharpe
ax.scatter(ms_port["volatility"], ms_port["return"],
           marker="*", color="gold", s=450, zorder=6,
           edgecolors="black", linewidths=0.8,
           label=f"Unconstrained Max Sharpe ({ms_sh:.2f})")
ax.annotate(f"Unc. Max Sharpe\n({ms_sh:.2f})",
            xy=(ms_port["volatility"], ms_port["return"]),
            xytext=(10, 12), textcoords="offset points", fontsize=8,
            arrowprops=dict(arrowstyle="-", color="black", lw=0.8))

# Constrained max sharpe (SLSQP)
ax.scatter(con_vol, con_ret,
           marker="*", color="deeppink", s=450, zorder=6,
           edgecolors="black", linewidths=0.8,
           label=f"Constrained Max Sharpe ({con_sharpe:.2f})")
ax.annotate(f"Con. Max Sharpe\n({con_sharpe:.2f})",
            xy=(con_vol, con_ret),
            xytext=(-70, -25), textcoords="offset points", fontsize=8,
            arrowprops=dict(arrowstyle="-", color="black", lw=0.8))

# Equal weight
ax.scatter(eq_vol, eq_ret,
           marker="o", color="darkorange", s=150, zorder=6,
           edgecolors="black", linewidths=0.8,
           label=f"Equal Weight ({eq_sharpe:.2f})")

ax.set_xlabel("Annualized Volatility", fontsize=12)
ax.set_ylabel("Annualized Return", fontsize=12)
ax.set_title("Efficient Frontier — Unconstrained vs Constrained (10,000 Portfolios Each)",
             fontsize=13, fontweight="bold")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.legend(fontsize=9, loc="upper left")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "efficient_frontier_v2.png", dpi=150)
plt.close(fig)
print("  Saved → analysis/efficient_frontier_v2.png")

# Chart 3: Monte Carlo (constrained)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (horizon, label) in zip(axes, zip(HORIZONS, HORIZON_LABELS)):
    paths    = con_paths[label]["paths"]
    terminal = con_paths[label]["terminal"]
    days     = np.arange(1, horizon + 1)
    sample200 = np.random.choice(N_SIMULATIONS, 200, replace=False)
    for path in paths[sample200]:
        ax.plot(days, path, alpha=0.04, color="mediumorchid", linewidth=0.5)
    pcts   = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    labels = ["P5", "P25", "P50 (Median)", "P75", "P95"]
    cols   = ["#d73027", "#fc8d59", "#1a9850", "#91cf60", "#4575b4"]
    stys   = ["--", "-", "-", "-", "--"]
    for pct, lbl, col, sty in zip(pcts, labels, cols, stys):
        ax.plot(days, pct, color=col, linewidth=2, linestyle=sty, label=lbl, zorder=3)
    ax.axhline(INITIAL, color="black", linewidth=1.2, linestyle=":", label="Initial $10k", zorder=4)
    ax.set_title(f"{label} Horizon\nMedian: ${np.median(terminal):,.0f}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Trading Days", fontsize=10)
    ax.set_ylabel("Portfolio Value ($)", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
fig.suptitle("Monte Carlo — Constrained Max Sharpe Portfolio (10,000 Paths)",
             fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(OUT / "monte_carlo.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved → analysis/monte_carlo.png")

print("\n" + "="*65)
print(" All steps complete.")
print("="*65)
