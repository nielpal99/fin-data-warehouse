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

OUT  = Path(__file__).parent
RISK_FREE = 0.05
HYSA_RATE = 0.045                           # 4.5% annual HYSA rate
HYSA_DAILY = (1 + HYSA_RATE) ** (1 / 252) - 1

WEIGHTS_SPEC = {
    "QQQ":     0.30,
    "GLD":     0.25,
    "BTC-USD": 0.25,
    "VTI":     0.10,
    "VOO":     0.10,
}
MONTHLY_CONTRIBUTION  = 500
HORIZONS              = [252,   756,   1260]
HORIZON_LABELS        = ["1Y",  "3Y",  "5Y"]
LUMP_AMOUNTS          = [6_000, 18_000, 30_000]   # 12 / 36 / 60 months × $500
N_SIMULATIONS         = 10_000
CONTRIB_INTERVAL      = 21   # every 21 trading days ≈ monthly


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


def mc_stats(terminal, invested, hysa_terminal):
    p5, p50, p95 = np.percentile(terminal, [5, 50, 95])
    return {
        "P5":              f"${p5:,.0f}",
        "Median":          f"${p50:,.0f}",
        "P95":             f"${p95:,.0f}",
        "Prob Loss vs HYSA": f"{(terminal < hysa_terminal).mean()*100:.1f}%",
        "Prob 2x":         f"{(terminal > invested * 2).mean()*100:.1f}%",
    }


def hysa_only_terminal(lump, horizon):
    """Full lump sum sits in HYSA for the entire horizon (deterministic)."""
    return lump * (1 + HYSA_RATE) ** (horizon / 252)


# ── STEP 1: pull data ──────────────────────────────────────────────────────────

print("\n" + "="*72)
print(" STEP 1 — Loading fct_daily_returns from Snowflake")
print("="*72)

tickers = sorted(WEIGHTS_SPEC.keys())
ticker_list = ", ".join(f"'{t}'" for t in tickers)

con = get_conn()
df = pd.read_sql(f"""
    SELECT date, ticker, daily_return
    FROM fct_daily_returns
    WHERE ticker IN ({ticker_list})
    ORDER BY date, ticker
""", con)
con.close()

df.columns = df.columns.str.lower()
df["date"] = pd.to_datetime(df["date"])

wide    = df.pivot(index="date", columns="ticker", values="daily_return").dropna()
wide    = wide[tickers]
weights = np.array([WEIGHTS_SPEC[t] for t in tickers])

ann_returns = wide.mean().values * 252
ann_vols    = wide.std().values  * np.sqrt(252)
cov_matrix  = wide.cov().values  * 252

daily_mean = ann_returns / 252
daily_cov  = cov_matrix  / 252

min_eig = np.linalg.eigvalsh(daily_cov).min()
if min_eig < 0:
    daily_cov += (-min_eig + 1e-10) * np.eye(len(tickers))

print(f"  Tickers      : {tickers}")
print(f"  Weights      : {weights}")
print(f"  Ann. returns : {dict(zip(tickers, np.round(ann_returns, 4)))}")
print(f"  Ann. vols    : {dict(zip(tickers, np.round(ann_vols, 4)))}")
print(f"  Trading days : {len(wide):,}")
print(f"  HYSA daily   : {HYSA_DAILY*100:.5f}%  ({HYSA_RATE*100:.1f}% annual)")


# ── STEP 2: simulation ────────────────────────────────────────────────────────

print("\n" + "="*72)
print(" STEP 2 — Running simulations (DCA+HYSA, Lump Sum, HYSA Only)")
print("="*72)

np.random.seed(42)

# Pre-build daily HYSA growth factor array (deterministic, same every sim)
# used to track cash balance between contributions
hysa_factor = 1 + HYSA_DAILY

all_results = {}

for horizon, label, lump in zip(HORIZONS, HORIZON_LABELS, LUMP_AMOUNTS):
    n_contribs = horizon // CONTRIB_INTERVAL
    total_dca  = n_contribs * MONTHLY_CONTRIBUTION

    # Simulate portfolio daily returns: shape (N_SIM, horizon)
    sim_asset_returns = np.random.multivariate_normal(
        daily_mean, daily_cov, size=(N_SIMULATIONS, horizon)
    )
    port_daily = sim_asset_returns @ weights      # (N_SIM, horizon)
    factors    = 1 + port_daily                   # (N_SIM, horizon)

    # ── Lump Sum ──────────────────────────────────────────────────────────────
    lump_paths    = lump * np.cumprod(factors, axis=1)
    lump_terminal = lump_paths[:, -1]

    # ── HYSA Only (deterministic — same value for all sims) ───────────────────
    hysa_value = hysa_only_terminal(lump, horizon)
    # For path plotting: full lump compounds in HYSA every day
    hysa_path = lump * hysa_factor ** np.arange(1, horizon + 1)

    # ── DCA + HYSA ────────────────────────────────────────────────────────────
    # Uninvested cash earns HYSA daily until the contribution date.
    # On each contribution day:
    #   - the month's $500 is deployed immediately into the portfolio
    #   - remaining cash balance (if any prior uncommitted cash) is also deployed
    #
    # Concretely: $500 is contributed monthly; between contribution dates the
    # investor earns HYSA on the *previously accumulated but not-yet-deployed*
    # portion.  Since all $500 is deployed each month on the contribution day,
    # the HYSA only applies during the inter-contribution window on that $500
    # from when it was "saved" to when it is invested.
    #
    # We model it as: the cash sits in HYSA from the previous contribution day
    # to the current one (20 trading days on average), then is invested.
    # The HYSA-boosted contribution on contribution day d (day index cd) is:
    #   effective_contribution = 500 * hysa_factor^(cd - prev_cd)
    # where prev_cd is the previous contribution day (or 0 for the first).
    #
    # Total wealth on any day d = portfolio_value(d) + cash_balance(d)
    # Between contribution days, the cash balance is 0 (all deployed).

    prefix_prod = np.cumprod(factors, axis=1)    # (N_SIM, horizon)

    dca_paths       = np.zeros((N_SIMULATIONS, horizon))
    running_inv_sum = np.zeros(N_SIMULATIONS)
    prev_contrib_day = -1

    for d in range(horizon):
        if d % CONTRIB_INTERVAL == 0:
            # Days since last contribution = d - prev_contrib_day
            days_in_hysa = d - prev_contrib_day        # ≥1; first time = d+1
            hysa_boost   = hysa_factor ** days_in_hysa  # HYSA earned on this $500
            effective_c  = MONTHLY_CONTRIBUTION * hysa_boost

            if d == 0:
                running_inv_sum += effective_c           # first contribution: factor=1
            else:
                running_inv_sum += effective_c / prefix_prod[:, d - 1]
            prev_contrib_day = d

        # portfolio value = running_inv_sum × prefix_prod (no separate cash balance
        # since all cash is deployed on contribution days)
        dca_paths[:, d] = running_inv_sum * prefix_prod[:, d]

    dca_terminal = dca_paths[:, -1]

    all_results[label] = {
        "lump":           lump,
        "total_dca":      total_dca,
        "lump_terminal":  lump_terminal,
        "dca_terminal":   dca_terminal,
        "lump_paths":     lump_paths,
        "dca_paths":      dca_paths,
        "hysa_value":     hysa_value,
        "hysa_path":      hysa_path,
    }

    print(f"  {label}: DCA+HYSA median=${np.median(dca_terminal):,.0f}  "
          f"Lump median=${np.median(lump_terminal):,.0f}  "
          f"HYSA only=${hysa_value:,.0f}")


# ── STEP 3: print comparison tables ───────────────────────────────────────────

print("\n" + "="*72)
print(" STEP 3 — Results")
print("="*72)

for label, lump in zip(HORIZON_LABELS, LUMP_AMOUNTS):
    r          = all_results[label]
    hysa_val   = r["hysa_value"]
    hysa_arr   = np.full(N_SIMULATIONS, hysa_val)  # deterministic benchmark

    rows = []
    for strategy, terminal, invested in [
        ("DCA + HYSA",  r["dca_terminal"],  r["total_dca"]),
        ("Lump Sum",    r["lump_terminal"], lump),
        ("HYSA Only",   hysa_arr,           lump),
    ]:
        row = {"Strategy": strategy, **mc_stats(terminal, invested, hysa_arr)}
        rows.append(row)

    print_table(
        f"{label} Horizon  (${lump:,} total capital  |  HYSA=${hysa_val:,.0f})",
        pd.DataFrame(rows),
    )


# ── STEP 4: crossover analysis ────────────────────────────────────────────────

print("\n" + "="*72)
print(" STEP 4 — Crossover: DCA+HYSA median = Lump Sum median")
print("="*72)
print("  (varying portfolio annual return while keeping historical covariance structure)")

np.random.seed(42)

crossover_returns = {}

for horizon, label, lump in zip(HORIZONS, HORIZON_LABELS, LUMP_AMOUNTS):
    n_contribs = horizon // CONTRIB_INTERVAL

    # Simulate a shared set of *unit* return paths (zero mean, historical cov)
    unit_asset = np.random.multivariate_normal(
        np.zeros(len(tickers)), daily_cov, size=(N_SIMULATIONS, horizon)
    )
    unit_port = unit_asset @ weights    # (N_SIM, horizon), zero-drift shocks

    # Sweep from 0.5% up to 50%. Crossover exists where portfolio return < HYSA
    # rate (~4.5%), so DCA+HYSA cash earns more than the portfolio while waiting.
    sweep_returns = np.arange(0.005, 0.51, 0.005)
    crossover_r   = None

    dca_meds  = []
    lump_meds = []

    for sweep_ann in sweep_returns:
        sweep_daily = sweep_ann / 252
        port_daily_swept = sweep_daily + unit_port
        factors_s        = 1 + port_daily_swept
        prefix_s         = np.cumprod(factors_s, axis=1)

        lump_med = np.median(lump * prefix_s[:, -1])

        running_s = np.zeros(N_SIMULATIONS)
        prev_cd   = -1
        for d in range(horizon):
            if d % CONTRIB_INTERVAL == 0:
                days_in_hysa = d - prev_cd
                eff_c        = MONTHLY_CONTRIBUTION * (hysa_factor ** days_in_hysa)
                if d == 0:
                    running_s += eff_c
                else:
                    running_s += eff_c / prefix_s[:, d - 1]
                prev_cd = d
        dca_med = np.median(running_s * prefix_s[:, -1])

        dca_meds.append(dca_med)
        lump_meds.append(lump_med)

    # Find where lump sum first surpasses DCA+HYSA (scan low→high)
    for i, (dm, lm, sr) in enumerate(zip(dca_meds, lump_meds, sweep_returns)):
        if lm > dm:
            # Interpolate for a finer estimate
            if i > 0:
                prev_gap  = dca_meds[i-1] - lump_meds[i-1]
                curr_gap  = dm - lm          # negative: lump now ahead
                frac      = prev_gap / (prev_gap - curr_gap)
                crossover_r = sweep_returns[i-1] + frac * 0.005
            else:
                crossover_r = sr
            break

    crossover_returns[label] = crossover_r

print()
for label in HORIZON_LABELS:
    cr = crossover_returns[label]
    if cr is not None:
        print(f"  {label}: Lump Sum beats DCA+HYSA when portfolio annual return > {cr*100:.1f}%")
        print(f"         (below {cr*100:.1f}%, HYSA cash yield > portfolio return → DCA+HYSA wins)")
    else:
        print(f"  {label}: DCA+HYSA leads even at 0.5% portfolio return (no crossover found in range)")

print()
print("  Interpretation: below these return thresholds the HYSA boost on")
print("  uninvested cash is enough to offset the compounding head-start of")
print("  deploying the full lump sum on day one.")


# ── STEP 5: chart ─────────────────────────────────────────────────────────────

print("\n" + "="*72)
print(" STEP 5 — Generating chart")
print("="*72)

fig, axes = plt.subplots(1, 3, figsize=(21, 7))
sample_idx = np.random.choice(N_SIMULATIONS, 200, replace=False)

pct_levels = [5, 50, 95]
pct_labels = ["P5", "Median", "P95"]
pct_styles = ["--", "-", "--"]
pct_widths = [1.2, 2.5, 1.2]

for ax, label, horizon, lump in zip(axes, HORIZON_LABELS, HORIZONS, LUMP_AMOUNTS):
    r    = all_results[label]
    days = np.arange(1, horizon + 1)
    cr   = crossover_returns[label]

    # ── 200 sampled paths ────────────────────────────────────────────────────
    for path in r["dca_paths"][sample_idx]:
        ax.plot(days, path, alpha=0.03, color="steelblue",  linewidth=0.4)
    for path in r["lump_paths"][sample_idx]:
        ax.plot(days, path, alpha=0.03, color="darkorange", linewidth=0.4)

    # ── Percentile bands ─────────────────────────────────────────────────────
    dca_pcts  = np.percentile(r["dca_paths"],  pct_levels, axis=0)
    lump_pcts = np.percentile(r["lump_paths"], pct_levels, axis=0)

    for pct_line, plbl, psty, pw in zip(dca_pcts, pct_labels, pct_styles, pct_widths):
        ax.plot(days, pct_line, color="steelblue", linewidth=pw, linestyle=psty,
                label=f"DCA+HYSA {plbl}" if psty == "-" else f"_DCA {plbl}")

    for pct_line, plbl, psty, pw in zip(lump_pcts, pct_labels, pct_styles, pct_widths):
        ax.plot(days, pct_line, color="darkorange", linewidth=pw, linestyle=psty,
                label=f"Lump Sum {plbl}" if psty == "-" else f"_Lump {plbl}")

    # ── HYSA Only path (deterministic) ───────────────────────────────────────
    ax.plot(days, r["hysa_path"], color="seagreen", linewidth=2.0,
            linestyle="-.", label="HYSA Only")

    # ── HYSA Only terminal horizontal line ───────────────────────────────────
    ax.axhline(r["hysa_value"], color="seagreen", linewidth=1.0,
               linestyle=":", alpha=0.7, zorder=4)

    # ── Formatting ───────────────────────────────────────────────────────────
    dca_med   = np.median(r["dca_terminal"])
    lump_med  = np.median(r["lump_terminal"])
    hysa_val  = r["hysa_value"]
    cr_str = f">{cr*100:.1f}%" if cr else "always"

    ax.set_title(
        f"{label} Horizon\n"
        f"DCA+HYSA: ${dca_med:,.0f}  |  Lump: ${lump_med:,.0f}  |  HYSA: ${hysa_val:,.0f}\n"
        f"Lump Sum wins when port. return {cr_str}",
        fontsize=9, fontweight="bold",
    )
    ax.set_xlabel("Trading Days", fontsize=9)
    ax.set_ylabel("Wealth ($)", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.3)

from matplotlib.patches import Patch
fig.legend(
    handles=[
        Patch(color="steelblue",  label="DCA + HYSA ($500/month, cash earns 4.5%)"),
        Patch(color="darkorange", label="Lump Sum (fully deployed day 1)"),
        Patch(color="seagreen",   label="HYSA Only (4.5% annual, no investment)"),
    ],
    loc="lower center", ncol=3, fontsize=10,
    bbox_to_anchor=(0.5, -0.05),
)

fig.suptitle(
    "DCA+HYSA vs Lump Sum vs HYSA Only\n"
    "Portfolio: QQQ 30% | GLD 25% | BTC 25% | VTI 10% | VOO 10%  "
    f"|  HYSA rate: {HYSA_RATE*100:.1f}%",
    fontsize=12, fontweight="bold",
)
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(OUT / "dca_hysa_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved → analysis/dca_hysa_comparison.png")

print("\n" + "="*72)
print(" All steps complete.")
print("="*72)
