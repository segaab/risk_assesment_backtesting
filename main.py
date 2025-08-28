# app.py
# Low-Cap Crypto Crash Simulator & Backtest (Long + Short) with Early Detection
# - CoinGecko Pro key hard-coded (per request)
# - Monte Carlo crash probabilities
# - ECDS early detection
# - 3 long strategies + 3 short-entry strategies
# - Decision logs and performance metrics
# - KeyError fix in summarize_performance

import streamlit as st
import pandas as pd
import numpy as np
import requests
from typing import Dict
from scipy.stats import norm

# ---------------------------
# Hard-coded CoinGecko Pro key (as requested)
# ---------------------------
COINGECKO_API_KEY = "CG-chRgqiH9ab4zsFTm2Zvst82a"
HEADERS: Dict[str, str] = {"x-cg-pro-api-key": COINGECKO_API_KEY} if COINGECKO_API_KEY else {}
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

st.set_page_config(page_title="Low-Cap Crypto Crash Backtest (Long + Short)", layout="wide")
st.title("Low-Cap Crypto Sector Crash Simulator & Backtest (Long + Short)")

st.markdown("""
This dashboard simulates sector crash risk using liquidity and imbalance dynamics, and backtests:
- 3 long-side risk-mitigation strategies (reduce/exit when risk rises).
- 3 short-entry strategies with an early detection system (ECDS).

Data: CoinGecko API (Pro header included). Research tool only; not financial advice.
""")

# ---------------------------
# Data helpers
# ---------------------------
@st.cache_data(show_spinner=False, ttl=60*30)
def cg_coin_market_chart(coin_id: str, vs_currency: str, days: int):
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days, "interval": "daily"}
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    def to_df(lst, col):
        df = pd.DataFrame(lst, columns=["ts_ms", col])
        df["date"] = pd.to_datetime(df["ts_ms"], unit="ms").dt.tz_localize(None)
        return df[["date", col]]
    prices = to_df(data.get("prices", []), "price")
    volumes = to_df(data.get("total_volumes", []), "volume")
    mktcap = to_df(data.get("market_caps", []), "market_cap")
    df = prices.merge(volumes, on="date", how="outer").merge(mktcap, on="date", how="outer")
    df.sort_values("date", inplace=True)
    return df

def derive_liquidity_and_imbalance(df: pd.DataFrame, vol_window: int = 14, flow_window: int = 7) -> pd.DataFrame:
    out = df.copy()
    out["ret"] = np.log(out["price"]).diff()
    out["volatility"] = out["ret"].rolling(vol_window).std()
    out["norm_volume"] = out["volume"].rolling(vol_window).mean()
    # Liquidity proxy: higher when volume is higher and volatility is lower
    out["liquidity"] = (out["norm_volume"] / (out["volatility"] + 1e-8)).fillna(method="bfill").fillna(0)
    # Imbalance proxy: recent signed drift
    out["imbalance"] = (out["ret"].rolling(flow_window).mean() - out["ret"].rolling(flow_window).median()).fillna(0)
    return out[["date","price","liquidity","imbalance","ret","volume","market_cap"]]

# ---------------------------
# Sidebar: data selection
# ---------------------------
st.sidebar.header("Data")
source = st.sidebar.selectbox("Source", ["CoinGecko (auto)", "Upload CSV"])

uploaded_df = None
if source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("CSV with columns: date, price, liquidity, imbalance", type="csv")
    if uploaded:
        uploaded_df = pd.read_csv(uploaded, parse_dates=["date"]).sort_values("date")

with st.sidebar.expander("Low-cap universe selection", expanded=True):
    vs_currency = st.selectbox("Quote currency", ["usd", "eur", "btc"], index=0)
    days_back = st.number_input("History (days)", min_value=30, max_value=1825, value=365, step=30)
    default_lowcaps = ["akash-network","celestia","injective-protocol","beam-2","sei-network","sui","near","render-token","optimism","arbitrum"]
    coin_id = st.text_input("CoinGecko coin id", value=default_lowcaps[0])

if source == "CoinGecko (auto)":
    try:
        raw = cg_coin_market_chart(coin_id, vs_currency, int(days_back))
        df = derive_liquidity_and_imbalance(raw)
        st.success(f"Loaded {len(df)} daily points for {coin_id}.")
        st.dataframe(df.tail(10), use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        df = None
else:
    df = uploaded_df
    if df is not None:
        st.success(f"Loaded uploaded data with {len(df)} rows.")
        st.dataframe(df.tail(10), use_container_width=True)

if df is None or len(df) < 60:
    st.stop()

# ---------------------------
# Simulation parameters (from your model)
# ---------------------------
st.sidebar.header("Simulation Parameters")
H = st.sidebar.number_input("Simulation horizon (days)", value=10, min_value=2, max_value=180)
N_MC = st.sidebar.number_input("Monte Carlo paths", value=500, min_value=50, max_value=5000, step=50)
alpha_L = st.sidebar.number_input("alpha_L", value=0.05, step=0.01, format="%.2f")
beta_L = st.sidebar.number_input("beta_L", value=0.10, step=0.01, format="%.2f")
rho_I = st.sidebar.number_input("rho_I", value=0.95, step=0.01, format="%.2f")
mu_I = st.sidebar.number_input("mu_I", value=-0.02, step=0.01, format="%.2f")
sigma0 = st.sidebar.number_input("sigma0", value=0.03, step=0.005, format="%.3f")
kappa = st.sidebar.number_input("kappa", value=0.5, step=0.05, format="%.2f")
gamma = st.sidebar.number_input("gamma", value=0.5, step=0.05, format="%.2f")
crash_pct = st.sidebar.number_input("Crash % threshold", value=0.25, min_value=0.05, max_value=0.9, step=0.05, format="%.2f")
L_min_ratio = st.sidebar.number_input("Liquidity min ratio", value=0.10, min_value=0.01, max_value=1.0, step=0.01, format="%.2f")
I_crit = st.sidebar.number_input("Imbalance threshold", value=0.20, min_value=0.01, max_value=2.0, step=0.01, format="%.2f")
persist_days = st.sidebar.number_input("Persistence days (liquidity crash)", value=3, min_value=1, max_value=30)

def run_hybrid_simulation(P0, L0, I0, H, params):
    alpha_L, beta_L, rho_I, mu_I, sigma0, kappa, gamma = params
    P = P0
    L = max(1e-6, L0)
    I = np.clip(I0, -1, 1)
    path = []
    for _ in range(int(H)):
        delta_flow = 0.0
        r_prev = 0.0
        L_prop = L * (1 + alpha_L * delta_flow - beta_L * (abs(r_prev) > 0.05))
        shock = 0.0
        I_prop = rho_I * I + shock + np.random.randn() * 0.01
        gL = 1 + kappa * (1.0 / (L_prop + 1e-6)) ** gamma
        r = mu_I * I_prop + sigma0 * gL * np.random.randn()
        P *= np.exp(r)
        L = max(1e-6, L_prop)
        I = np.clip(I_prop, -1, 1)
        path.append([P, L, I, r])
    return np.array(path)

# ---------------------------
# Rolling crash probabilities (forecast series)
# ---------------------------
with st.expander("Compute rolling crash probabilities", expanded=True):
    progress = st.progress(0, text="Running Monte Carlo backtest...")
    crash_probs = []
    idxs = list(range(0, len(df) - int(H)))
    for i, idx in enumerate(idxs):
        P0 = df["price"].iloc[idx]; L0 = df["liquidity"].iloc[idx]; I0 = df["imbalance"].iloc[idx]
        crash_count = 0
        for _ in range(int(N_MC)):
            sim = run_hybrid_simulation(P0, L0, I0, int(H), [alpha_L, beta_L, rho_I, mu_I, sigma0, kappa, gamma])
            crash_price = sim[:,0] <= P0 * (1 - crash_pct)
            crash_liq = (sim[:,1] <= L_min_ratio * L0) & (sim[:,2] <= -I_crit)
            if persist_days <= len(crash_liq):
                crash_liq_persist = np.convolve(crash_liq.astype(int), np.ones(int(persist_days), dtype=int), "valid") >= persist_days
            else:
                crash_liq_persist = np.array([False])
            if np.any(crash_price) or np.any(crash_liq_persist):
                crash_count += 1
        crash_probs.append(crash_count / float(N_MC))
        progress.progress((i + 1) / len(idxs), text=f"Running... {i+1}/{len(idxs)}")

    df_bt = df.iloc[: len(crash_probs)].copy()
    df_bt["pred_crash_prob"] = crash_probs
    st.line_chart(df_bt.set_index("date")[["pred_crash_prob"]], use_container_width=True)

# ---------------------------
# Early Detection System (composite score for shorts)
# ---------------------------
st.subheader("Early Crash Detection System (for short entries)")

with st.sidebar.expander("Early detection parameters", expanded=True):
    ed_vol_window = st.number_input("Vol window (days)", value=14, min_value=5, max_value=60)
    ed_liq_window = st.number_input("Liquidity slope window", value=10, min_value=3, max_value=60)
    ed_prob_window = st.number_input("Crash prob slope window", value=10, min_value=3, max_value=60)
    ed_weight_prob = st.number_input("Weight: crash prob", value=0.5, min_value=0.0, max_value=1.0, step=0.05)
    ed_weight_liq = st.number_input("Weight: liquidity inverse", value=0.3, min_value=0.0, max_value=1.0, step=0.05)
    ed_weight_mom = st.number_input("Weight: negative momentum", value=0.2, min_value=0.0, max_value=1.0, step=0.05)
    ed_threshold = st.number_input("ECDS trigger threshold", value=0.6, min_value=0.0, max_value=2.0, step=0.05)

df_ed = df_bt.copy()
df_ed["ret"] = np.log(df_ed["price"]).diff()
df_ed["vol"] = df_ed["ret"].rolling(int(ed_vol_window)).std()
df_ed["liq_inv"] = 1.0 / (df_ed["liquidity"] + 1e-8)
df_ed["liq_slope"] = df_ed["liquidity"].diff(int(ed_liq_window)) / (np.abs(df_ed["liquidity"].shift(int(ed_liq_window))) + 1e-8)
df_ed["prob_slope"] = df_ed["pred_crash_prob"].diff(int(ed_prob_window)).fillna(0)
df_ed["neg_mom"] = (-df_ed["ret"].rolling(5).mean()).clip(lower=0)  # short-term negative drift

# Normalize components to 0-1 (robust)
for c in ["pred_crash_prob", "liq_inv", "liq_slope", "prob_slope", "neg_mom", "vol"]:
    x = df_ed[c].replace([np.inf, -np.inf], np.nan).fillna(0)
    p1, p99 = np.nanpercentile(x, [1, 99]) if x.notna().sum() > 0 else (0.0, 1.0)
    scale = (p99 - p1) if (p99 - p1) != 0 else 1.0
    x = (x - p1) / (scale + 1e-9)
    df_ed[c + "_z"] = x.clip(0, 1)

# Composite Early Crash Detection Score (ECDS)
df_ed["ECDS"] = (
    ed_weight_prob * df_ed["pred_crash_prob_z"] +
    ed_weight_liq * df_ed["liq_inv_z"] +
    ed_weight_mom * df_ed["neg_mom_z"]
)
# Optional boost when vol and slopes are rising
df_ed["ECDS"] += 0.2 * df_ed["vol_z"] + 0.2 * df_ed["prob_slope_z"] + 0.1 * df_ed["liq_slope_z"]
st.line_chart(df_ed.set_index("date")[["ECDS"]], use_container_width=True)

# ---------------------------
# Long strategies (3)
# ---------------------------
st.subheader("Long-side strategies")
with st.sidebar.expander("Long strategy parameters", expanded=True):
    # Strategy 1
    T1 = st.number_input("S1: Crash prob threshold", value=0.30, min_value=0.0, max_value=1.0, step=0.05)
    q_liq = st.number_input("S1: Liquidity percentile q", value=0.20, min_value=0.0, max_value=1.0, step=0.05)
    # Strategy 2
    k_mom = st.number_input("S2: Momentum lookback", value=20, min_value=5, max_value=120, step=5)
    T2 = st.number_input("S2: Crash prob soft threshold", value=0.25, min_value=0.0, max_value=1.0, step=0.05)
    reduce_factor = st.number_input("S2: Size reduce factor", value=0.5, min_value=0.0, max_value=1.0, step=0.1)
    # Strategy 3
    T3 = st.number_input("S3: Posterior mean threshold", value=0.35, min_value=0.0, max_value=1.0, step=0.05)
    cred_level = st.number_input("S3: Credible level", value=0.95, min_value=0.5, max_value=0.999, step=0.01)

df_long = df_ed.copy()
df_long["mom_k"] = df_long["price"] / df_long["price"].shift(int(k_mom)) - 1.0
liq_thresh = df_long["liquidity"].quantile(q_liq)

def rolling_beta_posterior(probs: pd.Series, window: int = 30, a0: float = 1.0, b0: float = 1.0, cred: float = 0.95):
    means, lowers, uppers = [], [], []
    for i in range(len(probs)):
        lo = max(0, i - window + 1)
        p_slice = probs.iloc[lo : i + 1].fillna(0).clip(0, 1)
        succ = p_slice.sum()
        trials = len(p_slice)
        a = a0 + succ
        b = b0 + (trials - succ)
        mean = a / (a + b)
        var = (a * b) / (((a + b) ** 2) * (a + b + 1))
        z = norm.ppf(0.5 + cred / 2.0)
        se = np.sqrt(var)
        means.append(mean); lowers.append(mean - z * se); uppers.append(mean + z * se)
    return pd.DataFrame({"post_mean": means, "post_low": lowers, "post_high": uppers}, index=probs.index)

post_df = rolling_beta_posterior(df_long["pred_crash_prob"], window=30, cred=cred_level)

def build_long_positions(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    x["position_s1"] = 1.0
    x["position_s2"] = 0.0
    x["position_s3"] = 1.0
    # S1: Liquidity Guardrail
    x.loc[(x["pred_crash_prob"] > T1) & (x["liquidity"] <= liq_thresh), "position_s1"] = 0.0
    # S2: Momentum + Liquidity Filter
    x.loc[(x["mom_k"] > 0) & (x["liquidity"] > x["liquidity"].median()), "position_s2"] = 1.0
    x.loc[x["pred_crash_prob"] > T2, "position_s2"] *= reduce_factor
    # S3: Bayesian threshold
    x["post_mean"] = post_df["post_mean"].reindex(x.index).values
    x.loc[x["post_mean"] > T3, "position_s3"] = 0.0
    return x

long_bt = build_long_positions(df_long)

# ---------------------------
# Short-entry strategies (3)
# ---------------------------
st.subheader("Short-entry strategies (early detection)")

with st.sidebar.expander("Short strategy parameters", expanded=True):
    # Short Strategy A: ECDS threshold + confirmation
    A_threshold = st.number_input("A: ECDS threshold", value=ed_threshold, min_value=0.0, max_value=2.0, step=0.05)
    A_confirm_prob = st.number_input("A: Min crash prob", value=0.30, min_value=0.0, max_value=1.0, step=0.05)
    A_confirm_liq_q = st.number_input("A: Max liquidity percentile (thin)", value=0.30, min_value=0.0, max_value=1.0, step=0.05)

    # Short Strategy B: Microstructure breakdown (vol/liquidity regime shift)
    B_vol_z = st.number_input("B: Volatility z-threshold (0-1 scaled)", value=0.7, min_value=0.0, max_value=1.0, step=0.05)
    B_liq_inv_z = st.number_input("B: Inverse Liquidity z-threshold", value=0.7, min_value=0.0, max_value=1.0, step=0.05)
    B_prob_slope_z = st.number_input("B: Crash prob slope z-threshold", value=0.6, min_value=0.0, max_value=1.0, step=0.05)

    # Short Strategy C: Bayesian crash signal
    C_thresh = st.number_input("C: Posterior mean crash prob", value=0.40, min_value=0.0, max_value=1.0, step=0.05)
    C_cred = st.number_input("C: Require credible level (lower bound > T)? 1=yes,0=no", value=0, min_value=0, max_value=1)

liq_q_thresh_A = df_long["liquidity"].quantile(A_confirm_liq_q)

def build_short_positions(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    # Initialize with no short
    x["short_A"] = 0.0
    x["short_B"] = 0.0
    x["short_C"] = 0.0

    # A) ECDS threshold + confirmations
    cond_A = (x["ECDS"] >= A_threshold) & \
             (x["pred_crash_prob"] >= A_confirm_prob) & \
             (x["liquidity"] <= liq_q_thresh_A)
    x.loc[cond_A, "short_A"] = 1.0

    # B) Regime shift: high vol, thin liquidity, rising crash slope
    cond_B = (x["vol_z"] >= B_vol_z) & (x["liq_inv_z"] >= B_liq_inv_z) & (x["prob_slope_z"] >= B_prob_slope_z)
    x.loc[cond_B, "short_B"] = 1.0

    # C) Bayesian crash signal using posterior mean of crash prob
    x["post_mean"] = post_df["post_mean"].reindex(x.index).values
    if C_cred == 1 and "post_low" in post_df.columns:
        post_low = post_df["post_low"].reindex(x.index).values
        x.loc[(x["post_mean"] >= C_thresh) & (post_low >= C_thresh), "short_C"] = 1.0
    else:
        x.loc[(x["post_mean"] >= C_thresh), "short_C"] = 1.0

    return x

short_bt = build_short_positions(df_long)

# ---------------------------
# PnL computation (long and short)
# ---------------------------
def compute_pnl(long_df: pd.DataFrame, fee_bps: float = 5.0, borrow_bps_daily: float = 1.0):
    d = long_df.copy()
    d["ret"] = d["ret"].fillna(0.0)

    # Long strategies: PnL = position * ret - transaction cost
    for col in ["position_s1", "position_s2", "position_s3"]:
        pos = d[col].fillna(0.0).values
        pos_shift = np.roll(pos, 1); pos_shift[0] = 0.0
        turnover = np.abs(pos - pos_shift)
        tc = turnover * (fee_bps / 1e4)
        d[f"{col}_pnl"] = pos * d["ret"] - tc

    # Short strategies: PnL = position * (-ret) - transaction cost - borrow cost
    borrow_cost = borrow_bps_daily / 1e4
    for scol in ["short_A", "short_B", "short_C"]:
        pos = short_bt[scol].reindex(d.index).fillna(0.0).values
        pos_shift = np.roll(pos, 1); pos_shift[0] = 0.0
        turnover = np.abs(pos - pos_shift)
        tc = turnover * (fee_bps / 1e4)
        d[f"{scol}_pnl"] = pos * (-d["ret"]) - tc - pos * borrow_cost

    # Benchmark: buy-and-hold long
    d["bh_pnl"] = d["ret"]

    # Equity curves (log-returns cumulated -> exp). Keep same column names with "_pnl".
    eq = d.set_index("date")[[
        "position_s1_pnl", "position_s2_pnl", "position_s3_pnl",
        "short_A_pnl", "short_B_pnl", "short_C_pnl",
        "bh_pnl"
    ]].cumsum().apply(np.exp)
    return d, eq

bt_row, equity_curves = compute_pnl(long_bt)

st.subheader("Equity curves (normalized)")
st.line_chart(equity_curves, use_container_width=True)

# ---------------------------
# Decision log
# ---------------------------
def build_decision_log(d: pd.DataFrame) -> pd.DataFrame:
    log = d[[
        "date","price","liquidity","imbalance","ret",
        "pred_crash_prob","ECDS","vol","liq_inv","prob_slope","neg_mom",
        "position_s1","position_s2","position_s3"
    ]].copy()
    # Append short signals from short_bt aligned to index
    log["short_A"] = short_bt["short_A"].reindex(log.index).values
    log["short_B"] = short_bt["short_B"].reindex(log.index).values
    log["short_C"] = short_bt["short_C"].reindex(log.index).values
    # Forward crash realization over H
    fwd = d["price"].shift(-int(H)) / d["price"] - 1.0
    log[f"fwd_return_{int(H)}d"] = fwd
    log[f"fwd_crash_{int(H)}d"] = (fwd <= -crash_pct).astype(int)
    return log

decision_log = build_decision_log(long_bt)
with st.expander("Decision log (exportable)"):
    st.dataframe(decision_log.tail(20), use_container_width=True)
    st.download_button("Download decision log CSV", data=decision_log.to_csv(index=False), file_name="decision_log.csv", mime="text/csv")

# ---------------------------
# Performance summary (KeyError-safe)
# ---------------------------
def summarize_performance(d: pd.DataFrame, eq: pd.DataFrame) -> pd.DataFrame:
    ann_factor = 365
    # Use the columns that actually exist in both d and eq (with "_pnl" names)
    desired = [
        "position_s1_pnl", "position_s2_pnl", "position_s3_pnl",
        "short_A_pnl", "short_B_pnl", "short_C_pnl",
        "bh_pnl"
    ]
    pnl_cols = [c for c in desired if c in d.columns and c in eq.columns]

    rows = []
    for col in pnl_cols:
        r = d[col].fillna(0.0)
        cum = np.exp(r.cumsum().iloc[-1]) - 1.0
        vol = r.std() * np.sqrt(ann_factor)
        sharpe = (r.mean() * ann_factor) / (vol + 1e-12)

        # Equity curve uses the same column name in eq
        curve = eq[col]
        mdd = (curve.cummax() / curve - 1.0).max()

        rows.append({
            "strategy": col.replace("_pnl", ""),
            "cum_return": cum,
            "ann_vol": vol,
            "sharpe": sharpe,
            "max_drawdown": mdd
        })
    return pd.DataFrame(rows)

perf = summarize_performance(bt_row, equity_curves)
st.subheader("Performance summary (long and short)")
st.dataframe(
    perf.style.format({"cum_return": "{:.1%}", "ann_vol": "{:.2f}", "sharpe": "{:.2f}", "max_drawdown": "{:.1%}"}),
    use_container_width=True
)

st.info("Notes: Shorting small-cap crypto can be illiquid and costly. Borrow costs and slippage are simplified here.")
