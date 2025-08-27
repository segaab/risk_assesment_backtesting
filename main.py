# app.py
# Streamlit Low-Cap Crypto Sector Crash Simulator & Strategy Backtest
# - Optional CoinGecko Pro API key support
# - Ingests data from CoinGecko
# - Monte Carlo crash simulations
# - Three mitigation strategies and evaluation

import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime as dt
from typing import Dict
from scipy.stats import norm

st.set_page_config(page_title="Low-Cap Crypto Crash Backtest", layout="wide")
st.title("Low-Cap Crypto Sector Crash Simulator & Strategy Backtest")

st.markdown("""
Simulates sector-level crash risk for low-cap crypto assets using liquidity and imbalance dynamics, and backtests three decision strategies.

Data source: CoinGecko API. Provide an API key for CoinGecko Pro if you have one (optional).
""")

# ---------------------------
# API key management
# ---------------------------
with st.sidebar.expander("API settings", expanded=True):
    st.caption("Optional: Enter your CoinGecko Pro API key.")
    api_key_input = st.text_input("CoinGecko API key (x-cg-pro-api-key)", value="", type="password", help="Leave blank for public API.")
    if api_key_input:
        st.session_state["CG_API_KEY"] = api_key_input.strip()
        st.success("API key saved in session.")
    headers = {}
    if "CG_API_KEY" in st.session_state and st.session_state["CG_API_KEY"]:
        headers["x-cg-pro-api-key"] = st.session_state["CG_API_KEY"]

# ---------------------------
# CoinGecko helpers
# ---------------------------
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

@st.cache_data(show_spinner=False, ttl=60*30)
def cg_coin_market_chart(coin_id: str, vs_currency: str, days: int, headers: Dict[str, str]):
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days, "interval": "daily"}
    r = requests.get(url, params=params, headers=headers, timeout=30)
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
    out["liquidity"] = (out["norm_volume"] / (out["volatility"] + 1e-8)).fillna(method="bfill").fillna(0)
    out["flow"] = out["volume"].pct_change(periods=flow_window).fillna(0).clip(-5, 5)
    out["imbalance"] = (out["ret"].rolling(flow_window).mean() - out["ret"].rolling(flow_window).median()).fillna(0)
    return out[["date", "price", "liquidity", "imbalance", "ret", "volume", "market_cap"]]

# ---------------------------
# Data selection
# ---------------------------
st.sidebar.header("Data")
source = st.sidebar.selectbox("Source", ["CoinGecko (auto)", "Upload CSV"])

uploaded_df = None
if source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("CSV with columns: date, price, liquidity, imbalance", type="csv")
    if uploaded:
        tmp = pd.read_csv(uploaded, parse_dates=["date"]).sort_values("date")
        uploaded_df = tmp

with st.sidebar.expander("Low-cap universe selection", expanded=True):
    vs_currency = st.selectbox("Quote currency", ["usd", "eur", "btc"], index=0)
    days_back = st.number_input("History (days)", min_value=30, max_value=1825, value=365, step=30)
    default_lowcaps = ["akash-network", "celestia", "injective-protocol", "beam-2", "sei-network", "sui", "near", "render-token", "optimism", "arbitrum"]
    coin_id = st.text_input("CoinGecko coin id", value=default_lowcaps[0])

if source == "CoinGecko (auto)":
    try:
        raw = cg_coin_market_chart(coin_id, vs_currency, int(days_back), headers)
        df = derive_liquidity_and_imbalance(raw)
        st.success(f"Loaded {len(df)} points for {coin_id}.")
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
# Simulation parameters
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
persist_days = st.sidebar.number_input("Persistence days", value=3, min_value=1, max_value=30)

def run_hybrid_simulation(P0, L0, I0, H, params):
    alpha_L, beta_L, rho_I, mu_I, sigma0, kappa, gamma = params
    P = P0
    L = max(1e-6, L0)
    I = np.clip(I0, -1, 1)
    path = []
    for _ in range(H):
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

# Rolling crash probabilities
with st.expander("Compute rolling crash probabilities", expanded=True):
    progress = st.progress(0, text="Running Monte Carlo backtest...")
    crash_probs = []
    idxs = range(0, len(df) - int(H))
    for i, idx in enumerate(idxs):
        P0 = df["price"].iloc[idx]; L0 = df["liquidity"].iloc[idx]; I0 = df["imbalance"].iloc[idx]
        crash_count = 0
        for _ in range(int(N_MC)):
            sim = run_hybrid_simulation(P0, L0, I0, int(H), [alpha_L, beta_L, rho_I, mu_I, sigma0, kappa, gamma])
            crash_price = sim[:, 0] <= P0 * (1 - crash_pct)
            crash_liq = (sim[:, 1] <= L_min_ratio * L0) & (sim[:, 2] <= -I_crit)
            if persist_days <= len(crash_liq):
                crash_liq_persist = np.convolve(crash_liq.astype(int), np.ones(int(persist_days), dtype=int), "valid") >= persist_days
            else:
                crash_liq_persist = np.array([False])
            if np.any(crash_price) or np.any(crash_liq_persist):
                crash_count += 1
        crash_probs.append(crash_count / float(N_MC))
        progress.progress((i + 1) / (len(df) - int(H)), text=f"Running... {i+1}/{len(df)-int(H)}")

    df_bt = df.iloc[: len(crash_probs)].copy()
    df_bt["pred_crash_prob"] = crash_probs
    st.line_chart(df_bt.set_index("date")[["pred_crash_prob"]], use_container_width=True)

# Strategies
st.subheader("Strategies")
st.markdown("""
1) Liquidity Guardrail
2) Momentum + Liquidity Filter
3) Bayesian Crash-Probability Threshold
""")

with st.sidebar.expander("Strategy parameters", expanded=True):
    T1 = st.number_input("S1: Crash prob threshold", value=0.30, min_value=0.0, max_value=1.0, step=0.05)
    q_liq = st.number_input("S1: Liquidity percentile q", value=0.20, min_value=0.0, max_value=1.0, step=0.05)
    k_mom = st.number_input("S2: Momentum lookback (days)", value=20, min_value=5, max_value=120, step=5)
    T2 = st.number_input("S2: Crash prob soft threshold", value=0.25, min_value=0.0, max_value=1.0, step=0.05)
    reduce_factor = st.number_input("S2: Size reduce factor", value=0.5, min_value=0.0, max_value=1.0, step=0.1)
    T3 = st.number_input("S3: Posterior mean threshold", value=0.35, min_value=0.0, max_value=1.0, step=0.05)
    cred_level = st.number_input("S3: Credible level", value=0.95, min_value=0.5, max_value=0.999, step=0.01)

df_tmp = df_bt.copy()
df_tmp["ret"] = np.log(df_tmp["price"]).diff()
df_tmp["mom_k"] = df_tmp["price"] / df_tmp["price"].shift(int(k_mom)) - 1.0
liq_thresh = df_tmp["liquidity"].quantile(q_liq)

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
        from math import sqrt
        var = (a * b) / (((a + b) ** 2) * (a + b + 1))
        z = norm.ppf(0.5 + cred / 2.0)
        se = np.sqrt(var)
        means.append(mean); lowers.append(mean - z * se); uppers.append(mean + z * se)
    return pd.DataFrame({"post_mean": means, "post_low": lowers, "post_high": uppers}, index=probs.index)

post_df = rolling_beta_posterior(df_tmp["pred_crash_prob"], window=30, cred=cred_level)

def backtest_strategies(df_in: pd.DataFrame) -> pd.DataFrame:
    d = df_in.copy()
    d["position_s1"] = 1.0
    d["position_s2"] = 0.0
    d["position_s3"] = 1.0
    d.loc[(d["pred_crash_prob"] > T1) & (d["liquidity"] <= liq_thresh), "position_s1"] = 0.0
    d.loc[(d["mom_k"] > 0) & (d["liquidity"] > d["liquidity"].median()), "position_s2"] = 1.0
    d.loc[d["pred_crash_prob"] > T2, "position_s2"] *= reduce_factor
    d["post_mean"] = post_df["post_mean"].reindex(d.index).values
    d.loc[d["post_mean"] > T3, "position_s3"] = 0.0
    return d

bt = backtest_strategies(df_tmp)

def compute_pnl(bt_df: pd.DataFrame, fee_bps: float = 5.0):
    d = bt_df.copy()
    d["ret"] = d["ret"].fillna(0.0)
    for col in ["position_s1", "position_s2", "position_s3"]:
        pos = d[col].fillna(0.0).values
        pos_shift = np.roll(pos, 1); pos_shift[0] = 0.0
        turnover = np.abs(pos - pos_shift)
        cost = turnover * (fee_bps / 1e4)
        d[f"{col}_pnl"] = pos * d["ret"] - cost
    d["bh_pnl"] = d["ret"]
    eq = d.set_index("date")[["position_s1_pnl", "position_s2_pnl", "position_s3_pnl", "bh_pnl"]].cumsum().apply(np.exp)
    return d, eq

bt_row, equity_curves = compute_pnl(bt)

st.subheader("Equity curves (normalized)")
st.line_chart(equity_curves, use_container_width=True)

def build_decision_log(bt_df: pd.DataFrame) -> pd.DataFrame:
    log = bt_df[["date","price","liquidity","imbalance","pred_crash_prob","mom_k","position_s1","position_s2","position_s3"]].copy()
    fwd = bt_df["price"].shift(-int(H)) / bt_df["price"] - 1.0
    log[f"fwd_return_{int(H)}d"] = fwd
    log[f"fwd_crash_{int(H)}d"] = (fwd <= -crash_pct).astype(int)
    return log

decision_log = build_decision_log(bt_row)
with st.expander("Decision log (exportable)"):
    st.dataframe(decision_log.tail(20), use_container_width=True)
    st.download_button("Download decision log CSV", data=decision_log.to_csv(index=False), file_name="decision_log.csv", mime="text/csv")

def summarize_performance(d: pd.DataFrame, eq: pd.DataFrame) -> pd.DataFrame:
    ann_factor = 365
    pnl_cols = ["position_s1_pnl", "position_s2_pnl", "position_s3_pnl", "bh_pnl"]
    rows = []
    for col in pnl_cols:
        r = d[col].fillna(0.0)
        cum = np.exp(r.cumsum().iloc[-1]) - 1.0
        vol = r.std() * np.sqrt(ann_factor)
        sharpe = (r.mean() * ann_factor) / (vol + 1e-12)
        curve = eq[col]  # <-- fixed KeyError here
        mdd = (curve.cummax() / curve - 1.0).max()
        rows.append({"strategy": col.replace("_pnl",""), "cum_return": cum, "ann_vol": vol, "sharpe": sharpe, "max_drawdown": mdd})
    return pd.DataFrame(rows)

perf = summarize_performance(bt_row, equity_curves)
st.subheader("Strategy performance summary")
st.dataframe(perf.style.format({"cum_return": "{:.1%}", "ann_vol": "{:.2f}", "sharpe": "{:.2f}", "max_drawdown": "{:.1%}"}), use_container_width=True)

st.info("This tool is for research only and not financial advice.")
