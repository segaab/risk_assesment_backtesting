# ---------------------------
# Chunk 1: Data & Simulation Setup
# ---------------------------

import streamlit as st
import pandas as pd
import numpy as np
import requests
from typing import Dict
from scipy.stats import norm

# CoinGecko Pro API setup
COINGECKO_API_KEY = "CG-chRgqiH9ab4zsFTm2Zvst82a"
HEADERS: Dict[str, str] = {"x-cg-pro-api-key": COINGECKO_API_KEY} if COINGECKO_API_KEY else {}
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

st.set_page_config(page_title="Low-Cap Crypto Decay Simulator (Strategy 2 Only)", layout="wide")
st.title("Low-Cap Crypto Early Decay Simulator & Backtest (Strategy 2 Only)")

st.markdown("""
This dashboard simulates early/mid/late decay for Strategy 2 (momentum + liquidity) in low-cap cryptos.
Data: CoinGecko API (Pro header included). Research tool only; not financial advice.
""")

# ---------------------------
# Data helpers (original method preserved)
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
    # Liquidity proxy
    out["liquidity"] = (out["norm_volume"] / (out["volatility"] + 1e-8)).fillna(method="bfill").fillna(0)
    # Imbalance proxy
    out["imbalance"] = (out["ret"].rolling(flow_window).mean() - out["ret"].rolling(flow_window).median()).fillna(0)
    return out[["date","price","liquidity","imbalance","ret","volume","market_cap"]]

# ---------------------------
# Sidebar: Data selection
# ---------------------------

st.sidebar.header("Data")
vs_currency = st.sidebar.selectbox("Quote currency", ["usd", "eur", "btc"], index=0)
days_back = st.sidebar.number_input("History (days)", min_value=30, max_value=1825, value=365, step=30)
default_lowcaps = ["akash-network","celestia","injective-protocol","beam-2","sei-network","sui","near","render-token","optimism","arbitrum"]
coin_id = st.text_input("CoinGecko coin id", value=default_lowcaps[0])

try:
    raw = cg_coin_market_chart(coin_id, vs_currency, int(days_back))
    df = derive_liquidity_and_imbalance(raw)
    st.success(f"Loaded {len(df)} daily points for {coin_id}.")
    st.dataframe(df.tail(10), use_container_width=True)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    df = None

if df is None or len(df) < 60:
    st.stop()

# ---------------------------
# Simulation parameters
# ---------------------------

st.sidebar.header("Simulation Parameters")
H = st.sidebar.number_input("Simulation horizon (days)", value=10, min_value=2, max_value=180)
N_MC = st.sidebar.number_input("Monte Carlo paths", value=300, min_value=50, max_value=5000, step=50)
alpha_L = st.sidebar.number_input("alpha_L", value=0.05, step=0.01)
beta_L = st.sidebar.number_input("beta_L", value=0.10, step=0.01)
rho_I = st.sidebar.number_input("rho_I", value=0.95, step=0.01)
mu_I = st.sidebar.number_input("mu_I", value=-0.02, step=0.01)
sigma0 = st.sidebar.number_input("sigma0", value=0.03, step=0.005)
kappa = st.sidebar.number_input("kappa", value=0.5, step=0.05)
gamma = st.sidebar.number_input("gamma", value=0.5, step=0.05)
crash_pct = st.sidebar.number_input("Crash % threshold", value=0.25, min_value=0.05, max_value=0.9, step=0.05)
L_min_ratio = st.sidebar.number_input("Liquidity min ratio", value=0.10, min_value=0.01, max_value=1.0, step=0.01)
I_crit = st.sidebar.number_input("Imbalance threshold", value=0.20, min_value=0.01, max_value=2.0, step=0.01)
persist_days = st.sidebar.number_input("Persistence days (liquidity crash)", value=3, min_value=1, max_value=30)

# ---------------------------
# Hybrid Monte Carlo simulation for early/mid/late decay
# ---------------------------

def run_hybrid_simulation(P0, L0, I0, H, params):
    alpha_L, beta_L, rho_I, mu_I, sigma0, kappa, gamma = params
    P, L, I = P0, max(1e-6, L0), np.clip(I0, -1, 1)
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
# Chunk 2: Strategy 2, Backtest & Decay Entries
# ---------------------------

st.subheader("Strategy 2: Momentum + Liquidity (Early/Mid/Late Entry)")

with st.sidebar.expander("Strategy 2 Parameters", expanded=True):
k_mom = st.number_input("Momentum lookback (days)", value=20, min_value=5, max_value=120, step=5)
T2 = st.number_input("Crash prob soft threshold", value=0.25, min_value=0.0, max_value=1.0, step=0.05)
reduce_factor = st.number_input("Position reduce factor on high crash prob", value=0.5, min_value=0.0, max_value=1.0, step=0.05)

# Early / Mid / Late decay windows
early_window = st.sidebar.number_input("Early entry lookback (days)", value=5, min_value=1, max_value=30)
mid_window = st.sidebar.number_input("Mid entry lookback (days)", value=15, min_value=1, max_value=60)
late_window = st.sidebar.number_input("Late entry lookback (days)", value=30, min_value=1, max_value=90)

df_bt = df.copy()
df_bt["mom_k"] = df_bt["price"] / df_bt["price"].shift(int(k_mom)) - 1.0
liq_median = df_bt["liquidity"].median()

# Function to build position with decay-based entries
def build_decay_positions(d: pd.DataFrame):
    x = d.copy()
    x["position_early"] = 0.0
    x["position_mid"] = 0.0
    x["position_late"] = 0.0

    # Early decay: shorter rolling window
    x.loc[(x["mom_k"].rolling(early_window).mean() > 0) & (x["liquidity"] > x["liquidity"].rolling(early_window).median()), "position_early"] = 1.0
    x.loc[x["pred_crash_prob"] > T2, "position_early"] *= reduce_factor

    # Mid decay: medium rolling window
    x.loc[(x["mom_k"].rolling(mid_window).mean() > 0) & (x["liquidity"] > x["liquidity"].rolling(mid_window).median()), "position_mid"] = 1.0
    x.loc[x["pred_crash_prob"] > T2, "position_mid"] *= reduce_factor

    # Late decay: longer rolling window
    x.loc[(x["mom_k"].rolling(late_window).mean() > 0) & (x["liquidity"] > x["liquidity"].rolling(late_window).median()), "position_late"] = 1.0
    x.loc[x["pred_crash_prob"] > T2, "position_late"] *= reduce_factor

    return x

# ---------------------------
# Simulate rolling crash probabilities
# ---------------------------

st.subheader("Rolling crash probabilities (Monte Carlo)")

progress = st.progress(0, text="Running Monte Carlo simulation...")
crash_probs = []

for i in range(len(df_bt) - int(H)):
    P0, L0, I0 = df_bt["price"].iloc[i], df_bt["liquidity"].iloc[i], df_bt["imbalance"].iloc[i]
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
    progress.progress((i + 1) / (len(df_bt) - int(H)), text=f"Running... {i+1}/{len(df_bt) - int(H)}")

df_bt = df_bt.iloc[:len(crash_probs)].copy()
df_bt["pred_crash_prob"] = crash_probs
st.line_chart(df_bt.set_index("date")[["pred_crash_prob"]], use_container_width=True)

# ---------------------------
# Build Strategy 2 decay positions
# ---------------------------

positions_bt = build_decay_positions(df_bt)

# ---------------------------
# PnL computation
# ---------------------------

def compute_strategy2_pnl(d: pd.DataFrame, fee_bps: float = 5.0):
    x = d.copy()
    x["ret"] = x["ret"].fillna(0.0)

    for col in ["position_early", "position_mid", "position_late"]:
        pos = x[col].fillna(0.0).values
        pos_shift = np.roll(pos, 1); pos_shift[0] = 0.0
        turnover = np.abs(pos - pos_shift)
        tc = turnover * (fee_bps / 1e4)
        x[f"{col}_pnl"] = pos * x["ret"] - tc

    # Buy-and-hold benchmark
    x["bh_pnl"] = x["ret"]

    eq = x.set_index("date")[[f"{col}_pnl" for col in ["position_early", "position_mid", "position_late"]] + ["bh_pnl"]].cumsum().apply(np.exp)
    return x, eq

bt_row, equity_curves = compute_strategy2_pnl(positions_bt)

st.subheader("Equity curves (normalized)")
st.line_chart(equity_curves, use_container_width=True)

# ---------------------------
# Performance summary
# ---------------------------

def summarize_strategy2(d: pd.DataFrame, eq: pd.DataFrame):
    ann_factor = 365
    cols = [c for c in eq.columns if "_pnl" in c]
    rows = []
    for col in cols:
        r = d[col].fillna(0.0)
        cum = np.exp(r.cumsum().iloc[-1]) - 1.0
        vol = r.std() * np.sqrt(ann_factor)
        sharpe = (r.mean() * ann_factor) / (vol + 1e-12)
        curve = eq[col]
        mdd = (curve.cummax() / curve - 1.0).max()
        rows.append({"strategy": col.replace("_pnl", ""), "cum_return": cum, "ann_vol": vol, "sharpe": sharpe, "max_drawdown": mdd})
    return pd.DataFrame(rows)

perf = summarize_strategy2(bt_row, equity_curves)
st.subheader("Performance summary")
st.dataframe(perf.style.format({"cum_return": "{:.1%}", "ann_vol": "{:.2f}", "sharpe": "{:.2f}", "max_drawdown": "{:.1%}"}), use_container_width=True)
