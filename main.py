# ---------------------------
# Chunk 1: Data Loading, Derivation, ECDS, Monte Carlo
# ---------------------------

import streamlit as st
import pandas as pd
import numpy as np
import requests
from typing import Dict
from scipy.stats import norm

# ---------------------------
# CoinGecko Pro key (hard-coded)
# ---------------------------
COINGECKO_API_KEY = "CG-chRgqiH9ab4zsFTm2Zvst82a"
HEADERS: Dict[str, str] = {"x-cg-pro-api-key": COINGECKO_API_KEY} if COINGECKO_API_KEY else {}
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

st.set_page_config(page_title="Low-Cap Crypto Strategy 2 Backtest", layout="wide")
st.title("Low-Cap Crypto Crash Simulator & Strategy 2 Backtest")

st.markdown("""
This dashboard backtests **Strategy 2 only** (momentum + liquidity) with early/mid/late entry timing
and overlays Monte Carlo crash probability. Data is fetched from CoinGecko.
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
    out["liquidity"] = (out["norm_volume"] / (out["volatility"] + 1e-8)).fillna(method="bfill").fillna(0)
    out["imbalance"] = (out["ret"].rolling(flow_window).mean() - out["ret"].rolling(flow_window).median()).fillna(0)
    return out[["date","price","liquidity","imbalance","ret","volume","market_cap"]]

# ---------------------------
# Sidebar: Data selection
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
    st.stop()# ---------------------------
# Chunk 2: Strategy 2, Monte Carlo, Early/Mid/Late entries, PnL, Visualization
# ---------------------------

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
crash_pct = st.sidebar.number_input("Crash % threshold", value=0.25, min_value=0.05, max_value=0.9, step=0.05)
L_min_ratio = st.sidebar.number_input("Liquidity min ratio", value=0.10, min_value=0.01, max_value=1.0, step=0.01, format="%.2f")
I_crit = st.sidebar.number_input("Imbalance threshold", value=0.20, min_value=0.01, max_value=2.0, step=0.01, format="%.2f")
persist_days = st.sidebar.number_input("Persistence days (liquidity crash)", value=3, min_value=1, max_value=30)

# ---------------------------
# Monte Carlo hybrid simulation
# ---------------------------
def run_hybrid_simulation(P0, L0, I0, H, params):
    alpha_L, beta_L, rho_I, mu_I, sigma0, kappa, gamma = params
    P, L, I = P0, max(1e-6,L0), np.clip(I0,-1,1)
    path = []
    for _ in range(int(H)):
        delta_flow = 0.0
        r_prev = 0.0
        L_prop = L * (1 + alpha_L*delta_flow - beta_L*(abs(r_prev)>0.05))
        shock = 0.0
        I_prop = rho_I*I + shock + np.random.randn()*0.01
        gL = 1 + kappa*(1.0/(L_prop+1e-6))**gamma
        r = mu_I*I_prop + sigma0*gL*np.random.randn()
        P *= np.exp(r)
        L = max(1e-6,L_prop)
        I = np.clip(I_prop,-1,1)
        path.append([P,L,I,r])
    return np.array(path)

# ---------------------------
# Compute rolling crash probabilities
# ---------------------------
with st.expander("Rolling crash probabilities"):
    progress = st.progress(0, text="Running Monte Carlo backtest...")
    crash_probs = []
    idxs = list(range(0, len(df)-int(H)))
    for i, idx in enumerate(idxs):
        P0, L0, I0 = df["price"].iloc[idx], df["liquidity"].iloc[idx], df["imbalance"].iloc[idx]
        crash_count = 0
        for _ in range(int(N_MC)):
            sim = run_hybrid_simulation(P0,L0,I0,int(H),[alpha_L,beta_L,rho_I,mu_I,sigma0,kappa,gamma])
            crash_price = sim[:,0] <= P0*(1-crash_pct)
            crash_liq = (sim[:,1]<=L_min_ratio*L0) & (sim[:,2]<=-I_crit)
            if persist_days <= len(crash_liq):
                crash_liq_persist = np.convolve(crash_liq.astype(int), np.ones(int(persist_days)), "valid") >= persist_days
            else:
                crash_liq_persist = np.array([False])
            if np.any(crash_price) or np.any(crash_liq_persist):
                crash_count += 1
        crash_probs.append(crash_count/float(N_MC))
        progress.progress((i+1)/len(idxs), text=f"Running... {i+1}/{len(idxs)}")

df_bt = df.iloc[:len(crash_probs)].copy()
df_bt["pred_crash_prob"] = crash_probs
st.line_chart(df_bt.set_index("date")[["pred_crash_prob"]], use_container_width=True)

# ---------------------------
# Strategy 2 only: momentum + liquidity
# ---------------------------
st.subheader("Strategy 2: Momentum + Liquidity (Early/Mid/Late entry)")
with st.sidebar.expander("Strategy 2 parameters"):
    k_mom = st.number_input("Momentum lookback", value=20, min_value=5, max_value=120, step=5)
    T2 = st.number_input("Crash prob soft threshold", value=0.25, min_value=0.0, max_value=1.0, step=0.05)
    reduce_factor = st.number_input("Size reduce factor", value=0.5, min_value=0.0, max_value=1.0, step=0.1)
    entry_window_days = st.number_input("Entry rolling window (days)", value=10, min_value=2, max_value=30, step=1)

df_bt["mom_k"] = df_bt["price"]/df_bt["price"].shift(int(k_mom)) - 1.0
liq_med = df_bt["liquidity"].median()

# Early/Mid/Late entry: rolling windows
def assign_entry_timing(d: pd.DataFrame, window: int):
    early, mid, late = np.zeros(len(d)), np.zeros(len(d)), np.zeros(len(d))
    for i in range(len(d)):
        start = max(0,i-window+1)
        mom_window = d["mom_k"].iloc[start:i+1]
        liq_window = d["liquidity"].iloc[start:i+1]
        prob_window = d["pred_crash_prob"].iloc[start:i+1]

        # Early entry: first signal in window
        if ((mom_window>0) & (liq_window>liq_med) & (prob_window<T2)).any():
            early[i] = 1.0
        # Mid entry: median signal
        if ((mom_window>0) & (liq_window>liq_med) & (prob_window<T2)).sum() >= window//2:
            mid[i] = 1.0
        # Late entry: persistent window signal
        if ((mom_window>0) & (liq_window>liq_med) & (prob_window<T2)).sum() == window:
            late[i] = 1.0
    return early, mid, late

df_bt["entry_early"], df_bt["entry_mid"], df_bt["entry_late"] = assign_entry_timing(df_bt,int(entry_window_days))
st.dataframe(df_bt[["date","mom_k","liquidity","pred_crash_prob","entry_early","entry_mid","entry_late"]].tail(20))
