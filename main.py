import streamlit as st
import pandas as pd
import numpy as np
import requests
from typing import Dict
from scipy.stats import norm

# ---------------------------
# CoinGecko API setup
# ---------------------------

COINGECKO_API_KEY = "CG-chRgqiH9ab4zsFTm2Zvst82a"
HEADERS: Dict[str, str] = {"x-cg-pro-api-key": COINGECKO_API_KEY} if COINGECKO_API_KEY else {}
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

st.set_page_config(page_title="Low-Cap Crypto Crash Backtest (Strategy 2)", layout="wide")
st.title("Low-Cap Crypto Crash Simulator & Strategy 2 Backtest")

st.markdown("""
This dashboard simulates sector crash risk using liquidity and imbalance dynamics and backtests:

- Strategy 2: Momentum + Liquidity risk mitigation with early/mid/late entries.

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
    out["liquidity"] = (out["norm_volume"] / (out["volatility"] + 1e-8)).fillna(method="bfill").fillna(0)
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
persist_days = st.sidebar.number_input("Persistence days (liquidity crash)", value=3, min_value=1, max_value=30)

# Strategy 2 parameters
with st.sidebar.expander("Strategy 2 parameters", expanded=True):
    k_mom = st.number_input("Momentum lookback (days)", value=20, min_value=5, max_value=120, step=5)
    T2 = st.number_input("Crash prob soft threshold", value=0.25, min_value=0.0, max_value=1.0, step=0.05)
    reduce_factor = st.number_input("Size reduce factor", value=0.5, min_value=0.0, max_value=1.0, step=0.1)
    early_window = st.number_input("Early entry rolling window", value=5, min_value=1, max_value=30, step=1)
    mid_window = st.number_input("Mid entry rolling window", value=15, min_value=1, max_value=60, step=1)
    late_window = st.number_input("Late entry rolling window", value=30, min_value=1, max_value=90, step=1)

# ---------------------------
# Monte Carlo simulation for early decay
# ---------------------------

def run_hybrid_simulation(P0, L0, I0, H, params):
    alpha_L, beta_L, rho_I, mu_I, sigma0, kappa, gamma = params
    P = P0
    L = max(1e-6, L0)
    I = np.clip(I0, -1, 1)
    path = []
    for _ in range(int(H)):
        delta_flow = 0.0
        r_prev = 0.0
        # Simulate early decay effect on liquidity
        decay_factor = np.random.choice([0.7, 0.85, 1.0])  # Early, mid, late decay
        L_prop = L * (1 + alpha_L * delta_flow - beta_L * (abs(r_prev) > 0.05)) * decay_factor
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
# Rolling crash probabilities
# ---------------------------

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
# Strategy 2: Momentum + Liquidity entries
# ---------------------------

df_long = df_bt.copy()
df_long["mom_k"] = df_long["price"] / df_long["price"].shift(int(k_mom)) - 1.0
liq_median = df_long["liquidity"].median()

def build_strategy2_positions(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    # Initialize all positions at zero
    x["position_early"] = 0.0
    x["position_mid"] = 0.0
    x["position_late"] = 0.0

    # Early entry: smaller rolling window
    cond_early = (x["mom_k"].rolling(early_window).mean() > 0) & (x["liquidity"].rolling(early_window).mean() > liq_median)
    x.loc[cond_early, "position_early"] = 1.0

    # Mid entry: medium rolling window
    cond_mid = (x["mom_k"].rolling(mid_window).mean() > 0) & (x["liquidity"].rolling(mid_window).mean() > liq_median)
    x.loc[cond_mid, "position_mid"] = 1.0 * reduce_factor

    # Late entry: long rolling window
    cond_late = (x["mom_k"].rolling(late_window).mean() > 0) & (x["liquidity"].rolling(late_window).mean() > liq_median)
    x.loc[cond_late, "position_late"] = 1.0 * reduce_factor

    return x

strategy2_bt = build_strategy2_positions(df_long)

# ---------------------------
# PnL computation
# ---------------------------

def compute_pnl(d: pd.DataFrame, fee_bps: float = 5.0):
    df = d.copy()
    df["ret"] = df["ret"].fillna(0.0)
    for col in ["position_early","position_mid","position_late"]:
        pos = df[col].fillna(0.0).values
        pos_shift = np.roll(pos,1); pos_shift[0] = 0.0
        turnover = np.abs(pos - pos_shift)
        tc = turnover * (fee_bps/1e4)
        df[f"{col}_pnl"] = pos * df["ret"] - tc
    # Benchmark: buy-and-hold
    df["bh_pnl"] = df["ret"]
    eq = df.set_index("date")[[c+"_pnl" for c in ["position_early","position_mid","position_late"]] + ["bh_pnl"]].cumsum().apply(np.exp)
    return df, eq

bt_row, equity_curves = compute_pnl(strategy2_bt)
st.subheader("Equity curves (Strategy 2)")
st.line_chart(equity_curves, use_container_width=True)

# ---------------------------
# Decision log
# ---------------------------

def build_decision_log(d: pd.DataFrame) -> pd.DataFrame:
    log = d[["date","price","liquidity","imbalance","ret","pred_crash_prob",
             "position_early","position_mid","position_late"]].copy()
    fwd = d["price"].shift(-int(H)) / d["price"] - 1.0
    log[f"fwd_return_{int(H)}d"] = fwd
    log[f"fwd_crash_{int(H)}d"] = (fwd <= -crash_pct).astype(int)
    return log

decision_log = build_decision_log(strategy2_bt)
with st.expander("Decision log (exportable)"):
    st.dataframe(decision_log.tail(20), use_container_width=True)
    st.download_button("Download decision log CSV", data=decision_log.to_csv(index=False),
                       file_name="decision_log.csv", mime="text/csv")

# ---------------------------
# Performance summary
# ---------------------------

def summarize_performance(d: pd.DataFrame, eq: pd.DataFrame) -> pd.DataFrame:
    ann_factor = 365
    desired = ["position_early_pnl","position_mid_pnl","position_late_pnl","bh_pnl"]
    pnl_cols = [c for c in desired if c in d.columns and c in eq.columns]
    rows = []
    for col in pnl_cols:
        r = d[col].fillna(0.0)
        cum = np.exp(r.cumsum().iloc[-1])-1.0
        vol = r.std()*np.sqrt(ann_factor)
        sharpe = (r.mean()*ann_factor)/(vol+1e-12)
        curve = eq[col]
        mdd = (curve.cummax()/curve - 1.0).max()
        rows.append({"strategy": col.replace("_pnl",""), "cum_return": cum, "ann_vol": vol, "sharpe": sharpe, "max_drawdown": mdd})
    return pd.DataFrame(rows)

perf = summarize_performance(bt_row, equity_curves)
st.subheader("Performance summary")
st.dataframe(perf.style.format({"cum_return":"{:.1%}","ann_vol":"{:.2f}","sharpe":"{:.2f}","max_drawdown":"{:.1%}"}), use_container_width=True)
