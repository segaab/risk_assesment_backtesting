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
# Long-side strategy: Strategy 2 only
# ---------------------------

st.subheader("Long-side Strategy 2 (Momentum + Crash Prob)")

df_long = df_ed.copy()
df_long["mom_k"] = df_long["price"] / df_long["price"].shift(int(k_mom)) - 1.0

liq_median = df_long["liquidity"].median()

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
        means.append(mean)
        lowers.append(mean - z * se)
        uppers.append(mean + z * se)
    return pd.DataFrame({"post_mean": means, "post_low": lowers, "post_high": uppers}, index=probs.index)

post_df = rolling_beta_posterior(df_long["pred_crash_prob"], window=30, cred=0.95)

def build_strategy2_positions(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    x["position_s2_early"] = 0.0
    x["position_s2_mid"] = 0.0
    x["position_s2_late"] = 0.0

    # Early entry: smaller rolling window
    early_mom = x["price"] / x["price"].shift(int(early_window)) - 1.0
    cond_early = (early_mom > 0) & (x["pred_crash_prob"] < T2) & (x["liquidity"] > liq_median)
    x.loc[cond_early, "position_s2_early"] = 1.0

    # Mid entry: medium rolling window
    mid_mom = x["price"] / x["price"].shift(int(mid_window)) - 1.0
    cond_mid = (mid_mom > 0) & (x["pred_crash_prob"] < T2) & (x["liquidity"] > liq_median)
    x.loc[cond_mid, "position_s2_mid"] = 1.0

    # Late entry: larger rolling window
    late_mom = x["price"] / x["price"].shift(int(late_window)) - 1.0
    cond_late = (late_mom > 0) & (x["pred_crash_prob"] < T2) & (x["liquidity"] > liq_median)
    x.loc[cond_late, "position_s2_late"] = 1.0

    # Reduce positions if crash probability is high
    x.loc[x["pred_crash_prob"] > T2, ["position_s2_early","position_s2_mid","position_s2_late"]] *= reduce_factor

    return x

strategy2_bt = build_strategy2_positions(df_long)

# ---------------------------
# PnL computation for Strategy 2
# ---------------------------

def compute_strategy2_pnl(d: pd.DataFrame, fee_bps: float = 5.0):
    df = d.copy()
    df["ret"] = df["ret"].fillna(0.0)
    for col in ["position_s2_early","position_s2_mid","position_s2_late"]:
        pos = df[col].fillna(0.0).values
        pos_shift = np.roll(pos, 1); pos_shift[0] = 0.0
        turnover = np.abs(pos - pos_shift)
        tc = turnover * (fee_bps / 1e4)
        df[f"{col}_pnl"] = pos * df["ret"] - tc
    df["bh_pnl"] = df["ret"]
    eq = df.set_index("date")[[c+"_pnl" for c in ["position_s2_early","position_s2_mid","position_s2_late"]]+["bh_pnl"]].cumsum().apply(np.exp)
    return df, eq

bt_row, equity_curves = compute_strategy2_pnl(strategy2_bt)

st.subheader("Equity curves (Strategy 2)")
st.line_chart(equity_curves, use_container_width=True)

# ---------------------------
# Decision log for Strategy 2
# ---------------------------

def build_decision_log_strategy2(d: pd.DataFrame) -> pd.DataFrame:
    log = d[["date","price","liquidity","imbalance","ret","pred_crash_prob"]].copy()
    log["position_s2_early"] = d["position_s2_early"]
    log["position_s2_mid"] = d["position_s2_mid"]
    log["position_s2_late"] = d["position_s2_late"]
    fwd = d["price"].shift(-int(H)) / d["price"] - 1.0
    log[f"fwd_return_{int(H)}d"] = fwd
    log[f"fwd_crash_{int(H)}d"] = (fwd <= -crash_pct).astype(int)
    return log

decision_log = build_decision_log_strategy2(strategy2_bt)
with st.expander("Decision log (exportable)"):
    st.dataframe(decision_log.tail(20), use_container_width=True)
    st.download_button("Download decision log CSV", data=decision_log.to_csv(index=False), file_name="decision_log.csv", mime="text/csv")

# ---------------------------
# Performance summary for Strategy 2
# ---------------------------

def summarize_strategy2_performance(d: pd.DataFrame, eq: pd.DataFrame) -> pd.DataFrame:
    ann_factor = 365
    pnl_cols = [c for c in eq.columns]
    rows = []
    for col in pnl_cols:
        r = d[col.replace("_pnl","")+"_pnl"].fillna(0.0)
        cum = np.exp(r.cumsum().iloc[-1]) - 1.0
        vol = r.std() * np.sqrt(ann_factor)
        sharpe = (r.mean() * ann_factor) / (vol + 1e-12)
        curve = eq[col]
        mdd = (curve.cummax() / curve - 1.0).max()
        rows.append({
            "strategy": col.replace("_pnl",""),
            "cum_return": cum,
            "ann_vol": vol,
            "sharpe": sharpe,
            "max_drawdown": mdd
        })
    return pd.DataFrame(rows)

perf = summarize_strategy2_performance(bt_row, equity_curves)
st.subheader("Performance summary (Strategy 2)")
st.dataframe(
    perf.style.format({"cum_return": "{:.1%}", "ann_vol": "{:.2f}", "sharpe": "{:.2f}", "max_drawdown": "{:.1%}"}),
    use_container_width=True
    )
