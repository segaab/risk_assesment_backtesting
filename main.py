import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict

# ---------------------------
# CoinGecko Pro API Key + Base URL
# ---------------------------
COINGECKO_API_KEY = "CG-chRgqiH9ab4zsFTm2Zvst82a"
HEADERS: Dict[str, str] = {"x-cg-pro-api-key": COINGECKO_API_KEY} if COINGECKO_API_KEY else {}
COINGECKO_BASE = "https://pro-api.coingecko.com/api/v3"

# ---------------------------
# Streamlit setup
# ---------------------------
st.set_page_config(page_title="Low-Cap Crypto Crash Backtest (Strategy 2, History API)", layout="wide")
st.title("Low-Cap Crypto Sector Crash Simulator & Backtest — Strategy 2 (History API)")

st.markdown("""
This dashboard simulates risk proxies and backtests **Strategy 2**  
using **CoinGecko Pro's `/coins/{id}/history` endpoint**.  

⚠️ Notice: `/history` only returns **one day snapshot at 00:00 UTC**.  
We loop over days to build the dataset. This can be slower.
""")

# ---------------------------
# Data helpers (CoinGecko Pro History API)
# ---------------------------
@st.cache_data(show_spinner=False, ttl=60*30)
def cg_coin_history_series(coin_id: str, vs_currency: str, days: int) -> pd.DataFrame:
    """
    Fetch historical data series by calling /coins/{id}/history for each date.
    """
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days)
    rows = []

    for i in range(days + 1):
        d = start_date + timedelta(days=i)
        d_str = d.strftime("%d-%m-%Y")  # CoinGecko expects DD-MM-YYYY
        url = f"{COINGECKO_BASE}/coins/{coin_id}/history"
        params = {"date": d_str, "localization": "false"}
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)

        if r.status_code != 200:
            continue

        data = r.json()
        mkt = data.get("market_data", {})
        if not mkt:
            continue

        row = {
            "date": pd.to_datetime(d),
            "price": mkt.get("current_price", {}).get(vs_currency),
            "volume": mkt.get("total_volume", {}).get(vs_currency),
            "market_cap": mkt.get("market_cap", {}).get(vs_currency),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["date","price","volume","market_cap"])

    df = pd.DataFrame(rows).dropna().sort_values("date").reset_index(drop=True)
    return df

def derive_liquidity_and_imbalance(df: pd.DataFrame, vol_window: int = 14, flow_window: int = 7) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date","price","liquidity","imbalance","ret","volume","market_cap"])
    out = df.copy()
    out["ret"] = np.log(out["price"]).diff()
    out["volatility"] = out["ret"].rolling(vol_window).std()
    out["norm_volume"] = out["volume"].rolling(vol_window).mean()
    out["liquidity"] = (out["norm_volume"] / (out["volatility"] + 1e-8)).fillna(0)
    out["imbalance"] = (out["ret"].rolling(flow_window).mean() - out["ret"].rolling(flow_window).median()).fillna(0)
    return out[["date","price","liquidity","imbalance","ret","volume","market_cap"]]

# ---------------------------
# Universe & controls
# ---------------------------
COINS = {
    "Akash Network": "akash-network",
    "Celestia": "celestia",
    "Injective": "injective-protocol",
    "Beam": "beam-2",
    "Sei": "sei-network",
    "Sui": "sui",
    "NEAR": "near",
    "Render": "render-token",
    "Optimism": "optimism",
    "Arbitrum": "arbitrum",
}

with st.sidebar:
    st.header("Data")
    coin_name = st.selectbox("Coin", list(COINS.keys()))
    coin_id = COINS[coin_name]
    vs_currency = st.selectbox("Quote currency", ["usd", "eur", "btc"], index=0)
    days = st.number_input("History (days)", min_value=30, max_value=720, value=180, step=30)

    st.header("Strategy 2 parameters")
    k_mom = st.number_input("Momentum lookback (days)", value=20, min_value=5, max_value=120, step=5)
    early_window = st.number_input("Early entry window", value=5, min_value=2, max_value=60, step=1)
    mid_window = st.number_input("Mid entry window", value=15, min_value=5, max_value=120, step=1)
    late_window = st.number_input("Late entry window", value=30, min_value=10, max_value=180, step=1)

    liq_quantile_gate = st.slider("Liquidity gate quantile", 0.0, 1.0, 0.50, 0.05)
    liq_scale = st.checkbox("Scale position by relative liquidity", value=True)

    st.markdown("---")
    st.subheader("Decay filter")
    decay_threshold = st.number_input("Decay threshold (log slope over k_mom)", value=-1.5, step=0.1)
    decay_reduce_factor = st.number_input("Reduce factor when decay flagged", value=0.0, min_value=0.0, max_value=1.0, step=0.1)

    st.markdown("---")
    fee_bps = st.number_input("Fees (bps per one-way)", value=5.0, min_value=0.0, max_value=100.0, step=1.0)

# ---------------------------
# Load & prepare data
# ---------------------------
try:
    raw = cg_coin_history_series(coin_id, vs_currency, int(days))
    if raw.empty:
        st.error("No historical data returned from CoinGecko. Try fewer days or another coin.")
        st.stop()

    df = derive_liquidity_and_imbalance(raw)

    if df.shape[0] < max(60, k_mom + late_window + 5):
        st.warning("Not enough history for selected parameters.")
        st.stop()

    st.subheader(f"Data Preview — {coin_name}")
    st.dataframe(df.tail(10), use_container_width=True)

    # ---------------------------
    # Momentum + decay flags
    # ---------------------------
    work = df.copy()
    work["momentum"] = work["price"].pct_change(int(k_mom))
    work["log_price"] = np.log(work["price"])
    work["log_slope"] = work["log_price"].diff(int(k_mom))
    work["decay_flag"] = (work["log_slope"] < float(decay_threshold)).astype(int)

    liq_gate_level = work["liquidity"].quantile(liq_quantile_gate)

    # ---------------------------
    # Strategy 2: early / mid / late positions
    # ---------------------------
    def make_positions(d: pd.DataFrame, entry_window: int, base_size: float) -> pd.Series:
        mom_ok = d["momentum"].rolling(entry_window).mean() > 0
        liq_ok = d["liquidity"].rolling(entry_window).mean() > liq_gate_level
        entry = (mom_ok & liq_ok).astype(float)

        if liq_scale:
            liq_z = (d["liquidity"] - d["liquidity"].rolling(60).mean()) / (d["liquidity"].rolling(60).std() + 1e-12)
            liq_z = liq_z.clip(lower=-2, upper=2).fillna(0.0)
            size = base_size * (1.0 + 0.25 * liq_z)
        else:
            size = base_size

        pos = entry * size
        if decay_reduce_factor < 1.0:
            pos = np.where(d["decay_flag"] == 1, pos * decay_reduce_factor, pos)
        return pd.Series(pos, index=d.index)

    res = work.copy()
    res["pos_early"] = make_positions(work, early_window, base_size=1.00)
    res["pos_mid"]   = make_positions(work, mid_window, base_size=0.80)
    res["pos_late"]  = make_positions(work, late_window, base_size=0.60)

    # ---------------------------
    # PnL and equity curves
    # ---------------------------
    def pnl_with_costs(series_pos: pd.Series, rets: pd.Series, fee_bps: float) -> pd.Series:
        pos = series_pos.fillna(0.0).values
        pos_prev = np.roll(pos, 1); pos_prev[0] = 0.0
        turnover = np.abs(pos - pos_prev)
        tc = turnover * (fee_bps / 1e4)
        return pd.Series(pos * rets.fillna(0.0).values - tc, index=series_pos.index)

    res["pnl_early"] = pnl_with_costs(res["pos_early"], res["ret"], fee_bps)
    res["pnl_mid"]   = pnl_with_costs(res["pos_mid"], res["ret"], fee_bps)
    res["pnl_late"]  = pnl_with_costs(res["pos_late"], res["ret"], fee_bps)
    res["pnl_bh"]    = res["ret"].fillna(0.0)

    if "date" not in res.columns:
        st.error("No valid 'date' column found in results. Cannot plot equity curves.")
        st.stop()

    eq = res.set_index("date")[["pnl_early","pnl_mid","pnl_late","pnl_bh"]].cumsum().apply(np.exp)

    st.subheader("Equity Curves")
    st.line_chart(eq, use_container_width=True)

    def summarize(d: pd.DataFrame, equity: pd.DataFrame) -> pd.DataFrame:
        ann = 365
        rows = []
        for key in ["pnl_early","pnl_mid","pnl_late","pnl_bh"]:
            r = d[key].fillna(0.0)
            cum = np.exp(r.cumsum().iloc[-1]) - 1.0
            vol = r.std() * np.sqrt(ann)
            sharpe = (r.mean() * ann) / (vol + 1e-12)
            curve = equity[key]
            mdd = (curve.cummax() / curve - 1.0).max()
            rows.append({"strategy": key.replace("pnl_",""),
                         "cum_return": cum, "ann_vol": vol, "sharpe": sharpe, "max_drawdown": mdd})
        return pd.DataFrame(rows)

    perf = summarize(res, eq)
    st.subheader("Performance Summary")
    st.dataframe(
        perf.style.format({"cum_return":"{:.1%}","ann_vol":"{:.2f}","sharpe":"{:.2f}","max_drawdown":"{:.1%}"}),
        use_container_width=True
    )

    log_cols = ["date","price","liquidity","imbalance","ret","momentum","log_slope","decay_flag",
                "pos_early","pos_mid","pos_late","pnl_early","pnl_mid","pnl_late","pnl_bh"]
    decision_log = res[log_cols].copy()
    with st.expander("Decision Log"):
        st.dataframe(decision_log.tail(25), use_container_width=True)
        st.download_button("Download decision_log.csv", data=decision_log.to_csv(index=False),
                           file_name="decision_log.csv", mime="text/csv")

except Exception as e:
    st.error(f"Failed to load data: {e}")
