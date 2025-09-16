"""
Fastest-to-Ship Auto Screener + Auto Trader
Stack: Streamlit UI + Finnhub (data) + Alpaca (paper trading) + pure-Python indicators

One-file prototype you can run locally:
    streamlit run app.py

What it does
- Lets you define a ticker universe and rules (EMA/MACD/KDJ/Volume)
- Pulls latest candles from Finnhub
- Computes indicators and produces BUY/SELL/HOLD signals
- Places MARKET orders to Alpaca PAPER account with simple risk controls
- Shows positions, orders, P&L, and logs in Streamlit

Disclaimers
- Educational prototype. Use paper trading first. Past performance ≠ future results.
- Respect vendor rate limits. Add sleep/batching for larger universes.

Setup
1) Create accounts & API keys
   - Alpaca: https://alpaca.markets  (use PAPER endpoint)
   - Finnhub: https://finnhub.io (free tier is OK to start)
2) Set env vars (Windows PowerShell example):
   $env:ALPACA_KEY="YOUR_KEY"
   $env:ALPACA_SECRET="YOUR_SECRET"
   $env:ALPACA_BASE="https://paper-api.alpaca.markets"  # PAPER trading endpoint
   $env:FINNHUB_KEY="YOUR_FINNHUB_KEY"
3) Install deps:
   pip install streamlit requests pandas numpy pydantic alpaca-trade-api

Notes
- Universe defaults to S&P 100 subset for speed. You can edit in the UI.
- KDJ is computed from Stochastic oscillator (K, D) and J = 3*K - 2*D.
- Position sizing: fixed risk per trade % and ATR-based stop distance (optional simple version here).
- For Bursa/MYX we skip; this is US-only.

"""

import os
import time
import math
import json
import hmac
import hashlib
import requests
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple

import streamlit as st
from pydantic import BaseModel, Field, ValidationError

# ========== CONFIG & CLIENTS ==========
# Prefer Streamlit Cloud secrets, fall back to env vars for local runs
ALPACA_KEY = st.secrets.get("ALPACA_KEY", os.getenv("ALPACA_KEY", ""))
ALPACA_SECRET = st.secrets.get("ALPACA_SECRET", os.getenv("ALPACA_SECRET", ""))
ALPACA_BASE = st.secrets.get("ALPACA_BASE", os.getenv("ALPACA_BASE", "https://paper-api.alpaca.markets"))
FINNHUB_KEY = st.secrets.get("FINNHUB_KEY", os.getenv("FINNHUB_KEY", ""))

ALPACA_HEADERS = {
    "APCA-API-KEY-ID": ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET,
}

# Minimal error guard
if not FINNHUB_KEY:
    st.warning("FINNHUB_KEY not set. Set environment variables before running live.")
if not (ALPACA_KEY and ALPACA_SECRET):
    st.warning("Alpaca keys not set. Orders will be disabled.")

# ========== UTILS ==========

def utcnow_iso():
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')

@st.cache_data(show_spinner=False)
def _sp100_subset() -> List[str]:
    # Lightweight subset to start; edit as you like
    return [
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK.B","LLY","JPM",
        "V","XOM","AVGO","UNH","TSLA","WMT","MA","PG","COST","JNJ",
        "HD","MRK","ORCL","ADBE","BAC","PEP","KO","NFLX","CSCO","ABT",
        "CRM","TMUS","LIN","ACN","DHR","CMCSA","MCD","WFC","AMD","TMO",
        "INTU","TXN","DIS","NKE","PM","IBM","RTX","ISRG","UPS","CAT"
    ]

# ========== DATA LAYER (FINNHUB) ==========

FINNHUB_BASE = "https://finnhub.io/api/v1"

class CandleRes(BaseModel):
    c: List[float]  # close
    h: List[float]  # high
    l: List[float]  # low
    o: List[float]  # open
    s: str          # status
    t: List[int]    # timestamps (seconds)
    v: List[float]  # volume


def get_candles(symbol: str, resolution: str = "60", lookback_days: int = 60) -> pd.DataFrame:
    """Fetch OHLCV candles from Finnhub.
    resolution: '1','5','15','30','60','D','W','M'
    """
    to_ts = int(datetime.now(timezone.utc).timestamp())
    from_ts = to_ts - lookback_days * 24 * 3600
    params = {
        "symbol": symbol,
        "resolution": resolution,
        "from": from_ts,
        "to": to_ts,
        "token": FINNHUB_KEY,
    }
    r = requests.get(f"{FINNHUB_BASE}/stock/candle", params=params, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"Finnhub candle error {r.status_code}: {r.text}")
    data = r.json()
    try:
        cr = CandleRes(**data)
    except ValidationError:
        # Sometimes Finnhub returns { 's': 'no_data' }
        if data.get("s") == "no_data":
            return pd.DataFrame()
        raise
    if cr.s != "ok":
        return pd.DataFrame()
    df = pd.DataFrame({
        "t": [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in cr.t],
        "open": cr.o,
        "high": cr.h,
        "low": cr.l,
        "close": cr.c,
        "volume": cr.v,
    })
    return df.set_index("t").sort_index()

# ========== INDICATORS ==========

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out

def stochastic_kd(df: pd.DataFrame, period: int = 14, k_smooth: int = 3, d_smooth: int = 3) -> Tuple[pd.Series, pd.Series]:
    lowest_low = df['low'].rolling(window=period).min()
    highest_high = df['high'].rolling(window=period).max()
    fast_k = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
    k = fast_k.rolling(window=k_smooth).mean()
    d = k.rolling(window=d_smooth).mean()
    return k, d

def kdj(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    k, d = stochastic_kd(df, period=period)
    j = 3 * k - 2 * d
    return k, d, j

# ========== SCREENING RULES ==========
@dataclass
class Rules:
    ema_fast: int = 5
    ema_mid: int = 20
    ema_slow: int = 50
    require_ema_rising: bool = True

    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    macd_hist_must_shrink_neg: bool = True
    macd_line_gt_signal: bool = True

    kdj_period: int = 14
    kdj_j_max: float = 80.0

    min_volume: int = 1_000_000

    rsi_min: float = 40.0
    rsi_max: float = 60.0


def apply_rules(df: pd.DataFrame, rules: Rules) -> Dict[str, any]:
    out = {"passes": False, "why": []}
    if df.empty or len(df) < max(rules.ema_slow, rules.kdj_period, rules.macd_slow) + 5:
        out["why"].append("insufficient data")
        return out

    df = df.copy()
    df["ema_fast"] = ema(df["close"], rules.ema_fast)
    df["ema_mid"] = ema(df["close"], rules.ema_mid)
    df["ema_slow"] = ema(df["close"], rules.ema_slow)

    macd_line, macd_signal, hist = macd(df["close"], rules.macd_fast, rules.macd_slow, rules.macd_signal)
    df["macd_line"], df["macd_signal"], df["macd_hist"] = macd_line, macd_signal, hist

    df["rsi14"] = rsi(df["close"], 14)

    k, d = stochastic_kd(df, period=rules.kdj_period)
    j = 3*k - 2*d
    df["K"], df["D"], df["J"] = k, d, j

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Volume rule
    if last["volume"] < rules.min_volume:
        out["why"].append(f"volume {int(last['volume'])} < {rules.min_volume}")

    # EMA structure
    if not (last["ema_fast"] > last["ema_mid"] > last["ema_slow"]):
        out["why"].append("EMA order not bullish (fast>mid>slow)")
    if rules.require_ema_rising:
        if not (last["ema_fast"] > prev["ema_fast"] and last["ema_mid"] > prev["ema_mid"] and last["ema_slow"] > prev["ema_slow"]):
            out["why"].append("EMAs not rising vs previous bar")

    # MACD
    if rules.macd_hist_must_shrink_neg:
        if not (last["macd_hist"] > prev["macd_hist"] and last["macd_hist"] < 0):
            out["why"].append("MACD hist not negative-shrinking")
    if rules.macd_line_gt_signal and not (last["macd_line"] > last["macd_signal"] and last["macd_line"] > prev["macd_line"]):
        out["why"].append("MACD line not > signal & rising")

    # KDJ
    if not (last["K"] > last["D"] and last["K"] > prev["K"] and last["D"] > prev["D"] and last["J"] > prev["J"] and last["J"] < rules.kdj_j_max):
        out["why"].append("KDJ not bullish-but-not-hot")

    # RSI window (optional neutral zone)
    if not (rules.rsi_min <= last["rsi14"] <= rules.rsi_max):
        out["why"].append(f"RSI {last['rsi14']:.1f} outside [{rules.rsi_min},{rules.rsi_max}]")

    out["passes"] = (len(out["why"]) == 0)
    out["last_row"] = last
    return out

# ========== RISK & ORDERS (ALPACA) ==========

def alpaca_get_account() -> dict:
    r = requests.get(f"{ALPACA_BASE}/v2/account", headers=ALPACA_HEADERS, timeout=20)
    if r.status_code != 200:
        return {"error": r.text}
    return r.json()

def alpaca_list_positions() -> List[dict]:
    r = requests.get(f"{ALPACA_BASE}/v2/positions", headers=ALPACA_HEADERS, timeout=20)
    if r.status_code != 200:
        return []
    return r.json()

def alpaca_get_clock() -> dict:
    r = requests.get(f"{ALPACA_BASE}/v2/clock", headers=ALPACA_HEADERS, timeout=20)
    return r.json() if r.status_code == 200 else {"is_open": False}

def alpaca_place_order(symbol: str, qty: float, side: str = "buy", order_type: str = "market", tif: str = "day") -> dict:
    payload = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": order_type,
        "time_in_force": tif,
    }
    r = requests.post(f"{ALPACA_BASE}/v2/orders", headers=ALPACA_HEADERS, json=payload, timeout=20)
    if r.status_code not in (200, 201):
        return {"error": r.text, "status": r.status_code}
    return r.json()

# Simple position sizing: fixed % of equity per trade, rounded to whole shares using last close

def calc_order_size(equity: float, risk_pct: float, price: float) -> int:
    if price <= 0:
        return 0
    dollars = equity * risk_pct
    qty = int(dollars // price)
    return max(qty, 0)

# ========== STREAMLIT UI ==========

st.set_page_config(page_title="Auto Screener + Trader", layout="wide")

st.title("⚡ Fastest-to-Ship Auto Screener + Trader (Alpaca + Finnhub)")
st.caption("Educational prototype. Use PAPER trading. Built for US equities.")

with st.sidebar:
    st.header("Settings")
    universe_default = ",".join(_sp100_subset())
    tickers_input = st.text_area("Universe (comma-separated)", universe_default, height=120)
    resolution = st.selectbox("Bar timeframe", ["60", "30", "15", "5", "D"], index=0, help="Finnhub resolution")
    lookback_days = st.slider("Lookback days", 20, 180, 60)

    st.subheader("Rules")
    ema_fast = st.number_input("EMA fast", 3, 50, 5)
    ema_mid = st.number_input("EMA mid", 5, 100, 20)
    ema_slow = st.number_input("EMA slow", 10, 200, 50)
    require_rising = st.checkbox("Require EMAs rising", True)

    macd_fast = st.number_input("MACD fast", 5, 50, 12)
    macd_slow = st.number_input("MACD slow", 10, 100, 26)
    macd_signal = st.number_input("MACD signal", 5, 50, 9)
    macd_shrink = st.checkbox("MACD hist negative but shrinking", True)

    kdj_period = st.number_input("KDJ period", 5, 50, 14)
    kdj_j_max = st.number_input("KDJ J max", 50.0, 100.0, 80.0)

    rsi_min = st.number_input("RSI min", 0.0, 100.0, 40.0)
    rsi_max = st.number_input("RSI max", 0.0, 100.0, 60.0)

    min_volume = st.number_input("Min Volume (last bar)", 0, 50_000_000, 1_000_000, step=100_000)

    st.subheader("Risk & Trading")
    scanner_only = st.checkbox("Scanner-only mode (no broker calls)", value=True, help="Skip Alpaca account/positions/clock calls and disable order placement.")
    live_trading = st.toggle("Enable ORDER placement (Alpaca PAPER)", value=False, disabled=True if scanner_only else False)
    risk_pct = st.slider("Dollars per trade (% of equity)", 0.0, 5.0, 1.0, 0.25) / 100.0
    max_positions = st.number_input("Max concurrent positions", 1, 50, 5)

    run_scan = st.button("Run Screener Now", type="primary")

# Fetch account/positions
colA, colB, colC = st.columns(3)
with colA:
    st.subheader("Account")
    if not scanner_only and (ALPACA_KEY and ALPACA_SECRET):
        acct = alpaca_get_account()
        st.json({k: acct.get(k) for k in ["status", "cash", "portfolio_value", "equity", "buying_power"] if k in acct})
    else:
        acct = {"status": "scanner-only"}
        st.info("Scanner-only: skipping broker account call")
with colB:
    st.subheader("Clock")
    if not scanner_only and (ALPACA_KEY and ALPACA_SECRET):
        st.json(alpaca_get_clock())
    else:
        st.info("Scanner-only: skipping market clock call")
with colC:
    st.subheader("Open Positions")
    if not scanner_only and (ALPACA_KEY and ALPACA_SECRET):
        st.json(alpaca_list_positions())
    else:
        st.info("Scanner-only: skipping positions call")

# Build rules
rules = Rules(
    ema_fast=ema_fast,
    ema_mid=ema_mid,
    ema_slow=ema_slow,
    require_ema_rising=require_rising,
    macd_fast=macd_fast,
    macd_slow=macd_slow,
    macd_signal=macd_signal,
    macd_hist_must_shrink_neg=macd_shrink,
    macd_line_gt_signal=True,
    kdj_period=kdj_period,
    kdj_j_max=kdj_j_max,
    min_volume=min_volume,
    rsi_min=rsi_min,
    rsi_max=rsi_max,
)

st.markdown("---")

if run_scan:
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    results = []
    buys = []

    progress = st.progress(0)

    for i, sym in enumerate(tickers, start=1):
        try:
            df = get_candles(sym, resolution=resolution, lookback_days=lookback_days)
            res = apply_rules(df, rules)
            last_close = float(df["close"].iloc[-1]) if not df.empty else float('nan')
            reasons = "; ".join(res.get("why", []))
            results.append({
                "symbol": sym,
                "passes": res["passes"],
                "last_close": last_close,
                "reasons": reasons,
            })
            if res["passes"]:
                buys.append((sym, last_close))
            time.sleep(0.12)  # be gentle with API
        except Exception as e:
            results.append({"symbol": sym, "passes": False, "last_close": float('nan'), "reasons": f"error: {e}"})
        progress.progress(i / max(len(tickers), 1))

    df_res = pd.DataFrame(results).sort_values(["passes","symbol"], ascending=[False, True])
    st.subheader("Scan Results")
st.dataframe(df_res, use_container_width=True)
# Offer CSV export in scanner mode
csv_bytes = df_res.to_csv(index=False).encode("utf-8")
st.download_button("Download results as CSV", data=csv_bytes, file_name=f"scan_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

# Trading logic: buy passed names until max_positions is reached; skip if already have position
if buys:
    st.markdown("### Trade Plan")
    # Determine positions held
    held = {p.get("symbol"): float(p.get("qty", 0)) for p in alpaca_list_positions()} if (ALPACA_KEY and ALPACA_SECRET) else {}
    acct_equity = float(acct.get("equity", 0)) if isinstance(acct, dict) else 0.0
    open_slots = max_positions - len(held)
    planned = []
    for sym, px in buys:
        if open_slots <= 0:
            break
        if sym in held:
            continue
        qty = calc_order_size(acct_equity, risk_pct, px)
        if qty > 0:
            planned.append({"symbol": sym, "price": px, "qty": qty})
            open_slots -= 1

    st.json({"planned_orders": planned})

    if planned and (not scanner_only) and live_trading and (ALPACA_KEY and ALPACA_SECRET):
        st.warning("Placing MARKET orders (PAPER)")
        placed = []
        for od in planned:
            resp = alpaca_place_order(od["symbol"], od["qty"], side="buy")
            placed.append({"symbol": od["symbol"], "qty": od["qty"], "resp": resp})
            time.sleep(0.25)
        st.subheader("Order Responses")
        st.json(placed)
    elif planned and not live_trading:
        st.info("Trading disabled. Enable toggle to place PAPER orders.")
    elif not planned:
        st.info("No eligible buys after risk/slots checks.")

st.markdown("---")
st.caption(f"Build finished at {utcnow_iso()} — Remember to keep this on PAPER trading while testing.")
