# streamlit_app.py — Alpaca-only screener/trader with dynamic universe (Top N by $ volume)

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# =========================
# Config: API keys & bases
# =========================
ALPACA_KEY = st.secrets.get("ALPACA_KEY") or os.getenv("ALPACA_KEY", "")
ALPACA_SECRET = st.secrets.get("ALPACA_SECRET") or os.getenv("ALPACA_SECRET", "")
ALPACA_BASE = st.secrets.get("ALPACA_BASE") or os.getenv("ALPACA_BASE", "https://paper-api.alpaca.markets")

ALPACA_HEADERS = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}
ALPACA_DATA_BASE = "https://data.alpaca.markets"  # market data host

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")


# =========================
# Universe helpers
# =========================
@st.cache_data(show_spinner=False)
def default_universe() -> List[str]:
    # small, liquid default set
    return [
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK.B","LLY","JPM",
        "V","XOM","AVGO","UNH","TSLA","WMT","MA","PG","COST","JNJ",
        "HD","MRK","ORCL","ADBE","BAC","PEP","KO","NFLX","CSCO","ABT",
        "CRM","TMUS","LIN","ACN","DHR","CMCSA","MCD","WFC","AMD","TMO",
        "INTU","TXN","DIS","NKE","PM","IBM","RTX","ISRG","UPS","CAT"
    ]

def _alpaca_tf(resolution: str) -> str:
    return {"5":"5Min","15":"15Min","30":"30Min","60":"1Hour","D":"1Day"}.get(resolution, "1Hour")

@st.cache_data(show_spinner=True)
def list_active_us_equities(limit_to:int=1500, require_tradable:bool=True, require_etb:bool=False) -> List[str]:
    """
    Pull active US equities from Alpaca /v2/assets and return a symbol list.
    limit_to: take first N to keep things snappy (the list can be large).
    """
    params = {"status": "active", "asset_class": "us_equity"}
    r = requests.get(f"{ALPACA_BASE}/v2/assets", headers=ALPACA_HEADERS, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Assets error {r.status_code}: {r.text}")
    assets = r.json()
    syms = []
    for a in assets:
        if require_tradable and not a.get("tradable", False):
            continue
        if require_etb and not a.get("easy_to_borrow", False):
            continue
        sym = a.get("symbol")
        if sym and "." not in sym:  # skip weird share classes like BRK.A (Alpaca uses BRK.A/BRK.B, keep .B only)
            syms.append(sym)
        elif sym and sym.endswith(".B"):
            syms.append(sym)
        if len(syms) >= limit_to:
            break
    return syms

def _chunk(lst: List[str], size: int) -> List[List[str]]:
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

@st.cache_data(show_spinner=True)
def top_n_by_dollar_volume(n:int=100, min_price:float=5.0, start_days:int=5, use_intraday:bool=False) -> List[str]:
    """
    Rank symbols by dollar volume using recent bars and return the Top N.
    - If use_intraday=True: 1Hour bars over last 3 days (fresher, but more API work).
    - Else: last 5 x 1Day bars (fast + stable).
    """
    symbols = list_active_us_equities()
    if not symbols:
        return []

    tf = "1Hour" if use_intraday else "1Day"
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=(3 if use_intraday else start_days))

    # Alpaca batch bars: up to ~200 symbols per request
    ranked: List[Tuple[str, float]] = []
    for batch in _chunk(symbols, 200):
        params = {
            "timeframe": tf,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": 1000,
            "feed": "iex",
            "adjustment": "all",
            "symbols": ",".join(batch),
        }
        r = requests.get(f"{ALPACA_DATA_BASE}/v2/stocks/bars", headers=ALPACA_HEADERS, params=params, timeout=40)
        if r.status_code != 200:
            # Skip the batch but keep going
            continue
        payload = r.json()  # {"bars": {"AAPL":[...], "MSFT":[...]}, "next_page_token":...}
        bars_by_sym = payload.get("bars", {})
        for sym, bars in bars_by_sym.items():
            if not bars:
                continue
            df = pd.DataFrame(bars)
            df["t"] = pd.to_datetime(df["t"], utc=True)
            df = df.sort_values("t")
            # filters
            if "c" not in df or "v" not in df:
                continue
            # average of close*volume over period (dollar volume)
            adv = float((df["c"] * df["v"]).mean())
            last_price = float(df["c"].iloc[-1])
            if last_price >= min_price:
                ranked.append((sym, adv))

        # be polite to API
        time.sleep(0.25)

        if len(ranked) > 6000:  # safety cap to keep compute light
            break

    ranked.sort(key=lambda x: x[1], reverse=True)
    top_syms = [s for s, _ in ranked[:n]]
    return top_syms


# =========================
# Market data (bars)
# =========================
def get_candles_alpaca(symbol: str, resolution: str = "60", lookback_days: int = 60) -> pd.DataFrame:
    if not (ALPACA_KEY and ALPACA_SECRET):
        raise RuntimeError("Alpaca keys missing; cannot fetch Alpaca data.")
    tf = _alpaca_tf(resolution)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    url = f"{ALPACA_DATA_BASE}/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": tf,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "limit": 10000,
        "feed": "iex",
        "adjustment": "all",
    }
    r = requests.get(url, headers=ALPACA_HEADERS, params=params, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"Alpaca bars error {r.status_code}: {r.text}")
    bars = r.json().get("bars", [])
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(
        [{"t": pd.to_datetime(b["t"], utc=True), "open": b["o"], "high": b["h"],
          "low": b["l"], "close": b["c"], "volume": b["v"]} for b in bars]
    )
    return df.set_index("t").sort_index()


# =========================
# Indicators & rules
# =========================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(close, fast); ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def stochastic_kd(df: pd.DataFrame, period: int = 14, k_smooth: int = 3, d_smooth: int = 3):
    lowest_low = df["low"].rolling(window=period).min()
    highest_high = df["high"].rolling(window=period).max()
    fast_k = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low)
    k = fast_k.rolling(window=k_smooth).mean()
    d = k.rolling(window=d_smooth).mean()
    return k, d

def kdj(df: pd.DataFrame, period: int = 14):
    k, d = stochastic_kd(df, period=period)
    j = 3 * k - 2 * d
    return k, d, j

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
    result = {
        "passes": False,
        "vol_ok": False,
        "ema_bullish": False,
        "ema_rising": False,
        "macd_ok": False,
        "kdj_ok": False,
        "rsi_ok": False,
        "why": [],
    }
    need = max(rules.ema_slow, rules.kdj_period, rules.macd_slow) + 5
    if df.empty or len(df) < need:
        result["why"].append("insufficient data")
        return result

    df = df.copy()
    df["ema_fast"] = ema(df["close"], rules.ema_fast)
    df["ema_mid"]  = ema(df["close"], rules.ema_mid)
    df["ema_slow"] = ema(df["close"], rules.ema_slow)

    macd_line, macd_signal, hist = macd(df["close"], rules.macd_fast, rules.macd_slow, rules.macd_signal)
    df["macd_line"], df["macd_signal"], df["macd_hist"] = macd_line, macd_signal, hist

    df["rsi14"] = rsi(df["close"], 14)

    k, d = stochastic_kd(df, period=rules.kdj_period)
    j = 3 * k - 2 * d
    df["K"], df["D"], df["J"] = k, d, j

    last, prev = df.iloc[-1], df.iloc[-2]

    # Volume
    result["vol_ok"] = bool(last["volume"] >= rules.min_volume)
    if not result["vol_ok"]:
        result["why"].append(f"volume {int(last['volume'])} < {rules.min_volume}")

    # EMA stack + rising
    result["ema_bullish"] = bool(last["ema_fast"] > last["ema_mid"] > last["ema_slow"])
    if not result["ema_bullish"]:
        result["why"].append("EMA order not bullish (fast>mid>slow)")
    if rules.require_ema_rising:
        result["ema_rising"] = bool(last["ema_fast"] > prev["ema_fast"] and
                                    last["ema_mid"]  > prev["ema_mid"]  and
                                    last["ema_slow"] > prev["ema_slow"])
        if not result["ema_rising"]:
            result["why"].append("EMAs not rising vs previous bar")
    else:
        result["ema_rising"] = True

    # MACD
    macd_hist_shrinking_neg = (last["macd_hist"] > prev["macd_hist"] and last["macd_hist"] < 0) if rules.macd_hist_must_shrink_neg else True
    macd_line_rising_gt_sig = (last["macd_line"] > last["macd_signal"] and last["macd_line"] > prev["macd_line"]) if rules.macd_line_gt_signal else True
    result["macd_ok"] = bool(macd_hist_shrinking_neg and macd_line_rising_gt_sig)
    if not result["macd_ok"]:
        result["why"].append("MACD not in desired state")

    # KDJ
    kdj_bullish_not_hot = (last["K"] > last["D"] and last["K"] > prev["K"] and last["D"] > prev["D"] and j.iloc[-1] > j.iloc[-2] and last["J"] < rules.kdj_j_max)
    result["kdj_ok"] = bool(kdj_bullish_not_hot)
    if not result["kdj_ok"]:
        result["why"].append("KDJ not bullish-but-not-hot")

    # RSI
    result["rsi_ok"] = bool(rules.rsi_min <= last["rsi14"] <= rules.rsi_max)
    if not result["rsi_ok"]:
        result["why"].append(f"RSI {last['rsi14']:.1f} outside [{rules.rsi_min},{rules.rsi_max}]")

    result["passes"] = all([result["vol_ok"], result["ema_bullish"], result["ema_rising"], result["macd_ok"], result["kdj_ok"], result["rsi_ok"]])
    result["last_close"] = float(last["close"])
    return result


# =========================
# Alpaca trading helpers
# =========================
def alpaca_get_account() -> dict:
    r = requests.get(f"{ALPACA_BASE}/v2/account", headers=ALPACA_HEADERS, timeout=20)
    return r.json() if r.status_code == 200 else {"error": r.text}

def alpaca_list_positions() -> List[dict]:
    r = requests.get(f"{ALPACA_BASE}/v2/positions", headers=ALPACA_HEADERS, timeout=20)
    return r.json() if r.status_code == 200 else []

def alpaca_get_clock() -> dict:
    r = requests.get(f"{ALPACA_BASE}/v2/clock", headers=ALPACA_HEADERS, timeout=20)
    return r.json() if r.status_code == 200 else {"is_open": False}

def alpaca_place_order(symbol: str, qty: float, side: str = "buy", order_type: str = "market", tif: str = "day") -> dict:
    payload = {"symbol": symbol, "qty": str(qty), "side": side, "type": order_type, "time_in_force": tif}
    r = requests.post(f"{ALPACA_BASE}/v2/orders", headers=ALPACA_HEADERS, json=payload, timeout=20)
    return r.json() if r.status_code in (200, 201) else {"error": r.text, "status": r.status_code}

def calc_order_size(equity: float, risk_pct: float, price: float) -> int:
    if price <= 0:
        return 0
    dollars = equity * risk_pct
    qty = int(dollars // price)
    return max(qty, 0)


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Auto Screener + Trader (Alpaca)", layout="wide")
st.title("⚡ Auto Screener + Trader — Alpaca only")
st.caption("Retail-friendly swing setup (1h bars) • Dynamic universe supported • Build: 2025-09-16")

# BIG run button in main area (always visible)
run_scan_main = st.button("🔍 Run Screener Now", type="primary")
run_scan = run_scan_main

with st.sidebar:
    st.header("Universe")
    # Source selector
    universe_source = st.selectbox("Universe source", ["Static list", "Top N by $ volume (recent)"], index=0)
    if "universe_cache" not in st.session_state:
        st.session_state.universe_cache = ",".join(default_universe())

    if universe_source == "Static list":
        tickers_input = st.text_area("Universe (comma-separated)", st.session_state.universe_cache, height=120)
        st.session_state.universe_cache = tickers_input
    else:
        n_top = st.slider("Top N (by dollar volume)", 20, 300, 100, step=10)
        min_px = st.number_input("Min last price ($)", 0.0, 1000.0, 5.0)
        intraday_rank = st.checkbox("Use intraday bars (1h) for ranking (slower)", value=False,
                                    help="If off, uses last ~5 daily bars (faster).")
        if st.button("⚡ Load dynamic universe"):
            with st.spinner("Building universe from Alpaca…"):
                try:
                    syms = top_n_by_dollar_volume(n=n_top, min_price=min_px, use_intraday=intraday_rank)
                    if syms:
                        st.session_state.universe_cache = ",".join(syms)
                        st.success(f"Loaded {len(syms)} symbols.")
                    else:
                        st.warning("No symbols returned. Try different filters.")
                except Exception as e:
                    st.error(str(e))
        tickers_input = st.text_area("Universe (comma-separated)", st.session_state.universe_cache, height=120)

    st.header("Bars & Lookback")
    resolution = st.selectbox("Bar timeframe", ["60","30","15","5","D"], index=0)
    lookback_days = st.slider("Lookback days", 20, 180, 60)

    st.header("Rules")
    ema_fast = st.number_input("EMA fast", 3, 50, 5)
    ema_mid  = st.number_input("EMA mid", 5, 100, 20)
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

    st.header("Risk & Trading")
    scanner_only = st.checkbox("Scanner-only mode (no order placement)", value=True)
    place_orders_checkbox = st.checkbox("Enable ORDER placement (Alpaca PAPER)", value=False, disabled=scanner_only)
    live_trading = bool(place_orders_checkbox) and (not scanner_only)
    risk_pct = st.slider("Dollars per trade (% of equity)", 0.0, 5.0, 1.0, 0.25) / 100.0
    max_positions = st.number_input("Max concurrent positions", 1, 50, 5)

# Broker panels (as metrics/tables)
colA, colB, colC = st.columns(3)
with colA:
    st.subheader("Account")
    if (ALPACA_KEY and ALPACA_SECRET):
        acct = alpaca_get_account()
        if "error" in acct:
            st.error(acct["error"])
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Status", acct.get("status", "-"))
            c2.metric("Cash", acct.get("cash", "-"))
            c3.metric("Equity", acct.get("equity", "-"))
            st.caption(f"Buying power: {acct.get('buying_power','-')}")
    else:
        acct = {"status": "no-keys"}
        st.info("Add Alpaca keys to view account.")

with colB:
    st.subheader("Clock")
    if (ALPACA_KEY and ALPACA_SECRET):
        clk = alpaca_get_clock()
        is_open = clk.get("is_open", False)
        next_open = clk.get("next_open", "-")
        next_close = clk.get("next_close", "-")
        c1, c2, c3 = st.columns(3)
        c1.metric("Market open?", "Yes" if is_open else "No")
        c2.metric("Next open", str(next_open))
        c3.metric("Next close", str(next_close))
    else:
        st.info("Add Alpaca keys to view market clock.")

with colC:
    st.subheader("Open Positions")
    if (ALPACA_KEY and ALPACA_SECRET):
        pos = alpaca_list_positions()
        if isinstance(pos, list) and pos:
            pos_df = pd.DataFrame([{
                "symbol": p.get("symbol"),
                "qty": float(p.get("qty", 0)),
                "avg_price": float(p.get("avg_entry_price", 0)),
                "market_value": float(p.get("market_value", 0)),
                "unrealized_pl": float(p.get("unrealized_pl", 0)),
            } for p in pos])
            st.dataframe(pos_df, use_container_width=True)
        else:
            st.info("No open positions.")
    else:
        st.info("Add Alpaca keys to view positions.")

# Build rules
rules = Rules(
    ema_fast=ema_fast, ema_mid=ema_mid, ema_slow=ema_slow, require_ema_rising=require_rising,
    macd_fast=macd_fast, macd_slow=macd_slow, macd_signal=macd_signal,
    macd_hist_must_shrink_neg=macd_shrink, macd_line_gt_signal=True,
    kdj_period=kdj_period, kdj_j_max=kdj_j_max,
    min_volume=min_volume, rsi_min=rsi_min, rsi_max=rsi_max
)

st.markdown("---")

# ====== Run scan ======
if run_scan:
    if not (ALPACA_KEY and ALPACA_SECRET):
        st.error("Alpaca keys missing; set ALPACA_KEY/ALPACA_SECRET/ALPACA_BASE in Secrets."); st.stop()

    tickers = [t.strip().upper() for t in st.session_state.universe_cache.split(",") if t.strip()]
    results_rows = []
    buys: List[Tuple[str, float]] = []

    progress = st.progress(0.0)
    for i, sym in enumerate(tickers, start=1):
        try:
            df = get_candles_alpaca(sym, resolution=resolution, lookback_days=lookback_days)
            res = apply_rules(df, rules)
            rows = {
                "symbol": sym,
                "passes": res["passes"],
                "last_close": res.get("last_close", float("nan")),
                "vol_ok": res["vol_ok"],
                "ema_bullish": res["ema_bullish"],
                "ema_rising": res["ema_rising"],
                "macd_ok": res["macd_ok"],
                "kdj_ok": res["kdj_ok"],
                "rsi_ok": res["rsi_ok"],
                "reasons": "; ".join(res["why"]),
            }
            results_rows.append(rows)
            if res["passes"]:
                buys.append((sym, rows["last_close"]))
            time.sleep(0.12)
        except Exception as e:
            results_rows.append({
                "symbol": sym, "passes": False, "last_close": float("nan"),
                "vol_ok": False, "ema_bullish": False, "ema_rising": False,
                "macd_ok": False, "kdj_ok": False, "rsi_ok": False,
                "reasons": f"error: {e}"
            })
        progress.progress(i / max(len(tickers), 1))

    df_res = pd.DataFrame(results_rows).sort_values(["passes","symbol"], ascending=[False, True])

    st.subheader("Scan Results")
    st.dataframe(df_res, use_container_width=True)

    csv_bytes = df_res.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download results as CSV",
        data=csv_bytes,
        file_name=f"scan_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    # ---- Trade plan (optional) ----
    if buys:
        st.markdown("### Trade Plan")
        held = {p.get("symbol"): float(p.get("qty", 0)) for p in alpaca_list_positions()} if not scanner_only else {}
        acct_equity = 0.0
        if 'acct' in locals() and isinstance(acct, dict):
            try: acct_equity = float(acct.get("equity", 0))
            except: acct_equity = 0.0
        open_slots = max_positions - len(held)
        planned = []
        for sym, px in buys:
            if open_slots <= 0: break
            if sym in held: continue
            qty = calc_order_size(acct_equity, risk_pct, px)
            if qty > 0:
                planned.append({"symbol": sym, "price": px, "qty": qty})
                open_slots -= 1

        st.json({"planned_orders": planned})

        if planned and (not scanner_only) and live_trading:
            st.warning("Placing MARKET orders (PAPER)")
            placed = []
            for od in planned:
                resp = alpaca_place_order(od["symbol"], od["qty"], side="buy")
                placed.append({"symbol": od["symbol"], "qty": od["qty"], "resp": resp})
                time.sleep(0.25)
            st.subheader("Order Responses")
            st.json(placed)
        elif planned and not live_trading:
            st.info("Trading disabled. Enable checkbox to place PAPER orders.")
        elif not planned:
            st.info("No eligible buys after risk/slots checks.")
else:
    st.info("Pick your universe (static or dynamic) and click **Run Screener Now**.")

st.markdown("---")
st.caption(f"Build finished at {utcnow_iso()} — PAPER mode recommended while testing.")
