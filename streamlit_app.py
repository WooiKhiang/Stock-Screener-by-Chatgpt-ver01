# streamlit_app.py — Resilient Market Scanner (Alpaca → Google Sheets)
# Key additions:
# - Fault-tolerant multi-symbol fetch (retry/backoff + split chunk on errors)
# - Soft fetch mode returns partial data instead of crashing
# - Diagnostics include fetch error counters
# - Everything else: Auto/Hourly/Daily/Dual, dynamic H1 lookback, vertical Context, safe Universe writes

import os, json, time, datetime as dt
import numpy as np
import pandas as pd
import pytz
import requests
import streamlit as st

# =========================
# Fixed config
# =========================
GOOGLE_SHEET_ID   = "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4"
TAB_UNIVERSE      = "Universe"
TAB_CONTEXT       = "Context"

ALPACA_KEY    = os.getenv("ALPACA_KEY",    "PKIG445MPT704CN8P0R8")
ALPACA_SECRET = os.getenv("ALPACA_SECRET", "3GQdWcnbMo6V8uvhVa6BK6EbvnH4EHinlsU6uvj4")
ALPACA_BASE   = "https://paper-api.alpaca.markets/v2"
ALPACA_DATA   = "https://data.alpaca.markets"
FEED          = "iex"  # free/paper plan

SA_RAW = st.secrets.get("GCP_SERVICE_ACCOUNT") or os.getenv("GCP_SERVICE_ACCOUNT")

ET = pytz.timezone("America/New_York")
HEADERS = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}

# =========================
# UI
# =========================
st.set_page_config(page_title="Market Scanner — Alpaca → Sheets", layout="wide")
st.title("📊 Market Scanner (Alpaca → Google Sheets)")

autorun = st.sidebar.checkbox("Auto-run every hour", value=True)
try:
    autorefresh = getattr(st, "autorefresh", None) or getattr(st, "experimental_autorefresh", None)
    if autorun and autorefresh:
        autorefresh(interval=60*60*1000, key="hourly_autorefresh")
    elif autorun:
        st.markdown("<script>setTimeout(()=>window.location.reload(),3600000);</script>", unsafe_allow_html=True)
except Exception:
    pass

st.sidebar.markdown("### Mode & Universe")
SCAN_MODE = st.sidebar.selectbox(
    "Scan mode",
    ["Auto (prefer Hourly)", "Hourly", "Daily", "Dual (Daily+Hourly)"],
    index=0
)
UNIVERSE_CAP = st.sidebar.slider("Universe cap (symbols)", 500, 6000, 4000, 250)
MANUAL_RISK  = st.sidebar.selectbox("Risk mode", ["auto", "normal", "tight"], index=0)

with st.sidebar.expander("Step 1 — Hygiene", expanded=True):
    PRICE_MIN = st.number_input("Min price ($)", value=5.0, step=0.5)
    PRICE_MAX = st.number_input("Max price ($)", value=100.0, step=1.0)
    st.caption("Liquidity (pass if either condition holds)")
    LIQ_MIN_AVG_VOL_D1 = st.number_input("Daily AvgVol20 ≥", value=1_000_000, step=100_000, format="%i")
    LIQ_MIN_AVG_DV_D1  = st.number_input("Daily Avg$Vol20 ≥", value=20_000_000, step=1_000_000, format="%i")
    LIQ_MIN_AVG_VOL_H1 = st.number_input("Hourly AvgVol20 ≥", value=100_000, step=25_000, format="%i")
    LIQ_MIN_AVG_DV_H1  = st.number_input("Hourly Avg$Vol20 ≥", value=5_000_000, step=500_000, format="%i")
    REQUIRE_DAILY_TREND = st.checkbox("Daily trend bias (Close ≥ SMA50 & SMA20 ≥ SMA50)", value=False)

with st.sidebar.expander("Step 2 — Activity", expanded=True):
    ATR_PCT_MIN_H1 = st.number_input("ATR% (H1) min", value=0.0010, step=0.0005, format="%.4f")
    ATR_PCT_MAX_H1 = st.number_input("ATR% (H1) max", value=0.0200, step=0.0005, format="%.4f")
    RVOL_MIN_H1    = st.number_input("RVOL (H1) ≥",   value=1.00,    step=0.05,   format="%.2f")
    ATR_PCT_MIN_D1 = st.number_input("ATR% (D1) min", value=0.010, step=0.001, format="%.3f")
    ATR_PCT_MAX_D1 = st.number_input("ATR% (D1) max", value=0.080, step=0.001, format="%.3f")
    RVOL_MIN_D1    = st.number_input("RVOL (D1) ≥",   value=0.90,  step=0.05,  format="%.2f")
    REQ_SMA50_GT_200 = st.checkbox("Require SMA50 > SMA200 (Daily, tight add-on)", value=False)

with st.sidebar.expander("Step 3 — Technical gates", expanded=True):
    GATE_EMA  = st.checkbox("EMA stack rising (EMA5>EMA20>EMA50; all rising)", value=True)
    GATE_MACD = st.checkbox("MACD turn (hist < 0 & rising; line > signal; line rising)", value=True)
    GATE_KDJ  = st.checkbox("KDJ align (K>D; K↑; D↑; J↑)", value=True)

with st.sidebar.expander("Lookback, Fallback & Fetch", expanded=False):
    DAILY_LOOKBACK  = st.slider("Daily lookback (days)", 60, 200, 100)
    HOURLY_LOOK_D   = st.slider("Hourly lookback (days)", 5, 30, 10)
    AUTO_EXTEND_H1  = st.checkbox("Auto-extend hourly lookback if bars insufficient", value=True)
    H1_MIN_BARS_REQ = st.slider("Minimum H1 bars required", 30, 70, 45, 1)
    H1_MAX_LOOK_D   = st.slider("Max hourly lookback (auto-extend cap, days)", 10, 45, 30, 1)
    FETCH_CHUNK     = st.slider("Symbols per request (start)", 40, 180, 140, 10)
    FETCH_SLEEP_MS  = st.slider("Throttle between requests (ms)", 0, 400, 60, 10)
    FETCH_RETRIES   = st.slider("Retries per request (429/5xx)", 0, 6, 4, 1)

SHOW_LIMIT = st.sidebar.slider("Rows to show (UI)", 10, 50, 20)

# =========================
# Helpers
# =========================
def now_et_str() -> str:
    return dt.datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")

def parse_sa(raw_json: str) -> dict:
    if not raw_json:
        raise RuntimeError("Missing GCP_SERVICE_ACCOUNT in secrets/env.")
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        if "-----BEGIN PRIVATE KEY-----" in raw_json and "\\n" not in raw_json:
            start = raw_json.find("-----BEGIN PRIVATE KEY-----")
            end   = raw_json.find("-----END PRIVATE KEY-----", start) + len("-----END PRIVATE KEY-----")
            block = raw_json[start:end]
            raw_json = raw_json.replace(block, block.replace("\r\n","\n").replace("\n","\\n"))
        return json.loads(raw_json)

def zscore(x: pd.Series) -> pd.Series:
    m = x.mean(); s = x.std(ddof=0)
    if s == 0 or not np.isfinite(s): return pd.Series(0.0, index=x.index)
    return (x - m) / s

def rth_only_hour(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    et_times = df["t"].dt.tz_convert(ET)
    mask = (et_times.dt.time >= dt.time(9,30)) & (et_times.dt.time <= dt.time(16,0))
    return df[mask].reset_index(drop=True)

# =========================
# yfinance context
# =========================
try:
    import yfinance as yf
    YF_AVAILABLE = True
except Exception:
    YF_AVAILABLE = False

SECTORS = ["XLF","XLK","XLY","XLP","XLV","XLE","XLI","XLU","XLRE","XLB","IYZ"]
INDUSTRY_ETFS = ["SMH","SOXX","XBI","KRE","ITB","XME","IYT","XOP","OIH","TAN"]
COMMODITY_ETFS = ["GLD","GDX","USO","XOP","OIH"]
MEGACAPS = ["AAPL","MSFT","NVDA","AMZN","GOOG","GOOGL","META","TSLA"]

def _yf_panel(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        panel = df["Adj Close"] if "Adj Close" in lvl0 else (df["Close"] if "Close" in lvl0 else df[lvl0[0]])
    else:
        panel = df
    return panel if isinstance(panel, pd.DataFrame) else panel.to_frame()

def fetch_refs():
    if not YF_AVAILABLE:
        return {"vix": np.nan, "sector1d":{}, "sector5d":{}, "industry1d":{}, "industry5d":{},
                "commodity1d":{}, "commodity5d":{}, "mega1d":{}, "mega5d":{}, "risk_auto":"normal"}
    tickers = ["SPY","^VIX"] + sorted(set(SECTORS + INDUSTRY_ETFS + COMMODITY_ETFS + MEGACAPS))
    data = yf.download(tickers=" ".join(tickers), period="10d", interval="1d", auto_adjust=False, progress=False)
    if data is None or len(data)==0:
        return {"vix": np.nan, "sector1d":{}, "sector5d":{}, "industry1d":{}, "industry5d":{},
                "commodity1d":{}, "commodity5d":{}, "mega1d":{}, "mega5d":{}, "risk_auto":"normal"}
    panel = _yf_panel(data).ffill()
    last = panel.iloc[-1]; prev = panel.iloc[-2] if len(panel)>=2 else last; prev5 = panel.iloc[-5] if len(panel)>=5 else prev
    ret1d = (last/prev - 1.0); ret5d = (last/prev5 - 1.0)
    vix = float(last.get("^VIX", np.nan))
    def sub(keys): 
        return {k: float(ret1d.get(k, np.nan)) for k in keys}, {k: float(ret5d.get(k, np.nan)) for k in keys}
    sec1, sec5 = sub(SECTORS); ind1, ind5 = sub(INDUSTRY_ETFS); com1, com5 = sub(COMMODITY_ETFS)
    mega1 = {m: float(ret1d.get(m, np.nan)) for m in MEGACAPS}
    mega5 = {m: float(ret5d.get(m, np.nan)) for m in MEGACAPS}
    defensives = ["XLV","XLP","XLU"]; cyclicals = ["XLK","XLY","XLF"]
    tight = (vix > 20.0) and (np.nanmean([sec1.get(s,np.nan) for s in defensives]) >
                              np.nanmean([sec1.get(s,np.nan) for s in cyclicals])) and \
                           (np.nanmean([sec5.get(s,np.nan) for s in defensives]) >
                              np.nanmean([sec5.get(s,np.nan) for s in cyclicals]))
    return {"vix": vix, "sector1d": sec1, "sector5d": sec5,
            "industry1d": ind1, "industry5d": ind5,
            "commodity1d": com1, "commodity5d": com5,
            "mega1d": mega1, "mega5d": mega5,
            "risk_auto": "tight" if tight else "normal"}

# =========================
# Alpaca assets & bars (resilient)
# =========================
def fetch_active_symbols(cap: int):
    url = f"{ALPACA_BASE}/assets"
    params = {"status": "active", "asset_class": "us_equity"}
    r = requests.get(url, headers=HEADERS, params=params, timeout=60); r.raise_for_status()
    keep_exch = {"NASDAQ","NYSE","AMEX"}
    syms = [x["symbol"] for x in r.json() if x.get("exchange") in keep_exch and x.get("tradable")]
    bad_suffixes = (".U",".W","WS","W","R",".P","-P")
    syms = [s for s in syms if not s.endswith(bad_suffixes)]
    return syms[:cap]

def _bars_call(params, retries=0):
    """Single HTTP call with backoff on 429/5xx. Returns (json or None, status_code)."""
    url = f"{ALPACA_DATA}/v2/stocks/bars"
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=60)
    except requests.RequestException:
        return None, 599
    if r.status_code in (429, 500, 502, 503, 504) and retries < FETCH_RETRIES:
        # exponential backoff
        delay = (0.4 * (2 ** retries))
        time.sleep(delay)
        return _bars_call(params, retries+1)
    if r.status_code >= 400:
        return {"__error__": r.text[:500]}, r.status_code
    try:
        return r.json(), r.status_code
    except Exception:
        return None, r.status_code

def _fetch_chunk_soft(symbols, timeframe, start_iso, end_iso, limit, throttle_ms, out, fetch_stats):
    """
    Soft fetch: if a chunk fails (400/422/403/413...), split it and keep going.
    - out: dict[symbol] -> DataFrame
    - fetch_stats: dict counters
    """
    if not symbols:
        return
    params = dict(timeframe=timeframe, symbols=",".join(symbols), start=start_iso, end=end_iso,
                  limit=limit, adjustment="raw", feed=FEED)
    js, code = _bars_call(params)
    if code < 400 and js:
        fetch_stats["ok_requests"] += 1
        bars = js.get("bars", [])
        if isinstance(bars, dict):
            for sym, arr in bars.items():
                if not arr: continue
                df = pd.DataFrame(arr)
                df["t"] = pd.to_datetime(df["t"], utc=True)
                df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"}, inplace=True)
                df = df[["t","open","high","low","close","volume"]]
                out[sym] = pd.concat([out.get(sym), df], ignore_index=True)
        else:
            if bars:
                df = pd.DataFrame(bars)
                df["t"] = pd.to_datetime(df["t"], utc=True)
                df.rename(columns={"S":"symbol","o":"open","h":"high","l":"low","c":"close","v":"volume"}, inplace=True)
                df = df[["symbol","t","open","high","low","close","volume"]]
                for sym, grp in df.groupby("symbol"):
                    out[sym] = pd.concat([out.get(sym), grp.drop(columns=["symbol"])], ignore_index=True)
        if throttle_ms > 0:
            time.sleep(throttle_ms/1000.0)
        return

    # Error path
    fetch_stats["http_errors"] += 1
    fetch_stats["last_error_code"] = code
    fetch_stats["last_error_msg"]  = (js or {}).get("__error__", "HTTP error")

    # Split on "bad request" style errors or large payloads; stop at 1 symbol
    if len(symbols) == 1:
        fetch_stats["skipped_symbols"] += 1
        fetch_stats["skipped_list"].append(symbols[0])
        return

    mid = len(symbols)//2
    left, right = symbols[:mid], symbols[mid:]
    _fetch_chunk_soft(left, timeframe, start_iso, end_iso, limit, throttle_ms, out, fetch_stats)
    _fetch_chunk_soft(right, timeframe, start_iso, end_iso, limit, throttle_ms, out, fetch_stats)

def fetch_bars_multi_soft(symbols, timeframe, start_iso, end_iso, start_chunk, limit=1000, throttle_ms=60):
    """
    Resilient multi-symbol fetch:
    - processes in batches of 'start_chunk' symbols
    - splits failing chunks recursively until single-symbol, skipping offenders
    - returns dict[symbol] -> DataFrame, and fetch_stats
    """
    out = {}
    fetch_stats = {"ok_requests":0, "http_errors":0, "skipped_symbols":0, "skipped_list":[], "last_error_code":None, "last_error_msg":None}
    for i in range(0, len(symbols), start_chunk):
        chunk = symbols[i:i+start_chunk]
        _fetch_chunk_soft(chunk, timeframe, start_iso, end_iso, limit, throttle_ms, out, fetch_stats)

    # sort & apply RTH filter for hourly
    for s, df in list(out.items()):
        if df is None or df.empty: 
            continue
        df = df.drop_duplicates(subset=["t"]).sort_values("t").reset_index(drop=True)
        if timeframe.lower() in ("1hour","1h","60min","60m"):
            df = rth_only_hour(df)
        out[s] = df
    return out, fetch_stats

# =========================
# Indicators
# =========================
def sma(x, n): return x.rolling(n, min_periods=n).mean()
def ema(x, n): return x.ewm(span=n, adjust=False).mean()
def atr(df, n=14):
    hi, lo, cl = df["high"], df["low"], df["close"]; pc = cl.shift(1)
    tr = pd.concat([(hi-lo).abs(), (hi-pc).abs(), (lo-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()
def macd(x, fast=12, slow=26, signal=9):
    ef, es = ema(x, fast), ema(x, slow); line = ef - es; sig = ema(line, signal); hist = line - sig
    return line, sig, hist
def kdj(df, n=9, k_period=3, d_period=3):
    low_min  = df["low"].rolling(n, min_periods=n).min()
    high_max = df["high"].rolling(n, min_periods=n).max()
    rsv = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    k = rsv.ewm(alpha=1.0/k_period, adjust=False).mean()
    d = k.ewm(alpha=1.0/d_period, adjust=False).mean()
    j = 3*k - 2*d
    return k, d, j

# =========================
# Risk tweaks
# =========================
def apply_risk_tweaks(risk_mode, rvol_h1, atr_min_h1, atr_max_h1, rvol_d1, atr_min_d1, atr_max_d1, req_sma50_gt200):
    if risk_mode == "tight":
        rvol_h = (rvol_h1 + 0.10)
        atr_min_h = max(atr_min_h1, 0.0015)
        atr_max_h = min(atr_max_h1, 0.0150)
        rvol_d = (rvol_d1 + 0.20)
        atr_min_d = max(atr_min_d1, 0.015)
        atr_max_d = min(atr_max_d1, 0.060)
        return rvol_h, atr_min_h, atr_max_h, rvol_d, atr_min_d, atr_max_d, True or req_sma50_gt200
    return rvol_h1, atr_min_h1, atr_max_h1, rvol_d1, atr_min_d1, atr_max_d1, req_sma50_gt200

# =========================
# Sheets I/O
# =========================
def _open_or_create_ws(gc, title, rows=200, cols=60):
    sh = gc.open_by_key(GOOGLE_SHEET_ID)
    try: return sh.worksheet(title)
    except Exception: return sh.add_worksheet(title=title, rows=rows, cols=cols)

def write_context_hard_replace(metric_value_rows: list):
    import gspread
    from google.oauth2.service_account import Credentials
    sa_info = parse_sa(SA_RAW)
    creds = Credentials.from_service_account_info(sa_info,
               scopes=["https://www.googleapis.com/auth/spreadsheets",
                       "https://www.googleapis.com/auth/drive"])
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(GOOGLE_SHEET_ID)
    temp_name = f"{TAB_CONTEXT}_TMP"
    try:
        try:
            ws_tmp = sh.worksheet(temp_name)
        except Exception:
            ws_tmp = sh.add_worksheet(title=temp_name, rows=500, cols=2)
        df_ctx = pd.DataFrame(metric_value_rows, columns=["Metric","Value"])
        ws_tmp.clear()
        ws_tmp.update("A1", [list(df_ctx.columns)] + df_ctx.fillna("").astype(str).values.tolist(), value_input_option="RAW")
        try:
            ws_old = sh.worksheet(TAB_CONTEXT); sh.del_worksheet(ws_old)
        except Exception:
            pass
        ws_tmp.update_title(TAB_CONTEXT)
    except Exception as e:
        ws = _open_or_create_ws(gc, TAB_CONTEXT)
        ws.clear(); ws.update("A1", [["Metric","Value"]] + metric_value_rows, value_input_option="RAW")

def write_universe_safe(df: pd.DataFrame):
    if df is None or df.empty:
        raise RuntimeError("No rows to write; preserving previous Universe sheet.")
    import gspread
    from google.oauth2.service_account import Credentials
    sa_info = parse_sa(SA_RAW)
    creds = Credentials.from_service_account_info(sa_info,
               scopes=["https://www.googleapis.com/auth/spreadsheets",
                       "https://www.googleapis.com/auth/drive"])
    gc = gspread.authorize(creds)
    ws = _open_or_create_ws(gc, TAB_UNIVERSE)

    try:
        old_vals = ws.get_all_values()
        old_rows = len(old_vals)
        old_cols = max((len(r) for r in old_vals), default=0)
    except Exception:
        old_rows = old_cols = 0

    new_values = [list(df.columns)] + df.fillna("").astype(str).values.tolist()
    ws.update("A1", new_values, value_input_option="RAW")

    try:
        def col_letter(n:int):
            s=""; 
            while n>0:
                n,r = divmod(n-1,26); s = chr(65+r)+s
            return s
        new_rows = len(new_values)
        new_cols = len(new_values[0]) if new_values else 0
        ranges = []
        if old_rows > new_rows and old_cols > 0:
            ranges.append(f"A{new_rows+1}:{col_letter(max(old_cols,new_cols))}")
        if old_cols > new_cols and new_rows > 0:
            ranges.append(f"{col_letter(new_cols+1)}1:{col_letter(old_cols)}{new_rows}")
        if ranges:
            ws.batch_clear(ranges)
    except Exception:
        pass

# =========================
# Pipeline
# =========================
def run_pipeline():
    # Context
    refs = fetch_refs()
    risk_used = MANUAL_RISK if MANUAL_RISK != "auto" else refs.get("risk_auto","normal")
    RVOL_MIN_H1_eff, ATR_MIN_H1_eff, ATR_MAX_H1_eff, RVOL_MIN_D1_eff, ATR_MIN_D1_eff, ATR_MAX_D1_eff, REQ_SMA50_GT_200_eff = \
        apply_risk_tweaks(risk_used, RVOL_MIN_H1, ATR_PCT_MIN_H1, ATR_PCT_MAX_H1, RVOL_MIN_D1, ATR_PCT_MIN_D1, ATR_PCT_MAX_D1, REQ_SMA50_GT_200)

    # Universe
    symbols = fetch_active_symbols(UNIVERSE_CAP)
    total_universe = len(symbols)

    # Lookbacks
    now_utc = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    end_iso = now_utc.isoformat().replace("+00:00","Z")
    hourly_days = HOURLY_LOOK_D
    daily_days  = DAILY_LOOKBACK

    def get_bars(mode):
        need_h = (mode in ["Auto (prefer Hourly)","Hourly","Dual (Daily+Hourly)"])
        need_d = (mode in ["Auto (prefer Hourly)","Daily","Dual (Daily+Hourly)"])
        bars_h = {}; bars_d = {}; min_h = 0
        stat_h = {"ok_requests":0,"http_errors":0,"skipped_symbols":0,"skipped_list":[], "last_error_code":None, "last_error_msg":None}
        stat_d = {"ok_requests":0,"http_errors":0,"skipped_symbols":0,"skipped_list":[], "last_error_code":None, "last_error_msg":None}

        if need_d:
            start_d = (now_utc - dt.timedelta(days=daily_days)).isoformat().replace("+00:00","Z")
            bars_d, stat_d = fetch_bars_multi_soft(symbols, "1Day", start_d, end_iso, FETCH_CHUNK, limit=1000, throttle_ms=FETCH_SLEEP_MS)

        if need_h:
            while True:
                start_h = (now_utc - dt.timedelta(days=hourly_days)).isoformat().replace("+00:00","Z")
                bars_h, stat_h = fetch_bars_multi_soft(symbols, "1Hour", start_h, end_iso, FETCH_CHUNK, limit=1000, throttle_ms=FETCH_SLEEP_MS)
                counts = [len(df) for df in bars_h.values() if df is not None]
                min_h = min(counts) if counts else 0
                if (not AUTO_EXTEND_H1) or (min_h >= H1_MIN_BARS_REQ) or (hourly_days >= H1_MAX_LOOK_D):
                    break
                hourly_days = min(H1_MAX_LOOK_D, hourly_days + 4)
        return bars_h, stat_h, bars_d, stat_d, min_h

    bars_h, stat_h, bars_d, stat_d, min_hbars = get_bars(SCAN_MODE)

    # Auto fallback decision
    mode_used = SCAN_MODE
    if SCAN_MODE == "Auto (prefer Hourly)":
        if min_hbars < H1_MIN_BARS_REQ:
            mode_used = "Daily"
        else:
            mode_used = "Hourly"

    # Big-flow context
    big_rvol_h1 = []
    if bars_h:
        rows = []
        for s, df in bars_h.items():
            if df is None or len(df) < 25: continue
            vol = df["volume"].astype(float).fillna(0)
            base = vol.rolling(20, min_periods=20).mean().iloc[-1]
            if base and base>0:
                rows.append((s, float(vol.iloc[-1]/base)))
        if rows:
            tmp = pd.DataFrame(rows, columns=["symbol","rvol_h1"]).sort_values("rvol_h1", ascending=False)
            big_rvol_h1 = tmp.head(10)["symbol"].tolist()

    big_dv_d1 = []
    if bars_d:
        rows = []
        for s, df in bars_d.items():
            if df is None or len(df) < 2: continue
            close = df["close"].astype(float); vol = df["volume"].astype(float).fillna(0)
            rows.append((s, float(close.iloc[-1]*vol.iloc[-1])))
        if rows:
            tmp = pd.DataFrame(rows, columns=["symbol","dollar_vol_d1"]).sort_values("dollar_vol_d1", ascending=False)
            big_dv_d1 = tmp.head(10)["symbol"].tolist()

    # Screening + diagnostics
    drops = {"insufficient_bars":0, "price_band":0, "liq":0, "daily_trend":0, "atr_rvol":0, "ema":0, "macd":0, "kdj":0}
    step1 = step2 = step3 = 0
    survivors = []

    for s in symbols:
        dfd = bars_d.get(s) if bars_d else None
        dfh = bars_h.get(s) if bars_h else None
        need_d = (mode_used in ["Daily","Dual (Daily+Hourly)"])
        need_h = (mode_used in ["Hourly","Dual (Daily+Hourly)"])

        enough = True
        if need_d and (dfd is None or len(dfd) < 60): enough = False
        if need_h and (dfh is None or len(dfh) < 45): enough = False
        if not enough:
            drops["insufficient_bars"] += 1
            continue

        # Step 1 (Hygiene)
        if mode_used in ["Daily","Dual (Daily+Hourly)"]:
            close_d = dfd["close"].astype(float); vol_d = dfd["volume"].astype(float).fillna(0)
            last_d  = float(close_d.iloc[-1])
            if not (PRICE_MIN <= last_d <= PRICE_MAX): drops["price_band"] += 1; continue
            avg_vol20_d = float(vol_d.rolling(20, min_periods=20).mean().iloc[-1])
            avg_dv20_d  = float((close_d*vol_d).rolling(20, min_periods=20).mean().iloc[-1])
            if not ((avg_vol20_d >= LIQ_MIN_AVG_VOL_D1) or (avg_dv20_d >= LIQ_MIN_AVG_DV_D1)):
                drops["liq"] += 1; continue
            if REQUIRE_DAILY_TREND:
                if not (last_d >= sma(close_d,50).iloc[-1] and sma(close_d,20).iloc[-1] >= sma(close_d,50).iloc[-1]):
                    drops["daily_trend"] += 1; continue
        else:
            close_h = dfh["close"].astype(float); vol_h = dfh["volume"].astype(float).fillna(0)
            last_h  = float(close_h.iloc[-1])
            if not (PRICE_MIN <= last_h <= PRICE_MAX): drops["price_band"] += 1; continue
            avg_vol20_h = float(vol_h.rolling(20, min_periods=20).mean().iloc[-1])
            avg_dv20_h  = float((close_h*vol_h).rolling(20, min_periods=20).mean().iloc[-1])
            if not ((avg_vol20_h >= LIQ_MIN_AVG_VOL_H1) or (avg_dv20_h >= LIQ_MIN_AVG_DV_H1)):
                drops["liq"] += 1; continue
            if REQUIRE_DAILY_TREND and dfd is not None and len(dfd) >= 60:
                cd = dfd["close"].astype(float)
                if not (cd.iloc[-1] >= sma(cd,50).iloc[-1] and sma(cd,20).iloc[-1] >= sma(cd,50).iloc[-1]):
                    drops["daily_trend"] += 1; continue
        step1 += 1

        # Step 2 (Activity)
        if mode_used == "Daily":
            vol_d = dfd["volume"].astype(float).fillna(0)
            base = vol_d.rolling(20, min_periods=20).mean().iloc[-1]
            rvol = float(vol_d.iloc[-1]/base) if base and base>0 else np.nan
            atr14 = float(atr(dfd,14).iloc[-1]); last = float(dfd["close"].iloc[-1])
            atr_pct = float(atr14/last) if last>0 else np.nan
            if not (ATR_MIN_D1_eff <= atr_pct <= ATR_MAX_D1_eff and rvol >= RVOL_MIN_D1_eff):
                drops["atr_rvol"] += 1; continue
            if REQ_SMA50_GT_200_eff and len(dfd) >= 200:
                if not (sma(dfd["close"].astype(float),50).iloc[-1] > sma(dfd["close"].astype(float),200).iloc[-1]):
                    drops["atr_rvol"] += 1; continue
        else:
            vol_h = dfh["volume"].astype(float).fillna(0); close_h = dfh["close"].astype(float)
            base = vol_h.rolling(20, min_periods=20).mean().iloc[-1]
            rvol = float(vol_h.iloc[-1]/base) if base and base>0 else np.nan
            atr14 = float(atr(dfh,14).iloc[-1]); last = float(close_h.iloc[-1])
            atr_pct = float(atr14/last) if last>0 else np.nan
            if not (ATR_MIN_H1_eff <= atr_pct <= ATR_MAX_H1_eff and rvol >= RVOL_MIN_H1_eff):
                drops["atr_rvol"] += 1; continue
            if mode_used == "Dual (Daily+Hourly)" and REQ_SMA50_GT_200_eff and dfd is not None and len(dfd) >= 200:
                if not (sma(dfd["close"].astype(float),50).iloc[-1] > sma(dfd["close"].astype(float),200).iloc[-1]):
                    drops["atr_rvol"] += 1; continue
        step2 += 1

        # Step 3 (Technical gates)
        dfI = dfd if mode_used=="Daily" else dfh
        clI = dfI["close"].astype(float)
        ok = True; ema_ok=macd_ok=kdj_ok=True
        if GATE_EMA:
            e5,e20,e50 = ema(clI,5), ema(clI,20), ema(clI,50)
            ema_ok = (e5.iloc[-1] > e20.iloc[-1] > e50.iloc[-1] and
                      e5.iloc[-1] > e5.iloc[-2] and e20.iloc[-1] > e20.iloc[-2] and e50.iloc[-1] > e50.iloc[-2])
            ok &= ema_ok
        else:
            e5=e20=e50=None
        if GATE_MACD:
            macd_line, macd_sig, macd_hist = macd(clI)
            macd_ok = ((macd_hist.iloc[-1] < 0) and (macd_hist.iloc[-1] > macd_hist.iloc[-2]) and
                       (macd_line.iloc[-1] > macd_sig.iloc[-1]) and (macd_line.iloc[-1] > macd_line.iloc[-2]))
            ok &= macd_ok
        else:
            macd_line, macd_sig, macd_hist = macd(clI)
        if GATE_KDJ:
            k,d_,j = kdj(dfI)
            kdj_ok = (k.iloc[-1] > d_.iloc[-1] and k.iloc[-1] > k.iloc[-2] and d_.iloc[-1] > d_.iloc[-2] and j.iloc[-1] > j.iloc[-2])
            ok &= kdj_ok
        else:
            k,d_,j = kdj(dfI)
        if not ok:
            if not ema_ok: drops["ema"] += 1
            elif not macd_ok: drops["macd"] += 1
            else: drops["kdj"] += 1
            continue
        step3 += 1

        # Features & flags
        roc20 = float(clI.iloc[-1] / clI.iloc[-20] - 1.0) if len(clI) > 20 else np.nan
        notes=[]; flag_d1_ema200=flag_d1_macd0=flag_h1_macd0=False
        if dfd is not None and len(dfd) >= 201:
            cd = dfd["close"].astype(float); e200 = ema(cd,200)
            if cd.iloc[-1] > e200.iloc[-1] and cd.iloc[-2] <= e200.iloc[-2]:
                flag_d1_ema200=True; notes.append("D1_EMA200_breakout")
            m_line_d, m_sig_d, _ = macd(cd)
            if len(m_line_d)>=2 and (m_line_d.iloc[-2] <= 0 < m_line_d.iloc[-1]) and (m_sig_d.iloc[-1] < 0):
                flag_d1_macd0=True; notes.append("D1_MACD_zero_cross_sig_neg")
        if dfh is not None and len(dfh) >= 2:
            ml_h, ms_h, _ = macd(dfh["close"].astype(float))
            if len(ml_h)>=2 and (ml_h.iloc[-2] <= 0 < ml_h.iloc[-1]) and (ms_h.iloc[-1] < 0):
                flag_h1_macd0=True; notes.append("H1_MACD_zero_cross_sig_neg")

        survivors.append({
            "symbol": s,
            "close": float(dfI["close"].iloc[-1]),
            "rvol": float(rvol),
            "atr_pct": float(atr14/last) if last>0 else np.nan,
            "ema5": float(e5.iloc[-1]) if e5 is not None else np.nan,
            "ema20": float(e20.iloc[-1]) if e20 is not None else np.nan,
            "ema50": float(e50.iloc[-1]) if e50 is not None else np.nan,
            "macd_line": float(macd_line.iloc[-1]),
            "macd_signal": float(macd_sig.iloc[-1]),
            "macd_hist": float(macd_hist.iloc[-1]),
            "kdj_k": float(k.iloc[-1]),
            "kdj_d": float(d_.iloc[-1]),
            "kdj_j": float(j.iloc[-1]),
            "roc20": float(roc20),
            "rank_score": 0.0,  # fill later
            "flag_d1_ema200_breakout": flag_d1_ema200,
            "flag_d1_macd_zero_cross_sig_neg": flag_d1_macd0,
            "flag_h1_macd_zero_cross_sig_neg": flag_h1_macd0,
            "notes": ", ".join(notes) if notes else ""
        })

    if not survivors:
        return refs, risk_used, mode_used, total_universe, step1, step2, step3, pd.DataFrame(), pd.DataFrame(), \
               big_rvol_h1, big_dv_d1, drops, min_hbars, stat_h, stat_d

    df = pd.DataFrame(survivors)
    df["inv_atr_pct"] = df["atr_pct"].replace(0, np.nan).rpow(-1)
    df["rank_score"]  = 0.4*zscore(df["roc20"]) + 0.3*zscore(df["rvol"]) + 0.3*zscore(df["inv_atr_pct"])
    df = df.sort_values("rank_score", ascending=False).reset_index(drop=True)

    df_ui  = df.head(SHOW_LIMIT).copy()
    df_out = df.head(20).copy()
    as_of = now_et_str()
    df_out.insert(0,"risk_mode", risk_used)
    df_out.insert(0,"as_of_date", as_of)

    cols = ["as_of_date","risk_mode","symbol","close","rvol","atr_pct",
            "ema5","ema20","ema50","macd_line","macd_signal","macd_hist",
            "kdj_k","kdj_d","kdj_j","roc20","rank_score",
            "flag_d1_ema200_breakout","flag_d1_macd_zero_cross_sig_neg","flag_h1_macd_zero_cross_sig_neg","notes"]
    for c in cols:
        if c not in df_out.columns: df_out[c] = np.nan
    df_out = df_out[cols]

    return refs, risk_used, mode_used, total_universe, step1, step2, step3, df_ui, df_out, \
           big_rvol_h1, big_dv_d1, drops, min_hbars, stat_h, stat_d

# =========================
# RUN
# =========================
left, right = st.columns([2,1])

with st.spinner("Running scan…"):
    refs, risk_used, mode_used, total_universe, s1, s2, s3, df_ui, df_out, big_rvol_h1, big_dv_d1, drops, min_hbars, stat_h, stat_d = run_pipeline()

# Context panel
with left:
    st.subheader("Market Context")
    vix_val = refs.get("vix", np.nan)
    st.write(
        f"**VIX:** {vix_val:.2f}" if isinstance(vix_val,(int,float)) and np.isfinite(vix_val) else "**VIX:** n/a",
        f" | **Risk (auto):** `{refs.get('risk_auto','normal')}`",
        f" | **Using:** `{risk_used}`",
        f" | **Mode:** `{mode_used}`"
    )
    if refs.get("sector1d"):
        sect_df = pd.DataFrame({"Sector": list(refs["sector1d"].keys()),
                                "1D": list(refs["sector1d"].values()),
                                "5D": [refs["sector5d"].get(k, np.nan) for k in refs["sector1d"].keys()]})
        st.dataframe(sect_df.sort_values("1D", ascending=False).style.format({"1D":"{:.2%}","5D":"{:.2%}"}),
                     use_container_width=True)

with right:
    st.subheader("Run Summary")
    st.write(f"**As of (ET):** {now_et_str()}")
    st.write(f"**Universe cap:** {UNIVERSE_CAP:,}")
    st.write(f"**Universe scanned:** {total_universe:,}")
    st.write(f"**Step1/2/3 survivors:** {s1:,} / {s2:,} / {s3:,}")
    st.write(f"**Min H1 bars (fetched):** {min_hbars}")

# UI table
st.subheader("Top Candidates (preview)")
if df_ui.empty:
    st.warning("No survivors under current gates.")
else:
    num_cols = ["close","rvol","atr_pct","ema5","ema20","ema50","macd_line","macd_signal","macd_hist",
                "kdj_k","kdj_d","kdj_j","roc20","rank_score"]
    for c in num_cols:
        if c in df_ui.columns: df_ui[c] = pd.to_numeric(df_ui[c], errors="coerce")
    fmt = {c:"{:.2f}" for c in num_cols if c in df_ui.columns}
    st.dataframe(df_ui.head(SHOW_LIMIT).style.format(fmt), use_container_width=True)

# Build Context (vertical; readable) and write (hard refresh)
ctx_rows = []
ctx_rows.append(["Timestamp (ET)", now_et_str()])
ctx_rows.append(["Risk mode (auto)", refs.get("risk_auto","")])
ctx_rows.append(["Risk mode (using)", risk_used])
ctx_rows.append(["Scan mode (effective)", mode_used])
ctx_rows.append(["Universe cap", f"{UNIVERSE_CAP}"])
ctx_rows.append(["Universe scanned", f"{total_universe}"])
ctx_rows.append(["Step 1 survivors", f"{s1}"])
ctx_rows.append(["Step 2 survivors", f"{s2}"])
ctx_rows.append(["Step 3 survivors", f"{s3}"])
ctx_rows.append(["VIX level", f"{refs.get('vix', np.nan):.2f}" if isinstance(refs.get('vix'),(int,float)) and np.isfinite(refs.get('vix')) else "n/a"])

def add_bucket(title, d1:dict, d5:dict):
    ctx_rows.append([title, ""])
    for k in sorted(d1.keys()):
        v1 = d1.get(k, np.nan); v5 = d5.get(k, np.nan)
        v1s = f"{v1:.2%}" if isinstance(v1,(int,float)) and np.isfinite(v1) else "n/a"
        v5s = f"{v5:.2%}" if isinstance(v5,(int,float)) and np.isfinite(v5) else "n/a"
        ctx_rows.append([f"  {k}", f"1D {v1s} | 5D {v5s}"])

add_bucket("Sectors",   refs.get("sector1d",{}),   refs.get("sector5d",{}))
add_bucket("Industries",refs.get("industry1d",{}), refs.get("industry5d",{}))
add_bucket("Gold/Oil",  refs.get("commodity1d",{}),refs.get("commodity5d",{}))

if big_rvol_h1: ctx_rows.append(["Top-10 RVOL (Hourly)", ", ".join(big_rvol_h1)])
if big_dv_d1:  ctx_rows.append(["Top-10 Dollar Volume (Daily)", ", ".join(big_dv_d1)])

try:
    write_context_hard_replace(ctx_rows)
    st.success("Context tab refreshed.")
except Exception as e:
    st.error(f"Failed to write Context: {e}")

# Write Universe
try:
    if df_out.empty:
        st.warning("No rows to write — Universe left unchanged.")
    else:
        write_universe_safe(df_out)
        st.success("Universe tab updated.")
except Exception as e:
    st.error(f"Failed to write Universe (previous data preserved): {e}")

# Diagnostics
with st.expander("🛠 Diagnostics", expanded=False):
    st.markdown("**Drop reasons (counts)**")
    st.json(drops)

    st.markdown("**Fetch stats — Hourly**")
    st.json({k:v for k,v in stat_h.items() if k != "skipped_list"})
    if stat_h.get("skipped_list"):
        st.write("Skipped symbols (H1, due to repeated errors):", ", ".join(stat_h["skipped_list"][:50]), ("…more" if len(stat_h["skipped_list"])>50 else ""))

    st.markdown("**Fetch stats — Daily**")
    st.json({k:v for k,v in stat_d.items() if k != "skipped_list"})
    if stat_d.get("skipped_list"):
        st.write("Skipped symbols (D1):", ", ".join(stat_d["skipped_list"][:50]), ("…more" if len(stat_d["skipped_list"])>50 else ""))

    probe = st.text_input("Probe symbol (e.g., AAPL)", value="AAPL").strip().upper()
    if probe:
        bars_h = None; bars_d = None  # avoid accidental leaks
        st.write("Use the main table to sanity-check indicator values for the probe symbol.")
