# streamlit_app.py — Auto/Dual/Hourly/Daily scan (Alpaca) → Google Sheets
# Robust Alpaca bars fetching with retries, chunk-down, per-symbol fallback, feed/timeframe fallbacks.
# Diagnostics shows last data error. Context vertical. Universe transactional overwrite.

import os, json, time, datetime as dt
import numpy as np
import pandas as pd
import pytz
import requests
import streamlit as st

# ========= Fixed config (IDs, endpoints) =========
GOOGLE_SHEET_ID   = "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4"
TAB_UNIVERSE      = "Universe"
TAB_CONTEXT       = "Context"

ALPACA_KEY    = os.getenv("ALPACA_KEY",    "PKIG445MPT704CN8P0R8")
ALPACA_SECRET = os.getenv("ALPACA_SECRET", "3GQdWcnbMo6V8uvhVa6BK6EbvnH4EHinlsU6uvj4")
ALPACA_BASE   = "https://paper-api.alpaca.markets/v2"
ALPACA_DATA   = "https://data.alpaca.markets"
FEED          = "iex"  # free/paper default

SA_RAW = st.secrets.get("GCP_SERVICE_ACCOUNT") or os.getenv("GCP_SERVICE_ACCOUNT")

ET = pytz.timezone("America/New_York")
HEADERS = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}

# ========= UI =========
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
    index=0,
    help="Auto: try Hourly; if insufficient bars, fallback to Daily automatically."
)
UNIVERSE_CAP = st.sidebar.slider("Universe cap (symbols)", 500, 6000, 4000, 250,
    help="Active & tradable US equities (NASDAQ/NYSE/AMEX). Higher = broader but slower.")
MANUAL_RISK  = st.sidebar.selectbox("Risk mode", ["auto", "normal", "tight"], index=0)

with st.sidebar.expander("Step 1 — Hygiene", expanded=True):
    PRICE_MIN = st.number_input("Min price ($)", value=5.0, step=0.5)
    PRICE_MAX = st.number_input("Max price ($)", value=100.0, step=1.0)
    st.caption("Liquidity (pass if either condition holds)")
    LIQ_MIN_AVG_VOL_D1 = st.number_input("Daily AvgVol20 ≥", value=1_000_000, step=100_000, format="%i")
    LIQ_MIN_AVG_DV_D1  = st.number_input("Daily Avg$Vol20 ≥", value=20_000_000, step=1_000_000, format="%i")
    LIQ_MIN_AVG_VOL_H1 = st.number_input("Hourly AvgVol20 ≥", value=100_000, step=25_000, format="%i")
    LIQ_MIN_AVG_DV_H1  = st.number_input("Hourly Avg$Vol20 ≥", value=5_000_000, step=500_000, format="%i")
    REQUIRE_DAILY_TREND = st.checkbox("Daily trend bias (Close ≥ SMA50 & SMA20 ≥ SMA50)", value=False,
        help="OFF keeps pure hourly viable. In Dual mode, Daily trend acts as Step 1 structure.")

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

with st.sidebar.expander("Lookback & Fallback", expanded=False):
    DAILY_LOOKBACK  = st.slider("Daily lookback (days)", 60, 200, 100,
        help="Used for Daily scan + flags + context.")
    HOURLY_LOOK_D   = st.slider("Hourly lookback (days)", 5, 30, 10,
        help="~65–70 RTH hours over ~10 trading days.")
    AUTO_EXTEND_H1  = st.checkbox("Auto-extend hourly lookback if bars insufficient", value=True)
    H1_MIN_BARS_REQ = st.slider("Minimum H1 bars required", 30, 70, 45, 1,
        help="If below in Auto/Hourly/Dual, we'll extend lookback; Auto may fallback to Daily.")
    H1_MAX_LOOK_D   = st.slider("Max hourly lookback when auto-extending (days)", 10, 45, 30, 1)

SHOW_LIMIT = st.sidebar.slider("Rows to show (UI)", 10, 50, 20)

# ========= Helpers =========
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

# ========= yfinance context =========
try:
    import yfinance as yf
    YF_AVAILABLE = True
except Exception:
    YF_AVAILABLE = False

SECTORS = ["XLF","XLK","XLY","XLP","XLV","XLE","XLI","XLU","XLRE","XLB","IYZ"]
INDUSTRY_ETFS = ["SMH","SOXX","XBI","KRE","ITB","XME","IYT","XOP","OIH","TAN"]
COMMODITY_ETFS = ["GLD","GDX","USO","XOP","OIH"]
MEGACAPS = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA"]  # keep GOOGL (drop GOOG dup)

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

# ========= Alpaca I/O =========
def fetch_active_symbols(cap: int):
    url = f"{ALPACA_BASE}/assets"
    params = {"status": "active", "asset_class": "us_equity"}
    r = requests.get(url, headers=HEADERS, params=params, timeout=60); r.raise_for_status()
    keep_exch = {"NASDAQ","NYSE","AMEX"}
    syms = [x["symbol"] for x in r.json() if x.get("exchange") in keep_exch and x.get("tradable")]
    bad_suffixes = (".U",".W","WS","W","R",".P","-P")
    syms = [s for s in syms if not s.endswith(bad_suffixes)]
    return syms[:cap]

_last_data_error = {"where":"", "status":None, "message":""}  # shown in Diagnostics

def fetch_bars_multi(symbols, timeframe, start_iso, end_iso, limit=1000, chunk_size=150, max_retries=3):
    """
    Robust multi-symbol bars fetch:
      - retries with exponential backoff (429/5xx)
      - chunk-down on 400/422
      - per-symbol fallback for a failing chunk
      - feed/timeframe fallbacks
    Returns: dict[symbol] -> DataFrame(t, open, high, low, close, volume)
    """
    base = f"{ALPACA_DATA}/v2/stocks/bars"
    out = {}
    tf_candidates = [timeframe]
    if timeframe.lower() in ("1hour","1h","60min","60m"):
        tf_candidates = [timeframe, "1H", "1Hour"]
    elif timeframe.lower() in ("1day","1d"):
        tf_candidates = [timeframe, "1Day"]

    def parse_and_merge(js, _tf):
        bars = js.get("bars", [])
        if isinstance(bars, dict):
            for sym, arr in bars.items():
                if not arr: 
                    if sym not in out: out[sym] = pd.DataFrame(columns=["t","open","high","low","close","volume"])
                    continue
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

    def do_request(params):
        # Try with feed; on entitlement issues, retry without feed once
        tried = []
        for try_feed in [params.get("feed"), None]:
            p = dict(params)
            if try_feed is None:
                p.pop("feed", None)
            tried.append("feed="+("none" if try_feed is None else str(try_feed)))
            for attempt in range(max_retries):
                r = requests.get(base, headers=HEADERS, params=p, timeout=60)
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(1.2 * (attempt+1))
                    continue
                if r.status_code >= 400:
                    try:
                        js = r.json()
                        msg = js.get("message") or js.get("error") or r.text[:300]
                    except Exception:
                        msg = r.text[:300]
                    _last_data_error.update({"where": f"/bars ({','.join(tried)})", "status": r.status_code, "message": str(msg)})
                    return None, r.status_code
                try:
                    return r.json(), None
                except Exception:
                    _last_data_error.update({"where": f"/bars parse ({','.join(tried)})", "status": r.status_code, "message": "JSON parse error"})
                    return None, r.status_code
        return None, 400

    i = 0
    while i < len(symbols):
        chunk = symbols[i:i+chunk_size]
        ok = False
        # Try timeframe variants first
        for tf in tf_candidates:
            params = dict(timeframe=tf, symbols=",".join(chunk), start=start_iso, end=end_iso,
                          limit=limit, adjustment="raw")
            if FEED: params["feed"] = FEED
            page = None
            # Page loop
            out_before = dict(out)
            while True:
                if page: params["page_token"] = page
                js, err = do_request(params)
                if js is None:
                    break
                parse_and_merge(js, tf)
                page = js.get("next_page_token")
                if not page:
                    ok = True
                    break
            if ok:
                break

        if not ok:
            # If the multi-call failed, try chunk-down first
            if chunk_size > 60:
                chunk_size = max(60, chunk_size // 2)
                continue
            # As a last resort, per-symbol fallback in this chunk
            for sym in chunk:
                for tf in tf_candidates:
                    params = dict(timeframe=tf, symbols=sym, start=start_iso, end=end_iso,
                                  limit=limit, adjustment="raw")
                    if FEED: params["feed"] = FEED
                    page = None; merged = False
                    for attempt in range(max_retries):
                        if page: params["page_token"] = page
                        js, err = do_request(params)
                        if js is None:
                            break
                        parse_and_merge(js, tf)
                        page = js.get("next_page_token")
                        if not page:
                            merged = True
                            break
                    if merged:
                        break
                # ensure key exists even if empty
                if sym not in out:
                    out[sym] = pd.DataFrame(columns=["t","open","high","low","close","volume"])
        i += chunk_size

    # sort & RTH filter for hourly
    for s, df in list(out.items()):
        if df is None or df.empty:
            continue
        df = df.drop_duplicates(subset=["t"]).sort_values("t").reset_index(drop=True)
        # Map timeframe string to an hourly check
        if any(x in str(timeframe).lower() for x in ["hour","1h","60m","60min"]):
            df = rth_only_hour(df)
        out[s] = df
    return out

# ========= Indicators =========
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

# ========= Risk-mode adjustments =========
def apply_risk_tweaks(mode_used, rvol_h1, atr_min_h1, atr_max_h1, rvol_d1, atr_min_d1, atr_max_d1, req_sma50_gt200):
    if MANUAL_RISK != "auto":
        risk_used = MANUAL_RISK
    else:
        risk_used = None
    def tweak(risk):
        if risk == "tight":
            rvol_h = (rvol_h1 + 0.10)
            atr_min_h = max(atr_min_h1, 0.0015)
            atr_max_h = min(atr_max_h1, 0.0150)
            rvol_d = (rvol_d1 + 0.20)
            atr_min_d = max(atr_min_d1, 0.015)
            atr_max_d = min(atr_max_d1, 0.060)
            return rvol_h, atr_min_h, atr_max_h, rvol_d, atr_min_d, atr_max_d, True or req_sma50_gt200
        else:
            return rvol_h1, atr_min_h1, atr_max_h1, rvol_d1, atr_min_d1, atr_max_d1, req_sma50_gt200
    return risk_used, tweak

# ========= Context writer (hard-refresh) =========
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
            ws_old = sh.worksheet(TAB_CONTEXT)
            sh.del_worksheet(ws_old)
        except Exception:
            pass
        ws_tmp.update_title(TAB_CONTEXT)
    except Exception:
        ws = _open_or_create_ws(gc, TAB_CONTEXT)
        ws.clear()
        ws.update("A1", [["Metric","Value"]] + metric_value_rows, value_input_option="RAW")

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

# ========= Pipeline =========
def run_pipeline():
    refs = fetch_refs()
    risk_auto = refs.get("risk_auto","normal")
    risk_used = MANUAL_RISK if MANUAL_RISK != "auto" else risk_auto

    risk_placeholder, tweak = apply_risk_tweaks(
        SCAN_MODE, RVOL_MIN_H1, ATR_PCT_MIN_H1, ATR_PCT_MAX_H1,
        RVOL_MIN_D1, ATR_PCT_MIN_D1, ATR_PCT_MAX_D1, REQ_SMA50_GT_200
    )
    RVOL_MIN_H1_eff, ATR_MIN_H1_eff, ATR_MAX_H1_eff, RVOL_MIN_D1_eff, ATR_MIN_D1_eff, ATR_MAX_D1_eff, REQ_SMA50_GT_200_eff = \
        tweak(risk_used)

    symbols = fetch_active_symbols(UNIVERSE_CAP)
    total_universe = len(symbols)

    now_utc = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    hourly_days = HOURLY_LOOK_D
    daily_days  = DAILY_LOOKBACK
    end_iso     = now_utc.isoformat().replace("+00:00","Z")

    def get_bars(mode):
        nonlocal hourly_days
        need_h = (mode in ["Auto (prefer Hourly)","Hourly","Dual (Daily+Hourly)"])
        need_d = (mode in ["Auto (prefer Hourly)","Daily","Dual (Daily+Hourly)"])
        bars_h = {}
        bars_d = {}

        if need_d:
            start_d = (now_utc - dt.timedelta(days=daily_days)).isoformat().replace("+00:00","Z")
            # Daily is cheap; chunk default is fine
            bars_d = fetch_bars_multi(symbols, "1Day", start_d, end_iso)

        if need_h:
            # Loop with auto-extend on minimal bars
            while True:
                start_h = (now_utc - dt.timedelta(days=hourly_days)).isoformat().replace("+00:00","Z")
                bars_h = fetch_bars_multi(symbols, "1Hour", start_h, end_iso)
                counts = [len(df) for df in bars_h.values() if df is not None]
                min_h = min(counts) if counts else 0
                if (not AUTO_EXTEND_H1) or (min_h >= H1_MIN_BARS_REQ) or (hourly_days >= H1_MAX_LOOK_D):
                    return bars_h, bars_d, min_h
                hourly_days = min(H1_MAX_LOOK_D, hourly_days + 4)

        return bars_h, bars_d, 0

    bars_h, bars_d, min_hbars = get_bars(SCAN_MODE)

    effective_mode = SCAN_MODE
    if SCAN_MODE == "Auto (prefer Hourly)":
        effective_mode = "Hourly" if min_hbars >= H1_MIN_BARS_REQ else "Daily"

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

    drops = {
        "insufficient_bars":0, "price_band":0, "liq":0, "daily_trend":0,
        "atr_rvol":0, "ema":0, "macd":0, "kdj":0
    }
    step1 = step2 = step3 = 0
    survivors = []

    for s in symbols:
        dfd = bars_d.get(s) if bars_d else None
        dfh = bars_h.get(s) if bars_h else None

        need_d = (effective_mode in ["Daily","Dual (Daily+Hourly)"])
        need_h = (effective_mode in ["Hourly","Dual (Daily+Hourly)"])

        enough = True
        if need_d and (dfd is None or len(dfd) < 60): enough = False
        if need_h and (dfh is None or len(dfh) < 45): enough = False
        if not enough:
            drops["insufficient_bars"] += 1
            continue

        # Step 1
        if effective_mode in ["Daily","Dual (Daily+Hourly)"]:
            close_d = dfd["close"].astype(float); vol_d = dfd["volume"].astype(float).fillna(0)
            last_d  = float(close_d.iloc[-1])
            if not (PRICE_MIN <= last_d <= PRICE_MAX):
                drops["price_band"] += 1; continue
            avg_vol20_d = float(vol_d.rolling(20, min_periods=20).mean().iloc[-1])
            avg_dv20_d  = float((close_d*vol_d).rolling(20, min_periods=20).mean().iloc[-1])
            if not ((avg_vol20_d >= LIQ_MIN_AVG_VOL_D1) or (avg_dv20_d >= LIQ_MIN_AVG_DV_D1)):
                drops["liq"] += 1; continue
            if REQUIRE_DAILY_TREND:
                sma20_d = float(sma(close_d,20).iloc[-1]); sma50_d = float(sma(close_d,50).iloc[-1])
                if not (last_d >= sma50_d and sma20_d >= sma50_d):
                    drops["daily_trend"] += 1; continue
        else:
            close_h = dfh["close"].astype(float); vol_h = dfh["volume"].astype(float).fillna(0)
            last_h  = float(close_h.iloc[-1])
            if not (PRICE_MIN <= last_h <= PRICE_MAX):
                drops["price_band"] += 1; continue
            avg_vol20_h = float(vol_h.rolling(20, min_periods=20).mean().iloc[-1])
            avg_dv20_h  = float((close_h*vol_h).rolling(20, min_periods=20).mean().iloc[-1])
            if not ((avg_vol20_h >= LIQ_MIN_AVG_VOL_H1) or (avg_dv20_h >= LIQ_MIN_AVG_DV_H1)):
                drops["liq"] += 1; continue
            if REQUIRE_DAILY_TREND:
                if dfd is None or len(dfd)<60:
                    drops["daily_trend"] += 1; continue
                cd = dfd["close"].astype(float)
                if not (cd.iloc[-1] >= sma(cd,50).iloc[-1] and sma(cd,20).iloc[-1] >= sma(cd,50).iloc[-1]):
                    drops["daily_trend"] += 1; continue
        step1 += 1

        # Step 2
        if effective_mode in ["Daily"]:
            atr14 = float(atr(dfd,14).iloc[-1]); last = float(dfd["close"].iloc[-1])
            atr_pct = float(atr14 / last) if last>0 else np.nan
            vol_d = dfd["volume"].astype(float).fillna(0)
            base = vol_d.rolling(20, min_periods=20).mean().iloc[-1]
            rvol = float(vol_d.iloc[-1]/base) if base and base>0 else np.nan
            sma50 = float(sma(dfd["close"].astype(float),50).iloc[-1]); sma200 = float(sma(dfd["close"].astype(float),200).iloc[-1]) if len(dfd)>=200 else np.nan
            if not (ATR_MIN_D1_eff <= atr_pct <= ATR_MAX_D1_eff and rvol >= RVOL_MIN_D1_eff):
                drops["atr_rvol"] += 1; continue
            if REQ_SMA50_GT_200_eff and not (sma50 > sma200):
                drops["atr_rvol"] += 1; continue
        else:
            use = dfh
            close_h = use["close"].astype(float); vol_h = use["volume"].astype(float).fillna(0)
            last = float(close_h.iloc[-1])
            atr14 = float(atr(use,14).iloc[-1]); atr_pct = float(atr14/last) if last>0 else np.nan
            base = vol_h.rolling(20, min_periods=20).mean().iloc[-1]
            rvol = float(vol_h.iloc[-1]/base) if base and base>0 else np.nan
            if not (ATR_MIN_H1_eff <= atr_pct <= ATR_MAX_H1_eff and rvol >= RVOL_MIN_H1_eff):
                drops["atr_rvol"] += 1; continue
            if effective_mode == "Dual (Daily+Hourly)" and REQ_SMA50_GT_200_eff:
                if dfd is None or len(dfd)<200:
                    drops["atr_rvol"] += 1; continue
                cd = dfd["close"].astype(float)
                if not (sma(cd,50).iloc[-1] > sma(cd,200).iloc[-1]):
                    drops["atr_rvol"] += 1; continue
        step2 += 1

        # Step 3
        dfI = dfd if effective_mode=="Daily" else dfh
        clI = dfI["close"].astype(float)
        ok = True; ema_ok=True; macd_ok=True; kdj_ok=True
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
            elif not kdj_ok: drops["kdj"] += 1
            else: drops["kdj"] += 1
            continue
        step3 += 1

        # Features & flags
        roc20 = float(clI.iloc[-1] / clI.iloc[-20] - 1.0) if len(clI) > 20 else np.nan
        inv_atr = (1.0 / (atr14/last)) if (last>0 and atr14>0) else np.nan

        notes=[]; flag_d1_ema200=False; flag_d1_macd0=False; flag_h1_macd0=False
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

        row = {
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
            "flag_d1_ema200_breakout": flag_d1_ema200,
            "flag_d1_macd_zero_cross_sig_neg": flag_d1_macd0,
            "flag_h1_macd_zero_cross_sig_neg": flag_h1_macd0,
            "notes": ", ".join(notes) if notes else ""
        }
        survivors.append(row)

    if not survivors:
        return refs, risk_used, effective_mode, total_universe, step1, step2, step3, pd.DataFrame(), pd.DataFrame(), [], [], {"drops":drops, "last_error":_last_data_error}, bars_h, bars_d, min_hbars

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

    return refs, risk_used, effective_mode, total_universe, step1, step2, step3, df_ui, df_out, big_rvol_h1, big_dv_d1, {"drops":drops, "last_error":_last_data_error}, bars_h, bars_d, min_hbars

# ========= RUN =========
left, right = st.columns([2,1])

with st.spinner("Running scan…"):
    refs, risk_used, mode_used, total_universe, s1, s2, s3, df_ui, df_out, big_rvol_h1, big_dv_d1, diag, bars_h, bars_d, min_hbars = run_pipeline()

# ---- Context ----
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
    st.write(f"**Min H1 bars (fetched):** {min_hbars if bars_h else 0}")

# ---- Candidates table ----
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

# ---- Context write (vertical, hard refresh) ----
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

# ---- Universe write ----
try:
    if df_out.empty:
        st.warning("No rows to write — Universe left unchanged.")
    else:
        write_universe_safe(df_out)
        st.success("Universe tab updated.")
except Exception as e:
    st.error(f"Failed to write Universe (previous data preserved): {e}")

# ========= Diagnostics =========
with st.expander("🛠 Diagnostics", expanded=False):
    st.markdown("**Drop reasons (counts)**")
    st.json(diag.get("drops", {}))
    le = diag.get("last_error", {})
    if le and le.get("status") is not None:
        st.markdown("**Last data error**")
        st.write(f"Where: `{le.get('where')}`")
        st.write(f"HTTP: `{le.get('status')}`")
        st.code(str(le.get("message"))[:500])
    if bars_h:
        counts = [len(df) for df in bars_h.values() if df is not None]
        if counts:
            st.write(f"H1 bars — min/median/max: **{int(np.min(counts))} / {int(np.median(counts))} / {int(np.max(counts))}**")
    if bars_d:
        counts = [len(df) for df in bars_d.values() if df is not None]
        if counts:
            st.write(f"D1 bars — min/median/max: **{int(np.min(counts))} / {int(np.median(counts))} / {int(np.max(counts))}**")

    probe = st.text_input("Probe symbol (e.g., AAPL)", value="AAPL").strip().upper()
    if probe:
        try:
            dfh = bars_h.get(probe); dfd = bars_d.get(probe)
            st.write(f"**{probe}** — H1 bars: {len(dfh) if isinstance(dfh,pd.DataFrame) else 0}, D1 bars: {len(dfd) if isinstance(dfd,pd.DataFrame) else 0}")
            if isinstance(dfh, pd.DataFrame) and not dfh.empty:
                ch = dfh["close"].astype(float); vh = dfh["volume"].astype(float).fillna(0)
                last = float(ch.iloc[-1]); atr14 = float(atr(dfh,14).iloc[-1]); base = vh.rolling(20, min_periods=20).mean().iloc[-1]
                rvol = float(vh.iloc[-1]/base) if base and base>0 else np.nan
                st.write(f"H1 close: {last:.2f} | ATR%: {((atr14/last)*100):.2f}% | RVOL_hour: {rvol:.2f}")
                if GATE_EMA:
                    e5,e20,e50 = ema(ch,5), ema(ch,20), ema(ch,50)
                    st.write(f"EMA5:{e5.iloc[-1]:.2f}  EMA20:{e20.iloc[-1]:.2f}  EMA50:{e50.iloc[-1]:.2f}")
                if GATE_MACD:
                    ml, ms, mh = macd(ch)
                    st.write(f"MACD line:{ml.iloc[-1]:.3f}  signal:{ms.iloc[-1]:.3f}  hist:{mh.iloc[-1]:.3f}")
            if isinstance(dfd, pd.DataFrame) and not dfd.empty:
                cd = dfd["close"].astype(float); vd = dfd["volume"].astype(float).fillna(0)
                base = vd.rolling(20, min_periods=20).mean().iloc[-1]
                rvol_d = float(vd.iloc[-1]/base) if base and base>0 else np.nan
                st.write(f"D1 close: {cd.iloc[-1]:.2f} | RVOL_today: {rvol_d:.2f}")
        except Exception as e:
            st.write(f"Probe error: {e}")

st.caption("Auto mode prefers Hourly; if H1 bars are insufficient, falls back to Daily. Robust bars fetching with retries, chunk-down, and per-symbol fallback.")
