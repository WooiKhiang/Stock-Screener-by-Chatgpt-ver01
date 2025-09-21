# streamlit_app.py — Dual-timeframe (D1+H1) candidates → Google Sheets
# - Universe cap slider (default 4000)
# - D1 structural gates + H1 activity/trigger gates
# - Flags: D1_EMA200_breakout, D1_MACD_zero_cross_sig_neg, H1_MACD_zero_cross_sig_neg
# - Context tab: vertical (Metric, Value), human-readable, hard-refresh each run
# - Adds sector + industry ETFs + Gold/Oil refs; Top-10 RVOL(H1) & DollarVol(D1)
# - Transaction-safe write for Universe tab

import os, json, datetime as dt
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
FEED          = "iex"

SA_RAW = st.secrets.get("GCP_SERVICE_ACCOUNT") or os.getenv("GCP_SERVICE_ACCOUNT")

# =========================
# UI
# =========================
st.set_page_config(page_title="Dual Scan — Alpaca → Sheets", layout="wide")
st.title("📡 Dual-Timeframe Candidates (Alpaca → Google Sheets)")

ET = pytz.timezone("America/New_York")
HEADERS = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}

# Auto-refresh hourly
autorun = st.sidebar.checkbox("Auto-run every hour", value=True)
try:
    autorefresh = getattr(st, "autorefresh", None) or getattr(st, "experimental_autorefresh", None)
    if autorun and autorefresh:
        autorefresh(interval=60*60*1000, key="hourly_autorefresh")
    elif autorun:
        st.markdown("<script>setTimeout(()=>window.location.reload(),3600000);</script>", unsafe_allow_html=True)
except Exception:
    pass

st.sidebar.markdown("### Scan settings")
SCAN_MODE = st.sidebar.selectbox("Scan timeframe", ["Dual (Daily+Hourly)", "Daily only", "Hourly only"], index=0)

UNIVERSE_CAP = st.sidebar.slider("Universe cap (symbols)", min_value=500, max_value=6000, value=4000, step=250,
                                 help="US equities (NASDAQ/NYSE/AMEX), active & tradable. Broader = slower.")

MANUAL_RISK  = st.sidebar.selectbox("Risk mode", ["auto", "normal", "tight"], index=0)

with st.sidebar.expander("Step 1 — Daily Hygiene", expanded=True):
    PRICE_MIN = st.number_input("Min price ($)", value=5.0, step=0.5)
    PRICE_MAX = st.number_input("Max price ($)", value=300.0, step=1.0)
    REQUIRE_SMA_BIAS = st.checkbox("Require (SMA20 ≥ SMA50) AND (Close ≥ SMA50)", value=True)
    st.caption("Liquidity (pass if either):")
    LIQ_MIN_AVG_VOL20 = st.number_input("AvgVol20 ≥", value=1_000_000, step=100_000, format="%i")
    LIQ_MIN_AVG_DV20  = st.number_input("Avg$Vol20 ≥", value=20_000_000, step=1_000_000, format="%i")

with st.sidebar.expander("Step 2 — Activity", expanded=True):
    if SCAN_MODE == "Daily only":
        ATR_PCT_MIN = st.number_input("ATR% (D1) min", value=0.010, step=0.001, format="%.3f")
        ATR_PCT_MAX = st.number_input("ATR% (D1) max", value=0.080, step=0.001, format="%.3f")
        RVOL_MIN    = st.number_input("RVOL_today (D1) ≥", value=0.90, step=0.05, format="%.2f")
    else:
        ATR_PCT_MIN = st.number_input("ATR% (H1) min", value=0.0015, step=0.0005, format="%.4f")
        ATR_PCT_MAX = st.number_input("ATR% (H1) max", value=0.0200, step=0.0005, format="%.4f")
        RVOL_MIN    = st.number_input("RVOL_hour (H1) ≥", value=1.00, step=0.05, format="%.2f")
    REQ_SMA50_GT_200 = st.checkbox("Require SMA50 > SMA200 (daily, tight add-on)", value=False)

with st.sidebar.expander("Step 3 — Technical gates", expanded=True):
    GATE_EMA  = st.checkbox("EMA stack rising", value=True,
                            help="EMA5 > EMA20 > EMA50, and each > prior bar (uses H1 in Dual/Hourly, D1 in Daily only).")
    GATE_MACD = st.checkbox("MACD turn", value=True,
                            help="Histogram < 0 & rising; MACD line > signal; MACD line rising.")
    GATE_KDJ  = st.checkbox("KDJ align", value=True,
                            help="K > D, K rising, D rising, J rising (no J<80 cap).")

with st.sidebar.expander("Other", expanded=False):
    SHOW_LIMIT     = st.slider("Rows to show (UI)", 10, 50, 20)
    DAILY_LOOKBACK = st.slider("Daily lookback (days)", 60, 140, 100)
    HOURLY_LOOK_D  = st.slider("Hourly lookback (days)", 5, 30, 10,
                               help="History window for H1 ATR/RVOL/indicators.")
    TAG_SURV_LIMIT = st.slider("Tagging cap (survivors)", 20, 200, 80, 10)

# =========================
# Helpers
# =========================
def now_et_str():
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
    # Keep hourly bars whose ET time is within RTH (09:30–16:00)
    if df is None or df.empty: return df
    et_times = df["t"].dt.tz_convert(ET)
    mask = (et_times.dt.time >= dt.time(9,30)) & (et_times.dt.time <= dt.time(16,0))
    return df[mask].reset_index(drop=True)

# =========================
# Reference context (yfinance)
# =========================
try:
    import yfinance as yf
    YF_AVAILABLE = True
except Exception:
    YF_AVAILABLE = False

SECTORS = ["XLF","XLK","XLY","XLP","XLV","XLE","XLI","XLU","XLRE","XLB","IYZ"]
INDUSTRY_ETFS = ["SMH","SOXX","XBI","KRE","ITB","XME","IYT","XOP","OIH","TAN"]
COMMODITY_ETFS = ["GLD","GDX","USO","XOP","OIH"]  # gold, miners, oil, energy exposures
MEGACAPS = ["AAPL","MSFT","NVDA","AMZN","GOOG","GOOGL","META","TSLA"]

def _yf_extract_close(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        panel = df["Adj Close"] if "Adj Close" in lvl0 else (df["Close"] if "Close" in lvl0 else df[lvl0[0]])
    else:
        panel = df
    return panel if isinstance(panel, pd.DataFrame) else panel.to_frame()

def fetch_sentiment_refs():
    if not YF_AVAILABLE:
        return {"vix": np.nan, "sector1d":{}, "sector5d":{}, "industry1d":{}, "industry5d":{},
                "mega1d":{}, "mega5d":{}, "commodity1d":{}, "commodity5d":{}, "risk_mode_auto":"normal"}
    tickers = ["SPY","^VIX"] + sorted(set(SECTORS + INDUSTRY_ETFS + COMMODITY_ETFS + MEGACAPS))
    data = yf.download(tickers=" ".join(tickers), period="10d", interval="1d",
                       auto_adjust=False, progress=False)
    if data is None or len(data)==0:
        return {"vix": np.nan, "sector1d":{}, "sector5d":{}, "industry1d":{}, "industry5d":{},
                "mega1d":{}, "mega5d":{}, "commodity1d":{}, "commodity5d":{}, "risk_mode_auto":"normal"}
    panel = _yf_extract_close(data).ffill()
    last = panel.iloc[-1]; prev = panel.iloc[-2] if len(panel)>=2 else last; prev5 = panel.iloc[-5] if len(panel)>=5 else prev
    ret1d = (last/prev - 1.0); ret5d = (last/prev5 - 1.0)
    vix = float(last.get("^VIX", np.nan))

    def submap(keys): 
        return {k: float(ret1d.get(k, np.nan)) for k in keys}, {k: float(ret5d.get(k, np.nan)) for k in keys}

    sector1d, sector5d = submap(SECTORS)
    industry1d, industry5d = submap(INDUSTRY_ETFS)
    commodity1d, commodity5d = submap(COMMODITY_ETFS)
    mega1d = {m: float(ret1d.get(m, np.nan)) for m in MEGACAPS}
    mega5d = {m: float(ret5d.get(m, np.nan)) for m in MEGACAPS}

    defensives = ["XLV","XLP","XLU"]; cyclicals = ["XLK","XLY","XLF"]
    d1 = np.nanmean([sector1d.get(s, np.nan) for s in defensives])
    c1 = np.nanmean([sector1d.get(s, np.nan) for s in cyclicals])
    d5 = np.nanmean([sector5d.get(s, np.nan) for s in defensives])
    c5 = np.nanmean([sector5d.get(s, np.nan) for s in cyclicals])
    tight = (vix > 20.0) and (d1 > c1) and (d5 > c5)

    return {"vix": vix, "sector1d": sector1d, "sector5d": sector5d,
            "industry1d": industry1d, "industry5d": industry5d,
            "commodity1d": commodity1d, "commodity5d": commodity5d,
            "mega1d": mega1d, "mega5d": mega5d,
            "risk_mode_auto": "tight" if tight else "normal"}

# =========================
# Universe & Bars
# =========================
def fetch_active_symbols(cap):
    url = f"{ALPACA_BASE}/assets"
    params = {"status": "active", "asset_class": "us_equity"}
    r = requests.get(url, headers=HEADERS, params=params, timeout=60); r.raise_for_status()
    keep_exch = {"NASDAQ","NYSE","AMEX"}
    syms = [x["symbol"] for x in r.json() if x.get("exchange") in keep_exch and x.get("tradable")]
    bad_suffixes = (".U",".W","WS","W","R",".P","-P")
    syms = [s for s in syms if not s.endswith(bad_suffixes)]
    return syms[:cap]

def fetch_bars_multi(symbols, timeframe, start_iso, end_iso, limit=1000):
    """Generic bar fetcher returning dict[symbol] -> DataFrame(t, open, high, low, close, volume)"""
    base = f"{ALPACA_DATA}/v2/stocks/bars"
    out = {}
    for i in range(0, len(symbols), 180):
        chunk = symbols[i:i+180]
        params = dict(timeframe=timeframe, symbols=",".join(chunk), start=start_iso, end=end_iso,
                      limit=limit, adjustment="raw", feed=FEED)
        page = None
        while True:
            if page: params["page_token"] = page
            r = requests.get(base, headers=HEADERS, params=params, timeout=60)
            if r.status_code >= 400: raise requests.HTTPError(f"/bars {r.status_code}: {r.text[:300]}")
            js = r.json() or {}
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
            page = js.get("next_page_token")
            if not page: break
    # sort & RTH filter for hourly
    for s, df in list(out.items()):
        if df is None or df.empty: 
            continue
        df = df.drop_duplicates(subset=["t"]).sort_values("t").reset_index(drop=True)
        if timeframe.lower() in ("1hour","1h","60min","60m"):
            df = rth_only_hour(df)
        out[s] = df
    return out

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
# Pipeline
# =========================
def run_pipeline():
    # 0) Sentiment / references
    refs = fetch_sentiment_refs()
    risk_mode = refs["risk_mode_auto"] if MANUAL_RISK=="auto" else MANUAL_RISK

    # 1) Universe
    symbols = fetch_active_symbols(UNIVERSE_CAP)
    total_universe = len(symbols)

    # 2) Bars
    now_utc = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    daily_start = (now_utc - dt.timedelta(days=DAILY_LOOKBACK)).isoformat().replace("+00:00","Z")
    hourly_start= (now_utc - dt.timedelta(days=HOURLY_LOOK_D)).isoformat().replace("+00:00","Z")
    end_iso     = now_utc.isoformat().replace("+00:00","Z")

    need_daily = (SCAN_MODE in ["Dual (Daily+Hourly)","Daily only"])
    need_hour  = (SCAN_MODE in ["Dual (Daily+Hourly)","Hourly only"])

    bars_d = fetch_bars_multi(symbols, "1Day", daily_start, end_iso) if need_daily else {}
    bars_h = fetch_bars_multi(symbols, "1Hour", hourly_start, end_iso) if need_hour  else {}

    # "Big players" — flow anomalies
    big_rvol_h1 = []
    if need_hour:
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
    if need_daily:
        rows = []
        for s, df in bars_d.items():
            if df is None or len(df) < 2: continue
            close = df["close"].astype(float); vol = df["volume"].astype(float).fillna(0)
            rows.append((s, float(close.iloc[-1]*vol.iloc[-1])))
        if rows:
            tmp = pd.DataFrame(rows, columns=["symbol","dollar_vol_d1"]).sort_values("dollar_vol_d1", ascending=False)
            big_dv_d1 = tmp.head(10)["symbol"].tolist()

    # 3) Screen
    step1 = step2 = step3 = 0
    survivors = []

    for s in symbols:
        # Assemble daily/hourly frames as needed
        dfd = bars_d.get(s)
        dfh = bars_h.get(s)

        # Must have enough data for whichever mode we're using
        if need_daily and (dfd is None or len(dfd) < 60): 
            continue
        if need_hour  and (dfh is None or len(dfh) < 60): 
            continue

        # ---- Step 1 (Daily hygiene) ----
        if need_daily:
            close_d = dfd["close"].astype(float); vol_d = dfd["volume"].astype(float).fillna(0)
            last_d  = close_d.iloc[-1]
            if not (PRICE_MIN <= last_d <= PRICE_MAX): 
                continue
            sma20_d = sma(close_d, 20); sma50_d = sma(close_d, 50)
            if REQUIRE_SMA_BIAS and ((sma20_d.iloc[-1] < sma50_d.iloc[-1]) or (last_d < sma50_d.iloc[-1])):
                continue
            avg_vol20_d  = vol_d.rolling(20, min_periods=20).mean().iloc[-1]
            avg_dv20_d   = (close_d*vol_d).rolling(20, min_periods=20).mean().iloc[-1]
            if not ((avg_vol20_d >= LIQ_MIN_AVG_VOL20) or (avg_dv20_d >= LIQ_MIN_AVG_DV20)):
                continue
        else:
            # Hourly-only price filter uses last hourly close
            close_h = dfh["close"].astype(float)
            last_h  = close_h.iloc[-1]
            if not (PRICE_MIN <= last_h <= PRICE_MAX): 
                continue

        step1 += 1

        # ---- Step 2 (Activity) ----
        if SCAN_MODE == "Daily only":
            # D1 ATR% & RVOL_today
            atr14 = atr(dfd, 14).iloc[-1]
            atr_pct = float(atr14 / close_d.iloc[-1]) if close_d.iloc[-1] > 0 else np.nan
            rvol_today = float(vol_d.iloc[-1] / vol_d.rolling(20, min_periods=20).mean().iloc[-1]) if vol_d.rolling(20,20).mean().iloc[-1] else np.nan
            sma50_d = sma(close_d,50).iloc[-1]; sma200_d = sma(close_d,200).iloc[-1] if len(close_d)>=200 else np.nan
            if not (ATR_PCT_MIN <= atr_pct <= ATR_PCT_MAX and rvol_today >= RVOL_MIN): 
                continue
            if REQ_SMA50_GT_200 and not (sma50_d > sma200_d): 
                continue
        else:
            # H1 ATR% & RVOL_hour
            close_h = dfh["close"].astype(float); vol_h = dfh["volume"].astype(float).fillna(0)
            atr14_h = atr(dfh, 14).iloc[-1]
            atr_pct = float(atr14_h / close_h.iloc[-1]) if close_h.iloc[-1] > 0 else np.nan
            base = vol_h.rolling(20, min_periods=20).mean().iloc[-1]
            rvol_hour = float(vol_h.iloc[-1]/base) if base and base>0 else np.nan
            if not (ATR_PCT_MIN <= atr_pct <= ATR_PCT_MAX and rvol_hour >= RVOL_MIN): 
                continue
            # Optional D1 add-on even in Dual/Hourly
            if need_daily and REQ_SMA50_GT_200:
                sma50_d = sma(dfd["close"].astype(float),50).iloc[-1]
                sma200_d = sma(dfd["close"].astype(float),200).iloc[-1] if len(dfd)>=200 else np.nan
                if not (sma50_d > sma200_d): 
                    continue

        step2 += 1

        # ---- Step 3 (Technical) ----
        # Pick the frame for indicators (H1 in Dual/Hourly, D1 in Daily-only)
        dfI = dfh if SCAN_MODE != "Daily only" else dfd
        clI = dfI["close"].astype(float)
        ok = True
        if GATE_EMA:
            e5, e20, e50 = ema(clI,5), ema(clI,20), ema(clI,50)
            ok &= (e5.iloc[-1] > e20.iloc[-1] > e50.iloc[-1] and
                   e5.iloc[-1] > e5.iloc[-2] and e20.iloc[-1] > e20.iloc[-2] and e50.iloc[-1] > e50.iloc[-2])
        if GATE_MACD:
            macd_line, macd_sig, macd_hist = macd(clI)
            ok &= ((macd_hist.iloc[-1] < 0) and (macd_hist.iloc[-1] > macd_hist.iloc[-2]) and
                   (macd_line.iloc[-1] > macd_sig.iloc[-1]) and (macd_line.iloc[-1] > macd_line.iloc[-2]))
        else:
            macd_line, macd_sig, macd_hist = macd(clI)
        if GATE_KDJ:
            k, d_, j = kdj(dfI)
            ok &= (k.iloc[-1] > d_.iloc[-1] and k.iloc[-1] > k.iloc[-2] and d_.iloc[-1] > d_.iloc[-2] and j.iloc[-1] > j.iloc[-2])
        else:
            k, d_, j = kdj(dfI)
        if not ok: 
            continue

        step3 += 1

        # Features for ranking (use the indicator frame for ROC; daily for liquidity cols)
        roc_window = 20
        roc = float(clI.iloc[-1] / clI.iloc[-roc_window] - 1.0) if len(clI) > roc_window else np.nan

        if need_daily:
            avg_vol20   = float(dfd["volume"].rolling(20, min_periods=20).mean().iloc[-1])
            avg_dv20    = float((dfd["close"]*dfd["volume"]).rolling(20, min_periods=20).mean().iloc[-1])
            close_last  = float(dfd["close"].iloc[-1])
            atr14_d     = float(atr(dfd,14).iloc[-1])
            atr_pct_d   = float(atr14_d/close_last) if close_last>0 else np.nan
            ema5_d, ema20_d, ema50_d = ema(dfd["close"].astype(float),5).iloc[-1], ema(dfd["close"].astype(float),20).iloc[-1], ema(dfd["close"].astype(float),50).iloc[-1]
            macd_line_d, macd_sig_d, macd_hist_d = macd(dfd["close"].astype(float))
        else:
            avg_vol20 = avg_dv20 = close_last = atr14_d = atr_pct_d = np.nan
            ema5_d = ema20_d = ema50_d = np.nan
            macd_line_d = pd.Series([],dtype=float); macd_sig_d = pd.Series([],dtype=float); macd_hist_d = pd.Series([],dtype=float)

        # Attention flags
        flag_d1_ema200 = False
        flag_d1_macd0  = False
        flag_h1_macd0  = False
        notes = []

        if need_daily:
            e200_d = ema(dfd["close"].astype(float),200)
            if len(e200_d) >= 201:
                c_now, c_prev = dfd["close"].iloc[-1], dfd["close"].iloc[-2]
                e_now, e_prev = e200_d.iloc[-1], e200_d.iloc[-2]
                if (c_now > e_now) and (c_prev <= e_prev):
                    flag_d1_ema200 = True; notes.append("D1_EMA200_breakout")
            if len(macd_line_d) >= 2:
                if (macd_line_d.iloc[-2] <= 0 < macd_line_d.iloc[-1]) and (macd_sig_d.iloc[-1] < 0):
                    flag_d1_macd0 = True; notes.append("D1_MACD_zero_cross_sig_neg")

        if need_hour:
            macd_line_h, macd_sig_h, _ = macd(dfh["close"].astype(float))
            if len(macd_line_h) >= 2:
                if (macd_line_h.iloc[-2] <= 0 < macd_line_h.iloc[-1]) and (macd_sig_h.iloc[-1] < 0):
                    flag_h1_macd0 = True; notes.append("H1_MACD_zero_cross_sig_neg")

        # Collect row
        survivors.append({
            "symbol": s,
            "close": float(close_last) if need_daily else float(dfh["close"].iloc[-1]),
            "avg_vol20": float(avg_vol20),
            "avg_dollar_vol20": float(avg_dv20),
            "atr14": float(atr14_d) if need_daily else float(atr(dfh,14).iloc[-1]),
            "atr_pct": float(atr_pct_d) if need_daily else float(atr(dfh,14).iloc[-1]/dfh["close"].iloc[-1]),
            "ema5": float(ema5_d) if need_daily else float(ema(dfh["close"].astype(float),5).iloc[-1]),
            "ema20": float(ema20_d) if need_daily else float(ema(dfh["close"].astype(float),20).iloc[-1]),
            "ema50": float(ema50_d) if need_daily else float(ema(dfh["close"].astype(float),50).iloc[-1]),
            "macd_line": float(macd_line.iloc[-1]) if 'macd_line' in locals() else (float(macd(dfh["close"].astype(float))[0].iloc[-1]) if need_hour else np.nan),
            "macd_signal": float(macd_sig.iloc[-1]) if 'macd_sig'  in locals() else (float(macd(dfh["close"].astype(float))[1].iloc[-1]) if need_hour else np.nan),
            "macd_hist": float(macd_hist.iloc[-1]) if 'macd_hist'   in locals() else (float(macd(dfh["close"].astype(float))[2].iloc[-1]) if need_hour else np.nan),
            "kdj_k": float(k.iloc[-1]),
            "kdj_d": float(d_.iloc[-1]),
            "kdj_j": float(j.iloc[-1]),
            "roc20": float(roc),
            "flag_d1_ema200_breakout": flag_d1_ema200,
            "flag_d1_macd_zero_cross_sig_neg": flag_d1_macd0,
            "flag_h1_macd_zero_cross_sig_neg": flag_h1_macd0,
            "notes": ", ".join(notes) if notes else ""
        })

    if not survivors:
        return refs, risk_mode, total_universe, step1, step2, step3, pd.DataFrame(), pd.DataFrame(), big_rvol_h1, big_dv_d1

    df = pd.DataFrame(survivors)

    # Rank (same formula; ROC from indicator frame; inverse ATR%)
    df["inv_atr_pct"] = df["atr_pct"].replace(0, np.nan).rpow(-1)
    df["rank_score"]  = 0.4*zscore(df["roc20"]) + 0.3*zscore(df["inv_atr_pct"])
    # Add a bit of flow: if hourly, mix RVOL_hour if available
    # (Optional; keeping simple for speed)

    df = df.sort_values("rank_score", ascending=False).reset_index(drop=True)

    df_ui  = df.head(SHOW_LIMIT).copy()
    df_out = df.head(20).copy()
    as_of = now_et_str()
    df_out.insert(0,"risk_mode", risk_mode)
    df_out.insert(0,"as_of_date", as_of)

    cols = ["as_of_date","risk_mode","symbol","close","avg_vol20","avg_dollar_vol20",
            "atr14","atr_pct","ema5","ema20","ema50","macd_line","macd_signal","macd_hist",
            "kdj_k","kdj_d","kdj_j","roc20","rank_score",
            "flag_d1_ema200_breakout","flag_d1_macd_zero_cross_sig_neg","flag_h1_macd_zero_cross_sig_neg","notes"]
    for c in cols:
        if c not in df_out.columns: df_out[c] = np.nan
    df_out = df_out[cols]

    return refs, risk_mode, total_universe, step1, step2, step3, df_ui, df_out, big_rvol_h1, big_dv_d1

# =========================
# Google Sheets I/O
# =========================
def _open_or_create_ws(gc, title, rows=200, cols=60):
    sh = gc.open_by_key(GOOGLE_SHEET_ID)
    try: return sh.worksheet(title)
    except Exception: return sh.add_worksheet(title=title, rows=rows, cols=cols)

def write_sheet_safe(df: pd.DataFrame, tab_name: str):
    """Transaction-safe overwrite: write new content first, then trim; old data preserved on failure."""
    if df is None or df.empty:
        raise RuntimeError("No rows to write; preserving previous sheet.")
    import gspread
    from google.oauth2.service_account import Credentials
    sa_info = parse_sa(SA_RAW)
    creds = Credentials.from_service_account_info(sa_info,
               scopes=["https://www.googleapis.com/auth/spreadsheets",
                       "https://www.googleapis.com/auth/drive"])
    gc = gspread.authorize(creds)
    ws = _open_or_create_ws(gc, tab_name)

    try:
        old_vals = ws.get_all_values()
        old_rows = len(old_vals)
        old_cols = max((len(r) for r in old_vals), default=0)
    except Exception:
        old_rows = old_cols = 0

    new_values = [list(df.columns)] + df.fillna("").astype(str).values.tolist()
    ws.update("A1", new_values, value_input_option="RAW")

    # Trim leftovers (not critical if fails)
    try:
        from string import ascii_uppercase as AU
        def col_letter(n):
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

def write_context_hard_replace(rows_metric_value: list):
    """Hard-refresh the Context tab: create temp, write, delete old, rename temp→Context (safe swap)."""
    import gspread
    from google.oauth2.service_account import Credentials
    sa_info = parse_sa(SA_RAW)
    creds = Credentials.from_service_account_info(sa_info,
               scopes=["https://www.googleapis.com/auth/spreadsheets",
                       "https://www.googleapis.com/auth/drive"])
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(GOOGLE_SHEET_ID)

    temp_name = f"{TAB_CONTEXT}_TMP"
    # Create temp & write
    try:
        try:
            ws_tmp = sh.worksheet(temp_name)
        except Exception:
            ws_tmp = sh.add_worksheet(title=temp_name, rows=500, cols=2)
        df_ctx = pd.DataFrame(rows_metric_value, columns=["Metric","Value"])
        ws_tmp.clear()
        ws_tmp.update("A1", [list(df_ctx.columns)] + df_ctx.fillna("").astype(str).values.tolist(), value_input_option="RAW")
        # Delete old Context (if exists), then rename temp
        try:
            ws_old = sh.worksheet(TAB_CONTEXT)
            sh.del_worksheet(ws_old)
        except Exception:
            pass
        ws_tmp.update_title(TAB_CONTEXT)
    except Exception as e:
        # Fallback: write to Context directly (clear then update)
        try:
            ws = _open_or_create_ws(gc, TAB_CONTEXT)
            ws.clear()
            ws.update("A1", [["Metric","Value"]] + rows_metric_value, value_input_option="RAW")
        except Exception as e2:
            raise RuntimeError(f"Failed to write Context: {e} / fallback: {e2}")

# =========================
# RUN
# =========================
sentiment_col, summary_col = st.columns([2,1])

with st.spinner("Scanning (Dual timeframe)…"):
    refs, risk_mode, total_universe, s1, s2, s3, df_ui, df_out, big_rvol_h1, big_dv_d1 = run_pipeline()

# Sentiment / references panel
with sentiment_col:
    st.subheader("Market & Breadth (context)")
    vix_val = refs.get("vix", np.nan)
    st.write(
        f"**VIX:** {vix_val:.2f}" if isinstance(vix_val,(int,float)) and np.isfinite(vix_val) else "**VIX:** n/a",
        f" | **Risk (auto):** `{refs.get('risk_mode_auto','normal')}`",
        f" | **Using:** `{risk_mode}`",
        f" | **Mode:** `{SCAN_MODE}`"
    )
    # Show sectors quickly
    if refs.get("sector1d"):
        sect_df = pd.DataFrame({"Sector": list(refs["sector1d"].keys()),
                                "1D": list(refs["sector1d"].values()),
                                "5D": [refs["sector5d"].get(k, np.nan) for k in refs["sector1d"].keys()]})
        st.dataframe(sect_df.sort_values("1D", ascending=False).style.format({"1D":"{:.2%}","5D":"{:.2%}"}),
                     use_container_width=True)

with summary_col:
    st.subheader("Run Summary")
    st.write(f"**As of (ET):** {now_et_str()}")
    st.write(f"**Universe cap:** {UNIVERSE_CAP:,}")
    st.write(f"**Universe scanned:** {total_universe:,}")
    st.write(f"**Step1/2/3 survivors:** {s1:,} / {s2:,} / {s3:,}")
    st.write(f"**Tabs:** `{TAB_UNIVERSE}`, `{TAB_CONTEXT}`")

# UI table
st.subheader("Top Candidates (preview)")
if df_ui.empty:
    st.warning("No survivors under current gates.")
else:
    # Coerce numerics for clean formatting
    num_cols = ["close","avg_vol20","avg_dollar_vol20","atr14","atr_pct","ema5","ema20","ema50",
                "macd_line","macd_signal","macd_hist","kdj_k","kdj_d","kdj_j","roc20","rank_score"]
    for c in num_cols:
        if c in df_ui.columns: df_ui[c] = pd.to_numeric(df_ui[c], errors="coerce")
    fmt = {c:"{:.2f}" for c in num_cols if c in df_ui.columns}
    st.dataframe(df_ui.head(SHOW_LIMIT).style.format(fmt), use_container_width=True)

# Build Context (vertical, readable) and write (hard refresh)
ctx_rows = []
ctx_rows.append(["Timestamp (ET)", now_et_str()])
ctx_rows.append(["Risk mode (auto)", refs.get("risk_mode_auto","")])
ctx_rows.append(["Risk mode (using)", risk_mode])
ctx_rows.append(["Scan mode", SCAN_MODE])
ctx_rows.append(["Universe cap", f"{UNIVERSE_CAP}"])
ctx_rows.append(["Universe scanned", f"{total_universe}"])
ctx_rows.append(["Step 1 survivors", f"{s1}"])
ctx_rows.append(["Step 2 survivors", f"{s2}"])
ctx_rows.append(["Step 3 survivors", f"{s3}"])
ctx_rows.append(["VIX level", f"{refs.get('vix', np.nan):.2f}" if isinstance(refs.get('vix'),(int,float)) and np.isfinite(refs.get('vix')) else "n/a"])

# Sector & industry & commodity (1D/5D)
def add_bucket_rows(title, d1:dict, d5:dict):
    ctx_rows.append([f"{title}", ""])
    keys = sorted(d1.keys())
    for k in keys:
        v1 = d1.get(k, np.nan); v5 = d5.get(k, np.nan)
        v1s = f"{v1:.2%}" if isinstance(v1,(int,float)) and np.isfinite(v1) else "n/a"
        v5s = f"{v5:.2%}" if isinstance(v5,(int,float)) and np.isfinite(v5) else "n/a"
        ctx_rows.append([f"  {k}", f"1D {v1s} | 5D {v5s}"])

add_bucket_rows("Sectors", refs.get("sector1d",{}), refs.get("sector5d",{}))
add_bucket_rows("Industries", refs.get("industry1d",{}), refs.get("industry5d",{}))
add_bucket_rows("Gold/Oil refs", refs.get("commodity1d",{}), refs.get("commodity5d",{}))

# Big players lists
if big_rvol_h1:
    ctx_rows.append(["Top-10 RVOL (Hourly)", ", ".join(big_rvol_h1)])
if big_dv_d1:
    ctx_rows.append(["Top-10 Dollar Volume (Daily)", ", ".join(big_dv_d1)])

# Always hard-refresh Context (delete/replace safely)
try:
    write_context_hard_replace(ctx_rows)
    st.success("Context tab refreshed.")
except Exception as e:
    st.error(f"Failed to write Context: {e}")

# Write Universe only if we have rows
try:
    if df_out.empty:
        st.warning("No rows to write — Universe left unchanged.")
    else:
        write_sheet_safe(df_out, TAB_UNIVERSE)
        st.success("Universe tab updated.")
except Exception as e:
    st.error(f"Failed to write Universe (previous data preserved): {e}")

st.caption("Dual-timeframe scan: D1 for structure, H1 for activity/triggers. Context is vertical and refreshed each run.")
