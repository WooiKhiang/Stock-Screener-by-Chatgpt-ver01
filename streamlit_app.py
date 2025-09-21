# streamlit_app.py — Daily candidates (Alpaca) → Sheets (safe overwrite) + Context tab; hourly auto-run
import os, json, datetime as dt
import numpy as np
import pandas as pd
import pytz
import requests
import streamlit as st

# =========================
# Hard-coded configuration
# =========================
GOOGLE_SHEET_ID   = "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4"
TAB_UNIVERSE      = "Universe"   # Top-20 candidates table (overwritten safely)
TAB_CONTEXT       = "Context"    # Sentiment, breadth, big players, funnel counts (overwritten safely)

# Alpaca credentials (you can move to Streamlit secrets)
ALPACA_KEY    = os.getenv("ALPACA_KEY",    "PKIG445MPT704CN8P0R8")
ALPACA_SECRET = os.getenv("ALPACA_SECRET", "3GQdWcnbMo6V8uvhVa6BK6EbvnH4EHinlsU6uvj4")
ALPACA_BASE   = "https://paper-api.alpaca.markets/v2"   # assets, account, calendar
ALPACA_DATA   = "https://data.alpaca.markets"           # market data v2
FEED          = "iex"                                   # required for paper/free tier

# Google service account JSON from Streamlit secrets or env
SA_RAW = st.secrets.get("GCP_SERVICE_ACCOUNT") or os.getenv("GCP_SERVICE_ACCOUNT")

# General knobs
MAX_SYMBOLS_SCAN = 1000
CHUNK_SIZE       = 180

# =========================
# UI: Sidebar
# =========================
st.set_page_config(page_title="Daily Candidates — Alpaca → Google Sheets", layout="wide")
st.sidebar.header("Run Control")

autorun = st.sidebar.checkbox("Auto-run every hour", value=True,
                              help="Refreshes this app every 60 minutes and pushes the latest candidates.")
try:
    st_autorefresh = getattr(st, "autorefresh", None) or getattr(st, "experimental_autorefresh", None)
    if autorun and st_autorefresh:
        st_autorefresh(interval=60*60*1000, key="hourly_autorefresh")  # 60 minutes
    elif autorun:
        st.markdown("<script>setTimeout(()=>window.location.reload(), 3600000);</script>", unsafe_allow_html=True)
except Exception:
    pass

st.sidebar.markdown("### Risk mode")
MANUAL_RISK  = st.sidebar.selectbox("Override", ["auto", "normal", "tight"], index=0,
                                    help="Leave 'auto' to infer from VIX+sectors; choose normal/tight to override.")

with st.sidebar.expander("Step 1 — Hygiene (Daily RTH)", expanded=True):
    PRICE_MIN = st.number_input("Min price", value=5.0, step=0.5)
    PRICE_MAX = st.number_input("Max price", value=300.0, step=1.0)
    REQUIRE_SMA_BIAS = st.checkbox("Require SMA20 ≥ SMA50 AND Close ≥ SMA50", value=True)
    st.caption("Liquidity: pass if (AvgVol20 ≥ X) **OR** (Avg$Vol20 ≥ Y)")
    LIQ_MIN_AVG_VOL20   = st.number_input("AvgVol20 ≥", value=1_000_000, step=100_000, format="%i")
    LIQ_MIN_AVG_DV20    = st.number_input("Avg$Vol20 ≥", value=20_000_000, step=1_000_000, format="%i")

with st.sidebar.expander("Step 2 — Activity", expanded=True):
    ATR_PCT_MIN = st.number_input("ATR% min", value=0.010, step=0.001, format="%.3f")
    ATR_PCT_MAX = st.number_input("ATR% max", value=0.080, step=0.001, format="%.3f")
    RVOL_MIN    = st.number_input("RVOL_today ≥", value=0.90, step=0.05, format="%.2f")
    REQ_SMA50_GT_200 = st.checkbox("Require SMA50 > SMA200 (tight add-on)", value=False)

with st.sidebar.expander("Step 3 — Technical gates", expanded=True):
    GATE_EMA  = st.checkbox("EMA5>EMA20>EMA50 and all rising", value=True)
    GATE_MACD = st.checkbox("MACD turn (hist < 0 & rising; DIF>DEA; DIF rising)", value=True)
    GATE_KDJ  = st.checkbox("KDJ alignment (K>D; K↑; D↑; J↑; J<80)", value=True)

with st.sidebar.expander("Other", expanded=False):
    SHOW_LIMIT    = st.slider("Rows to show (UI)", 10, 50, 20)
    STALE_DAYS_OK = st.slider("Daily bars lookback (days)", 60, 140, 100)
    TAG_SURV_LIMIT = st.slider("Tagging survivors cap", 20, 200, 80, 10,
                               help="Upper bound of survivors to compute hourly/PM/halts/tags on.")

st.title("🧭 Daily Candidates — Clean, fast, PDT-safe (Alpaca → Sheets)")

# =========================
# Basics
# =========================
ET = pytz.timezone("America/New_York")
HEADERS = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}

def now_et_str():
    return dt.datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")

def parse_sa(raw_json: str) -> dict:
    if not raw_json:
        raise RuntimeError("Missing GCP_SERVICE_ACCOUNT in secrets/env.")
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        if "-----BEGIN PRIVATE KEY-----" in raw_json and "\\n" not in raw_json:
            start = raw_json.find("-----BEGIN PRIVATE KEY-----"); end = raw_json.find("-----END PRIVATE KEY-----", start) + len("-----END PRIVATE KEY-----")
            raw_json = raw_json.replace(raw_json[start:end], raw_json[start:end].replace("\r\n","\n").replace("\n","\\n"))
        return json.loads(raw_json)

def zscore(x: pd.Series) -> pd.Series:
    m = x.mean(); s = x.std(ddof=0)
    if s == 0 or not np.isfinite(s): return pd.Series(0.0, index=x.index)
    return (x - m) / s

# =========================
# Sentiment (yfinance) — context only
# =========================
try:
    import yfinance as yf
    YF_AVAILABLE = True
except Exception:
    YF_AVAILABLE = False

SECTORS = ["XLF","XLK","XLY","XLP","XLV","XLE","XLI","XLU","XLRE","XLB","IYZ"]
INDUSTRY_ETFS = ["SMH","SOXX","XBI","KRE","ITB","XME","IYT","XOP","OIH","TAN"]
MEGACAPS = ["AAPL","MSFT","NVDA","AMZN","GOOG","GOOGL","META","TSLA"]

def _yf_extract_close(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        panel = df["Adj Close"] if "Adj Close" in lvl0 else (df["Close"] if "Close" in lvl0 else df[lvl0[0]])
    else:
        panel = df
    return panel if isinstance(panel, pd.DataFrame) else panel.to_frame()

def fetch_sentiment():
    if not YF_AVAILABLE:
        return {"vix": float("nan"), "ret1d": {}, "ret5d": {}, "risk_mode_auto": "normal",
                "mega1d":{}, "mega5d":{}, "ind1d":{}, "ind5d":{}}
    tickers = ["SPY","^VIX"] + SECTORS + INDUSTRY_ETFS + MEGACAPS
    data = yf.download(tickers=" ".join(tickers), period="10d", interval="1d",
                       auto_adjust=False, progress=False)
    if data is None or len(data)==0:
        return {"vix": float("nan"), "ret1d": {}, "ret5d": {}, "risk_mode_auto": "normal",
                "mega1d":{}, "mega5d":{}, "ind1d":{}, "ind5d":{}}
    panel = _yf_extract_close(data).ffill()
    last = panel.iloc[-1]; prev = panel.iloc[-2] if len(panel)>=2 else last; prev5 = panel.iloc[-5] if len(panel)>=5 else prev
    ret1d = (last/prev - 1.0); ret5d = (last/prev5 - 1.0)
    vix = float(last.get("^VIX", np.nan))
    defensives = ["XLV","XLP","XLU"]; cyclicals = ["XLK","XLY","XLF"]
    def_mean_1 = np.nanmean([ret1d.get(s, np.nan) for s in defensives]); cyc_mean_1 = np.nanmean([ret1d.get(s, np.nan) for s in cyclicals])
    def_mean_5 = np.nanmean([ret5d.get(s, np.nan) for s in defensives]); cyc_mean_5 = np.nanmean([ret5d.get(s, np.nan) for s in cyclicals])
    tight = (vix > 20.0) and (def_mean_1 > cyc_mean_1) and (def_mean_5 > cyc_mean_5)
    return {
        "vix": vix,
        "ret1d": {s:float(ret1d.get(s, np.nan)) for s in SECTORS},
        "ret5d": {s:float(ret5d.get(s, np.nan)) for s in SECTORS},
        "ind1d": {s:float(ret1d.get(s, np.nan)) for s in INDUSTRY_ETFS},
        "ind5d": {s:float(ret5d.get(s, np.nan)) for s in INDUSTRY_ETFS},
        "mega1d": {m:float(ret1d.get(m, np.nan)) for m in MEGACAPS},
        "mega5d": {m:float(ret5d.get(m, np.nan)) for m in MEGACAPS},
        "risk_mode_auto":"tight" if tight else "normal"
    }

# =========================
# Universe (US equities; active, tradable)
# =========================
def fetch_active_symbols(max_symbols=MAX_SYMBOLS_SCAN):
    url = f"{ALPACA_BASE}/assets"
    params = {"status": "active", "asset_class": "us_equity"}
    r = requests.get(url, headers=HEADERS, params=params, timeout=60); r.raise_for_status()
    keep_exch = {"NASDAQ","NYSE","AMEX"}
    syms = [x["symbol"] for x in r.json() if x.get("exchange") in keep_exch and x.get("tradable")]
    bad_suffixes = (".U",".W","WS","W","R",".P","-P")
    return [s for s in syms if not s.endswith(bad_suffixes)][:max_symbols]

# =========================
# Bars helpers (RTH)
# =========================
def ny_open_close_utc(day_utc: dt.datetime):
    try:
        d = day_utc.astimezone(ET).date().isoformat()
        r = requests.get(f"{ALPACA_BASE}/calendar", headers=HEADERS, params={"start": d, "end": d}, timeout=30)
        r.raise_for_status(); cal = r.json()
        if cal:
            o = dt.datetime.fromisoformat(cal[0]["open"]).astimezone(ET)
            c = dt.datetime.fromisoformat(cal[0]["close"]).astimezone(ET)
            return o.astimezone(dt.timezone.utc), c.astimezone(dt.timezone.utc)
    except Exception:
        pass
    et_date = day_utc.astimezone(ET).date()
    o = ET.localize(dt.datetime.combine(et_date, dt.time(9,30))).astimezone(dt.timezone.utc)
    c = ET.localize(dt.datetime.combine(et_date, dt.time(16,0))).astimezone(dt.timezone.utc)
    return o, c

def fetch_daily_bars_multi(symbols, start_iso, end_iso, timeframe="1Day", limit=1000):
    base = f"{ALPACA_DATA}/v2/stocks/bars"; result = {}
    for i in range(0, len(symbols), CHUNK_SIZE):
        chunk = symbols[i:i+CHUNK_SIZE]
        params = dict(timeframe=timeframe, symbols=",".join(chunk), start=start_iso, end=end_iso,
                      limit=limit, adjustment="raw", feed=FEED)
        page = None
        while True:
            if page: params["page_token"] = page
            r = requests.get(base, headers=HEADERS, params=params, timeout=60)
            if r.status_code >= 400: raise requests.HTTPError(f"/bars {r.status_code}: {r.text[:300]}")
            js = r.json(); bars = js.get("bars", [])
            if isinstance(bars, dict):
                for sym, arr in bars.items():
                    add = pd.DataFrame(arr); 
                    if add.empty: continue
                    add["t"] = pd.to_datetime(add["t"], utc=True)
                    add.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"}, inplace=True)
                    add = add[["t","open","high","low","close","volume"]]
                    result[sym] = pd.concat([result.get(sym), add], ignore_index=True)
            else:
                if bars:
                    add = pd.DataFrame(bars); add["t"] = pd.to_datetime(add["t"], utc=True)
                    add.rename(columns={"S":"symbol","o":"open","h":"high","l":"low","c":"close","v":"volume"}, inplace=True)
                    add = add[["symbol","t","open","high","low","close","volume"]]
                    for sym, grp in add.groupby("symbol"):
                        g = grp.drop(columns=["symbol"]).copy()
                        result[sym] = pd.concat([result.get(sym), g], ignore_index=True)
            page = js.get("next_page_token")
            if not page: break
    for s, df in list(result.items()):
        result[s] = df.drop_duplicates(subset=["t"]).sort_values("t").reset_index(drop=True)
    return result

def fetch_hourly_bars(symbols, start_utc, end_utc):
    base = f"{ALPACA_DATA}/v2/stocks/bars"; result = {}
    for i in range(0, len(symbols), CHUNK_SIZE):
        chunk = symbols[i:i+CHUNK_SIZE]
        params = dict(timeframe="1Hour", symbols=",".join(chunk),
                      start=start_utc.isoformat().replace("+00:00","Z"),
                      end=end_utc.isoformat().replace("+00:00","Z"),
                      limit=1000, adjustment="raw", feed=FEED)
        r = requests.get(base, headers=HEADERS, params=params, timeout=60)
        if r.status_code >= 400: continue
        js = r.json(); bars = js.get("bars", [])
        if isinstance(bars, dict):
            for sym, arr in bars.items():
                add = pd.DataFrame(arr); 
                if add.empty: continue
                add["t"] = pd.to_datetime(add["t"], utc=True)
                add.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"}, inplace=True)
                result[sym] = add[["t","open","high","low","close","volume"]]
        else:
            if bars:
                add = pd.DataFrame(bars); add["t"] = pd.to_datetime(add["t"], utc=True)
                add.rename(columns={"S":"symbol","o":"open","h":"high","l":"low","c":"close","v":"volume"}, inplace=True)
                add = add[["symbol","t","open","high","low","close","volume"]]
                for sym, grp in add.groupby("symbol"):
                    result[sym] = grp.drop(columns=["symbol"]).copy()
    o, c = ny_open_close_utc(dt.datetime.now(dt.timezone.utc))
    for s, df in list(result.items()):
        result[s] = df[(df["t"]>=o) & (df["t"]<=c)].reset_index(drop=True)
    return result

# =========================
# Indicators & features
# =========================
def sma(x, n): return x.rolling(n, min_periods=n).mean()
def ema(x, n): return x.ewm(span=n, adjust=False).mean()

def atr(df: pd.DataFrame, n=14) -> pd.Series:
    hi, lo, cl = df["high"], df["low"], df["close"]; pc = cl.shift(1)
    tr = pd.concat([(hi-lo).abs(), (hi-pc).abs(), (lo-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def macd(x, fast=12, slow=26, signal=9):
    ef, es = ema(x, fast), ema(x, slow); line = ef - es; sig = ema(line, signal); hist = line - sig
    return line, sig, hist

def kdj(df, n=9, k_period=3, d_period=3):
    low_min = df["low"].rolling(n, min_periods=n).min(); high_max = df["high"].rolling(n, min_periods=n).max()
    rsv = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    k = rsv.ewm(alpha=1.0/k_period, adjust=False).mean(); d = k.ewm(alpha=1.0/d_period, adjust=False).mean(); j = 3*k - 2*d
    return k, d, j

# =========================
# Trading halts check
# =========================
def check_halts(symbols):
    out = {}; base = f"{ALPACA_DATA}/v2/stocks/snapshots"
    for i in range(0, len(symbols), CHUNK_SIZE):
        chunk = symbols[i:i+CHUNK_SIZE]; params = {"symbols": ",".join(chunk), "feed": FEED}
        r = requests.get(base, headers=HEADERS, params=params, timeout=60)
        if r.status_code >= 400:
            for s in chunk: out[s] = None
            continue
        snap = (r.json() or {}).get("snapshots", {})
        for s, info in snap.items():
            out[s] = info.get("trading_halted", None)
    return out

# =========================
# Pre-market context (not used for indicators)
# =========================
def premkt_context(symbols):
    res = {s: {"gap_pm_pct": None, "pm_vol": None} for s in symbols}
    if not symbols: return res
    now_utc = dt.datetime.now(dt.timezone.utc)
    et_date = now_utc.astimezone(ET).date()
    pm_start = ET.localize(dt.datetime.combine(et_date, dt.time(4,0))).astimezone(dt.timezone.utc)
    rth_open = ET.localize(dt.datetime.combine(et_date, dt.time(9,30))).astimezone(dt.timezone.utc)

    daily = fetch_daily_bars_multi(symbols, (now_utc - dt.timedelta(days=10)).isoformat().replace("+00:00","Z"),
                                   now_utc.isoformat().replace("+00:00","Z"))
    base = f"{ALPACA_DATA}/v2/stocks/bars"; pm_vols = {s:0.0 for s in symbols}
    for i in range(0, len(symbols), CHUNK_SIZE):
        chunk = symbols[i:i+CHUNK_SIZE]
        params = {"timeframe":"1Min","symbols":",".join(chunk),
                  "start":pm_start.isoformat().replace("+00:00","Z"),
                  "end":rth_open.isoformat().replace("+00:00","Z"),
                  "limit":10000, "adjustment":"raw", "feed":FEED}
        r = requests.get(base, headers=HEADERS, params=params, timeout=60)
        if r.status_code >= 400: continue
        bars = (r.json() or {}).get("bars", {})
        if isinstance(bars, dict):
            for sym, arr in bars.items():
                if not arr: continue
                df = pd.DataFrame(arr); 
                if df.empty: continue
                pm_vols[sym] += float(df["v"].sum())

    for s in symbols:
        res[s] = {"gap_pm_pct": None, "pm_vol": pm_vols.get(s, 0.0)}
    return res

# =========================
# Daily pipeline
# =========================
def run_pipeline():
    # 0) Sentiment (context)
    sent = fetch_sentiment()
    risk_mode = sent["risk_mode_auto"] if MANUAL_RISK=="auto" else MANUAL_RISK

    # 1) Universe (US equities, active+tradable on NASDAQ/NYSE/AMEX)
    syms_all = fetch_active_symbols(MAX_SYMBOLS_SCAN)
    total_universe = len(syms_all)
    step1_count = step2_count = step3_count = 0

    # 2) Daily bars (RTH)
    end_utc   = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    start_utc = end_utc - dt.timedelta(days=STALE_DAYS_OK)
    bars = fetch_daily_bars_multi(syms_all, start_utc.isoformat().replace("+00:00","Z"),
                                  end_utc.isoformat().replace("+00:00","Z"))

    # For "big players" context (across the whole universe)
    big_rows = []
    for s, df in bars.items():
        if df is None or len(df) < 25: continue
        close = df["close"].astype(float); vol = df["volume"].astype(float).fillna(0)
        dv20 = (close*vol).rolling(20, min_periods=20).mean().iloc[-1] if len(df)>=20 else np.nan
        rvol_today = (vol.iloc[-1] / vol.rolling(20, min_periods=20).mean().iloc[-1]) if len(df)>=20 and vol.rolling(20, min_periods=20).mean().iloc[-1]>0 else np.nan
        today_dv = float(close.iloc[-1] * vol.iloc[-1])
        big_rows.append((s, today_dv, dv20, rvol_today))
    big_df = pd.DataFrame(big_rows, columns=["symbol","today_dollar_vol","avg_dollar_vol20","rvol_today"]).dropna()
    big_top_rvol = big_df.sort_values("rvol_today", ascending=False).head(10)["symbol"].tolist() if not big_df.empty else []
    big_top_dv   = big_df.sort_values("today_dollar_vol", ascending=False).head(10)["symbol"].tolist() if not big_df.empty else []

    # Step 1→3 screening
    rows = []
    for s in syms_all:
        df = bars.get(s)
        if df is None or len(df) < 60:  # need history
            continue
        close = df["close"].astype(float); high  = df["high"].astype(float); low = df["low"].astype(float)
        vol   = df["volume"].astype(float).fillna(0)

        last = close.iloc[-1]
        if not (PRICE_MIN <= last <= PRICE_MAX):
            continue

        sma20 = sma(close, 20); sma50 = sma(close, 50)
        if REQUIRE_SMA_BIAS and ((sma20.iloc[-1] < sma50.iloc[-1]) or (last < sma50.iloc[-1])):
            continue

        avg_vol20   = vol.rolling(20, min_periods=20).mean().iloc[-1]
        avg_dvol20  = (close*vol).rolling(20, min_periods=20).mean().iloc[-1]
        if not ((avg_vol20 >= LIQ_MIN_AVG_VOL20) or (avg_dvol20 >= LIQ_MIN_AVG_DV20)):
            continue
        step1_count += 1

        _atr = atr(df, 14); atr14 = _atr.iloc[-1]
        atr_pct = (atr14 / last) if last > 0 else np.nan
        rvol_today = (vol.iloc[-1] / (vol.rolling(20, min_periods=20).mean().iloc[-1])) if avg_vol20>0 else np.nan

        sma200 = sma(close, 200).iloc[-1] if len(close)>=200 else np.nan
        if not (ATR_PCT_MIN <= atr_pct <= ATR_PCT_MAX and rvol_today >= RVOL_MIN):
            continue
        if REQ_SMA50_GT_200 and not (sma50.iloc[-1] > sma200):
            continue
        step2_count += 1

        # Gates (only enforce checked ones)
        ema5  = ema(close, 5); ema20 = ema(close, 20); ema50 = ema(close, 50)
        ok = True
        if GATE_EMA:
            ok = ok and (ema5.iloc[-1] > ema20.iloc[-1] > ema50.iloc[-1] and
                         ema5.iloc[-1] > ema5.iloc[-2] and ema20.iloc[-1] > ema20.iloc[-2] and ema50.iloc[-1] > ema50.iloc[-2])

        if GATE_MACD:
            macd_line, macd_sig, macd_hist = macd(close)
            ok = ok and ((macd_hist.iloc[-1] < 0) and (macd_hist.iloc[-1] > macd_hist.iloc[-2]) and
                         (macd_line.iloc[-1] > macd_sig.iloc[-1]) and (macd_line.iloc[-1] > macd_line.iloc[-2]))
        else:
            macd_line, macd_sig, macd_hist = macd(close)

        if GATE_KDJ:
            k, d_, j = kdj(df)
            ok = ok and (k.iloc[-1] > d_.iloc[-1] and
                         k.iloc[-1] > k.iloc[-2] and d_.iloc[-1] > d_.iloc[-2] and j.iloc[-1] > j.iloc[-2] and j.iloc[-1] < 80)
        else:
            k, d_, j = kdj(df)

        if not ok: 
            continue
        step3_count += 1

        roc20 = (last / close.iloc[-20] - 1.0) if len(close) >= 21 else np.nan

        rows.append({
            "symbol": s,
            "close": float(last),
            "avg_vol20": float(avg_vol20),
            "avg_dollar_vol20": float(avg_dvol20),
            "atr14": float(atr14),
            "atr_pct": float(atr_pct),
            "sma20": float(sma20.iloc[-1]),
            "sma50": float(sma50.iloc[-1]),
            "ema5": float(ema5.iloc[-1]),
            "ema20": float(ema20.iloc[-1]),
            "ema50": float(ema50.iloc[-1]),
            "macd_line": float(macd_line.iloc[-1]),
            "macd_signal": float(macd_sig.iloc[-1]),
            "macd_hist": float(macd_hist.iloc[-1]),
            "kdj_k": float(k.iloc[-1]),
            "kdj_d": float(d_.iloc[-1]),
            "kdj_j": float(j.iloc[-1]),
            "rvol_today": float(rvol_today),
            "roc20": float(roc20),
        })

    if not rows:
        # Still return context + counts, but empty candidates
        df_ui = pd.DataFrame(); df_out = pd.DataFrame()
        return sent, risk_mode, total_universe, step1_count, step2_count, step3_count, df_ui, df_out, big_top_rvol, big_top_dv

    df = pd.DataFrame(rows)

    # Rank & select
    df["inv_atr_pct"] = df["atr_pct"].replace(0, np.nan).rpow(-1)
    df["rank_score"]  = 0.4*zscore(df["roc20"]) + 0.3*zscore(df["rvol_today"]) + 0.3*zscore(df["inv_atr_pct"])
    df = df.sort_values("rank_score", ascending=False).reset_index(drop=True)

    # Tags & context on survivors (cap for performance)
    def tag_ema200(daily_df):
        c = daily_df["close"]; e200 = ema(c,200); tags=[]
        if len(e200) >= 201:
            if c.iloc[-1] > e200.iloc[-1] and c.iloc[-2] <= e200.iloc[-2]: tags.append("EMA200_cross_up")
            if c.iloc[-1] < e200.iloc[-1] and c.iloc[-2] >= e200.iloc[-2]: tags.append("EMA200_dip")
            last20 = (c.tail(20) > e200.tail(20)).astype(int)
            if last20.iloc[-1]==1 and last20.sum()>=18 and (c.iloc[-2] < e200.iloc[-2]): tags.append("EMA200_rebreak")
        return tags

    survivors_syms = df["symbol"].head(TAG_SURV_LIMIT).tolist()
    o, c = ny_open_close_utc(dt.datetime.now(dt.timezone.utc))
    h1 = fetch_hourly_bars(survivors_syms, o, c)
    pm = premkt_context(survivors_syms)
    halts = check_halts(survivors_syms)

    for i, row in df.iterrows():
        sym = row["symbol"]; notes = []
        dfd = bars.get(sym); notes += tag_ema200(dfd)
        if sym in h1 and len(h1[sym]) >= 35:
            macd_line_h, macd_sig_h, _ = macd(h1[sym]["close"])
            if macd_line_h.iloc[-1] > 0 and macd_sig_h.iloc[-1] < 0:
                notes.append("H1_MACD_zeroline_fakeout_watch")
        halted = halts.get(sym)
        if halted is True: notes.append("halted")
        elif halted is None: notes.append("halt_check_failed")
        ctx = pm.get(sym, {})
        df.loc[i,"gap_pm_pct"] = ctx.get("gap_pm_pct")
        df.loc[i,"pm_vol"]     = ctx.get("pm_vol")
        df.loc[i,"notes"]      = ", ".join(notes) if notes else ""

    df_ui  = df.head(SHOW_LIMIT).copy()
    df_out = df.head(20).copy()
    as_of = now_et_str()
    df_out.insert(0,"risk_mode", risk_mode)
    df_out.insert(0,"as_of_date", as_of)

    cols = ["as_of_date","risk_mode","symbol","close","avg_vol20","avg_dollar_vol20",
            "atr14","atr_pct","sma20","sma50","ema5","ema20","ema50",
            "macd_line","macd_signal","macd_hist","kdj_k","kdj_d","kdj_j",
            "rvol_today","roc20","rank_score","gap_pm_pct","pm_vol",
            "feasible_qty_at_0p7pct_risk","stop_price_2p5atr","notes"]
    for m in cols:
        if m not in df_out.columns: df_out[m] = np.nan
    df_out = df_out[cols]

    return sent, risk_mode, total_universe, step1_count, step2_count, step3_count, df_ui, df_out, big_top_rvol, big_top_dv

# =========================
# Google Sheets (safe overwrite)
# =========================
def _col_letter(n:int) -> str:
    s=""; 
    while n>0:
        n, r = divmod(n-1, 26); s = chr(65+r) + s
    return s

def _open_or_create_ws(gc, title, rows=200, cols=60):
    sh = gc.open_by_key(GOOGLE_SHEET_ID)
    try: return sh.worksheet(title)
    except Exception: return sh.add_worksheet(title=title, rows=rows, cols=cols)

def write_sheet_safe(df: pd.DataFrame, tab_name: str):
    """Transaction-safe overwrite: write new content first, then trim extras."""
    if df is None or df.empty:
        raise RuntimeError("No rows to write; preserving previous sheet content.")
    import gspread
    from google.oauth2.service_account import Credentials
    sa_info = parse_sa(SA_RAW)
    creds = Credentials.from_service_account_info(sa_info,
               scopes=["https://www.googleapis.com/auth/spreadsheets",
                       "https://www.googleapis.com/auth/drive"])
    gc = gspread.authorize(creds); ws = _open_or_create_ws(gc, tab_name)

    try:
        old_vals = ws.get_all_values(); old_rows = len(old_vals); old_cols = max((len(r) for r in old_vals), default=0)
    except Exception:
        old_rows = old_cols = 0

    new_values = [list(df.columns)] + df.fillna("").astype(str).values.tolist()
    new_rows = len(new_values); new_cols = len(new_values[0]) if new_values else 0

    ws.update("A1", new_values, value_input_option="RAW")

    ranges_to_clear = []
    if old_rows > new_rows and old_cols > 0:
        ranges_to_clear.append(f"A{new_rows+1}:{_col_letter(max(old_cols,new_cols))}")
    if old_cols > new_cols and new_rows > 0:
        ranges_to_clear.append(f"{_col_letter(new_cols+1)}1:{_col_letter(old_cols)}{new_rows}")
    if ranges_to_clear:
        try: ws.batch_clear(ranges_to_clear)
        except Exception: pass

def write_context_tab(sent, risk_mode, counts_tuple, big_top_rvol, big_top_dv):
    """Write 1-row context snapshot to TAB_CONTEXT (always runs)."""
    total_universe, step1, step2, step3 = counts_tuple
    as_of = now_et_str()
    row = {
        "as_of_date": as_of,
        "risk_mode": risk_mode,
        "vix": sent.get("vix", np.nan),
        "universe_size": total_universe,
        "step1_survivors": step1,
        "step2_survivors": step2,
        "step3_survivors": step3,
        "big_rvol_top10": ",".join(big_top_rvol) if big_top_rvol else "",
        "big_dollarvol_top10": ",".join(big_top_dv) if big_top_dv else "",
    }
    for s in SECTORS:
        row[f"sector1d_{s}"] = sent.get("ret1d", {}).get(s, np.nan)
        row[f"sector5d_{s}"] = sent.get("ret5d", {}).get(s, np.nan)
    for s in INDUSTRY_ETFS:
        row[f"industry1d_{s}"] = sent.get("ind1d", {}).get(s, np.nan)
        row[f"industry5d_{s}"] = sent.get("ind5d", {}).get(s, np.nan)
    for m in MEGACAPS:
        row[f"mega1d_{m}"] = sent.get("mega1d", {}).get(m, np.nan)
        row[f"mega5d_{m}"] = sent.get("mega5d", {}).get(m, np.nan)
    df = pd.DataFrame([row]); write_sheet_safe(df, TAB_CONTEXT)

# =========================
# RUN
# =========================
sentiment_col, summary_col = st.columns([2,1])

with st.spinner("Running daily pipeline (Alpaca + yfinance)…"):
    sent, risk_mode, total_universe, step1_c, step2_c, step3_c, df_ui, df_out, big_top_rvol, big_top_dv = run_pipeline()

# Sentiment panel
with sentiment_col:
    st.subheader("Market Sentiment (context only)")
    vix = sent.get("vix", np.nan)
    st.write(
        f"**VIX:** {vix:.2f}" if isinstance(vix,(int,float)) and np.isfinite(vix) else "**VIX:** n/a",
        f" | **Risk mode (auto):** `{sent.get('risk_mode_auto','normal')}`",
        f" | **Using:** `{risk_mode}`"
    )
    st.caption("Universe scanned: **US equities (NASDAQ/NYSE/AMEX)** — Alpaca active & tradable.")
    sect_df = pd.DataFrame({
        "sector": SECTORS,
        "ret1d": [sent.get("ret1d", {}).get(s, np.nan) for s in SECTORS],
        "ret5d": [sent.get("ret5d", {}).get(s, np.nan) for s in SECTORS],
    }).sort_values("ret1d", ascending=False, na_position="last")
    st.dataframe(sect_df.style.format({"ret1d":"{:.2%}","ret5d":"{:.2%}"}), use_container_width=True)

# Summary / controls
with summary_col:
    st.subheader("Run Summary")
    st.write(f"**As of (ET):** {now_et_str()}")
    st.write(f"**Universe size:** {total_universe:,}")
    st.write(f"**Step1/2/3 survivors:** {step1_c:,} / {step2_c:,} / {step3_c:,}")
    st.write(f"**Tabs:** `{TAB_UNIVERSE}`, `{TAB_CONTEXT}`")

# UI table
st.subheader("Top Candidates (UI preview)")
if df_ui.empty:
    st.warning("No survivors today under current gates.")
else:
    fmt = {c:"{:.2f}" for c in ["close","atr14","sma20","sma50","ema5","ema20","ema50",
                                "macd_line","macd_signal","macd_hist","kdj_k","kdj_d","kdj_j",
                                "rvol_today","roc20","rank_score","gap_pm_pct","pm_vol","avg_vol20","avg_dollar_vol20","atr_pct"]}
    st.dataframe(df_ui.head(SHOW_LIMIT).style.format(fmt), use_container_width=True)

# Write to Google Sheets (Context always; Universe only if rows)
try:
    write_context_tab(sent, risk_mode, (total_universe, step1_c, step2_c, step3_c), big_top_rvol, big_top_dv)
    if df_out.empty:
        st.warning("No survivors — Universe tab left unchanged. Context tab updated.")
    else:
        write_sheet_safe(df_out, TAB_UNIVERSE)
        st.success(f"Wrote Top-20 to `{TAB_UNIVERSE}` and updated `{TAB_CONTEXT}` at {now_et_str()} ET.")
except Exception as e:
    st.error(f"Sheets write failed (previous data preserved): {e}")

st.caption("RTH-only indicators; pre-/post-market used only for context. ET timestamps are written into every row. Sidebar controls let you tune Step 1/2/3 thresholds live.")
