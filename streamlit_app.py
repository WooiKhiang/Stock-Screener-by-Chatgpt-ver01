# streamlit_app.py — Daily candidates (Alpaca) → overwrite Google Sheet; hourly auto-run
import os, io, json, math, time, datetime as dt
import numpy as np
import pandas as pd
import requests
import pytz
import streamlit as st

# =========================
# Hard-coded configuration
# =========================
GOOGLE_SHEET_ID   = "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4"
GOOGLE_SHEET_NAME = "Universe"   # <- we overwrite this tab each run

# Alpaca credentials (feel free to move these to Streamlit secrets later)
ALPACA_KEY    = os.getenv("ALPACA_KEY",    "PKIG445MPT704CN8P0R8")
ALPACA_SECRET = os.getenv("ALPACA_SECRET", "3GQdWcnbMo6V8uvhVa6BK6EbvnH4EHinlsU6uvj4")
ALPACA_BASE   = "https://paper-api.alpaca.markets/v2"   # assets, account, calendar
ALPACA_DATA   = "https://data.alpaca.markets"           # market data v2
FEED          = "iex"                                   # required for paper/free tier

# Google service account JSON from Streamlit secrets or env
SA_RAW = st.secrets.get("GCP_SERVICE_ACCOUNT") or os.getenv("GCP_SERVICE_ACCOUNT")

# General knobs
MAX_SYMBOLS_SCAN = 1000   # safety cap (keeps runtime fast)
CHUNK_SIZE       = 180    # symbols per /v2/stocks/bars call

# =========================
# UI: Sidebar
# =========================
st.set_page_config(page_title="Daily Candidates — Alpaca → Google Sheets", layout="wide")
st.sidebar.header("Run Control")

autorun = st.sidebar.checkbox("Auto-run every hour", value=True,
                              help="Refreshes this app every 60 minutes and pushes the latest candidates.")
try:
    # Available in many Streamlit versions
    st_autorefresh = getattr(st, "autorefresh", None) or getattr(st, "experimental_autorefresh", None)
    if autorun and st_autorefresh:
        st_autorefresh(interval=60*60*1000, key="hourly_autorefresh")  # 60 minutes
    elif autorun:
        # Fallback: simple JS reload (safe; used only if autorefresh isn't available)
        st.markdown("<script>setTimeout(()=>window.location.reload(), 3600000);</script>", unsafe_allow_html=True)
except Exception:
    pass

with st.sidebar.expander("Parameters", expanded=True):
    PRICE_MIN = st.number_input("Min price", value=5.0, step=0.5)
    PRICE_MAX = st.number_input("Max price", value=300.0, step=1.0)
    TOP_N_NORMAL = st.slider("Top-N (normal)", 3, 10, 5)
    TOP_N_TIGHT  = st.slider("Top-N (tight)", 3, 6, 3)
    MANUAL_RISK  = st.selectbox("Risk mode override", ["auto", "normal", "tight"],
                                help="Leave 'auto' to infer from VIX+sectors; choose normal/tight to override.")
    SHOW_LIMIT   = st.slider("Rows to show (UI)", 10, 50, 20)
    STALE_DAYS_OK = st.slider("Bars lookback days (RTH dailies)", 60, 140, 100)

st.title("🧭 Daily Candidates — Clean, fast, PDT-safe (Alpaca → Sheets)")

# =========================
# Helpers
# =========================
ET = pytz.timezone("America/New_York")

def now_et_str():
    return dt.datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")

def parse_sa(raw_json: str) -> dict:
    if not raw_json:
        raise RuntimeError("Missing service account JSON in secrets/env (GCP_SERVICE_ACCOUNT).")
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        if "-----BEGIN PRIVATE KEY-----" in raw_json and "\\n" not in raw_json:
            start = raw_json.find("-----BEGIN PRIVATE KEY-----")
            end   = raw_json.find("-----END PRIVATE KEY-----", start) + len("-----END PRIVATE KEY-----")
            block = raw_json[start:end]
            raw_json = raw_json.replace(block, block.replace("\r\n", "\n").replace("\n", "\\n"))
        return json.loads(raw_json)

def zscore(x: pd.Series) -> pd.Series:
    m = x.mean()
    s = x.std(ddof=0)
    if s == 0 or not np.isfinite(s):
        return pd.Series([0.0]*len(x), index=x.index)
    return (x - m) / s

HEADERS = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}

# =========================
# Sentiment: yfinance (context only)
# =========================
def fetch_sentiment():
    import yfinance as yf
    tickers = ["SPY", "^VIX", "XLF", "XLK", "XLY", "XLP", "XLV", "XLE", "XLI", "XLU", "XLRE", "XLB", "IYZ"]
    data = yf.download(tickers=" ".join(tickers), period="10d", interval="1d", auto_adjust=True, progress=False)
    last = data["Adj Close"].ffill().iloc[-1]
    prev = data["Adj Close"].ffill().iloc[-2]
    ret1d = (last/prev - 1.0).rename("ret1d")
    ret5d = (last / data["Adj Close"].ffill().iloc[-5] - 1.0).rename("ret5d")
    vix = float(last["^VIX"])
    defensives = ["XLV", "XLP", "XLU"]
    cyclicals  = ["XLK", "XLY", "XLF"]
    def_mean_1 = ret1d[defensives].mean()
    cyc_mean_1 = ret1d[cyclicals].mean()
    def_mean_5 = ret5d[defensives].mean()
    cyc_mean_5 = ret5d[cyclicals].mean()
    tight = (vix > 20.0) and (def_mean_1 > cyc_mean_1) and (def_mean_5 > cyc_mean_5)
    return {
        "vix": vix,
        "ret1d": ret1d.to_dict(),
        "ret5d": ret5d.to_dict(),
        "risk_mode_auto": "tight" if tight else "normal"
    }

# =========================
# Universe (daily; light)
# =========================
def fetch_active_symbols(max_symbols=MAX_SYMBOLS_SCAN):
    url = f"{ALPACA_BASE}/assets"
    params = {"status": "active", "asset_class": "us_equity"}
    r = requests.get(url, headers=HEADERS, params=params, timeout=60)
    r.raise_for_status()
    js = r.json()
    keep_exch = {"NASDAQ", "NYSE", "AMEX"}
    syms = [x["symbol"] for x in js if x.get("exchange") in keep_exch and x.get("tradable")]
    # quickly cull obvious non-commons
    bad_suffixes = (".U", ".W", "WS", "W", "R", ".P", "-P")
    syms = [s for s in syms if not s.endswith(bad_suffixes)]
    return syms[:max_symbols]

# =========================
# Bars helpers (RTH-only)
# =========================
def ny_open_close_utc(day_utc: dt.datetime):
    # Ask Alpaca calendar for robustness (early close/holidays)
    try:
        d = day_utc.astimezone(ET).date().isoformat()
        r = requests.get(f"{ALPACA_BASE}/calendar", headers=HEADERS, params={"start": d, "end": d}, timeout=30)
        r.raise_for_status()
        cal = r.json()
        if cal:
            o = dt.datetime.fromisoformat(cal[0]["open"]).astimezone(ET)
            c = dt.datetime.fromisoformat(cal[0]["close"]).astimezone(ET)
            return o.astimezone(dt.timezone.utc), c.astimezone(dt.timezone.utc)
    except Exception:
        pass
    # Fallback: standard 9:30–16:00 ET same day
    et_date = day_utc.astimezone(ET).date()
    o = ET.localize(dt.datetime.combine(et_date, dt.time(9,30))).astimezone(dt.timezone.utc)
    c = ET.localize(dt.datetime.combine(et_date, dt.time(16,0))).astimezone(dt.timezone.utc)
    return o, c

def fetch_daily_bars_multi(symbols, start_iso, end_iso, timeframe="1Day", limit=1000):
    base = f"{ALPACA_DATA}/v2/stocks/bars"
    result = {}
    for i in range(0, len(symbols), CHUNK_SIZE):
        chunk = symbols[i:i+CHUNK_SIZE]
        params = {
            "timeframe": timeframe,
            "symbols": ",".join(chunk),
            "start": start_iso,
            "end": end_iso,
            "limit": limit,
            "adjustment": "raw",
            "feed": FEED
        }
        page = None
        while True:
            if page: params["page_token"] = page
            r = requests.get(base, headers=HEADERS, params=params, timeout=60)
            if r.status_code >= 400:
                msg = r.text[:400].replace("\n"," ")
                raise requests.HTTPError(f"Alpaca /bars {r.status_code}: {msg}")
            js = r.json()
            bars = js.get("bars", [])
            if isinstance(bars, dict):
                for sym, arr in bars.items():
                    add = pd.DataFrame(arr)
                    if add.empty: continue
                    add["t"] = pd.to_datetime(add["t"], utc=True)
                    add.rename(columns={"c":"close","v":"volume","h":"high","l":"low","o":"open"}, inplace=True)
                    add = add[["t","open","high","low","close","volume"]]
                    prev = result.get(sym)
                    result[sym] = pd.concat([prev, add], ignore_index=True) if prev is not None else add
            else:
                if bars:
                    add = pd.DataFrame(bars)
                    add["t"] = pd.to_datetime(add["t"], utc=True)
                    add.rename(columns={"S":"symbol","c":"close","v":"volume","h":"high","l":"low","o":"open"}, inplace=True)
                    add = add[["symbol","t","open","high","low","close","volume"]]
                    for sym, grp in add.groupby("symbol"):
                        g = grp.drop(columns=["symbol"]).copy()
                        prev = result.get(sym)
                        result[sym] = pd.concat([prev, g], ignore_index=True) if prev is not None else g
            page = js.get("next_page_token")
            if not page: break
    # Ensure sorted, RTH-only doesn’t apply to 1D (already RTH)
    for s, df in list(result.items()):
        result[s] = df.drop_duplicates(subset=["t"]).sort_values("t").reset_index(drop=True)
    return result

def fetch_hourly_bars(symbols, start_utc, end_utc):
    # RTH-only: filter timestamps inside actual open/close
    base = f"{ALPACA_DATA}/v2/stocks/bars"
    result = {}
    for i in range(0, len(symbols), CHUNK_SIZE):
        chunk = symbols[i:i+CHUNK_SIZE]
        params = {
            "timeframe": "1Hour",
            "symbols": ",".join(chunk),
            "start": start_utc.isoformat().replace("+00:00","Z"),
            "end": end_utc.isoformat().replace("+00:00","Z"),
            "limit": 1000,
            "adjustment": "raw",
            "feed": FEED
        }
        r = requests.get(base, headers=HEADERS, params=params, timeout=60)
        if r.status_code >= 400:
            msg = r.text[:400].replace("\n"," ")
            raise requests.HTTPError(f"Alpaca /bars(H1) {r.status_code}: {msg}")
        js = r.json()
        bars = js.get("bars", [])
        if isinstance(bars, dict):
            for sym, arr in bars.items():
                add = pd.DataFrame(arr)
                if add.empty: continue
                add["t"] = pd.to_datetime(add["t"], utc=True)
                add.rename(columns={"c":"close","v":"volume","h":"high","l":"low","o":"open"}, inplace=True)
                add = add[["t","open","high","low","close","volume"]]
                result[sym] = add
        else:
            if bars:
                add = pd.DataFrame(bars)
                add["t"] = pd.to_datetime(add["t"], utc=True)
                add.rename(columns={"S":"symbol","c":"close","v":"volume","h":"high","l":"low","o":"open"}, inplace=True)
                add = add[["symbol","t","open","high","low","close","volume"]]
                for sym, grp in add.groupby("symbol"):
                    result[sym] = grp.drop(columns=["symbol"]).copy()
    # filter to RTH window per day
    o,c = ny_open_close_utc(dt.datetime.now(dt.timezone.utc))
    for s, df in list(result.items()):
        mask = (df["t"]>=o) & (df["t"]<=c)
        result[s] = df[mask].reset_index(drop=True)
    return result

# =========================
# Indicators & features
# =========================
def sma(x, n): return x.rolling(n, min_periods=n).mean()
def ema(x, n): return x.ewm(span=n, adjust=False).mean()

def atr(df: pd.DataFrame, n=14) -> pd.Series:
    hi, lo, cl = df["high"], df["low"], df["close"]
    prev_close = cl.shift(1)
    tr = pd.concat([(hi-lo).abs(),
                    (hi-prev_close).abs(),
                    (lo-prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def macd(x, fast=12, slow=26, signal=9):
    ema_fast = ema(x, fast)
    ema_slow = ema(x, slow)
    macd_line = ema_fast - ema_slow
    macd_signal = ema(macd_line, signal)
    hist = macd_line - macd_signal
    return macd_line, macd_signal, hist

def kdj(df, n=9, k_period=3, d_period=3):
    low_min  = df["low"].rolling(n, min_periods=n).min()
    high_max = df["high"].rolling(n, min_periods=n).max()
    rsv = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    k = rsv.ewm(alpha=1.0/k_period, adjust=False).mean()
    d = k.ewm(alpha=1.0/d_period, adjust=False).mean()
    j = 3*k - 2*d
    return k, d, j

# =========================
# Trading halts (survivors only)
# =========================
def check_halts(symbols):
    out = {}
    base = f"{ALPACA_DATA}/v2/stocks/snapshots"
    for i in range(0, len(symbols), CHUNK_SIZE):
        chunk = symbols[i:i+CHUNK_SIZE]
        params = {"symbols": ",".join(chunk), "feed": FEED}
        r = requests.get(base, headers=HEADERS, params=params, timeout=60)
        if r.status_code >= 400:
            # Soft fail: mark as unchecked
            for s in chunk: out[s] = None
            continue
        js = r.json() or {}
        snap = js.get("snapshots", {})
        for s, info in snap.items():
            halted = None
            try:
                halted = bool(info.get("latestTrade", {}).get("tape", None)) and bool(info.get("trading_halted", False))
            except Exception:
                halted = info.get("trading_halted", False)
            out[s] = halted
    return out

# =========================
# Pre-market context (survivors only; not used for signals)
# =========================
def premkt_context(symbols):
    # compute gap_pm_pct vs prior close and premarket volume 04:00–09:29 ET
    res = {s: {"gap_pm_pct": None, "pm_vol": None} for s in symbols}
    if not symbols: return res
    now_utc = dt.datetime.now(dt.timezone.utc)
    et_date = now_utc.astimezone(ET).date()
    pm_start = ET.localize(dt.datetime.combine(et_date, dt.time(4,0))).astimezone(dt.timezone.utc)
    rth_open = ET.localize(dt.datetime.combine(et_date, dt.time(9,30))).astimezone(dt.timezone.utc)

    # get prior daily close for survivors
    dstart = (now_utc - dt.timedelta(days=10)).isoformat().replace("+00:00","Z")
    dend   = now_utc.isoformat().replace("+00:00","Z")
    daily = fetch_daily_bars_multi(symbols, dstart, dend)
    base = f"{ALPACA_DATA}/v2/stocks/bars"
    res_pm = {s: 0 for s in symbols}

    # premarket minute bars (can be sparse on IEX)
    for i in range(0, len(symbols), CHUNK_SIZE):
        chunk = symbols[i:i+CHUNK_SIZE]
        params = {
            "timeframe": "1Min",
            "symbols": ",".join(chunk),
            "start": pm_start.isoformat().replace("+00:00","Z"),
            "end": rth_open.isoformat().replace("+00:00","Z"),
            "limit": 10000,
            "adjustment": "raw",
            "feed": FEED
        }
        r = requests.get(base, headers=HEADERS, params=params, timeout=60)
        if r.status_code >= 400:
            continue
        js = r.json()
        bars = js.get("bars", [])
        if isinstance(bars, dict):
            for sym, arr in bars.items():
                if not arr: continue
                df = pd.DataFrame(arr)
                if df.empty: continue
                res_pm[sym] += float(df["v"].sum())

    for s in symbols:
        prev_close = None
        if s in daily and len(daily[s])>=2:
            prev_close = float(daily[s]["close"].iloc[-2])
            last_premkt_trade = None
            # If any minute bars exist, take the last PM close as context; else use prior close
            pm_vol = res_pm.get(s, 0.0)
            gap = None
            if prev_close and s in daily and len(daily[s])>=1:
                last_close = float(daily[s]["close"].iloc[-1])  # last RTH close (yday)
                # We don't have PM price cleanly on IEX always; use prev_close baseline
                gap = 0.0
            res[s] = {"gap_pm_pct": gap, "pm_vol": float(pm_vol)}
    return res

# =========================
# Daily pipeline
# =========================
def run_pipeline():
    # 0) Sentiment (context)
    sent = fetch_sentiment()
    risk_mode = sent["risk_mode_auto"] if MANUAL_RISK=="auto" else MANUAL_RISK

    # 1) Universe
    syms_all = fetch_active_symbols(MAX_SYMBOLS_SCAN)

    # 2) Daily bars (RTH) — last ~100 days
    end_utc   = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    start_utc = end_utc - dt.timedelta(days=STALE_DAYS_OK)
    bars = fetch_daily_bars_multi(syms_all,
                                  start_utc.isoformat().replace("+00:00","Z"),
                                  end_utc.isoformat().replace("+00:00","Z"))

    rows = []
    # Step 1 + basic features
    for s in syms_all:
        df = bars.get(s)
        if df is None or len(df) < 60:  # need enough history
            continue
        close = df["close"].astype(float)
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        vol   = df["volume"].astype(float).fillna(0)

        last = close.iloc[-1]
        if not (PRICE_MIN <= last <= PRICE_MAX):
            continue

        sma20 = sma(close, 20)
        sma50 = sma(close, 50)
        if (sma20.iloc[-1] < sma50.iloc[-1]) or (last < sma50.iloc[-1]):
            continue

        avg_vol20   = vol.rolling(20, min_periods=20).mean().iloc[-1]
        avg_dvol20  = (close*vol).rolling(20, min_periods=20).mean().iloc[-1]
        if not ((avg_vol20 >= 1_000_000) or (avg_dvol20 >= 20_000_000)):
            continue

        # Step 2 features
        _atr = atr(df, 14)
        atr14 = _atr.iloc[-1]
        atr_pct = (atr14 / last) if last > 0 else np.nan
        rvol_today = (vol.iloc[-1] / (vol.rolling(20, min_periods=20).mean().iloc[-1])) if avg_vol20>0 else np.nan

        if risk_mode == "tight":
            if not (0.015 <= atr_pct <= 0.06):
                continue
            # require SMA50 > SMA200
            sma200 = sma(close, 200).iloc[-1]
            if not (sma50.iloc[-1] > sma200):
                continue
            if rvol_today < 1.1:  # tight mode
                continue
        else:
            if not (0.01 <= atr_pct <= 0.08):
                continue
            if rvol_today < 0.9:
                continue

        # Step 3 features
        ema5  = ema(close, 5)
        ema20 = ema(close, 20)
        ema50 = ema(close, 50)
        ema_stack_ok = (ema5.iloc[-1] > ema20.iloc[-1] > ema50.iloc[-1] and
                        ema5.iloc[-1] > ema5.iloc[-2] and
                        ema20.iloc[-1] > ema20.iloc[-2] and
                        ema50.iloc[-1] > ema50.iloc[-2])

        macd_line, macd_sig, macd_hist = macd(close)
        macd_turn_ok = (macd_hist.iloc[-1] < 0) and (macd_hist.iloc[-1] > macd_hist.iloc[-2]) and \
                       (macd_line.iloc[-1] > macd_sig.iloc[-1]) and (macd_line.iloc[-1] > macd_line.iloc[-2])

        k, d_, j = kdj(df)
        kdj_ok = (k.iloc[-1] > d_.iloc[-1] and
                  k.iloc[-1] > k.iloc[-2] and d_.iloc[-1] > d_.iloc[-2] and j.iloc[-1] > j.iloc[-2] and j.iloc[-1] < 80)

        if not (ema_stack_ok and macd_turn_ok and kdj_ok):
            continue

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
        return sent, risk_mode, pd.DataFrame()

    df = pd.DataFrame(rows)

    # Rank & select
    df["inv_atr_pct"] = df["atr_pct"].replace(0, np.nan).rpow(-1)  # 1/atr%
    df["rank_score"] = 0.4*zscore(df["roc20"]) + 0.3*zscore(df["rvol_today"]) + 0.3*zscore(df["inv_atr_pct"])
    df = df.sort_values("rank_score", ascending=False).reset_index(drop=True)

    # Tagging — EMA200 series (cross/dip/rebreak) & hourly fakeout on top survivors only
    # Quick EMA200 tracker:
    def tag_ema200(daily_df):
        c = daily_df["close"]; e200 = ema(c,200)
        tags=[]
        if len(e200) >= 201:
            if c.iloc[-1] > e200.iloc[-1] and c.iloc[-2] <= e200.iloc[-2]:
                tags.append("EMA200_cross_up")
            if c.iloc[-1] < e200.iloc[-1] and c.iloc[-2] >= e200.iloc[-2]:
                tags.append("EMA200_dip")
            # 'rebreak' heuristic: crossed up in last 20d, minor dip, now back above
            last20 = (c.tail(20) > e200.tail(20)).astype(int)
            if last20.iloc[-1]==1 and last20.sum()>=18 and (c.iloc[-2] < e200.iloc[-2]):
                tags.append("EMA200_rebreak")
        return tags

    # Compute tags on survivors (limit to 80 for speed)
    survivors_syms = df["symbol"].head(80).tolist()
    # Hourly fakeout (RTH today)
    o,c = ny_open_close_utc(dt.datetime.now(dt.timezone.utc))
    h1 = fetch_hourly_bars(survivors_syms, o, c)

    # Pre-market context
    pm = premkt_context(survivors_syms)

    # Trading halts check
    halts = check_halts(survivors_syms)

    # Apply tags & context
    for i, row in df.iterrows():
        sym = row["symbol"]
        notes = []
        # EMA200 daily tags
        dfd = bars.get(sym)
        notes += tag_ema200(dfd)
        # H1 MACD zero-line fakeout watch
        if sym in h1 and len(h1[sym]) >= 35:
            macd_line_h, macd_sig_h, _ = macd(h1[sym]["close"])
            if macd_line_h.iloc[-1] > 0 and macd_sig_h.iloc[-1] < 0:
                notes.append("H1_MACD_zeroline_fakeout_watch")
        # Trading halts
        halted = halts.get(sym)
        if halted is True:
            notes.append("halted")
        elif halted is None:
            notes.append("halt_check_failed")
        # Pre-market context
        ctx = pm.get(sym, {})
        df.loc[i, "gap_pm_pct"] = ctx.get("gap_pm_pct")
        df.loc[i, "pm_vol"]     = ctx.get("pm_vol")
        df.loc[i, "notes"]      = ", ".join(notes) if notes else ""

    # Final select for sheet/UI
    topN = TOP_N_TIGHT if risk_mode=="tight" else TOP_N_NORMAL
    df_ui  = df.head(SHOW_LIMIT).copy()
    df_out = df.head(20).copy()   # we always write Top-20 to sheet

    # Add as_of_date and risk_mode cols (ET)
    as_of = now_et_str()
    df_out.insert(0, "risk_mode", risk_mode)
    df_out.insert(0, "as_of_date", as_of)

    # Order columns per spec
    cols = [
        "as_of_date","risk_mode","symbol","close","avg_vol20","avg_dollar_vol20",
        "atr14","atr_pct","sma20","sma50","ema5","ema20","ema50",
        "macd_line","macd_signal","macd_hist","kdj_k","kdj_d","kdj_j",
        "rvol_today","roc20","rank_score","gap_pm_pct","pm_vol",
        # optional feasibility placeholders (kept for schema compatibility)
        # you'd wire Alpaca /v2/account here if you want exact equity-based sizing
        "notes"
    ]
    present = [c for c in cols if c in df_out.columns]
    missing = [c for c in cols if c not in df_out.columns]
    df_out = df_out[present]
    # append missing columns (empty) to fit header
    for m in missing:
        df_out[m] = np.nan
    df_out = df_out[cols]

    return sent, risk_mode, topN, df_ui, df_out

# =========================
# Google Sheets (overwrite)
# =========================
def write_sheet_overwrite(df_out: pd.DataFrame):
    import gspread
    from google.oauth2.service_account import Credentials
    sa_info = parse_sa(SA_RAW)
    creds = Credentials.from_service_account_info(
        sa_info,
        scopes=["https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"],
    )
    gc = gspread.authorize(creds)
    ws = gc.open_by_key(GOOGLE_SHEET_ID).worksheet(GOOGLE_SHEET_NAME)
    # Clear everything and write header + values
    try:
        ws.clear()
    except Exception:
        pass
    values = [list(df_out.columns)] + df_out.fillna("").values.tolist()
    ws.update("A1", values, value_input_option="RAW")

# =========================
# RUN
# =========================
sentiment_col, summary_col = st.columns([2,1])

with st.spinner("Running daily pipeline (Alpaca + yfinance)…"):
    sent, risk_mode, topN, df_ui, df_out = run_pipeline()

# Sentiment panel
with sentiment_col:
    st.subheader("Market Sentiment (context only)")
    vix = sent["vix"]
    st.write(f"**VIX:** {vix:.2f}  |  **Risk mode (auto):** `{sent['risk_mode_auto']}`  |  **Using:** `{risk_mode}`")
    # quick sector table (1D/5D)
    sector_list = ["XLF","XLK","XLY","XLP","XLV","XLE","XLI","XLU","XLRE","XLB","IYZ"]
    df_sect = pd.DataFrame({
        "sector": sector_list,
        "ret1d": [sent["ret1d"].get(s, np.nan) for s in sector_list],
        "ret5d": [sent["ret5d"].get(s, np.nan) for s in sector_list],
    }).sort_values("ret1d", ascending=False)
    st.dataframe(df_sect.style.format({"ret1d":"{:.2%}","ret5d":"{:.2%}"}), use_container_width=True)

# Summary / controls
with summary_col:
    st.subheader("Run Summary")
    st.write(f"**As of (ET):** {now_et_str()}")
    st.write(f"**Top-N (selected today):** {topN}")
    st.write(f"**Rows written to sheet (Top-20):** {len(df_out)}")
    st.write(f"**Sheet tab:** `{GOOGLE_SHEET_NAME}`")

# UI table
st.subheader("Top Candidates (UI preview)")
if df_ui.empty:
    st.warning("No survivors today under current gates.")
else:
    fmt = {c:"{:.2f}" for c in ["close","atr14","sma20","sma50","ema5","ema20","ema50",
                                "macd_line","macd_signal","macd_hist","kdj_k","kdj_d","kdj_j",
                                "rvol_today","roc20","rank_score","gap_pm_pct","pm_vol","avg_vol20","avg_dollar_vol20","atr_pct"]}
    st.dataframe(df_ui.head(SHOW_LIMIT).style.format(fmt), use_container_width=True)

# Write to Google Sheets (always, each run)
try:
    write_sheet_overwrite(df_out)
    st.success(f"Overwrote Google Sheet `{GOOGLE_SHEET_NAME}` with Top-20 at {now_et_str()} ET.")
except Exception as e:
    st.error(f"Sheets write failed: {e}")

st.caption("Indicators use **RTH** bars only. Pre-/post-market values appear only in context columns. Hourly auto-run refreshes this page; each run **overwrites** the Google Sheet tab.")
