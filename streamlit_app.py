# Market Scanner — Volume-first (no-lookback), minimal hydration, optional signals → Google Sheets
# Tabs (overwrite each run): ActiveNow, SignalsH1, SignalsD1, Context
# Data: Alpaca snapshots (IEX). No historical bars except: (a) one small global hydration of daily bars (capped), (b) optional signals on Top-K only.

import os, json, time, random, datetime as dt
import numpy as np
import pandas as pd
import pytz, requests, streamlit as st

# ────────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────────
GOOGLE_SHEET_ID = "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4"

TAB_ACTIVENOW = "ActiveNow"
TAB_SIG_H1    = "SignalsH1"
TAB_SIG_D1    = "SignalsD1"
TAB_CONTEXT   = "Context"

# Alpaca creds (env may override)
ALPACA_KEY    = os.getenv("ALPACA_KEY",    "PKIG445MPT704CN8P0R8")
ALPACA_SECRET = os.getenv("ALPACA_SECRET", "3GQdWcnbMo6V8uvhVa6BK6EbvnH4EHinlsU6uvj4")
ALPACA_BASE   = "https://paper-api.alpaca.markets/v2"
ALPACA_DATA   = "https://data.alpaca.markets"
FEED          = "iex"   # avoid SIP entitlement issues

SA_RAW = st.secrets.get("GCP_SERVICE_ACCOUNT") or os.getenv("GCP_SERVICE_ACCOUNT")

ET = pytz.timezone("America/New_York")
HEADERS = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}

# ────────────────────────────────────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Market Scanner — Volume-first", layout="wide")
st.title("⚡ Market Scanner — Volume-first (Snapshots only, minimal hydration)")

# Auto-refresh (pure JS, no components)
st.sidebar.markdown("### Auto-refresh")
REFRESH_MIN = st.sidebar.slider("Refresh every (minutes)", 5, 120, 60, 5)
if st.sidebar.checkbox("Enable auto-refresh", value=True, key="auto_js"):
    st.markdown(
        f"<script>setTimeout(()=>window.location.reload(), {REFRESH_MIN*60*1000});</script>",
        unsafe_allow_html=True
    )

# Universe & filters
st.sidebar.markdown("### Universe")
UNIVERSE_CAP = st.sidebar.slider("Symbols to scan (cap)", 500, 12000, 6000, 500)

st.sidebar.markdown("### Filters")
PRICE_MIN = st.sidebar.number_input("Min price ($)", value=5.0, step=0.5)
PRICE_MAX = st.sidebar.number_input("Max price ($)", value=100.0, step=1.0)
TOP_K_ACT = st.sidebar.slider("Top K to keep (by volume)", 100, 3000, 800, 100)
DHP_DELTA = st.sidebar.slider("Day-high proximity δ", 0.001, 0.02, 0.005, 0.001,
                              help="Flag if last ≥ today's high × (1−δ). Requires today's high to exist.")

st.sidebar.markdown("### Network / Degradation Controls")
SNAPSHOT_CHUNK = st.sidebar.slider("Snapshot chunk size", 60, 240, 120, 20)
MAX_HYDRATE    = st.sidebar.slider("Max symbols to hydrate with 1-Day bars (0 = disable)", 0, 1000, 250, 50)

st.sidebar.markdown("### Signals (Top-K only, optional)")
DO_SIGNALS   = st.sidebar.checkbox("Compute signals (Top-K only)", value=False)
TOP_K_SIG    = st.sidebar.slider("Symbols to compute signals on", 50, 800, 200, 50, disabled=not DO_SIGNALS)
H1_LIMIT     = st.sidebar.slider("Hourly bars (for MACD)", 40, 120, 80, 20, disabled=not DO_SIGNALS)
D1_LIMIT     = st.sidebar.slider("Daily bars (for EMA200)", 210, 260, 220, 10, disabled=not DO_SIGNALS)
BARS_CHUNK   = st.sidebar.slider("Bars chunk size (signals only)", 40, 160, 100, 20, disabled=not DO_SIGNALS)

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────
def now_et():
    return dt.datetime.now(ET)

def now_et_str():
    return now_et().strftime("%Y-%m-%d %H:%M:%S")

def session_status():
    n = now_et()
    if n.weekday() >= 5: return "closed"
    t = n.time()
    if dt.time(4,0)  <= t < dt.time(9,30): return "pre"
    if dt.time(9,30) <= t <= dt.time(16,0): return "regular"
    if dt.time(16,0) < t <= dt.time(20,0):  return "post"
    return "closed"

def is_market_open_now():
    return session_status() == "regular"

def parse_sa(raw_json: str) -> dict:
    if not raw_json: raise RuntimeError("Missing GCP_SERVICE_ACCOUNT in secrets/env.")
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        if "-----BEGIN PRIVATE KEY-----" in raw_json and "\\n" not in raw_json:
            s = raw_json.find("-----BEGIN PRIVATE KEY-----")
            e = raw_json.find("-----END PRIVATE KEY-----", s) + len("-----END PRIVATE KEY-----")
            raw_json = raw_json.replace(raw_json[s:e], raw_json[s:e].replace("\r\n","\n").replace("\n","\\n"))
        return json.loads(raw_json)

def safe_float(x):
    try:
        v = float(x);  return v if v > 0 else None
    except Exception:
        return None

# ────────────────────────────────────────────────────────────────────────────────
# Alpaca I/O
# ────────────────────────────────────────────────────────────────────────────────
_last_data_error = {}
diag = {
    "snapshot_batches": 0,
    "symbols_snap_nonempty": 0,
    "symbols_snap_empty": 0,
    "hydrated_symbols": 0,
}
src_counts = {"price": {"trade":0,"minbar":0,"daybar":0,"prevday":0,"quote_mid":0,"none":0},
              "vol":   {"minute":0,"daily":0,"prev_daily":0}}

def fetch_active_symbols(cap: int):
    url = f"{ALPACA_BASE}/assets"
    params = {"status": "active", "asset_class": "us_equity"}
    r = requests.get(url, headers=HEADERS, params=params, timeout=60); r.raise_for_status()
    keep_exch = {"NASDAQ","NYSE","AMEX"}
    syms = []
    for x in r.json():
        if x.get("exchange") in keep_exch and x.get("tradable"):
            s = x.get("symbol","")
            if not s: continue
            if s.endswith((".U",".W","WS","W","R",".P","-P")):  # strip units/warrants/prefs
                continue
            syms.append(s)
        if len(syms) >= cap: break
    return syms[:cap]

def _snapshots_request(symbols_batch, max_retries=4, timeout_s=90):
    url = f"{ALPACA_DATA}/v2/stocks/snapshots"
    params = {"symbols": ",".join(symbols_batch), "feed": FEED}
    backoff = 1.0
    for _ in range(max_retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=timeout_s)
        except requests.Timeout:
            pass
        else:
            if r.status_code in (429,500,502,503,504):
                pass
            elif r.status_code >= 400:
                try: msg = r.json().get("message", r.text[:300])
                except Exception: msg = r.text[:300]
                _last_data_error["snapshots"] = (r.status_code, msg); return None
            else:
                try: return r.json()
                except Exception:
                    _last_data_error["snapshots"] = (r.status_code, "JSON parse error"); return None
        time.sleep(backoff + random.random()*0.4)
        backoff = min(backoff*1.8, 6.0)
    _last_data_error["snapshots"] = (408, "retry timeout"); return None

def _bars_request(symbols, timeframe, limit, chunk):
    """Lightweight bars pull — used only for (a) limited hydration, (b) optional signals."""
    out = {}
    url = f"{ALPACA_DATA}/v2/stocks/bars"
    end = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    # generous window, capped by limit
    start = end - (dt.timedelta(days=90) if timeframe == "1Hour" else dt.timedelta(days=400))
    start_iso = start.isoformat().replace("+00:00","Z"); end_iso = end.isoformat().replace("+00:00","Z")
    i = 0
    while i < len(symbols):
        batch = symbols[i:i+chunk]
        params = {"symbols": ",".join(batch), "timeframe": timeframe, "limit": limit,
                  "adjustment": "raw", "feed": FEED, "start": start_iso, "end": end_iso}
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=60)
            if r.status_code >= 400:
                try: msg = r.json().get("message", r.text[:300])
                except Exception: msg = r.text[:300]
                _last_data_error[f"bars_{timeframe}"] = (r.status_code, msg); js = {}
            else:
                js = r.json()
        except Exception:
            js = {}
        bars = js.get("bars", [])
        if isinstance(bars, dict):
            for sym, arr in bars.items():
                df = pd.DataFrame(arr) if arr else pd.DataFrame()
                if not df.empty:
                    df["t"] = pd.to_datetime(df["t"], utc=True)
                    df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"}, inplace=True)
                    out[sym] = df[["t","open","high","low","close","volume"]].sort_values("t").reset_index(drop=True)
                else:
                    out[sym] = pd.DataFrame(columns=["t","open","high","low","close","volume"])
        else:
            if bars:
                df = pd.DataFrame(bars)
                df["t"] = pd.to_datetime(df["t"], utc=True)
                df.rename(columns={"S":"symbol","o":"open","h":"high","l":"low","c":"close","v":"volume"}, inplace=True)
                for sym, grp in df.groupby("symbol"):
                    out[sym] = grp[["t","open","high","low","close","volume"]].sort_values("t").reset_index(drop=True)
        i += len(batch); time.sleep(0.08)
    return out

def fetch_snapshots_multi(symbols, chunk=120):
    """One full snapshot sweep, no hydration inside batches. Return dict symbol→snapshot (possibly empty)."""
    out = {}
    i = 0
    while i < len(symbols):
        batch = symbols[i:i+chunk]
        js = _snapshots_request(batch, max_retries=4, timeout_s=90)
        diag["snapshot_batches"] += 1
        snaps = (js or {}).get("snapshots") or {}
        for s in batch:
            snap = snaps.get(s, {}) or {}
            out[s] = snap
            if any(k in snap and bool(snap.get(k)) for k in ("latestTrade","minuteBar","dailyBar","prevDailyBar","latestQuote")):
                diag["symbols_snap_nonempty"] += 1
            else:
                diag["symbols_snap_empty"] += 1
        i += len(batch); time.sleep(0.12)
    return out

# pickers (with source counters)
def pick_price(snap):
    lt  = snap.get("latestTrade") or {}
    mb  = snap.get("minuteBar")   or {}
    db  = snap.get("dailyBar")    or {}
    pdB = snap.get("prevDailyBar")or {}
    qt  = snap.get("latestQuote") or {}
    for cand, tag in ((lt.get("p"),"trade"), (mb.get("c"),"minbar"),
                      (db.get("c"),"daybar"), (pdB.get("c"),"prevday")):
        v = safe_float(cand)
        if v: src_counts["price"][tag] += 1; return v
    bp = safe_float(qt.get("bp")); ap = safe_float(qt.get("ap"))
    if bp and ap: src_counts["price"]["quote_mid"] += 1; return (bp+ap)/2.0
    if ap:        src_counts["price"]["quote_mid"] += 1; return ap
    if bp:        src_counts["price"]["quote_mid"] += 1; return bp
    src_counts["price"]["none"] += 1; return None

def pick_active_volume(snap, rth: bool):
    mb = snap.get("minuteBar") or {}
    db = snap.get("dailyBar") or {}
    pdB = snap.get("prevDailyBar") or {}
    mv = safe_float(mb.get("v")) or 0.0
    dv = safe_float(db.get("v")) or 0.0
    pv = safe_float(pdB.get("v")) or 0.0
    if rth and mv > 0: src_counts["vol"]["minute"] += 1; return mv, "minute"
    if dv > 0:         src_counts["vol"]["daily"]  += 1; return dv, "daily"
    src_counts["vol"]["prev_daily"] += 1; return pv, "prev_daily"

# ────────────────────────────────────────────────────────────────────────────────
# Google Sheets helpers (overwrite tabs)
# ────────────────────────────────────────────────────────────────────────────────
def get_gc():
    import gspread
    from google.oauth2.service_account import Credentials
    sa_info = parse_sa(SA_RAW)
    creds = Credentials.from_service_account_info(sa_info,
        scopes=["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"])
    gc = gspread.authorize(creds)
    return gc.open_by_key(GOOGLE_SHEET_ID)

def drop_and_create(sh, title, rows=200, cols=40):
    try:
        titles = {w.title: w for w in sh.worksheets()}
        if title in titles:
            sh.del_worksheet(titles[title]); time.sleep(0.3)
        ws = sh.add_worksheet(title=title, rows=rows, cols=cols)
        return ws
    except Exception:
        try:
            ws = sh.worksheet(title)
        except Exception:
            ws = sh.add_worksheet(title=title, rows=rows, cols=cols)
        ws.batch_clear(["A1:ZZ100000"])
        return ws

def write_frame_to_ws(sh, title, df: pd.DataFrame):
    ws = drop_and_create(sh, title, rows=max(200, len(df)+10), cols=max(10, len(df.columns)+2))
    values = [list(df.columns)] + df.fillna("").astype(str).values.tolist()
    ws.update("A1", values, value_input_option="RAW")

# ────────────────────────────────────────────────────────────────────────────────
# Indicators (signals)
# ────────────────────────────────────────────────────────────────────────────────
def ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

def macd(series: pd.Series):
    ema12 = ema(series, 12)
    ema26 = ema(series, 26)
    line  = ema12 - ema26
    signal= line.ewm(span=9, adjust=False, min_periods=9).mean()
    hist  = line - signal
    return line, signal, hist

# ────────────────────────────────────────────────────────────────────────────────
# Stage-A: ActiveNow (snapshots only; one small global hydration at the end)
# ────────────────────────────────────────────────────────────────────────────────
def stage_a_activenow():
    # 1) sweep snapshots once
    symbols = fetch_active_symbols(UNIVERSE_CAP)
    snaps   = fetch_snapshots_multi(symbols, chunk=SNAPSHOT_CHUNK)
    rth     = is_market_open_now()

    # 2) first pass (snapshots only)
    rows, price_drops = [], 0
    empty_symbols = []
    for s in symbols:
        snap = snaps.get(s) or {}
        p = pick_price(snap)
        if p is None:
            empty_symbols.append(s)
            continue
        if not (PRICE_MIN <= p <= PRICE_MAX):
            price_drops += 1; continue
        active_vol, vol_src = pick_active_volume(snap, rth)
        db = snap.get("dailyBar") or {}; pdB = snap.get("prevDailyBar") or {}
        prev_high  = pdB.get("h"); today_high = db.get("h")
        flag_pdh_break = bool(prev_high is not None and p > float(prev_high))
        flag_day_high_prox = False
        if today_high is not None and safe_float(today_high):
            try: flag_day_high_prox = p >= float(today_high) * (1.0 - float(DHP_DELTA))
            except Exception: flag_day_high_prox = False
        mb = snap.get("minuteBar") or {}
        rows.append({
            "symbol": s,
            "price": p,
            "minute_vol": float(safe_float(mb.get("v")) or 0.0),
            "daily_vol":  float(safe_float(db.get("v")) or 0.0),
            "active_vol": float(active_vol),
            "vol_source": vol_src,
            "flag_pdh_break": flag_pdh_break,
            "flag_day_high_proximity": flag_day_high_prox
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "symbol","price","minute_vol","daily_vol","active_vol","vol_source",
        "flag_pdh_break","flag_day_high_proximity"
    ])

    # 3) If not enough rows to cover TOP_K_ACT and hydration is allowed, hydrate up to MAX_HYDRATE empties with 1-Day bars
    needed = max(0, TOP_K_ACT - len(df))
    if needed > 0 and MAX_HYDRATE > 0 and empty_symbols:
        hydrate_list = empty_symbols[:min(MAX_HYDRATE, needed*3)]  # hydrate a small multiple of what's needed
        bars1 = _bars_request(hydrate_list, "1Day", 1, 80)
        bars2 = _bars_request(hydrate_list, "1Day", 2, 80)
        for s in hydrate_list:
            tbar = bars1.get(s); pbar = bars2.get(s)
            if tbar is None and pbar is None: continue
            # synthesize a pseudo-snapshot from bars
            db  = {}; pdB = {}
            if tbar is not None and not tbar.empty:
                last = tbar.iloc[-1]
                db = {"c": float(last["close"]), "h": float(last["high"]), "v": float(last["volume"])}
            if pbar is not None and len(pbar) >= 2:
                prev = pbar.iloc[-2]
                pdB = {"c": float(prev["close"]), "h": float(prev["high"]), "v": float(prev["volume"])}
            # price fallback: db.c -> pdB.c
            p = safe_float(db.get("c")) or safe_float(pdB.get("c"))
            if p is None or not (PRICE_MIN <= p <= PRICE_MAX): continue
            dv = safe_float(db.get("v")) or 0.0
            pv = safe_float(pdB.get("v")) or 0.0
            active_vol, vol_src = (dv, "daily") if dv > 0 else (pv, "prev_daily")
            prev_high  = pdB.get("h")
            today_high = db.get("h")
            flag_pdh_break = bool(prev_high is not None and p > float(prev_high))
            flag_day_high_prox = False
            if today_high is not None and safe_float(today_high):
                try: flag_day_high_prox = p >= float(today_high) * (1.0 - float(DHP_DELTA))
                except Exception: flag_day_high_prox = False
            df.loc[len(df)] = [s, p, 0.0, float(dv or 0.0), float(active_vol), vol_src, flag_pdh_break, flag_day_high_prox]
            diag["hydrated_symbols"] += 1

    if df.empty:
        return df, symbols, {"price_drops": price_drops, "session": session_status()}

    df = df.sort_values(["active_vol","daily_vol","minute_vol"], ascending=[False, False, False]).reset_index(drop=True)
    df_out = df.head(TOP_K_ACT).copy()
    df_out.insert(0, "as_of_et", now_et_str())
    df_out.insert(1, "session", session_status())
    return df_out, symbols, {"price_drops": price_drops, "session": session_status()}

# ────────────────────────────────────────────────────────────────────────────────
# Signals (optional) — on Top-K
# ────────────────────────────────────────────────────────────────────────────────
def ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

def macd(series: pd.Series):
    ema12 = ema(series, 12); ema26 = ema(series, 26)
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False, min_periods=9).mean()
    hist = line - signal
    return line, signal, hist

def signals_h1_macd(top_syms, h1_limit, chunk):
    if not top_syms: return pd.DataFrame(columns=["as_of_et","symbol","macd_line","macd_signal","macd_hist","note"])
    bars = _bars_request(top_syms, "1Hour", h1_limit, chunk)
    rows = []
    for s, df in bars.items():
        if df is None or df.empty or len(df) < 35:  # need enough bars
            continue
        closes = df["close"].astype(float)
        line, sig, hist = macd(closes)
        if line.isna().iloc[-1] or sig.isna().iloc[-1]: continue
        mline, msignal, mhist = float(line.iloc[-1]), float(sig.iloc[-1]), float(hist.iloc[-1])
        if (mline > 0.0) and (msignal < 0.0):
            rows.append({"as_of_et": now_et_str(), "symbol": s,
                         "macd_line": round(mline, 6), "macd_signal": round(msignal, 6),
                         "macd_hist": round(mhist, 6), "note": "H1_MACD_zeroline_watch"})
    return pd.DataFrame(rows)

def signals_d1_ema200(top_syms, d1_limit, chunk):
    if not top_syms: return pd.DataFrame(columns=["as_of_et","symbol","close","ema200","note"])
    bars = _bars_request(top_syms, "1Day", d1_limit, chunk)
    rows = []
    for s, df in bars.items():
        if df is None or df.empty or len(df) < 205: continue
        closes = df["close"].astype(float)
        e200 = closes.ewm(span=200, adjust=False, min_periods=200).mean()
        if e200.isna().iloc[-1] or e200.isna().iloc[-2]: continue
        c0, e0 = float(closes.iloc[-1]), float(e200.iloc[-1])
        c1, e1 = float(closes.iloc[-2]), float(e200.iloc[-2])
        if (c0 > e0) and (c1 <= e1):
            rows.append({"as_of_et": now_et_str(), "symbol": s, "close": round(c0, 4),
                         "ema200": round(e0, 4), "note": "EMA200_cross_up"})
    return pd.DataFrame(rows)

# ────────────────────────────────────────────────────────────────────────────────
# Context (SPY/VIXY/GLD/USO/SLV + sectors, 1D/5D)
# ────────────────────────────────────────────────────────────────────────────────
SECTORS = ["XLF","XLK","XLY","XLP","XLV","XLE","XLI","XLU","XLRE","XLB","XLC"]
CTX_TICKERS = ["SPY","VIXY","GLD","USO","SLV"] + SECTORS

def pct(series: pd.Series, idx_a, idx_b):
    try:
        a, b = float(series.iloc[idx_a]), float(series.iloc[idx_b])
        if b != 0: return 100.0 * (a-b)/b
    except Exception:
        pass
    return None

def build_context_df():
    bars = _bars_request(CTX_TICKERS, "1Day", 6, 60)  # small & cheap
    rows = [{"key":"as_of_et","value":now_et_str()},
            {"key":"session","value":session_status()}]
    sector_1d, sector_5d = {}, {}
    for s in CTX_TICKERS:
        df = bars.get(s)
        if df is None or df.empty: continue
        c = df["close"].astype(float).reset_index(drop=True)
        p1 = pct(c, -1, -2)
        p5 = pct(c, -1, -6 if len(c) >= 6 else 0)
        if p1 is not None: rows.append({"key":f"{s}_1d_pct","value":f"{p1:+.2f}%"})
        if p5 is not None: rows.append({"key":f"{s}_5d_pct","value":f"{p5:+.2f}%"})
        if s in SECTORS:
            if p1 is not None: sector_1d[s] = p1
            if p5 is not None: sector_5d[s] = p5
    if sector_1d:
        top1 = sorted(sector_1d.items(), key=lambda x: x[1], reverse=True)[:3]
        rows.append({"key":"sectors_top3_1d","value":", ".join([f'{k}({v:+.2f}%)' for k,v in top1])})
    if sector_5d:
        top5 = sorted(sector_5d.items(), key=lambda x: x[1], reverse=True)[:3]
        rows.append({"key":"sectors_top3_5d","value":", ".join([f'{k}({v:+.2f}%)' for k,v in top5])})
    # Simple risk proxy (uses VIXY 1D% as stress proxy)
    vixy_1d = None
    for r in rows:
        if r["key"] == "VIXY_1d_pct":
            try: vixy_1d = float(r["value"].replace("%",""))
            except Exception: vixy_1d = None
    def_avg = np.nanmean([sector_1d.get(x, np.nan) for x in ("XLV","XLP","XLU")])
    cyc_avg = np.nanmean([sector_1d.get(x, np.nan) for x in ("XLK","XLY","XLF")])
    risk_mode = "tight" if (vixy_1d is not None and vixy_1d > 3.0 and not np.isnan(def_avg) and not np.isnan(cyc_avg) and def_avg > cyc_avg) else "normal"
    rows.append({"key":"risk_mode","value":risk_mode})
    return pd.DataFrame(rows, columns=["key","value"])

# ────────────────────────────────────────────────────────────────────────────────
# Run
# ────────────────────────────────────────────────────────────────────────────────
left, right = st.columns([2,1])

with st.spinner("Stage-A: scanning snapshots…"):
    df_active, all_syms, a_diag = stage_a_activenow()

with left:
    st.subheader(f"ActiveNow — Top {len(df_active) if not df_active.empty else 0} by volume (Price ${PRICE_MIN:.0f}–${PRICE_MAX:.0f})")
    if df_active.empty:
        st.warning("No matches under current settings (data may be limited off-hours or by plan).")
    else:
        fmt = {"price":"{:.2f}","minute_vol":"{:.0f}","daily_vol":"{:.0f}","active_vol":"{:.0f}"}
        st.dataframe(df_active.style.format(fmt), use_container_width=True)

with right:
    st.subheader("Run Summary")
    st.write(f"**As of (ET):** {now_et_str()}")
    st.write(f"**Session:** {a_diag.get('session')}")
    st.write(f"**Universe cap:** {UNIVERSE_CAP:,}")
    st.write(f"**Scanned:** {len(all_syms):,}")
    st.write(f"**Snapshot non-empty:** {diag['symbols_snap_nonempty']:,}")
    st.write(f"**Snapshot empty:** {diag['symbols_snap_empty']:,}")
    st.write(f"**Hydrated (1-Day bars):** {diag['hydrated_symbols']:,}")
    st.write(f"**Price-band drops:** {a_diag.get('price_drops',0)}")
    st.caption("Minute volume used only during **regular** session; otherwise daily/prev-day. Hydration capped to avoid 429s.")

# Optional Stage-B signals (Top-K)
if DO_SIGNALS and not df_active.empty:
    sig_syms = df_active["symbol"].tolist()[:TOP_K_SIG]
    with st.spinner(f"Signals — H1 MACD on {len(sig_syms)} symbols…"):
        df_h1 = signals_h1_macd(sig_syms, H1_LIMIT, BARS_CHUNK)
    with st.spinner(f"Signals — D1 EMA200 on {len(sig_syms)} symbols…"):
        df_d1 = signals_d1_ema200(sig_syms, D1_LIMIT, BARS_CHUNK)
else:
    df_h1 = pd.DataFrame(columns=["as_of_et","symbol","macd_line","macd_signal","macd_hist","note"])
    df_d1 = pd.DataFrame(columns=["as_of_et","symbol","close","ema200","note"])

# Context
with st.spinner("Context (SPY/VIXY/GLD/USO/SLV + sectors)…"):
    df_ctx = build_context_df()

st.markdown("### Signals — Hourly (H1)")
if df_h1.empty: st.info("Signals disabled or no H1 MACD zero-line watch right now.")
else: st.dataframe(df_h1, use_container_width=True)

st.markdown("### Signals — Daily (D1)")
if df_d1.empty: st.info("Signals disabled or no EMA200 cross-up right now.")
else: st.dataframe(df_d1, use_container_width=True)

st.markdown("### Context")
st.dataframe(df_ctx, use_container_width=True)

# Write Sheets (overwrite each run)
try:
    sh = get_gc()
    write_frame_to_ws(sh, TAB_ACTIVENOW, df_active if not df_active.empty else pd.DataFrame(columns=[
        "as_of_et","session","symbol","price","minute_vol","daily_vol","active_vol","vol_source",
        "flag_pdh_break","flag_day_high_proximity"
    ]))
    write_frame_to_ws(sh, TAB_SIG_H1, df_h1)
    write_frame_to_ws(sh, TAB_SIG_D1, df_d1)
    write_frame_to_ws(sh, TAB_CONTEXT, df_ctx)
    st.success("Sheets updated: ActiveNow, SignalsH1, SignalsD1, Context.")
except Exception as e:
    st.error(f"Sheets write error: {e}")

# Diagnostics
with st.expander("🛠 Diagnostics", expanded=False):
    if _last_data_error:
        for k, v in _last_data_error.items():
            st.write(f"**{k}** → {v}")
    st.write("**Snapshot batches**:", diag["snapshot_batches"])
    st.write("**Hydrated symbols (global)**:", diag["hydrated_symbols"])
    st.write("**Price source counts**:", src_counts["price"])
    st.write("**Volume source counts**:", src_counts["vol"])
