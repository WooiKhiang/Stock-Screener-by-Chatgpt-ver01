# Market Scanner — ActiveNow + Signals (Alpaca → Google Sheets)
# Stage-A: snapshots-only volume screen (no lookback)
# Stage-B (Top-K only): MACD H1 watch, EMA200 D1 cross-up
# Context: SPY/VIXY/GLD/USO/SLV + sector ETFs (1D/5D, rankings, risk_mode)
# Tabs (overwrite each run): ActiveNow, SignalsH1, SignalsD1, Context

import os, json, time, random, datetime as dt
import numpy as np
import pandas as pd
import pytz, requests, streamlit as st
from streamlit_autorefresh import st_autorefresh

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
FEED          = "iex"   # use IEX to avoid SIP entitlement issues

SA_RAW = st.secrets.get("GCP_SERVICE_ACCOUNT") or os.getenv("GCP_SERVICE_ACCOUNT")

ET = pytz.timezone("America/New_York")
HEADERS = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}

# ────────────────────────────────────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Market Scanner — ActiveNow + Signals", layout="wide")
st.title("⚡ Market Scanner — ActiveNow + Signals (Alpaca → Google Sheets)")

# Auto-run hourly
if st.sidebar.checkbox("Auto-run hourly", value=True, key="autorun"):
    st_autorefresh(interval=60*60*1000, key="hourly_refresh")

# Universe & filters
st.sidebar.markdown("### Universe")
UNIVERSE_CAP = st.sidebar.slider("Symbols to scan (cap)", 1000, 12000, 8000, 500)

st.sidebar.markdown("### Stage-A Filters")
PRICE_MIN = st.sidebar.number_input("Min price ($)", value=5.0, step=0.5)
PRICE_MAX = st.sidebar.number_input("Max price ($)", value=100.0, step=1.0)
TOP_K_ACT = st.sidebar.slider("Top K to keep (by volume)", 200, 3000, 1200, 100)
DHP_DELTA = st.sidebar.slider("Day-high proximity δ", 0.001, 0.02, 0.005, 0.001,
                              help="Flag if last ≥ today's high × (1−δ). Requires today's high to exist.")

st.sidebar.markdown("### Stage-B Bars (Top-K only)")
TOP_K_SIG  = st.sidebar.slider("Symbols to compute signals on (from Top-K ActiveNow)", 50, 1000, 200, 50)
H1_LIMIT   = st.sidebar.slider("Hourly bars (for MACD)", 40, 160, 80, 20)
D1_LIMIT   = st.sidebar.slider("Daily bars (for EMA200)", 210, 280, 220, 10)

st.sidebar.markdown("### Network")
SNAPSHOT_CHUNK = st.sidebar.slider("Snapshot chunk size", 80, 300, 120, 20)
BARS_CHUNK     = st.sidebar.slider("Bars chunk size (Top-K only)", 40, 200, 120, 20)

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
_diag = {"snap_batches":0, "snap_empty_batches":0, "hydrated_symbols":0}
_src_counts = {"price": {"trade":0,"minbar":0,"daybar":0,"prevday":0,"quote_mid":0,"none":0},
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

def _snapshots_request(symbols_batch, max_retries=5, timeout_s=90):
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
        time.sleep(backoff + random.random()*0.5)
        backoff = min(backoff*1.8, 8.0)
    _last_data_error["snapshots"] = (408, "retry timeout"); return None

def _bars_request(symbols, timeframe, limit, chunk):
    """Lightweight bars pull — used for Top-K signals and for hydrating empty snapshots."""
    out = {}
    url = f"{ALPACA_DATA}/v2/stocks/bars"
    end = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    # generous window just to satisfy API; limit caps rows anyway
    start = end - (dt.timedelta(days=90) if timeframe == "1Hour" else dt.timedelta(days=400))
    start_iso = start.isoformat().replace("+00:00","Z"); end_iso = end.isoformat().replace("+00:00","Z")
    i = 0
    while i < len(symbols):
        batch = symbols[i:i+chunk]
        params = {"symbols": ",".join(batch), "timeframe": timeframe, "limit": limit,
                  "adjustment":"raw", "feed": FEED, "start": start_iso, "end": end_iso}
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
        i += len(batch); time.sleep(0.1)
    return out

def fetch_snapshots_multi(symbols, chunk=120):
    out = {}
    i = 0
    while i < len(symbols):
        batch = symbols[i:i+chunk]
        js = _snapshots_request(batch, max_retries=5, timeout_s=90)
        _diag["snap_batches"] += 1
        snaps = (js or {}).get("snapshots") or {}
        # store snaps
        for s in batch:
            out[s] = snaps.get(s, {}) or {}
        # hydrate empties with 1–2 daily bars
        needs = []
        for s in batch:
            snap = out.get(s, {})
            has_any = False
            for key in ("latestTrade","minuteBar","dailyBar","prevDailyBar","latestQuote"):
                if key in snap and bool(snap.get(key)): has_any = True; break
            if not has_any: needs.append(s)
        if needs:
            _diag["snap_empty_batches"] += 1
            bars1 = _bars_request(needs, "1Day", 1, max(40, min(80, len(needs))))
            bars2 = _bars_request(needs, "1Day", 2, max(40, min(80, len(needs))))
            for s in needs:
                snap = out.get(s, {}) or {}
                tbar = bars1.get(s); pbar = bars2.get(s)
                if tbar is not None and not tbar.empty:
                    last = tbar.iloc[-1]
                    snap["dailyBar"] = {"c": float(last["close"]), "h": float(last["high"]), "v": float(last["volume"])}
                if pbar is not None and len(pbar) >= 2:
                    prev = pbar.iloc[-2]
                    snap["prevDailyBar"] = {"c": float(prev["close"]), "h": float(prev["high"]), "v": float(prev["volume"])}
                if snap: out[s] = snap; _diag["hydrated_symbols"] += 1
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
        if v:
            _src_counts["price"][tag] += 1; return v
    bp = safe_float(qt.get("bp")); ap = safe_float(qt.get("ap"))
    if bp and ap: _src_counts["price"]["quote_mid"] += 1; return (bp+ap)/2.0
    if ap:        _src_counts["price"]["quote_mid"] += 1; return ap
    if bp:        _src_counts["price"]["quote_mid"] += 1; return bp
    _src_counts["price"]["none"] += 1; return None

def pick_active_volume(snap, rth: bool):
    mb = snap.get("minuteBar") or {}
    db = snap.get("dailyBar") or {}
    pdB = snap.get("prevDailyBar") or {}
    mv = safe_float(mb.get("v")) or 0.0
    dv = safe_float(db.get("v")) or 0.0
    pv = safe_float(pdB.get("v")) or 0.0
    if rth and mv > 0: _src_counts["vol"]["minute"] += 1; return mv, "minute"
    if dv > 0:         _src_counts["vol"]["daily"]  += 1; return dv, "daily"
    _src_counts["vol"]["prev_daily"] += 1; return pv, "prev_daily"

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

def cleanup_conflict_tabs(sh, base_title: str):
    try:
        for ws in sh.worksheets():
            if ws.title.startswith(f"{base_title}_conflict"):
                try: sh.del_worksheet(ws)
                except Exception: pass
    except Exception:
        pass

def drop_and_create(sh, title, rows=200, cols=40):
    try:
        titles = {w.title: w for w in sh.worksheets()}
        if title in titles:
            ws = titles[title]
            if len(titles) == 1:
                tmp = sh.add_worksheet(title="__tmp__", rows=1, cols=1)
                sh.del_worksheet(ws)
                ws = sh.add_worksheet(title=title, rows=rows, cols=cols)
                sh.del_worksheet(tmp)
            else:
                sh.del_worksheet(ws); time.sleep(0.4)
                ws = sh.add_worksheet(title=title, rows=rows, cols=cols)
        else:
            ws = sh.add_worksheet(title=title, rows=rows, cols=cols)
        cleanup_conflict_tabs(sh, title); return ws
    except Exception:
        try: ws = sh.worksheet(title)
        except Exception: ws = sh.add_worksheet(title=title, rows=rows, cols=cols)
        ws.batch_clear(["A1:ZZ100000"]); cleanup_conflict_tabs(sh, title); return ws

def write_frame_to_ws(sh, title, df: pd.DataFrame):
    ws = drop_and_create(sh, title, rows=max(200, len(df)+10), cols=max(10, len(df.columns)+2))
    values = [list(df.columns)] + df.fillna("").astype(str).values.tolist()
    ws.update("A1", values, value_input_option="RAW")

# ────────────────────────────────────────────────────────────────────────────────
# Indicators (for Top-K only)
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
# Stage-A: ActiveNow (no lookback)
# ────────────────────────────────────────────────────────────────────────────────
def stage_a_activenow():
    symbols = fetch_active_symbols(UNIVERSE_CAP)
    snaps = fetch_snapshots_multi(symbols, chunk=SNAPSHOT_CHUNK)
    rth = is_market_open_now()

    rows = []; price_drops = 0
    for s in symbols:
        snap = snaps.get(s) or {}
        p = pick_price(snap)
        if p is None or not (PRICE_MIN <= p <= PRICE_MAX):
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

    if not rows:
        return pd.DataFrame(), symbols, {"price_drops": price_drops, "session": session_status()}

    df = pd.DataFrame(rows).sort_values(["active_vol","daily_vol","minute_vol"], ascending=[False, False, False]).reset_index(drop=True)
    df_out = df.head(TOP_K_ACT).copy()
    df_out.insert(0, "as_of_et", now_et_str())
    df_out.insert(1, "session", session_status())
    return df_out, symbols, {"price_drops": price_drops, "session": session_status()}

# ────────────────────────────────────────────────────────────────────────────────
# Stage-B: Signals on Top-K
# ────────────────────────────────────────────────────────────────────────────────
def signals_h1_macd(top_syms):
    if not top_syms: return pd.DataFrame(columns=["as_of_et","symbol","macd_line","macd_signal","macd_hist","note"])
    bars = _bars_request(top_syms, "1Hour", H1_LIMIT, BARS_CHUNK)
    rows = []
    for s, df in bars.items():
        if df is None or df.empty or len(df) < 35:  # need enough bars for MACD
            continue
        closes = df["close"].astype(float)
        line, sig, hist = macd(closes)
        if line.isna().iloc[-1] or sig.isna().iloc[-1]:  # incomplete
            continue
        mline = float(line.iloc[-1]); msignal = float(sig.iloc[-1]); mhist = float(hist.iloc[-1])
        if (mline > 0.0) and (msignal < 0.0):
            rows.append({"as_of_et": now_et_str(), "symbol": s,
                         "macd_line": round(mline, 6), "macd_signal": round(msignal, 6),
                         "macd_hist": round(mhist, 6),
                         "note": "H1_MACD_zeroline_watch"})
    return pd.DataFrame(rows)

def signals_d1_ema200(top_syms):
    if not top_syms: return pd.DataFrame(columns=["as_of_et","symbol","close","ema200","note"])
    bars = _bars_request(top_syms, "1Day", D1_LIMIT, BARS_CHUNK)
    rows = []
    for s, df in bars.items():
        if df is None or df.empty or len(df) < 205:
            continue
        closes = df["close"].astype(float)
        ema200 = ema(closes, 200)
        if ema200.isna().iloc[-1] or ema(closes,200).isna().iloc[-2]:
            continue
        c0, e0 = float(closes.iloc[-1]), float(ema200.iloc[-1])
        c1, e1 = float(closes.iloc[-2]), float(ema200.iloc[-2])
        if (c0 > e0) and (c1 <= e1):
            rows.append({"as_of_et": now_et_str(), "symbol": s, "close": round(c0, 4),
                         "ema200": round(e0, 4), "note": "EMA200_cross_up"})
    return pd.DataFrame(rows)

# ────────────────────────────────────────────────────────────────────────────────
# Context (SPY/VIXY/GLD/USO/SLV + sector ETFs)
# ────────────────────────────────────────────────────────────────────────────────
SECTORS = ["XLF","XLK","XLY","XLP","XLV","XLE","XLI","XLU","XLRE","XLB","XLC"]  # 11 GICS sectors (XLC included)
CTX_TICKERS = ["SPY","VIXY","GLD","USO","SLV"] + SECTORS

def pct(series, idx_a, idx_b):
    a, b = safe_float(series.iloc[idx_a]), safe_float(series.iloc[idx_b])
    if a and b: return 100.0 * (a-b)/b
    return None

def build_context_df():
    bars = _bars_request(CTX_TICKERS, "1Day", 6, 60)
    rows = [{"key":"as_of_et", "value": now_et_str()},
            {"key":"session",  "value": session_status()}]

    sector_1d = {}
    sector_5d = {}
    for s in CTX_TICKERS:
        df = bars.get(s)
        if df is None or df.empty: continue
        c = df["close"].astype(float).reset_index(drop=True)
        # 1D % = last vs prev
        p1 = pct(c, -1, -2)
        # 5D % = last vs 5 bars ago (or closest available)
        p5 = pct(c, -1, -6 if len(c) >= 6 else 0)
        if p1 is not None: rows.append({"key": f"{s}_1d_pct", "value": f"{p1:+.2f}%"})
        if p5 is not None: rows.append({"key": f"{s}_5d_pct", "value": f"{p5:+.2f}%"})
        if s in SECTORS:
            if p1 is not None: sector_1d[s] = p1
            if p5 is not None: sector_5d[s] = p5

    # Sector rankings
    if sector_1d:
        top1 = sorted(sector_1d.items(), key=lambda x: x[1], reverse=True)[:3]
        rows.append({"key":"sectors_top3_1d", "value": ", ".join([f"{k}({v:+.2f}%)" for k,v in top1])})
    if sector_5d:
        top5 = sorted(sector_5d.items(), key=lambda x: x[1], reverse=True)[:3]
        rows.append({"key":"sectors_top3_5d", "value": ", ".join([f"{k}({v:+.2f}%)" for k,v in top5])})

    # Risk mode (simple rule): VIXY > 20 and defensives outperform cyclicals on 1D
    def_avg = np.nanmean([sector_1d.get(x, np.nan) for x in ("XLV","XLP","XLU")])
    cyc_avg = np.nanmean([sector_1d.get(x, np.nan) for x in ("XLK","XLY","XLF")])
    vixy_1d = None
    for r in rows:
        if r["key"] == "VIXY_1d_pct":
            # not needed actually; rule uses level, but we don't have level—use proxy: if VIXY 1D% > +3% mark as high
            try:
                vixy_1d = float(r["value"].replace("%",""))
            except Exception:
                vixy_1d = None
    risk_mode = "normal"
    if (vixy_1d is not None and vixy_1d > 3.0) and (not np.isnan(def_avg) and not np.isnan(cyc_avg) and def_avg > cyc_avg):
        risk_mode = "tight"
    rows.append({"key":"risk_mode", "value": risk_mode})

    return pd.DataFrame(rows, columns=["key","value"])

# ────────────────────────────────────────────────────────────────────────────────
# Run pipeline
# ────────────────────────────────────────────────────────────────────────────────
left, right = st.columns([2,1])

with st.spinner("Stage-A: scanning snapshots (no lookback)…"):
    df_active, all_syms, a_diag = stage_a_activenow()

with left:
    st.subheader(f"ActiveNow — Top {len(df_active) if not df_active.empty else 0} by volume (Price ${PRICE_MIN:.0f}–${PRICE_MAX:.0f})")
    if df_active.empty:
        st.warning("No matches under current settings.")
    else:
        fmt = {"price":"{:.2f}","minute_vol":"{:.0f}","daily_vol":"{:.0f}","active_vol":"{:.0f}"}
        st.dataframe(df_active.style.format(fmt), use_container_width=True)

with right:
    st.subheader("Run Summary (Stage-A)")
    st.write(f"**As of (ET):** {now_et_str()}")
    st.write(f"**Session:** {a_diag.get('session')}")
    st.write(f"**Universe cap:** {UNIVERSE_CAP:,}")
    st.write(f"**Universe scanned:** {len(all_syms):,}")
    st.write(f"**Price-band drops:** {a_diag.get('price_drops',0)}")
    st.caption("Minute volume used only during **regular** session; otherwise daily/prev-day volume.")

# Stage-B (signals) on Top-K from ActiveNow
sig_syms = df_active["symbol"].tolist()[:TOP_K_SIG] if not df_active.empty else []

with st.spinner(f"Stage-B: H1 MACD on {len(sig_syms)} symbols…"):
    df_h1 = signals_h1_macd(sig_syms)
with st.spinner(f"Stage-B: D1 EMA200 cross on {len(sig_syms)} symbols…"):
    df_d1 = signals_d1_ema200(sig_syms)

# Context (always)
with st.spinner("Refreshing Context…"):
    df_ctx = build_context_df()

# Display signals
st.markdown("### Signals — Hourly (H1)")
if df_h1.empty:
    st.info("No H1 MACD zero-line watch signals right now.")
else:
    st.dataframe(df_h1, use_container_width=True)

st.markdown("### Signals — Daily (D1)")
if df_d1.empty:
    st.info("No EMA200 cross-up signals right now.")
else:
    st.dataframe(df_d1, use_container_width=True)

st.markdown("### Context (SPY / VIXY / GLD / USO / SLV + sectors)")
st.dataframe(df_ctx, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────────
# Write to Google Sheets (overwrite)
# ────────────────────────────────────────────────────────────────────────────────
try:
    sh = get_gc()
    # ActiveNow
    if df_active.empty:
        write_frame_to_ws(sh, TAB_ACTIVENOW, pd.DataFrame(columns=[
            "as_of_et","session","symbol","price","minute_vol","daily_vol","active_vol","vol_source",
            "flag_pdh_break","flag_day_high_proximity"
        ]))
    else:
        write_frame_to_ws(sh, TAB_ACTIVENOW, df_active)

    # Signals
    write_frame_to_ws(sh, TAB_SIG_H1, df_h1 if not df_h1.empty else pd.DataFrame(columns=["as_of_et","symbol","macd_line","macd_signal","macd_hist","note"]))
    write_frame_to_ws(sh, TAB_SIG_D1, df_d1 if not df_d1.empty else pd.DataFrame(columns=["as_of_et","symbol","close","ema200","note"]))

    # Context
    write_frame_to_ws(sh, TAB_CONTEXT, df_ctx)

    st.success("Sheets updated: ActiveNow, SignalsH1, SignalsD1, Context.")
except Exception as e:
    st.error(f"Sheets write error: {e}")

# ────────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ────────────────────────────────────────────────────────────────────────────────
with st.expander("🛠 Diagnostics", expanded=False):
    if _last_data_error:
        for k, v in _last_data_error.items():
            st.write(f"**{k}** → {v}")
    st.write("**Snapshot batches**:", _diag["snap_batches"])
    st.write("**Batches with empty snapshots (hydrated)**:", _diag["snap_empty_batches"])
    st.write("**Symbols hydrated via tiny daily bars**:", _diag["hydrated_symbols"])
    st.write("**Price source counts**:", _src_counts["price"])
    st.write("**Volume source counts**:", _src_counts["vol"])
