# Ultra-lean Market Scanner — Volume-only (no lookback) → Google Sheets
# Tabs (drop & recreate safely): ActiveNow, Context
# Data: Alpaca snapshots (IEX). No historical lookback; only a tiny 1–2 daily-bar fallback inside snapshots helper.
# Ranking: minute volume (if RTH) else daily volume; if missing pre/post → prev-day volume.
# Price fallback: latestTrade.p → minuteBar.c → dailyBar.c → prevDailyBar.c → latestQuote mid (bp/ap)
# Flags: PDH break, Day-high proximity (only when today's high exists)
# UI: Context panel shows SPY, GLD (gold), USO (oil), SLV (silver) prices + 1D %; also session label.

import os, json, time, random, datetime as dt
import numpy as np
import pandas as pd
import pytz, requests, streamlit as st

# =========================
# Config & Secrets
# =========================
GOOGLE_SHEET_ID = "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4"

TAB_ACTIVENOW = "ActiveNow"
TAB_CONTEXT   = "Context"

# Alpaca creds (env preferred; fallbacks supplied)
ALPACA_KEY    = os.getenv("ALPACA_KEY",    "PKIG445MPT704CN8P0R8")
ALPACA_SECRET = os.getenv("ALPACA_SECRET", "3GQdWcnbMo6V8uvhVa6BK6EbvnH4EHinlsU6uvj4")
ALPACA_BASE   = "https://paper-api.alpaca.markets/v2"
ALPACA_DATA   = "https://data.alpaca.markets"
FEED          = "iex"   # avoid SIP entitlement issues

# Google SA JSON in Streamlit secrets
SA_RAW = st.secrets.get("GCP_SERVICE_ACCOUNT") or os.getenv("GCP_SERVICE_ACCOUNT")

ET = pytz.timezone("America/New_York")
HEADERS = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}

# =========================
# UI
# =========================
st.set_page_config(page_title="Volume Scanner — ActiveNow", layout="wide")
st.title("⚡ Volume Scanner — ActiveNow (Snapshots only)")

autorun = st.sidebar.checkbox("Auto-run hourly", value=True)
try:
    autorefresh = getattr(st, "autorefresh", None) or getattr(st, "experimental_autorefresh", None)
    if autorun and autorefresh:
        autorefresh(interval=60*60*1000, key="auto_hr_scanner")
    elif autorun:
        st.markdown("<script>setTimeout(()=>window.location.reload(),3600000);</script>", unsafe_allow_html=True)
except Exception:
    pass

st.sidebar.markdown("### Universe")
UNIVERSE_CAP = st.sidebar.slider("Symbols to scan (cap)", 1000, 12000, 8000, 500)

st.sidebar.markdown("### Filters")
PRICE_MIN = st.sidebar.number_input("Min price ($)", value=5.0, step=0.5)
PRICE_MAX = st.sidebar.number_input("Max price ($)", value=100.0, step=1.0)
TOP_K     = st.sidebar.slider("Top K to keep (by volume)", 200, 3000, 1200, 100)

st.sidebar.markdown("### Flags (no lookback)")
DHP_DELTA = st.sidebar.slider("Day-high proximity δ", 0.001, 0.02, 0.005, 0.001,
                              help="Flag if last ≥ today's high × (1−δ). Only when today's high exists.")

st.sidebar.markdown("### Network")
SNAPSHOT_CHUNK = st.sidebar.slider("Snapshot chunk size (start)", 100, 600, 300, 50)

# =========================
# Helpers
# =========================
def now_et():
    return dt.datetime.now(ET)

def now_et_str():
    return now_et().strftime("%Y-%m-%d %H:%M:%S")

def session_status():
    n = now_et()
    if n.weekday() >= 5:
        return "closed"
    t = n.time()
    if dt.time(4,0) <= t < dt.time(9,30):
        return "pre"
    if dt.time(9,30) <= t <= dt.time(16,0):
        return "regular"
    if dt.time(16,0) < t <= dt.time(20,0):
        return "post"
    return "closed"

def is_market_open_now():
    return session_status() == "regular"

def parse_sa(raw_json: str) -> dict:
    if not raw_json: raise RuntimeError("Missing GCP_SERVICE_ACCOUNT in secrets/env.")
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        if "-----BEGIN PRIVATE KEY-----" in raw_json and "\\n" not in raw_json:
            start = raw_json.find("-----BEGIN PRIVATE KEY-----")
            end   = raw_json.find("-----END PRIVATE KEY-----", start) + len("-----END PRIVATE KEY-----")
            block = raw_json[start:end]
            raw_json = raw_json.replace(block, block.replace("\r\n","\n").replace("\n","\\n"))
        return json.loads(raw_json)

def safe_float(x):
    try:
        v = float(x)
        return v if (v is not None and v > 0) else None
    except Exception:
        return None

# =========================
# Alpaca I/O (robust snapshots + tiny daily fallback)
# =========================
_last_data_error = {}
_diag_counts = {"price_src": {"trade":0,"minbar":0,"daybar":0,"prevday":0,"quote_mid":0,"none":0},
                "vol_src": {"minute":0,"daily":0,"prev_daily":0}}

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
            if s.endswith((".U",".W","WS","W","R",".P","-P")):
                continue
            syms.append(s)
        if len(syms) >= cap: break
    return syms[:cap]

def _snapshots_request(symbols_batch, max_retries=5, timeout_s=120):
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
                _last_data_error["snapshots"] = (r.status_code, msg)
                return None
            else:
                try:
                    return r.json()
                except Exception:
                    _last_data_error["snapshots"] = (r.status_code, "JSON parse error")
                    return None
        time.sleep(backoff + random.random()*0.5)
        backoff = min(backoff*1.8, 8.0)
    _last_data_error["snapshots"] = (408, "retry timeout")
    return None

def fetch_bars_multi(symbols, timeframe="1Day", limit=2, chunk=150):
    # used ONLY as a lightweight fallback to synthesize daily & prevDaily
    out = {}
    end = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    start = end - dt.timedelta(days=10)
    start_iso = start.isoformat().replace("+00:00","Z")
    end_iso   = end.isoformat().replace("+00:00","Z")

    def _bars_request(params, max_retries=3):
        url = f"{ALPACA_DATA}/v2/stocks/bars"
        p = dict(params); p["feed"] = FEED
        backoff = 1.0
        for _ in range(max_retries):
            try:
                r = requests.get(url, headers=HEADERS, params=p, timeout=60)
            except requests.Timeout:
                pass
            else:
                if r.status_code in (429,500,502,503,504):
                    pass
                elif r.status_code >= 400:
                    try: msg = r.json().get("message", r.text[:300])
                    except Exception: msg = r.text[:300]
                    _last_data_error[f"bars_{p.get('timeframe')}"] = (r.status_code, msg)
                    return None
                else:
                    try:
                        return r.json()
                    except Exception:
                        _last_data_error[f"bars_{p.get('timeframe')}"] = (r.status_code, "JSON parse error")
                        return None
            time.sleep(backoff + random.random()*0.3)
            backoff = min(backoff*1.7, 6.0)
        _last_data_error[f"bars_{p.get('timeframe')}"] = (408, "retry timeout")
        return None

    def merge_json(js):
        bars = js.get("bars", [])
        if isinstance(bars, dict):
            for sym, arr in bars.items():
                if not arr:
                    out[sym] = pd.DataFrame(columns=["t","open","high","low","close","volume"]); continue
                df = pd.DataFrame(arr)
                df["t"] = pd.to_datetime(df["t"], utc=True)
                df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"}, inplace=True)
                out[sym] = df[["t","open","high","low","close","volume"]].sort_values("t").reset_index(drop=True)
        else:
            if bars:
                df = pd.DataFrame(bars)
                df["t"] = pd.to_datetime(df["t"], utc=True)
                df.rename(columns={"S":"symbol","o":"open","h":"high","l":"low","c":"close","v":"volume"}, inplace=True)
                for sym, grp in df.groupby("symbol"):
                    out[sym] = grp[["t","open","high","low","close","volume"]].sort_values("t").reset_index(drop=True)

    i = 0
    cur_chunk = max(60, chunk)
    while i < len(symbols):
        batch = symbols[i:i+cur_chunk]
        params = dict(timeframe=timeframe, symbols=",".join(batch), limit=limit,
                      start=start_iso, end=end_iso, adjustment="raw")
        js = _bars_request(params)
        if js is None and cur_chunk > 60:
            cur_chunk = max(40, cur_chunk // 2)
            continue
        if js is not None:
            merge_json(js)
        i += len(batch)
        time.sleep(0.1)
    return out

def fetch_snapshots_multi(symbols, chunk=300):
    out = {}
    i = 0
    cur_chunk = max(40, chunk)
    while i < len(symbols):
        batch = symbols[i:i+cur_chunk]
        js = _snapshots_request(batch, max_retries=5, timeout_s=120)
        if js is None:
            # shrink chunk and retry this slice
            if cur_chunk > 40:
                cur_chunk = max(20, cur_chunk // 2)
                continue
            # robust fallback using 1–2 daily bars
            try:
                bars_today = fetch_bars_multi(batch, timeframe="1Day", limit=1,  chunk=max(60, cur_chunk))
                bars_prev2 = fetch_bars_multi(batch, timeframe="1Day", limit=2,  chunk=max(60, cur_chunk))
                for s in batch:
                    snap = {}
                    tbar = bars_today.get(s)
                    pbar = bars_prev2.get(s)
                    if tbar is not None and not tbar.empty:
                        last = tbar.iloc[-1]
                        snap["dailyBar"] = {"c": float(last["close"]), "h": float(last["high"]), "v": float(last["volume"])}
                    if pbar is not None and len(pbar) >= 2:
                        prev = pbar.iloc[-2]
                        snap["prevDailyBar"] = {"c": float(prev["close"]), "h": float(prev["high"]), "v": float(prev["volume"])}
                    out[s] = snap
            except Exception:
                for s in batch: out[s] = {}
            i += cur_chunk
            time.sleep(0.15)
            continue

        snaps = js.get("snapshots") or {}
        for s in batch:
            out[s] = snaps.get(s, {}) or {}
        i += cur_chunk
        time.sleep(0.15)
        if cur_chunk < chunk:
            cur_chunk = min(chunk, int(cur_chunk * 1.25))
    return out

# ----- price/volume pickers (with source counters) -----
def pick_price(snap):
    lt  = snap.get("latestTrade") or {}
    mb  = snap.get("minuteBar") or {}
    db  = snap.get("dailyBar") or {}
    pdB = snap.get("prevDailyBar") or {}
    qt  = snap.get("latestQuote") or {}

    for cand, label in ((lt.get("p"),"trade"), (mb.get("c"),"minbar"),
                        (db.get("c"),"daybar"), (pdB.get("c"),"prevday")):
        val = safe_float(cand)
        if val:
            _diag_counts["price_src"][label] += 1
            return val

    # last resort: quote mid (or bid/ask if only one side)
    bp = safe_float(qt.get("bp"))
    ap = safe_float(qt.get("ap"))
    if bp and ap:
        _diag_counts["price_src"]["quote_mid"] += 1
        return (bp + ap) / 2.0
    if ap:
        _diag_counts["price_src"]["quote_mid"] += 1
        return ap
    if bp:
        _diag_counts["price_src"]["quote_mid"] += 1
        return bp

    _diag_counts["price_src"]["none"] += 1
    return None

def pick_active_volume(snap, rth: bool):
    mb = snap.get("minuteBar") or {}
    db = snap.get("dailyBar") or {}
    pdB = snap.get("prevDailyBar") or {}
    mv = safe_float(mb.get("v")) or 0.0
    dv = safe_float(db.get("v")) or 0.0
    pv = safe_float(pdB.get("v")) or 0.0

    if rth and mv > 0:
        _diag_counts["vol_src"]["minute"] += 1
        return mv, "minute"
    if dv > 0:
        _diag_counts["vol_src"]["daily"] += 1
        return dv, "daily"
    _diag_counts["vol_src"]["prev_daily"] += 1
    return pv, "prev_daily"

# =========================
# Google Sheets (conflict-safe)
# =========================
def get_gc():
    import gspread
    from google.oauth2.service_account import Credentials
    sa_info = parse_sa(SA_RAW)
    creds = Credentials.from_service_account_info(sa_info,
        scopes=["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"])
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(GOOGLE_SHEET_ID)
    return sh

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
                sh.del_worksheet(ws)
                time.sleep(0.4)
                ws = sh.add_worksheet(title=title, rows=rows, cols=cols)
        else:
            ws = sh.add_worksheet(title=title, rows=rows, cols=cols)
        cleanup_conflict_tabs(sh, title)
        return ws
    except Exception:
        try:
            ws = sh.worksheet(title)
        except Exception:
            ws = sh.add_worksheet(title=title, rows=rows, cols=cols)
        ws.batch_clear(["A1:ZZ100000"])
        cleanup_conflict_tabs(sh, title)
        return ws

def write_frame_to_ws(sh, title, df: pd.DataFrame):
    ws = drop_and_create(sh, title, rows=max(200, len(df)+10), cols=max(10, len(df.columns)+2))
    values = [list(df.columns)] + df.fillna("").astype(str).values.tolist()
    ws.update("A1", values, value_input_option="RAW")

# =========================
# Stage A — snapshots → ActiveNow (no lookback)
# =========================
def stage_a_activenow():
    symbols = fetch_active_symbols(UNIVERSE_CAP)
    snaps = fetch_snapshots_multi(symbols, chunk=SNAPSHOT_CHUNK)
    rth = is_market_open_now()

    rows = []
    price_drops = 0
    for s in symbols:
        snap = snaps.get(s) or {}
        p = pick_price(snap)
        if p is None or not (PRICE_MIN <= p <= PRICE_MAX):
            price_drops += 1
            continue

        active_vol, vol_src = pick_active_volume(snap, rth)
        db = snap.get("dailyBar") or {}
        pdB = snap.get("prevDailyBar") or {}

        prev_high  = pdB.get("h")
        today_high = db.get("h")
        flag_pdh_break = bool(prev_high is not None and safe_float(p) and (p > float(prev_high)))
        flag_day_high_prox = False
        if today_high is not None and safe_float(today_high):
            try:
                flag_day_high_prox = p >= float(today_high) * (1.0 - float(DHP_DELTA))
            except Exception:
                flag_day_high_prox = False

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
    df_out = df.head(TOP_K).copy()
    df_out.insert(0, "as_of_et", now_et_str())
    df_out.insert(1, "session", session_status())
    return df_out, symbols, {"price_drops": price_drops, "session": session_status()}

# =========================
# Context (tiny; no lookback) + SPY/GLD/USO/SLV
# =========================
CONTEXT_TICKERS = [
    "SPY",  # S&P 500 proxy
    "GLD",  # Gold
    "USO",  # Oil
    "SLV",  # Silver
]

def build_context_rows():
    snaps = fetch_snapshots_multi(CONTEXT_TICKERS, chunk=min(40, len(CONTEXT_TICKERS)))
    rows = [
        {"key": "as_of_et", "value": now_et_str()},
        {"key": "session",  "value": session_status()},
        {"key": "notes",    "value": "Minute volume is used only in regular session; otherwise daily/prev-day volume."}
    ]
    for s in CONTEXT_TICKERS:
        snap = snaps.get(s) or {}
        px = pick_price(snap)
        db = snap.get("dailyBar") or {}
        pdB = snap.get("prevDailyBar") or {}
        pct = ""
        try:
            c  = safe_float(db.get("c"))
            pc = safe_float(pdB.get("c"))
            if c and pc:
                pct = f"{100.0*(c-pc)/pc:+.2f}%"
        except Exception:
            pct = ""
        if px is not None:
            rows.append({"key": f"{s}_price", "value": f"{px:.2f}"})
        if pct:
            rows.append({"key": f"{s}_1d_pct", "value": pct})
    return pd.DataFrame(rows, columns=["key","value"])

# =========================
# Run
# =========================
left, right = st.columns([2,1])

with st.spinner("Scanning snapshots (no lookback)…"):
    df_active, all_syms, a_diag = stage_a_activenow()

with left:
    st.subheader(f"ActiveNow — Top {len(df_active) if not df_active.empty else 0} by volume (Price ${PRICE_MIN:.0f}–${PRICE_MAX:.0f})")
    if df_active.empty:
        st.warning("No matches under current settings.")
    else:
        fmt = {"price":"{:.2f}","minute_vol":"{:.0f}","daily_vol":"{:.0f}","active_vol":"{:.0f}"}
        st.dataframe(df_active.style.format(fmt), use_container_width=True)

    # --- Context panel (SPY, GLD, USO, SLV) ---
    st.markdown("### Context — SPY / Gold (GLD) / Oil (USO) / Silver (SLV)")
    df_ctx_ui = build_context_rows()
    if not df_ctx_ui.empty:
        st.dataframe(df_ctx_ui, use_container_width=True)

with right:
    st.subheader("Run Summary")
    st.write(f"**As of (ET):** {now_et_str()}")
    st.write(f"**Session:** {a_diag.get('session')}")
    st.write(f"**Universe cap:** {UNIVERSE_CAP:,}")
    st.write(f"**Universe scanned:** {len(all_syms):,}")
    st.write(f"**Price-band drops:** {a_diag.get('price_drops',0)}")
    st.caption("Diagnostics below show price/volume source usage (trade/minbar/daybar/prevday/quote).")

# Write ActiveNow (conflict-safe)
try:
    sh = get_gc()
    if not df_active.empty:
        write_frame_to_ws(sh, TAB_ACTIVENOW, df_active)
        st.success(f"{TAB_ACTIVENOW} updated.")
    else:
        write_frame_to_ws(sh, TAB_ACTIVENOW, pd.DataFrame(columns=[
            "as_of_et","session","symbol","price","minute_vol","daily_vol","active_vol","vol_source",
            "flag_pdh_break","flag_day_high_proximity"
        ]))
        st.info(f"{TAB_ACTIVENOW} created (empty).")
except Exception as e:
    st.error(f"Failed to write {TAB_ACTIVENOW}: {e}")

# Write Context (vertical key/value)
with st.spinner("Updating Context sheet…"):
    try:
        write_frame_to_ws(sh, TAB_CONTEXT, df_ctx_ui)
        st.success(f"{TAB_CONTEXT} refreshed.")
    except Exception as e:
        st.error(f"Failed to write {TAB_CONTEXT}: {e}")

# Diagnostics
with st.expander("🛠 Diagnostics", expanded=False):
    if _last_data_error:
        for k, v in _last_data_error.items():
            st.write(f"**{k}** → {v}")
    st.write("**Price source counts**:", _diag_counts["price_src"])
    st.write("**Volume source counts**:", _diag_counts["vol_src"])
