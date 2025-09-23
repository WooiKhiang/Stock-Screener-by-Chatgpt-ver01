# Ultra-lean Volume Scanner — ActiveNow (Snapshots only + tiny daily fallback) → Google Sheets
# Ranking: minute volume (if regular session) else daily volume; if missing → previous day volume.
# Price fallback order: latestTrade.p → minuteBar.c → dailyBar.c → prevDailyBar.c → latestQuote mid (bp/ap).
# Flags: PDH break, Day-high proximity (only when today's high exists).
# UI also shows SPY / GLD / USO / SLV prices + 1D % and clear session status.
# No historical lookback except 1–2 daily bars only for symbols whose snapshot is empty.

import os, json, time, random, datetime as dt
import numpy as np
import pandas as pd
import pytz, requests, streamlit as st

# ────────────────────────────────────────────────────────────────────────────────
# Config & Secrets
# ────────────────────────────────────────────────────────────────────────────────
GOOGLE_SHEET_ID = "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4"
TAB_ACTIVENOW   = "ActiveNow"
TAB_CONTEXT     = "Context"

ALPACA_KEY    = os.getenv("ALPACA_KEY",    "PKIG445MPT704CN8P0R8")
ALPACA_SECRET = os.getenv("ALPACA_SECRET", "3GQdWcnbMo6V8uvhVa6BK6EbvnH4EHinlsU6uvj4")
ALPACA_BASE   = "https://paper-api.alpaca.markets/v2"
ALPACA_DATA   = "https://data.alpaca.markets"
FEED          = "iex"   # stick to IEX to avoid SIP entitlement errors

SA_RAW = st.secrets.get("GCP_SERVICE_ACCOUNT") or os.getenv("GCP_SERVICE_ACCOUNT")

ET = pytz.timezone("America/New_York")
HEADERS = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}

# ────────────────────────────────────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────────────────────────────────────
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
UNIVERSE_CAP = st.sidebar.slider("Symbols to scan (cap)", 1000, 12000, 6000, 500)

st.sidebar.markdown("### Filters")
PRICE_MIN = st.sidebar.number_input("Min price ($)", value=5.0, step=0.5)
PRICE_MAX = st.sidebar.number_input("Max price ($)", value=100.0, step=1.0)
TOP_K     = st.sidebar.slider("Top K to keep (by volume)", 200, 3000, 1200, 100)

st.sidebar.markdown("### Flags (no lookback)")
DHP_DELTA = st.sidebar.slider("Day-high proximity δ", 0.001, 0.02, 0.005, 0.001,
                              help="Flag if last ≥ today's high × (1−δ). Only when today's high exists.")

st.sidebar.markdown("### Network")
SNAPSHOT_CHUNK = st.sidebar.slider("Snapshot chunk size (start)", 80, 300, 120, 20)

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
        v = float(x)
        return v if (v is not None and v > 0) else None
    except Exception:
        return None

# ────────────────────────────────────────────────────────────────────────────────
# Alpaca I/O (robust snapshots + per-batch hydration with tiny daily bars)
# ────────────────────────────────────────────────────────────────────────────────
_last_data_error = {}
_diag = {"snap_batches":0, "snap_empty_batches":0, "bars_hydrated_symbols":0}
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
                _last_data_error["snapshots"] = (r.status_code, msg); return None
            else:
                try: return r.json()
                except Exception:
                    _last_data_error["snapshots"] = (r.status_code, "JSON parse error"); return None
        time.sleep(backoff + random.random()*0.5)
        backoff = min(backoff*1.8, 8.0)
    _last_data_error["snapshots"] = (408, "retry timeout"); return None

def _bars_request(symbols, limit=2, chunk=80):
    # ultra-light daily bars for hydration only
    out = {}
    url = f"{ALPACA_DATA}/v2/stocks/bars"
    end = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    start = end - dt.timedelta(days=10)
    start_iso = start.isoformat().replace("+00:00","Z")
    end_iso   = end.isoformat().replace("+00:00","Z")
    i = 0
    while i < len(symbols):
        batch = symbols[i:i+chunk]
        params = {"symbols": ",".join(batch), "timeframe":"1Day","limit":limit,
                  "adjustment":"raw","feed":FEED,"start":start_iso,"end":end_iso}
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=60)
            if r.status_code >= 400:
                try: msg = r.json().get("message", r.text[:300])
                except Exception: msg = r.text[:300]
                _last_data_error["bars_1Day"] = (r.status_code, msg)
                js = {}
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
                    out[sym] = pd.DataFrame()
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
    out = {}
    i = 0
    while i < len(symbols):
        batch = symbols[i:i+chunk]
        js = _snapshots_request(batch, max_retries=5, timeout_s=90)
        _diag["snap_batches"] += 1
        snaps = (js or {}).get("snapshots") or {}
        # collect snaps for batch
        for s in batch:
            out[s] = snaps.get(s, {}) or {}
        # hydrate empties: if a symbol has neither (dailyBar nor prevDailyBar nor latestTrade/minuteBar/latestQuote), fill with tiny daily bars
        needs = []
        for s in batch:
            snap = out.get(s, {})
            has_any = False
            for key in ("latestTrade","minuteBar","dailyBar","prevDailyBar","latestQuote"):
                if key in snap and bool(snap.get(key)):
                    has_any = True; break
            if not has_any:
                needs.append(s)
        if needs:
            _diag["snap_empty_batches"] += 1
            bars1 = _bars_request(needs, limit=1, chunk=max(40, min(80, len(needs))))
            bars2 = _bars_request(needs, limit=2, chunk=max(40, min(80, len(needs))))
            for s in needs:
                snap = out.get(s, {}) or {}
                tbar = bars1.get(s)
                pbar = bars2.get(s)
                if tbar is not None and not tbar.empty:
                    last = tbar.iloc[-1]
                    snap["dailyBar"] = {"c": float(last["close"]), "h": float(last["high"]), "v": float(last["volume"])}
                if pbar is not None and len(pbar) >= 2:
                    prev = pbar.iloc[-2]
                    snap["prevDailyBar"] = {"c": float(prev["close"]), "h": float(prev["high"]), "v": float(prev["volume"])}
                if snap:
                    out[s] = snap
                    _diag["bars_hydrated_symbols"] += 1
        i += len(batch); time.sleep(0.12)
    return out

# price & volume pickers (with source counters)
def pick_price(snap):
    lt  = snap.get("latestTrade") or {}
    mb  = snap.get("minuteBar")   or {}
    db  = snap.get("dailyBar")    or {}
    pdB = snap.get("prevDailyBar")or {}
    qt  = snap.get("latestQuote") or {}
    for cand, tag in ((lt.get("p"),"trade"),
                      (mb.get("c"),"minbar"),
                      (db.get("c"),"daybar"),
                      (pdB.get("c"),"prevday")):
        v = safe_float(cand)
        if v:
            _src_counts["price"][tag] += 1
            return v
    # last resort: quote mid (or one side)
    bp = safe_float(qt.get("bp")); ap = safe_float(qt.get("ap"))
    if bp and ap:
        _src_counts["price"]["quote_mid"] += 1; return (bp+ap)/2.0
    if ap:
        _src_counts["price"]["quote_mid"] += 1; return ap
    if bp:
        _src_counts["price"]["quote_mid"] += 1; return bp
    _src_counts["price"]["none"] += 1; return None

def pick_active_volume(snap, rth: bool):
    mb = snap.get("minuteBar") or {}
    db = snap.get("dailyBar") or {}
    pdB = snap.get("prevDailyBar") or {}
    mv = safe_float(mb.get("v")) or 0.0
    dv = safe_float(db.get("v")) or 0.0
    pv = safe_float(pdB.get("v")) or 0.0
    if rth and mv > 0:
        _src_counts["vol"]["minute"] += 1; return mv, "minute"
    if dv > 0:
        _src_counts["vol"]["daily"] += 1; return dv, "daily"
    _src_counts["vol"]["prev_daily"] += 1; return pv, "prev_daily"

# ────────────────────────────────────────────────────────────────────────────────
# Google Sheets (conflict-safe)
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
# Stage A — snapshots → ActiveNow (no lookback)
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
    df_out = df.head(TOP_K).copy()
    df_out.insert(0, "as_of_et", now_et_str())
    df_out.insert(1, "session", session_status())
    return df_out, symbols, {"price_drops": price_drops, "session": session_status()}

# ────────────────────────────────────────────────────────────────────────────────
# Context rows (SPY / GLD / USO / SLV)
# ────────────────────────────────────────────────────────────────────────────────
CONTEXT_TICKERS = ["SPY","GLD","USO","SLV"]

def build_context_rows():
    snaps = fetch_snapshots_multi(CONTEXT_TICKERS, chunk=min(40, len(CONTEXT_TICKERS)))
    rows = [{"key":"as_of_et","value":now_et_str()},
            {"key":"session","value":session_status()},
            {"key":"note","value":"Minute vol used only in regular session; otherwise daily/prev-day volume."}]
    for s in CONTEXT_TICKERS:
        snap = snaps.get(s) or {}
        px = pick_price(snap)
        db = snap.get("dailyBar") or {}; pdB = snap.get("prevDailyBar") or {}
        pct = ""
        try:
            c = safe_float(db.get("c")); pc = safe_float(pdB.get("c"))
            if c and pc: pct = f"{100.0*(c-pc)/pc:+.2f}%"
        except Exception: pct = ""
        if px is not None: rows.append({"key":f"{s}_price","value":f"{px:.2f}"})
        if pct: rows.append({"key":f"{s}_1d_pct","value":pct})
    return pd.DataFrame(rows, columns=["key","value"])

# ────────────────────────────────────────────────────────────────────────────────
# Run
# ────────────────────────────────────────────────────────────────────────────────
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
    st.caption("If pre/post/closed, ranks by daily or prev-day volume with price fallbacks.")

# Write Sheets
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
    # Context sheet (vertical key/value)
    write_frame_to_ws(sh, TAB_CONTEXT, df_ctx_ui)
    st.success(f"{TAB_CONTEXT} refreshed.")
except Exception as e:
    st.error(f"Sheets write error: {e}")

# Diagnostics
with st.expander("🛠 Diagnostics", expanded=False):
    if _last_data_error:
        for k, v in _last_data_error.items():
            st.write(f"**{k}** → {v}")
    st.write("**Snapshot batches**:", _diag["snap_batches"])
    st.write("**Batches with empty snapshots (hydrated by bars)**:", _diag["snap_empty_batches"])
    st.write("**Symbols hydrated via tiny daily bars**:", _diag["bars_hydrated_symbols"])
    st.write("**Price source counts**:", _src_counts["price"])
    st.write("**Volume source counts**:", _src_counts["vol"])
