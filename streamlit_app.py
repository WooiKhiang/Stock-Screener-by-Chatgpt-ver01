# Ultra-lean Hourly Scanner — Alpaca → Google Sheets
# Filters: Price in [5, 100], RVOL_hour >= 1.0
# Rank: RVOL_hour desc
# Output: overwrite "Universe" tab each run
# Notes: feed forced to IEX to avoid SIP 403; RTH-only for hourly bars

import os, json, time, datetime as dt
import numpy as np
import pandas as pd
import pytz, requests, streamlit as st

# ========= Config =========
GOOGLE_SHEET_ID = "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4"
TAB_UNIVERSE    = "Universe"

# Alpaca creds — prefer env; literals here as fallback
ALPACA_KEY    = os.getenv("ALPACA_KEY",    "PKIG445MPT704CN8P0R8")
ALPACA_SECRET = os.getenv("ALPACA_SECRET", "3GQdWcnbMo6V8uvhVa6BK6EbvnH4EHinlsU6uvj4")
ALPACA_BASE   = "https://paper-api.alpaca.markets/v2"
ALPACA_DATA   = "https://data.alpaca.markets"
FEED          = "iex"  # force IEX to avoid SIP 403

SA_RAW = st.secrets.get("GCP_SERVICE_ACCOUNT") or os.getenv("GCP_SERVICE_ACCOUNT")

ET = pytz.timezone("America/New_York")
HEADERS = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}

# ========= UI =========
st.set_page_config(page_title="Hourly RVOL Scanner — Alpaca → Sheets", layout="wide")
st.title("⏱️ Hourly RVOL Scanner — Alpaca → Google Sheets")

autorun = st.sidebar.checkbox("Auto-run hourly", value=True)
try:
    autorefresh = getattr(st, "autorefresh", None) or getattr(st, "experimental_autorefresh", None)
    if autorun and autorefresh:
        autorefresh(interval=60*60*1000, key="auto_1h_rvol")
    elif autorun:
        st.markdown("<script>setTimeout(()=>window.location.reload(),3600000);</script>", unsafe_allow_html=True)
except Exception:
    pass

st.sidebar.markdown("### Universe")
UNIVERSE_CAP = st.sidebar.slider("Symbols to scan (cap)", 200, 6000, 4000, 200)

st.sidebar.markdown("### Filters")
PRICE_MIN = st.sidebar.number_input("Min price ($)", value=5.0, step=0.5)
PRICE_MAX = st.sidebar.number_input("Max price ($)", value=100.0, step=1.0)
RVOL_MIN  = st.sidebar.number_input("RVOL_hour ≥", value=1.00, step=0.05, format="%.2f")

st.sidebar.markdown("### Lookback / Speed")
H1_LIMIT_BARS = st.sidebar.slider("Hourly bars to fetch (limit)", 40, 120, 80, 5,
                                  help="80 bars covers RVOL20 comfortably.")
H1_MIN_BARS_REQ = st.sidebar.slider("Minimum H1 bars required", 20, 80, 30, 1)
CHUNK_SIZE = st.sidebar.slider("Fetch chunk size", 60, 200, 150, 20)
SHOW_LIMIT = st.sidebar.slider("Rows to show", 10, 100, 50)

# ========= Helpers =========
def now_et_str():
    return dt.datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")

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

def rth_only_hour(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    et_times = df["t"].dt.tz_convert(ET)
    mask = (et_times.dt.time >= dt.time(9,30)) & (et_times.dt.time <= dt.time(16,0))
    return df[mask].reset_index(drop=True)

# ========= Alpaca I/O =========
_last_data_error = {"where":"", "status":None, "message":""}

def fetch_active_symbols(cap: int):
    url = f"{ALPACA_BASE}/assets"
    params = {"status": "active", "asset_class": "us_equity"}
    r = requests.get(url, headers=HEADERS, params=params, timeout=60); r.raise_for_status()
    keep_exch = {"NASDAQ","NYSE","AMEX"}
    syms = [x["symbol"] for x in r.json() if x.get("exchange") in keep_exch and x.get("tradable")]
    bad_suffixes = (".U",".W","WS","W","R",".P","-P")
    syms = [s for s in syms if not s.endswith(bad_suffixes)]
    return syms[:cap]

def _bars_request(params, max_retries=3):
    base = f"{ALPACA_DATA}/v2/stocks/bars"
    p = dict(params); p["feed"] = FEED  # force IEX
    for attempt in range(max_retries):
        r = requests.get(base, headers=HEADERS, params=p, timeout=60)
        if r.status_code in (429,500,502,503,504):
            time.sleep(1.2*(attempt+1)); continue
        if r.status_code >= 400:
            try: msg = r.json().get("message", r.text[:300])
            except Exception: msg = r.text[:300]
            _last_data_error.update({"where": f"/bars {p.get('timeframe')} feed=iex",
                                     "status": r.status_code, "message": msg})
            return None
        try: return r.json()
        except Exception:
            _last_data_error.update({"where": "parse feed=iex", "status": r.status_code, "message": "JSON parse error"})
            return None
    _last_data_error.update({"where": f"/bars {p.get('timeframe')} feed=iex", "status": 408, "message": "retry timeout"})
    return None

def fetch_bars_multi_limit(symbols, timeframe="1Hour", limit=80, chunk_size=150, max_retries=3):
    out = {}
    end = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    start = end - dt.timedelta(days=40)  # advisory
    start_iso = start.isoformat().replace("+00:00","Z")
    end_iso   = end.isoformat().replace("+00:00","Z")

    def merge_json(js):
        bars = js.get("bars", [])
        if isinstance(bars, dict):
            for sym, arr in bars.items():
                if not arr:
                    if sym not in out:
                        out[sym] = pd.DataFrame(columns=["t","open","high","low","close","volume"])
                    continue
                df = pd.DataFrame(arr)
                df["t"] = pd.to_datetime(df["t"], utc=True)
                df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"}, inplace=True)
                out[sym] = pd.concat([out.get(sym), df[["t","open","high","low","close","volume"]]], ignore_index=True)
        else:
            if bars:
                df = pd.DataFrame(bars)
                df["t"] = pd.to_datetime(df["t"], utc=True)
                df.rename(columns={"S":"symbol","o":"open","h":"high","l":"low","c":"close","v":"volume"}, inplace=True)
                for sym, grp in df.groupby("symbol"):
                    out[sym] = pd.concat([out.get(sym), grp[["t","open","high","low","close","volume"]]], ignore_index=True)

    i = 0
    while i < len(symbols):
        chunk = symbols[i:i+chunk_size]
        params = dict(timeframe=timeframe, symbols=",".join(chunk), limit=limit,
                      start=start_iso, end=end_iso, adjustment="raw")
        page=None; ok=False
        while True:
            p = dict(params)
            if page: p["page_token"] = page
            js = _bars_request(p, max_retries=max_retries)
            if js is None: break
            merge_json(js)
            page = js.get("next_page_token")
            if not page:
                ok=True; break
        if not ok:
            # per-symbol fallback inside this chunk
            if chunk_size > 60:
                chunk_size = max(60, chunk_size//2)
                continue
            for sym in chunk:
                params = dict(timeframe=timeframe, symbols=sym, limit=limit,
                              start=start_iso, end=end_iso, adjustment="raw")
                page=None
                for attempt in range(max_retries):
                    p = dict(params); 
                    if page: p["page_token"] = page
                    js = _bars_request(p, max_retries=max_retries)
                    if js is None: break
                    merge_json(js)
                    page = js.get("next_page_token")
                    if not page: break
                if sym not in out:
                    out[sym] = pd.DataFrame(columns=["t","open","high","low","close","volume"])
        i += chunk_size

    # RTH filter for hourly
    if timeframe.lower() in ("1hour","1h","60m","60min"):
        for s, df in list(out.items()):
            if df is None or df.empty: continue
            df = df.drop_duplicates(subset=["t"]).sort_values("t").reset_index(drop=True)
            out[s] = rth_only_hour(df)
    return out

# ========= Sheets write (overwrite + trim) =========
def _open_or_create_ws(gc, title, rows=200, cols=40):
    sh = gc.open_by_key(GOOGLE_SHEET_ID)
    try: return sh.worksheet(title)
    except Exception: return sh.add_worksheet(title=title, rows=rows, cols=cols)

def write_universe_safe(df: pd.DataFrame):
    import gspread
    from google.oauth2.service_account import Credentials
    sa_info = parse_sa(SA_RAW)
    creds = Credentials.from_service_account_info(sa_info,
        scopes=["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"])
    gc = gspread.authorize(creds)
    ws = _open_or_create_ws(gc, TAB_UNIVERSE)

    values = [list(df.columns)] + df.fillna("").astype(str).values.tolist()
    ws.update("A1", values, value_input_option="RAW")

    # Trim stale cells if new sheet is smaller than previous
    try:
        old_vals = ws.get_all_values()
        old_rows = len(old_vals); old_cols = max((len(r) for r in old_vals), default=0)
        new_rows = len(values); new_cols = len(values[0]) if values else 0
        def col_letter(n:int):
            s=""
            while n>0:
                n,r = divmod(n-1,26); s = chr(65+r)+s
            return s
        ranges=[]
        if old_rows>new_rows and old_cols>0:
            ranges.append(f"A{new_rows+1}:{col_letter(max(old_cols,new_cols))}")
        if old_cols>new_cols and new_rows>0:
            ranges.append(f"{col_letter(new_cols+1)}1:{col_letter(old_cols)}{new_rows}")
        if ranges:
            ws.batch_clear(ranges)
    except Exception:
        pass

# ========= Runner =========
def run_scan():
    # Universe
    symbols = fetch_active_symbols(UNIVERSE_CAP)
    total_universe = len(symbols)

    # Fetch hourly bars (IEX)
    bars_h = fetch_bars_multi_limit(symbols, timeframe="1Hour", limit=H1_LIMIT_BARS, chunk_size=CHUNK_SIZE)
    counts = [len(df) for df in bars_h.values() if isinstance(df, pd.DataFrame)]
    min_hbars = min(counts) if counts else 0

    drops = {"insufficient_bars":0, "price":0, "rvol":0}
    rows = []

    for s in symbols:
        df = bars_h.get(s)
        if df is None or len(df) < max(21, H1_MIN_BARS_REQ):
            drops["insufficient_bars"] += 1; continue

        cl = df["close"].astype(float)
        vol = df["volume"].astype(float).fillna(0)
        last_close = float(cl.iloc[-1])
        last_vol   = float(vol.iloc[-1])

        # Filter 1: Price
        if not (PRICE_MIN <= last_close <= PRICE_MAX):
            drops["price"] += 1; continue

        # Filter 2: RVOL_hour (vs 20-hour simple avg)
        base = float(vol.rolling(20, min_periods=20).mean().iloc[-1])
        if not (base and base>0):
            drops["rvol"] += 1; continue
        rvol = last_vol / base
        if rvol < RVOL_MIN:
            drops["rvol"] += 1; continue

        rows.append({
            "symbol": s,
            "close": last_close,
            "vol_last": last_vol,
            "avg_vol20": base,
            "rvol_hour": float(rvol),
        })

    if not rows:
        df_out = pd.DataFrame()
    else:
        df = pd.DataFrame(rows).sort_values("rvol_hour", ascending=False).reset_index(drop=True)
        df_out = df.copy()
        df_out.insert(0, "as_of_et", now_et_str())

    return total_universe, min_hbars, df_out, {"last_error": _last_data_error, "drops": drops}

# ========= Run =========
left, right = st.columns([2,1])

with st.spinner("Scanning…"):
    total_universe, min_hbars, df_out, diag = run_scan()

with right:
    st.subheader("Run Summary")
    st.write(f"**As of (ET):** {now_et_str()}")
    st.write(f"**Universe cap:** {UNIVERSE_CAP:,}")
    st.write(f"**Universe scanned:** {total_universe:,}")
    st.write(f"**Min H1 bars (fetched):** {min_hbars}")
    st.write("**Feed:** iex")

with left:
    st.subheader("Candidates (RVOL ≥ threshold; Price $5–$100)")
    if df_out.empty:
        st.warning("No matches under current thresholds.")
    else:
        fmt = {"close":"{:.2f}", "vol_last":"{:.0f}", "avg_vol20":"{:.0f}", "rvol_hour":"{:.2f}"}
        st.dataframe(df_out.head(SHOW_LIMIT).style.format(fmt), use_container_width=True)

# Write to Sheets (overwrite) if we have rows
if not df_out.empty:
    try:
        write_universe_safe(df_out)
        st.success("Universe tab updated.")
    except Exception as e:
        st.error(f"Failed to write Universe: {e}")
else:
    st.info("No rows to write — Universe left unchanged.")

# Diagnostics
with st.expander("🛠 Diagnostics", expanded=False):
    st.markdown("**Drop reasons (counts)**")
    st.json(diag.get("drops", {}))
    le = diag.get("last_error", {})
    if le and le.get("status") is not None:
        st.markdown("**Last data error**")
        st.write(f"Where: `{le.get('where')}`")
        st.write(f"HTTP: `{le.get('status')}`")
        st.code(str(le.get("message"))[:500])
