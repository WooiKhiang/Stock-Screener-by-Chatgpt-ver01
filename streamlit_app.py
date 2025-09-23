# Hourly Scanner (lean) — Alpaca → Google Sheets
# FIX: Force feed=iex (no SIP fallback) to avoid 403 "subscription does not permit querying recent SIP data".
# Gates: price band, H1 liquidity, RVOL_hour, EMA stack, optional KDJ cross
# Flags: H1 MACD zero cross w/ signal<0; D1 EMA200 breakout (survivors-only)
# Rank: 0.6*Z(RVOL_hour) + 0.4*Z(1/ATR%)
# Output: overwrite "Universe" tab each run (safe replace). RTH-only for hourly bars.

import os, json, time, datetime as dt
import numpy as np
import pandas as pd
import pytz, requests, streamlit as st

# ========= Config =========
GOOGLE_SHEET_ID = "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4"
TAB_UNIVERSE    = "Universe"

# Alpaca creds — prefer env; replace literals if needed
ALPACA_KEY    = os.getenv("ALPACA_KEY",    "PKIG445MPT704CN8P0R8")
ALPACA_SECRET = os.getenv("ALPACA_SECRET", "3GQdWcnbMo6V8uvhVa6BK6EbvnH4EHinlsU6uvj4")
ALPACA_BASE   = "https://paper-api.alpaca.markets/v2"
ALPACA_DATA   = "https://data.alpaca.markets"
FEED          = "iex"  # FORCE IEX

SA_RAW = st.secrets.get("GCP_SERVICE_ACCOUNT") or os.getenv("GCP_SERVICE_ACCOUNT")

ET = pytz.timezone("America/New_York")
HEADERS = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}

# ========= UI =========
st.set_page_config(page_title="Hourly Scanner — Alpaca → Sheets", layout="wide")
st.title("⏱️ Hourly Scanner (lean) — Alpaca → Google Sheets")

autorun = st.sidebar.checkbox("Auto-run hourly", value=True)
try:
    autorefresh = getattr(st, "autorefresh", None) or getattr(st, "experimental_autorefresh", None)
    if autorun and autorefresh:
        autorefresh(interval=60*60*1000, key="auto_1h")
    elif autorun:
        st.markdown("<script>setTimeout(()=>window.location.reload(),3600000);</script>", unsafe_allow_html=True)
except Exception:
    pass

st.sidebar.markdown("### Universe")
UNIVERSE_CAP = st.sidebar.slider("Symbols to scan (cap)", 200, 6000, 4000, 200)

st.sidebar.markdown("### Step 1 — Hygiene")
PRICE_MIN = st.sidebar.number_input("Min price ($)", value=5.0, step=0.5)
PRICE_MAX = st.sidebar.number_input("Max price ($)", value=300.0, step=1.0)
LIQ_MIN_AVG_VOL_H1 = st.sidebar.number_input("Hourly AvgVol20 ≥", value=100_000, step=25_000, format="%i")
LIQ_MIN_AVG_DV_H1  = st.sidebar.number_input("Hourly Avg$Vol20 ≥", value=5_000_000, step=500_000, format="%i")

st.sidebar.markdown("### Step 2 — Activity")
RVOL_MIN_H1 = st.sidebar.number_input("RVOL_hour ≥", value=1.00, step=0.05, format="%.2f")

st.sidebar.markdown("### Step 3 — Structure & Signal")
REQ_EMA_STACK = st.sidebar.checkbox("Require trend stack (close>EMA50>EMA100 & EMA10>EMA20)", value=True)
REQ_KDJ_CROSS = st.sidebar.checkbox("Require KDJ bullish cross (K>D with fresh cross)", value=True)

st.sidebar.markdown("### Lookback / Speed")
H1_LIMIT_BARS = st.sidebar.slider("Hourly bars to fetch (limit)", 50, 120, 80, 5,
                                  help="80 bars covers EMA100, ATR14, RVOL20 comfortably.")
H1_MIN_BARS_REQ = st.sidebar.slider("Minimum H1 bars required", 30, 80, 45, 1)
CHUNK_SIZE = st.sidebar.slider("Fetch chunk size", 60, 200, 150, 20)
SHOW_LIMIT = st.sidebar.slider("Rows to show", 10, 50, 20)

# ========= Helpers =========
def now_et_str():
    return dt.datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")

def parse_sa(raw_json: str) -> dict:
    if not raw_json:
        raise RuntimeError("Missing GCP_SERVICE_ACCOUNT")
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

# ========= I/O (Alpaca) =========
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
    """ Single HTTP call using FEED=iex only; no SIP fallback. """
    base = f"{ALPACA_DATA}/v2/stocks/bars"
    p = dict(params)
    p["feed"] = FEED  # FORCE IEX
    for attempt in range(max_retries):
        r = requests.get(base, headers=HEADERS, params=p, timeout=60)
        if r.status_code in (429,500,502,503,504):
            time.sleep(1.2*(attempt+1)); continue
        if r.status_code >= 400:
            try:
                msg = r.json().get("message", r.text[:300])
            except Exception:
                msg = r.text[:300]
            _last_data_error.update({"where": f"/bars {p.get('timeframe')} feed=iex",
                                     "status": r.status_code, "message": msg})
            return None
        try:
            return r.json()
        except Exception:
            _last_data_error.update({"where": f"/bars parse feed=iex", "status": r.status_code, "message": "JSON parse error"})
            return None
    _last_data_error.update({"where": f"/bars {p.get('timeframe')} feed=iex", "status": 408, "message": "retry timeout"})
    return None

def fetch_bars_multi_limit(symbols, timeframe="1Hour", limit=80, chunk_size=150, max_retries=3):
    """
    Multi-symbol bars fetch using limit; retries and per-symbol fallback.
    RTH filter is applied for hourly.
    FEED is forced to IEX to avoid SIP 403.
    """
    out = {}
    end = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    start = end - dt.timedelta(days=40)
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
        page = None; ok=False
        while True:
            p = dict(params)
            if page: p["page_token"] = page
            js = _bars_request(p, max_retries=max_retries)
            if js is None: break
            merge_json(js)
            page = js.get("next_page_token")
            if not page:
                ok = True; break
        if not ok:
            # per-symbol fallback
            if chunk_size > 60:
                # Try smaller chunks first
                i = i  # no-op; we'll re-loop with reduced chunk size
                chunk_size = max(60, chunk_size//2)
                continue
            for sym in chunk:
                params = dict(timeframe=timeframe, symbols=sym, limit=limit,
                              start=start_iso, end=end_iso, adjustment="raw")
                page=None; merged=False
                for attempt in range(max_retries):
                    pp = dict(params)
                    if page: pp["page_token"]=page
                    js = _bars_request(pp, max_retries=max_retries)
                    if js is None: break
                    merge_json(js)
                    page = js.get("next_page_token")
                    if not page:
                        merged=True; break
                if sym not in out:
                    out[sym] = pd.DataFrame(columns=["t","open","high","low","close","volume"])
        i += chunk_size

    # Clean + RTH filter for hourly
    if timeframe.lower() in ("1hour","1h","60min","60m","1hour"):
        for s, df in list(out.items()):
            if df is None or df.empty: continue
            df = df.drop_duplicates(subset=["t"]).sort_values("t").reset_index(drop=True)
            out[s] = rth_only_hour(df)
    return out

# ========= Indicators =========
def ema(x, n): return x.ewm(span=n, adjust=False).mean()
def atr(df, n=14):
    hi, lo, cl = df["high"], df["low"], df["close"]; pc = cl.shift(1)
    tr = pd.concat([(hi-lo).abs(), (hi-pc).abs(), (lo-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()
def kdj(df, n=9, k_period=3, d_period=3):
    low_min  = df["low"].rolling(n, min_periods=n).min()
    high_max = df["high"].rolling(n, min_periods=n).max()
    rsv = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    k = rsv.ewm(alpha=1.0/k_period, adjust=False).mean()
    d = k.ewm(alpha=1.0/d_period, adjust=False).mean()
    j = 3*k - 2*d
    return k, d, j
def macd_line_sig(x, fast=12, slow=26, signal=9):
    ef, es = ema(x, fast), ema(x, slow)
    line = ef - es
    sig  = line.ewm(span=signal, adjust=False).mean()
    return line, sig

# ========= Sheets write (safe replace) =========
def _open_or_create_ws(gc, title, rows=200, cols=60):
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

    # Trim stale cells
    try:
        old_vals = ws.get_all_values()
        old_rows = len(old_vals); old_cols = max((len(r) for r in old_vals), default=0)
        new_rows = len(values); new_cols = len(values[0]) if values else 0
        def col_letter(n:int):
            s=""; 
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

    # Fetch hourly bars (feed=iex enforced)
    bars_h = fetch_bars_multi_limit(symbols, timeframe="1Hour", limit=H1_LIMIT_BARS, chunk_size=CHUNK_SIZE)
    counts = [len(df) for df in bars_h.values() if isinstance(df, pd.DataFrame)]
    min_hbars = min(counts) if counts else 0
    if min_hbars < H1_MIN_BARS_REQ:
        return total_universe, min_hbars, pd.DataFrame(), {"last_error": _last_data_error, "drops": {"insufficient_bars": total_universe}}

    # Screen
    drops = {"insufficient_bars":0, "price":0, "liq":0, "rvol":0, "ema":0, "kdj":0}
    survivors = []

    for s in symbols:
        df = bars_h.get(s)
        if df is None or len(df) < H1_MIN_BARS_REQ:
            drops["insufficient_bars"] += 1; continue

        cl = df["close"].astype(float); hi = df["high"].astype(float); lo = df["low"].astype(float)
        vol = df["volume"].astype(float).fillna(0)
        last = float(cl.iloc[-1])

        # Step 1: Price
        if not (PRICE_MIN <= last <= PRICE_MAX):
            drops["price"] += 1; continue

        # Step 1: Liquidity (hourly 20-bar averages)
        avg_vol20 = float(vol.rolling(20, min_periods=20).mean().iloc[-1])
        avg_dv20  = float((cl*vol).rolling(20, min_periods=20).mean().iloc[-1])
        if not ((avg_vol20 >= LIQ_MIN_AVG_VOL_H1) or (avg_dv20 >= LIQ_MIN_AVG_DV_H1)):
            drops["liq"] += 1; continue

        # Step 2: Activity
        base = avg_vol20
        rvol = float(vol.iloc[-1]/base) if base and base>0 else np.nan
        if not (np.isfinite(rvol) and rvol >= RVOL_MIN_H1):
            drops["rvol"] += 1; continue

        # Step 3: Structure
        e10, e20, e50, e100 = ema(cl,10), ema(cl,20), ema(cl,50), ema(cl,100)
        if REQ_EMA_STACK:
            if not (last > e50.iloc[-1] > e100.iloc[-1] and e10.iloc[-1] > e20.iloc[-1]):
                drops["ema"] += 1; continue

        # Step 3: Signal (KDJ cross)
        k, d_, j = kdj(df)
        kdj_ok = True
        if REQ_KDJ_CROSS:
            kdj_ok = (k.iloc[-1] > d_.iloc[-1]) and (k.iloc[-2] <= d_.iloc[-2])
        if not kdj_ok:
            drops["kdj"] += 1; continue

        # Features & rank pieces
        atr14 = float(atr(df,14).iloc[-1])
        atr_pct = float(atr14/last) if last>0 else np.nan
        inv_atr = (1.0/atr_pct) if (np.isfinite(atr_pct) and atr_pct>0) else np.nan

        # H1 MACD attention flag (line crosses > 0 while signal < 0)
        ml, ms = macd_line_sig(cl)
        flag_h1_macd0 = bool(len(ml)>=2 and (ml.iloc[-2] <= 0 < ml.iloc[-1]) and (ms.iloc[-1] < 0))

        survivors.append({
            "symbol": s,
            "close": last,
            "avg_vol20": avg_vol20,
            "avg_dollar_vol20": avg_dv20,
            "rvol_hour": rvol,
            "ema10": float(e10.iloc[-1]), "ema20": float(e20.iloc[-1]),
            "ema50": float(e50.iloc[-1]), "ema100": float(e100.iloc[-1]),
            "kdj_k": float(k.iloc[-1]), "kdj_d": float(d_.iloc[-1]), "kdj_j": float(j.iloc[-1]),
            "atr_pct": atr_pct,
            "inv_atr_pct": inv_atr,
            "flag_h1_macd_zero_cross_sig_neg": flag_h1_macd0,
            "flag_d1_ema200_breakout": False,  # fill later
            "notes": "H1_MACD_zero_cross_sig_neg" if flag_h1_macd0 else ""
        })

    if not survivors:
        return total_universe, min_hbars, pd.DataFrame(), {"last_error": _last_data_error, "drops": drops}

    df = pd.DataFrame(survivors)

    # Fetch D1 bars only for survivors and tag EMA200 breakout (cheap; feed=iex)
    surv_syms = df["symbol"].tolist()
    bars_d = fetch_bars_multi_limit(surv_syms, timeframe="1Day", limit=220, chunk_size=min(CHUNK_SIZE, 120))
    ema200_break = {}
    for s in surv_syms:
        dfd = bars_d.get(s)
        if dfd is None or len(dfd) < 201:
            ema200_break[s] = False
            continue
        cd = dfd["close"].astype(float)
        e200 = ema(cd, 200)
        brk = bool(cd.iloc[-1] > e200.iloc[-1] and cd.iloc[-2] <= e200.iloc[-2])
        ema200_break[s] = brk

    df["flag_d1_ema200_breakout"] = df["symbol"].map(ema200_break).fillna(False)
    # Append to notes
    mask = df["flag_d1_ema200_breakout"]
    df.loc[mask, "notes"] = (df.loc[mask, "notes"].str.replace("^\\s*$", "D1_EMA200_breakout", regex=True))
    df.loc[mask & df["notes"].ne("D1_EMA200_breakout"), "notes"] = df.loc[mask & df["notes"].ne("D1_EMA200_breakout"), "notes"] + ", D1_EMA200_breakout"

    # Rank
    df["rank_score"] = 0.6*zscore(df["rvol_hour"]) + 0.4*zscore(df["inv_atr_pct"])
    df = df.sort_values("rank_score", ascending=False).reset_index(drop=True)

    # Output (Top-20)
    df_out = df.head(20).copy()
    df_out.insert(0, "as_of_et", now_et_str())

    cols = ["as_of_et","symbol","close","avg_vol20","avg_dollar_vol20","rvol_hour",
            "ema10","ema20","ema50","ema100","kdj_k","kdj_d","kdj_j","atr_pct","rank_score",
            "flag_h1_macd_zero_cross_sig_neg","flag_d1_ema200_breakout","notes"]
    df_out = df_out[cols]
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
    st.subheader("Top candidates (preview)")
    if df_out.empty:
        st.warning("No survivors under current gates.")
    else:
        fmt = {c:"{:.2f}" for c in ["close","rvol_hour","ema10","ema20","ema50","ema100","kdj_k","kdj_d","kdj_j","atr_pct","rank_score"] if c in df_out.columns}
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
