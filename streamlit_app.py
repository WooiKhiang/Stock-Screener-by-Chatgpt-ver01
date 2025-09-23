# Market Scanner — ActiveNow (fast) + Signals (light) → Google Sheets
# Tabs (always drop & recreate): ActiveNow, SignalsH1, SignalsD1, Context
# Stage A: Alpaca snapshots (IEX), no lookback → price band + active volume rank + PDH/DHP flags
# Stage B: On Top-K only → Alpaca H1/D1 bars (IEX) w/ fallback to yfinance → Donchian10, RSI(3), EMA stack, MACD zero-line, EMA200 breakout
# Robust snapshots: dynamic batch shrinking + longer timeout/retries + daily-bar fallback if needed

import os, json, time, random, datetime as dt
import numpy as np
import pandas as pd
import pytz, requests, streamlit as st

# =========================
# Config & Secrets
# =========================
GOOGLE_SHEET_ID = "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4"

TAB_ACTIVENOW = "ActiveNow"
TAB_SIG_H1    = "SignalsH1"
TAB_SIG_D1    = "SignalsD1"
TAB_CONTEXT   = "Context"

# Alpaca creds (prefer env; literals as fallback)
ALPACA_KEY    = os.getenv("ALPACA_KEY",    "PKIG445MPT704CN8P0R8")
ALPACA_SECRET = os.getenv("ALPACA_SECRET", "3GQdWcnbMo6V8uvhVa6BK6EbvnH4EHinlsU6uvj4")
ALPACA_BASE   = "https://paper-api.alpaca.markets/v2"
ALPACA_DATA   = "https://data.alpaca.markets"
FEED          = "iex"   # force IEX to avoid SIP entitlement issues

# Google SA JSON must be in Streamlit secrets
SA_RAW = st.secrets.get("GCP_SERVICE_ACCOUNT") or os.getenv("GCP_SERVICE_ACCOUNT")

ET = pytz.timezone("America/New_York")
HEADERS = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}

# =========================
# UI
# =========================
st.set_page_config(page_title="Market Scanner — ActiveNow + Signals", layout="wide")
st.title("⚡ Market Scanner — ActiveNow + Signals (Alpaca → Google Sheets)")

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

st.sidebar.markdown("### Filters (Stage A)")
PRICE_MIN = st.sidebar.number_input("Min price ($)", value=5.0, step=0.5)
PRICE_MAX = st.sidebar.number_input("Max price ($)", value=100.0, step=1.0)
TOP_K     = st.sidebar.slider("Top K to keep (by active volume)", 200, 3000, 1200, 100)

st.sidebar.markdown("### Flags (Stage A)")
DHP_DELTA = st.sidebar.slider("Day-high proximity δ", 0.001, 0.01, 0.005, 0.001,
                              help="Flag if last ≥ today's high × (1−δ). 0.005 = within 0.5% of day high")

st.sidebar.markdown("### Bars (Stage B — survivors only, automatic)")
H1_LIMIT = st.sidebar.slider("Hourly bars limit (H1)", 30, 80, 60, 5)
D1_LIMIT = st.sidebar.slider("Daily bars limit (D1)", 180, 260, 220, 10)
SNAPSHOT_CHUNK = st.sidebar.slider("Snapshot chunk size (start)", 100, 600, 300, 50)

# =========================
# Helpers
# =========================
def now_et():
    return dt.datetime.now(ET)

def now_et_str():
    return now_et().strftime("%Y-%m-%d %H:%M:%S")

def is_market_open_now():
    n = now_et()
    if n.weekday() >= 5:  # Sat/Sun
        return False
    return dt.time(9,30) <= n.time() <= dt.time(16,0)

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

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def macd_line_sig(x: pd.Series, fast=12, slow=26, signal=9):
    ef, es = ema(x, fast), ema(x, slow)
    line = ef - es
    sig  = line.ewm(span=signal, adjust=False).mean()
    return line, sig

def rsi(series: pd.Series, n: int = 3) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    r = 100 - (100 / (1 + rs))
    return r

# =========================
# Alpaca I/O
# =========================
_last_data_error = {}

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
            if s.endswith((".U",".W","WS","W","R",".P","-P")):  # sanitize tails
                continue
            syms.append(s)
        if len(syms) >= cap: break
    return syms[:cap]

# ---------- Robust snapshots ----------
def _snapshots_request(symbols_batch, max_retries=5, timeout_s=120):
    url = f"{ALPACA_DATA}/v2/stocks/snapshots"
    params = {"symbols": ",".join(symbols_batch), "feed": FEED}
    backoff = 1.0
    for attempt in range(max_retries):
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
        time.sleep(backoff + random.random()*0.5)  # jitter
        backoff = min(backoff*1.8, 8.0)
    _last_data_error["snapshots"] = (408, "retry timeout")
    return None

def fetch_bars_multi(symbols, timeframe="1Hour", limit=60, chunk=150):
    out = {}
    end = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    start = end - dt.timedelta(days=60 if timeframe=="1Hour" else 400)
    start_iso = start.isoformat().replace("+00:00","Z")
    end_iso   = end.isoformat().replace("+00:00","Z")

    def _bars_request(params, max_retries=4):
        url = f"{ALPACA_DATA}/v2/stocks/bars"
        p = dict(params); p["feed"] = FEED
        backoff = 1.0
        for attempt in range(max_retries):
            try:
                r = requests.get(url, headers=HEADERS, params=p, timeout=90)
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
                    if sym not in out: out[sym] = pd.DataFrame(columns=["t","open","high","low","close","volume"])
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
    cur_chunk = max(60, chunk)
    while i < len(symbols):
        batch = symbols[i:i+cur_chunk]
        params = dict(timeframe=timeframe, symbols=",".join(batch), limit=limit,
                      start=start_iso, end=end_iso, adjustment="raw")
        page=None; ok=False
        while True:
            p = dict(params)
            if page: p["page_token"] = page
            js = _bars_request(p)
            if js is None: break
            merge_json(js)
            page = js.get("next_page_token")
            if not page: ok=True; break
        if not ok and cur_chunk > 60:
            cur_chunk = max(40, cur_chunk // 2)
            continue
        i += len(batch)
        time.sleep(0.12)
    # Clean & RTH for H1
    for s, df in list(out.items()):
        if df is None or df.empty: continue
        df = df.drop_duplicates(subset=["t"]).sort_values("t").reset_index(drop=True)
        if timeframe.lower() in ("1hour","1h","60m","60min"):
            df = rth_only_hour(df)
        out[s] = df
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
            # still failing at small size → fallback to daily bars for this batch
            try:
                # today daily bar for vol & close
                bars_today = fetch_bars_multi(batch, timeframe="1Day", limit=1, chunk=max(60, cur_chunk))
                # prev daily bar for PDH
                bars_prev2 = fetch_bars_multi(batch, timeframe="1Day", limit=2, chunk=max(60, cur_chunk))
                for s in batch:
                    snap = {}
                    tbar = bars_today.get(s)
                    pbar2 = bars_prev2.get(s)
                    if tbar is not None and not tbar.empty:
                        last = tbar.iloc[-1]
                        snap["dailyBar"] = {
                            "c": float(last["close"]),
                            "h": float(last["high"]),
                            "v": float(last["volume"])
                        }
                    if pbar2 is not None and len(pbar2) >= 2:
                        prev = pbar2.iloc[-2]
                        snap["prevDailyBar"] = {"h": float(prev["high"]), "c": float(prev["close"])}
                    # minuteBar and latestTrade absent in fallback
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
        # cautiously bump chunk back up if stable
        if cur_chunk < chunk:
            cur_chunk = min(chunk, int(cur_chunk * 1.25))
    return out

# =========================
# Yahoo fallback (only if needed)
# =========================
def fetch_yf_hourly(symbols):
    try:
        import yfinance as yf
    except Exception:
        return {}
    out = {}
    for s in symbols:
        try:
            df = yf.download(s, period="30d", interval="1h", auto_adjust=False, progress=False, prepost=False, threads=False)
            if df is None or df.empty:
                out[s] = pd.DataFrame(); continue
            df = df.reset_index().rename(columns={"Datetime":"t","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
            df["t"] = pd.to_datetime(df["t"], utc=True)
            df = rth_only_hour(df)
            out[s] = df[["t","open","high","low","close","volume"]].copy()
        except Exception:
            out[s] = pd.DataFrame()
    return out

def fetch_yf_daily(symbols):
    try:
        import yfinance as yf
    except Exception:
        return {}
    out = {}
    for s in symbols:
        try:
            df = yf.download(s, period="250d", interval="1d", auto_adjust=False, progress=False, prepost=False, threads=False)
            if df is None or df.empty:
                out[s] = pd.DataFrame(); continue
            df = df.reset_index().rename(columns={"Date":"t","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
            df["t"] = pd.to_datetime(df["t"], utc=True)
            out[s] = df[["t","open","high","low","close","volume"]].copy()
        except Exception:
            out[s] = pd.DataFrame()
    return out

# =========================
# Google Sheets (drop & recreate)
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

def drop_and_create(sh, title, rows=200, cols=40):
    try:
        try:
            ws = sh.worksheet(title)
            # create temp to allow deletion if it's the last worksheet
            if len(sh.worksheets()) == 1:
                tmp = sh.add_worksheet(title="__tmp__", rows=1, cols=1)
                sh.del_worksheet(ws)
                ws = sh.add_worksheet(title=title, rows=rows, cols=cols)
                sh.del_worksheet(tmp)
                return ws
            else:
                sh.del_worksheet(ws)
        except Exception:
            pass
        ws = sh.add_worksheet(title=title, rows=rows, cols=cols)
        return ws
    except Exception:
        # fallback clear if cannot delete due to perms
        ws = sh.worksheet(title) if title in [w.title for w in sh.worksheets()] else sh.add_worksheet(title=title, rows=rows, cols=cols)
        ws.batch_clear(["A1:Z100000"])
        return ws

def write_frame_to_ws(sh, title, df: pd.DataFrame):
    ws = drop_and_create(sh, title, rows=max(200, len(df)+10), cols=max(10, len(df.columns)+2))
    values = [list(df.columns)] + df.fillna("").astype(str).values.tolist()
    ws.update("A1", values, value_input_option="RAW")

# =========================
# Stage A — snapshots → ActiveNow
# =========================
def stage_a_activenow():
    # Universe
    symbols = fetch_active_symbols(UNIVERSE_CAP)
    # Snapshots (robust)
    snaps = fetch_snapshots_multi(symbols, chunk=SNAPSHOT_CHUNK)
    rth = is_market_open_now()

    rows = []
    price_drops = 0
    for s in symbols:
        snap = snaps.get(s) or {}
        lt = snap.get("latestTrade") or {}
        mb = snap.get("minuteBar") or {}
        db = snap.get("dailyBar") or {}
        pdbar = snap.get("prevDailyBar") or {}

        # last price: latestTrade.p -> minuteBar.c -> dailyBar.c
        p = None
        try:
            if lt.get("p") is not None: p = float(lt["p"])
            elif mb.get("c") is not None: p = float(mb["c"])
            elif db.get("c") is not None: p = float(db["c"])
        except Exception:
            p = None
        if p is None or not (PRICE_MIN <= p <= PRICE_MAX):
            price_drops += 1; continue

        # volumes
        mv = float(mb.get("v") or 0.0)
        dv = float(db.get("v") or 0.0)
        active_vol = mv if (rth and mv > 0) else dv
        vol_source = "minute" if (rth and mv > 0) else "daily"

        # flags from snapshots (no lookback)
        prev_high = pdbar.get("h")
        today_high = db.get("h")

        flag_pdh_break = bool(prev_high is not None and p > float(prev_high))
        flag_day_high_prox = False
        if today_high is not None:
            try:
                flag_day_high_prox = p >= float(today_high) * (1.0 - float(DHP_DELTA))
            except Exception:
                flag_day_high_prox = False

        rows.append({
            "symbol": s,
            "price": p,
            "minute_vol": mv,
            "daily_vol": dv,
            "active_vol": active_vol,
            "vol_source": vol_source,
            "flag_pdh_break": flag_pdh_break,
            "flag_day_high_proximity": flag_day_high_prox
        })

    if not rows:
        return pd.DataFrame(), symbols, {"price_drops": price_drops, "rth": rth}

    df = pd.DataFrame(rows).sort_values(["active_vol","daily_vol","minute_vol"], ascending=[False, False, False]).reset_index(drop=True)
    df_out = df.head(TOP_K).copy()
    df_out.insert(0, "as_of_et", now_et_str())
    return df_out, symbols, {"price_drops": price_drops, "rth": rth}

# =========================
# Stage B — bars → SignalsH1 / SignalsD1
# =========================
def stage_b_signals(top_symbols):
    # ----- Hourly bars (H1) -----
    bars_h = fetch_bars_multi(top_symbols, timeframe="1Hour", limit=H1_LIMIT, chunk=150)
    need_yf_h = [s for s in top_symbols if (bars_h.get(s) is None or bars_h.get(s).empty or len(bars_h.get(s)) < 12)]
    if need_yf_h:
        yf_h = fetch_yf_hourly(need_yf_h)
        for s in need_yf_h:
            if yf_h.get(s) is not None and not yf_h[s].empty:
                bars_h[s] = yf_h[s]

    # ----- Daily bars (D1) -----
    bars_d = fetch_bars_multi(top_symbols, timeframe="1Day", limit=D1_LIMIT, chunk=150)
    need_yf_d = [s for s in top_symbols if (bars_d.get(s) is None or bars_d.get(s).empty or len(bars_d.get(s)) < 201)]
    if need_yf_d:
        yf_d = fetch_yf_daily(need_yf_d)
        for s in need_yf_d:
            if yf_d.get(s) is not None and not yf_d[s].empty:
                bars_d[s] = yf_d[s]

    # ----- Compute H1 flags -----
    rows_h1 = []
    for s in top_symbols:
        df = bars_h.get(s)
        if df is None or df.empty or len(df) < 12:
            continue
        cl = df["close"].astype(float); hi = df["high"].astype(float); vol = df["volume"].astype(float)

        # Donchian-10 breakout: close(t) > max(high[t-10 .. t-1])
        hh10_prev = hi.shift(1).rolling(10, min_periods=10).max().iloc[-1]
        donchian10_break = bool(np.isfinite(hh10_prev) and cl.iloc[-1] > hh10_prev)

        # RSI(3) cross above 50
        rsi3 = rsi(cl, 3)
        rsi3_cross50 = bool(len(rsi3) >= 2 and rsi3.iloc[-2] <= 50 and rsi3.iloc[-1] > 50)

        # EMA stack (info flag)
        e10, e20, e50, e100 = ema(cl,10), ema(cl,20), ema(cl,50), ema(cl,100)
        stack_h1_bull = bool(cl.iloc[-1] > e50.iloc[-1] > e100.iloc[-1] and e10.iloc[-1] > e20.iloc[-1])

        # MACD zero-line cross while signal < 0
        ml, ms = macd_line_sig(cl)
        macd_zero_cross_sig_neg = bool(len(ml)>=2 and (ml.iloc[-2] <= 0 < ml.iloc[-1]) and (ms.iloc[-1] < 0))

        rows_h1.append({
            "symbol": s,
            "last_h1_close": float(cl.iloc[-1]),
            "last_h1_vol": float(vol.iloc[-1]),
            "flag_donchian10_breakout": donchian10_break,
            "flag_rsi3_cross50": rsi3_cross50,
            "stack_h1_bull": stack_h1_bull,
            "flag_h1_macd_zero_cross_sig_neg": macd_zero_cross_sig_neg
        })

    df_h1 = pd.DataFrame(rows_h1)
    if not df_h1.empty:
        df_h1.insert(0, "as_of_et", now_et_str())

    # ----- Compute D1 flag -----
    rows_d1 = []
    for s in top_symbols:
        df = bars_d.get(s)
        if df is None or df.empty or len(df) < 201:
            continue
        cl = df["close"].astype(float)
        e200 = ema(cl,200)
        # Cross up today: close[t] > EMA200[t] and close[t-1] <= EMA200[t-1]
        flag = bool(cl.iloc[-1] > e200.iloc[-1] and cl.iloc[-2] <= e200.iloc[-2])
        rows_d1.append({
            "symbol": s,
            "last_d1_close": float(cl.iloc[-1]),
            "flag_d1_ema200_breakout": flag
        })

    df_d1 = pd.DataFrame(rows_d1)
    if not df_d1.empty:
        df_d1.insert(0, "as_of_et", now_et_str())

    return df_h1, df_d1

# =========================
# Context (no lookback)
# =========================
CONTEXT_TICKERS = ["SPY", "VIXY", "XLF", "XLK", "XLY", "XLP", "XLV", "XLE", "XLI", "XLU",
                   "XLRE", "XLB", "SMH", "XOP", "XBI", "XME", "KRE", "ITB", "IYT", "TAN"]

def build_context():
    snaps = fetch_snapshots_multi(CONTEXT_TICKERS, chunk=min(100, len(CONTEXT_TICKERS)))
    rows = [{"key":"as_of_et", "value": now_et_str()},
            {"key":"market_open", "value": str(is_market_open_now())}]
    for s in CONTEXT_TICKERS:
        snap = snaps.get(s) or {}
        db = snap.get("dailyBar") or {}
        pdbar = snap.get("prevDailyBar") or {}
        try:
            c = float(db.get("c") or np.nan)
            pc = float(pdbar.get("c") or np.nan)
            if np.isfinite(c) and np.isfinite(pc) and pc != 0:
                pct = 100.0*(c-pc)/pc
                rows.append({"key": f"{s}_1d_pct", "value": f"{pct:+.2f}%"})
        except Exception:
            pass
    return pd.DataFrame(rows, columns=["key","value"])

# =========================
# Run
# =========================
left, right = st.columns([2,1])

with st.spinner("Stage A — scanning snapshots…"):
    df_active, all_syms, a_diag = stage_a_activenow()

with left:
    st.subheader(f"ActiveNow — Top {len(df_active) if not df_active.empty else 0} by active volume (Price ${PRICE_MIN:.0f}–${PRICE_MAX:.0f})")
    if df_active.empty:
        st.warning("No matches under current settings.")
    else:
        fmt = {"price":"{:.2f}","minute_vol":"{:.0f}","daily_vol":"{:.0f}","active_vol":"{:.0f}"}
        st.dataframe(df_active.style.format(fmt), use_container_width=True)

with right:
    st.subheader("Run Summary (Stage A)")
    st.write(f"**As of (ET):** {now_et_str()}")
    st.write(f"**Universe cap:** {UNIVERSE_CAP:,}")
    st.write(f"**Universe scanned:** {len(all_syms):,}")
    st.write(f"**Market open:** {a_diag.get('rth')}")
    st.write(f"**Price-band drops:** {a_diag.get('price_drops',0)}")

# Write ActiveNow immediately (drop & recreate)
try:
    sh = get_gc()
    if not df_active.empty:
        write_frame_to_ws(sh, TAB_ACTIVENOW, df_active)
        st.success(f"{TAB_ACTIVENOW} updated.")
    else:
        write_frame_to_ws(sh, TAB_ACTIVENOW, pd.DataFrame(columns=[
            "as_of_et","symbol","price","minute_vol","daily_vol","active_vol","vol_source",
            "flag_pdh_break","flag_day_high_proximity"
        ]))
        st.info(f"{TAB_ACTIVENOW} created (empty).")
except Exception as e:
    st.error(f"Failed to write {TAB_ACTIVENOW}: {e}")

# Stage B — compute signals on survivors, then write tabs
if not df_active.empty:
    with st.spinner("Stage B — computing SignalsH1 / SignalsD1 on Top-K survivors…"):
        top_syms = df_active["symbol"].tolist()
        df_h1, df_d1 = stage_b_signals(top_syms)

        # Show brief preview
        st.subheader("Signals (preview)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**SignalsH1**")
            if df_h1.empty:
                st.warning("SignalsH1: no rows.")
            else:
                st.dataframe(df_h1.head(20), use_container_width=True)
        with col2:
            st.markdown("**SignalsD1**")
            if df_d1.empty:
                st.warning("SignalsD1: no rows.")
            else:
                st.dataframe(df_d1.head(20), use_container_width=True)

        # Write Sheets (drop & recreate)
        try:
            if not df_h1.empty:
                write_frame_to_ws(sh, TAB_SIG_H1, df_h1)
            else:
                write_frame_to_ws(sh, TAB_SIG_H1, pd.DataFrame(columns=[
                    "as_of_et","symbol","last_h1_close","last_h1_vol",
                    "flag_donchian10_breakout","flag_rsi3_cross50","stack_h1_bull","flag_h1_macd_zero_cross_sig_neg"
                ]))
            st.success(f"{TAB_SIG_H1} refreshed.")
        except Exception as e:
            st.error(f"Failed to write {TAB_SIG_H1}: {e}")
        try:
            if not df_d1.empty:
                write_frame_to_ws(sh, TAB_SIG_D1, df_d1)
            else:
                write_frame_to_ws(sh, TAB_SIG_D1, pd.DataFrame(columns=[
                    "as_of_et","symbol","last_d1_close","flag_d1_ema200_breakout"
                ]))
            st.success(f"{TAB_SIG_D1} refreshed.")
        except Exception as e:
            st.error(f"Failed to write {TAB_SIG_D1}: {e}")

# Context (small, no lookback)
with st.spinner("Updating Context…"):
    try:
        df_ctx = build_context()
        write_frame_to_ws(sh, TAB_CONTEXT, df_ctx)
        st.success(f"{TAB_CONTEXT} refreshed.")
    except Exception as e:
        st.error(f"Failed to write {TAB_CONTEXT}: {e}")

# Diagnostics
with st.expander("🛠 Diagnostics", expanded=False):
    if _last_data_error:
        for k, v in _last_data_error.items():
            st.write(f"**{k}** → {v}")
    else:
        st.write("No errors recorded.")
