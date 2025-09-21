# streamlit_app.py — Pure Hourly Scan (H1) + Daily Flags → Google Sheets
# - Universe cap slider (default 4000)
# - Steps 1/2/3 entirely on H1 bars
# - Attention flags (not gates): D1_EMA200_breakout, D1_MACD_zero_cross_sig_neg, H1_MACD_zero_cross_sig_neg
# - Context tab: vertical (Metric, Value), readable, hard-refresh each run
# - Universe tab: transaction-safe overwrite
# - Uses GCP service account JSON from Streamlit secrets as GCP_SERVICE_ACCOUNT

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

ET = pytz.timezone("America/New_York")
HEADERS = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}

# =========================
# UI
# =========================
st.set_page_config(page_title="Pure Hourly Scan — Alpaca → Sheets", layout="wide")
st.title("⏱️ Pure Hourly Scan (H1) — Candidates to Google Sheets")

autorun = st.sidebar.checkbox("Auto-run every hour", value=True)
try:
    autorefresh = getattr(st, "autorefresh", None) or getattr(st, "experimental_autorefresh", None)
    if autorun and autorefresh:
        autorefresh(interval=60*60*1000, key="hourly_autorefresh")
    elif autorun:
        st.markdown("<script>setTimeout(()=>window.location.reload(),3600000);</script>", unsafe_allow_html=True)
except Exception:
    pass

st.sidebar.markdown("### Universe & Mode")
UNIVERSE_CAP = st.sidebar.slider("Universe cap (symbols)", 500, 6000, 4000, 250,
    help="Active & tradable US equities (NASDAQ/NYSE/AMEX). Higher = broader but slower.")
MANUAL_RISK  = st.sidebar.selectbox("Risk mode", ["auto", "normal", "tight"], index=0)

with st.sidebar.expander("Step 1 — Hourly Hygiene", expanded=True):
    PRICE_MIN = st.number_input("Min price ($)", value=5.0, step=0.5)
    PRICE_MAX = st.number_input("Max price ($)", value=100.0, step=1.0)
    st.caption("Liquidity (H1): pass if either condition holds over last 20 hours")
    LIQ_MIN_AVG_VOLH20 = st.number_input("AvgVolH20 ≥", value=100_000, step=25_000, format="%i")
    LIQ_MIN_AVG_DVH20  = st.number_input("Avg$VolH20 ≥", value=5_000_000, step=500_000, format="%i")
    REQUIRE_DAILY_TREND = st.checkbox("Also require Daily trend bias (Close ≥ SMA50 & SMA20 ≥ SMA50)", value=False,
        help="Optional: consult D1 to keep structural bias. OFF keeps scan purely hourly.")

with st.sidebar.expander("Step 2 — Hourly Activity", expanded=True):
    ATR_PCT_MIN = st.number_input("ATR% (H1) min", value=0.0010, step=0.0005, format="%.4f",
        help="0.0010 = 0.10% of price per hour")
    ATR_PCT_MAX = st.number_input("ATR% (H1) max", value=0.0200, step=0.0005, format="%.4f",
        help="2.00% cap avoids hyper-volatility")
    RVOL_MIN    = st.number_input("RVOL_hour (H1) ≥", value=1.00, step=0.05, format="%.2f",
        help="This hour's volume vs 20-hour average")

with st.sidebar.expander("Step 3 — Hourly Technical Gates", expanded=True):
    GATE_EMA  = st.checkbox("EMA stack rising (EMA5>EMA20>EMA50; all rising)", value=True)
    GATE_MACD = st.checkbox("MACD turn (hist<0 & rising; line>signal; line rising)", value=True)
    GATE_KDJ  = st.checkbox("KDJ align (K>D; K↑; D↑; J↑)", value=True)

with st.sidebar.expander("Lookbacks & Output", expanded=False):
    HOURLY_LOOK_D  = st.slider("Hourly lookback (days)", 5, 30, 10,
        help="~65–70 RTH hours → stable ATR(14), MACD(26/12/9), KDJ(9) & RVOL baselines.")
    DAILY_LOOKBACK = st.slider("Daily lookback (days) for flags/context", 60, 200, 100)
    SHOW_LIMIT     = st.slider("Rows to show (UI)", 10, 50, 20)
    TAG_SURV_LIMIT = st.slider("Tagging cap (survivors for flags)", 20, 200, 80, 10)

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
COMMODITY_ETFS = ["GLD","GDX","USO","XOP","OIH"]
MEGACAPS = ["AAPL","MSFT","NVDA","AMZN","GOOG","GOOGL","META","TSLA"]

def _yf_close_panel(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        panel = df["Adj Close"] if "Adj Close" in lvl0 else (df["Close"] if "Close" in lvl0 else df[lvl0[0]])
    else:
        panel = df
    return panel if isinstance(panel, pd.DataFrame) else panel.to_frame()

def fetch_refs():
    if not YF_AVAILABLE:
        return {"vix": np.nan, "sector1d":{}, "sector5d":{}, "industry1d":{}, "industry5d":{},
                "commodity1d":{}, "commodity5d":{}, "mega1d":{}, "mega5d":{}, "risk_auto":"normal"}
    tickers = ["SPY","^VIX"] + sorted(set(SECTORS + INDUSTRY_ETFS + COMMODITY_ETFS + MEGACAPS))
    data = yf.download(tickers=" ".join(tickers), period="10d", interval="1d", auto_adjust=False, progress=False)
    if data is None or len(data)==0:
        return {"vix": np.nan, "sector1d":{}, "sector5d":{}, "industry1d":{}, "industry5d":{},
                "commodity1d":{}, "commodity5d":{}, "mega1d":{}, "mega5d":{}, "risk_auto":"normal"}
    panel = _yf_close_panel(data).ffill()
    last = panel.iloc[-1]; prev = panel.iloc[-2] if len(panel)>=2 else last; prev5 = panel.iloc[-5] if len(panel)>=5 else prev
    ret1d = (last/prev - 1.0); ret5d = (last/prev5 - 1.0)
    vix = float(last.get("^VIX", np.nan))
    def sub(keys): 
        return {k: float(ret1d.get(k, np.nan)) for k in keys}, {k: float(ret5d.get(k, np.nan)) for k in keys}
    sec1, sec5 = sub(SECTORS); ind1, ind5 = sub(INDUSTRY_ETFS); com1, com5 = sub(COMMODITY_ETFS)
    mega1 = {m: float(ret1d.get(m, np.nan)) for m in MEGACAPS}
    mega5 = {m: float(ret5d.get(m, np.nan)) for m in MEGACAPS}
    defensives = ["XLV","XLP","XLU"]; cyclicals = ["XLK","XLY","XLF"]
    tight = (vix > 20.0) and (np.nanmean([sec1.get(s,np.nan) for s in defensives]) >
                              np.nanmean([sec1.get(s,np.nan) for s in cyclicals])) and \
                           (np.nanmean([sec5.get(s,np.nan) for s in defensives]) >
                              np.nanmean([sec5.get(s,np.nan) for s in cyclicals]))
    return {"vix": vix, "sector1d": sec1, "sector5d": sec5,
            "industry1d": ind1, "industry5d": ind5,
            "commodity1d": com1, "commodity5d": com5,
            "mega1d": mega1, "mega5d": mega5,
            "risk_auto": "tight" if tight else "normal"}

# =========================
# Universe & Bars
# =========================
def fetch_active_symbols(cap: int):
    url = f"{ALPACA_BASE}/assets"
    params = {"status": "active", "asset_class": "us_equity"}
    r = requests.get(url, headers=HEADERS, params=params, timeout=60); r.raise_for_status()
    keep_exch = {"NASDAQ","NYSE","AMEX"}
    syms = [x["symbol"] for x in r.json() if x.get("exchange") in keep_exch and x.get("tradable")]
    bad_suffixes = (".U",".W","WS","W","R",".P","-P")
    syms = [s for s in syms if not s.endswith(bad_suffixes)]
    return syms[:cap]

def fetch_bars_multi(symbols, timeframe, start_iso, end_iso, limit=1000):
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
        if df is None or df.empty: continue
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
# Pipeline (pure hourly scan)
# =========================
def run_pipeline_pure_hourly():
    # Context references
    refs = fetch_refs()
    risk_mode = refs["risk_auto"] if MANUAL_RISK=="auto" else MANUAL_RISK

    # Universe
    symbols = fetch_active_symbols(UNIVERSE_CAP)
    total_universe = len(symbols)

    # Bars: hourly for scan; daily for flags/context only
    now_utc = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    hourly_start = (now_utc - dt.timedelta(days=HOURLY_LOOK_D)).isoformat().replace("+00:00","Z")
    daily_start  = (now_utc - dt.timedelta(days=DAILY_LOOKBACK)).isoformat().replace("+00:00","Z")
    end_iso      = now_utc.isoformat().replace("+00:00","Z")

    bars_h = fetch_bars_multi(symbols, "1Hour", hourly_start, end_iso)
    bars_d = fetch_bars_multi(symbols, "1Day",  daily_start,  end_iso)

    # Big players lists
    big_rvol_h1 = []
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
    rows = []
    for s, df in bars_d.items():
        if df is None or len(df) < 2: continue
        close = df["close"].astype(float); vol = df["volume"].astype(float).fillna(0)
        rows.append((s, float(close.iloc[-1]*vol.iloc[-1])))
    if rows:
        tmp = pd.DataFrame(rows, columns=["symbol","dollar_vol_d1"]).sort_values("dollar_vol_d1", ascending=False)
        big_dv_d1 = tmp.head(10)["symbol"].tolist()

    # Screening
    step1 = step2 = step3 = 0
    survivors = []

    for s in symbols:
        dfh = bars_h.get(s)
        if dfh is None or len(dfh) < 60:
            continue

        close_h = dfh["close"].astype(float); vol_h = dfh["volume"].astype(float).fillna(0)
        last_h  = float(close_h.iloc[-1])
        if not (PRICE_MIN <= last_h <= PRICE_MAX):
            continue

        avg_vol_h20 = float(vol_h.rolling(20, min_periods=20).mean().iloc[-1])
        avg_dv_h20  = float((close_h*vol_h).rolling(20, min_periods=20).mean().iloc[-1])
        if not ((avg_vol_h20 >= LIQ_MIN_AVG_VOLH20) or (avg_dv_h20 >= LIQ_MIN_AVG_DVH20)):
            continue

        if REQUIRE_DAILY_TREND:
            dfd = bars_d.get(s)
            if dfd is None or len(dfd) < 60:
                continue
            cd = dfd["close"].astype(float)
            sma20_d = sma(cd,20).iloc[-1]; sma50_d = sma(cd,50).iloc[-1]
            if not (cd.iloc[-1] >= sma50_d and sma20_d >= sma50_d):
                continue

        step1 += 1

        atr14_h = float(atr(dfh, 14).iloc[-1])
        atr_pct = float(atr14_h / last_h) if last_h > 0 else np.nan
        base = vol_h.rolling(20, min_periods=20).mean().iloc[-1]
        rvol_hour = float(vol_h.iloc[-1]/base) if base and base>0 else np.nan
        if not (ATR_PCT_MIN <= atr_pct <= ATR_PCT_MAX and rvol_hour >= RVOL_MIN):
            continue

        step2 += 1

        # Technical gates on H1
        ok = True
        e5, e20, e50 = ema(close_h,5), ema(close_h,20), ema(close_h,50)
        if GATE_EMA:
            ok &= (e5.iloc[-1] > e20.iloc[-1] > e50.iloc[-1] and
                   e5.iloc[-1] > e5.iloc[-2] and e20.iloc[-1] > e20.iloc[-2] and e50.iloc[-1] > e50.iloc[-2])

        macd_line_h, macd_sig_h, macd_hist_h = macd(close_h)
        if GATE_MACD:
            ok &= ((macd_hist_h.iloc[-1] < 0) and (macd_hist_h.iloc[-1] > macd_hist_h.iloc[-2]) and
                   (macd_line_h.iloc[-1] > macd_sig_h.iloc[-1]) and (macd_line_h.iloc[-1] > macd_line_h.iloc[-2]))

        k, d_, j = kdj(dfh)
        if GATE_KDJ:
            ok &= (k.iloc[-1] > d_.iloc[-1] and k.iloc[-1] > k.iloc[-2] and d_.iloc[-1] > d_.iloc[-2] and j.iloc[-1] > j.iloc[-2])

        if not ok:
            continue

        step3 += 1

        # Ranking features (H1)
        roc20 = float(close_h.iloc[-1] / close_h.iloc[-20] - 1.0) if len(close_h) > 20 else np.nan
        inv_atr = (1.0 / atr_pct) if (atr_pct and atr_pct>0) else np.nan

        # Attention flags (not gates)
        notes = []
        flag_d1_ema200 = False
        flag_d1_macd0  = False
        flag_h1_macd0  = False

        # Daily flags
        dfd = bars_d.get(s)
        if dfd is not None and len(dfd) >= 201:
            cd = dfd["close"].astype(float)
            e200_d = ema(cd,200)
            if cd.iloc[-1] > e200_d.iloc[-1] and cd.iloc[-2] <= e200_d.iloc[-2]:
                flag_d1_ema200 = True; notes.append("D1_EMA200_breakout")
            m_line_d, m_sig_d, _ = macd(cd)
            if len(m_line_d) >= 2 and (m_line_d.iloc[-2] <= 0 < m_line_d.iloc[-1]) and (m_sig_d.iloc[-1] < 0):
                flag_d1_macd0 = True; notes.append("D1_MACD_zero_cross_sig_neg")

        # Hourly flag
        if len(macd_line_h) >= 2 and (macd_line_h.iloc[-2] <= 0 < macd_line_h.iloc[-1]) and (macd_sig_h.iloc[-1] < 0):
            flag_h1_macd0 = True; notes.append("H1_MACD_zero_cross_sig_neg")

        survivors.append({
            "symbol": s,
            "close": last_h,
            "avg_vol_h20": avg_vol_h20,
            "avg_dollar_vol_h20": avg_dv_h20,
            "atr14_h1": atr14_h,
            "atr_pct_h1": atr_pct,
            "ema5_h1": float(e5.iloc[-1]),
            "ema20_h1": float(e20.iloc[-1]),
            "ema50_h1": float(e50.iloc[-1]),
            "macd_line_h1": float(macd_line_h.iloc[-1]),
            "macd_signal_h1": float(macd_sig_h.iloc[-1]),
            "macd_hist_h1": float(macd_hist_h.iloc[-1]),
            "kdj_k_h1": float(k.iloc[-1]),
            "kdj_d_h1": float(d_.iloc[-1]),
            "kdj_j_h1": float(j.iloc[-1]),
            "rvol_hour": rvol_hour,
            "roc20_h1": roc20,
            "flag_d1_ema200_breakout": flag_d1_ema200,
            "flag_d1_macd_zero_cross_sig_neg": flag_d1_macd0,
            "flag_h1_macd_zero_cross_sig_neg": flag_h1_macd0,
            "notes": ", ".join(notes) if notes else ""
        })

    if not survivors:
        return refs, risk_mode, total_universe, step1, step2, step3, pd.DataFrame(), pd.DataFrame(), big_rvol_h1, big_dv_d1

    df = pd.DataFrame(survivors)
    df["inv_atr_pct_h1"] = df["atr_pct_h1"].replace(0, np.nan).rpow(-1)
    # Rank: ROC20 (H1) + RVOL_hour + 1/ATR% (H1)
    df["rank_score"] = 0.4*zscore(df["roc20_h1"]) + 0.3*zscore(df["rvol_hour"]) + 0.3*zscore(df["inv_atr_pct_h1"])
    df = df.sort_values("rank_score", ascending=False).reset_index(drop=True)

    df_ui  = df.head(SHOW_LIMIT).copy()
    df_out = df.head(20).copy()
    as_of = now_et_str()
    df_out.insert(0, "risk_mode", risk_mode)
    df_out.insert(0, "as_of_date", as_of)

    # Universe output columns
    cols = ["as_of_date","risk_mode","symbol","close",
            "avg_vol_h20","avg_dollar_vol_h20","atr14_h1","atr_pct_h1",
            "ema5_h1","ema20_h1","ema50_h1","macd_line_h1","macd_signal_h1","macd_hist_h1",
            "kdj_k_h1","kdj_d_h1","kdj_j_h1","rvol_hour","roc20_h1","rank_score",
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
    """Transaction-safe overwrite: write new content then trim leftovers."""
    if df is None or df.empty:
        raise RuntimeError("No rows to write; preserving previous data.")
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

    # Trim leftovers safely
    try:
        def col_letter(n:int):
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

def write_context_hard_replace(metric_value_rows: list):
    """Hard-refresh Context tab via temp worksheet swap (atomic)."""
    import gspread
    from google.oauth2.service_account import Credentials
    sa_info = parse_sa(SA_RAW)
    creds = Credentials.from_service_account_info(sa_info,
               scopes=["https://www.googleapis.com/auth/spreadsheets",
                       "https://www.googleapis.com/auth/drive"])
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(GOOGLE_SHEET_ID)

    temp_name = f"{TAB_CONTEXT}_TMP"
    try:
        try:
            ws_tmp = sh.worksheet(temp_name)
        except Exception:
            ws_tmp = sh.add_worksheet(title=temp_name, rows=500, cols=2)
        df_ctx = pd.DataFrame(metric_value_rows, columns=["Metric","Value"])
        ws_tmp.clear()
        ws_tmp.update("A1", [list(df_ctx.columns)] + df_ctx.fillna("").astype(str).values.tolist(), value_input_option="RAW")
        try:
            ws_old = sh.worksheet(TAB_CONTEXT)
            sh.del_worksheet(ws_old)
        except Exception:
            pass
        ws_tmp.update_title(TAB_CONTEXT)
    except Exception as e:
        # Fallback: direct clear & write
        ws = _open_or_create_ws(gc, TAB_CONTEXT)
        ws.clear()
        ws.update("A1", [["Metric","Value"]] + metric_value_rows, value_input_option="RAW")

# =========================
# RUN
# =========================
sentiment_col, summary_col = st.columns([2,1])

with st.spinner("Scanning on Hourly (H1)…"):
    refs, risk_mode, total_universe, s1, s2, s3, df_ui, df_out, big_rvol_h1, big_dv_d1 = run_pipeline_pure_hourly()

# Sentiment / references
with sentiment_col:
    st.subheader("Market Context & Breadth")
    vix_val = refs.get("vix", np.nan)
    st.write(
        f"**VIX:** {vix_val:.2f}" if isinstance(vix_val,(int,float)) and np.isfinite(vix_val) else "**VIX:** n/a",
        f" | **Risk (auto):** `{refs.get('risk_auto','normal')}`",
        f" | **Using:** `{risk_mode}`"
    )
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
    st.warning("No survivors under current hourly gates.")
else:
    num_cols = ["close","avg_vol_h20","avg_dollar_vol_h20","atr14_h1","atr_pct_h1",
                "ema5_h1","ema20_h1","ema50_h1","macd_line_h1","macd_signal_h1","macd_hist_h1",
                "kdj_k_h1","kdj_d_h1","kdj_j_h1","rvol_hour","roc20_h1","rank_score"]
    for c in num_cols:
        if c in df_ui.columns: df_ui[c] = pd.to_numeric(df_ui[c], errors="coerce")
    fmt = {c:"{:.2f}" for c in num_cols if c in df_ui.columns}
    st.dataframe(df_ui.head(SHOW_LIMIT).style.format(fmt), use_container_width=True)

# Build Context (vertical) and write (hard refresh)
ctx = []
ctx.append(["Timestamp (ET)", now_et_str()])
ctx.append(["Risk mode (auto)", refs.get("risk_auto","")])
ctx.append(["Risk mode (using)", risk_mode])
ctx.append(["Scan timeframe", "Hourly (H1)"])
ctx.append(["Universe cap", f"{UNIVERSE_CAP}"])
ctx.append(["Universe scanned", f"{total_universe}"])
ctx.append(["Step 1 survivors", f"{s1}"])
ctx.append(["Step 2 survivors", f"{s2}"])
ctx.append(["Step 3 survivors", f"{s3}"])
ctx.append(["VIX level", f"{refs.get('vix', np.nan):.2f}" if isinstance(refs.get('vix'),(int,float)) and np.isfinite(refs.get('vix')) else "n/a"])

def add_bucket(title, d1:dict, d5:dict):
    ctx.append([title, ""])
    for k in sorted(d1.keys()):
        v1 = d1.get(k, np.nan); v5 = d5.get(k, np.nan)
        v1s = f"{v1:.2%}" if isinstance(v1,(int,float)) and np.isfinite(v1) else "n/a"
        v5s = f"{v5:.2%}" if isinstance(v5,(int,float)) and np.isfinite(v5) else "n/a"
        ctx.append([f"  {k}", f"1D {v1s} | 5D {v5s}"])

add_bucket("Sectors",   refs.get("sector1d",{}),   refs.get("sector5d",{}))
add_bucket("Industries",refs.get("industry1d",{}), refs.get("industry5d",{}))
add_bucket("Gold/Oil",  refs.get("commodity1d",{}),refs.get("commodity5d",{}))

if big_rvol_h1:
    ctx.append(["Top-10 RVOL (Hourly)", ", ".join(big_rvol_h1)])
if big_dv_d1:
    ctx.append(["Top-10 Dollar Volume (Daily)", ", ".join(big_dv_d1)])

try:
    write_context_hard_replace(ctx)
    st.success("Context tab refreshed.")
except Exception as e:
    st.error(f"Failed to write Context: {e}")

# Write Universe (only when we have rows)
try:
    if df_out.empty:
        st.warning("No rows to write — Universe left unchanged.")
    else:
        write_sheet_safe(df_out, TAB_UNIVERSE)
        st.success("Universe tab updated.")
except Exception as e:
    st.error(f"Failed to write Universe (previous data preserved): {e}")

st.caption("All gates run on H1. Daily is used only for attention flags and market context. Context is vertical and hard-refreshed each run.")
