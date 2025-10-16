# streamlit_app.py — Stage A Universe Builder (Yahoo) + Hourly Up-Bias • Auto Sheet Overwrite
# - Daily ADV + trend filter (Top 1000, $5–$1000) -> overwrite "Universe" once/day after close
# - Hourly light 1h up-bias check on the Top 1000 (regular session only, no pre/post)
# - Writes a timestamped summary to "Scanned Result" every run
# - Designed to be the *feeder* for Stage B screener (separate app)

import os
import time
import json
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials

# Optional hourly auto-refresh (won't crash if package not installed)
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

# =========================
# Config
# =========================
GOOGLE_SHEET_ID = "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4"
UNIVERSE_SHEET_NAME = "Universe"
RESULT_SHEET_NAME = "Scanned Result"

# Universe build targets
PRICE_MIN = 5.0
PRICE_MAX = 1000.0
TOP_N = 1000

# Timing: US equities (NYSE/Nasdaq) — close at 16:00 ET
US_EASTERN_OFFSET = -4  # EDT ~ -4; (Cloud will handle DST imperfectly; we only need "post-close" buffer)
DAILY_REBUILD_BUFFER_MIN = 15   # run daily rebuild ~15 minutes after close
HOURLY_REFRESH_MINUTE_OFFSET = 5  # refresh at HH:05-ish to ensure bar finalized

REGULAR_SESSION_PREPOST = False  # ignore pre/post for swing

# =========================
# Utilities
# =========================
def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def utcnow_iso() -> str:
    return utcnow().strftime("%Y-%m-%d %H:%M:%S %Z")

def _local_us_et_now() -> datetime:
    # Simple offset; OK for "post-close buffer" logic. (For precise DST, use pytz/zoneinfo if desired.)
    return datetime.utcnow().replace(tzinfo=timezone.utc) + timedelta(hours=US_EASTERN_OFFSET)

# =========================
# Google Sheets helpers
# =========================
def _get_gspread_client():
    raw = st.secrets.get("gcp_service_account")
    if not raw:
        raise RuntimeError("Missing [gcp_service_account] in secrets.")
    info = json.loads(raw) if isinstance(raw, str) else dict(raw)
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    credentials = Credentials.from_service_account_info(info, scopes=scopes)
    return gspread.authorize(credentials)

@st.cache_data(show_spinner=False)
def read_sheet_columnwise(sheet_id: str, tab_name: str) -> List[str]:
    gc = _get_gspread_client()
    ws = gc.open_by_key(sheet_id).worksheet(tab_name)
    values = ws.get_all_values()
    out = []
    for row in values:
        for cell in row:
            s = (cell or "").strip().upper()
            if s and all(ch.isalnum() or ch in (".","-","_") for ch in s):
                out.append(s)
    # dedupe preserving order
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            uniq.append(t); seen.add(t)
    return uniq

def write_sheet_overwrite(sheet_id: str, tab_name: str, df: pd.DataFrame):
    gc = _get_gspread_client()
    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(tab_name)
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=tab_name, rows=str(max(len(df)+10, 1000)), cols=str(max(len(df.columns)+5, 10)))
    vals = [list(df.columns)] + df.astype(object).where(pd.notnull(df), "").values.tolist()
    ws.update("A1", vals, value_input_option="RAW")

def append_or_overwrite_results(sheet_id: str, tab_name: str, df: pd.DataFrame):
    # overwrite with current run (keeps it simple + deterministic)
    write_sheet_overwrite(sheet_id, tab_name, df)

# =========================
# Yahoo helpers
# =========================
def _chunks(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

@st.cache_data(show_spinner=True)
def yf_download_daily(tickers: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
    """
    Download daily bars in batches (auto_adjust = True).
    Returns symbol -> DataFrame columns: Open, High, Low, Close, Volume (lower-case renamed).
    """
    out: Dict[str, pd.DataFrame] = {}
    if not tickers:
        return out
    for batch in _chunks(tickers, 150):
        data = yf.download(" ".join(batch), interval="1d", period=period, group_by="ticker",
                           auto_adjust=True, threads=True, progress=False, prepost=False)
        if isinstance(data.columns, pd.MultiIndex):
            for sym in batch:
                if sym in data.columns.get_level_values(0):
                    df = data[sym].rename(columns=str.lower)
                    if not df.empty:
                        out[sym] = df[["open","high","low","close","volume"]].dropna()
        else:
            df = data.rename(columns=str.lower)
            if not df.empty:
                out[batch[0]] = df[["open","high","low","close","volume"]].dropna()
        time.sleep(0.03)
    return out

@st.cache_data(show_spinner=True)
def yf_download_1h_light(tickers: List[str], period: str = "30d") -> Dict[str, pd.DataFrame]:
    """
    Download 1h bars for a shortlist (Top 1000). Regular session only (prepost=False).
    Returns symbol -> DataFrame (open, high, low, close, volume).
    """
    out: Dict[str, pd.DataFrame] = {}
    if not tickers:
        return out
    for batch in _chunks(tickers, 80):
        data = yf.download(" ".join(batch), interval="1h", period=period, group_by="ticker",
                           auto_adjust=True, threads=True, progress=False, prepost=REGULAR_SESSION_PREPOST)
        if isinstance(data.columns, pd.MultiIndex):
            for sym in batch:
                if sym in data.columns.get_level_values(0):
                    df = data[sym].rename(columns=str.lower)
                    if not df.empty:
                        out[sym] = df[["open","high","low","close","volume"]].dropna()
        else:
            df = data.rename(columns=str.lower)
            if not df.empty:
                out[batch[0]] = df[["open","high","low","close","volume"]].dropna()
        time.sleep(0.05)
    return out

# =========================
# Stage A: Daily universe build
# =========================
def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()

def build_universe_daily(seed_tickers: List[str]) -> pd.DataFrame:
    """
    Build liquid up-trend universe on DAILY data.
    - price in [PRICE_MIN, PRICE_MAX]
    - rank by ADV (close*volume)
    - up-trend: close>SMA200, SMA50>SMA200, SMA50 slope > 0 (last 5 vs prior 5)
    Returns DataFrame with columns: symbol, last_close, adv, sma50, sma200, sma50_slope_pos, kept, rank
    """
    daily = yf_download_daily(seed_tickers, period="1y")
    rows = []
    for sym, df in daily.items():
        if df.empty or "close" not in df or "volume" not in df:
            continue
        c = df["close"]; v = df["volume"]
        last_close = float(c.iloc[-1])
        if not (PRICE_MIN <= last_close <= PRICE_MAX):
            continue
        adv = float((c * v).mean())
        s50 = sma(c, 50)
        s200 = sma(c, 200)
        if pd.isna(s200.iloc[-1]) or pd.isna(s50.iloc[-1]):
            continue
        up_trend = (last_close > s200.iloc[-1]) and (s50.iloc[-1] > s200.iloc[-1])

        # simple slope over last ~5 trading days vs prior 5
        if len(s50.dropna()) >= 15:
            recent = s50.tail(5).mean()
            prev = s50.tail(10).head(5).mean()
            slope_pos = bool(recent > prev)
        else:
            slope_pos = False

        kept = bool(up_trend and slope_pos)
        rows.append({
            "symbol": sym,
            "last_close": last_close,
            "adv": adv,
            "sma50": float(s50.iloc[-1]),
            "sma200": float(s200.iloc[-1]),
            "sma50_slope_pos": slope_pos,
            "kept": kept
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Keep only "kept", then rank by ADV and cap to TOP_N
    df = df[df["kept"]].copy()
    if df.empty:
        return df
    df = df.sort_values("adv", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df)+1)
    df = df.head(TOP_N)
    return df[["symbol","last_close","adv","sma50","sma200","sma50_slope_pos","rank"]]

# =========================
# Stage A-light: Hourly up-bias (cheap)
# =========================
def compute_hourly_bias(df_1h: pd.DataFrame) -> float:
    """
    Return a small bias score [0..1] from 1h bars:
    - recent momentum: close > SMA20 > SMA50 (uses last 50 bars)
    - optional bonus if last close > previous close
    """
    if df_1h is None or df_1h.empty or len(df_1h) < 50:
        return 0.0
    close = df_1h["close"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    last = close.iloc[-1]
    prev = close.iloc[-2]
    cond_trend = (last > sma20.iloc[-1] > sma50.iloc[-1])
    bonus = 0.2 if (last > prev) else 0.0
    return float((1.0 if cond_trend else 0.0) + bonus)

def attach_hourly_bias(top_symbols: List[str]) -> pd.DataFrame:
    if not top_symbols:
        return pd.DataFrame(columns=["symbol","hourly_bias"])
    bars = yf_download_1h_light(top_symbols, period="30d")
    rows = []
    for sym in top_symbols:
        df = bars.get(sym)
        score = compute_hourly_bias(df) if df is not None else 0.0
        rows.append({"symbol": sym, "hourly_bias": round(score, 3)})
    return pd.DataFrame(rows)

# =========================
# Auto-refresh logic
# =========================
def hourly_autorefresh():
    # Try to refresh every hour at ~HH:05
    if st_autorefresh:
        st_autorefresh(interval=60*60*1000, key="hourly-refresh")
    # Else: best-effort banner showing the intended cadence
    st.caption("⏱️ Auto-refresh cadence: hourly (~HH:05) — if not auto, click 'Rerun'.")

def should_rebuild_daily_now() -> bool:
    et_now = _local_us_et_now()
    # After US close + buffer OR first run of the day
    closed_buffer = et_now.hour > 16 or (et_now.hour == 16 and et_now.minute >= DAILY_REBUILD_BUFFER_MIN)
    last_day = st.session_state.get("last_daily_rebuild_day")
    today = et_now.date()
    if last_day != today and closed_buffer:
        return True
    # Also, on very first run (nothing built yet)
    if last_day is None:
        return True
    return False

def mark_daily_rebuilt():
    et_now = _local_us_et_now()
    st.session_state["last_daily_rebuild_day"] = et_now.date()

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Stage A — Universe Builder (Yahoo)", layout="wide")
st.title("📈 Stage A — Yahoo Universe Builder (Top 1000, Up-Trend) + Hourly Up-Bias")
st.caption("Daily: build/overwrite 'Universe' (Top 1000 by ADV, $5–$1000, up-trend). Hourly: add a light 1h bias score (no pre/post).")

hourly_autorefresh()

with st.expander("Info & Criteria", expanded=True):
    st.markdown("""
**Universe Build (Daily, once per day after close)**
- Price filter: **$5 to $1000**
- Liquidity: **Top 1000 by ADV** (mean of **Close × Volume** over ~1 year daily)
- Up-trend (daily): **Close > SMA200**, **SMA50 > SMA200**, **SMA50 rising** (recent 5d vs prior 5d)
- Writes back to Google Sheet: **Universe** (auto-overwrite)

**Hourly Up-Bias (light)**
- 1h (regular session only): **Close > SMA20 > SMA50**, +0.2 bonus if last close > previous close
- Writes a timestamped summary to **Scanned Result**
    """)

# Load seed tickers (initial run or if Universe is currently empty)
try:
    seed = read_sheet_columnwise(GOOGLE_SHEET_ID, UNIVERSE_SHEET_NAME)
except Exception as e:
    st.error(f"Failed to read 'Universe' sheet: {e}")
    seed = []

# Decide daily rebuild
if should_rebuild_daily_now():
    if not seed:
        st.warning("Universe sheet empty; please add a broad US list (e.g., your maintained symbols).")
    else:
        with st.spinner("Building daily universe (Top 1000 by ADV + up-trend)…"):
            df_uni = build_universe_daily(seed)
        if df_uni.empty:
            st.error("Daily universe build produced no symbols. (Seed may be too narrow or filters too strict.)")
        else:
            # Overwrite Universe
            write_sheet_overwrite(GOOGLE_SHEET_ID, UNIVERSE_SHEET_NAME, df_uni[["symbol","last_close","adv","sma50","sma200","sma50_slope_pos","rank"]])
            mark_daily_rebuilt()
            st.success(f"Universe overwritten with {len(df_uni)} symbols at {utcnow_iso()}.")

# Use current Universe (post-rebuild or existing)
try:
    universe_now = read_sheet_columnwise(GOOGLE_SHEET_ID, UNIVERSE_SHEET_NAME)
except Exception as e:
    st.error(f"Failed to read 'Universe' sheet after rebuild: {e}")
    universe_now = []

# Limit to TOP_N if sheet contains more
universe_now = universe_now[:TOP_N]

# Hourly up-bias summary (only if we have symbols)
summary_rows = []
if universe_now:
    with st.spinner(f"Fetching 1h (regular session) for {len(universe_now)} symbols to compute light up-bias…"):
        df_bias = attach_hourly_bias(universe_now)
    # Merge (some rows might be missing if yfinance returns nothing for a symbol)
    out = pd.DataFrame({"symbol": universe_now})
    out = out.merge(df_bias, on="symbol", how="left").fillna({"hourly_bias": 0.0})
    out["timestamp_utc"] = utcnow().strftime("%Y-%m-%d %H:%M:%S")
    out = out[["timestamp_utc","symbol","hourly_bias"]]
    st.subheader("Hourly Up-Bias (light) on Current Universe")
    st.dataframe(out.head(200), use_container_width=True)

    # Write summary to Scanned Result (overwrite with latest snapshot)
    try:
        append_or_overwrite_results(GOOGLE_SHEET_ID, RESULT_SHEET_NAME, out)
        st.success(f"Snapshot written to '{RESULT_SHEET_NAME}' at {utcnow_iso()}.")
    except Exception as e:
        st.error(f"Failed to write 'Scanned Result': {e}")
else:
    st.info("No symbols found in 'Universe'. Add seed tickers first, or wait for the next daily rebuild.")

st.markdown("---")
st.caption("Notes: pre/post **ignored** for stability; hourly refresh scheduled ~HH:05. This app feeds Stage B (1h strategy screener).")
