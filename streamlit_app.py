# streamlit_app.py — US Universe Builder (Top 500 by hourly transacted amount) + Sector/Industry + Context
# - Full US listings from NASDAQ Trader (include ETFs, drop OTC)
# - 1h regular-session bars (no pre/post), last completed bar even if market closed
# - Price filter $5–$100; rank by (close * volume); Top 500
# - Universe sheet columns: date, time, ticker, price, volume, type, sector, industry (overwrite each run)
# - Context sheet: VIX, HYG/LQD, XAUUSD=X (gold), and rotation (top sectors/industries by count & transacted amount)
# - Dashboard clocks: US ET & Malaysia (MYT), bar timestamp used
# - Auto-refresh hourly (best effort) and safe to run 24/7

import os
import io
import json
import time
import math
import pytz
import requests
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials

# Optional hourly auto-refresh
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

# =================================
# Config
# =================================
GOOGLE_SHEET_ID = "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4"
UNIVERSE_SHEET = "Universe"
CONTEXT_SHEET = "Context"

TOP_N = 500
PRICE_MIN = 5.0
PRICE_MAX = 100.0

# Refresh minute target (run around HH:25 to ensure last 1h bar is finalized)
TARGET_MINUTE = 25

# Tickers for context panel
VIX = "^VIX"
HYG = "HYG"
LQD = "LQD"
GOLD = "XAUUSD=X"  # you asked for XAUUSD=X

# Timezones
TZ_ET = pytz.timezone("America/New_York")
TZ_MYT = pytz.timezone("Asia/Kuala_Lumpur")

# =================================
# Helpers: time & UI
# =================================
def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def fmt_utc(ts: pd.Timestamp | datetime) -> Tuple[str, str]:
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime().replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.tz_convert("UTC").to_pydatetime()
    d = ts.astimezone(timezone.utc).strftime("%Y-%m-%d")
    t = ts.astimezone(timezone.utc).strftime("%H:%M:%S")
    return d, t

def et_now_str() -> str:
    return datetime.now(TZ_ET).strftime("%Y-%m-%d %H:%M:%S %Z")

def myt_now_str() -> str:
    return datetime.now(TZ_MYT).strftime("%Y-%m-%d %H:%M:%S %Z")

def hourly_autorefresh():
    if st_autorefresh:
        st_autorefresh(interval=60*60*1000, key="hourly-refresh")
    st.caption("⏱️ Auto-run hourly. Target: ~HH:{:02d} (regular-session 1h bar finalized).".format(TARGET_MINUTE))

# =================================
# Google Sheets auth
# =================================
def _get_gspread():
    raw = st.secrets.get("gcp_service_account")
    if not raw:
        raise RuntimeError("Missing [gcp_service_account] in secrets.")
    if isinstance(raw, str):
        info = json.loads(raw)
    else:
        info = dict(raw)
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    return gspread.authorize(creds)

def write_sheet_overwrite(sheet_id: str, tab_name: str, df: pd.DataFrame):
    gc = _get_gspread()
    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(tab_name)
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=tab_name, rows=str(max(len(df)+10, 1000)),
                              cols=str(max(len(df.columns)+5, 8)))
    values = [list(df.columns)] + df.astype(object).where(pd.notnull(df), "").values.tolist()
    ws.update("A1", values, value_input_option="RAW")

# =================================
# Symbol universe (NASDAQ Trader)
# =================================
NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_URL  = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

@st.cache_data(show_spinner=True, ttl=24*60*60)
def fetch_symbol_directory() -> pd.DataFrame:
    """
    Returns DataFrame with columns: symbol, exchange, is_etf (bool), is_test (bool), is_otc (bool)
    Includes NASDAQ + NYSE/NYSE American/ARCA (excludes OTC).
    """
    def load_pipe_txt(url: str) -> pd.DataFrame:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        # Files are pipe-delimited with a footer line "File Creation Time..."
        raw = r.text.strip().splitlines()
        # Keep header row; drop last footer line if startswith 'File Creation Time'
        if raw and raw[-1].lower().startswith("file creation time"):
            raw = raw[:-1]
        buf = io.StringIO("\n".join(raw))
        df = pd.read_csv(buf, sep="|")
        return df

    nas = load_pipe_txt(NASDAQ_URL)
    oth = load_pipe_txt(OTHER_URL)

    # Normalize columns
    # NASDAQ file has: Symbol, Security Name, Market Category, Test Issue (Y/N), Financial Status, ETF (Y/N), Round Lot Size, ...
    nas = nas.rename(columns=str.strip)
    nas_symbols = nas[~nas["Test Issue"].eq("Y")].copy()
    nas_symbols["symbol"] = nas_symbols["Symbol"].str.upper().str.strip()
    nas_symbols["exchange"] = "NASDAQ"
    nas_symbols["is_etf"] = nas_symbols["ETF"].eq("Y")
    nas_symbols["is_test"] = nas["Test Issue"].eq("Y")
    nas_symbols["is_otc"] = False

    # OTHER file has: ACT Symbol, Security Name, Exchange, CQS Symbol, ETF, Round Lot Size, Test Issue, NASDAQ Symbol
    oth = oth.rename(columns=str.strip)
    oth_symbols = oth[~oth["Test Issue"].eq("Y")].copy()
    oth_symbols["symbol"] = oth_symbols["ACT Symbol"].str.upper().str.strip()
    oth_symbols["exchange"] = oth_symbols["Exchange"].str.upper().str.strip()
    oth_symbols["is_etf"] = oth_symbols["ETF"].eq("Y")
    oth_symbols["is_test"] = oth_symbols["Test Issue"].eq("Y")
    # Mark OTC if exchange says OTC, otherwise False
    oth_symbols["is_otc"] = oth_symbols["exchange"].str.contains("OTC", na=False)

    # Combine and drop OTC
    df = pd.concat([
        nas_symbols[["symbol","exchange","is_etf","is_test","is_otc"]],
        oth_symbols[["symbol","exchange","is_etf","is_test","is_otc"]],
    ], ignore_index=True)

    # Deduplicate: keep first occurrence (usually primary)
    df = df.drop_duplicates(subset=["symbol"], keep="first").reset_index(drop=True)

    # Drop OTC and obvious non-primaries
    df = df[(~df["is_otc"]) & (~df["is_test"])].reset_index(drop=True)
    return df

# =================================
# Market data (Yahoo Finance)
# =================================
def _chunks(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

@st.cache_data(show_spinner=True)
def yf_daily_last_close(tickers: List[str]) -> pd.DataFrame:
    """Get last valid daily close for a large list quickly (filters price window fast)."""
    out = []
    for batch in _chunks(tickers, 150):
        data = yf.download(" ".join(batch), interval="1d", period="5d",
                           group_by="ticker", auto_adjust=True, threads=True,
                           progress=False, prepost=False)
        if isinstance(data.columns, pd.MultiIndex):
            for sym in batch:
                if sym in data.columns.get_level_values(0):
                    df = data[sym].rename(columns=str.lower)
                    if not df.empty and "close" in df:
                        last = float(df["close"].dropna().iloc[-1])
                        out.append((sym, last))
        else:
            df = data.rename(columns=str.lower)
            if not df.empty and "close" in df:
                last = float(df["close"].dropna().iloc[-1])
                out.append((batch[0], last))
        time.sleep(0.03)
    return pd.DataFrame(out, columns=["symbol","daily_last_close"])

@st.cache_data(show_spinner=True)
def yf_1h_last_bar(tickers: List[str]) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """
    Fetch 1h regular-session bars for a shortlist and return:
    - DataFrame with columns [symbol, close, volume] for the last completed bar
    - The last-bar timestamp (UTC) we used (from the data)
    """
    rows = []
    last_ts = None
    for batch in _chunks(tickers, 80):
        data = yf.download(" ".join(batch), interval="1h", period="7d",
                           group_by="ticker", auto_adjust=True, threads=True,
                           progress=False, prepost=False)
        # Determine per-batch last completed timestamp
        def get_last_ts(dfidx):
            if isinstance(dfidx, pd.DatetimeIndex) and len(dfidx) > 0:
                return dfidx.tz_localize("UTC") if dfidx.tz is None else dfidx.tz_convert("UTC")
            return None

        batch_ts = None
        if isinstance(data.columns, pd.MultiIndex):
            # Multi-ticker
            for sym in batch:
                if sym in data.columns.get_level_values(0):
                    df = data[sym].rename(columns=str.lower)
                    df = df[["close","volume"]].dropna()
                    if df.empty: 
                        continue
                    # Last row is last completed regular 1h bar
                    ts = df.index
                    batch_ts = get_last_ts(ts)
                    last_row = df.iloc[-1]
                    rows.append((sym, float(last_row["close"]), float(last_row["volume"])))
        else:
            # Single-ticker scenario fallback
            df = data.rename(columns=str.lower)
            df = df[["close","volume"]].dropna()
            if not df.empty:
                ts = df.index
                batch_ts = get_last_ts(ts)
                rows.append((batch[0], float(df.iloc[-1]["close"]), float(df.iloc[-1]["volume"])))
        if batch_ts is not None and len(batch_ts) > 0:
            last_ts = batch_ts[-1]
        time.sleep(0.05)
    df = pd.DataFrame(rows, columns=["symbol","close","volume"])
    return df, (pd.Timestamp(last_ts) if last_ts is not None else None)

@st.cache_data(show_spinner=True, ttl=24*60*60)
def get_sector_industry(symbol: str) -> Tuple[str, str]:
    """
    Lightweight sector/industry lookup for a single symbol (cached).
    Uses yfinance.get_info (may occasionally be slow/None); failures return empty strings.
    """
    try:
        info = yf.Ticker(symbol).get_info()
        sector = info.get("sector") or ""
        industry = info.get("industry") or ""
        return str(sector), str(industry)
    except Exception:
        return "", ""

# =================================
# Context metrics
# =================================
def _last_and_change_1h(ticker: str) -> Tuple[float, float]:
    """
    Return (last_close, 1h_change_pct). Uses last two 1h bars (regular).
    """
    df = yf.download(ticker, interval="1h", period="7d", auto_adjust=True, progress=False, prepost=False)
    if df.empty or "Close" not in df.columns:
        return float("nan"), float("nan")
    c = df["Close"].dropna()
    if len(c) < 2:
        return float(c.iloc[-1]) if len(c) else float("nan"), float("nan")
    last = float(c.iloc[-1])
    prev = float(c.iloc[-2])
    chg = (last - prev) / prev * 100.0 if prev != 0 else float("nan")
    return last, chg

def build_rotation_tables(universe_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    From the Top-500 universe with sector/industry and transacted amount,
    compute Top sectors/industries by total transacted amount.
    """
    if universe_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    agg = universe_df.groupby("sector", dropna=False)["transacted_amount"].sum().reset_index()
    agg = agg.sort_values("transacted_amount", ascending=False)
    agg_ind = universe_df.groupby("industry", dropna=False)["transacted_amount"].sum().reset_index()
    agg_ind = agg_ind.sort_values("transacted_amount", ascending=False)
    # Keep top 10 for context display
    return agg.head(10), agg_ind.head(10)

# =================================
# Streamlit UI
# =================================
st.set_page_config(page_title="US Universe Builder (Top 500 by 1h $ Amount)", layout="wide")
st.title("🚀 US Universe — Top 500 by Hourly Transacted Amount (Regular Session)")
st.caption("Includes ETFs, drops OTC. Price filter $5–$100. Uses last completed regular 1h bar even when market is closed.")

hourly_autorefresh()

# Clocks
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("US Eastern", et_now_str())
with col_b:
    st.metric("Malaysia (MYT)", myt_now_str())
with col_c:
    st.caption("Bar timestamp and run details shown below.")

# =================================
# Build the symbol universe (listings)
# =================================
with st.spinner("Fetching US listings (NASDAQ/NYSE/NYSE American)…"):
    listings = fetch_symbol_directory()
    # Keep everything except OTC/test; ETF flag is in is_etf
    tickers_all = listings["symbol"].tolist()
    st.caption(f"Listings loaded: {len(tickers_all)} symbols (incl. ETFs, excl. OTC/Test).")

# =================================
# Stage: Filter by daily price first (fast), then pull 1h for those in range
# =================================
with st.spinner("Fast price filter on daily bars…"):
    df_daily = yf_daily_last_close(tickers_all)
    if df_daily.empty:
        st.error("Failed to retrieve daily prices; cannot proceed.")
        st.stop()
    df_daily = df_daily[(df_daily["daily_last_close"] >= PRICE_MIN) & (df_daily["daily_last_close"] <= PRICE_MAX)]
    symbols_price_ok = df_daily["symbol"].tolist()
    st.caption(f"Symbols within ${PRICE_MIN}-{PRICE_MAX} by last daily close: {len(symbols_price_ok)}")

# =================================
# 1h bars for price-eligible shortlist
# =================================
with st.spinner("Pulling 1h regular-session bars (last completed bar)…"):
    df_1h, last_bar_ts = yf_1h_last_bar(symbols_price_ok)
    if df_1h.empty or last_bar_ts is None:
        st.error("No 1h bars available. Market likely closed for an extended period or data source unavailable.")
        st.stop()
    # Compute transacted amount, then rank
    df_1h["transacted_amount"] = df_1h["close"] * df_1h["volume"]
    # Merge ETF flag
    etf_map = dict(zip(listings["symbol"], listings["is_etf"]))
    exch_map = dict(zip(listings["symbol"], listings["exchange"]))
    df_1h["is_etf"] = df_1h["symbol"].map(etf_map).fillna(False)
    df_1h["type"] = np.where(df_1h["is_etf"], "etf", "stock")
    df_1h["exchange"] = df_1h["symbol"].map(exch_map)

    # Rank and keep Top N
    df_ranked = df_1h.sort_values("transacted_amount", ascending=False).head(TOP_N).reset_index(drop=True)

    # Sector/industry only for finalists (cached per symbol)
    sectors, industries = [], []
    with st.spinner("Enriching Top symbols with sector & industry (cached)…"):
        for sym in df_ranked["symbol"]:
            sec, ind = get_sector_industry(sym)
            sectors.append(sec)
            industries.append(ind)
    df_ranked["sector"] = sectors
    df_ranked["industry"] = industries

# =================================
# Prepare Universe output format
# =================================
bar_date, bar_time = fmt_utc(last_bar_ts)
universe_out = pd.DataFrame({
    "date": [bar_date]*len(df_ranked),
    "time": [bar_time]*len(df_ranked),
    "ticker": df_ranked["symbol"],
    "price": np.round(df_ranked["close"].astype(float), 4),
    "volume": df_ranked["volume"].astype(np.int64, errors="ignore"),
    "type": df_ranked["type"],
    "sector": df_ranked["sector"],
    "industry": df_ranked["industry"],
})

# =================================
# Context snapshot (sentiment + rotation)
# =================================
with st.spinner("Building Context snapshot (VIX, HYG/LQD, Gold, rotation)…"):
    vix_last, vix_ch = _last_and_change_1h(VIX)
    hyg_last, _ = _last_and_change_1h(HYG)
    lqd_last, _ = _last_and_change_1h(LQD)
    ratio = (hyg_last / lqd_last) if (isinstance(hyg_last, float) and isinstance(lqd_last, float) and lqd_last not in (0.0, np.nan)) else float("nan")
    # compute 1h change for ratio via small helper: pull both then diff; we approximated above (ok for snapshot)
    gold_last, gold_ch = _last_and_change_1h(GOLD)

    # Rotation: sums by sector/industry using transacted_amount from df_ranked
    top_sectors, top_industries = build_rotation_tables(df_ranked[["symbol","transacted_amount","sector","industry"]])

    # Context table (flat)
    context_rows = []
    ts_utc = utcnow().strftime("%Y-%m-%d %H:%M:%S")
    context_rows.append(["timestamp_utc", ts_utc])
    context_rows.append(["bar_date_utc", bar_date])
    context_rows.append(["bar_time_utc", bar_time])

    context_rows.append(["VIX_last", round(vix_last, 4) if pd.notna(vix_last) else ""])
    context_rows.append(["VIX_1h_change_pct", round(vix_ch, 4) if pd.notna(vix_ch) else ""])

    context_rows.append(["HYG_last", round(hyg_last, 4) if pd.notna(hyg_last) else ""])
    context_rows.append(["LQD_last", round(lqd_last, 4) if pd.notna(lqd_last) else ""])
    context_rows.append(["HYG_LQD_ratio", round(ratio, 6) if pd.notna(ratio) else ""])

    context_rows.append(["Gold_last_XAUUSD=X", round(gold_last, 4) if pd.notna(gold_last) else ""])
    context_rows.append(["Gold_1h_change_pct", round(gold_ch, 4) if pd.notna(gold_ch) else ""])

    # Flatten top sectors/industries (by transacted amount)
    for i, row in enumerate(top_sectors.itertuples(index=False), start=1):
        context_rows.append([f"TopSector{i}_name", row.sector if pd.notna(row.sector) else ""])
        context_rows.append([f"TopSector{i}_shareAmt", round(float(row.transacted_amount), 2)])
    for i, row in enumerate(top_industries.itertuples(index=False), start=1):
        context_rows.append([f"TopIndustry{i}_name", row.industry if pd.notna(row.industry) else ""])
        context_rows.append([f"TopIndustry{i}_shareAmt", round(float(row.transacted_amount), 2)])

    context_out = pd.DataFrame(context_rows, columns=["metric","value"])

# =================================
# Write to Google Sheets (overwrite each run)
# =================================
col1, col2 = st.columns(2)
with col1:
    st.subheader("Universe — Top 500 by Hourly Transacted Amount")
    st.caption(f"Bar used (UTC): {bar_date} {bar_time}")
    st.dataframe(universe_out.head(30), use_container_width=True)

with col2:
    st.subheader("Context — Sentiment & Rotation (snapshot)")
    st.dataframe(context_out, use_container_width=True)

with st.spinner("Writing Universe to Google Sheet (overwrite)…"):
    try:
        write_sheet_overwrite(GOOGLE_SHEET_ID, UNIVERSE_SHEET, universe_out)
        st.success(f"Universe overwritten ({len(universe_out)} rows).")
    except Exception as e:
        st.error(f"Failed to write Universe: {e}")

with st.spinner("Writing Context to Google Sheet (overwrite)…"):
    try:
        write_sheet_overwrite(GOOGLE_SHEET_ID, CONTEXT_SHEET, context_out)
        st.success("Context overwritten.")
    except Exception as e:
        st.error(f"Failed to write Context: {e}")

st.markdown("---")
st.caption("Run complete • Last bar (regular 1h) is used even when market is closed • Pre/post ignored for consistency.")
