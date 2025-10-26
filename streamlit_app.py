# streamlit_app.py — US Universe Builder (Top 500 by 1h $ Amount) + Sector/Industry + Context
# - Full US listings (NASDAQ/NYSE/NYSE American) from NASDAQ Trader
# - Include ETFs, drop OTC
# - Use last completed regular 1h bar (even when market is closed)
# - Filter price $5–$100; rank by (close * volume); Top 500
# - Universe sheet columns: date, time, ticker, price, volume, type, sector, industry (overwrite)
# - Context sheet: VIX, HYG/LQD, XAUUSD=X, rotation (by transacted amount)
# - Dashboard: ET & MYT clocks, bar timestamp used
# - Auto-refresh hourly; robust to empty frames and partial data

import os
import io
import json
import time
import pytz
import requests
from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

# =========================
# Config
# =========================
GOOGLE_SHEET_ID = "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4"
UNIVERSE_SHEET = "Universe"
CONTEXT_SHEET = "Context"

TOP_N = 500
PRICE_MIN = 5.0
PRICE_MAX = 100.0

TARGET_MINUTE = 25  # aim to run around HH:25

VIX = "^VIX"
HYG = "HYG"
LQD = "LQD"
GOLD = "XAUUSD=X"  # per your choice

TZ_ET = pytz.timezone("America/New_York")
TZ_MYT = pytz.timezone("Asia/Kuala_Lumpur")

# =========================
# Time helpers & UI
# =========================
def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def fmt_utc(ts: pd.Timestamp | datetime) -> Tuple[str, str]:
    if isinstance(ts, pd.Timestamp):
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        ts = ts.to_pydatetime()
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
    st.caption(f"⏱️ Auto-run hourly. Target: ~HH:{TARGET_MINUTE:02d} (uses last completed regular 1h bar).")

# =========================
# Google Sheets
# =========================
def _get_gspread():
    raw = st.secrets.get("gcp_service_account")
    if not raw:
        raise RuntimeError("Missing [gcp_service_account] in secrets.")
    info = json.loads(raw) if isinstance(raw, str) else dict(raw)
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
        ws = sh.add_worksheet(
            title=tab_name,
            rows=str(max(len(df) + 10, 1000)),
            cols=str(max(len(df.columns) + 5, 8)),
        )
    values = [list(df.columns)] + df.astype(object).where(pd.notnull(df), "").values.tolist()
    ws.update("A1", values, value_input_option="RAW")

# =========================
# Symbol universe (NASDAQ Trader)
# =========================
NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_URL  = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

@st.cache_data(show_spinner=True, ttl=24*60*60)
def fetch_symbol_directory() -> pd.DataFrame:
    """Return DataFrame: symbol, exchange, is_etf, is_test, is_otc; drop OTC/Test."""
    def load_pipe_txt(url: str) -> pd.DataFrame:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        lines = r.text.strip().splitlines()
        if lines and lines[-1].lower().startswith("file creation time"):
            lines = lines[:-1]
        buf = io.StringIO("\n".join(lines))
        return pd.read_csv(buf, sep="|")

    nas = load_pipe_txt(NASDAQ_URL).rename(columns=str.strip)
    oth = load_pipe_txt(OTHER_URL).rename(columns=str.strip)

    nas_symbols = nas[~nas["Test Issue"].eq("Y")].copy()
    nas_symbols["symbol"] = nas_symbols["Symbol"].str.upper().str.strip()
    nas_symbols["exchange"] = "NASDAQ"
    nas_symbols["is_etf"] = nas_symbols["ETF"].eq("Y")
    nas_symbols["is_test"] = nas["Test Issue"].eq("Y")
    nas_symbols["is_otc"] = False

    oth_symbols = oth[~oth["Test Issue"].eq("Y")].copy()
    oth_symbols["symbol"] = oth_symbols["ACT Symbol"].str.upper().str.strip()
    oth_symbols["exchange"] = oth_symbols["Exchange"].str.upper().str.strip()
    oth_symbols["is_etf"] = oth_symbols["ETF"].eq("Y")
    oth_symbols["is_test"] = oth_symbols["Test Issue"].eq("Y")
    oth_symbols["is_otc"] = oth_symbols["exchange"].str.contains("OTC", na=False)

    df = pd.concat(
        [nas_symbols[["symbol","exchange","is_etf","is_test","is_otc"]],
         oth_symbols[["symbol","exchange","is_etf","is_test","is_otc"]]],
        ignore_index=True
    ).drop_duplicates(subset=["symbol"], keep="first")

    df = df[(~df["is_otc"]) & (~df["is_test"])].reset_index(drop=True)
    return df

# =========================
# Yahoo helpers (hardened)
# =========================
def _chunks(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

@st.cache_data(show_spinner=True)
def yf_daily_last_close(tickers: List[str]) -> pd.DataFrame:
    """Get last valid daily close for many symbols; robust to empties."""
    out = []
    for batch in _chunks(tickers, 150):
        try:
            data = yf.download(
                " ".join(batch),
                interval="1d",
                period="5d",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
                prepost=False,
            )
        except Exception:
            time.sleep(0.05)
            continue

        if data is None or len(data) == 0:
            continue

        if isinstance(data.columns, pd.MultiIndex):
            syms_present = sorted(set(data.columns.get_level_values(0)))
            for sym in syms_present:
                try:
                    df = data[sym]
                except KeyError:
                    continue
                if df is None or df.empty:
                    continue
                df = df.rename(columns=str.lower)
                if "close" not in df.columns:
                    continue
                s = df["close"].dropna()
                if s.empty:
                    continue
                last = float(s.iloc[-1])
                out.append((sym, last))
        else:
            df = data.rename(columns=str.lower)
            if "close" in df.columns:
                s = df["close"].dropna()
                if not s.empty:
                    sym = batch[0]
                    last = float(s.iloc[-1])
                    out.append((sym, last))
        time.sleep(0.03)
    return pd.DataFrame(out, columns=["symbol", "daily_last_close"])

@st.cache_data(show_spinner=True)
def yf_1h_last_bar(tickers: List[str]) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """
    Fetch last completed 1h regular-session bar for symbols; robust to empties.
    Returns (df[symbol, close, volume], last_bar_timestamp_utc)
    """
    rows = []
    last_ts_global = None

    def normalize_ts(idx: pd.DatetimeIndex | None):
        if idx is None or len(idx) == 0:
            return None
        if idx.tz is None:
            return idx.tz_localize("UTC")
        return idx.tz_convert("UTC")

    for batch in _chunks(tickers, 80):
        try:
            data = yf.download(
                " ".join(batch),
                interval="1h",
                period="7d",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
                prepost=False,  # regular session only
            )
        except Exception:
            time.sleep(0.05)
            continue

        if data is None or len(data) == 0:
            continue

        if isinstance(data.columns, pd.MultiIndex):
            syms_present = sorted(set(data.columns.get_level_values(0)))
            for sym in syms_present:
                try:
                    df = data[sym]
                except KeyError:
                    continue
                if df is None or df.empty:
                    continue
                df = df.rename(columns=str.lower)
                if not {"close","volume"} <= set(df.columns):
                    continue
                df = df[["close","volume"]].dropna()
                if df.empty:
                    continue
                ts = normalize_ts(df.index)
                if ts is None:
                    continue
                last_row = df.iloc[-1]
                rows.append((sym, float(last_row["close"]), float(last_row["volume"])))
                if last_ts_global is None or ts[-1] > last_ts_global:
                    last_ts_global = ts[-1]
        else:
            df = data.rename(columns=str.lower)
            if {"close","volume"} <= set(df.columns):
                df = df[["close","volume"]].dropna()
                if not df.empty:
                    ts = normalize_ts(df.index)
                    if ts is not None:
                        last_row = df.iloc[-1]
                        rows.append((batch[0], float(last_row["close"]), float(last_row["volume"])))
                        if last_ts_global is None or ts[-1] > last_ts_global:
                            last_ts_global = ts[-1]
        time.sleep(0.05)

    df = pd.DataFrame(rows, columns=["symbol","close","volume"])
    return df, (pd.Timestamp(last_ts_global) if last_ts_global is not None else None)

@st.cache_data(show_spinner=True, ttl=24*60*60)
def get_sector_industry(symbol: str) -> Tuple[str, str]:
    """Sector/industry lookup (cached). Fail-safe returns empty strings."""
    try:
        info = yf.Ticker(symbol).get_info()
        return str(info.get("sector") or ""), str(info.get("industry") or "")
    except Exception:
        return "", ""

# =========================
# Context metrics
# =========================
def _last_and_change_1h(ticker: str) -> Tuple[float, float]:
    """Return (last_close, 1h_change_pct) using last two 1h regular bars."""
    df = yf.download(ticker, interval="1h", period="7d", auto_adjust=True, progress=False, prepost=False)
    if df is None or df.empty or "Close" not in df.columns:
        return float("nan"), float("nan")
    c = df["Close"].dropna()
    if c.empty:
        return float("nan"), float("nan")
    if len(c) == 1:
        return float(c.iloc[-1]), float("nan")
    last = float(c.iloc[-1])
    prev = float(c.iloc[-2])
    chg = (last - prev) / prev * 100.0 if prev else float("nan")
    return last, chg

def build_rotation_tables(universe_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Top sectors/industries by total transacted amount."""
    if universe_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    agg_s = universe_df.groupby("sector", dropna=False)["transacted_amount"].sum().reset_index()
    agg_i = universe_df.groupby("industry", dropna=False)["transacted_amount"].sum().reset_index()
    agg_s = agg_s.sort_values("transacted_amount", ascending=False).head(10)
    agg_i = agg_i.sort_values("transacted_amount", ascending=False).head(10)
    return agg_s, agg_i

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="US Universe Builder (Top 500 by 1h $ Amount)", layout="wide")
st.title("🚀 US Universe — Top 500 by Hourly Transacted Amount (Regular Session)")
st.caption("Includes ETFs, drops OTC • Price filter $5–$100 • Uses last completed regular 1h bar, even when market is closed.")

hourly_autorefresh()

# Clocks
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("US Eastern", et_now_str())
with c2:
    st.metric("Malaysia (MYT)", myt_now_str())
with c3:
    st.caption("Bar timestamp used is shown below.")

# =========================
# Load US listings (seed universe)
# =========================
with st.spinner("Fetching US listings (NASDAQ/NYSE/NYSE American)…"):
    listings = fetch_symbol_directory()
    tickers_all = listings["symbol"].tolist()
    st.caption(f"Listings loaded: {len(tickers_all)} symbols (incl. ETFs, excl. OTC/Test).")

# =========================
# Fast daily filter ($5–$100)
# =========================
with st.spinner("Fast price filter on daily bars…"):
    df_daily = yf_daily_last_close(tickers_all)
    if df_daily.empty:
        st.error("Failed to retrieve daily prices; cannot proceed.")
        st.stop()

    df_daily = df_daily[(df_daily["daily_last_close"] >= PRICE_MIN) & (df_daily["daily_last_close"] <= PRICE_MAX)]
    symbols_price_ok = df_daily["symbol"].tolist()
    st.caption(f"Symbols within ${PRICE_MIN}-{PRICE_MAX} by last daily close: {len(symbols_price_ok)}")

    if not symbols_price_ok:
        st.error("No symbols passed the $5–$100 daily price filter. Try widening the range or wait for next session.")
        st.stop()

# =========================
# Pull 1h bars for shortlist; rank Top 500 by $ amount
# =========================
with st.spinner("Pulling 1h regular-session bars (last completed bar)…"):
    df_1h, last_bar_ts = yf_1h_last_bar(symbols_price_ok)
    if df_1h.empty or last_bar_ts is None:
        st.error("No 1h bars available. Market likely closed for an extended period or data is unavailable.")
        st.stop()

    # Compute transacted amount
    df_1h["transacted_amount"] = df_1h["close"] * df_1h["volume"]

    # Map ETF flag & exchange
    etf_map = dict(zip(listings["symbol"], listings["is_etf"]))
    exch_map = dict(zip(listings["symbol"], listings["exchange"]))
    df_1h["is_etf"] = df_1h["symbol"].map(etf_map).fillna(False)
    df_1h["type"] = np.where(df_1h["is_etf"], "etf", "stock")
    df_1h["exchange"] = df_1h["symbol"].map(exch_map)

    # Rank and keep Top N
    df_ranked = df_1h.sort_values("transacted_amount", ascending=False).head(TOP_N).reset_index(drop=True)

    # Enrich with sector/industry (cached per symbol)
    sectors, industries = [], []
    with st.spinner("Enriching Top symbols with sector & industry (cached)…"):
        for sym in df_ranked["symbol"]:
            sec, ind = get_sector_industry(sym)
            sectors.append(sec)
            industries.append(ind)
    df_ranked["sector"] = sectors
    df_ranked["industry"] = industries

# =========================
# Prepare Universe output
# =========================
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

# =========================
# Context snapshot (sentiment + rotation)
# =========================
with st.spinner("Building Context snapshot (VIX, HYG/LQD, Gold, rotation)…"):
    vix_last, vix_ch = _last_and_change_1h(VIX)
    hyg_last, _ = _last_and_change_1h(HYG)
    lqd_last, _ = _last_and_change_1h(LQD)
    ratio = float("nan")
    if isinstance(hyg_last, float) and isinstance(lqd_last, float) and (lqd_last not in (0.0, np.nan)):
        ratio = hyg_last / lqd_last
    gold_last, gold_ch = _last_and_change_1h(GOLD)

    top_sectors, top_industries = build_rotation_tables(
        df_ranked[["symbol","transacted_amount","sector","industry"]]
    )

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

    for i, row in enumerate(top_sectors.itertuples(index=False), start=1):
        context_rows.append([f"TopSector{i}_name", row.sector if pd.notna(row.sector) else ""])
        context_rows.append([f"TopSector{i}_shareAmt", round(float(row.transacted_amount), 2)])
    for i, row in enumerate(top_industries.itertuples(index=False), start=1):
        context_rows.append([f"TopIndustry{i}_name", row.industry if pd.notna(row.industry) else ""])
        context_rows.append([f"TopIndustry{i}_shareAmt", round(float(row.transacted_amount), 2)])

    context_out = pd.DataFrame(context_rows, columns=["metric","value"])

# =========================
# Write to Google Sheets
# =========================
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
st.caption("Run complete • Regular-session 1h bar used (ignores pre/post). When market is closed, uses the most recent completed regular bar.")
