# streamlit_app.py — Dynamic Universe (Yahoo-based) → Google Sheet
# Purpose: build Top-N universe by $-flow share and write to Sheet (replace old).
# Sheet schema: 2 columns — "Timestamp (ET)" and "Ticker", one ticker per row.

import os, json, time, re
from datetime import datetime, timezone
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import pytz
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Dynamic Tickers — Universe Builder", layout="centered")
st.title("🗂️ Dynamic Tickers — Universe Builder")

# -----------------------
# Config / Secrets
# -----------------------
GOOGLE_SHEET_ID = st.secrets.get("GOOGLE_SHEET_ID") or os.getenv("GOOGLE_SHEET_ID", "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4")
GOOGLE_SHEET_NAME = st.secrets.get("GOOGLE_SHEET_NAME") or os.getenv("GOOGLE_SHEET_NAME", "Universe")

ET_TZ = pytz.timezone("US/Eastern")

# Support BOTH secret styles:
# 1) st.secrets["gcp_service_account"] as a dict (Streamlit recommended)
# 2) st.secrets["GCP_SERVICE_ACCOUNT"] / env as a JSON string
def _load_sa_dict():
    # style 1: dict already parsed
    if "gcp_service_account" in st.secrets:
        sa = dict(st.secrets["gcp_service_account"])
        # If private_key accidentally stored as array of lines, join
        if isinstance(sa.get("private_key"), list):
            sa["private_key"] = "\n".join(sa["private_key"])
        return sa
    # style 2: JSON string
    raw = st.secrets.get("GCP_SERVICE_ACCOUNT") or os.getenv("GCP_SERVICE_ACCOUNT")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Repair case: raw newlines inside private key
        if "-----BEGIN PRIVATE KEY-----" in raw and "\\n" not in raw:
            block_start = raw.find("-----BEGIN PRIVATE KEY-----")
            block_end = raw.find("-----END PRIVATE KEY-----", block_start)
            if block_start != -1 and block_end != -1:
                block_end += len("-----END PRIVATE KEY-----")
                block = raw[block_start:block_end]
                fixed = block.replace("\r\n", "\n").replace("\n", "\\n")
                raw = raw.replace(block, fixed)
        return json.loads(raw)

def _now_et_str() -> str:
    return datetime.now(timezone.utc).astimezone(ET_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")

# -----------------------
# Google Sheets (gspread)
# -----------------------
@st.cache_resource(show_spinner=False)
def _gs_client():
    import gspread
    from google.oauth2.service_account import Credentials
    sa = _load_sa_dict()
    if not sa:
        raise RuntimeError("Missing Google service account secrets. Add either `gcp_service_account` (dict) or `GCP_SERVICE_ACCOUNT` (JSON).")
    creds = Credentials.from_service_account_info(
        sa,
        scopes=["https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"]
    )
    return gspread.authorize(creds)

def replace_universe_sheet(tickers: List[str]) -> Tuple[bool, str]:
    """Replace entire Universe sheet with header + rows of [Timestamp (ET), Ticker]."""
    import gspread
    client = _gs_client()
    sh = client.open_by_key(GOOGLE_SHEET_ID)
    try:
        ws = sh.worksheet(GOOGLE_SHEET_NAME)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=GOOGLE_SHEET_NAME, rows=1, cols=2)

    rows = [["Timestamp (ET)", "Ticker"]] + [[_now_et_str(), t] for t in tickers]
    ws.clear()
    # Use batch update to minimize API calls
    ws.update("A1", rows, value_input_option="USER_ENTERED")
    return True, f"Wrote {len(tickers)} tickers."

# -----------------------
# Symbol directory (NASDAQ + NYSE)
# -----------------------
NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_URL  = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

def _to_yahoo_symbol(sym: str) -> str:
    # Yahoo usually uses dash class (BRK.B -> BRK-B)
    return sym.replace(".", "-").upper()

@st.cache_data(ttl=60*60*6, show_spinner=False)
def download_symbol_files_qn_only(exclude_funds: bool = True, exclude_derivs: bool = True) -> List[str]:
    frames = []
    try:
        nasdaq = pd.read_csv(NASDAQ_URL, sep="|")
        nasdaq = nasdaq[nasdaq["Symbol"].notna() & (nasdaq["Symbol"] != "Symbol")]
        dfq = pd.DataFrame({"symbol": nasdaq["Symbol"].astype(str).str.upper().str.strip(),
                            "security_name": nasdaq.get("Security Name"),
                            "exchange": "Q",
                            "etf": nasdaq.get("ETF", "N"),
                            "test_issue": nasdaq.get("Test Issue", "N"),
                            "nextshares": nasdaq.get("NextShares", "N")})
        frames.append(dfq)
    except Exception:
        pass
    try:
        other = pd.read_csv(OTHER_URL, sep="|")
        other = other[other["ACT Symbol"].notna()]
        # Keep NYSE only
        other = other[other.get("Exchange", "").astype(str).str.upper().eq("N")]
        dfn = pd.DataFrame({"symbol": other["ACT Symbol"].astype(str).str.upper().str.strip(),
                            "security_name": other.get("Security Name"),
                            "exchange": "N",
                            "etf": other.get("ETF", "N"),
                            "test_issue": other.get("Test Issue", "N")})
        frames.append(dfn)
    except Exception:
        pass

    if not frames:
        return ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AMD","NFLX","QCOM"]

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["symbol"])
    if exclude_funds:
        df = df[(df["etf"].astype(str) != "Y") & (df["test_issue"].astype(str) != "Y")]
        if "nextshares" in df.columns:
            df = df[df["nextshares"].astype(str) != "Y"]

    if exclude_derivs and "security_name" in df.columns:
        pat = re.compile(r"(WARRANT|RIGHTS?|UNITS?|PREF|PREFERRED|NOTE|BOND|TRUST|FUND|ETF|ETN|DEPOSITARY|SPAC)", re.IGNORECASE)
        df = df[~df["security_name"].astype(str).str.contains(pat, na=False)]

    syms = sorted({_to_yahoo_symbol(s) for s in df["symbol"].dropna().unique().tolist()})
    return syms

# -----------------------
# Yahoo download (daily)
# -----------------------
@st.cache_data(ttl=60*30, show_spinner=True)
def fetch_daily(tickers: List[str], period: str = "2mo") -> Dict[str, pd.DataFrame]:
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = [_to_yahoo_symbol(t) for t in tickers]
    data: Dict[str, pd.DataFrame] = {}
    CHUNK = 200
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i:i+CHUNK]
        try:
            raw = yf.download(" ".join(chunk), interval="1d", period=period,
                              group_by="ticker", auto_adjust=False, progress=False, threads=False)
        except Exception:
            continue
        if len(chunk) == 1 or not isinstance(raw.columns, pd.MultiIndex):
            if raw is None or raw.empty:
                continue
            sub = raw.rename(columns={c: c.split(" ")[0] for c in raw.columns})
            sub = sub[["Open","High","Low","Close","Volume"]].rename(columns=str.lower).dropna()
            # yfinance daily index is date; treat as ET midnight (good enough for averages)
            data[chunk[0]] = sub
        else:
            for t in chunk:
                try:
                    sub = raw[t]
                except Exception:
                    continue
                sub = sub.rename(columns={c: c.split(" ")[0] for c in sub.columns})
                try:
                    sub = sub[["Open","High","Low","Close","Volume"]].rename(columns=str.lower).dropna()
                except Exception:
                    continue
                data[t] = sub
    return data

# -----------------------
# Universe builder (flow-ranked)
# -----------------------
@st.cache_data(ttl=60*60, show_spinner=True)
def build_universe_flow_rank(
    n_top:int = 800,
    min_price:float = 5.0,
    max_price:float = 100.0,
    exclude_top_volume_pct:float = 1.5,
    gentle_momentum:bool = True,
    daily_bars:int = 5,
    max_pool:int = 6000
) -> Tuple[List[str], Dict]:
    """Return top tickers by $-flow share with filters."""
    pool = download_symbol_files_qn_only()
    pool = pool[:max_pool]
    if not pool:
        return [], {"evaluated": 0, "kept": 0, "after_excl": 0}

    daily = fetch_daily(pool, period="2mo")
    candidates: List[Tuple[str, float, float, float]] = []  # (sym, adv, last_vol, last_close)

    for sym, df in daily.items():
        if df is None or df.empty:
            continue
        if len(df) < max(daily_bars + 1, 21):  # need yesterday + EMA20 horizon
            continue

        last_close = float(df["close"].iloc[-1])
        prev_close = float(df["close"].iloc[-2])
        if not (min_price <= last_close <= max_price):
            continue

        recent = df.tail(daily_bars)
        adv = float((recent["close"] * recent["volume"]).mean())
        if not np.isfinite(adv) or adv <= 0:
            continue

        if gentle_momentum:
            ema20 = float(df["close"].ewm(span=20, adjust=False).mean().iloc[-1])
            if not ((last_close >= prev_close) or (last_close >= ema20)):
                continue

        last_vol = float(df["volume"].iloc[-1])
        candidates.append((sym, adv, last_vol, last_close))

    kept = len(candidates)
    if kept == 0:
        return [], {"evaluated": len(daily), "kept": 0, "after_excl": 0}

    # Exclude top X% by raw volume (noise control)
    after_excl = kept
    if exclude_top_volume_pct and exclude_top_volume_pct > 0 and kept >= 50:
        vols = np.array([lv for _, _, lv, _ in candidates], dtype=float)
        cutoff = float(np.quantile(vols, 1.0 - (exclude_top_volume_pct / 100.0)))
        candidates = [row for row in candidates if row[2] < cutoff]
        after_excl = len(candidates)

    if not candidates:
        return [], {"evaluated": len(daily), "kept": kept, "after_excl": 0}

    total_adv = float(sum(adv for _, adv, _lv, _px in candidates)) or 1.0
    scored = [(sym, adv / total_adv, adv) for (sym, adv, _lv, _px) in candidates]
    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)

    # Fail-soft: if too few, relax momentum once
    if len(scored) < max(100, int(0.5 * n_top)) and gentle_momentum:
        candidates2 = []
        for sym, df in daily.items():
            if df is None or df.empty or len(df) < daily_bars + 1:
                continue
            last_close = float(df["close"].iloc[-1])
            if not (min_price <= last_close <= max_price):
                continue
            adv = float((df.tail(daily_bars)["close"] * df.tail(daily_bars)["volume"]).mean())
            if not np.isfinite(adv) or adv <= 0:
                continue
            last_vol = float(df["volume"].iloc[-1])
            candidates2.append((sym, adv, last_vol, last_close))
        if candidates2:
            if exclude_top_volume_pct and exclude_top_volume_pct > 0 and len(candidates2) >= 50:
                vols2 = np.array([lv for _, _, lv, _ in candidates2], dtype=float)
                cutoff2 = float(np.quantile(vols2, 1.0 - (exclude_top_volume_pct / 100.0)))
                candidates2 = [row for row in candidates2 if row[2] < cutoff2]
            total_adv2 = float(sum(adv for _, adv, _lv, _px in candidates2)) or 1.0
            scored = [(sym, adv / total_adv2, adv) for (sym, adv, _lv, _px) in candidates2]
            scored.sort(key=lambda x: (x[1], x[2]), reverse=True)

    top_syms = [sym for sym, _share, _adv in scored[:n_top]]
    stats = {"evaluated": len(daily), "kept": kept, "after_excl": after_excl}
    return top_syms, stats

# -----------------------
# UI — minimal controls
# -----------------------
with st.expander("Criteria (adjust then click Run)", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        n_top = st.slider("Top N (by $-flow share)", 100, 1200, 800, step=50)
        daily_bars = st.slider("Daily bars for $-flow avg", 3, 10, 5)
        gentle_momentum = st.checkbox("Gentle momentum (last ≥ prev OR ≥ EMA20)", value=True)
    with c2:
        min_price = st.number_input("Min price ($)", value=5.0, step=0.5)
        max_price = st.number_input("Max price ($)", value=100.0, step=1.0)
        exclude_top_volume_pct = st.number_input("Exclude top volume (%)", value=1.5, min_value=0.0, max_value=25.0, step=0.5)

run_it = st.button("🚀 Run Screener & Update Sheet", type="primary")

# -----------------------
# Run + write
# -----------------------
if run_it:
    with st.spinner("Building universe…"):
        syms, stats = build_universe_flow_rank(
            n_top=int(n_top),
            min_price=float(min_price),
            max_price=float(max_price),
            exclude_top_volume_pct=float(exclude_top_volume_pct),
            gentle_momentum=bool(gentle_momentum),
            daily_bars=int(daily_bars),
            max_pool=6000,
        )
    if not syms:
        st.error("No tickers found. Try relaxing filters or reducing exclusions.")
    else:
        st.success(f"Selected {len(syms)} tickers. (Evaluated: {stats.get('evaluated',0)}, kept: {stats.get('kept',0)}, after excl: {stats.get('after_excl',0)})")
        st.code(", ".join(syms[:100]), language="text")
        try:
            ok, msg = replace_universe_sheet(syms)
            if ok:
                st.success(f"Google Sheet updated: {msg}")
            else:
                st.error("Failed to update Google Sheet.")
        except Exception as e:
            st.error(f"Google Sheet write error: {e}")

st.caption("Writes to Google Sheet → Universe (replaces content). Data: Yahoo Finance daily bars.")
