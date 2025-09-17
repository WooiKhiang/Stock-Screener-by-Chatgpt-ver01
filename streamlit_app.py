# streamlit_app.py — Dynamic Universe (scan-on-click; fast boot)
# Writes 2 columns to Google Sheet "Universe": [Timestamp (ET), Ticker]

import os, json, re, time
from datetime import datetime, timezone
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import pytz
import streamlit as st

# ----------------------------------------
# Fast page load: no heavy imports here.
# ----------------------------------------

st.set_page_config(page_title="Dynamic Tickers — Universe Builder", layout="centered")
st.title("🗂️ Dynamic Tickers — Universe Builder")

# --- Sheet config (hardcoded for speed; creds still via secrets)
GOOGLE_SHEET_ID = "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4"
GOOGLE_SHEET_NAME = "Universe"
GOOGLE_SHEET_ID = st.secrets.get("GOOGLE_SHEET_ID", GOOGLE_SHEET_ID)
GOOGLE_SHEET_NAME = st.secrets.get("GOOGLE_SHEET_NAME", GOOGLE_SHEET_NAME)

ET_TZ = pytz.timezone("US/Eastern")
def now_et_str() -> str:
    return datetime.now(timezone.utc).astimezone(ET_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")

# --- Tiny helpers (no heavy deps)
NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_URL  = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

def to_yahoo_symbol(sym: str) -> str:
    return sym.replace(".", "-").upper()

def require_yfinance():
    # Import yfinance only when scanning is requested
    try:
        import yfinance as yf  # noqa
        return yf
    except Exception:
        st.error("`yfinance` is not installed. Ensure your branch has a `requirements.txt` at repo root with `yfinance==0.2.40`, then redeploy.")
        return None

def require_gspread_client():
    # Import google libs only when writing is requested
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except Exception:
        st.error("Google libs missing. Ensure `gspread==6.1.2` and `google-auth==2.33.0` are in requirements.txt, then redeploy.")
        return None, None

    # Load service account from secrets (dict or one-line JSON)
    if "gcp_service_account" in st.secrets:
        sa = dict(st.secrets["gcp_service_account"])
        if isinstance(sa.get("private_key"), list):
            sa["private_key"] = "\n".join(sa["private_key"])
    else:
        raw = st.secrets.get("GCP_SERVICE_ACCOUNT") or os.getenv("GCP_SERVICE_ACCOUNT")
        if not raw:
            st.error("Missing Google credentials in secrets. Add [gcp_service_account] or GCP_SERVICE_ACCOUNT.")
            return None, None
        try:
            sa = json.loads(raw)
        except json.JSONDecodeError:
            # self-heal stray newlines in private_key
            if "-----BEGIN PRIVATE KEY-----" in raw and "\\n" not in raw:
                s = raw.find("-----BEGIN PRIVATE KEY-----"); e = raw.find("-----END PRIVATE KEY-----", s)
                if s != -1 and e != -1:
                    e += len("-----END PRIVATE KEY-----")
                    raw = raw.replace(raw[s:e], raw[s:e].replace("\r\n","\n").replace("\n","\\n"))
            sa = json.loads(raw)

    creds = Credentials.from_service_account_info(
        sa,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    return gspread.authorize(creds), gspread

@st.cache_data(ttl=60*60*6, show_spinner=False)
def download_symbol_files_qn_only(exclude_funds=True, exclude_derivs=True) -> List[str]:
    frames = []
    # NASDAQ
    try:
        nasdaq = pd.read_csv(NASDAQ_URL, sep="|")
        nasdaq = nasdaq[nasdaq["Symbol"].notna() & (nasdaq["Symbol"] != "Symbol")]
        dfq = pd.DataFrame({
            "symbol": nasdaq["Symbol"].astype(str).str.upper().str.strip(),
            "security_name": nasdaq.get("Security Name"),
            "exchange": "Q",
            "etf": nasdaq.get("ETF","N"),
            "test_issue": nasdaq.get("Test Issue","N"),
            "nextshares": nasdaq.get("NextShares","N"),
        })
        frames.append(dfq)
    except Exception:
        pass
    # NYSE-only from otherlisted
    try:
        other = pd.read_csv(OTHER_URL, sep="|")
        other = other[other["ACT Symbol"].notna()]
        other = other[other.get("Exchange","").astype(str).str.upper().eq("N")]
        dfn = pd.DataFrame({
            "symbol": other["ACT Symbol"].astype(str).str.upper().str.strip(),
            "security_name": other.get("Security Name"),
            "exchange": "N",
            "etf": other.get("ETF","N"),
            "test_issue": other.get("Test Issue","N"),
        })
        frames.append(dfn)
    except Exception:
        pass

    if not frames:
        return ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AMD","NFLX","QCOM"]

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["symbol"])

    if exclude_funds:
        df = df[(df["etf"].astype(str)!="Y") & (df["test_issue"].astype(str)!="Y")]
        if "nextshares" in df.columns:
            df = df[df["nextshares"].astype(str)!="Y"]

    if exclude_derivs and "security_name" in df.columns:
        pat = re.compile(r"(WARRANT|RIGHTS?|UNITS?|PREF|PREFERRED|NOTE|BOND|TRUST|FUND|ETF|ETN|DEPOSITARY|SPAC)", re.IGNORECASE)
        df = df[~df["security_name"].astype(str).str.contains(pat, na=False)]

    return sorted({to_yahoo_symbol(s) for s in df["symbol"].dropna().unique().tolist()})

@st.cache_data(ttl=60*30, show_spinner=True)
def fetch_daily_yahoo(tickers: List[str], period: str = "6w", chunk: int = 300, threads_on: bool = True) -> Dict[str, pd.DataFrame]:
    yf = require_yfinance()
    if yf is None:
        return {}

    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = [to_yahoo_symbol(t) for t in tickers]
    data: Dict[str, pd.DataFrame] = {}
    CHUNK = int(chunk)
    total = len(tickers)
    progress = st.progress(0.0)

    for i in range(0, total, CHUNK):
        sub = tickers[i:i+CHUNK]
        try:
            raw = yf.download(
                " ".join(sub),
                interval="1d",
                period=period,
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=threads_on
            )
        except Exception:
            progress.progress(min(1.0, (i+CHUNK)/max(1,total)))
            continue

        if len(sub) == 1 or not isinstance(raw.columns, pd.MultiIndex):
            if raw is not None and not raw.empty:
                df = raw.rename(columns={c: c.split(" ")[0] for c in raw.columns})
                df = df[["Open","High","Low","Close","Volume"]].rename(columns=str.lower).dropna()
                data[sub[0]] = df
        else:
            for t in sub:
                try:
                    df = raw[t]
                    df = df.rename(columns={c: c.split(" ")[0] for c in df.columns})
                    df = df[["Open","High","Low","Close","Volume"]].rename(columns=str.lower).dropna()
                    data[t] = df
                except Exception:
                    continue

        progress.progress(min(1.0, (i+CHUNK)/max(1,total)))

    return data

@st.cache_data(ttl=60*60, show_spinner=True)
def build_universe_flow_rank(
    n_top:int = 800,
    min_price:float = 5.0,
    max_price:float = 100.0,
    exclude_top_volume_pct:float = 1.5,
    gentle_momentum:bool = True,
    daily_bars:int = 5,
    max_pool:int = 6000,
    yf_chunk:int = 300,
    yf_threads:bool = True,
) -> Tuple[List[str], Dict]:
    t0 = time.perf_counter()
    pool = download_symbol_files_qn_only()[:max_pool]
    t1 = time.perf_counter()
    daily = fetch_daily_yahoo(pool, period="6w", chunk=yf_chunk, threads_on=yf_threads)
    t2 = time.perf_counter()

    candidates: List[Tuple[str, float, float, float]] = []  # (sym, adv, last_vol, last_close)
    need_bars = max(daily_bars + 1, 21)

    for sym, df in daily.items():
        if df is None or df.empty or len(df) < need_bars:
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
    after_excl = kept
    if exclude_top_volume_pct and exclude_top_volume_pct > 0 and kept >= 50:
        vols = np.array([lv for _, _, lv, _ in candidates], dtype=float)
        cutoff = float(np.quantile(vols, 1.0 - (exclude_top_volume_pct / 100.0)))
        candidates = [row for row in candidates if row[2] < cutoff]
        after_excl = len(candidates)

    if not candidates:
        return [], {"evaluated": len(daily), "kept": 0, "after_excl": 0, "t_sym": t1 - t0, "t_dl": t2 - t1}

    total_adv = float(sum(adv for _, adv, _lv, _px in candidates)) or 1.0
    scored = [(sym, adv / total_adv, adv) for (sym, adv, _lv, _px) in candidates]
    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)

    # Fail-soft: relax momentum if too few
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
    return top_syms, {
        "evaluated": len(daily),
        "kept": kept,
        "after_excl": after_excl,
        "t_sym": t1 - t0,
        "t_dl": t2 - t1,
    }

def replace_universe_sheet(tickers: List[str]) -> Tuple[bool, str]:
    client, gspread = require_gspread_client()
    if not client:
        return False, "Missing Google libs or credentials."
    sh = client.open_by_key(GOOGLE_SHEET_ID)
    try:
        ws = sh.worksheet(GOOGLE_SHEET_NAME)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=GOOGLE_SHEET_NAME, rows=1, cols=2)
    rows = [["Timestamp (ET)", "Ticker"]] + [[now_et_str(), t] for t in tickers]
    ws.clear()
    ws.update("A1", rows, value_input_option="USER_ENTERED")
    return True, f"Wrote {len(tickers)} tickers."

# =========================
# UI — simple, fast, scan on click
# =========================
with st.expander("Criteria (adjust then click Run)", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        n_top = st.slider("Top N (by $-flow share)", 100, 1200, 800, step=50)
        daily_bars = st.slider("Daily bars for $-flow avg", 3, 10, 5)
        gentle_momentum = st.checkbox("Gentle momentum (last ≥ prev OR ≥ EMA20)", value=True)
        fast_mode = st.checkbox("⚡ Fast mode (smaller pool + threaded yfinance + bigger batches)", value=True)
    with c2:
        min_price = st.number_input("Min price ($)", value=5.0, step=0.5)
        max_price = st.number_input("Max price ($)", value=100.0, step=1.0)
        exclude_top_volume_pct = st.number_input("Exclude top volume (%)", value=1.5, min_value=0.0, max_value=25.0, step=0.5)
        if st.button("🧹 Clear caches"):
            download_symbol_files_qn_only.clear()
            fetch_daily_yahoo.clear()
            build_universe_flow_rank.clear()
            st.toast("Caches cleared. Re-run to rebuild.", icon="🧹")

colA, colB = st.columns(2)
with colA:
    run_scan = st.button("🚀 Run Screener (compute only)")
with colB:
    run_scan_and_write = st.button("📝 Run & Update Google Sheet")

if run_scan or run_scan_and_write:
    max_pool = 4000 if fast_mode else 6000
    yf_chunk = 300 if fast_mode else 200
    yf_threads = True if fast_mode else False

    with st.spinner("Building universe…"):
        syms, stats = build_universe_flow_rank(
            n_top=int(n_top),
            min_price=float(min_price),
            max_price=float(max_price),
            exclude_top_volume_pct=float(exclude_top_volume_pct),
            gentle_momentum=bool(gentle_momentum),
            daily_bars=int(daily_bars),
            max_pool=max_pool,
            yf_chunk=yf_chunk,
            yf_threads=yf_threads,
        )

    if not syms:
        st.error("No tickers found. Try relaxing filters or reducing exclusions.")
    else:
        st.success(
            f"Selected {len(syms)} tickers. "
            f"(Evaluated: {stats.get('evaluated',0)}, kept: {stats.get('kept',0)}, after excl: {stats.get('after_excl',0)})"
        )
        st.code(", ".join(syms[:100]), language="text")
        with st.expander("Diagnostics", expanded=False):
            st.caption(f"Symbols load: {stats.get('t_sym',0):.2f}s · Daily bars download: {stats.get('t_dl',0):.2f}s")

        if run_scan_and_write:
            try:
                ok, msg = replace_universe_sheet(syms)
                if ok:
                    st.success(f"Google Sheet updated: {msg}")
                else:
                    st.error("Failed to update Google Sheet.")
            except Exception as e:
                st.error(f"Google Sheet write error: {e}")

st.caption("Fast boot: no heavy imports until you click a button. If yfinance is missing, fix requirements.txt at repo root and redeploy.")
