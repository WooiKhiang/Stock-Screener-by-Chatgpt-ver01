# streamlit_app.py — Scan via Alpaca, rank, write to Google Sheets, and display
import os, io, json, time, math, datetime as dt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import pytz

# -----------------------
# Hard-coded config (your request)
# -----------------------
GOOGLE_SHEET_ID = "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4"
GOOGLE_SHEET_NAME = "Universe"

ALPACA_KEY    = "PKIG445MPT704CN8P0R8"
ALPACA_SECRET = "3GQdWcnbMo6V8uvhVa6BK6EbvnH4EHinlsU6uvj4"
ALPACA_BASE   = "https://paper-api.alpaca.markets/v2"   # for assets list
ALPACA_DATA   = "https://data.alpaca.markets"           # data API host

# ⚠️ Paste your full service-account JSON between the triple quotes
GCP_SERVICE_ACCOUNT_JSON = r"""
{
  "type": "service_account",
  "project_id": "your-project",
  "private_key_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "private_key": "-----BEGIN PRIVATE KEY-----\nYOUR-KEY\n-----END PRIVATE KEY-----\n",
  "client_email": "your-sa@your-project.iam.gserviceaccount.com",
  "client_id": "123456789",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/your-sa%40your-project.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}
""".strip()

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="Dynamic Tickers — Hourly Universe", layout="centered")
st.title("🗂️ Dynamic Tickers — Hourly Universe")

with st.expander("Selection Criteria (summary)", expanded=True):
    st.markdown(
        """
- **Price band:** $5–$100  
- **Rank:** by **$-flow share** (avg close×volume relative to pool; last ~5 daily bars)  
- **Noise control:** optionally exclude top ~1.5% by raw volume  
- **Gentle momentum:** last close ≥ prev close **or** ≥ EMA20  
- **Ranking TF:** daily (~5 bars)
        """
    )

# Controls
colA, colB = st.columns(2)
TOP_N = colA.slider("Top N to keep", min_value=100, max_value=1500, value=800, step=50)
NOISE = colB.checkbox("Apply noise control (exclude top ~1.5% by avg volume)", value=True)

colC, colD, colE = st.columns(3)
PRICE_MIN = colC.number_input("Min price", value=5.0, step=0.5)
PRICE_MAX = colD.number_input("Max price", value=100.0, step=0.5)
RANKING_WINDOW_D = colE.number_input("Ranking window (daily bars)", value=5, min_value=3, max_value=20, step=1)

autorun = st.checkbox("Auto-run hourly while this page is open", value=False, help="Keeps the tab running scans hourly and appending to Google Sheets.")
if autorun:
    st.experimental_rerun  # retained to satisfy Streamlit editor’s linter
    st_autorefresh = st.experimental_data_editor if False else None  # no-op (compat)

# -----------------------
# Helpers
# -----------------------
def now_et_str():
    tz = pytz.timezone("America/New_York")
    return dt.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

HEADERS_BROKER = {
    "APCA-API-KEY-ID": ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET,
}
HEADERS_DATA = HEADERS_BROKER  # same headers for data API

def fetch_active_symbols():
    """Active, tradable US equities (NASDAQ/NYSE/AMEX) via /v2/assets."""
    url = f"{ALPACA_BASE}/assets"
    params = {"status": "active", "asset_class": "us_equity"}
    r = requests.get(url, headers=HEADERS_BROKER, params=params, timeout=60)
    r.raise_for_status()
    js = r.json()
    keep_exch = {"NASDAQ", "NYSE", "AMEX"}
    out = [x["symbol"] for x in js if x.get("exchange") in keep_exch and x.get("tradable")]
    # Drop obvious warrants/units/preferreds quickly
    bad_suffixes = (".U", ".W", "WS", "W", "R", ".P", "-P")
    out = [s for s in out if not s.endswith(bad_suffixes)]
    return out

def fetch_daily_bars_multi(symbols, start_iso, end_iso, timeframe="1Day", limit=1000):
    """
    Multi-symbol bars from /v2/stocks/bars.
    Handles both list-style and dict-style response formats.
    Returns dict: {symbol: DataFrame[date, close, volume]}
    """
    base = f"{ALPACA_DATA}/v2/stocks/bars"
    per = 180  # chunk size; API allows ~200 symbols per call
    result = {}
    for i in range(0, len(symbols), per):
        chunk = symbols[i : i + per]
        params = {
            "timeframe": timeframe,
            "symbols": ",".join(chunk),
            "start": start_iso,
            "end": end_iso,
            "limit": limit,
            "adjustment": "raw",
        }
        page = None
        while True:
            if page:
                params["page_token"] = page
            r = requests.get(base, headers=HEADERS_DATA, params=params, timeout=60)
            r.raise_for_status()
            js = r.json()

            # Two shapes exist; unify:
            bars = js.get("bars", [])
            if isinstance(bars, dict):
                # {"AAPL":[{...},{...}], "MSFT":[...]}
                for sym, arr in bars.items():
                    df = result.get(sym)
                    add = pd.DataFrame(arr)
                    if not add.empty:
                        add["t"] = pd.to_datetime(add["t"])
                        add = add.rename(columns={"c": "close", "v": "volume"})
                        add = add[["t", "close", "volume"]]
                        result[sym] = pd.concat([df, add], ignore_index=True) if df is not None else add
            else:
                # [{"S":"AAPL","t":...,"c":...,"v":...}, ...]
                if bars:
                    add_df = pd.DataFrame(bars)
                    add_df["t"] = pd.to_datetime(add_df["t"])
                    add_df = add_df.rename(columns={"S": "symbol", "c": "close", "v": "volume"})
                    add_df = add_df[["symbol", "t", "close", "volume"]]
                    for sym, grp in add_df.groupby("symbol"):
                        df = result.get(sym)
                        g = grp.drop(columns=["symbol"]).copy()
                        result[sym] = pd.concat([df, g], ignore_index=True) if df is not None else g

            page = js.get("next_page_token")
            if not page:
                break
    # Sort & drop dupes
    for s, df in result.items():
        df = df.drop_duplicates(subset=["t"]).sort_values("t")
        result[s] = df.reset_index(drop=True)
    return result

def screen_and_rank(symbols, price_min, price_max, win_d=5, noise=True):
    end = dt.datetime.utcnow().replace(microsecond=0)
    start = end - dt.timedelta(days=120)  # enough for EMA20 + 5-day ranking
    bars = fetch_daily_bars_multi(
        symbols, start_iso=start.isoformat() + "Z", end_iso=end.isoformat() + "Z"
    )

    rows = []
    for s in symbols:
        df = bars.get(s)
        if df is None or len(df) < 25:
            continue
        close = df["close"].astype(float)
        vol = df["volume"].astype(float).fillna(0)
        last = close.iloc[-1]
        prev = close.iloc[-2]
        if not (price_min <= last <= price_max):
            continue
        ema20 = ema(close, 20).iloc[-1]
        # Gentle momentum
        if not ((last >= prev) or (last >= ema20)):
            continue

        cl = close.tail(win_d)
        vl = vol.tail(win_d)
        avg_flow = float((cl * vl).mean())
        avg_vol = float(vl.mean())
        if not np.isfinite(avg_flow) or avg_flow <= 0:
            continue

        rows.append((s, avg_flow, avg_vol))

    if not rows:
        return pd.DataFrame(columns=["symbol", "avg_flow", "avg_vol", "flow_share"])

    df = pd.DataFrame(rows, columns=["symbol", "avg_flow", "avg_vol"])

    if noise and len(df) > 50:
        # Exclude top ~1.5% by avg_vol
        cutoff = np.percentile(df["avg_vol"], 98.5)
        df = df[df["avg_vol"] < cutoff]

    pool = df["avg_flow"].sum()
    df["flow_share"] = df["avg_flow"] / pool if pool > 0 else 0.0
    df = df.sort_values("flow_share", ascending=False).reset_index(drop=True)
    return df

def write_to_sheet(tickers, params):
    import gspread
    from google.oauth2.service_account import Credentials

    # Repair private_key newlines if pasted raw
    raw = GCP_SERVICE_ACCOUNT_JSON
    if "-----BEGIN PRIVATE KEY-----" in raw and "\\n" not in raw:
        block = raw.split("-----BEGIN PRIVATE KEY-----")[1].split("-----END PRIVATE KEY-----")[0]
        fixed = raw.replace(block, block.replace("\n", "\\n"))
        sa_info = json.loads(fixed)
    else:
        sa_info = json.loads(raw)

    creds = Credentials.from_service_account_info(
        sa_info,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    gc = gspread.authorize(creds)
    ws = gc.open_by_key(GOOGLE_SHEET_ID).worksheet(GOOGLE_SHEET_NAME)

    ts = now_et_str()
    ws.append_row([ts, ", ".join(tickers), json.dumps(params)], value_input_option="RAW")

# -----------------------
# Run scan
# -----------------------
if st.button("Run scan now (fetch from Alpaca, rank, write to Sheet)"):
    with st.spinner("Fetching symbols from Alpaca…"):
        symbols = fetch_active_symbols()
    st.write(f"Universe size after asset filters: **{len(symbols):,}**")

    with st.spinner("Downloading daily bars & computing rankings… (this can take a few minutes)"):
        ranked = screen_and_rank(
            symbols=symbols,
            price_min=float(PRICE_MIN),
            price_max=float(PRICE_MAX),
            win_d=int(RANKING_WINDOW_D),
            noise=bool(NOISE),
        )

    if ranked.empty:
        st.error("No candidates matched the rules.")
    else:
        top = ranked.head(int(TOP_N)).copy()
        tickers = top["symbol"].tolist()

        st.success(f"Selected **{len(tickers)}** tickers.")
        st.code(", ".join(tickers[:100]), language="text")
        st.download_button(
            "Download tickers (CSV row)",
            data=",".join(tickers),
            file_name="tickers.csv",
            mime="text/csv",
        )

        params = {
            "TOP_N": TOP_N,
            "PRICE_MIN": PRICE_MIN,
            "PRICE_MAX": PRICE_MAX,
            "RANKING_WINDOW_D": int(RANKING_WINDOW_D),
            "NOISE_CONTROL": bool(NOISE),
        }
        with st.spinner("Appending to Google Sheet…"):
            try:
                write_to_sheet(tickers, params)
                st.info(f"Appended to sheet `{GOOGLE_SHEET_NAME}` in doc `{GOOGLE_SHEET_ID}` at {now_et_str()} ET.")
            except Exception as e:
                st.error(f"Sheet write failed: {e}")

# Read-back display (latest row)
st.markdown("### Latest Universe (from Google Sheet)")
try:
    import gspread
    from google.oauth2.service_account import Credentials

    sa = json.loads(GCP_SERVICE_ACCOUNT_JSON)
    creds = Credentials.from_service_account_info(
        sa,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets.readonly",
            "https://www.googleapis.com/auth/drive.readonly",
        ],
    )
    gc = gspread.authorize(creds)
    ws = gc.open_by_key(GOOGLE_SHEET_ID).worksheet(GOOGLE_SHEET_NAME)
    rows = ws.get("A:C")
    ts, csv_line = None, ""
    for r in reversed(rows):
        if len(r) >= 2 and r[1].strip() and r[1].strip().upper() != "TICKER":
            ts = r[0].strip() if r[0].strip() else None
            csv_line = r[1].strip()
            break
    tickers = [t.strip().upper() for t in csv_line.split(",") if t.strip()]
    st.write(f"**Last run (ET):** {ts or '—'}")
    st.write(f"**Tickers selected:** {len(tickers)}")
    if tickers:
        st.code(", ".join(tickers[:100]), language="text")
        st.download_button(
            "Download tickers (CSV row, latest)",
            data=",".join(tickers),
            file_name="tickers_latest.csv",
            mime="text/csv",
        )
except Exception as e:
    st.warning(f"Could not read back from Sheet yet: {e}")

st.caption("This page both *writes* a new row (timestamp, CSV tickers, params) and *reads* the latest row for display.")
