# streamlit_app.py — Alpaca scan → rank → write to Google Sheets (VERTICAL) → display latest
import os, json, datetime as dt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import pytz

# =========================
# Hard-coded configuration
# =========================
GOOGLE_SHEET_ID   = "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4"
GOOGLE_SHEET_NAME = "Universe"

ALPACA_KEY    = "PKIG445MPT704CN8P0R8"
ALPACA_SECRET = "3GQdWcnbMo6V8uvhVa6BK6EbvnH4EHinlsU6uvj4"
ALPACA_BASE   = "https://paper-api.alpaca.markets/v2"   # assets endpoint
ALPACA_DATA   = "https://data.alpaca.markets"           # data endpoint
FEED          = "iex"  # REQUIRED for paper/free keys

# Service account JSON from Streamlit secrets or env (DO NOT hard-code in repo)
SA_RAW = st.secrets.get("GCP_SERVICE_ACCOUNT") or os.getenv("GCP_SERVICE_ACCOUNT")

# =========================
# UI
# =========================
st.set_page_config(page_title="Dynamic Tickers — Hourly Universe", layout="centered")
st.title("🗂️ Dynamic Tickers — Hourly Universe")

with st.expander("Selection Criteria (summary)", expanded=True):
    st.markdown("""
- **Price band:** $5–$100  
- **Rank:** by **$-flow share** (avg close×volume relative to pool; last ~5 daily bars)  
- **Noise control:** optionally exclude top ~1.5% by raw volume  
- **Gentle momentum:** last close ≥ prev close **or** ≥ EMA20  
- **Ranking TF:** daily (~5 bars)
    """)

c1, c2 = st.columns(2)
TOP_N  = c1.slider("Top N to keep", 100, 1500, 800, 50)
NOISE  = c2.checkbox("Noise control (exclude top ~1.5% by avg volume)", value=True)

c3, c4, c5 = st.columns(3)
PRICE_MIN = c3.number_input("Min price", value=5.0, step=0.5)
PRICE_MAX = c4.number_input("Max price", value=100.0, step=0.5)
RANKING_WINDOW_D = c5.number_input("Ranking window (daily bars)", value=5, min_value=3, max_value=20, step=1)

# =========================
# Helpers
# =========================
def now_et_str():
    tz = pytz.timezone("America/New_York")
    return dt.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def parse_sa(raw_json: str) -> dict:
    if not raw_json:
        raise RuntimeError("Missing GCP service account JSON in secrets/env (GCP_SERVICE_ACCOUNT).")
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        if "-----BEGIN PRIVATE KEY-----" in raw_json and "\\n" not in raw_json:
            start = raw_json.find("-----BEGIN PRIVATE KEY-----")
            end   = raw_json.find("-----END PRIVATE KEY-----", start)
            if start != -1 and end != -1:
                end += len("-----END PRIVATE KEY-----")
                block = raw_json[start:end]
                fixed = block.replace("\r\n", "\n").replace("\n", "\\n")
                raw_json = raw_json.replace(block, fixed)
        return json.loads(raw_json)

HEADERS_BROKER = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}
HEADERS_DATA   = HEADERS_BROKER

# ================
# Alpaca fetchers
# ================
def fetch_active_symbols():
    """Active, tradable US equities from /v2/assets (NASDAQ/NYSE/AMEX)."""
    url = f"{ALPACA_BASE}/assets"
    params = {"status": "active", "asset_class": "us_equity"}
    r = requests.get(url, headers=HEADERS_BROKER, params=params, timeout=60)
    r.raise_for_status()
    js = r.json()
    keep_exch = {"NASDAQ", "NYSE", "AMEX"}
    out = [x["symbol"] for x in js if x.get("exchange") in keep_exch and x.get("tradable")]
    # cull obvious warrants/units/preferreds quickly
    bad_suffixes = (".U", ".W", "WS", "W", "R", ".P", "-P")
    out = [s for s in out if not s.endswith(bad_suffixes)]
    return out

def fetch_daily_bars_multi(symbols, start_iso, end_iso, timeframe="1Day", limit=1000):
    """
    Multi-symbol bars from /v2/stocks/bars. Uses feed=iex for paper/free keys.
    Returns {symbol: DataFrame[t, close, volume]}.
    """
    base = f"{ALPACA_DATA}/v2/stocks/bars"
    per = 180  # API allows ~200 symbols per call
    result = {}

    for i in range(0, len(symbols), per):
        chunk = symbols[i:i+per]
        params = {
            "timeframe": timeframe,
            "symbols": ",".join(chunk),
            "start": start_iso,
            "end": end_iso,
            "limit": limit,
            "adjustment": "raw",
            "feed": "iex",
        }
        page = None
        while True:
            if page:
                params["page_token"] = page
            r = requests.get(base, headers=HEADERS_DATA, params=params, timeout=60)
            if r.status_code >= 400:
                msg = r.text[:400].replace("\n", " ")
                raise requests.HTTPError(f"Alpaca /bars error {r.status_code}: {msg}")
            js = r.json()
            bars = js.get("bars", [])

            if isinstance(bars, dict):
                for sym, arr in bars.items():
                    add = pd.DataFrame(arr)
                    if not add.empty:
                        add["t"] = pd.to_datetime(add["t"])
                        add = add.rename(columns={"c": "close", "v": "volume"})
                        add = add[["t", "close", "volume"]]
                        prev = result.get(sym)
                        result[sym] = pd.concat([prev, add], ignore_index=True) if prev is not None else add
            else:
                if bars:
                    add_df = pd.DataFrame(bars)
                    add_df["t"] = pd.to_datetime(add_df["t"])
                    add_df = add_df.rename(columns={"S": "symbol", "c": "close", "v": "volume"})
                    add_df = add_df[["symbol", "t", "close", "volume"]]
                    for sym, grp in add_df.groupby("symbol"):
                        g = grp.drop(columns=["symbol"]).copy()
                        prev = result.get(sym)
                        result[sym] = pd.concat([prev, g], ignore_index=True) if prev is not None else g

            page = js.get("next_page_token")
            if not page:
                break

    for s, df in list(result.items()):
        df = df.drop_duplicates(subset=["t"]).sort_values("t").reset_index(drop=True)
        result[s] = df
    return result

# ======================
# Screening & Ranking
# ======================
def screen_and_rank(symbols, price_min, price_max, win_d=5, noise=True):
    end = dt.datetime.utcnow().replace(microsecond=0)
    start = end - dt.timedelta(days=120)  # EMA20 + ranking window buffer
    bars = fetch_daily_bars_multi(
        symbols, start_iso=start.isoformat() + "Z", end_iso=end.isoformat() + "Z"
    )

    rows = []
    for s in symbols:
        df = bars.get(s)
        if df is None or len(df) < 25:
            continue
        close = df["close"].astype(float)
        vol   = df["volume"].astype(float).fillna(0)

        last = close.iloc[-1]
        prev = close.iloc[-2]
        if not (price_min <= last <= price_max):
            continue

        ema20 = ema(close, 20).iloc[-1]
        if not ((last >= prev) or (last >= ema20)):  # Gentle momentum
            continue

        cl = close.tail(win_d)
        vl = vol.tail(win_d)
        avg_flow = float((cl * vl).mean())
        avg_vol  = float(vl.mean())
        if not np.isfinite(avg_flow) or avg_flow <= 0:
            continue

        rows.append((s, avg_flow, avg_vol))

    if not rows:
        return pd.DataFrame(columns=["symbol", "avg_flow", "avg_vol", "flow_share"])

    df = pd.DataFrame(rows, columns=["symbol", "avg_flow", "avg_vol"])

    if noise and len(df) > 50:
        cutoff = np.percentile(df["avg_vol"], 98.5)  # exclude ~top 1.5% by volume
        df = df[df["avg_vol"] < cutoff]

    pool = df["avg_flow"].sum()
    df["flow_share"] = df["avg_flow"] / pool if pool > 0 else 0.0
    return df.sort_values("flow_share", ascending=False).reset_index(drop=True)

# ======================
# Sheet writer / reader (VERTICAL layout)
# ======================
def write_to_sheet_vertical(tickers, params):
    import gspread
    from google.oauth2.service_account import Credentials

    sa_info = parse_sa(SA_RAW)
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
    rows = [[ts, t] for t in tickers]  # A: timestamp, B: ticker (one per row)

    # Batch-append for speed; fallback to per-row if needed
    try:
        ws.append_rows(rows, value_input_option="RAW")
    except Exception:
        for r in rows:
            ws.append_row(r, value_input_option="RAW")

def read_latest_from_sheet_vertical():
    import gspread
    from google.oauth2.service_account import Credentials

    sa_info = parse_sa(SA_RAW)
    creds = Credentials.from_service_account_info(
        sa_info,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets.readonly",
            "https://www.googleapis.com/auth/drive.readonly",
        ],
    )
    gc = gspread.authorize(creds)
    ws = gc.open_by_key(GOOGLE_SHEET_ID).worksheet(GOOGLE_SHEET_NAME)

    rows = ws.get("A:B")
    # Filter out headers/blank lines
    data = [r for r in rows if len(r) >= 2 and r[0].strip() and r[1].strip() and r[0].strip().upper() not in ("TIMESTAMP","TIME","TS")]
    if not data:
        return None, []

    # Most recent timestamp is in the last non-empty row's A
    last_ts = data[-1][0].strip()
    tickers = [r[1].strip().upper() for r in data if r[0].strip() == last_ts]
    return last_ts, tickers

# ======================
# Actions
# ======================
if st.button("Run scan now (fetch from Alpaca, rank, write vertical to Sheet)"):
    try:
        with st.spinner("Fetching symbols from Alpaca…"):
            symbols = fetch_active_symbols()
        st.write(f"Universe size after asset filters: **{len(symbols):,}**")

        with st.spinner("Downloading daily bars & computing rankings…"):
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
            tickers = ranked.head(int(TOP_N))["symbol"].tolist()
            st.success(f"Selected **{len(tickers)}** tickers.")
            st.code(", ".join(tickers[:100]), language="text")
            st.download_button("Download tickers (CSV row)", data=",".join(tickers), file_name="tickers.csv", mime="text/csv")

            params = {
                "TOP_N": int(TOP_N),
                "PRICE_MIN": float(PRICE_MIN),
                "PRICE_MAX": float(PRICE_MAX),
                "RANKING_WINDOW_D": int(RANKING_WINDOW_D),
                "NOISE_CONTROL": bool(NOISE),
                "LAYOUT": "vertical",
            }
            with st.spinner("Appending vertical rows to Google Sheet…"):
                write_to_sheet_vertical(tickers, params)
            st.info(f"Wrote {len(tickers)} rows to `{GOOGLE_SHEET_NAME}` at {now_et_str()} ET.")

    except requests.HTTPError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Scan failed: {e}")

# Always show latest run (vertical)
st.markdown("### Latest Universe (from Google Sheet — vertical)")
try:
    ts, tickers = read_latest_from_sheet_vertical()
    st.write(f"**Last run (ET):** {ts or '—'}")
    st.write(f"**Tickers selected:** {len(tickers)}")
    if tickers:
        st.code(", ".join(tickers[:100]), language="text")
        st.download_button("Download tickers (CSV row, latest)", data=",".join(tickers), file_name="tickers_latest.csv", mime="text/csv")
except Exception as e:
    st.warning(f"Could not read back from Sheet yet: {e}")

st.caption("Writes one row per ticker: Column A = timestamp (ET), Column B = ticker. Reader aggregates the most recent timestamp’s block.")
