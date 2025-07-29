import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

# ---- CONFIG ----
GOOGLE_SHEET_ID = "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4"
GOOGLE_SHEET_NAME = "MACD Cross"

sp500 = [
    'AAPL','ABBV','ABT','ACN','ADBE','AIG','AMGN','AMT','AMZN','AVGO','AXP','BA','BAC','BK','BKNG','BLK','BMY','BRK-B','C','CAT',
    'CHTR','CL','CMCSA','COF','COP','COST','CRM','CSCO','CVS','CVX','DHR','DIS','DOW','DUK','EMR','EXC','F','FDX','FOX','FOXA',
    'GD','GE','GILD','GM','GOOG','GOOGL','GS','HD','HON','IBM','INTC','JNJ','JPM','KHC','KMI','KO','LIN','LLY','LMT','LOW',
    'MA','MCD','MDLZ','MDT','MET','META','MMM','MO','MRK','MS','MSFT','NEE','NFLX','NKE','NVDA','ORCL','PEP','PFE','PG','PM',
    'PYPL','QCOM','RTX','SBUX','SCHW','SO','SPG','T','TGT','TMO','TMUS','TSLA','TXN','UNH','UNP','UPS','USB','V','VZ','WBA',
    'WFC','WMT','XOM'
    # ... add the rest of S&P 500 tickers as you wish ...
]

def formatn(num, d=2):
    try:
        if num is None or num == "" or np.isnan(num): return "-"
        if isinstance(num, int) or d == 0:
            return f"{int(num):,}"
        return f"{num:,.{d}f}"
    except Exception:
        return str(num)

def norm(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x).lower() for x in c if x]) for c in df.columns]
    else:
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    return df

def ensure_core_cols(df):
    req = ["close","open","high","low","volume"]
    col_map = {c: c for c in df.columns}
    for col in df.columns:
        cstr = str(col).lower()
        for r in req:
            if r in cstr and r not in col_map.values():
                col_map[col] = r
    df = df.rename(columns=col_map)
    missing = [x for x in req if x not in df.columns]
    if missing:
        raise Exception(f"Missing columns: {missing}")
    return df

def calc_indicators(df):
    df['ema10'] = df['close'].ewm(span=10, min_periods=10).mean()
    df['ema20'] = df['close'].ewm(span=20, min_periods=20).mean()
    # RSI
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(window=14).mean()
    roll_down = down.rolling(window=14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['rsi14'] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = df['close'].ewm(span=12, min_periods=12).mean()
    ema26 = df['close'].ewm(span=26, min_periods=26).mean()
    df['macd'] = ema12 - ema26
    df['macdsignal'] = df['macd'].ewm(span=9, min_periods=9).mean()
    df['hist'] = df['macd'] - df['macdsignal']
    return df

def get_gspread_client_from_secrets():
    info = st.secrets["gcp_service_account"]
    creds_dict = {k: v for k, v in info.items()}
    if isinstance(creds_dict["private_key"], list):
        creds_dict["private_key"] = "\n".join(creds_dict["private_key"])
    creds_json = json.dumps(creds_dict)
    scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(creds_json), scope)
    return gspread.authorize(creds)

def append_to_gsheet(rows, sheet_name):
    try:
        client = get_gspread_client_from_secrets()
        sheet = client.open_by_key(GOOGLE_SHEET_ID).worksheet(sheet_name)
        for row in rows:
            sheet.append_row(row, value_input_option="USER_ENTERED")
    except Exception as e:
        st.warning(f"Google Sheet ({sheet_name}): {e}")

# ---- Sidebar ----
st.sidebar.header("KIV Signal Parameters")
min_volume = st.sidebar.number_input("Min Avg Volume (10 bars)", value=100_000)
min_price = st.sidebar.number_input("Min Price ($)", value=10.0)
max_price = st.sidebar.number_input("Max Price ($)", value=2000.0)
rsi_thresh = st.sidebar.number_input("RSI threshold (below)", value=60.0)
lookback_days = st.sidebar.slider("Lookback (max days to load)", 2, 7, 4)

# ---- Signal Table ----
st.title("AI-Powered US Stocks Screener: 1h MACD Cross")
st.caption(f"Last run: {datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S')} US/Eastern")

# -- Fetch & Prepare Sheet History for De-dupe
recent_signals = set()
try:
    client = get_gspread_client_from_secrets()
    sheet = client.open_by_key(GOOGLE_SHEET_ID).worksheet(GOOGLE_SHEET_NAME)
    data = sheet.get_all_values()
    if data and len(data) > 1:
        headers, rows = data[0], data[1:]
        # Only keep signals from last 7 days for dedupe (adjust as needed)
        for row in rows:
            if len(row) >= 2:
                ktime, kticker = row[0], row[1]
                recent_signals.add((ktime, kticker))
except Exception as e:
    st.warning(f"Could not load recent sheet data: {e}")

kiv_results, rows_to_append = [], []
progress = st.progress(0.0)
for n, ticker in enumerate(sp500):
    try:
        df = yf.download(ticker, period=f"{lookback_days+3}d", interval="1h", progress=False, threads=False)
        if df.empty or len(df) < 20: continue
        df = norm(df)
        df = ensure_core_cols(df)
        df = calc_indicators(df)
        df = df.dropna()
        df['us_time'] = df.index.tz_convert('US/Eastern')
        df['avg_vol_10'] = df['volume'].rolling(10).mean()
        curr = df.iloc[-1]
        dt_bar = curr['us_time']
        price = curr['close']
        macd = curr['macd']
        macdsignal = curr['macdsignal']
        hist = curr['hist']
        rsi = curr['rsi14']
        ema10 = curr['ema10']
        ema20 = curr['ema20']
        avg_vol = curr['avg_vol_10']
        # --- Signal Criteria: NO LOOKBACK
        if (
            macd > 0 and macdsignal < 0 and
            hist > 0 and
            rsi < rsi_thresh and
            ema10 > ema20 and
            min_price <= price <= max_price and
            avg_vol >= min_volume
        ):
            row_key = (dt_bar.strftime("%Y-%m-%d %H:%M"), ticker)
            if row_key in recent_signals: continue  # dedupe against sheet
            kiv_results.append({
                "Time": dt_bar.strftime("%Y-%m-%d %H:%M"),
                "Ticker": ticker,
                "MACD": formatn(macd,4),
                "MACD Signal": formatn(macdsignal,4),
                "RSI": formatn(rsi,2),
                "EMA10": formatn(ema10,2),
                "EMA20": formatn(ema20,2),
                "Hist": formatn(hist,4),
                "Price": formatn(price,2),
                "AvgVol10": formatn(avg_vol,0)
            })
            rows_to_append.append([
                dt_bar.strftime("%Y-%m-%d %H:%M"), ticker,
                formatn(price,2), formatn(rsi,2), formatn(macd,4), formatn(macdsignal,4),
                formatn(hist,4), formatn(ema10,2), formatn(ema20,2), formatn(avg_vol,0)
            ])
            recent_signals.add(row_key)
    except Exception as e:
        continue
    progress.progress((n+1)/len(sp500))
progress.empty()

# ---- Show Table ----
if kiv_results:
    st.subheader("⭐ 1h MACD Cross Current Signals (No Lookback)")
    df_out = pd.DataFrame(kiv_results)
    st.dataframe(df_out, use_container_width=True)
    try:
        if rows_to_append:
            append_to_gsheet(rows_to_append, GOOGLE_SHEET_NAME)
    except Exception as e:
        st.warning(f"Google Sheet update error: {e}")
else:
    st.info("No current 1h MACD signals found.")

# --- Criteria Recap
with st.expander("📜 Screener Signal Criteria (Click to expand)"):
    st.markdown("""
    - **Chart:** 1H bar, US/Eastern time.
    - **MACD > 0**, **MACD Signal < 0** (current bar only).
    - **MACD Histogram > 0**
    - **RSI(14) below threshold** (sidebar adjustable, default 60).
    - **EMA10 > EMA20** (trend confirmation).
    - **Min price, min avg volume (10 bars)** (sidebar).
    - **No previous bar/cross-up logic:** Only most recent bar is used.
    - **No duplicate alerts per ticker/time (sheet-guarded).**
    - **Pushes new hits to Google Sheet for tracking.**
    """)

st.caption("© AI Screener | S&P 500. 1h chart. Only current-bar signals. No previous-bar lookback. Sheet push enabled.")

