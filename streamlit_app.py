import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta, time as dt_time
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

# ---- CONFIG ----
GOOGLE_SHEET_ID = "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4"
GOOGLE_SHEET_NAME = "MACD Cross"

sp500 = [
    # S&P 500 tickers, shortened for brevity; use your full list here
    "AAPL","MSFT","AMZN","NVDA","GOOGL","META","GOOG","BRK-B","LLY","UNH",
    "TSLA","JPM","V","XOM","MA","AVGO","PG","JNJ","COST","HD","MRK","ADBE",
    "ABBV","CRM","WMT","AMD","PEP","KO","CVX","BAC","MCD","NFLX","DIS",
    # ... add the rest of S&P500 tickers here
]

# ---- Helper Functions ----
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
    # Histogram
    df['macdhist'] = df['macd'] - df['macdsignal']
    # ATR
    high_low = df['high'] - df['low']
    high_prevclose = np.abs(df['high'] - df['close'].shift(1))
    low_prevclose = np.abs(df['low'] - df['close'].shift(1))
    ranges = pd.concat([high_low, high_prevclose, low_prevclose], axis=1)
    df['atr14'] = ranges.max(axis=1).rolling(14).mean()
    return df

def us_now():
    return datetime.now(pytz.timezone("US/Eastern"))

def format_us_time(dt):
    return dt.astimezone(pytz.timezone("US/Eastern")).strftime('%Y-%m-%d %H:%M:%S')

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
st.sidebar.header("Filter Settings")
min_price = st.sidebar.number_input("Min Price ($)", value=10.0)
max_price = st.sidebar.number_input("Max Price ($)", value=2000.0)
min_avg_vol = st.sidebar.number_input("Min Avg Vol (20 bars)", value=200_000)
rsi_max = st.sidebar.number_input("Max RSI (default 60)", value=60, min_value=10, max_value=80, step=1)

st.title("MACD Cross Zero Screener - S&P 500 (1H US Time)")
st.caption(f"Last run: {format_us_time(us_now())}")

if "alerted_today" not in st.session_state:
    st.session_state["alerted_today"] = set()

results = []
reference_rows = []
debug_issues = []
max_days = st.sidebar.number_input("Lookback Days (default 7)", min_value=1, max_value=30, value=7)

for ticker in sp500:
    try:
        df = yf.download(ticker, period=f"{max_days}d", interval='1h', progress=False, threads=False)
        if df.empty or len(df) < 25: continue
        df = norm(df)
        df = ensure_core_cols(df)
        df = calc_indicators(df)
        # Remove pre/post bars by filtering only regular session: 9:30–16:00
        df = df.between_time('09:30', '16:00')
        if len(df) < 25: continue

        # --- MACD Cross Zero (Current Bar only) ---
        curr, prev = df.iloc[-1], df.iloc[-2]
        # Check MACD crosses up zero
        macd_cross = (prev['macd'] < 0) and (curr['macd'] > 0)
        macd_signal_below_zero = curr['macdsignal'] < 0
        rsi_ok = curr['rsi14'] < rsi_max
        ema_ok = curr['ema10'] > curr['ema20']
        hist_pos = curr['macdhist'] > 0
        avg_vol = df['volume'][-20:].mean()
        price = curr['close']
        vol_ok = avg_vol >= min_avg_vol
        time_str = format_us_time(df.index[-1].to_pydatetime())

        # --- Table 1: MACD Cross Signal
        if all([macd_cross, macd_signal_below_zero, rsi_ok, ema_ok, hist_pos, vol_ok, min_price <= price <= max_price]):
            row = {
                "Time": time_str,
                "Ticker": ticker,
                "Price": formatn(price),
                "MACD": formatn(curr['macd'], 4),
                "MACD Signal": formatn(curr['macdsignal'], 4),
                "RSI": formatn(curr['rsi14'], 2),
                "EMA10": formatn(curr['ema10'], 2),
                "EMA20": formatn(curr['ema20'], 2),
                "Volume": formatn(curr['volume'], 0),
                "AvgVol20": formatn(avg_vol, 0),
            }
            results.append(row)

        # --- Table 2: Reference MACD Cross Up Zero, Signal < 0, any RSI/EMA ---
        # Show all such events in the last 20 bars (reference table)
        ref_events = []
        for i in range(-min(20, len(df)-1), 0):
            prev, curr = df.iloc[i-1], df.iloc[i]
            if (prev['macd'] < 0) and (curr['macd'] > 0) and (curr['macdsignal'] < 0):
                ref_events.append({
                    "Time": format_us_time(df.index[i].to_pydatetime()),
                    "Ticker": ticker,
                    "Price": formatn(curr['close']),
                    "MACD": formatn(curr['macd'], 4),
                    "MACD Signal": formatn(curr['macdsignal'], 4),
                    "RSI": formatn(curr['rsi14'], 2),
                    "EMA10": formatn(curr['ema10'], 2),
                    "EMA20": formatn(curr['ema20'], 2),
                    "Volume": formatn(curr['volume'], 0),
                })
        reference_rows.extend(ref_events)
    except Exception as e:
        debug_issues.append({"Ticker": ticker, "Issue": str(e)})
        continue

# --- Output Main Table ---
st.subheader("🔥 MACD Cross Zero Signals (Current Bar)")
if results:
    df_results = pd.DataFrame(results)
    st.dataframe(df_results, use_container_width=True)
    # --- Push to Google Sheets ---
    try:
        gsheet_rows = df_results.values.tolist()
        append_to_gsheet(gsheet_rows, GOOGLE_SHEET_NAME)
    except Exception as e:
        st.warning(f"Google Sheet log error: {e}")
else:
    st.info("No MACD Cross signals found for your settings.")

# --- Output Reference Table ---
st.subheader("🔍 Reference: MACD Cross Up Zero (Last 20 Bars, any RSI/EMA)")
if reference_rows:
    df_ref = pd.DataFrame(reference_rows)
    st.dataframe(df_ref, use_container_width=True)
else:
    st.info("No reference events found.")

# --- Output Debug Table ---
if st.sidebar.checkbox("Show Debug Info", value=False):
    st.subheader("Debug: Issues Encountered")
    if debug_issues:
        st.dataframe(pd.DataFrame(debug_issues))
    else:
        st.info("No issues.")

st.caption("MACD Cross Zero Screener - S&P 500, US market hours only. One row per bar. Manual page refresh required for new data.")

