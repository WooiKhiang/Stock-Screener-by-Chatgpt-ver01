import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, time as dt_time
import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import time

# ---- Config ----
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"
GOOGLE_SHEET_ID = "YOUR_GOOGLE_SHEET_ID"
GOOGLE_SHEET_NAME = "SnP"

sp100 = [
    'AAPL','ABBV','ABT','ACN','ADBE','AIG','AMGN','AMT','AMZN','AVGO',
    'AXP','BA','BAC','BK','BKNG','BLK','BMY','BRK-B','C','CAT',
    'CHTR','CL','CMCSA','COF','COP','COST','CRM','CSCO','CVS','CVX',
    'DHR','DIS','DOW','DUK','EMR','EXC','F','FDX','FOX','FOXA',
    'GD','GE','GILD','GM','GOOG','GOOGL','GS','HD','HON','IBM',
    'INTC','JNJ','JPM','KHC','KMI','KO','LIN','LLY','LMT','LOW',
    'MA','MCD','MDLZ','MDT','MET','META','MMM','MO','MRK','MS',
    'MSFT','NEE','NFLX','NKE','NVDA','ORCL','PEP','PFE','PG','PM',
    'PYPL','QCOM','RTX','SBUX','SCHW','SO','SPG','T','TGT','TMO',
    'TMUS','TSLA','TXN','UNH','UNP','UPS','USB','V','VZ','WBA',
    'WFC','WMT','XOM'
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
    df['ema40'] = df['close'].ewm(span=40, min_periods=40).mean()
    df['ema200'] = df['close'].ewm(span=200, min_periods=200).mean()
    df['ema10'] = df['close'].ewm(span=10, min_periods=10).mean()
    df['ema20'] = df['close'].ewm(span=20, min_periods=20).mean()
    df['ema50'] = df['close'].ewm(span=50, min_periods=50).mean()
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
    # ATR
    high_low = df['high'] - df['low']
    high_prevclose = np.abs(df['high'] - df['close'].shift(1))
    low_prevclose = np.abs(df['low'] - df['close'].shift(1))
    ranges = pd.concat([high_low, high_prevclose, low_prevclose], axis=1)
    df['atr14'] = ranges.max(axis=1).rolling(14).mean()
    return df

def local_time_str():
    return datetime.now(pytz.timezone("Asia/Singapore")).strftime('%Y-%m-%d %H:%M:%S')

# ---- Spam Prevention: alert memory ----
if "alerted_today" not in st.session_state:
    st.session_state["alerted_today"] = set()

# ---- Sidebar ----
st.sidebar.header("Filter Settings (Debug Mode: All Wide Open)")
min_price = st.sidebar.number_input("Min Price ($)", value=1.0)
max_price = st.sidebar.number_input("Max Price ($)", value=10000.0)
min_vol = st.sidebar.number_input("Min Volume (last bar)", value=0)
capital_per_trade = st.sidebar.number_input("Capital per Trade ($)", value=1000.0, step=100.0)
min_target_pct = st.sidebar.number_input("Min Target (%)", value=0.1, min_value=0.05, step=0.05) / 100
min_cutloss_pct = st.sidebar.number_input("Min Cutloss (%)", value=0.1, min_value=0.05, step=0.05) / 100
macd_stack_on = st.sidebar.checkbox("MACD Stack (Momentum)", value=True)
hybrid_on = st.sidebar.checkbox("Hybrid 5min+1h Confirm", value=True)
ema200_on = st.sidebar.checkbox("EMA200 Breakout (Swing)", value=True)
debug_strategy_on = st.sidebar.checkbox("DEBUG: Always True Strategy", value=True)
show_debug = st.sidebar.checkbox("Show Debug Info", value=True)

st.title("AI-Powered US Stocks Screener (DEBUG: All Filters Wide Open)")
st.caption(f"Last run: {local_time_str()}")

# ---- DEBUG "Always True" strategy ----
def debug_always_signal(df):
    return True, "DEBUG: Always True", 50  # Always fires

debug_issues, results = [], []

for ticker in sp100:
    try:
        # --- Data load ---
        df5 = yf.download(ticker, period='3d', interval='5m', progress=False, threads=False)
        if df5.empty: continue
        df5 = norm(df5)
        df5 = ensure_core_cols(df5)
        df5 = calc_indicators(df5)

        # --- DEBUG STRATEGY (always fires if enabled) ---
        if debug_strategy_on:
            hit, reason, score = debug_always_signal(df5)
            if hit:
                price = df5['close'].iloc[-1]
                atr = df5['atr14'].iloc[-1]
                results.append({
                    "Ticker": ticker,
                    "Strategy": "DEBUG: Always True",
                    "Score": score,
                    "Entry": formatn(price),
                    "Target Price": formatn(price + 0.5, 2),
                    "Cut Loss Price": formatn(price - 0.3, 2),
                    "Shares": int(capital_per_trade // price),
                    "ATR": formatn(atr,2),
                    "Reason": reason,
                    "Type": "Debug",
                    "Time Picked": local_time_str(),
                    "SigID": (ticker, "DEBUG: Always True")
                })
        # --- (Add your other real strategies below if you want) ---

    except Exception as e:
        debug_issues.append({"Ticker": ticker, "Issue": str(e)})
        continue

# ---- OUTPUT TABLE & ALERTS ----
if results:
    df_out = pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)
    st.subheader("⭐ AI-Picked Stock Setups (DEBUG: Should Always Fire!)")
    st.dataframe(df_out.head(10), use_container_width=True)
else:
    st.warning("No current signals found. (This should never happen in debug mode!)")

# ---- Debug Table ----
if show_debug:
    st.subheader("Debug: Issues Encountered")
    if debug_issues:
        st.dataframe(pd.DataFrame(debug_issues))
    else:
        st.info("No issues.")

st.caption("© AI Screener | S&P 100 (Debug). All filters off. 'DEBUG: Always True' should fire for every ticker every run.")
