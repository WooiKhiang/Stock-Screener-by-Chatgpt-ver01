import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yfinance as yf
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, time as dt_time
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

# ---- CONFIG ----
GOOGLE_SHEET_ID = "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4"
GOOGLE_SHEET_NAME = "MACD Cross"

# S&P 500 Tickers (shortened here for brevity - use your full list!)
sp500 = [
    'AAPL','ABBV','ABT','ACN','ADBE','AIG','AMGN','AMT','AMZN','AVGO','AXP','BA','BAC','BK','BKNG','BLK','BMY','BRK-B','C','CAT',
    'CHTR','CL','CMCSA','COF','COP','COST','CRM','CSCO','CVS','CVX','DHR','DIS','DOW','DUK','EMR','EXC','F','FDX','FOX','FOXA',
    'GD','GE','GILD','GM','GOOG','GOOGL','GS','HD','HON','IBM','INTC','JNJ','JPM','KHC','KMI','KO','LIN','LLY','LMT','LOW',
    'MA','MCD','MDLZ','MDT','MET','META','MMM','MO','MRK','MS','MSFT','NEE','NFLX','NKE','NVDA','ORCL','PEP','PFE','PG','PM',
    'PYPL','QCOM','RTX','SBUX','SCHW','SO','SPG','T','TGT','TMO','TMUS','TSLA','TXN','UNH','UNP','UPS','USB','V','VZ','WBA',
    'WFC','WMT','XOM',
    # ... (add the rest if needed)
]

# ---- SIDEBAR FILTERS ----
st.sidebar.header("Screener Filters")
min_price = st.sidebar.number_input("Minimum Price ($)", value=10.0, min_value=0.01)
min_avg_vol = st.sidebar.number_input("Minimum Avg Volume (last 20 bars)", value=300_000)
rsi_threshold = st.sidebar.slider("Max RSI (14)", min_value=30, max_value=70, value=60, step=1)
max_days = st.sidebar.slider("Max Days To Scan", min_value=1, max_value=14, value=7, step=1)

# ---- PAGE TITLE & AUTO-REFRESH ----
st.title("AI-Powered US Stocks Screener: 1h MACD Cross (S&P 500)")
st_autorefresh(interval=300000, key="screener_autorefresh")  # 5 minutes
st.caption("⏳ Page auto-refreshes every 5 minutes. All times in US/Eastern.")

# ---- GOOGLE SHEETS HELPER ----
def get_gspread_client_from_secrets():
    info = st.secrets["gcp_service_account"]
    creds_dict = {k: v for k, v in info.items()}
    if isinstance(creds_dict["private_key"], list):
        creds_dict["private_key"] = "\n".join(creds_dict["private_key"])
    creds_json = json.dumps(creds_dict)
    scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(creds_json), scope)
    return gspread.authorize(creds)

def append_to_gsheet(rows):
    try:
        client = get_gspread_client_from_secrets()
        sheet = client.open_by_key(GOOGLE_SHEET_ID).worksheet(GOOGLE_SHEET_NAME)
        for row in rows:
            sheet.append_row(row, value_input_option="USER_ENTERED")
    except Exception as e:
        st.warning(f"Google Sheet Error: {e}")

# ---- INDICATOR FUNCTIONS ----
def norm(df):
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    return df

def ensure_core_cols(df):
    req = ["close","open","high","low","volume"]
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

def formatn(x, d=2):
    try: return f"{x:,.{d}f}"
    except: return x

def us_time_str():
    return datetime.now(pytz.timezone("US/Eastern")).strftime('%Y-%m-%d %H:%M:%S')

# ---- MARKET SENTIMENT ----
def get_market_sentiment():
    try:
        spy = yf.download('SPY', period='1d', interval='60m', progress=False, threads=False)
        if spy.empty: return 0, "Sentiment: Unknown"
        spy = norm(spy)
        spy = ensure_core_cols(spy)
        open_, last = spy['open'].iloc[0], spy['close'].iloc[-1]
        pct = (last - open_) / open_ * 100
        if pct > 0.5: return pct, "🟢 Bullish"
        elif pct < -0.5: return pct, "🔴 Bearish"
        else: return pct, "🟡 Sideways"
    except Exception:
        return 0, "Sentiment: Unknown"

sentiment_pct, sentiment_text = get_market_sentiment()
st.subheader(f"Market Sentiment: {sentiment_text} ({sentiment_pct:.2f}%)")

# ---- MAIN SCREENER ----
scan_days = max_days
macd_signals = []
for ticker in sp500:
    try:
        df = yf.download(ticker, period=f"{scan_days+2}d", interval='1h', progress=False, threads=False)
        if df.empty or len(df) < 30: continue
        df = norm(df)
        df = ensure_core_cols(df)
        df = calc_indicators(df)

        # Check only most recent bar (current hour)
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        avg_vol = df['volume'][-20:].mean()

        # Main Screener Criteria (NO lookback)
        conds = [
            prev['macd'] < 0 and curr['macd'] > 0,       # MACD cross up zero
            curr['macdsignal'] < 0,                      # MACD signal below zero
            curr['rsi14'] < rsi_threshold,               # RSI below threshold (sidebar)
            curr['ema10'] > curr['ema20'],               # EMA10 > EMA20
            curr['hist'] > 0,                            # MACD histogram positive
            curr['close'] > min_price,                   # Price filter
            avg_vol > min_avg_vol,                       # Average volume filter
        ]
        if all(conds):
            macd_signals.append({
                "Ticker": ticker,
                "Price": formatn(curr['close']),
                "RSI": formatn(curr['rsi14'], 1),
                "MACD": formatn(curr['macd'], 3),
                "MACD Signal": formatn(curr['macdsignal'], 3),
                "Hist": formatn(curr['hist'], 3),
                "EMA10": formatn(curr['ema10']),
                "EMA20": formatn(curr['ema20']),
                "Avg Vol": formatn(avg_vol, 0),
                "Time": df.index[-1].tz_localize(None).strftime("%Y-%m-%d %H:%M"),
                "US Time": us_time_str()
            })
    except Exception as e:
        continue

# ---- SHOW MAIN TABLE ----
st.subheader(f"⭐ S&P 500: 1h MACD Cross Screener (Last {scan_days} days)")
if macd_signals:
    df_signals = pd.DataFrame(macd_signals)
    st.dataframe(df_signals, use_container_width=True)
    # ---- PUSH TO GOOGLE SHEET ----
    try:
        rows = [[x['Time'], x['Ticker'], x['Price'], x['RSI'], x['MACD'], x['MACD Signal'], x['Hist'], x['EMA10'], x['EMA20'], x['Avg Vol']] for x in macd_signals]
        append_to_gsheet(rows)
    except Exception as e:
        st.warning(f"Google Sheet Error: {e}")
else:
    st.info("No signals found meeting all criteria in the latest scan.")

# ---- SHOW RECENT HISTORY TABLE ----
st.subheader("🕒 Last 10 MACD Cross Signals (from Sheet)")
try:
    client = get_gspread_client_from_secrets()
    sheet = client.open_by_key(GOOGLE_SHEET_ID).worksheet(GOOGLE_SHEET_NAME)
    data = sheet.get_all_values()
    if data and len(data) > 1:
        headers, rows = data[0], data[-10:]
        df_last = pd.DataFrame(rows, columns=headers)
        st.dataframe(df_last, use_container_width=True)
except Exception as e:
    st.warning(f"Error loading recent signals: {e}")

# ---- CRITERIA DISPLAY ----
with st.expander("ℹ️ Screener Strategy & Criteria", expanded=True):
    st.markdown(f"""
- **Timeframe:** 1 Hour (US/Eastern)
- **Universe:** S&P 500 (all tickers)
- **MACD Crosses Up Zero:** Previous bar MACD < 0, Current MACD > 0
- **MACD Signal Line Below Zero** (bullish cross still in early phase)
- **RSI(14) Below {rsi_threshold}** (customizable in sidebar)
- **EMA10 > EMA20** (trend confirmation)
- **MACD Histogram Positive**
- **Minimum Price:** ${min_price}
- **Minimum Avg Volume:** {min_avg_vol:,.0f} (last 20 bars)
- **Data auto-refreshes every 5 minutes**
- **Pushes all signals to Google Sheet (MACD Cross tab)**
- **All times in US/Eastern (matches Moomoo & most US brokers)**
    """)

st.caption("© AI Screener | S&P 500 | MACD/EMA/RSI volume filter | v2025-07")
