import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import time

# --- CONFIG ---
GOOGLE_SHEET_ID = "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4"
GOOGLE_SHEET_NAME = "KIV_SIGNALS"
TELEGRAM_BOT_TOKEN = "xxxxxxx"   # optional for alert, fill if needed
TELEGRAM_CHAT_ID = "xxxxxxx"

# --- Ticker Lists (S&P100 + NASDAQ100, deduplicated) ---
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
nasdaq100 = [
    'AAPL','MSFT','GOOGL','AMZN','NVDA','META','TSLA','AVGO','COST','PEP','CSCO','ADBE','CMCSA','NFLX','TXN','QCOM','AMD','INTC','AMGN','AMAT',
    'SBUX','BKNG','INTU','ISRG','LRCX','MU','ADI','GILD','MDLZ','REGN','VRTX','CSX','KLAC','PDD','ADP','MAR','CRWD','CDNS','MNST','SNPS','KDP',
    'AEP','MRNA','DXCM','TEAM','IDXX','EXC','FISV','CHTR','PANW','ORLY','EA','FAST','ROST','ODFL','MELI','CTAS','PAYX','XEL','PCAR','CEG','CTSH',
    'WBD','BKR','SIRI','BIIB','DLTR','WBA','SPLK','VRSK','SGEN','ANSS','TTD','ZS','SBAC','GFS','ON','LCID','ALGN','CPRT','VRSN','VTRS','PEP','FTNT'
]
universe = sorted(list(set(sp100 + nasdaq100)))

# --- HELPERS ---
def norm(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x).lower() for x in c if x]) for c in df.columns]
    else:
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    return df

def ensure_core_cols(df):
    req = ["close","open","high","low","volume"]
    missing = [x for x in req if x not in df.columns]
    if missing:
        raise Exception(f"Missing columns: {missing}")
    return df

def calc_indicators(df):
    # EMA, MACD, RSI
    df['ema10'] = df['close'].ewm(span=10, min_periods=10).mean()
    df['ema20'] = df['close'].ewm(span=20, min_periods=20).mean()
    df['ema50'] = df['close'].ewm(span=50, min_periods=50).mean()
    ema12 = df['close'].ewm(span=12, min_periods=12).mean()
    ema26 = df['close'].ewm(span=26, min_periods=26).mean()
    df['macd'] = ema12 - ema26
    df['macdsignal'] = df['macd'].ewm(span=9, min_periods=9).mean()
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(window=14).mean()
    roll_down = down.rolling(window=14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['rsi14'] = 100 - (100 / (1 + rs))
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

def local_time_str():
    return datetime.now(pytz.timezone("Asia/Singapore")).strftime('%Y-%m-%d %H:%M:%S')

def formatn(x, d=2):
    try: return f"{x:,.{d}f}"
    except: return x

# --- Streamlit UI ---
st.title("US Stock Screener: 1h KIV (Keep-In-View) Signal Bot")
st.caption("Detects 1h MACD/EMA/RSI setups for S&P 100 + NASDAQ 100, flags for pullback buy.")

st.sidebar.header("Screening Controls")
min_vol = st.sidebar.number_input("Min Volume (last bar)", value=100_000)
lookback_macd = st.sidebar.number_input("MACD Lookback (trend growth)", value=10)
rsi_min = st.sidebar.number_input("Min RSI", value=35)
rsi_max = st.sidebar.number_input("Max RSI", value=65)

# --- Main Signal Scan ---
results = []
for ticker in universe:
    try:
        df = yf.download(ticker, period="45d", interval="1h", progress=False, threads=False)
        if df.empty or len(df) < lookback_macd + 5: continue
        df = norm(df)
        df = ensure_core_cols(df)
        df = calc_indicators(df)
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        # --- Criteria:
        # 1. MACD crosses up signal (bull cross) and MACD crosses up baseline (0)
        macd_cross = (df['macd'].iloc[-2] < df['macdsignal'].iloc[-2]) and (df['macd'].iloc[-1] > df['macdsignal'].iloc[-1])
        macd_base_cross = (df['macd'].iloc[-1] > 0) and (df['macd'].iloc[-2] < 0)
        macd_trend = (df['macd'].iloc[-lookback_macd] < df['macd'].iloc[-5] < df['macd'].iloc[-1])
        macd_signal_below0 = df['macdsignal'].iloc[-1] < 0
        # 2. RSI within range and rising from lower, not dropping from high
        rsi_good = rsi_min <= curr['rsi14'] <= rsi_max and (df['rsi14'].iloc[-2] < curr['rsi14'])
        # 3. EMA 10 rising above EMA 20
        ema_good = curr['ema10'] > curr['ema20'] and curr['ema10'] > prev['ema10']
        # 4. Volume filter
        if curr['volume'] < min_vol: continue
        # --- All condition ---
        if macd_cross and macd_base_cross and macd_trend and macd_signal_below0 and rsi_good and ema_good:
            results.append({
                "Ticker": ticker,
                "Signal Time": local_time_str(),
                "MACD": formatn(curr['macd'], 4),
                "MACD Signal": formatn(curr['macdsignal'], 4),
                "RSI": formatn(curr['rsi14'], 2),
                "Price": formatn(curr['close'], 2),
                "Volume": int(curr['volume']),
                "Note": "KIV for 2% dip, aim for +5% TP, -2.5% SL"
            })
    except Exception as e:
        continue

# --- Results Table and Logging ---
if results:
    df_out = pd.DataFrame(results)
    st.subheader("🕵️ 1H KIV Signals (Potential Pullback Buys)")
    st.dataframe(df_out, use_container_width=True)
    try:
        append_to_gsheet(df_out.values.tolist(), GOOGLE_SHEET_NAME)
    except Exception as e:
        st.warning(f"GSHEET LOG: {e}")
else:
    st.info("No current 1h KIV signals found.")

# --- Show Recent Sheet Data (last 10) ---
try:
    client = get_gspread_client_from_secrets()
    sheet = client.open_by_key(GOOGLE_SHEET_ID).worksheet(GOOGLE_SHEET_NAME)
    data = sheet.get_all_values()
    if data and len(data) > 1:
        headers, rows = data[0], data[-10:]
        df_last = pd.DataFrame(rows, columns=headers)
        st.subheader("📜 Last 10 KIV Signals")
        st.dataframe(df_last)
except Exception as e:
    st.warning(f"Could not load recent sheet data: {e}")

st.caption("This dashboard only flags 'setup' stocks for watchlist. For each, consider entry if price dips 2% below signal, with +5% target, -2.5% stop.")
