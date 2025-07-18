import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, time as dt_time
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

# ---- Config ----
GOOGLE_SHEET_ID = "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4"
GOOGLE_SHEET_NAME = "MACD Cross"

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
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","GOOG","COST","PEP",
    "AVGO","QCOM","AMGN","TXN","CSCO","ADBE","TMUS","NFLX","AMD","HON",
    "SBUX","BKNG","INTC","AMAT","ADI","MDLZ","PDD","REGN","ISRG","VRTX",
    "GILD","ZM","FISV","MAR","PANW","MELI","LRCX","ADP","CRWD","ASML",
    "CTSH","CHTR","ORLY","KLAC","IDXX","CDNS","AEP","MNST","MRNA","BIDU",
    "ROST","CPRT","CTAS","WDAY","FAST","SGEN","DLTR","BIIB","ODFL","PAYX",
    "XEL","KDP","DXCM","PCAR","EA","CSGP","DDOG","SIRI","EXC","ANSS",
    "SPLK","VRSK","SGEN","ALGN","WBA","WBD","LCID","CEG","ZS","JD","FOXA",
    "OKTA","GEN","TEAM","VERU","PINS","DOCU","BKR","MTCH","MRVL","LULU",
    "ILMN","PEAK","KHC","HBAN","CDW","SGEN","TECH","WDC","TTD"
]
universe = sorted(set(sp100 + nasdaq100))

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
    # MACD
    ema12 = df['close'].ewm(span=12, min_periods=12).mean()
    ema26 = df['close'].ewm(span=26, min_periods=26).mean()
    df['macd'] = ema12 - ema26
    df['macdsignal'] = df['macd'].ewm(span=9, min_periods=9).mean()
    df['hist'] = df['macd'] - df['macdsignal']
    # RSI
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(window=14).mean()
    roll_down = down.rolling(window=14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['rsi14'] = 100 - (100 / (1 + rs))
    # Volume avg
    df['avgvol20'] = df['volume'].rolling(20).mean()
    return df

def get_market_sentiment():
    try:
        spy = yf.download('SPY', period='1d', interval='5m', progress=False, threads=False)
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

def local_time_str():
    return datetime.now(pytz.timezone("Asia/Singapore")).strftime('%Y-%m-%d %H:%M:%S')

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
min_vol = st.sidebar.number_input("Min Volume (last bar)", value=100_000)
min_avgvol = st.sidebar.number_input("Min Avg Volume (20 bars)", value=200_000)
rsi_min = st.sidebar.number_input("RSI Min", value=35, min_value=1, max_value=99)
rsi_max = st.sidebar.number_input("RSI Max", value=60, min_value=1, max_value=99)
show_debug = st.sidebar.checkbox("Show Debug Info", value=False)

# ---- Main ----
sentiment_pct, sentiment_text = get_market_sentiment()
st.title("MACD/EMA/RSI Stock Screener (S&P 100 + NASDAQ 100, 1h)")
st.caption(f"Last run: {local_time_str()}")
st.caption(f"Market Sentiment: {sentiment_text} ({sentiment_pct:.2f}%)")

if "alerted_today" not in st.session_state: st.session_state["alerted_today"] = set()
debug_issues, results, reference = [], [], []

for ticker in universe:
    try:
        df = yf.download(ticker, period='15d', interval='1h', progress=False, threads=False)
        if df.empty or len(df) < 35: continue
        df = norm(df)
        df = ensure_core_cols(df)
        df = calc_indicators(df)
        # Reference events: MACD cross up zero, signal still below zero
        curr, prev = df.iloc[-1], df.iloc[-2]
        cross_up = (prev['macd'] < 0) and (curr['macd'] > 0) and (curr['macdsignal'] < 0)
        if cross_up:
            reference.append({
                "Ticker": ticker,
                "Time": local_time_str(),
                "Close": formatn(curr['close']),
                "MACD": formatn(curr['macd'], 3),
                "MACD Signal": formatn(curr['macdsignal'], 3),
                "Hist": formatn(curr['hist'], 3),
                "RSI": formatn(curr['rsi14'], 2),
                "EMA10": formatn(curr['ema10'], 2),
                "EMA20": formatn(curr['ema20'], 2),
                "Vol": formatn(curr['volume'], 0)
            })
        # Screener: only apply main logic
        screener_cond = (
            cross_up and
            (rsi_min <= curr['rsi14'] <= rsi_max) and
            (curr['ema10'] > curr['ema20']) and
            (curr['hist'] > 0) and
            (min_price <= curr['close'] <= max_price) and
            (curr['volume'] >= min_vol) and
            (curr['avgvol20'] >= min_avgvol)
        )
        if screener_cond:
            results.append({
                "Ticker": ticker,
                "Time": local_time_str(),
                "Close": formatn(curr['close']),
                "MACD": formatn(curr['macd'], 3),
                "MACD Signal": formatn(curr['macdsignal'], 3),
                "Hist": formatn(curr['hist'], 3),
                "RSI": formatn(curr['rsi14'], 2),
                "EMA10": formatn(curr['ema10'], 2),
                "EMA20": formatn(curr['ema20'], 2),
                "Volume": formatn(curr['volume'], 0),
                "AvgVol20": formatn(curr['avgvol20'], 0)
            })
    except Exception as e:
        debug_issues.append({"Ticker": ticker, "Issue": str(e)})
        continue

# ---- OUTPUT TABLES ----
if results:
    df_out = pd.DataFrame(results)
    st.subheader("⭐ Filtered Signals (MACD/RSI/EMA/Vol)")
    st.dataframe(df_out, use_container_width=True)
    try:
        append_to_gsheet(df_out.values.tolist(), GOOGLE_SHEET_NAME)
    except Exception as e:
        st.warning(f"Failed to write to sheet: {e}")
else:
    st.info("No current signals found.")

if reference:
    df_ref = pd.DataFrame(reference)
    st.subheader("📝 Reference: MACD Cross Up Zero, Signal Below Zero (No Filter)")
    st.dataframe(df_ref, use_container_width=True)
else:
    st.info("No recent MACD cross-up-zero events for reference.")

if show_debug:
    st.subheader("Debug: Issues Encountered")
    if debug_issues:
        st.dataframe(pd.DataFrame(debug_issues))
    else:
        st.info("No issues.")

st.caption("© MACD Screener | S&P 100 & NASDAQ 100 | 1-hour bar. Screener saves all results to Google Sheet tab 'MACD Cross'.")
