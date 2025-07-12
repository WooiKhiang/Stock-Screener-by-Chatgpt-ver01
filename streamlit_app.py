import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, time as dt_time
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import time

# ---- Config ----
GOOGLE_SHEET_ID = "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4"
GOOGLE_SHEET_NAME = "KIV_SIGNALS"

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
    'AAPL','MSFT','AMZN','NVDA','GOOGL','META','GOOG','AVGO','TSLA','COST',
    'PEP','ADBE','CSCO','CMCSA','NFLX','AMD','TXN','HON','QCOM','INTC',
    'AMGN','INTU','AMAT','ISRG','BKNG','MDLZ','LRCX','ADI','SBUX','VRTX',
    'PDD','GILD','ZM','REGN','MELI','CRWD','KDP','SNPS','CTAS','PANW',
    'KHC','CDNS','MNST','ADP','AEP','WDAY','MAR','ASML','MU','NXPI',
    'DXCM','EXC','IDXX','CSX','ORLY','ODFL','FAST','PCAR','PAYX','MCHP',
    'XEL','MRVL','SGEN','LCID','TEAM','ROST','CTSH','BKR','EBAY','BIIB',
    'DLTR','ANSS','EA','SIRI','CEG','CHTR','JD','SWKS','VRSK','ALGN',
    'ON','SPLK','WBA','ILMN','ZS','SIRI','OKTA','PTON','VRSN','LULU',
    'DOCU','FOXA','FOX','MKTX','TTD','NDAQ','TTWO','WDC','SIRI'
]

universe = sorted(set(sp100 + nasdaq100))

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
    # MACD
    ema12 = df['close'].ewm(span=12, min_periods=12).mean()
    ema26 = df['close'].ewm(span=26, min_periods=26).mean()
    df['macd'] = ema12 - ema26
    df['macdsignal'] = df['macd'].ewm(span=9, min_periods=9).mean()
    # RSI
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(window=14).mean()
    roll_down = down.rolling(window=14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['rsi14'] = 100 - (100 / (1 + rs))
    # EMA
    df['ema10'] = df['close'].ewm(span=10, min_periods=10).mean()
    df['ema20'] = df['close'].ewm(span=20, min_periods=20).mean()
    return df

def local_time_str():
    return datetime.now(pytz.timezone("Asia/Singapore")).strftime('%Y-%m-%d %H:%M:%S')

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

# ---- Market Sentiment ----
sentiment_pct, sentiment_text = get_market_sentiment()
st.title("AI-Powered US Stocks Screener: 1h KIV + MACD Reference Table")
st.caption(f"Last run: {local_time_str()}")
st.subheader(f"Market Sentiment: {sentiment_text} ({sentiment_pct:.2f}%)")

# ---- Screening ----
st.sidebar.header("KIV Signal Parameters")
min_vol = st.sidebar.number_input("Min Volume (last 1h bar)", value=100000)
min_price = st.sidebar.number_input("Min Price ($)", value=10.0)
max_price = st.sidebar.number_input("Max Price ($)", value=2000.0)

kiv_results = []
macd_crosses = []

for ticker in universe:
    try:
        df = yf.download(ticker, period='20d', interval='1h', progress=False, threads=False)
        if df.empty or len(df) < 15:
            continue
        df = norm(df)
        df = ensure_core_cols(df)
        df = calc_indicators(df)
        # Main KIV Signal logic
        macd, macdsignal = df['macd'].iloc[-1], df['macdsignal'].iloc[-1]
        macd_prev = df['macd'].iloc[-2]
        rsi = df['rsi14'].iloc[-1]
        ema10, ema20 = df['ema10'].iloc[-1], df['ema20'].iloc[-1]
        close = df['close'].iloc[-1]
        vol = df['volume'].iloc[-1]
        # Criteria 1: KIV
        # 1. MACD crosses up zero, MACD signal below zero
        macd_cross_up = (macd_prev < 0 and macd > 0)
        macdsignal_still_neg = macdsignal < 0
        # 2. MACD trend 10d < 5d < now
        macd_10ago = df['macd'].iloc[-10] if len(df) >= 11 else None
        macd_5ago = df['macd'].iloc[-5] if len(df) >= 6 else None
        macd_trending = macd_10ago is not None and macd_5ago is not None and macd_10ago < macd_5ago < macd
        # 3. RSI 35-65, rising (from below, not falling from 70)
        rsi_rising = rsi > 35 and rsi < 65 and (df['rsi14'].iloc[-5] < rsi)
        # 4. EMA10 above EMA20, EMA10 rising
        ema_ok = ema10 > ema20 and (df['ema10'].iloc[-5] < ema10)
        # All conditions for KIV:
        cond_kiv = all([
            macd_cross_up,
            macdsignal_still_neg,
            macd_trending,
            rsi_rising,
            ema_ok,
            min_price <= close <= max_price,
            vol >= min_vol
        ])
        if cond_kiv:
            kiv_results.append({
                "Ticker": ticker,
                "Signal Time": local_time_str(),
                "MACD": formatn(macd, 4),
                "MACD Signal": formatn(macdsignal, 4),
                "RSI": formatn(rsi, 2),
                "Price": formatn(close, 2),
                "Volume": int(vol),
                "Note": "KIV for 2% dip, aim for +5% TP, -2.5% SL"
            })
        # 2. Reference Table: MACD Crosses Zero (no other filter)
        macd_ref_cross = (macd_prev < 0 and macd > 0) and (macdsignal < 0)
        if macd_ref_cross:
            macd_crosses.append({
                "Ticker": ticker,
                "Signal Time": local_time_str(),
                "MACD": formatn(macd, 4),
                "MACD Signal": formatn(macdsignal, 4),
                "Price": formatn(close, 2),
                "Volume": int(vol),
            })
        time.sleep(0.04)
    except Exception as e:
        continue

# ---- Display KIV Table ----
st.subheader("📋 1h KIV Signal Setups (Strict All-Criteria Matches)")
if kiv_results:
    df_kiv = pd.DataFrame(kiv_results)
    st.dataframe(df_kiv, use_container_width=True)
    try:
        append_to_gsheet(df_kiv.values.tolist(), GOOGLE_SHEET_NAME)
    except Exception as e:
        st.warning(f"Google Sheet log error: {e}")
else:
    st.info("No current 1h KIV signals found.")

# ---- Show last 10 signals from Google Sheet ----
try:
    client = get_gspread_client_from_secrets()
    sheet = client.open_by_key(GOOGLE_SHEET_ID).worksheet(GOOGLE_SHEET_NAME)
    data = sheet.get_all_values()
    if data and len(data) > 1:
        headers, rows = data[0], data[-10:]
        df_last = pd.DataFrame(rows, columns=headers)
        st.subheader("🕒 Last 10 KIV Signals (from Google Sheet)")
        st.dataframe(df_last)
except Exception as e:
    st.warning(f"Could not load recent sheet data: {GOOGLE_SHEET_NAME}")

# ---- Reference: MACD Crosses Zero Table ----
st.subheader("📑 Reference: MACD Crosses Zero, Signal Still Below Zero (No RSI/EMA Filter)")
if macd_crosses:
    df_cross = pd.DataFrame(macd_crosses)
    st.dataframe(df_cross, use_container_width=True)
else:
    st.info("No MACD cross-zero signals at this moment.")

st.caption("© AI Screener | S&P 100 + NASDAQ 100. 1h chart. Market sentiment adjusts screening.")
