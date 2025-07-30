import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yfinance as yf
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import time

# ---- Config ----
GOOGLE_SHEET_ID = "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4"
GOOGLE_SHEET_NAME = "MACD Cross"

sp500 = [
    "AAPL","MSFT","AMZN","NVDA","GOOGL","META","GOOG","BRK-B","LLY","UNH","TSLA","JPM","V","XOM","MA","AVGO",
    "PG","JNJ","COST","HD","MRK","ADBE","ABBV","CRM","WMT","AMD","PEP","KO","CVX","BAC","MCD","NFLX","DIS",
    "LIN","ACN","ABT","TMO","INTC","DHR","TXN","NEE","WFC","BMY","PM","UNP","QCOM","HON","LOW","RTX","AMGN",
    "SBUX","MS","NKE","VZ","COP","SCHW","GS","AMAT","INTU","MDT","ISRG","CAT","T","GE","SPGI","IBM","LMT",
    "BLK","ELV","EL","DE","ADP","CB","NOW","MDLZ","PLD","PYPL","CSCO","MU","VRTX","DUK","SYK","SO","GILD",
    "AXP","ZTS","BKNG","TJX","REGN","BDX","MMC","CI","ADI","PGR","TGT","ITW","SLB","MO","EW","APD","HCA",
    "C","PNC","SHW","FI","USB","LRCX","EMR","ORCL","WM","FISV","AON","FDX","ICE","TRV","ETN","GM","DG",
    "ROP","MCO","NSC","KLAC","PRU","AIG","EOG","MCK","AFL","SPG","SRE","IDXX","ALL","ODFL","AEP","PSA",
    "MAR","ADM","D","CTAS","STZ","ROST","DOW","YUM","BIIB","HLT","PEG","CMG","EXC","CDNS","MCHP","TT",
    "DLR","CTVA","MSCI","HSY","CME","WELL","F","GWW","SYY","HAL","TROW","XEL","CPRT","NEM","TSCO","IQV",
    "WMB","OXY","KHC","OTIS","PH","MRNA","SBAC","VRSK","PPG","KMB","ED","NUE","RMD","LEN","PCAR","STT",
    "MTD","ILMN","MNST","A","HUM","ECL","AEE","XYL","AWK","AME","CMI","AZO","FAST","BAX","PPL","CHD",
    "CNC","TSN","EFX","EIX","DVN","FLT","AMP","FRC","RSG","AAL","CL","BALL","AVB","HRL","BXP",
    "ESS","DGX","LH","QRVO","VTR","WAT","BKR","BEN","COO","NDAQ","MGM","PWR","ZBH","UHS","LYB","WAB",
    "AKAM","NVR","NTAP","PFG","MAS","KEYS","MTB","J","L","CBOE","UAL","APA","MKTX","RJF","VNO",
    "DHI","FFIV","HSIC","NWL","SEE","HWM","GL","RF","BIO","IRM","WRB","HOLX","NRG","CNP","ALK","HII",
    "ALLE","VFC","WY","NOV","GNRC","IPG","AOS","LUMN","NWSA","FOX","NWS","LW","CPB","JBHT",
    "DISCK","DISCA","DVA","ZION","LKQ","IVZ","CF","NDSN","ROL","FRT","NCLH","CMA","AIZ","FANG","PKG",
    "AAP","DRI","LNT","STX","NRZ","MOS","KIM","TPR","WHR","IP","SWK","HAS","CZR","EMN","UA","UAA",
    "AES","ANET","BR","BRO","CHRW","CTLT","DLTR","DOV","EBAY",
    "EPAM","ETSY","EXPD","FDS","FTNT","GEN","GRMN","HUBB","INVH","KEY",
    "KDP","LDOS","MKC","MPWR","NTRS","OGN","ON","PARA","PAYC",
    "PAYX","PEAK","PTC","PWR","RE","RHI","SNPS","STE","TDG",
    "TECH","TER","URI","VMC","WDC","WTW","ZBRA","DASH", "TKO", "WSM", "EXE", "COIN", "TTD", "SQ"
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
    df['histogram'] = df['macd'] - df['macdsignal']
    return df

def us_local_time():
    return datetime.now(pytz.timezone("US/Eastern")).strftime('%Y-%m-%d %H:%M:%S')

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

# --- Auto Refresh (every 5min) ---
st_autorefresh(interval=5*60*1000, key="autorefresh")  # 5 min

# --- Sidebar ---
st.sidebar.header("MACD Cross Screener Filters")
min_price = st.sidebar.number_input("Min Price ($)", value=5.0)
max_price = st.sidebar.number_input("Max Price ($)", value=2000.0)
min_vol = st.sidebar.number_input("Min Avg Volume (10 bars)", value=100_000)
max_rsi = st.sidebar.number_input("Max RSI (14)", value=60)

# --- Strategy Explanation ---
with st.expander("📘 Strategy: MACD Cross Screener Logic", expanded=True):
    st.markdown("""
**Entry Criteria:**  
- 1H bar (US/Eastern, S&P 500 universe)
- MACD crosses above zero (**current MACD > 0**; MACD signal < 0)
- RSI(14) below threshold (default 60, adjustable in sidebar)
- EMA10 > EMA20 (trend confirmation)
- MACD histogram positive (bullish momentum)
- Current price in your min/max range
- 10-bar average volume above threshold

**Reference table below** shows any ticker with MACD>0, Signal<0 regardless of other filters.
""")

# --- Main Table ---
st.title("MACD Cross Zero (US S&P 500 Screener)")
st.caption(f"Last run: {us_local_time()}")

results, reference_rows = [], []
progress = st.progress(0)
for i, ticker in enumerate(sp500):
    try:
        df = yf.download(ticker, period="10d", interval="1h", progress=False, threads=False)
        if df.empty or len(df) < 40: continue
        df = norm(df)
        df = ensure_core_cols(df)
        df = calc_indicators(df)
        last = df.iloc[-1]
        # Main Screener Table (with filters)
        if (
            last['macd'] > 0 and
            last['macdsignal'] < 0 and
            last['rsi14'] <= max_rsi and
            last['ema10'] > last['ema20'] and
            last['histogram'] > 0 and
            min_price <= last['close'] <= max_price and
            df['volume'][-10:].mean() >= min_vol
        ):
            results.append({
                "Ticker": ticker,
                "US Time": us_local_time(),
                "Price": formatn(last['close']),
                "RSI": formatn(last['rsi14'], 2),
                "MACD": formatn(last['macd'], 4),
                "MACD Signal": formatn(last['macdsignal'], 4),
                "Hist": formatn(last['histogram'], 4),
                "EMA10": formatn(last['ema10'], 2),
                "EMA20": formatn(last['ema20'], 2),
                "AvgVol10": formatn(df['volume'][-10:].mean(), 0)
            })
        # Reference Table: Any cross up (no RSI/EMA/vol filters)
        if last['macd'] > 0 and last['macdsignal'] < 0:
            reference_rows.append({
                "Ticker": ticker,
                "US Time": us_local_time(),
                "Price": formatn(last['close']),
                "RSI": formatn(last['rsi14'], 2),
                "MACD": formatn(last['macd'], 4),
                "MACD Signal": formatn(last['macdsignal'], 4),
                "Hist": formatn(last['histogram'], 4),
                "EMA10": formatn(last['ema10'], 2),
                "EMA20": formatn(last['ema20'], 2),
                "AvgVol10": formatn(df['volume'][-10:].mean(), 0)
            })
    except Exception as e:
        continue
    progress.progress((i + 1) / len(sp500))

# --- Display Main Screener Table ---
if results:
    st.subheader("⭐ MACD Cross Signals (Filtered)")
    df_out = pd.DataFrame(results)
    st.dataframe(df_out, use_container_width=True)
    # ---- Google Sheet Export ----
    try:
        append_to_gsheet(df_out.values.tolist(), GOOGLE_SHEET_NAME)
    except Exception as e:
        st.warning(f"Google Sheet Export Error: {e}")
else:
    st.info("No filtered MACD cross signals found for current bar.")

# --- Reference Table (all MACD>0, Signal<0, no filters) ---
st.subheader("🕵️ Reference: MACD Cross Up (MACD > 0, Signal < 0, All tickers, no filters)")
if reference_rows:
    st.dataframe(pd.DataFrame(reference_rows), use_container_width=True)
else:
    st.info("No reference MACD cross events found for current bar.")

st.caption("Auto-refreshes every 5 min. S&P500, US/Eastern time. Main screener pushes to Google Sheet.")
