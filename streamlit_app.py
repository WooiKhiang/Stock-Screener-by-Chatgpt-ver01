import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, time as dt_time
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

# --- CONFIG ---
GOOGLE_SHEET_ID = "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4"
GOOGLE_SHEET_NAME = "MACD Cross"

# --- Tickers (S&P 500) ---
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
    "CNC","TSN","EFX","EIX","DVN","FLT","AMP","FRC","RSG","AAL","CL","BALL","AVB","HRL","XYL","BXP",
    "ESS","DGX","LH","QRVO","VTR","WAT","BKR","BEN","COO","NDAQ","MGM","PWR","ZBH","UHS","LYB","WAB",
    "AKAM","NVR","NTAP","PFG","MAS","KEYS","MTB","HES","J","L","CBOE","UAL","APA","MKTX","RJF","VNO",
    "DHI","FFIV","HSIC","NWL","SEE","HWM","GL","RF","BIO","IRM","WRB","HOLX","NRG","CNP","ALK","HII",
    "ALLE","VFC","WY","NOV","GNRC","IPG","AOS","LUMN","NWSA","FOX","NWS","FOX","FMC","LW","CPB","JBHT",
    "DISCK","DISCA","DVA","ZION","LKQ","IVZ","CF","NDSN","ROL","FRT","NCLH","CMA","AIZ","FANG","PKG",
    "AAP","DRI","LNT","STX","NRZ","MOS","KIM","TPR","WHR","IP","SWK","HAS","CZR","EMN","UA","UAA","AAL"
]

# --- Helper Functions ---
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

def local_time_str_us(dt):
    # Convert pandas Timestamp to US/Eastern
    eastern = pytz.timezone("US/Eastern")
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    dt_et = dt.astimezone(eastern)
    return dt_et.strftime('%Y-%m-%d %H:%M:%S')

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

# --- Sidebar (Filters) ---
st.sidebar.header("MACD Cross Strategy Filters")
min_price = st.sidebar.number_input("Min Price ($)", value=10.0)
min_vol = st.sidebar.number_input("Min Avg Vol (last 10 bars)", value=100_000)
rsi_max = st.sidebar.number_input("Max RSI (default 60)", value=60)
max_rows = st.sidebar.number_input("Events to Display (History)", value=10, min_value=5, max_value=50)

# --- Market Sentiment ---
sentiment_pct, sentiment_text = get_market_sentiment()
st.title("AI MACD Cross Screener: S&P 500, 1H, Zero Cross")
st.caption(f"Market Sentiment: {sentiment_text} ({sentiment_pct:.2f}%)")

# --- Anti-spam session state (only unique per ticker/bar timestamp in session) ---
if "alerted_today" not in st.session_state:
    st.session_state["alerted_today"] = set()

results = []
ref_crosses = []

# --- Main Screener Loop ---
for ticker in sp500:
    try:
        df = yf.download(ticker, period='60d', interval='1h', progress=False, threads=False)
        if df.empty or len(df) < 30:
            continue
        df = norm(df)
        df = ensure_core_cols(df)
        df = calc_indicators(df)

        # Only include regular trading hours
        eastern = pytz.timezone("US/Eastern")
        df.index = pd.to_datetime(df.index).tz_localize(None).tz_localize("UTC").tz_convert(eastern)
        df = df.between_time("09:30", "16:00")

        avg_vol = df['volume'].rolling(10).mean()
        for i in range(1, len(df)):
            # --- Reference cross table (any cross-up, regardless of filters) ---
            cross_up = (df['macd'].iloc[i-1] < 0) and (df['macd'].iloc[i] >= 0) and (df['macdsignal'].iloc[i] < 0)
            if cross_up:
                row = df.iloc[i]
                ref_crosses.append({
                    "Ticker": ticker,
                    "Bar Time (US/Eastern)": row.name.strftime('%Y-%m-%d %H:%M'),
                    "Close": formatn(row['close']),
                    "MACD": formatn(row['macd'], 4),
                    "MACD Signal": formatn(row['macdsignal'], 4),
                    "RSI": formatn(row['rsi14'], 2),
                    "EMA10": formatn(row['ema10'], 2),
                    "EMA20": formatn(row['ema20'], 2),
                    "Volume": formatn(row['volume'], 0),
                })
            # --- Main screener logic ---
            if cross_up:
                row = df.iloc[i]
                # Illiquid filter
                if row['close'] < min_price: continue
                if avg_vol.iloc[i] < min_vol: continue
                # Screener filters:
                if not (row['rsi14'] < rsi_max): continue
                if not (row['ema10'] > row['ema20']): continue
                if not (row['hist'] > 0): continue
                # Anti-duplicate (per ticker, per bar timestamp)
                sigid = (ticker, row.name)
                if sigid in st.session_state["alerted_today"]:
                    continue
                st.session_state["alerted_today"].add(sigid)
                results.append({
                    "Ticker": ticker,
                    "Bar Time (US/Eastern)": row.name.strftime('%Y-%m-%d %H:%M'),
                    "Close": formatn(row['close']),
                    "MACD": formatn(row['macd'], 4),
                    "MACD Signal": formatn(row['macdsignal'], 4),
                    "RSI": formatn(row['rsi14'], 2),
                    "EMA10": formatn(row['ema10'], 2),
                    "EMA20": formatn(row['ema20'], 2),
                    "Volume": formatn(row['volume'], 0),
                })
    except Exception as e:
        continue

# --- Screener Results Table ---
if results:
    df_out = pd.DataFrame(results)
    st.subheader("🟢 MACD Cross Up Zero (Screener, All Criteria Met)")
    st.dataframe(df_out.tail(max_rows), use_container_width=True)
    # Google Sheets push
    try:
        append_to_gsheet(df_out.values.tolist(), GOOGLE_SHEET_NAME)
    except Exception as e:
        st.warning(f"Sheet log error: {e}")
else:
    st.info("No MACD cross-up signals found with current filters.")

# --- Reference Table (All MACD Cross-Up Events, No Filter) ---
if ref_crosses:
    df_ref = pd.DataFrame(ref_crosses)
    st.subheader("📊 Reference: All MACD Zero Crosses (Signal < 0, No RSI/EMA/Vol Filter)")
    st.dataframe(df_ref.tail(max_rows), use_container_width=True)
else:
    st.info("No reference MACD cross-up events found.")

# --- Strategy Description ---
st.markdown("""
---
**MACD Cross Strategy (Screener Criteria)**  
- 1H Chart, US regular hours only (09:30–16:00 US/Eastern)  
- MACD crosses up zero (prev < 0, curr >= 0), MACD signal still < 0  
- RSI (14) below sidebar max (default 60)  
- EMA10 > EMA20  
- MACD histogram > 0  
- Min close price & avg volume (last 10 bars), adjustable in sidebar  
- Only one alert per ticker per bar (anti-duplicate, session-based)  

**Reference Table:**  
Shows all bars where MACD crossed up zero with MACD signal < 0, no further filters.
""")
