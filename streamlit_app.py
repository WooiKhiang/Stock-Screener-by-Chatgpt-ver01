import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta, time as dt_time
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import time

# ---- Config ----
GOOGLE_SHEET_ID = "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4"
GOOGLE_SHEET_NAME = "MACD Cross"

# Load S&P 500 tickers (truncated for brevity; you can paste the full list!)
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
        if hasattr(sheet, 'append_rows'):
            sheet.append_rows(rows, value_input_option="USER_ENTERED")
        else:
            for row in rows:
                sheet.append_row(row, value_input_option="USER_ENTERED")
    except Exception as e:
        st.warning(f"Google Sheet ({sheet_name}): {e}")

def us_eastern_now():
    eastern = pytz.timezone("US/Eastern")
    return datetime.now(pytz.utc).astimezone(eastern)

def convert_to_eastern(idx):
    eastern = pytz.timezone("US/Eastern")
    if idx.tzinfo is None:
        idx = idx.tz_localize("UTC").tz_convert(eastern)
    else:
        idx = idx.tz_convert(eastern)
    return idx

# ---- Sidebar ----
st.sidebar.header("Filter Settings")
min_price = st.sidebar.number_input("Min Price ($)", value=10.0)
max_price = st.sidebar.number_input("Max Price ($)", value=2000.0)
min_vol = st.sidebar.number_input("Min Volume (1H)", value=100000)
rsi_thresh = st.sidebar.number_input("RSI Max", value=60)
max_days = st.sidebar.number_input("Max Signal Days (History)", value=7, min_value=1, max_value=30, step=1)

st.title("S&P 500: MACD Cross Screener (1H, Auto, Recency-Filtered, Google Sheet Push)")
st.caption(f"US/Eastern Now: {us_eastern_now().strftime('%Y-%m-%d %H:%M:%S')}")

results = []
batch_rows = []
eastern = pytz.timezone("US/Eastern")
now_et = us_eastern_now()

# --- Main Loop ---
for ticker in sp500:
    try:
        df = yf.download(ticker, period="15d", interval="1h", progress=False, threads=False)
        if df.empty or len(df) < 30: continue
        df = norm(df)
        df = ensure_core_cols(df)
        df = calc_indicators(df)
        # Make sure index is US/Eastern for time filters
        df.index = convert_to_eastern(pd.to_datetime(df.index))
        # Filter to only last max_days
        recent_limit = now_et - timedelta(days=int(max_days))
        df = df[df.index >= recent_limit]
        for i in range(1, len(df)):
            row, prev = df.iloc[i], df.iloc[i-1]
            # Criteria:
            # MACD crosses up zero, prev < 0, curr > 0
            cross = prev['macd'] < 0 and row['macd'] > 0
            macdsig_below = row['macdsignal'] < 0
            rsi_ok = row['rsi14'] < rsi_thresh
            ema_trend = row['ema10'] > row['ema20']
            hist_pos = row['hist'] > 0
            price = row['close']
            vol = row['volume']
            if (cross and macdsig_below and rsi_ok and ema_trend and hist_pos and
                (min_price <= price <= max_price) and (vol >= min_vol)):
                signal_time = row.name.strftime('%Y-%m-%d %H:%M')
                results.append({
                    "Time (US/Eastern)": signal_time,
                    "Ticker": ticker,
                    "Close": formatn(price,2),
                    "RSI": formatn(row['rsi14'],2),
                    "MACD": formatn(row['macd'],4),
                    "MACD Signal": formatn(row['macdsignal'],4),
                    "Hist": formatn(row['hist'],4),
                    "EMA10": formatn(row['ema10'],2),
                    "EMA20": formatn(row['ema20'],2),
                    "Vol": formatn(vol,0)
                })
                batch_rows.append([
                    signal_time, ticker, formatn(price,2), formatn(row['rsi14'],2),
                    formatn(row['macd'],4), formatn(row['macdsignal'],4), formatn(row['hist'],4),
                    formatn(row['ema10'],2), formatn(row['ema20'],2), formatn(vol,0)
                ])
    except Exception as e:
        continue

# ---- Output Table ----
st.subheader(f"MACD Cross Up Events (Last {int(max_days)} Days)")
if results:
    df_out = pd.DataFrame(results)
    st.dataframe(df_out, use_container_width=True)
    # --- Push to Google Sheets (once per run) ---
    try:
        append_to_gsheet(batch_rows, GOOGLE_SHEET_NAME)
    except Exception as e:
        st.warning(f"Google Sheet batch append error: {e}")
else:
    st.info("No MACD Cross events found within the last days filter.")

# ---- Signal Criteria Section ----
st.markdown("""
### Strategy Criteria
- **1H chart (US/Eastern)**
- **MACD crosses up zero** (prev bar < 0, curr bar > 0)
- **MACD signal line below zero** at cross
- **RSI (14) < sidebar threshold** (default 60)
- **EMA10 > EMA20** (trend confirmation)
- **MACD histogram positive**
- **Min price/volume filters**
- **Results shown for last X days only (sidebar filter)**
""")

st.caption("© AI Screener | S&P 500. Signals pushed to Google Sheet. Recency and volume filtering help ensure signal relevance and avoid overload.")
