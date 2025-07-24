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

# ---- FUNCTIONS ----

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

def get_us_eastern_time(dt=None):
    eastern = pytz.timezone("US/Eastern")
    if dt is None:
        return datetime.now(pytz.utc).astimezone(eastern)
    return dt.astimezone(eastern)

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

def load_recent_signals(sheet_name, days=7):
    try:
        client = get_gspread_client_from_secrets()
        sheet = client.open_by_key(GOOGLE_SHEET_ID).worksheet(sheet_name)
        data = sheet.get_all_values()
        # (Time, Ticker) for deduplication
        recents = set()
        if data and len(data) > 1:
            for row in data[1:]:
                date_str, ticker = row[0], row[1]
                try:
                    dt = pd.to_datetime(date_str)
                    if dt >= pd.Timestamp.now(tz=pytz.timezone("US/Eastern")) - pd.Timedelta(days=days):
                        recents.add((date_str, ticker))
                except Exception:
                    continue
        return recents
    except Exception as e:
        st.warning(f"Sheet load error: {e}")
        return set()

def get_market_sentiment():
    try:
        spy = yf.download('SPY', period='1d', interval='5m', progress=False, threads=False)
        if spy.empty: return 0, "Unknown"
        spy = norm(spy)
        spy = ensure_core_cols(spy)
        open_, last = spy['open'].iloc[0], spy['close'].iloc[-1]
        pct = (last - open_) / open_ * 100
        if pct > 0.5: return pct, "🟢 Bullish"
        elif pct < -0.5: return pct, "🔴 Bearish"
        else: return pct, "🟡 Sideways"
    except Exception:
        return 0, "Unknown"

# ---- SIDEBAR ----
st.sidebar.header("KIV Signal Parameters")
min_volume = st.sidebar.number_input("Min Avg Volume (last 10 bars)", value=100000)
min_price = st.sidebar.number_input("Min Price ($)", value=10.0)
max_price = st.sidebar.number_input("Max Price ($)", value=2000.0)
rsi_thresh = st.sidebar.slider("RSI (14) threshold", 40, 80, 60)
lookback_days = st.sidebar.slider("Lookback Days", 1, 7, 7)

# ---- SENTIMENT ----
sentiment_pct, sentiment_text = get_market_sentiment()
st.title("AI-Powered US Stocks Screener: MACD Crosses Zero (S&P 500)")
st.caption(f"Last run: {get_us_eastern_time().strftime('%Y-%m-%d %H:%M:%S')} US/Eastern")
st.markdown(f"**Market Sentiment:** {sentiment_text} ({sentiment_pct:.2f}%)")

# ---- STRATEGY SUMMARY ----
st.info("""
**Current Screener Criteria**  
• 1h chart (US/Eastern time, pre/post/regular)  
• MACD crosses up zero (prev bar MACD < 0, curr MACD > 0)  
• MACD signal line is still < 0 at cross  
• RSI (14) below sidebar threshold (default 60)  
• EMA10 > EMA20 (trend up confirmation)  
• MACD histogram positive  
• Min price and min average volume filter (adjustable)  
• All signals in last N days (default 7), deduplicated for Google Sheets  
""")

# ---- RECENT SIGNALS FROM SHEET ----
recent_signals = load_recent_signals(GOOGLE_SHEET_NAME, days=lookback_days)
rows_to_append = []
kiv_results = []
history_tbl = []

# ---- MAIN SCAN ----
for ticker in stqdm(sp500, desc="Scanning tickers..."):
    try:
        df = yf.download(ticker, period=f"{lookback_days+3}d", interval="1h", progress=False, threads=False)
        if df.empty or len(df) < 20: continue
        df = norm(df)
        df = ensure_core_cols(df)
        df = calc_indicators(df)
        df = df.dropna()
        df['us_time'] = df.index.tz_convert('US/Eastern')
        # Min avg volume last 10 bars
        df['avg_vol_10'] = df['volume'].rolling(10).mean()
        # For each bar in last lookback_days, check the signal logic
        for i in range(1, len(df)):
            dt_bar = df['us_time'].iloc[i]
            # Only keep bars in lookback period
            if dt_bar < pd.Timestamp.now(tz=pytz.timezone("US/Eastern")) - pd.Timedelta(days=lookback_days):
                continue
            prev_macd = df['macd'].iloc[i-1]
            curr_macd = df['macd'].iloc[i]
            curr_signal = df['macdsignal'].iloc[i]
            curr_hist = df['hist'].iloc[i]
            curr_rsi = df['rsi14'].iloc[i]
            curr_ema10 = df['ema10'].iloc[i]
            curr_ema20 = df['ema20'].iloc[i]
            price = df['close'].iloc[i]
            avg_vol = df['avg_vol_10'].iloc[i]
            hist_row = {
                "Time": dt_bar.strftime("%Y-%m-%d %H:%M"),
                "Ticker": ticker,
                "MACD": formatn(curr_macd,4),
                "MACD Signal": formatn(curr_signal,4),
                "RSI": formatn(curr_rsi,2),
                "EMA10": formatn(curr_ema10,2),
                "EMA20": formatn(curr_ema20,2),
                "Hist": formatn(curr_hist,4),
                "Price": formatn(price,2),
                "AvgVol10": formatn(avg_vol,0)
            }
            # History table (all MACD crosses in last lookback_days)
            if prev_macd < 0 and curr_macd > 0 and curr_signal < 0:
                history_tbl.append(hist_row)
            # Screener criteria (main signal)
            if (
                prev_macd < 0 and curr_macd > 0 and curr_signal < 0 and
                curr_hist > 0 and
                curr_rsi < rsi_thresh and
                curr_ema10 > curr_ema20 and
                min_price <= price <= max_price and
                avg_vol >= min_volume
            ):
                row_key = (dt_bar.strftime("%Y-%m-%d %H:%M"), ticker)
                if row_key in recent_signals: continue # dedupe
                kiv_results.append(hist_row)
                rows_to_append.append([
                    dt_bar.strftime("%Y-%m-%d %H:%M"), ticker,
                    formatn(price,2), formatn(curr_rsi,2), formatn(curr_macd,4), formatn(curr_signal,4),
                    formatn(curr_hist,4), formatn(curr_ema10,2), formatn(curr_ema20,2), formatn(avg_vol,0)
                ])
                recent_signals.add(row_key)
    except Exception as e:
        continue

# ---- DISPLAY TABLES ----
st.subheader("📈 1h MACD Cross Signals (Current Screener Criteria)")
if kiv_results:
    df_out = pd.DataFrame(kiv_results)
    st.dataframe(df_out, use_container_width=True)
    if rows_to_append:
        append_to_gsheet(rows_to_append, GOOGLE_SHEET_NAME)
else:
    st.info("No MACD cross signals found with all criteria.")

st.subheader("🕒 MACD Crosses Zero Events (Last 10 Bars History, No Filter)")
if history_tbl:
    df_hist = pd.DataFrame(history_tbl).tail(50) # show last 50 for reference
    st.dataframe(df_hist, use_container_width=True)
else:
    st.info("No recent MACD cross-up events found.")

# ---- END ----
st.caption("© AI Screener | S&P 500 | 1h Chart. MACD Cross. US/Eastern time. See sidebar for settings. All results auto-deduped to Google Sheet.")
