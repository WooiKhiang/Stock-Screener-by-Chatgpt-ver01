import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

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

if "sent_today" not in st.session_state:
    st.session_state["sent_today"] = set()

def is_already_sent(ticker, dt_et):
    key = f"{ticker}-{dt_et.strftime('%Y-%m-%d %H:%M')}"
    if key in st.session_state["sent_today"]:
        return True
    st.session_state["sent_today"].add(key)
    return False

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

def formatn(num, d=2):
    try:
        if num is None or num == "" or np.isnan(num): return "-"
        if isinstance(num, int) or d == 0:
            return f"{int(num):,}"
        return f"{num:,.{d}f}"
    except Exception:
        return str(num)

def norm(df):
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    return df

st.sidebar.header("MACD Cross Screener Settings")
rsi_max = st.sidebar.number_input("Max RSI", value=60, min_value=20, max_value=100)
min_price = st.sidebar.number_input("Min Price ($)", value=10.0)
max_price = st.sidebar.number_input("Max Price ($)", value=2000.0)
min_vol = st.sidebar.number_input("Min Volume (avg last 10 bars)", value=100_000)

with st.expander("ℹ️ **Screener Strategy Explained**", expanded=True):
    st.markdown("""
**Signal Criteria (Main Table):**
- 1H Chart (US/Eastern time).
- MACD crosses UP zero (prev bar MACD < 0, curr MACD > 0).
- MACD signal line still < 0 at cross.
- RSI (14) below sidebar threshold (default 60).
- EMA10 > EMA20 (trend confirmation).
- MACD histogram positive.
- Min price/volume filter (customizable).
- Only one alert per ticker per hour-bar (spam-guarded).

**Reference Table (All Crosses):**
- Shows all S&P 500 tickers that had *any* MACD cross up zero with signal < 0, **last 10 bars**, regardless of RSI or EMA.
- For study/historical checks—no filters.

**History Table (Signals):**
- Recent signals that fired (met ALL main criteria), last 10 rows from Google Sheet, for research and review.
""")

st.title("MACD Cross Up Zero Screener (S&P 500, US ET)")
st.caption(f"Last run: {datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M')} (US/Eastern)")

results = []
ref_crosses = []
bars_to_check = 10  # For reference/history, last 10 bars

tickers_checked = 0
bars_fetched = 0

for ticker in sp500:
    try:
        df = yf.download(ticker, period="30d", interval="1h", progress=False, threads=False)
        if df.empty or len(df) < bars_to_check+2:
            continue
        df = norm(df)
        df["ema10"] = df["close"].ewm(span=10, min_periods=10).mean()
        df["ema20"] = df["close"].ewm(span=20, min_periods=20).mean()
        delta = df["close"].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        roll_up = up.rolling(window=14).mean()
        roll_down = down.rolling(window=14).mean()
        rs = roll_up / (roll_down + 1e-9)
        df["rsi14"] = 100 - (100 / (1 + rs))
        ema12 = df["close"].ewm(span=12, min_periods=12).mean()
        ema26 = df["close"].ewm(span=26, min_periods=26).mean()
        df["macd"] = ema12 - ema26
        df["macdsignal"] = df["macd"].ewm(span=9, min_periods=9).mean()
        df["hist"] = df["macd"] - df["macdsignal"]
        df["avgvol"] = df["volume"].rolling(10).mean()

        tickers_checked += 1
        bars_fetched += len(df)

        # Reference: Show all MACD cross up events (last 10 bars)
        for idx in range(-bars_to_check, 0):
            curr = df.iloc[idx]
            prev = df.iloc[idx-1]
            dt_utc = curr.name.tz_localize("UTC") if curr.name.tzinfo is None else curr.name
            dt_et = dt_utc.tz_convert("US/Eastern")
            cond_macd_cross = prev["macd"] < 0 and prev["macd"] < prev["macdsignal"] and curr["macd"] > 0 and curr["macd"] > curr["macdsignal"] and curr["macdsignal"] < 0
            if cond_macd_cross:
                ref_crosses.append({
                    "Ticker": ticker,
                    "Time": dt_et.strftime("%Y-%m-%d %H:%M"),
                    "Price": formatn(curr["close"]),
                    "MACD": formatn(curr["macd"], 4),
                    "Signal": formatn(curr["macdsignal"], 4),
                    "RSI": formatn(curr["rsi14"], 2),
                    "EMA10": formatn(curr["ema10"]),
                    "EMA20": formatn(curr["ema20"]),
                    "Volume": formatn(curr["volume"], 0),
                    "Hist": formatn(curr["hist"], 4)
                })

        # Main Signal: Only fire on current bar if meets all criteria
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        dt_utc = curr.name.tz_localize("UTC") if curr.name.tzinfo is None else curr.name
        dt_et = dt_utc.tz_convert("US/Eastern")
        cond_macd_cross = prev["macd"] < 0 and prev["macd"] < prev["macdsignal"] and curr["macd"] > 0 and curr["macd"] > curr["macdsignal"] and curr["macdsignal"] < 0
        cond_rsi = curr["rsi14"] < rsi_max
        cond_ema = curr["ema10"] > curr["ema20"]
        cond_hist = curr["hist"] > 0
        cond_price = min_price <= curr["close"] <= max_price
        cond_vol = curr["avgvol"] > min_vol

        if cond_macd_cross and cond_rsi and cond_ema and cond_hist and cond_price and cond_vol:
            if is_already_sent(ticker, dt_et):
                continue
            row = {
                "Ticker": ticker,
                "Price": formatn(curr["close"]),
                "MACD": formatn(curr["macd"], 4),
                "Signal": formatn(curr["macdsignal"], 4),
                "RSI": formatn(curr["rsi14"], 2),
                "EMA10": formatn(curr["ema10"]),
                "EMA20": formatn(curr["ema20"]),
                "Volume": formatn(curr["volume"], 0),
                "Hist": formatn(curr["hist"], 4),
                "Time": dt_et.strftime("%Y-%m-%d %H:%M (ET)")
            }
            results.append(row)
    except Exception as e:
        continue

# --- Main Signals Table ---
if results:
    df_out = pd.DataFrame(results)
    st.subheader("📈 New MACD Cross Up Zero Signals (S&P 500, US ET, No Spam)")
    st.dataframe(df_out, use_container_width=True)
    # Push to Google Sheet
    gsheet_rows = [
        [row["Time"], row["Ticker"], row["Price"], row["MACD"], row["Signal"], row["RSI"], row["EMA10"], row["EMA20"], row["Volume"], row["Hist"]]
        for row in results
    ]
    append_to_gsheet(gsheet_rows, GOOGLE_SHEET_NAME)
else:
    st.info("No current MACD cross up zero signals found (spam guard active).")

# --- History Table: Last 10 rows from Google Sheet ---
try:
    client = get_gspread_client_from_secrets()
    sheet = client.open_by_key(GOOGLE_SHEET_ID).worksheet(GOOGLE_SHEET_NAME)
    data = sheet.get_all_values()
    if data and len(data) > 1:
        headers, rows = data[0], data[-10:]
        df_hist = pd.DataFrame(rows, columns=headers)
        st.subheader("🕒 History: Last 10 Signals Sent")
        st.dataframe(df_hist, use_container_width=True)
except Exception as e:
    st.warning(f"Could not load persistent sheet data: {e}")

# --- Reference Table: All MACD cross up (no filter) in last 10 bars ---
if ref_crosses:
    st.subheader("🔍 Reference: Recent MACD Cross Up Zero Events (last 10 bars, no filter)")
    df_ref = pd.DataFrame(ref_crosses)
    st.dataframe(df_ref, use_container_width=True)
else:
    st.info("No MACD cross up events found in last 10 bars.")

st.caption(f"Checked {tickers_checked} tickers; {bars_fetched} bars fetched. Screener logic and reference tables above for research.")
