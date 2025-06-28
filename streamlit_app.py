import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import pytz
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
from datetime import datetime

# --- Config ---
TELEGRAM_BOT_TOKEN = "7280991990:AAEk5x4XFCW_sTohAQGUujy1ECAQHjSY_OU"
TELEGRAM_CHAT_ID = "713264762"
GOOGLE_SHEET_ID = "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4"
GOOGLE_SHEET_NAME = "Sheet1"

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

# --- Sidebar for settings ---
st.sidebar.header("Trade Settings")
min_price = st.sidebar.number_input("Min Price ($)", value=10.0)
max_price = st.sidebar.number_input("Max Price ($)", value=2000.0)
min_vol = st.sidebar.number_input("Min Avg Vol (40)", value=100000)
capital = st.sidebar.number_input("Capital per Trade ($)", value=1000.0, step=100.0)
show_debug = st.sidebar.checkbox("Show Debug Info", value=False)

def norm(df):
    # Flatten MultiIndex and force all columns to lower case, no spaces
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [('_'.join([str(x) for x in col if x not in [None,'','nan']])).lower().replace(" ","_") for col in df.columns]
    else:
        df.columns = [str(c).lower().replace(" ","_") for c in df.columns]
    return df

def ensure_core_cols(df):
    col_map = {}
    for c in df.columns:
        if c == "close" or "close" in c: col_map[c] = "close"
        if c == "open" or "open" in c: col_map[c] = "open"
        if c == "high" or "high" in c: col_map[c] = "high"
        if c == "low" or "low" in c: col_map[c] = "low"
        if c == "volume" or "vol" in c: col_map[c] = "volume"
    df = df.rename(columns=col_map)
    need = ["close", "open", "high", "low", "volume"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise Exception(f"Missing columns: {missing}")
    return df

def formatn(x, d=2):
    try: return f"{x:,.{d}f}"
    except: return x

def calc_indicators(df):
    df['ema200'] = df['close'].ewm(span=200, min_periods=200).mean()
    df['ema40'] = df['close'].ewm(span=40, min_periods=40).mean()
    high_low = df['high'] - df['low']
    high_prevclose = np.abs(df['high'] - df['close'].shift(1))
    low_prevclose = np.abs(df['low'] - df['close'].shift(1))
    ranges = pd.concat([high_low, high_prevclose, low_prevclose], axis=1)
    df['atr14'] = ranges.max(axis=1).rolling(14).mean()
    # RSI
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-8)
    df['rsi14'] = 100 - (100 / (1 + rs))
    return df

def ema200_breakout_daily(df):
    if len(df) < 210 or 'ema200' not in df.columns: return False, "", 0
    prev, curr = df.iloc[-2], df.iloc[-1]
    breakout = (prev['close'] < prev['ema200']) and (curr['close'] > curr['ema200'])
    vol_avg = df['volume'][-20:-1].mean() + 1e-8
    vol_mult = curr['volume'] / vol_avg
    rsi_ok = curr['rsi14'] < 70
    pct_above_ema = (curr['close'] - curr['ema200']) / curr['ema200'] * 100
    # Confidence score = % above EMA200 + volume multiplier, rescaled
    score = int(60 + pct_above_ema*10 + (vol_mult-1)*15)
    score = max(60, min(score, 100))
    cond = breakout and (vol_mult > 1.5) and rsi_ok
    return cond, f"Daily EMA200 Breakout (+{pct_above_ema:.2f}%, Vol x{vol_mult:.2f})", score

def hybrid_signal(df5, df1h):
    if len(df5) < 50 or len(df1h) < 50: return False, "", 0
    c, ema40 = df5['close'].iloc[-1], df5['ema40'].iloc[-1]
    prev_c, prev_ema = df5['close'].iloc[-2], df5['ema40'].iloc[-2]
    breakout = (prev_c < prev_ema) and (c > ema40)
    vol_avg = df5['volume'][-40:-1].mean() + 1e-8
    vol_spike = df5['volume'].iloc[-1] / vol_avg
    curr_1h = df1h.iloc[-1]
    above_ema200 = curr_1h['close'] > curr_1h['ema200']
    pct_breakout = (c - ema40) / ema40 * 100
    pct_above_ema200 = (curr_1h['close'] - curr_1h['ema200']) / curr_1h['ema200'] * 100
    rsi_bonus = 5 if 55 <= curr_1h['rsi14'] <= 65 else 0
    # Score = breakout % + volume spike + 1hr EMA200 + RSI bonus
    score = int(55 + pct_breakout*10 + (vol_spike-1)*20 + pct_above_ema200*5 + rsi_bonus)
    score = max(55, min(score, 99))
    cond = breakout and (vol_spike > 1.2) and above_ema200
    return cond, f"5m EMA40 Breakout + 1h Confirm (Breakout: {pct_breakout:.2f}%, Vol x{vol_spike:.2f}, Above EMA200: {pct_above_ema200:.2f}%)", score

# --- Google Sheets & Telegram ---
def get_gspread_client_from_secrets():
    info = st.secrets["gcp_service_account"]
    creds_dict = {k: v for k, v in info.items()}
    if isinstance(creds_dict["private_key"], list):
        creds_dict["private_key"] = "\n".join(creds_dict["private_key"])
    creds_json = json.dumps(creds_dict)
    scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(creds_json), scope)
    client = gspread.authorize(creds)
    return client

def append_to_gsheet(data_rows):
    try:
        client = get_gspread_client_from_secrets()
        sheet = client.open_by_key(GOOGLE_SHEET_ID).worksheet(GOOGLE_SHEET_NAME)
        for row in data_rows:
            sheet.append_row(row, value_input_option="USER_ENTERED")
    except Exception as e:
        st.warning(f"Failed to append to Google Sheet: {e}")

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        resp = requests.post(url, data=payload)
        if not resp.ok:
            st.warning(f"Telegram alert failed: {resp.text}")
    except Exception as e:
        st.warning(f"Telegram alert failed: {e}")

results = []
debug_issues = []

for ticker in sp100:
    try:
        # --- Daily chart for EMA200 breakout
        dfd = yf.download(ticker, period='1y', interval='1d', progress=False, threads=False)
        if dfd.empty or len(dfd) < 210:
            debug_issues.append({"Ticker": ticker, "Issue": "Daily data empty or short"})
            continue
        dfd = norm(dfd)
        try:
            dfd = ensure_core_cols(dfd)
        except Exception as e:
            debug_issues.append({"Ticker": ticker, "Issue": str(e)})
            continue
        dfd = calc_indicators(dfd)
        hit, reason, score = ema200_breakout_daily(dfd)
        if hit:
            curr = dfd.iloc[-1]
            price = curr['close']
            atr = curr['atr14']
            if not (min_price <= price <= max_price): continue
            if curr['volume'] < min_vol: continue
            shares = int(capital // price)
            target_price = round(price + 2*atr, 2)
            cut_loss_price = round(price - 1.5*atr, 2)
            local_time = datetime.now(pytz.timezone("Asia/Singapore")).strftime('%Y-%m-%d %H:%M:%S')
            results.append({
                "Ticker": ticker,
                "Strategy": "EMA200 Breakout (Daily)",
                "Score": score,
                "Entry": formatn(price),
                "Target Price": formatn(target_price),
                "Cut Loss Price": formatn(cut_loss_price),
                "Shares": shares,
                "ATR": formatn(atr,2),
                "Reason": reason,
                "Type": "Swing",
                "Time Picked": local_time
            })
            continue  # if breakout fires, don't double-count for hybrid

        # --- Hybrid: 5-min entry + 1hr confirm
        df5 = yf.download(ticker, period='3d', interval='5m', progress=False, threads=False)
        df1h = yf.download(ticker, period='30d', interval='60m', progress=False, threads=False)
        if df5.empty or df1h.empty:
            debug_issues.append({"Ticker": ticker, "Issue": "5m/1h data empty"})
            continue
        df5, df1h = norm(df5), norm(df1h)
        try:
            df5, df1h = ensure_core_cols(df5), ensure_core_cols(df1h)
        except Exception as e:
            debug_issues.append({"Ticker": ticker, "Issue": str(e)})
            continue
        df5 = calc_indicators(df5)
        df1h = calc_indicators(df1h)
        hit, reason, score = hybrid_signal(df5, df1h)
        if hit:
            price = df5['close'].iloc[-1]
            atr = df5['atr14'].iloc[-1]
            if not (min_price <= price <= max_price): continue
            if df5['volume'].iloc[-1] < min_vol: continue
            shares = int(capital // price)
            target_price = round(price + 2*atr, 2)
            cut_loss_price = round(price - 1.5*atr, 2)
            local_time = datetime.now(pytz.timezone("Asia/Singapore")).strftime('%Y-%m-%d %H:%M:%S')
            results.append({
                "Ticker": ticker,
                "Strategy": "Hybrid 5min+1hr Confirm",
                "Score": score,
                "Entry": formatn(price),
                "Target Price": formatn(target_price),
                "Cut Loss Price": formatn(cut_loss_price),
                "Shares": shares,
                "ATR": formatn(atr,2),
                "Reason": reason,
                "Type": "Hybrid",
                "Time Picked": local_time
            })
    except Exception as e:
        debug_issues.append({"Ticker": ticker, "Issue": str(e)})
        continue

# ---- OUTPUT TABLE & ALERTS ----
if results:
    df_out = pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)
    st.subheader("⭐ AI-Picked Stock Setups (Top 5)")
    st.dataframe(df_out.head(5), use_container_width=True)
    
    # ---- TELEGRAM/GOOGLE SHEETS: Top 3 only ----
    now_local = datetime.now(pytz.timezone("Asia/Singapore")).strftime('%Y-%m-%d %H:%M:%S')
    telegram_msgs, gsheet_rows = [], []
    for idx, row in df_out.head(3).iterrows():
        msg = (
            f"#AI_StockPick Rank #{idx+1}\n"
            f"{row['Ticker']} | Strategy: {row['Strategy']}\n"
            f"Entry: ${row['Entry']} | Target: ${row['Target Price']} | Cut Loss: ${row['Cut Loss Price']}\n"
            f"Shares: {row['Shares']} | ATR(14): {row['ATR']}\n"
            f"Reason: {row['Reason']}\n"
            f"Confidence Score: {row['Score']}\n"
            f"Time Picked: {row['Time Picked']}"
        )
        telegram_msgs.append(msg)
        gsheet_rows.append([
            row['Time Picked'], row['Ticker'], row['Entry'], row['Target Price'], row['Cut Loss Price'],
            row['Strategy'], row['Score'], row['Reason']
        ])
    for msg in telegram_msgs:
        send_telegram_alert(msg)
    append_to_gsheet(gsheet_rows)

else:
    st.info("No stocks met high-quality hybrid or EMA200 breakout criteria right now.")

if show_debug and debug_issues:
    st.sidebar.subheader("Debug Info (tickers with data issues)")
    df_debug = pd.DataFrame(debug_issues)
    st.sidebar.dataframe(df_debug)

st.markdown("""
---
**Strategy Summary:**  
- **Hybrid:** Enter on 5min EMA40 breakout *only if* 1hr price > EMA200 (trend confirmed). Score is proportional to breakout/volume/confirmation strength.  
- **EMA200 Breakout:** Daily close above EMA200 + volume + RSI filter (the classic, rare but high conviction, score reflects strength).

**Targets/Cut Loss (dynamic ATR):**  
- Take Profit: Entry + 2×ATR(14)  
- Cut Loss: Entry - 1.5×ATR(14)

**All times shown: GMT+8 (MY/SG time)**
""")
