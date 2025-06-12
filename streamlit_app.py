import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import pytz
import gspread
from oauth2client.service_account import ServiceAccountCredentials

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

# --- Utility Functions ---
def safe_scalar(val):
    if isinstance(val, pd.Series) or isinstance(val, np.ndarray):
        if len(val) == 1:
            return float(val.item())
        elif len(val) > 0:
            return float(val[-1])
        else:
            return np.nan
    try:
        return float(val)
    except Exception:
        return np.nan

def format_number(num, decimals=2):
    try:
        if np.isnan(num): return "-"
        if isinstance(num, int) or decimals == 0:
            return f"{int(num):,}"
        return f"{num:,.{decimals}f}"
    except Exception:
        return str(num)

def calc_indicators(df):
    df['SMA40'] = df['Close'].rolling(window=40).mean()
    df['EMA40'] = df['Close'].ewm(span=40, min_periods=40).mean()
    df['EMA8'] = df['Close'].ewm(span=8, min_periods=8).mean()
    df['EMA21'] = df['Close'].ewm(span=21, min_periods=21).mean()
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(window=3).mean()
    roll_down = down.rolling(window=3).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['RSI3'] = 100 - (100 / (1 + rs))
    ema12 = df['Close'].ewm(span=12, min_periods=12).mean()
    ema26 = df['Close'].ewm(span=26, min_periods=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, min_periods=9).mean()
    df['AvgVol40'] = df['Volume'].replace(0, np.nan).rolling(window=40, min_periods=1).mean().fillna(0)
    return df

def mean_reversion_signal(df):
    c = float(safe_scalar(df['Close'].iloc[-1]))
    sma = float(safe_scalar(df['SMA40'].iloc[-1]))
    rsi = float(safe_scalar(df['RSI3'].iloc[-1]))
    if np.isnan(c) or np.isnan(sma) or np.isnan(rsi):
        return False, None, 0
    cond = (c > sma) and (rsi < 15)
    score = 75 + max(0, 15 - rsi) if cond else 0
    return bool(cond), "Mean Reversion: Price > SMA40 & RSI(3)<15", score

def ema40_breakout_signal(df):
    c = float(safe_scalar(df['Close'].iloc[-1]))
    ema = float(safe_scalar(df['EMA40'].iloc[-1]))
    pc = float(safe_scalar(df['Close'].iloc[-2]))
    pema = float(safe_scalar(df['EMA40'].iloc[-2]))
    if np.isnan(c) or np.isnan(ema) or np.isnan(pc) or np.isnan(pema):
        return False, None, 0
    left_vals = df['Close'].iloc[-10:-1].values
    right_vals = df['EMA40'].iloc[-10:-1].values
    dipped = False
    if len(left_vals) == len(right_vals) and len(left_vals) > 0:
        dipped = np.any(left_vals < right_vals)
    cond = (c > ema) and ((pc < pema) or dipped)
    score = 70 + min(20, c - ema) if cond else 0
    return bool(cond), "EMA40 Breakout: Price reclaimed EMA40 (with shakeout)", score

def macd_ema_signal(df):
    macd = float(safe_scalar(df['MACD'].iloc[-1]))
    macd_signal = float(safe_scalar(df['MACD_signal'].iloc[-1]))
    macd_prev = float(safe_scalar(df['MACD'].iloc[-2]))
    macd_signal_prev = float(safe_scalar(df['MACD_signal'].iloc[-2]))
    ema8 = float(safe_scalar(df['EMA8'].iloc[-1]))
    ema21 = float(safe_scalar(df['EMA21'].iloc[-1]))
    if any(np.isnan(x) for x in [macd, macd_signal, macd_prev, macd_signal_prev, ema8, ema21]):
        return False, None, 0
    cross = (macd_prev < macd_signal_prev) and (macd > macd_signal) and (macd < 0)
    cond = cross and (ema8 > ema21)
    score = 65 + int(abs(macd)*5) if cond else 0
    return bool(cond), "MACD+EMA: MACD cross up <0 & EMA8>EMA21", score

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        resp = requests.post(url, data=payload)
        if not resp.ok:
            st.warning(f"Telegram alert failed: {resp.text}")
    except Exception as e:
        st.warning(f"Telegram alert failed: {e}")

def append_to_gsheet(data_rows, creds_path):
    try:
        scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(GOOGLE_SHEET_ID).worksheet(GOOGLE_SHEET_NAME)
        for row in data_rows:
            sheet.append_row(row, value_input_option="USER_ENTERED")
    except Exception as e:
        st.warning(f"Failed to append to Google Sheet: {e}")

def get_market_sentiment():
    spy = yf.download('SPY', period='1d', interval='5m', progress=False)
    qqq = yf.download('QQQ', period='1d', interval='5m', progress=False)
    try:
        spy_open = spy['Open'].iloc[0]
        spy_last = spy['Close'].iloc[-1]
        qqq_open = qqq['Open'].iloc[0]
        qqq_last = qqq['Close'].iloc[-1]
        spy_pct = (spy_last - spy_open) / spy_open * 100
        qqq_pct = (qqq_last - qqq_open) / qqq_open * 100
        avg_pct = (spy_pct + qqq_pct) / 2
        if avg_pct > 0.5:
            sentiment = f"🟢 Bullish (+{avg_pct:.2f}%)"
        elif avg_pct < -0.5:
            sentiment = f"🔴 Bearish ({avg_pct:.2f}%)"
        else:
            sentiment = f"🟡 Sideways ({avg_pct:.2f}%)"
        return sentiment
    except Exception as e:
        return f"Market sentiment unavailable ({e})"

def get_market_status():
    now_utc = datetime.utcnow()
    eastern = pytz.timezone("US/Eastern")
    now_et = now_utc.replace(tzinfo=pytz.utc).astimezone(eastern)
    time_now = now_et.time()
    open_time = datetime.strptime("09:30", "%H:%M").time()
    close_time = datetime.strptime("16:00", "%H:%M").time()
    if time_now < open_time:
        return f"Pre-market ({now_et.strftime('%I:%M %p %Z')})"
    elif time_now > close_time:
        return f"After-market ({now_et.strftime('%I:%M %p %Z')})"
    else:
        return f"Market Open ({now_et.strftime('%I:%M %p %Z')})"

# --- Streamlit Sidebar ---
st.sidebar.header("Filter Settings")
min_price = st.sidebar.number_input("Min Price ($)", value=10.0, key="min_price")
max_price = st.sidebar.number_input("Max Price ($)", value=2000.0, key="max_price")
min_volume = st.sidebar.number_input("Min Avg Vol (40 bars)", value=100000, key="min_vol")
capital_per_trade = st.sidebar.number_input("Capital per Trade ($)", value=1000.0, step=100.0, key="capital_trade")
gcreds_path = st.sidebar.text_input("Google Credentials Path", value="gcreds.json", help="Path to your gcreds.json service account file for Google Sheets")

# --- Dashboard Header ---
st.title("🔍 S&P 100 Intraday Screener & AI Top 3 Stock Picks")
st.caption(f"Last run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with st.expander("About the 3 AI Strategies", expanded=True):
    st.markdown("""
**Mean Reversion:**  
Looks for short-term oversold opportunities when price is above SMA40 and RSI(3) is below 15.  
**EMA40 Breakout:**  
Identifies when price breaks above its 40-period EMA, especially if it recently dipped below and recovers.  
**MACD+EMA:**  
Finds bullish MACD cross-ups below zero, with EMA8 > EMA21 to confirm trend shift.
""")

# --- Show market sentiment & status ---
sentiment = get_market_sentiment()
status = get_market_status()
st.subheader(f"Market Sentiment: {sentiment}")
st.caption(f"Market Status: {status}")

results = []
top10_active = []

for ticker in sp100:
    try:
        df = yf.download(ticker, period="5d", interval="5m", progress=False)
        if df.empty or len(df) < 50:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join([str(x) for x in c]).lower() for c in df.columns]
        else:
            df.columns = [str(c).lower() for c in df.columns]
        close_col = None
        candidates = ['close', 'adjclose', 'adj close']
        for cand in candidates:
            for col in df.columns:
                if cand == col.strip().replace(' ', '').lower():
                    close_col = col
                    break
            if close_col: break
        if not close_col:
            for col in df.columns:
                if 'close' in col.replace(' ', '').lower():
                    close_col = col
                    break
        if not close_col:
            continue
        if isinstance(df[close_col], pd.DataFrame):
            df['Close'] = df[close_col].iloc[:,0]
        else:
            df['Close'] = df[close_col]
        vol_col = None
        candidates = ['volume', 'regularmarketvolume']
        for cand in candidates:
            for col in df.columns:
                if cand == col.strip().replace(' ', '').lower():
                    vol_col = col
                    break
            if vol_col: break
        if not vol_col:
            for col in df.columns:
                if 'vol' in col.replace(' ', '').lower():
                    vol_col = col
                    break
        if not vol_col:
            continue
        if isinstance(df[vol_col], pd.DataFrame):
            df['Volume'] = df[vol_col].iloc[:,0]
        else:
            df['Volume'] = df[vol_col]

        df = calc_indicators(df)
        last = df.iloc[-1]
        close_price = float(safe_scalar(last['Close']))
        avgvol40 = float(safe_scalar(last['AvgVol40']))
        volume_now = int(safe_scalar(df['Volume'][df['Volume'] > 0].iloc[-1])) if (df['Volume'] > 0).any() else 0
        traded_value = close_price * volume_now

        top10_active.append({
            "Ticker": ticker,
            "Last Price": close_price,
            "Last Volume": volume_now,
            "Traded Value": traded_value,
            "Avg Vol (40)": int(avgvol40)
        })

        if any(np.isnan([close_price, avgvol40])):
            continue
        if not (min_price <= close_price <= max_price):
            continue
        if avgvol40 < min_volume:
            continue

        triggers = {}
        picks = []
        for func, strat_name in [
            (mean_reversion_signal, "Mean Reversion"),
            (ema40_breakout_signal, "EMA40 Breakout"),
            (macd_ema_signal, "MACD+EMA")
        ]:
            sig, reason, score = func(df)
            triggers[strat_name] = sig
            if sig:
                picks.append((strat_name, reason, score))

        if picks:
            picks.sort(key=lambda x: -x[2])
            strat_name, reason, score = picks[0]
            entry = close_price
            shares = int(capital_per_trade // entry)
            invested = shares * entry
            results.append({
                "Ticker": ticker,
                "Strategy": strat_name,
                "AI Score": score,
                "Entry Price": entry,
                "Capital Used": invested,
                "Shares": shares,
                "Reason": reason,
                "Volume": volume_now,
                "Avg Vol (40)": int(avgvol40),
                "RSI(3)": float(safe_scalar(last['RSI3'])),
                "EMA40": float(safe_scalar(last['EMA40'])),
                "SMA40": float(safe_scalar(last['SMA40'])),
                "Mean Reversion": triggers.get("Mean Reversion", False),
                "EMA40 Breakout": triggers.get("EMA40 Breakout", False),
                "MACD+EMA": triggers.get("MACD+EMA", False)
            })
    except Exception as e:
        pass

# --- Top 10 Active Stocks (by traded value) ---
if top10_active:
    df_top10 = pd.DataFrame(top10_active)
    df_top10 = df_top10.sort_values("Traded Value", ascending=False).head(10)
    # Format numbers
    df_top10['Last Price'] = df_top10['Last Price'].apply(lambda x: format_number(x, 2))
    df_top10['Last Volume'] = df_top10['Last Volume'].apply(lambda x: format_number(x, 0))
    df_top10['Traded Value'] = df_top10['Traded Value'].apply(lambda x: format_number(x, 2))
    df_top10['Avg Vol (40)'] = df_top10['Avg Vol (40)'].apply(lambda x: format_number(x, 0))
    st.subheader("🔥 Top 10 Most Active Stocks (by 5-min Traded Value)")
    st.dataframe(df_top10.reset_index(drop=True))

# --- Main AI picks output ---
df_results = pd.DataFrame(results)
st.header("AI-Powered Top 3 Intraday Stock Picks (S&P 100)")

if not df_results.empty:
    df_results = df_results.sort_values(["AI Score", "Strategy", "RSI(3)"], ascending=[False, True, True]).reset_index(drop=True)
    df_results.insert(0, "Rank", df_results.index + 1)
    # Format numbers for display
    for col in ["Entry Price", "Capital Used", "RSI(3)", "EMA40", "SMA40"]:
        df_results[col] = df_results[col].apply(lambda x: format_number(x, 2))
    for col in ["Volume", "Avg Vol (40)", "Shares"]:
        df_results[col] = df_results[col].apply(lambda x: format_number(x, 0))
    st.dataframe(df_results[[
        "Rank","Ticker","Entry Price","Strategy","AI Score",
        "Mean Reversion","EMA40 Breakout","MACD+EMA","Reason","Capital Used","Shares","RSI(3)","Volume","Avg Vol (40)","EMA40","SMA40"
    ]], use_container_width=True)

    # --- Top 3 Recommendations (sorted by AI Score) ---
    top3 = df_results.head(3)
    st.subheader("📈 Today's AI Stock Recommendations")
    telegram_messages = []
    gsheet_rows = []
    for idx, row in top3.iterrows():
        entry = float(str(row['Entry Price']).replace(",",""))
        target_price = round(entry * 1.02, 2)
        cut_loss_price = round(entry * 0.99, 2)
        now_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        strategy_list = []
        if row["Mean Reversion"]: strategy_list.append("Mean Reversion")
        if row["EMA40 Breakout"]: strategy_list.append("EMA40 Breakout")
        if row["MACD+EMA"]: strategy_list.append("MACD+EMA")
        triggered_strat = ", ".join(strategy_list) if strategy_list else row["Strategy"]

        msg = (
            f"🔥AI Pick #{idx+1}: {row['Ticker']}\n"
            f"Entry Price: ${format_number(entry,2)}\n"
            f"Target Price (+2%): ${format_number(target_price,2)}\n"
            f"Cut Loss (-1%): ${format_number(cut_loss_price,2)}\n"
            f"Reason: {row['Reason']}\n"
            f"Strategy: {triggered_strat}\n"
            f"AI Score: {row['AI Score']}\n"
            f"Time: {now_time}"
        )
        telegram_messages.append(msg)
        gsheet_rows.append([
            now_time, row['Ticker'], format_number(entry,2), format_number(target_price,2), format_number(cut_loss_price,2),
            triggered_strat, row['AI Score'], row['Reason']
        ])

        # Show in dashboard
        rank_emoji = "🥇" if idx==0 else ("🥈" if idx==1 else "🥉")
        st.markdown(f"""
        {rank_emoji} **Rank #{idx+1}: {row['Ticker']}**  
        **Signal(s):** {triggered_strat}  
        **AI Confidence Score:** {row['AI Score']}  
        **Reason:** {row['Reason']}  
        **Entry Price:** ${format_number(entry,2)} | **Target:** ${format_number(target_price,2)} | **Cut Loss:** ${format_number(cut_loss_price,2)}  
        **RSI(3):** {row['RSI(3)']} | **EMA40:** {row['EMA40']} | **SMA40:** {row['SMA40']} | **Vol:** {row['Volume']}
        """)
        if idx == 0:
            st.info("Rank #1: Highest confidence and strongest signal based on all indicators and strategy score. Most potential for intraday growth today.")
        elif idx == 1:
            st.info("Rank #2: Good signal, but a bit less compelling than #1. Still has solid edge today.")
        elif idx == 2:
            st.info("Rank #3: Valid setup, but less optimal than #1 or #2 based on strategy and indicator strength.")

    # --- Send Telegram Alerts ---
    for msg in telegram_messages:
        send_telegram_alert(msg)
    # --- Log to Google Sheets ---
    append_to_gsheet(gsheet_rows, gcreds_path)
else:
    st.info("No stocks meet your filter/strategy criteria right now.")

st.caption("Each recommendation above is ranked by a composite confidence score. Top 10 active stocks are shown for general market activity reference.")
