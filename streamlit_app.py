import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import pytz
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import time

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

def format_number(num, decimals=2):
    try:
        if num is None or num == "" or np.isnan(num): return "-"
        if isinstance(num, int) or decimals == 0:
            return f"{int(num):,}"
        return f"{num:,.{decimals}f}"
    except Exception:
        return str(num)

def get_live_price(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get('regularMarketPrice', None)
    except Exception:
        return None

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
    df['ATR'] = df['Close'].rolling(14).apply(lambda x: np.mean(np.abs(np.diff(x))), raw=True)
    return df

def mean_reversion_signal(df):
    c = float(safe_scalar(df['Close'].iloc[-1]))
    sma = float(safe_scalar(df['SMA40'].iloc[-1]))
    rsi = float(safe_scalar(df['RSI3'].iloc[-1]))
    # DEBUG: All filters removed except basic signal
    cond = (c > sma) and (rsi < 15)
    score = 75 + max(0, 15 - rsi) if cond else 0
    return bool(cond), "DEBUG: Mean Reversion (no filter)", float(f"{score:.2f}")

def ema40_breakout_signal(df):
    c = float(safe_scalar(df['Close'].iloc[-1]))
    ema = float(safe_scalar(df['EMA40'].iloc[-1]))
    pc = float(safe_scalar(df['Close'].iloc[-2]))
    pema = float(safe_scalar(df['EMA40'].iloc[-2]))
    dipped = np.any(df['Close'].iloc[-10:-1] < df['EMA40'].iloc[-10:-1])
    # DEBUG: All filters removed except basic signal
    cond = (c > ema) and ((pc < pema) or dipped)
    score = 70 + min(20, c - ema) if cond else 0
    return bool(cond), "DEBUG: EMA40 Breakout (no filter)", float(f"{score:.2f}")

def macd_ema_signal(df):
    macd = float(safe_scalar(df['MACD'].iloc[-1]))
    macd_signal = float(safe_scalar(df['MACD_signal'].iloc[-1]))
    macd_prev = float(safe_scalar(df['MACD'].iloc[-2]))
    macd_signal_prev = float(safe_scalar(df['MACD_signal'].iloc[-2]))
    ema8 = float(safe_scalar(df['EMA8'].iloc[-1]))
    ema21 = float(safe_scalar(df['EMA21'].iloc[-1]))
    # DEBUG: All filters removed except basic signal
    cross = (macd_prev < macd_signal_prev) and (macd > macd_signal) and (macd < 0)
    cond = cross and (ema8 > ema21)
    score = 65 + int(abs(macd)*5) if cond else 0
    return bool(cond), "DEBUG: MACD+EMA (no filter)", float(f"{score:.2f}")

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

def get_market_sentiment():
    spy = yf.download('SPY', period='1d', interval='5m', progress=False)
    qqq = yf.download('QQQ', period='1d', interval='5m', progress=False)
    try:
        if spy.empty or qqq.empty:
            return "Market sentiment unavailable (data missing)"
        spy_open = spy['Open'].iloc[0]
        spy_last = spy['Close'].iloc[-1]
        qqq_open = qqq['Open'].iloc[0]
        qqq_last = qqq['Close'].iloc[-1]
        spy_pct = float(spy_last - spy_open) / float(spy_open) * 100
        qqq_pct = float(qqq_last - qqq_open) / float(qqq_open) * 100
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

# --- Streamlit Sidebar ---
st.sidebar.header("Trade Settings")
min_price = st.sidebar.number_input("Min Price ($)", value=10.0)
max_price = st.sidebar.number_input("Max Price ($)", value=2000.0)
min_volume = st.sidebar.number_input("Min Avg Vol (40 bars)", value=100000)
capital_per_trade = st.sidebar.number_input("Capital per Trade ($)", value=1000.0, step=100.0)
show_live_price = st.sidebar.checkbox("Show 'Live' (Ticker.info) price", value=True)

st.title("🔍 AI-Powered US Stocks Screener – Intraday, Swing, and Catalyst Hybrid")
st.caption(f"Last run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption(f"Market Status: {get_market_status()}")
st.subheader(f"Market Sentiment: {get_market_sentiment()}")

# ------------------ Top 10 Most Active Stocks (Today) ------------------
st.subheader("Top 10 Most Active S&P 100 Stocks Today (5m data)")

active_rows = []
for ticker in sp100:
    try:
        df = yf.download(ticker, period="1d", interval="5m", progress=False)
        if df.empty or len(df) < 2:
            continue
        vol = int(df['Volume'].sum())
        open_price = float(df['Open'].iloc[0])
        last_price = float(df['Close'].iloc[-1])
        change_pct = 100 * (last_price - open_price) / open_price
        # RSI(14)
        delta = df['Close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        roll_up = up.rolling(14).mean()
        roll_down = down.rolling(14).mean()
        rs = roll_up / (roll_down + 1e-9)
        rsi14 = 100 - (100 / (1 + rs))
        rsi = float(rsi14.iloc[-1])
        # MACD
        ema12 = df['Close'].ewm(span=12, min_periods=12).mean()
        ema26 = df['Close'].ewm(span=26, min_periods=26).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, min_periods=9).mean()
        macd_val = float(macd.iloc[-1])
        macd_sig = float(macd_signal.iloc[-1])
        active_rows.append({
            "Ticker": ticker,
            "Volume": vol,
            "Open": format_number(open_price, 2),
            "Last Price": format_number(last_price, 2),
            "Change (%)": format_number(change_pct, 2),
            "RSI(14)": format_number(rsi, 2),
            "MACD": format_number(macd_val, 2),
            "MACD Sig": format_number(macd_sig, 2)
        })
    except Exception:
        continue

if active_rows:
    df_active = pd.DataFrame(active_rows).sort_values("Volume", ascending=False).head(10)
    st.dataframe(df_active.reset_index(drop=True), use_container_width=True)
else:
    st.info("No active stocks found today.")

# --------- Relative Strength Leaders (Top 10) ---------
def get_relative_strength_leaders():
    base = yf.download('SPY', period='6d', interval='1d', progress=False)['Close']
    leaders = []
    for ticker in sp100:
        try:
            prices = yf.download(ticker, period='6d', interval='1d', progress=False)['Close']
            if len(prices) < 5 or len(base) < 5:
                continue
            # Align indices and ensure both are Series of length 5
            prices = prices[-5:]
            base_aligned = base[-5:]
            rel = float(prices.pct_change().sum() - base_aligned.pct_change().sum())
            leaders.append((ticker, rel))
            time.sleep(0.1)  # avoid rate limit
        except Exception as e:
            continue
    # Keep only good floats (not NaN)
    leaders = [x for x in leaders if isinstance(x[1], float) and not np.isnan(x[1])]
    leaders.sort(key=lambda x: -x[1])
    return [x[0] for x in leaders[:10]]

rel_leaders = get_relative_strength_leaders()
st.markdown(f"**Relative Strength Top 10 Leaders (5d):** {'  '.join(rel_leaders)}")

# --------- New Catalyst/Gap+Volume Strategy ---------
def catalyst_gap_signal(df):
    # Identify if today opened >2% above yesterday's close and volume is >2x avg
    if len(df) < 2: return False, None, 0
    prev = df.iloc[-2]
    today = df.iloc[-1]
    gap = (today['Open'] - prev['Close']) / prev['Close']
    vol = today['Volume']
    avgvol = df['Volume'].iloc[-20:].mean()
    cond = (gap > 0.02) and (vol > 2 * avgvol)
    score = 85 if cond else 0
    return bool(cond), "Gap+Volume Breakout (>2% gap up, >2x vol)", float(score)

# --------- MAIN SCREENERS ---------
results_intraday = []
results_rel = []
results_catalyst = []
results_swing = []

for ticker in sp100:
    try:
        # --------- Intraday (5m) ---------
        df = yf.download(ticker, period="5d", interval="5m", progress=False)
        if df.empty or len(df) < 50:
            continue
        df['Close'] = df['Close']
        df['Volume'] = df['Volume']
        df = calc_indicators(df)
        last = df.iloc[-1]
        close_price = float(safe_scalar(last['Close']))
        avgvol40 = float(safe_scalar(last['AvgVol40']))
        volume_now = int(safe_scalar(last['Volume']))
        live_price = get_live_price(ticker) if show_live_price else None

        # Main filter
        if not (min_price <= close_price <= max_price): continue
        if avgvol40 < min_volume: continue

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
            results_intraday.append({
                "Ticker": ticker,
                "Strategy": strat_name,
                "AI Score": float(f"{score:.2f}"),
                "Entry Price (5m close)": entry,
                "Live Price (info)": live_price if live_price else "-",
                "Capital Used": invested,
                "Shares": shares,
                "Reason": reason,
                "Volume": volume_now,
                "Avg Vol (40)": int(avgvol40),
                "RSI(3)": float(safe_scalar(last['RSI3'])),
                "EMA40": float(safe_scalar(last['EMA40'])),
                "SMA40": float(safe_scalar(last['SMA40']))
            })

        # --------- Relative Strength + Signal (top 10 only) ---------
        if ticker in rel_leaders:
            for func, strat_name in [
                (mean_reversion_signal, "Mean Reversion"),
                (ema40_breakout_signal, "EMA40 Breakout"),
                (macd_ema_signal, "MACD+EMA")
            ]:
                sig, reason, score = func(df)
                if sig:
                    entry = close_price
                    shares = int(capital_per_trade // entry)
                    invested = shares * entry
                    results_rel.append({
                        "Ticker": ticker,
                        "Strategy": strat_name,
                        "AI Score": float(f"{score:.2f}"),
                        "Entry Price (5m close)": entry,
                        "Live Price (info)": live_price if live_price else "-",
                        "Capital Used": invested,
                        "Shares": shares,
                        "Reason": f"RelStrength: {reason}",
                        "Volume": volume_now,
                        "Avg Vol (40)": int(avgvol40),
                        "RSI(3)": float(safe_scalar(last['RSI3'])),
                        "EMA40": float(safe_scalar(last['EMA40'])),
                        "SMA40": float(safe_scalar(last['SMA40']))
                    })

        # --------- Catalyst/Gap+Volume (daily) ---------
        dfd = yf.download(ticker, period='3d', interval='1d', progress=False)
        if not dfd.empty and len(dfd) >= 2:
            sig, reason, score = catalyst_gap_signal(dfd)
            if sig:
                today = dfd.iloc[-1]
                entry = float(today['Open'])
                shares = int(capital_per_trade // entry)
                invested = shares * entry
                results_catalyst.append({
                    "Ticker": ticker,
                    "Strategy": "Gap+Volume",
                    "AI Score": float(f"{score:.2f}"),
                    "Entry Price (gap)": entry,
                    "Live Price (info)": live_price if live_price else "-",
                    "Capital Used": invested,
                    "Shares": shares,
                    "Reason": reason,
                    "Volume": int(today['Volume']),
                    "Avg Vol (20d)": int(dfd['Volume'][-20:].mean()),
                    "Gap %": round(100*(today['Open']/dfd.iloc[-2]['Close']-1),2)
                })

        # --------- Swing (EMA200 breakout, daily) ---------
        dfd = yf.download(ticker, period="1y", interval="1d", progress=False)
        if dfd.empty or len(dfd) < 210: continue
        dfd['EMA200'] = dfd['Close'].ewm(span=200, min_periods=200).mean()
        dfd['VolumeAvg20'] = dfd['Volume'].rolling(20).mean()
        today = dfd.iloc[-1]
        prev = dfd.iloc[-2]
        price_crossed = prev['Close'] < prev['EMA200'] and today['Close'] > today['EMA200']
        volume_confirm = today['Volume'] > 1.2 * today['VolumeAvg20']
        if price_crossed and volume_confirm and today['Volume'] > 500000:
            entry = today['Close']
            ema200 = today['EMA200']
            target = round(entry * 1.04, 2)   # +4% target
            stop = round(entry * 0.98, 2)     # -2% stop
            results_swing.append({
                "Date": today.name.strftime("%Y-%m-%d"),
                "Ticker": ticker,
                "Entry Price": entry,
                "Live Price (info)": live_price if live_price else "-",
                "EMA200": ema200,
                "Target Price": target,
                "Stop Loss": stop,
                "Volume": int(today['Volume']),
                "VolumeAvg20": int(today['VolumeAvg20']),
                "Reason": "EMA200 Breakout + Volume"
            })

    except Exception as e:
        continue

# --------------- DASHBOARD OUTPUT ---------------

def show_top_section(results, title, entry_col):
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(["AI Score"], ascending=False).head(3).reset_index(drop=True)
        df.insert(0, "Rank", df.index+1)
        for col in ["Entry Price (5m close)", "Live Price (info)", "EMA40", "SMA40", "RSI(3)", "AI Score"]:
            if col in df:
                df[col] = df[col].apply(lambda x: format_number(x,2))
        for col in ["Volume", "Avg Vol (40)", "Shares", "VolumeAvg20"]:
            if col in df:
                df[col] = df[col].apply(lambda x: format_number(x,0))
        st.subheader(title)
        st.dataframe(df, use_container_width=True)

        # --- Send top 3 to Telegram & Google Sheet ---
        eastern = pytz.timezone("US/Eastern")
        now_et = datetime.now(pytz.utc).astimezone(eastern)
        now_time = now_et.strftime('%Y-%m-%d %H:%M:%S %Z')
        telegram_msgs, gsheet_rows = [], []
        for idx, row in df.iterrows():
            entry = float(str(row[entry_col]).replace(",",""))
            target_price = round(entry * 1.02, 2)
            cut_loss_price = round(entry * 0.99, 2)
            msg = (
                f"#{title}\nRank #{row['Rank']}: {row['Ticker']}\n"
                f"Entry: ${row[entry_col]} | Live: {row['Live Price (info)']}\n"
                f"Target: ${format_number(target_price,2)} | Stop: ${format_number(cut_loss_price,2)}\n"
                f"Reason: {row['Reason']}\n"
                f"Time: {now_time}"
            )
            telegram_msgs.append(msg)
            gsheet_rows.append([
                now_time, row['Ticker'], format_number(entry,2), format_number(target_price,2), format_number(cut_loss_price,2),
                row.get('Strategy', ''), row.get('AI Score', ''), row.get('Reason', '')
            ])
        for msg in telegram_msgs:
            send_telegram_alert(msg)
        append_to_gsheet(gsheet_rows)
    else:
        st.info(f"No stocks meet {title} criteria right now.")

show_top_section(results_intraday, "Intraday AI Strategies (5m, Filtered)", "Entry Price (5m close)")
show_top_section(results_rel, "Relative Strength Leaders: Top Signals", "Entry Price (5m close)")
show_top_section(results_catalyst, "Catalyst/Gaps: Top Gap+Volume Picks", "Entry Price (gap)")

# --------- SWING SECTION (Daily EMA200) ---------
if results_swing:
    df_swing = pd.DataFrame(results_swing).sort_values("Volume", ascending=False).head(3)
    for col in ["Entry Price", "Live Price (info)", "EMA200", "Target Price", "Stop Loss"]:
        df_swing[col] = df_swing[col].apply(lambda x: format_number(x, 2))
    for col in ["Volume", "VolumeAvg20"]:
        df_swing[col] = df_swing[col].apply(lambda x: format_number(x, 0))
    st.subheader("Swing Picks (EMA200 Breakout, Daily)")
    st.dataframe(df_swing.reset_index(drop=True), use_container_width=True)
    # log to Telegram & Google Sheets
    eastern = pytz.timezone("US/Eastern")
    now_et = datetime.now(pytz.utc).astimezone(eastern)
    now_time = now_et.strftime('%Y-%m-%d %H:%M:%S %Z')
    swing_msgs, swing_rows = [], []
    for idx, row in df_swing.iterrows():
        entry = float(str(row['Entry Price']).replace(",", ""))
        target_price = float(str(row['Target Price']).replace(",", ""))
        stop_price = float(str(row['Stop Loss']).replace(",", ""))
        msg = (
            f"#SwingPicks\n{row['Date']} {row['Ticker']} Entry: ${row['Entry Price']} | Live: {row['Live Price (info)']}\n"
            f"Target: ${row['Target Price']} | Stop: ${row['Stop Loss']}\n"
            f"Reason: {row['Reason']}\n"
            f"Time: {now_time}"
        )
        swing_msgs.append(msg)
        swing_rows.append([
            now_time, row['Ticker'], row['Entry Price'], row['Target Price'], row['Stop Loss'],
            "EMA200 Breakout + Volume", "", row['Reason']
        ])
    for msg in swing_msgs:
        send_telegram_alert(msg)
    append_to_gsheet(swing_rows)
else:
    st.info("No swing trade setups found today.")
