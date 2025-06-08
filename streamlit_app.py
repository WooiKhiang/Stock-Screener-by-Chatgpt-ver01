import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, date
import requests
import os
import csv
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# === CONFIGURATION ===
TELEGRAM_BOT_TOKEN = "7280991990:AAEk5x4XFCW_sTohAQGUujy1ECAQHjSY_OU"
TELEGRAM_CHAT_ID = "713264762"
GOOGLE_SHEET_ID = "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4"
GOOGLE_SHEET_NAME = "Sheet1"
ALERT_LOG = "alerts_log.csv"

st.set_page_config(page_title="US Market Day Trading Screener", layout="wide")
st.title("US Market Go/No-Go Dashboard")

# === SIDEBAR SETTINGS ===
with st.sidebar.expander("🔎 Screener Criteria", expanded=True):
    min_price = st.number_input("Min Price ($)", value=5.0)
    max_price = st.number_input("Max Price ($)", value=500.0)
    min_avg_vol = st.number_input("Min Average Volume", value=1_000_000)
    min_index_change = st.number_input("Min Index Change (%)", value=0.05)
    max_atr_percent = st.number_input("Max ATR (%)", value=0.015)
    volume_factor = st.number_input("Min Volume Factor", value=0.7)
    ema200_lookback = st.number_input("EMA200 Breakout Lookback Bars", value=6, min_value=1, max_value=48)
    pullback_lookback = st.number_input("VWAP/EMA Pullback Lookback Bars", value=6, min_value=2, max_value=20)

with st.sidebar.expander("💰 Profit & Risk Planner", expanded=True):
    capital = st.number_input("Capital per Trade ($)", value=1000.0, step=100.0)
    take_profit_pct = st.number_input("Take Profit (%)", value=2.0, min_value=0.5, max_value=10.0, step=0.1) / 100
    cut_loss_pct = st.number_input("Cut Loss (%)", value=1.0, min_value=0.2, max_value=5.0, step=0.1) / 100

with st.sidebar.expander("🕰️ Backtest (GO Day Open Breakout)", expanded=False):
    run_backtest = st.checkbox("Run GO Day Backtest Now")
    lookback_days = st.number_input("Backtest: Days", value=10, min_value=2, max_value=30, step=1)
    backtest_tp = st.number_input("Backtest: Take Profit %", value=2.0, min_value=0.5, max_value=10.0, step=0.1) / 100
    backtest_sl = st.number_input("Backtest: Stop Loss %", value=1.0, min_value=0.2, max_value=5.0, step=0.1) / 100

# === WATCHLIST ===
watchlist = [
    "AAPL","ABBV","ABT","ACN","ADBE","AIG","AMGN","AMT","AMZN","AVGO",
    "AXP","BA","BAC","BK","BKNG","BLK","BMY","BRK.B","C","CAT",
    "CHTR","CL","CMCSA","COF","COP","COST","CRM","CSCO","CVS","CVX",
    "DHR","DIS","DOW","DUK","EMR","EXC","F","FDX","FOX","FOXA",
    "GD","GE","GILD","GM","GOOG","GOOGL","GS","HD","HON","IBM",
    "INTC","JNJ","JPM","KHC","KMI","KO","LIN","LLY","LMT","LOW",
    "MA","MCD","MDLZ","MDT","MET","META","MMM","MO","MRK","MS",
    "MSFT","NEE","NFLX","NKE","NVDA","ORCL","PEP","PFE","PG","PM",
    "PYPL","QCOM","RTX","SBUX","SCHW","SO","SPG","T","TGT","TMO",
    "TMUS","TSLA","TXN","UNH","UNP","UPS","USB","V","VZ","WBA",
    "WFC","WMT","XOM"
]

# === UTILITY FUNCTIONS ===
def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        st.warning(f"Failed to send Telegram alert: {e}")

def log_to_google_sheet(row_dict, section_name):
    try:
        SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        if "gcp_service_account" in st.secrets:
            creds_dict = st.secrets["gcp_service_account"]
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, SCOPE)
        else:
            creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", SCOPE)
        gc = gspread.authorize(creds)
        worksheet = gc.open_by_key(GOOGLE_SHEET_ID).worksheet(GOOGLE_SHEET_NAME)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_row = [
            now, section_name,
            row_dict.get("Ticker"), row_dict.get("Price"),
            row_dict.get("Target Price"), row_dict.get("Cut Loss Price"),
            row_dict.get("Position Size"), row_dict.get("Max Loss at Stop"),
            row_dict.get("Support"), row_dict.get("Resistance"),
            row_dict.get("Reasons")
        ]
        worksheet.append_row(log_row, value_input_option="USER_ENTERED")
    except Exception as e:
        st.warning(f"Failed to log to Google Sheet: {e}")

def was_alerted_today(ticker, section_name):
    today_str = str(date.today())
    if os.path.exists(ALERT_LOG):
        with open(ALERT_LOG, "r") as f:
            for row in csv.reader(f):
                if len(row) == 3 and row[0] == today_str and row[1] == section_name and row[2] == ticker:
                    return True
    return False

def add_to_alert_log(ticker, section_name):
    today_str = str(date.today())
    with open(ALERT_LOG, "a", newline="") as f:
        csv.writer(f).writerow([today_str, section_name, ticker])

def calc_support_resistance(prices, lookback=48):
    sub = prices.iloc[-lookback:] if len(prices) >= lookback else prices
    if sub.empty or sub.isnull().values.any():
        return "", ""
    support = sub.min()
    resistance = sub.max()
    return f"{support:.2f}", f"{resistance:.2f}"

def get_data(ticker):
    end = datetime.now()
    start = end - timedelta(days=5)
    df = yf.download(ticker, start=start, end=end, interval='5m', progress=False)
    return df

def get_intraday_data(ticker):
    try:
        df = yf.download(ticker, period="2d", interval="5m", progress=False)
    except Exception:
        return None
    return df

# === STOCK SCAN LOGIC ===
def scan_stock_all(ticker, spy_change, min_price, max_price, min_avg_vol, ema200_lookback, pullback_lookback, capital, take_profit_pct, cut_loss_pct):
    df = get_intraday_data(ticker)
    if df is None or len(df) < 22:
        return None
    today = df[df.index.date == df.index[-1].date()]
    if today.empty or len(today) < 10:
        return None

    try:
        today_open = float(today["Open"].iloc[0])
        today_close = float(today["Close"].iloc[-1])
        today_high = float(today["High"].max())
        today_low = float(today["Low"].min())
        price = today_close
        today_change = (today_close - today_open) / today_open * 100
    except Exception:
        return None

    try:
        avg_vol = float(today["Volume"].rolling(10).mean().iloc[-1])
        volume = float(today["Volume"].iloc[-1])
    except Exception:
        avg_vol = volume = 0

    if not (min_price <= price <= max_price and avg_vol >= min_avg_vol):
        return None

    rs_score = today_change - spy_change
    support, resistance = calc_support_resistance(today["Close"], lookback=48)

    go_criteria = rs_score > 1
    nogo_criteria = not go_criteria
    accumulation_criteria = rs_score > 0

    # EMA200 / VWAP features, basic version
    ema200 = today["Close"].ewm(span=200).mean().iloc[-1]
    ema200_breakout = (today_close > ema200) and (today["Close"].iloc[-2] < ema200)
    vwap = (today["Volume"] * (today["High"] + today["Low"] + today["Close"]) / 3).cumsum() / today["Volume"].cumsum()
    vwap_pullback = today_close > vwap.iloc[-1] and (abs(today["Low"].iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1] < 0.01)

    target_price = price * (1 + take_profit_pct)
    cut_loss_price = price * (1 - cut_loss_pct)
    risk_per_share = price - cut_loss_price
    position_size = int(capital / price) if price > 0 else 0
    max_loss = position_size * risk_per_share

    # Simple AI Scoring for demo
    ai_score = (rs_score + (price - float(support)) / (float(resistance) - float(support) + 1e-9)) if support and resistance else rs_score
    confidence = min(99, max(60, 70 + (ai_score / 3)))
    ai_reason = "Momentum + above support" if ai_score > 2 else "Normal setup"

    return {
        "Ticker": ticker,
        "Price": f"{price:.2f}",
        "Change %": f"{today_change:.2f}",
        "Support": support,
        "Resistance": resistance,
        "GO": go_criteria,
        "NO-GO": nogo_criteria,
        "Accumulation": accumulation_criteria,
        "EMA200 Breakout": ema200_breakout,
        "VWAP Pullback": vwap_pullback,
        "Target Price": f"{target_price:.2f}",
        "Cut Loss Price": f"{cut_loss_price:.2f}",
        "Position Size": position_size,
        "Max Loss at Stop": f"{max_loss:.2f}",
        "AI Score": f"{ai_score:.2f}",
        "Confidence Level": f"{confidence:.0f}%",
        "AI Reasoning": ai_reason,
        "Reasons": f"RS: {rs_score:.2f}, EMA200: {ema200_breakout}, VWAP: {vwap_pullback}"
    }

# === MARKET CONDITIONS ===
spy = get_data('SPY')
qqq = get_data('QQQ')
spy_change = qqq_change = 0
if not spy.empty and not qqq.empty:
    try:
        spy_change = (float(spy['Close'].iloc[-1]) - float(spy['Close'].iloc[-2])) / float(spy['Close'].iloc[-2]) * 100
        qqq_change = (float(qqq['Close'].iloc[-1]) - float(qqq['Close'].iloc[-2])) / float(qqq['Close'].iloc[-2]) * 100
    except Exception:
        pass

    st.subheader("Today's Market Conditions")
    st.markdown(f"**SPY Change:** {spy_change:.2f}%")
    st.markdown(f"**QQQ Change:** {qqq_change:.2f}%")
    st.markdown("**Sentiment:** " + ("🟢 GO DAY! Risk-on" if spy_change >= min_index_change and qqq_change >= min_index_change else "🔴 NO-GO DAY! Risk-off"))
    st.caption("Data source: Yahoo Finance")
else:
    st.warning("⚠️ Could not load recent market data. Try again later.")

# === MAIN SCREENER ===
results = []
for ticker in watchlist:
    result = scan_stock_all(
        ticker,
        spy_change,
        min_price, max_price, min_avg_vol,
        ema200_lookback, pullback_lookback,
        capital, take_profit_pct, cut_loss_pct
    )
    if result:
        results.append(result)
df_results = pd.DataFrame(results) if results else pd.DataFrame()

# === PRESENTATION ===
trade_cols = ["Target Price", "Cut Loss Price", "Position Size", "Max Loss at Stop", "Support", "Resistance", "Reasons"]
def show_section(title, filter_column):
    st.subheader(title)
    if not df_results.empty and filter_column in df_results.columns:
        df = df_results[df_results[filter_column] == True]
        if not df.empty:
            st.dataframe(df[["Ticker", "Price", "Change %", "Support", "Resistance", "AI Score", "Confidence Level", "AI Reasoning"] + trade_cols])
        else:
            st.info(f"⚠️ No stocks meet {title.lower()} criteria. Try relaxing the filter settings.")
    else:
        st.info("No data for stock screening.")

show_section("Go Day Stock Recommendations (Momentum/Breakout)", "GO")
show_section("No-Go Day Stock Recommendations (Defensive/Resilient)", "NO-GO")
show_section("Potential Institutional Accumulation", "Accumulation")
show_section("EMA200 Breakout", "EMA200 Breakout")
show_section("VWAP Pullback Bounce", "VWAP Pullback")

# === AI TOP 5 PICKS ===
if not df_results.empty:
    st.subheader("🤖 AI Top 5 Picks of the Day")
    df_ai = df_results.sort_values(by="AI Score", ascending=False).head(5)
    st.dataframe(df_ai[["Ticker", "Price", "AI Score", "Confidence Level", "AI Reasoning"] + trade_cols])
    for i, row in df_ai.iterrows():
        st.markdown(f"**{row['Ticker']}** | Reason: {row['AI Reasoning']} | Confidence: {row['Confidence Level']}")
else:
    st.info("No AI picks for today.")

# === ALERTS & LOGGING ===
if 'alerted_tickers' not in st.session_state:
    st.session_state['alerted_tickers'] = set()
alerted_tickers = st.session_state['alerted_tickers']

for _, row in df_results.iterrows():
    for section_name, filter_col in [
        ("GO Day", "GO"),
        ("No-Go Day", "NO-GO"),
        ("Institutional Accumulation", "Accumulation"),
        ("EMA200 Breakout", "EMA200 Breakout"),
        ("VWAP Pullback", "VWAP Pullback")
    ]:
        if filter_col in row and row[filter_col]:
            ticker = row["Ticker"]
            if not was_alerted_today(ticker, section_name):
                msg = (
                    f"📈 {section_name} ALERT!\n"
                    f"Ticker: {ticker}\n"
                    f"Price: ${row['Price']} | Target: ${row['Target Price']} | Cut Loss: ${row['Cut Loss Price']}\n"
                    f"Support: {row['Support']} | Resistance: {row['Resistance']}\n"
                    f"Position: {row['Position Size']} shares | Max Loss: ${row['Max Loss at Stop']}\n"
                    f"Confidence: {row['Confidence Level']}\n"
                    f"Reasons: {row['Reasons']}"
                )
                send_telegram_alert(msg)
                add_to_alert_log(ticker, section_name)
                log_to_google_sheet(row, section_name)
                alerted_tickers.add((ticker, section_name))
st.session_state['alerted_tickers'] = alerted_tickers

# === BACKTEST MODULE ===
if run_backtest:
    st.subheader(f"GO Day Backtest Results (last {lookback_days} days)")
    bt_results = []
    for ticker in watchlist:
        df = yf.download(ticker, period=f"{lookback_days+2}d", interval='5m', progress=False)
        if df.empty: continue
        df['Date'] = df.index.date
        day_groups = df.groupby('Date')
        for d, group in day_groups:
            if len(group) < 20: continue
            o = group['Open'].iloc[0]
            h = group['High'].max()
            l = group['Low'].min()
            c = group['Close'].iloc[-1]
            tp = o * (1 + backtest_tp)
            sl = o * (1 - backtest_sl)
            hit_tp = (group['High'] >= tp).any()
            hit_sl = (group['Low'] <= sl).any()
            win = None
            if hit_tp and hit_sl:
                tp_idx = group[group['High'] >= tp].index[0]
                sl_idx = group[group['Low'] <= sl].index[0]
                win = tp_idx < sl_idx
            elif hit_tp:
                win = True
            elif hit_sl:
                win = False
            else:
                win = c > o
            bt_results.append({'Ticker': ticker, 'Date': d, 'Open': o, 'Close': c, 'TP_hit': hit_tp, 'SL_hit': hit_sl, 'Win': win})
    if bt_results:
        df_bt = pd.DataFrame(bt_results)
        win_rate = df_bt['Win'].mean() * 100
        st.markdown(f"**Backtest Win Rate:** {win_rate:.2f}% (sample size: {len(df_bt)})")
        st.dataframe(df_bt.tail(100))
    else:
        st.info("No backtest results found. Try a smaller watchlist or different settings.")
