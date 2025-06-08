import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, date
import requests
import os
import csv

# --- SETTINGS ---
TELEGRAM_BOT_TOKEN = "7280991990:AAEk5x4XFCW_sTohAQGUujy1ECAQHjSY_OU"
TELEGRAM_CHAT_ID = "713264762"
GOOGLE_SHEET_ID = "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4"
GOOGLE_SHEET_NAME = "Sheet1"
ALERT_LOG = "alerts_log.csv"

st.set_page_config(page_title="US Market Day Trading Screener", layout="wide")
st.title("US Market Go/No-Go Dashboard")

# ---- Screener settings (sidebar) ----
with st.sidebar.expander("Screener Criteria", expanded=True):
    min_price = st.number_input("Min Price ($)", value=5.0)
    max_price = st.number_input("Max Price ($)", value=500.0)
    min_avg_vol = st.number_input("Min Average Volume", value=1_000_000)
    min_index_change = st.number_input("Min Index Change (%)", value=0.05)
    max_atr_percent = st.number_input("Max ATR (%)", value=0.015)
    volume_factor = st.number_input("Min Volume Factor", value=0.7)
    ema200_lookback = st.number_input("EMA200 Breakout Lookback Bars", value=6, min_value=1, max_value=48)
    pullback_lookback = st.number_input("VWAP/EMA Pullback Lookback Bars", value=6, min_value=2, max_value=20)

with st.sidebar.expander("Profit & Risk Planner", expanded=True):
    capital = st.number_input("Capital per Trade ($)", value=1000.0, step=100.0)
    take_profit_pct = st.number_input("Take Profit (%)", value=2.0, min_value=0.5, max_value=10.0, step=0.1) / 100
    cut_loss_pct = st.number_input("Cut Loss (%)", value=1.0, min_value=0.2, max_value=5.0, step=0.1) / 100

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

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        st.warning(f"Failed to send Telegram alert: {e}")

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

def scan_stock_all(ticker, spy_change, min_price, max_price, min_avg_vol, ema200_lookback, pullback_lookback,
                   capital, take_profit_pct, cut_loss_pct):
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
        today_change = (today_close - today_open) / today_open * 100
        price = today_close
        avg_vol = float(today["Volume"].rolling(10).mean().iloc[-1])
        volume = float(today["Volume"].iloc[-1])
        rel_vol = volume / avg_vol if avg_vol else 0
    except Exception:
        return None

    if not (min_price <= price <= max_price and avg_vol >= min_avg_vol):
        return None

    # --- Indicators
    rs_score = today_change - spy_change

    vwap = (today["Volume"] * (today["High"] + today["Low"] + today["Close"]) / 3).cumsum() / today["Volume"].cumsum()
    price_above_vwap = today_close > float(vwap.iloc[-1])
    ema10 = float(today["Close"].ewm(span=10).mean().iloc[-1])
    ema20 = float(today["Close"].ewm(span=20).mean().iloc[-1])
    ema50 = float(today["Close"].ewm(span=50).mean().iloc[-1])
    ema200_series = today["Close"].ewm(span=200).mean()
    ema200 = float(ema200_series.iloc[-1])
    price_above_ema10 = today_close > ema10
    price_above_ema20 = today_close > ema20
    price_above_ema50 = today_close > ema50
    price_above_ema200 = today_close > ema200

    # EMA200 Breakout
    ema200_breakout = False
    if len(today) > ema200_lookback:
        for i in range(1, ema200_lookback + 1):
            prev_close = float(today["Close"].iloc[-i-1])
            prev_ema200 = float(ema200_series.iloc[-i-1])
            curr_close = float(today["Close"].iloc[-i])
            curr_ema200 = float(ema200_series.iloc[-i])
            if (prev_close < prev_ema200) and (curr_close > curr_ema200):
                ema200_breakout = True
                break
        ema200_breakout = ema200_breakout and (today_close > ema200)

    # MACD
    exp12 = today["Close"].ewm(span=12).mean()
    exp26 = today["Close"].ewm(span=26).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9).mean()
    macd_bullish = float(macd.iloc[-1]) > float(signal.iloc[-1])

    # Volume Spike
    last_hour = today.iloc[-12:]
    volume_spike = any(last_hour["Volume"] > 2 * avg_vol)

    # Reasons
    reasons = []
    if rs_score > 0: reasons.append("Strong RS")
    if price_above_vwap: reasons.append("VWAP Reclaim")
    if price_above_ema10 and price_above_ema20: reasons.append("EMA Alignment")
    if volume_spike: reasons.append("Volume Spike")
    if macd_bullish: reasons.append("MACD Bullish")
    if ema200_breakout: reasons.append(f"EMA200 Breakout ({ema200_lookback} bars)")

    # --- Trade management values ---
    target_price = price * (1 + take_profit_pct)
    cut_loss_price = price * (1 - cut_loss_pct)
    risk_per_share = price - cut_loss_price
    position_size = int(capital / price) if price > 0 else 0
    max_loss = position_size * risk_per_share

    # --- Prediction columns (simple extrapolation for fun)
    pred_1d = price + price * (today_change/100)
    pred_3d = price + price * (today_change/100 * 3)

    # --- GO/NO-GO/ACCUMULATION (strategy as before) ---
    go_criteria = (
        price_above_ema10 and price_above_ema20 and price_above_ema50 and
        macd_bullish and volume_spike and rs_score > 0
    )
    nogo_criteria = (
        price_above_ema50 and not (price_above_ema10 and price_above_ema20) and rs_score > -1
    )
    accumulation_criteria = (
        rs_score > 0 and price_above_ema10 and price_above_ema20
    )

    # --- AI PICK column ---
    ai_pick = ""
    pick_reason = ""
    # Only pick 1 "AI pick" per run, we'll do it after all data loaded

    return {
        "Ticker": ticker,
        "Price": f"{price:.2f}",
        "Open": f"{today_open:.2f}",
        "Close": f"{today_close:.2f}",
        "High": f"{today_high:.2f}",
        "Low": f"{today_low:.2f}",
        "Change %": f"{today_change:.2f}",
        "Avg Volume": f"{avg_vol:,.0f}",
        "Volume": f"{volume:,.0f}",
        "Rel Vol": f"{rel_vol:.2f}",
        "RS vs SPY": f"{rs_score:.2f}",
        "VWAP": "Yes" if price_above_vwap else "No",
        "EMA 10": "Yes" if price_above_ema10 else "No",
        "EMA 20": "Yes" if price_above_ema20 else "No",
        "EMA 50": "Yes" if price_above_ema50 else "No",
        "EMA 200": f"{ema200:.2f}",
        "MACD Bullish": "Yes" if macd_bullish else "No",
        "Volume Spike": "Yes" if volume_spike else "No",
        "Reasons": ", ".join(reasons),
        "GO": go_criteria,
        "NO-GO": nogo_criteria,
        "Accumulation": accumulation_criteria,
        "Target Price": f"{target_price:.2f}",
        "Cut Loss Price": f"{cut_loss_price:.2f}",
        "Position Size": position_size,
        "Max Loss at Stop": f"{max_loss:.2f}",
        "+1d Prediction": f"{pred_1d:.2f}",
        "+3d Prediction": f"{pred_3d:.2f}",
        "AI Pick": ai_pick,      # to be filled after
        "Pick Reason": pick_reason
    }

# ---- Market Health
spy = get_data('SPY')
qqq = get_data('QQQ')

if not spy.empty and not qqq.empty:
    try:
        spy_change = (float(spy['Close'].iloc[-1]) - float(spy['Close'].iloc[-2])) / float(spy['Close'].iloc[-2]) * 100
        qqq_change = (float(qqq['Close'].iloc[-1]) - float(qqq['Close'].iloc[-2])) / float(qqq['Close'].iloc[-2]) * 100
    except Exception:
        spy_change = qqq_change = 0
    st.subheader("Today's Market Conditions")
    st.markdown(f"**SPY Change:** {spy_change:.2f}%")
    st.markdown(f"**QQQ Change:** {qqq_change:.2f}%")
    go_day = spy_change >= min_index_change and qqq_change >= min_index_change
else:
    st.warning("Could not load recent market data. Try again later.")
    go_day = False

results = []
for ticker in watchlist:
    result = scan_stock_all(
        ticker,
        spy_change if 'spy_change' in locals() else 0,
        min_price, max_price, min_avg_vol,
        ema200_lookback, pullback_lookback,
        capital, take_profit_pct, cut_loss_pct
    )
    if result:
        results.append(result)
df_results = pd.DataFrame(results) if results else pd.DataFrame()

# --- AI PICK column logic ---
if not df_results.empty:
    df = df_results.copy()
    if go_day:
        # Pick: GO candidates with highest RS, high rel vol, price between 10~100
        picks = df[(df['GO']) & (df['Rel Vol'].astype(float) > 1) & (df['Price'].astype(float) > 10) & (df['Price'].astype(float) < 100)]
        if not picks.empty:
            best_idx = picks['RS vs SPY'].astype(float).idxmax()
            df.at[best_idx, 'AI Pick'] = "YES"
            df.at[best_idx, 'Pick Reason'] = "Top RS, high volume, breakout play"
    else:
        # Pick: NO-GO, lowest drop, highest volume
        picks = df[(df['NO-GO'])]
        if not picks.empty:
            best_idx = picks['Change %'].astype(float).idxmax()
            df.at[best_idx, 'AI Pick'] = "YES"
            df.at[best_idx, 'Pick Reason'] = "Most resilient, defensive setup"
    # If no above, fallback to Accumulation/EMA200 Breakout
    if "YES" not in df['AI Pick'].values:
        picks = df[(df['Accumulation'])]
        if not picks.empty:
            best_idx = picks['Rel Vol'].astype(float).idxmax()
            df.at[best_idx, 'AI Pick'] = "YES"
            df.at[best_idx, 'Pick Reason'] = "High volume, early institutional move"
    df_results = df

# ---- SHOW DASHBOARDS ----
cols = [
    "Ticker", "Price", "Open", "Close", "High", "Low", "Change %", "Volume", "Avg Volume", "Rel Vol",
    "RS vs SPY", "VWAP", "EMA 10", "EMA 20", "EMA 50", "EMA 200",
    "MACD Bullish", "Volume Spike", "Reasons",
    "Target Price", "Cut Loss Price", "Position Size", "Max Loss at Stop",
    "+1d Prediction", "+3d Prediction", "AI Pick", "Pick Reason"
]

def show_section(title, filter_column):
    st.subheader(title)
    if not df_results.empty:
        df = df_results[df_results[filter_column] == True]
        if not df.empty:
            st.dataframe(df[cols])
        else:
            st.info(f"No stocks meet {title.lower()} criteria.")
    else:
        st.info("No data for stock screening.")

show_section("Go Day Stock Recommendations", "GO")
show_section("No-Go Day Stock Recommendations", "NO-GO")
show_section("Potential Institutional Accumulation", "Accumulation")

# ---- ALERTS: Only for new, one per session ----
if 'alerted_tickers' not in st.session_state:
    st.session_state['alerted_tickers'] = set()
alerted_tickers = st.session_state['alerted_tickers']

for _, row in df_results.iterrows():
    for section_name, filter_col in [
        ("GO Day", "GO"),
        ("No-Go Day", "NO-GO"),
        ("Institutional Accumulation", "Accumulation")
    ]:
        if row[filter_col]:
            ticker = row["Ticker"]
            if not was_alerted_today(ticker, section_name):
                msg = (
                    f"📈 {section_name} ALERT!\n"
                    f"Ticker: {ticker}\n"
                    f"Price: ${row['Price']} | Target: ${row['Target Price']} | Cut Loss: ${row['Cut Loss Price']}\n"
                    f"Position: {row['Position Size']} shares | Max Loss: ${row['Max Loss at Stop']}\n"
                    f"Reasons: {row['Reasons']}"
                )
                send_telegram_alert(msg)
                add_to_alert_log(ticker, section_name)
                alerted_tickers.add((ticker, section_name))
st.session_state['alerted_tickers'] = alerted_tickers

# END OF SCRIPT
