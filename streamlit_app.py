import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, date
import numpy as np
import requests

# --- SETTINGS ---
TELEGRAM_BOT_TOKEN = "7280991990:AAEk5x4XFCW_sTohAQGUujy1ECAQHjSY_OU"
TELEGRAM_CHAT_ID = "713264762"
ALERT_LOG = "alerts_log.csv"

st.set_page_config(page_title="US Market Day Trading Screener", layout="wide")
st.title("US Market Go/No-Go Dashboard")

# --- Sidebar: Screener Criteria ---
with st.sidebar.expander("Screener Criteria", expanded=True):
    min_price = st.number_input("Min Price ($)", value=5.0)
    max_price = st.number_input("Max Price ($)", value=500.0)
    min_avg_vol = st.number_input("Min Avg Volume", value=1_000_000)
    min_index_change = st.number_input("Min Index Change (%)", value=0.05)
    ema200_lookback = st.number_input("EMA200 Breakout Lookback", value=6, min_value=1, max_value=48)
    pullback_lookback = st.number_input("VWAP/EMA Pullback Lookback", value=6, min_value=2, max_value=20)

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
    try:
        with open(ALERT_LOG, "r") as f:
            for row in f:
                cols = row.strip().split(",")
                if len(cols) == 3 and cols[0] == today_str and cols[1] == section_name and cols[2] == ticker:
                    return True
    except FileNotFoundError:
        return False
    return False

def add_to_alert_log(ticker, section_name):
    today_str = str(date.today())
    with open(ALERT_LOG, "a") as f:
        f.write(f"{today_str},{section_name},{ticker}\n")

def get_intraday_data(ticker):
    try:
        df = yf.download(ticker, period="2d", interval="5m", progress=False)
    except Exception:
        return None
    return df

def get_daily_data(ticker, days=15):
    try:
        df = yf.download(ticker, period=f"{days}d", interval="1d", progress=False)
    except Exception:
        return None
    return df

def scan_stock_all(
    ticker, spy_change,
    min_price, max_price, min_avg_vol,
    ema200_lookback, pullback_lookback,
    capital, take_profit_pct, cut_loss_pct
):
    df = get_intraday_data(ticker)
    if df is None or len(df) < 22:
        return None
    today = df[df.index.date == df.index[-1].date()]
    if today.empty or len(today) < 10:
        return None

    try:
        today_open = float(today["Open"].iloc[0])
        today_close = float(today["Close"].iloc[-1])
        price = today_close
        avg_vol = float(today["Volume"].rolling(10).mean().iloc[-1])
        volume = float(today["Volume"].iloc[-1])
    except Exception:
        return None

    if not (min_price <= price <= max_price and avg_vol >= min_avg_vol):
        return None

    try:
        today_change = (today_close - today_open) / today_open * 100
        rs_score = today_change - spy_change
    except Exception:
        today_change = 0
        rs_score = 0

    # --- EMA/VWAP, MACD, RSI, ATR, etc. ---
    notes = []
    ai_score = 0

    # --- EMA Cross (200) ---
    try:
        ema200_series = today["Close"].ewm(span=200).mean()
        ema200 = float(ema200_series.iloc[-1])
        crossed = False
        for i in range(1, min(ema200_lookback, len(today)-1)):
            if today["Close"].iloc[-i-1] < ema200_series.iloc[-i-1] and today["Close"].iloc[-i] > ema200_series.iloc[-i]:
                crossed = True
                break
        if crossed:
            ai_score += 0.6
            notes.append("EMA200 Cross ↑")
    except Exception:
        ema200, crossed = 0, False

    # --- VWAP ---
    try:
        vwap = (today["Volume"] * (today["High"] + today["Low"] + today["Close"]) / 3).cumsum() / today["Volume"].cumsum()
        vwap_signal = today_close > float(vwap.iloc[-1])
        if vwap_signal:
            ai_score += 0.5
            notes.append("VWAP Above")
    except Exception:
        vwap_signal = False

    # --- MACD ---
    try:
        exp12 = today["Close"].ewm(span=12).mean()
        exp26 = today["Close"].ewm(span=26).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9).mean()
        if float(macd.iloc[-1]) > float(signal.iloc[-1]):
            ai_score += 0.4
            notes.append("MACD Bullish")
    except Exception:
        pass

    # --- RSI ---
    try:
        delta = today["Close"].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        roll_up = up.rolling(14).mean()
        roll_down = down.rolling(14).mean()
        rs = roll_up / (roll_down + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        if rsi.iloc[-1] > 70:
            notes.append("RSI Overbought")
        elif rsi.iloc[-1] < 30:
            notes.append("RSI Oversold")
    except Exception:
        pass

    # --- ATR (Dynamic Stop) ---
    try:
        high = today["High"]
        low = today["Low"]
        close = today["Close"]
        tr = pd.concat([
            (high - low),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        notes.append(f"ATR: {atr:.2f}")
    except Exception:
        atr = 0

    # --- High Liquidity (Order Flow) ---
    try:
        avg5m_vol = today["Volume"].rolling(10).mean()
        recent_bars = today.tail(12)
        if (recent_bars["Volume"] > 2 * avg5m_vol).any():
            ai_score += 0.3
            notes.append("Order Flow Spike")
    except Exception:
        pass

    # --- 10-bar High/Low Breakout ---
    try:
        high_10 = today["High"].rolling(10).max()
        low_10 = today["Low"].rolling(10).min()
        if today_close >= high_10.iloc[-1]:
            ai_score += 0.5
            notes.append("Breakout: 10-bar High")
        elif today_close <= low_10.iloc[-1]:
            ai_score -= 0.2
            notes.append("Breakdown: 10-bar Low")
    except Exception:
        pass

    # --- Range Contraction/Expansion ---
    try:
        range_10 = today["High"].rolling(10).max() - today["Low"].rolling(10).min()
        range_20 = today["High"].rolling(20).max() - today["Low"].rolling(20).min()
        if range_10.iloc[-1] < 0.7 * range_20.iloc[-1]:
            notes.append("Range Contraction")
        elif range_10.iloc[-1] > 1.5 * range_20.iloc[-1]:
            notes.append("Range Expansion")
    except Exception:
        pass

    # --- Relative Strength ---
    if rs_score > 0:
        ai_score += 0.4
        notes.append("Strong RS")

    # --- Trade Management ---
    target_price = price * (1 + take_profit_pct)
    cut_loss_price = price * (1 - cut_loss_pct)
    position_size = int(capital / price) if price > 0 else 0

    return {
        "Ticker": ticker,
        "Price": f"{price:.2f}",
        "Change %": f"{today_change:.2f}",
        "RS Score": f"{rs_score:.2f}",
        "Volume": f"{volume:,.0f}",
        "Avg Volume": f"{avg_vol:,.0f}",
        "Target Price": f"{target_price:.2f}",
        "Cut Loss Price": f"{cut_loss_price:.2f}",
        "Position Size": position_size,
        "AI Score": ai_score,
        "AI Reasoning": ", ".join(notes) if notes else "-"
    }

# --- Market Sentiment Analysis ---
spy = get_intraday_data('SPY')
qqq = get_intraday_data('QQQ')
go_day = False

if spy is not None and not spy.empty and qqq is not None and not qqq.empty:
    spy_today = spy[spy.index.date == spy.index[-1].date()]
    qqq_today = qqq[qqq.index.date == qqq.index[-1].date()]
    if len(spy_today) > 10 and len(qqq_today) > 10:
        try:
            spy_change = (spy_today["Close"].iloc[-1] - spy_today["Open"].iloc[0]) / spy_today["Open"].iloc[0] * 100
            qqq_change = (qqq_today["Close"].iloc[-1] - qqq_today["Open"].iloc[0]) / qqq_today["Open"].iloc[0] * 100
        except Exception:
            spy_change = qqq_change = 0
        # Market trend: basic
        go_day = float(spy_change) > float(min_index_change) and float(qqq_change) > float(min_index_change)
        st.subheader("Market Sentiment Panel")
        st.markdown(f"**SPY:** {spy_change:.2f}% | **QQQ:** {qqq_change:.2f}%")
        st.markdown("**Trend:** " + ("🟢 GO Day – Favorable" if go_day else "🔴 NO-GO Day – Defensive"))
    else:
        st.warning("Could not load enough recent market data. Try again later.")
        spy_change = qqq_change = 0
else:
    st.warning("Could not load recent market data. Try again later.")
    spy_change = qqq_change = 0

# --- Screener ---
results = []
for ticker in watchlist:
    result = scan_stock_all(
        ticker, spy_change,
        min_price, max_price, min_avg_vol,
        ema200_lookback, pullback_lookback,
        capital, take_profit_pct, cut_loss_pct
    )
    if result:
        results.append(result)
df_results = pd.DataFrame(results) if results else pd.DataFrame()

# --- Show results, AI Picks, Alerts ---
trade_cols = [
    "Ticker", "Price", "Change %", "RS Score", "Volume", "Avg Volume",
    "Target Price", "Cut Loss Price", "Position Size", "AI Score", "AI Reasoning"
]

if not df_results.empty:
    # Show all that pass the screener (sorted)
    st.subheader("All Screener Results")
    st.dataframe(df_results[trade_cols].sort_values("AI Score", ascending=False).reset_index(drop=True), use_container_width=True)
    
    # AI Top 5 Picks
    df_ai_top = df_results.sort_values("AI Score", ascending=False).head(5).copy()
    st.subheader("⭐ Top 5 AI Picks of the Day")
    st.dataframe(df_ai_top[trade_cols].reset_index(drop=True), use_container_width=True)
else:
    st.warning("⚠️ No stocks meet your screener criteria. Try relaxing the filter settings.")

# --- Alerts (for Top 5 only, to avoid spam) ---
if 'alerted_tickers' not in st.session_state:
    st.session_state['alerted_tickers'] = set()
alerted_tickers = st.session_state['alerted_tickers']

for _, row in df_results.sort_values("AI Score", ascending=False).head(5).iterrows():
    section_name = "AI Top 5"
    ticker = row["Ticker"]
    if not was_alerted_today(ticker, section_name):
        msg = (
            f"📈 {section_name} ALERT!\n"
            f"Ticker: {ticker}\n"
            f"Price: ${row['Price']} | Target: ${row['Target Price']} | Cut Loss: ${row['Cut Loss Price']}\n"
            f"Reason: {row['AI Reasoning']}"
        )
        send_telegram_alert(msg)
        add_to_alert_log(ticker, section_name)
        alerted_tickers.add((ticker, section_name))
st.session_state['alerted_tickers'] = alerted_tickers

# --- END ---
