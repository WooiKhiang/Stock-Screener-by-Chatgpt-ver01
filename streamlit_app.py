import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
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
    min_avg_vol = st.number_input("Min Avg Volume", value=100000)
    min_index_change = st.number_input("Min Index Change (%)", value=0.05)
    ema200_lookback = st.number_input("EMA200 Breakout Lookback", value=6, min_value=1, max_value=48)
    pullback_lookback = st.number_input("VWAP/EMA Pullback Lookback", value=6, min_value=2, max_value=20)

with st.sidebar.expander("Profit & Risk Planner", expanded=True):
    capital = st.number_input("Capital per Trade ($)", value=1000.0, step=100.0)
    atr_risk_multiple = st.number_input("ATR Multiplier for Stop Loss", value=1.2)
    atr_reward_multiple = st.number_input("ATR Multiplier for Target", value=2.4)

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

def scan_stock_all(
    ticker, spy_change,
    min_price, max_price, min_avg_vol,
    ema200_lookback, pullback_lookback,
    capital, atr_risk_multiple, atr_reward_multiple
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

    notes = []
    ai_score = 0.0
    signal_count = 0

    # --- EMA200 Crossover: Only if price is above and crossed recently ---
    try:
        ema200_series = today["Close"].ewm(span=200).mean()
        crossed = False
        for i in range(1, min(int(ema200_lookback), len(today)-1)):
            if today["Close"].iloc[-i-1] < ema200_series.iloc[-i-1] and today["Close"].iloc[-i] > ema200_series.iloc[-i]:
                crossed = True
                break
        if crossed and today_close > ema200_series.iloc[-1]:
            ai_score += 20
            signal_count += 1
            notes.append("EMA200 Cross ↑")
    except Exception:
        crossed = False

    # --- VWAP: only if >0.2% above VWAP ---
    try:
        vwap = (today["Volume"] * (today["High"] + today["Low"] + today["Close"]) / 3).cumsum() / today["Volume"].cumsum()
        vwap_signal = today_close > float(vwap.iloc[-1]) * 1.002
        if vwap_signal:
            ai_score += 20
            signal_count += 1
            notes.append("VWAP Above")
    except Exception:
        vwap_signal = False

    # --- MACD: momentum check ---
    try:
        exp12 = today["Close"].ewm(span=12).mean()
        exp26 = today["Close"].ewm(span=26).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9).mean()
        if float(macd.iloc[-1]) > float(signal.iloc[-1]) and macd.iloc[-1] > 0:
            ai_score += 15
            signal_count += 1
            notes.append("MACD Bullish")
    except Exception:
        pass

    # --- Order Flow Spike: recent volume 3x avg ---
    try:
        avg5m_vol = today["Volume"].rolling(10).mean()
        recent_bars = today.tail(12)
        if (recent_bars["Volume"] > 3 * avg5m_vol).any():
            ai_score += 10
            signal_count += 1
            notes.append("Order Flow Spike")
    except Exception:
        pass

    # --- Breakout: price > 0.5% above 10-bar high ---
    try:
        high_10 = today["High"].rolling(10).max()
        if today_close >= high_10.iloc[-1] * 1.005:
            ai_score += 20
            signal_count += 1
            notes.append("Breakout: 10-bar High")
    except Exception:
        pass

    # --- Relative Strength: >0.5% ---
    if rs_score > 0.5:
        ai_score += 15
        signal_count += 1
        notes.append("Strong RS")

    # --- RSI (overbought/oversold) ---
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

    # --- ATR calculation for dynamic stops/targets ---
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
    except Exception:
        atr = None

    # --- Dynamic risk/reward: ATR-based stops/targets ---
    if atr and atr > 0:
        stop_loss = price - atr * atr_risk_multiple
        take_profit = price + atr * atr_reward_multiple
        risk = price - stop_loss
        reward = take_profit - price
        if risk > 0:
            risk_reward_ratio = round(reward / risk, 2)
        else:
            risk_reward_ratio = "-"
    else:
        stop_loss = price * 0.99
        take_profit = price * 1.02
        risk_reward_ratio = "-"

    position_size = int(capital / price) if price > 0 else 0

    # --- Final Score/Confidence ---
    ai_score = min(100, ai_score)
    confidence_level = min(100, 25 + 15 * signal_count)

    # --- Section classifications ---
    is_go = (rs_score > 1) and vwap_signal and crossed and volume > avg_vol
    is_nogo = (rs_score < -1) or (not vwap_signal) or (not crossed)
    is_ema200 = crossed
    is_vwap = vwap_signal
    is_institutional = rs_score > 0 and volume > avg_vol

    return {
        "Ticker": ticker,
        "Price": f"{price:.2f}",
        "Change %": f"{today_change:.2f}",
        "RS Score": f"{rs_score:.2f}",
        "Volume": f"{volume:,.0f}",
        "Avg Volume": f"{avg_vol:,.0f}",
        "ATR": f"{atr:.2f}" if atr else "-",
        "Target Price": f"{take_profit:.2f}",
        "Cut Loss Price": f"{stop_loss:.2f}",
        "Position Size": position_size,
        "Risk:Reward": risk_reward_ratio,
        "AI Score": ai_score,
        "Confidence Level (%)": confidence_level,
        "AI Reasoning": ", ".join(notes) if notes else "-",
        "GO": is_go,
        "NO-GO": is_nogo,
        "EMA200": is_ema200,
        "VWAP": is_vwap,
        "Institutional": is_institutional
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
            spy_change = float((spy_today["Close"].iloc[-1] - spy_today["Open"].iloc[0]) / spy_today["Open"].iloc[0] * 100)
            qqq_change = float((qqq_today["Close"].iloc[-1] - qqq_today["Open"].iloc[0]) / qqq_today["Open"].iloc[0] * 100)
        except Exception:
            spy_change = qqq_change = 0.0
        try:
            st.subheader("Market Sentiment Panel")
            st.markdown(f"**SPY:** {spy_change:.2f}% | **QQQ:** {qqq_change:.2f}%")
            go_day = spy_change > float(min_index_change) and qqq_change > float(min_index_change)
            st.markdown("**Trend:** " + ("🟢 GO Day – Favorable" if go_day else "🔴 NO-GO Day – Defensive"))
        except Exception:
            st.markdown("**SPY/QQQ Data Error**")
    else:
        st.warning("Could not load enough recent market data. Try again later.")
        spy_change = qqq_change = 0.0
else:
    st.warning("Could not load recent market data. Try again later.")
    spy_change = qqq_change = 0.0

# --- Screener ---
results = []
for ticker in watchlist:
    result = scan_stock_all(
        ticker, spy_change,
        min_price, max_price, min_avg_vol,
        ema200_lookback, pullback_lookback,
        capital, atr_risk_multiple, atr_reward_multiple
    )
    if result:
        results.append(result)
df_results = pd.DataFrame(results) if results else pd.DataFrame()

trade_cols = [
    "Ticker", "Price", "Change %", "RS Score", "Volume", "Avg Volume", "ATR",
    "Target Price", "Cut Loss Price", "Position Size", "Risk:Reward",
    "AI Score", "Confidence Level (%)", "AI Reasoning"
]

# --- Section Tables ---
def show_section(title, filter_col):
    st.subheader(title)
    if not df_results.empty:
        df = df_results[df_results[filter_col] == True]
        if not df.empty:
            st.dataframe(df[trade_cols].sort_values("AI Score", ascending=False).reset_index(drop=True), use_container_width=True)
        else:
            st.info(f"No stocks meet {title.lower()} criteria.")
    else:
        st.info("No data for stock screening.")

show_section("Go Day Stock Recommendations (Momentum/Breakout)", "GO")
show_section("No-Go Day Stock Recommendations (Defensive)", "NO-GO")
show_section("EMA200 Crossover Screener", "EMA200")
show_section("VWAP Above Screener", "VWAP")
show_section("Potential Institutional Accumulation (RS > 0 & High Volume)", "Institutional")

# --- AI Top 5 Picks ---
if not df_results.empty:
    df_ai_top = df_results.sort_values("AI Score", ascending=False).head(5).copy()
    st.subheader("⭐ Top 5 AI Picks of the Day")
    st.dataframe(df_ai_top[trade_cols].reset_index(drop=True), use_container_width=True)
else:
    st.warning("⚠️ No stocks meet your screener criteria. Try relaxing the filter settings.")

# --- Alerts (for Top 5 only) ---
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
            f"Risk:Reward: {row['Risk:Reward']}\n"
            f"Reason: {row['AI Reasoning']}"
        )
        send_telegram_alert(msg)
        add_to_alert_log(ticker, section_name)
        alerted_tickers.add((ticker, section_name))
st.session_state['alerted_tickers'] = alerted_tickers

# --- END ---
