import streamlit as st
import yfinance as yf
import pandas as pd
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
    min_avg_vol = st.number_input("Min Avg Volume", value=1_000_000)
    min_index_change = st.number_input("Min Index Change (%)", value=0.05)
    max_atr_percent = st.number_input("Max ATR (%)", value=0.015)
    volume_factor = st.number_input("Min Volume Factor", value=0.7)
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

def scan_stock_all(
    ticker, spy_change,
    min_price, max_price, min_avg_vol, ema200_lookback, pullback_lookback,
    capital, take_profit_pct, cut_loss_pct
):
    df = get_intraday_data(ticker)
    if df is None or len(df) < 50:
        return None

    today = df[df.index.date == df.index[-1].date()]
    if today.empty or len(today) < 20:
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

    # --- Indicators ---
    try:
        vwap = (today["Volume"] * (today["High"] + today["Low"] + today["Close"]) / 3).cumsum() / today["Volume"].cumsum()
        vwap_signal = today_close > float(vwap.iloc[-1])
    except Exception:
        vwap_signal = False

    try:
        ema10 = today["Close"].ewm(span=10).mean().iloc[-1]
        ema20 = today["Close"].ewm(span=20).mean().iloc[-1]
        ema50 = today["Close"].ewm(span=50).mean().iloc[-1]
        ema200_series = today["Close"].ewm(span=200).mean()
        ema200 = ema200_series.iloc[-1]
        ema10up = today_close > ema10
        ema20up = today_close > ema20
        ema50up = today_close > ema50
        ema200up = today_close > ema200
        # Check for recent EMA200 crossover
        crossed_ema200 = False
        for i in range(1, min(ema200_lookback, len(today)-1)):
            if today["Close"].iloc[-i-1] < ema200_series.iloc[-i-1] and today["Close"].iloc[-i] > ema200_series.iloc[-i]:
                crossed_ema200 = True
                break
    except Exception:
        ema10up = ema20up = ema50up = ema200up = crossed_ema200 = False

    # Volume spike: is latest volume > 2x avg 10-bar volume
    try:
        volume_spike = today["Volume"].iloc[-1] > 2 * today["Volume"].rolling(10).mean().iloc[-1]
    except Exception:
        volume_spike = False

    # MACD Bullish
    try:
        exp12 = today["Close"].ewm(span=12).mean()
        exp26 = today["Close"].ewm(span=26).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9).mean()
        macd_bullish = macd.iloc[-1] > signal.iloc[-1]
    except Exception:
        macd_bullish = False

    # RSI
    try:
        diff = today["Close"].diff()
        up = diff.clip(lower=0)
        down = -diff.clip(upper=0)
        rs = up.rolling(14).mean() / down.rolling(14).mean()
        rsi = 100 - (100 / (1 + rs))
        rsi_signal = rsi.iloc[-1] > 55
    except Exception:
        rsi_signal = False

    # --- Trade Management ---
    target_price = price * (1 + take_profit_pct)
    cut_loss_price = price * (1 - cut_loss_pct)
    position_size = int(capital / price) if price > 0 else 0

    # --- AI Score (0-100): composite scoring, can adjust weights
    ai_score = 0
    reason = []
    weights = {
        "go_day": 16,
        "rs": 20,
        "vwap": 12,
        "volume_spike": 10,
        "ema_align": 14,
        "ema200_cross": 8,
        "macd": 8,
        "rsi": 6,
        "liquidity": 6
    }
    # Market day, RS, VWAP, Volume, EMA align, EMA200 cross, MACD, RSI, liquidity
    if st.session_state.get("GO_DAY", False):
        ai_score += weights["go_day"]
        reason.append("Market GO Day")
    if rs_score > 1:
        ai_score += weights["rs"]
        reason.append("Strong RS")
    if vwap_signal:
        ai_score += weights["vwap"]
        reason.append("VWAP breakout")
    if volume_spike:
        ai_score += weights["volume_spike"]
        reason.append("Volume spike")
    if ema10up and ema20up and ema50up and ema200up:
        ai_score += weights["ema_align"]
        reason.append("All EMA align up")
    if crossed_ema200:
        ai_score += weights["ema200_cross"]
        reason.append("Recent EMA200 cross")
    if macd_bullish:
        ai_score += weights["macd"]
        reason.append("MACD bullish")
    if rsi_signal:
        ai_score += weights["rsi"]
        reason.append("RSI > 55")
    if volume > 2 * min_avg_vol:
        ai_score += weights["liquidity"]
        reason.append("High liquidity")

    return {
        "Ticker": ticker,
        "Price": f"{price:.2f}",
        "Change %": f"{today_change:.2f}",
        "RS Score": f"{rs_score:.2f}",
        "Volume": f"{volume:,.0f}",
        "Target Price": f"{target_price:.2f}",
        "Cut Loss Price": f"{cut_loss_price:.2f}",
        "Position Size": position_size,
        "AI Score": ai_score,
        "AI Reasoning": ", ".join(reason) if reason else "-"
    }

# --- Market Sentiment Analysis ---
spy = get_intraday_data('SPY')
qqq = get_intraday_data('QQQ')
spy_change = qqq_change = 0

if spy is not None and not spy.empty and qqq is not None and not qqq.empty:
    spy_today = spy[spy.index.date == spy.index[-1].date()]
    qqq_today = qqq[qqq.index.date == qqq.index[-1].date()]
    if len(spy_today) > 10 and len(qqq_today) > 10:
        try:
            spy_change = (spy_today["Close"].iloc[-1] - spy_today["Open"].iloc[0]) / spy_today["Open"].iloc[0] * 100
            qqq_change = (qqq_today["Close"].iloc[-1] - qqq_today["Open"].iloc[0]) / qqq_today["Open"].iloc[0] * 100
        except Exception:
            spy_change = qqq_change = 0
        go_day = spy_change > min_index_change and qqq_change > min_index_change
        st.session_state["GO_DAY"] = go_day
        st.subheader("Market Sentiment Panel")
        st.markdown(f"**SPY:** {spy_change:.2f}% | **QQQ:** {qqq_change:.2f}%")
        st.markdown("**Trend:** " + ("🟢 GO Day – Favorable" if go_day else "🔴 NO-GO Day – Defensive"))
    else:
        st.warning("Could not load enough recent market data. Try again later.")
        st.session_state["GO_DAY"] = False
else:
    st.warning("Could not load recent market data. Try again later.")
    st.session_state["GO_DAY"] = False

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

trade_cols = [
    "Ticker", "Price", "Change %", "RS Score", "Volume",
    "Target Price", "Cut Loss Price", "Position Size", "AI Score", "AI Reasoning"
]

if not df_results.empty:
    st.subheader("All Screener Results")
    st.dataframe(df_results[trade_cols].sort_values("AI Score", ascending=False).reset_index(drop=True), use_container_width=True)
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
            f"AI Score: {row['AI Score']} | Reason: {row['AI Reasoning']}"
        )
        send_telegram_alert(msg)
        add_to_alert_log(ticker, section_name)
        alerted_tickers.add((ticker, section_name))
st.session_state['alerted_tickers'] = alerted_tickers

# --- END ---
