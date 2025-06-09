import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, date
import requests

TELEGRAM_BOT_TOKEN = "7280991990:AAEk5x4XFCW_sTohAQGUujy1ECAQHjSY_OU"
TELEGRAM_CHAT_ID = "713264762"
ALERT_LOG = "alerts_log.csv"

st.set_page_config(page_title="US Market Day Trading Screener", layout="wide")
st.title("US Market Go/No-Go Dashboard")

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

# For debug: use smaller list
watchlist = ["AAPL","MSFT"]  # Try just two first!

def get_intraday_data(ticker):
    try:
        df = yf.download(ticker, period="2d", interval="5m", progress=False)
        st.write(f"Fetched {ticker}: {len(df)} rows")  # Debug info
        return df
    except Exception as e:
        st.error(f"Error fetching {ticker}: {e}")
        return None

def get_daily_data(ticker, days=3):
    try:
        df = yf.download(ticker, period=f"{days}d", interval="1d", progress=False)
        st.write(f"Fetched {ticker} daily: {len(df)} rows")  # Debug info
        return df
    except Exception as e:
        st.error(f"Daily error for {ticker}: {e}")
        return None

def calc_pivot_points(df_day):
    try:
        high = df_day["High"]
        low = df_day["Low"]
        close = df_day["Close"]
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        return f"{s1:.2f}", f"{r1:.2f}", f"{pivot:.2f}"
    except Exception:
        return "-", "-", "-"

def scan_stock_all(ticker, spy_change, min_price, max_price, min_avg_vol, ema200_lookback, pullback_lookback,
                   capital, take_profit_pct, cut_loss_pct):
    df = get_intraday_data(ticker)
    if df is None or len(df) < 22:
        st.warning(f"No intraday data for {ticker}")
        return None

    today = df[df.index.date == df.index[-1].date()]
    if today.empty or len(today) < 10:
        st.warning(f"Not enough intraday data for {ticker}")
        return None

    try:
        today_open = float(today["Open"].iloc[0])
        today_close = float(today["Close"].iloc[-1])
        price = today_close
        avg_vol = float(today["Volume"].rolling(10).mean().iloc[-1])
        volume = float(today["Volume"].iloc[-1])
    except Exception as e:
        st.error(f"Error parsing price/vol for {ticker}: {e}")
        return None

    if not (min_price <= price <= max_price and avg_vol >= min_avg_vol):
        st.info(f"{ticker} filtered out by price/vol filter")
        return None

    try:
        today_change = (today_close - today_open) / today_open * 100
        rs_score = today_change - spy_change
    except Exception as e:
        today_change = 0
        rs_score = 0
        st.error(f"Error calculating RS for {ticker}: {e}")

    # EMA200, VWAP logic, skip for debug
    crossed, vwap_signal = False, False

    target_price = price * (1 + take_profit_pct)
    cut_loss_price = price * (1 - cut_loss_pct)
    position_size = int(capital / price) if price > 0 else 0

    df_day = get_daily_data(ticker, days=3)
    if df_day is not None and len(df_day) >= 2:
        prev_day = df_day.iloc[-2]
        support, resistance, pivot = calc_pivot_points(prev_day)
    else:
        support, resistance, pivot = "-", "-", "-"

    ai_score = 0.6 * rs_score + 0.4 * (volume / avg_vol if avg_vol else 0)
    notes = "Test run"

    return {
        "Ticker": ticker,
        "Price": f"{price:.2f}",
        "Change %": f"{today_change:.2f}",
        "RS Score": f"{rs_score:.2f}",
        "Volume": f"{volume:,.0f}",
        "Support": support,
        "Resistance": resistance,
        "Pivot": pivot,
        "AI Score": ai_score,
        "Target Price": f"{target_price:.2f}",
        "Cut Loss Price": f"{cut_loss_price:.2f}",
        "Position Size": position_size,
        "AI Reasoning": notes
    }

spy = get_intraday_data('SPY')
qqq = get_intraday_data('QQQ')
if spy is not None and not spy.empty and qqq is not None and not qqq.empty:
    spy_today = spy[spy.index.date == spy.index[-1].date()]
    qqq_today = qqq[qqq.index.date == qqq.index[-1].date()]
    if len(spy_today) > 10 and len(qqq_today) > 10:
        try:
            spy_change = (spy_today["Close"].iloc[-1] - spy_today["Open"].iloc[0]) / spy_today["Open"].iloc[0] * 100
            qqq_change = (qqq_today["Close"].iloc[-1] - qqq_today["Open"].iloc[0]) / qqq_today["Open"].iloc[0] * 100
        except Exception:
            spy_change = qqq_change = 0
        st.subheader("Market Sentiment Panel")
        st.markdown(f"**SPY:** {spy_change:.2f}% | **QQQ:** {qqq_change:.2f}%")
    else:
        st.warning("Not enough SPY/QQQ intraday bars")
        spy_change = qqq_change = 0
else:
    st.warning("Could not load SPY/QQQ data. Try again later.")
    spy_change = qqq_change = 0

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
    "Support", "Resistance", "Pivot",
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
