import streamlit as st
import yfinance as yf
import pandas as pd
import ta

st.set_page_config(page_title="US Mean Reversion Screener", layout="wide")
st.title("US Market Mean Reversion (Trend-Filtered) Screener")

# --- Sidebar: Screener Criteria ---
with st.sidebar.expander("Screener Criteria", expanded=True):
    min_price = st.number_input("Min Price ($)", value=5.0)
    max_price = st.number_input("Max Price ($)", value=500.0)
    min_avg_vol = st.number_input("Min Avg Volume", value=100_000)

watchlist = [ ... ]  # your tickers

def mean_reversion_signal(ticker, min_price, max_price, min_avg_vol):
    df = yf.download(ticker, period="1y", interval="1d", progress=False)
    if df is None or df.empty or len(df) < 210:
        return None
    df['SMA200'] = ta.trend.sma_indicator(df['Close'], window=200)
    df['RSI2'] = ta.momentum.rsi(df['Close'], window=2)
    df['AvgVol20'] = ta.trend.sma_indicator(df['Volume'], window=20)
    latest = df.iloc[-1]
    if (
        (min_price <= latest['Close'] <= max_price)
        and (latest['Close'] > latest['SMA200'])
        and (latest['RSI2'] < 10)
        and (latest['Volume'] > min_avg_vol)
        and (latest['Volume'] > latest['AvgVol20'])
    ):
        return {
            "Ticker": ticker,
            "Price": f"{latest['Close']:.2f}",
            "RSI(2)": f"{latest['RSI2']:.2f}",
            "SMA200": f"{latest['SMA200']:.2f}",
            "Volume": f"{latest['Volume']:,}",
            "Avg Volume (20)": f"{latest['AvgVol20']:,}",
            "Signal": "BUY Mean Reversion"
        }
    else:
        return None

results = []
for ticker in watchlist:
    signal = mean_reversion_signal(ticker, min_price, max_price, min_avg_vol)
    if signal:
        results.append(signal)
df_results = pd.DataFrame(results) if results else pd.DataFrame()

if not df_results.empty:
    df_results = df_results.sort_values("RSI(2)", ascending=True)

st.subheader("📉 Mean Reversion Buy Signals (Above SMA200, RSI(2)<10)")
if not df_results.empty:
    st.dataframe(df_results.reset_index(drop=True), use_container_width=True)
else:
    st.info("No stocks meet the mean reversion + trend criteria. Try adjusting filter settings.")
