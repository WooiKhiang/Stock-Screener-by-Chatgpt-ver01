import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

st.title("Market Go/No-Go Dashboard")

# Get user inputs for parameters (optional)
min_index_change = st.number_input("Min Index Change (%)", value=0.05)
max_atr_percent = st.number_input("Max ATR (%)", value=0.015)
volume_factor = st.number_input("Min Volume Factor", value=0.7)

# Load SPY, QQQ, VXZ data (last 2 days for pre-market, or adjust as needed)
def get_data(ticker):
    end = datetime.now()
    start = end - timedelta(days=5)
    df = yf.download(ticker, start=start, end=end, interval='5m')
    return df

spy = get_data('SPY')
qqq = get_data('QQQ')
vxx = get_data('VXZ')  # Use VXZ as VXX is discontinued

if not spy.empty and not qqq.empty:
    spy_change = (spy['Close'].iloc[-1] - spy['Close'].iloc[-2]) / spy['Close'].iloc[-2] * 100
    qqq_change = (qqq['Close'].iloc[-1] - qqq['Close'].iloc[-2]) / qqq['Close'].iloc[-2] * 100
    spy_volume = spy['Volume'].iloc[-1]
    spy_avg_volume = spy['Volume'].rolling(window=10).mean().iloc[-1]
    spy_atr = (spy['High'] - spy['Low']).rolling(window=14).mean().iloc[-1]
    atr_percent = spy_atr / spy['Close'].iloc[-1]
    vxx_fast = vxx['Close'].rolling(window=5).mean().iloc[-1]
    vxx_slow = vxx['Close'].rolling(window=20).mean().iloc[-1]
    vxx_diff = (vxx_fast - vxx_slow) / vxx_slow if vxx_slow != 0 else 0

    # Conditions
    market_health = spy_change >= min_index_change and qqq_change >= min_index_change
    liquidity = spy_volume >= spy_avg_volume * volume_factor
    volatility = atr_percent <= max_atr_percent
    sentiment = vxx_diff < 0.03

    go_day = market_health and liquidity and volatility and sentiment

    st.subheader("Today's Market Conditions")
    st.markdown(f"**SPY Change:** {spy_change:.2f}%")
    st.markdown(f"**QQQ Change:** {qqq_change:.2f}%")
    st.markdown(f"**Volume Factor:** {spy_volume / spy_avg_volume:.2f}x")
    st.markdown(f"**ATR %:** {atr_percent:.3f}")
    st.markdown(f"**VXZ Fast/Slow Diff:** {vxx_diff:.3f}")

    st.subheader("GO/NO-GO Signal")
    if go_day:
        st.success("GO DAY! Risk-on opportunities detected.")
    else:
        st.error("NO-GO DAY. Consider capital protection strategies.")

else:
    st.warning("Could not load recent market data. Try again later.")

st.caption("Data source: Yahoo Finance")
