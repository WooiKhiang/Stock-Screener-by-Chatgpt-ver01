import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

st.title("Market Go/No-Go Dashboard")

# --- User Parameters ---
min_index_change = st.number_input("Min Index Change (%)", value=0.05)
max_atr_percent = st.number_input("Max ATR (%)", value=0.015)
volume_factor = st.number_input("Min Volume Factor", value=0.7)

# --- Data Fetching ---
def get_data(ticker):
    end = datetime.now()
    start = end - timedelta(days=5)
    df = yf.download(ticker, start=start, end=end, interval='5m', progress=False)
    return df

spy = get_data('SPY')
qqq = get_data('QQQ')
vxx = get_data('VXZ')  # VXZ as replacement for VXX

if not spy.empty and not qqq.empty and not vxx.empty:
    # --- Change Calculations ---
    if len(spy) > 2 and len(qqq) > 2:
        spy_change = (spy['Close'].iloc[-1] - spy['Close'].iloc[-2]) / spy['Close'].iloc[-2] * 100
        qqq_change = (qqq['Close'].iloc[-1] - qqq['Close'].iloc[-2]) / qqq['Close'].iloc[-2] * 100
    else:
        spy_change = qqq_change = 0

    # --- Volume & ATR ---
    if len(spy) > 15:
        spy_volume = spy['Volume'].iloc[-1]
        spy_avg_volume = spy['Volume'].rolling(window=10).mean().iloc[-1]
        spy_atr = (spy['High'] - spy['Low']).rolling(window=14).mean().iloc[-1]
        close_last = spy['Close'].iloc[-1]
        if pd.notna(close_last) and close_last != 0:
            atr_percent = spy_atr / close_last
        else:
            atr_percent = 0
        vol_factor_val = spy_volume / spy_avg_volume if spy_avg_volume != 0 else 0
    else:
        spy_volume = spy_avg_volume = spy_atr = atr_percent = vol_factor_val = 0

    # --- Sentiment (VXZ) ---
    if len(vxx) > 20:
        vxx_fast = vxx['Close'].rolling(window=5).mean().iloc[-1]
        vxx_slow = vxx['Close'].rolling(window=20).mean().iloc[-1]
        if pd.notna(vxx_slow) and vxx_slow != 0:
            vxx_diff = (vxx_fast - vxx_slow) / vxx_slow
        else:
            vxx_diff = 0
    else:
        vxx_diff = 0

    # --- Go/No-Go Logic ---
    market_health = spy_change >= min_index_change and qqq_change >= min_index_change
    liquidity = vol_factor_val >= volume_factor
    volatility = atr_percent <= max_atr_percent
    sentiment = vxx_diff < 0.03

    go_day = market_health and liquidity and volatility and sentiment

    # --- Dashboard ---
    st.subheader("Today's Market Conditions")
    st.markdown(f"**SPY Change:** {spy_change:.2f}%")
    st.markdown(f"**QQQ Change:** {qqq_change:.2f}%")
    st.markdown(f"**Volume Factor:** {vol_factor_val:.2f}x")
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
