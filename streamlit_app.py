import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.title("🔍 Mini Debug Screener: S&P Tickers")

# --- Select tickers for quick test ---
tickers = st.multiselect(
    "Pick tickers (test with a few, then expand)",
    options=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA', 'SPY', 'META'],
    default=['AAPL', 'MSFT', 'GOOGL']
)

min_price = st.number_input("Min Price ($)", value=0.0)
max_price = st.number_input("Max Price ($)", value=2000.0)
min_vol = st.number_input("Min Avg Vol (40 bars)", value=0)

results = []

for ticker in tickers:
    df = yf.download(ticker, period='5d', interval='5m', progress=False, threads=False)
    if df.empty or len(df) < 3:
        results.append({'Ticker': ticker, 'Status': 'No data'})
        continue

    # Normalize columns to lowercase and underscore (robust!)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Compute indicators
    df['sma3'] = df['close'].rolling(3).mean()
    df['avgvol40'] = df['volume'].rolling(40, min_periods=1).mean()

    last = df.iloc[-1]
    close_price = last['close']
    avgvol40 = last['avgvol40']

    # Apply simple filters
    if not (min_price <= close_price <= max_price):
        results.append({'Ticker': ticker, 'Status': 'Filtered: Price', 'Last Close': close_price, 'AvgVol40': avgvol40})
        continue
    if avgvol40 < min_vol:
        results.append({'Ticker': ticker, 'Status': 'Filtered: Volume', 'Last Close': close_price, 'AvgVol40': avgvol40})
        continue

    # Simple debug signal: Close > SMA3
    if close_price > last['sma3']:
        results.append({
            'Ticker': ticker,
            'Status': 'DEBUG SIGNAL FIRED',
            'Last Close': round(close_price, 2),
            'SMA3': round(last['sma3'], 2),
            'AvgVol40': int(avgvol40)
        })
    else:
        results.append({
            'Ticker': ticker,
            'Status': 'No Debug Signal',
            'Last Close': round(close_price, 2),
            'SMA3': round(last['sma3'], 2),
            'AvgVol40': int(avgvol40)
        })

st.dataframe(pd.DataFrame(results))

st.markdown("""
---
**Instructions:**  
- If you see "DEBUG SIGNAL FIRED" for any ticker, your data pipeline is working.
- Once you see signals, expand your ticker list and start layering in real strategy logic, one-by-one.
""")
