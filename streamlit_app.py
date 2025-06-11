import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# --- Use only 2 tickers for quick debug ---
sp500 = ['AAPL', 'MSFT']  # Add more later

# --- Sidebar: Filters ---
st.sidebar.header("Filter Settings")
min_price = st.sidebar.number_input("Min Price ($)", value=0.0)
max_price = st.sidebar.number_input("Max Price ($)", value=10000.0)
min_volume = st.sidebar.number_input("Min Avg Vol (40 bars)", value=0)
capital_per_trade = st.sidebar.number_input("Capital per Trade ($)", value=1000.0, step=100.0)

st.title("⚡ S&P 500 - 5-Minute Intraday Trade Screener (DEBUG MODE)")
st.caption(f"Last run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (5-min chart, 5-day lookback)")

def calc_indicators(df):
    df['SMA40'] = df['Close'].rolling(window=40).mean()
    df['EMA40'] = df['Close'].ewm(span=40, min_periods=40).mean()
    df['EMA8'] = df['Close'].ewm(span=8, min_periods=8).mean()
    df['EMA21'] = df['Close'].ewm(span=21, min_periods=21).mean()
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(window=3).mean()
    roll_down = down.rolling(window=3).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['RSI3'] = 100 - (100 / (1 + rs))
    ema12 = df['Close'].ewm(span=12, min_periods=12).mean()
    ema26 = df['Close'].ewm(span=26, min_periods=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, min_periods=9).mean()
    df['AvgVol40'] = df['Volume'].rolling(window=40).mean()
    return df

def mean_reversion_signal(df):
    # For debug: just always return True if data is present!
    return True, "DEBUG: Data present (not using indicators)"

def ema40_breakout_signal(df):
    return False, None

def macd_ema_signal(df):
    return False, None

results = []
st.sidebar.subheader("Ticker Status:")
for ticker in sp500:
    try:
        df = yf.download(ticker, period="5d", interval="5m", progress=False)
        if df.empty or len(df) < 50:
            st.sidebar.write(f"{ticker}: No data or <50 bars")
            continue
        st.sidebar.write(f"{ticker}: {len(df)} bars OK")
        df = calc_indicators(df)
        last = df.iloc[-1]
        close_price = float(last['Close'])
        avgvol40 = float(last['AvgVol40'])
        # Show most recent bars in main app
        st.subheader(f"Raw 5-min data for {ticker}")
        st.dataframe(df.tail(5))
        # Loosen all filters for debug
        strat, reason, rank = "DEBUG", "Showing all results (bypass filters)", 0
        entry = close_price
        tp = entry * 1.01
        sl = entry * 0.995
        shares = int(capital_per_trade // entry)
        invested = shares * entry
        results.append({
            "Ticker": ticker,
            "Strategy": strat,
            "Rank": rank,
            "Entry Price": round(entry, 2),
            "Capital Used": round(invested, 2),
            "Shares": shares,
            "TP (if fixed)": round(tp, 2) if tp else "-",
            "SL": round(sl, 2),
            "Reason": reason,
            "Volume": int(last['Volume']),
            "Avg Vol (40)": int(avgvol40),
            "RSI(3)": round(last['RSI3'], 2),
            "EMA40": round(last['EMA40'], 2),
            "SMA40": round(last['SMA40'], 2)
        })
    except Exception as e:
        st.sidebar.write(f"{ticker}: Exception - {e}")
        continue

df_results = pd.DataFrame(results)

st.header("DEBUG: Results (all tickers shown, no real strategy)")
if not df_results.empty:
    st.dataframe(df_results.reset_index(drop=True), use_container_width=True)
else:
    st.info("No stocks data loaded at all. Check internet connection or Yahoo API status.")

# --- Top 5 by Capital Invested ---
if not df_results.empty:
    st.subheader("💰 Top by Capital Invested (DEBUG)")
    top_cap = df_results.sort_values("Capital Used", ascending=False).head(5)
    st.table(top_cap[["Ticker", "Capital Used", "Strategy", "Entry Price", "Shares", "Volume"]].reset_index(drop=True))

# --- Top 5 by Volume ---
if not df_results.empty:
    st.subheader("🔥 Top by Volume (Latest Bar, DEBUG)")
    top_vol = df_results.sort_values("Volume", ascending=False).head(5)
    st.table(top_vol[["Ticker", "Volume", "Strategy", "Entry Price", "Shares", "Capital Used"]].reset_index(drop=True))

# --- Top 5 Performers by % Change in Last 5-Min Bar (robust version) ---
perf_results = []
for ticker in sp500:
    try:
        df = yf.download(ticker, period="2d", interval="5m", progress=False)
        if df.empty:
            st.sidebar.write(f"{ticker}: Perf, no data")
            continue
        closes = df['Close'].dropna()
        if len(closes) < 2:
            st.sidebar.write(f"{ticker}: Perf, <2 closes")
            continue
        last_close = closes.iloc[-1]
        prev_close = closes.iloc[-2]
        last_idx = closes.index[-1]
        last_row = df.loc[last_idx]
        last_vol = last_row['Volume'] if not np.isnan(last_row['Volume']) else 0
        if prev_close == 0:
            st.sidebar.write(f"{ticker}: Perf, prev_close=0")
            continue
        pct_change = 100 * (last_close - prev_close) / prev_close
        perf_results.append({
            "Ticker": ticker,
            "Last Close": round(last_close, 2),
            "Prev Close": round(prev_close, 2),
            "Change (%)": pct_change,
            "Volume": int(last_vol)
        })
    except Exception as e:
        st.sidebar.write(f"{ticker}: Perf Exception - {e}")
        continue

df_perf = pd.DataFrame(perf_results)
if not df_perf.empty:
    df_perf = df_perf.dropna(subset=["Change (%)"])
    df_perf["Change (%)"] = pd.to_numeric(df_perf["Change (%)"], errors="coerce")
    df_perf = df_perf.dropna(subset=["Change (%)"])
    if not df_perf.empty:
        st.subheader("🚀 Top 5 Performers (Last 5-Min Session, DEBUG)")
        top_perf = df_perf.sort_values("Change (%)", ascending=False).head(5)
        st.table(top_perf.reset_index(drop=True))
    else:
        st.info("No valid performance data for this session.")
else:
    st.info("No recent 5-min data for performance check.")

st.caption("DEBUG MODE: All results shown regardless of indicators. If you see data here, Yahoo API works!")

