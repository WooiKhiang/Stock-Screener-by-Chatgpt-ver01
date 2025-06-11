import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# --- S&P 500 Shortlist (edit or expand to full S&P 500 as needed) ---
sp500 = [
    'AAPL','MSFT','GOOGL','AMZN','NVDA','META','BRK-B','JPM','UNH','XOM',
    'LLY','JNJ','V','PG','MA','AVGO','HD','MRK','COST','ADBE'
    # ...expand to full S&P 500 list if running locally and not limited by Streamlit Cloud
]

# --- Sidebar: Filters ---
st.sidebar.header("Filter Settings")
min_price = st.sidebar.number_input("Min Price ($)", value=10.0)
max_price = st.sidebar.number_input("Max Price ($)", value=2000.0)
min_volume = st.sidebar.number_input("Min Avg Vol (40 bars)", value=50_000)
capital_per_trade = st.sidebar.number_input("Capital per Trade ($)", value=1000.0, step=100.0)

st.title("⚡ S&P 500 - 5-Minute Intraday Trade Screener")
st.caption(f"Last run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (5-min chart, 5-day lookback)")

# --- Helper Functions: Indicators ---
def calc_indicators(df):
    df['SMA40'] = df['Close'].rolling(window=40).mean()
    df['EMA40'] = df['Close'].ewm(span=40, min_periods=40).mean()
    df['EMA8'] = df['Close'].ewm(span=8, min_periods=8).mean()
    df['EMA21'] = df['Close'].ewm(span=21, min_periods=21).mean()
    # RSI(3) for intraday oversold
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(window=3).mean()
    roll_down = down.rolling(window=3).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['RSI3'] = 100 - (100 / (1 + rs))
    # MACD (intraday fast)
    ema12 = df['Close'].ewm(span=12, min_periods=12).mean()
    ema26 = df['Close'].ewm(span=26, min_periods=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, min_periods=9).mean()
    # Avg 5-min Vol (40 bars ~ 3.5 trading hours)
    df['AvgVol40'] = df['Volume'].rolling(window=40).mean()
    return df

# --- Strategy Functions ---
def mean_reversion_signal(df):
    cond = (
        (df['Close'].iloc[-1] > df['SMA40'].iloc[-1]) &
        (df['RSI3'].iloc[-1] < 15)
    )
    if cond:
        return True, "Mean Reversion: Price > SMA40 & RSI(3)<15"
    return False, None

def ema40_breakout_signal(df):
    close = df['Close'].iloc[-1]
    ema40 = df['EMA40'].iloc[-1]
    prev_close = df['Close'].iloc[-2]
    prev_ema40 = df['EMA40'].iloc[-2]
    dipped = (df['Close'].iloc[-10:-1] < df['EMA40'].iloc[-10:-1]).any()
    cond = (close > ema40) and ((prev_close < prev_ema40) or dipped)
    if cond:
        return True, "EMA40 Breakout: Price reclaimed EMA40 (with shakeout)"
    return False, None

def macd_ema_signal(df):
    macd = df['MACD'].iloc[-1]
    macd_signal = df['MACD_signal'].iloc[-1]
    macd_prev = df['MACD'].iloc[-2]
    macd_signal_prev = df['MACD_signal'].iloc[-2]
    ema8 = df['EMA8'].iloc[-1]
    ema21 = df['EMA21'].iloc[-1]
    cross = (macd_prev < macd_signal_prev) and (macd > macd_signal) and (macd < 0)
    cond = cross and (ema8 > ema21)
    if cond:
        return True, "MACD+EMA: MACD cross up <0 & EMA8>EMA21"
    return False, None

# --- Main Scan Loop ---
results = []
for ticker in sp500:
    try:
        df = yf.download(ticker, period="5d", interval="5m", progress=False)
        if df.empty or len(df) < 50:
            continue
        df = calc_indicators(df)
        last = df.iloc[-1]
        close_price = float(last['Close'])
        avgvol40 = float(last['AvgVol40'])
        if np.isnan(close_price) or not (min_price <= close_price <= max_price):
            continue
        if np.isnan(avgvol40) or avgvol40 < min_volume:
            continue

        strat, reason = None, ""
        rank = 0

        mr, mr_reason = mean_reversion_signal(df)
        ema, ema_reason = ema40_breakout_signal(df)
        macdema, macdema_reason = macd_ema_signal(df)

        if mr:
            strat, reason, rank = "Mean Reversion", mr_reason, 1
        elif ema:
            strat, reason, rank = "EMA40 Breakout", ema_reason, 2
        elif macdema:
            strat, reason, rank = "MACD+EMA", macdema_reason, 3

        if strat:
            entry = close_price
            if strat == "Mean Reversion":
                tp = entry * 1.01  # 1% target
                sl = entry * 0.995 # 0.5% stop
            elif strat == "EMA40 Breakout":
                tp = None
                sl = entry * 0.99
            elif strat == "MACD+EMA":
                tp = None
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
        continue

df_results = pd.DataFrame(results)

# --- Dashboard Output ---
st.header("5-Minute Trade Recommendations (S&P 500 sample)")
if not df_results.empty:
    df_results = df_results.sort_values(["Rank", "Strategy", "RSI(3)"])
    st.dataframe(df_results.reset_index(drop=True), use_container_width=True)
else:
    st.info("No stocks meet your filter/strategy criteria right now.")

# --- Summary Section ---
if not df_results.empty:
    st.subheader("🔔 Summary & Highlight")
    picks = df_results.groupby("Strategy").first().sort_values("Rank")
    for idx, row in picks.iterrows():
        st.markdown(f"**[{row['Strategy']}] {row['Ticker']}** | Entry: ${row['Entry Price']} | Shares: {row['Shares']} | Reason: {row['Reason']}")
else:
    st.info("No strategy triggered this 5-min bar. Try next run.")

# --- Top 5 by Capital Invested ---
if not df_results.empty:
    st.subheader("💰 Top 5 by Capital Invested")
    top_cap = df_results.sort_values("Capital Used", ascending=False).head(5)
    st.table(top_cap[["Ticker", "Capital Used", "Strategy", "Entry Price", "Shares", "Volume"]].reset_index(drop=True))

# --- Top 5 by Volume ---
if not df_results.empty:
    st.subheader("🔥 Top 5 by Volume (Latest Bar)")
    top_vol = df_results.sort_values("Volume", ascending=False).head(5)
    st.table(top_vol[["Ticker", "Volume", "Strategy", "Entry Price", "Shares", "Capital Used"]].reset_index(drop=True))

st.caption("Strategies: 1) Mean Reversion, 2) EMA40 Breakout, 3) MACD+EMA. Exits: Mean Reversion = TP +1%/SL -0.5%. EMA40/Combo = trailing stop or indicator exit.")

# --- Top 5 Performers by % Change in Last 5-Min Bar ---
perf_results = []
for ticker in sp500:
    try:
        df = yf.download(ticker, period="2d", interval="5m", progress=False)
        if df.empty or len(df) < 2:
            continue
        last = df.iloc[-1]
        prev = df.iloc[-2]
        pct_change = 100 * (last['Close'] - prev['Close']) / prev['Close']
        perf_results.append({
            "Ticker": ticker,
            "Last Close": round(last['Close'], 2),
            "Prev Close": round(prev['Close'], 2),
            "Change (%)": round(pct_change, 2),
            "Volume": int(last['Volume'])
        })
    except Exception as e:
        continue

df_perf = pd.DataFrame(perf_results)
if not df_perf.empty:
    st.subheader("🚀 Top 5 Performers (Last 5-Min Session)")
    top_perf = df_perf.sort_values("Change (%)", ascending=False).head(5)
    st.table(top_perf.reset_index(drop=True))
else:
    st.info("No recent 5-min data for performance check.")

