import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# --- S&P 100 Tickers ---
sp100 = [
    'AAPL','ABBV','ABT','ACN','AIG','AMGN','AMT','AMZN','AVGO','AXP',
    'BA','BAC','BK','BKNG','BLK','BMY','BRK-B','C','CAT','CHTR','CL',
    'CMCSA','COF','COP','COST','CRM','CSCO','CVS','CVX','DHR','DIS',
    'DOW','DUK','EMR','EXC','F','FDX','FOX','FOXA','GD','GE','GILD',
    'GM','GOOG','GOOGL','GS','HD','HON','IBM','INTC','JNJ','JPM',
    'KHC','KMI','KO','LIN','LLY','LMT','LOW','MA','MCD','MDLZ','MDT',
    'MET','META','MMM','MO','MRK','MS','MSFT','NEE','NFLX','NKE',
    'NVDA','ORCL','PEP','PFE','PG','PM','PYPL','QCOM','RTX','SBUX',
    'SCHW','SO','SPG','T','TGT','TMO','TMUS','TSLA','TXN','UNH','UNP',
    'UPS','USB','V','VZ','WBA','WFC','WMT','XOM'
]

# --- Sidebar: Filters ---
st.sidebar.header("Filter Settings")
min_price = st.sidebar.number_input("Min Price ($)", value=10.0)
max_price = st.sidebar.number_input("Max Price ($)", value=500.0)
min_volume = st.sidebar.number_input("Min Avg Volume (20d)", value=1_000_000)
capital_per_trade = st.sidebar.number_input("Capital per Trade ($)", value=1000.0, step=100.0)

st.title("📊 S&P 100 Trade Screener (3 Strategies Combo)")
st.caption(f"Last run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- Helper Functions: Indicators ---
def calc_indicators(df):
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['EMA200'] = df['Close'].ewm(span=200, min_periods=200).mean()
    df['EMA10'] = df['Close'].ewm(span=10, min_periods=10).mean()
    df['EMA20'] = df['Close'].ewm(span=20, min_periods=20).mean()
    # RSI(2)
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(window=2).mean()
    roll_down = down.rolling(window=2).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['RSI2'] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = df['Close'].ewm(span=12, min_periods=12).mean()
    ema26 = df['Close'].ewm(span=26, min_periods=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, min_periods=9).mean()
    # Avg Vol
    df['AvgVol20'] = df['Volume'].rolling(window=20).mean()
    return df

# --- Strategy Functions ---
def mean_reversion_signal(df):
    cond = (
        (df['Close'].iloc[-1] > df['SMA200'].iloc[-1]) &
        (df['RSI2'].iloc[-1] < 10)
    )
    if cond:
        return True, "Mean Reversion: Price above SMA200 and RSI(2)<10"
    return False, None

def ema200_breakout_signal(df):
    # Check close above EMA200, with previous close below (breakout), OR a recent dip below then reclaim
    close = df['Close'].iloc[-1]
    ema200 = df['EMA200'].iloc[-1]
    prev_close = df['Close'].iloc[-2]
    prev_ema200 = df['EMA200'].iloc[-2]
    # "Shakeout" - dipped below EMA200 in last 5 days, now back above
    dipped = (df['Close'].iloc[-6:-1] < df['EMA200'].iloc[-6:-1]).any()
    cond = (close > ema200) and ((prev_close < prev_ema200) or dipped)
    if cond:
        return True, "EMA200 Breakout: Price reclaimed EMA200 (with shakeout)"
    return False, None

def macd_ema_signal(df):
    macd = df['MACD'].iloc[-1]
    macd_signal = df['MACD_signal'].iloc[-1]
    macd_prev = df['MACD'].iloc[-2]
    macd_signal_prev = df['MACD_signal'].iloc[-2]
    ema10 = df['EMA10'].iloc[-1]
    ema20 = df['EMA20'].iloc[-1]
    # MACD crosses up from below, still < 0
    cross = (macd_prev < macd_signal_prev) and (macd > macd_signal) and (macd < 0)
    cond = cross and (ema10 > ema20)
    if cond:
        return True, "MACD+EMA: MACD crosses up below 0 & EMA10>EMA20"
    return False, None

# --- Main Scan Loop ---
results = []
for ticker in sp100:
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df.empty or len(df) < 210:
            continue
        df = calc_indicators(df)
        last = df.iloc[-1]
        close_price = float(last['Close'])
        avgvol20 = float(last['AvgVol20'])
        # Check for NaN/invalid values
        if np.isnan(close_price) or not (min_price <= close_price <= max_price):
            continue
        if np.isnan(avgvol20) or avgvol20 < min_volume:
            continue

        strat, reason = None, ""
        rank = 0

        mr, mr_reason = mean_reversion_signal(df)
        ema, ema_reason = ema200_breakout_signal(df)
        macdema, macdema_reason = macd_ema_signal(df)

        if mr:
            strat, reason, rank = "Mean Reversion", mr_reason, 1
        elif ema:
            strat, reason, rank = "EMA200 Breakout", ema_reason, 2
        elif macdema:
            strat, reason, rank = "MACD+EMA", macdema_reason, 3

        if strat:
            entry = close_price
            if strat == "Mean Reversion":
                tp = entry * 1.02
                sl = entry * 0.99
            elif strat == "EMA200 Breakout":
                tp = None  # Trailing stop logic can be handled on execution
                sl = entry * 0.98
            elif strat == "MACD+EMA":
                tp = None  # Trailing stop logic can be handled on execution
                sl = entry * 0.99
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
                "Avg Vol (20d)": int(avgvol20),
                "RSI(2)": round(last['RSI2'], 2),
                "EMA200": round(last['EMA200'], 2),
                "SMA200": round(last['SMA200'], 2)
            })
    except Exception as e:
        # Safely skip ticker on error
        continue

df_results = pd.DataFrame(results)

# --- Dashboard Output ---
st.header("Today's Trade Recommendations")
if not df_results.empty:
    df_results = df_results.sort_values(["Rank", "Strategy", "RSI(2)"])
    st.dataframe(df_results.reset_index(drop=True), use_container_width=True)
else:
    st.info("No stocks meet your filter/strategy criteria today.")

# --- Summary Section ---
if not df_results.empty:
    st.subheader("🔔 Summary & Highlight")
    picks = df_results.groupby("Strategy").first().sort_values("Rank")
    for idx, row in picks.iterrows():
        st.markdown(f"**[{row['Strategy']}] {row['Ticker']}** | Entry: ${row['Entry Price']} | Shares: {row['Shares']} | Reason: {row['Reason']}")
else:
    st.info("No strategy triggered today. Adjust filters or check back tomorrow.")

st.caption("Strategies ranked: 1) Mean Reversion, 2) EMA200 Breakout, 3) MACD+EMA. Exits: Mean Reversion = TP +2%/SL -1%. EMA200/Combo = trailing stop 1–1.5% after profit or exit on trend flip.")

