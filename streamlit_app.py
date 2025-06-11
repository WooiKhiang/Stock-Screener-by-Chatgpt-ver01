import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# --- S&P 500 Ticker List ---
# Downloaded static list for speed, but you can also fetch with pandas.read_html() if you want it always fresh
sp500 = [
    'AAPL','MSFT','GOOGL','AMZN','NVDA','META','BRK-B','JPM','UNH','XOM','LLY','JNJ','V','PG','MA','AVGO','HD','MRK','COST','ADBE','ABBV',
    'CVX','CRM','PEP','TMO','WMT','KO','ACN','MCD','NFLX','WFC','AMD','LIN','ABT','BMY','DHR','NEE','TXN','NKE','UNP','INTC','LOW','RTX',
    'PM','PFE','SCHW','MS','DIS','SBUX','AMAT','GS','BA','HON','CAT','BLK','IBM','SPGI','ELV','ISRG','MDT','LMT','T','C','GE','SYK','BKNG',
    'NOW','ADP','MO','MDLZ','CVS','VRTX','SO','PLD','TGT','ZTS','GILD','DE','ADSK','DUK','AXP','CSCO','AMGN','MMC','ADI','APD','CB','USB',
    'REGN','CME','BDX','ETN','EW','CL','FCX','AON','PNC','ITW','BSX','SHW','FISV','PGR','ORLY','HCA','CARR','F','MCO','TRV','EMR','GM','D',
    # ... (truncated for brevity, but you can get the full S&P 500 ticker list from any online source)
]

# --- Sidebar: Filters ---
st.sidebar.header("Filter Settings")
min_price = st.sidebar.number_input("Min Price ($)", value=10.0)
max_price = st.sidebar.number_input("Max Price ($)", value=500.0)
min_volume = st.sidebar.number_input("Min Avg Vol (20hr)", value=100_000)
capital_per_trade = st.sidebar.number_input("Capital per Trade ($)", value=1000.0, step=100.0)

st.title("📊 S&P 500 1-Hour Trade Screener (3 Strategies Combo)")
st.caption(f"Last run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- Helper Functions: Indicators ---
def calc_indicators(df):
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['EMA50'] = df['Close'].ewm(span=50, min_periods=50).mean()
    df['EMA10'] = df['Close'].ewm(span=10, min_periods=10).mean()
    df['EMA20'] = df['Close'].ewm(span=20, min_periods=20).mean()
    # RSI(5)
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(window=5).mean()
    roll_down = down.rolling(window=5).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['RSI5'] = 100 - (100 / (1 + rs))
    # MACD (fast for intraday)
    ema12 = df['Close'].ewm(span=12, min_periods=12).mean()
    ema26 = df['Close'].ewm(span=26, min_periods=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, min_periods=9).mean()
    # Avg hourly Vol
    df['AvgVol20'] = df['Volume'].rolling(window=20).mean()
    return df

# --- Strategy Functions ---
def mean_reversion_signal(df):
    cond = (
        (df['Close'].iloc[-1] > df['SMA50'].iloc[-1]) &
        (df['RSI5'].iloc[-1] < 15)
    )
    if cond:
        return True, "Mean Reversion: Price above SMA50 and RSI(5)<15"
    return False, None

def ema50_breakout_signal(df):
    # Check close above EMA50, with previous close below (breakout), OR a recent dip below then reclaim
    close = df['Close'].iloc[-1]
    ema50 = df['EMA50'].iloc[-1]
    prev_close = df['Close'].iloc[-2]
    prev_ema50 = df['EMA50'].iloc[-2]
    dipped = (df['Close'].iloc[-6:-1] < df['EMA50'].iloc[-6:-1]).any()
    cond = (close > ema50) and ((prev_close < prev_ema50) or dipped)
    if cond:
        return True, "EMA50 Breakout: Price reclaimed EMA50 (with shakeout)"
    return False, None

def macd_ema_signal(df):
    macd = df['MACD'].iloc[-1]
    macd_signal = df['MACD_signal'].iloc[-1]
    macd_prev = df['MACD'].iloc[-2]
    macd_signal_prev = df['MACD_signal'].iloc[-2]
    ema10 = df['EMA10'].iloc[-1]
    ema20 = df['EMA20'].iloc[-1]
    cross = (macd_prev < macd_signal_prev) and (macd > macd_signal) and (macd < 0)
    cond = cross and (ema10 > ema20)
    if cond:
        return True, "MACD+EMA: MACD crosses up below 0 & EMA10>EMA20"
    return False, None

# --- Main Scan Loop ---
results = []
for ticker in sp500:
    try:
        df = yf.download(ticker, period="60d", interval="1h", progress=False)
        if df.empty or len(df) < 60:
            continue
        df = calc_indicators(df)
        last = df.iloc[-1]
        close_price = float(last['Close'])
        avgvol20 = float(last['AvgVol20'])
        if np.isnan(close_price) or not (min_price <= close_price <= max_price):
            continue
        if np.isnan(avgvol20) or avgvol20 < min_volume:
            continue

        strat, reason = None, ""
        rank = 0

        mr, mr_reason = mean_reversion_signal(df)
        ema, ema_reason = ema50_breakout_signal(df)
        macdema, macdema_reason = macd_ema_signal(df)

        if mr:
            strat, reason, rank = "Mean Reversion", mr_reason, 1
        elif ema:
            strat, reason, rank = "EMA50 Breakout", ema_reason, 2
        elif macdema:
            strat, reason, rank = "MACD+EMA", macdema_reason, 3

        if strat:
            entry = close_price
            if strat == "Mean Reversion":
                tp = entry * 1.02
                sl = entry * 0.99
            elif strat == "EMA50 Breakout":
                tp = None
                sl = entry * 0.98
            elif strat == "MACD+EMA":
                tp = None
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
                "Avg Vol (20hr)": int(avgvol20),
                "RSI(5)": round(last['RSI5'], 2),
                "EMA50": round(last['EMA50'], 2),
                "SMA50": round(last['SMA50'], 2)
            })
    except Exception as e:
        continue

df_results = pd.DataFrame(results)

# --- Dashboard Output ---
st.header("1-Hour Trade Recommendations (S&P 500)")
if not df_results.empty:
    df_results = df_results.sort_values(["Rank", "Strategy", "RSI(5)"])
    st.dataframe(df_results.reset_index(drop=True), use_container_width=True)
else:
    st.info("No stocks meet your filter/strategy criteria this hour.")

# --- Summary Section ---
if not df_results.empty:
    st.subheader("🔔 Summary & Highlight")
    picks = df_results.groupby("Strategy").first().sort_values("Rank")
    for idx, row in picks.iterrows():
        st.markdown(f"**[{row['Strategy']}] {row['Ticker']}** | Entry: ${row['Entry Price']} | Shares: {row['Shares']} | Reason: {row['Reason']}")
else:
    st.info("No strategy triggered this hour. Adjust filters or check back next hour.")

st.caption("Strategies ranked: 1) Mean Reversion (SMA50+RSI5), 2) EMA50 Breakout, 3) MACD+EMA. Exits: Mean Reversion = TP +2%/SL -1%. EMA50/Combo = trailing stop 1–1.5% after profit or exit on trend flip.")

