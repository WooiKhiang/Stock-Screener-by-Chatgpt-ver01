import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="AI US Stock Screener", layout="wide")
st.title("🔍 AI-Powered US Stock Screener (S&P 100 Top 3 Picks)")

# -------- S&P 100 List --------
sp100 = [
    'AAPL','ABBV','ABT','ACN','ADBE','AIG','AMGN','AMT','AMZN','AVGO',
    'AXP','BA','BAC','BK','BKNG','BLK','BMY','BRK-B','C','CAT',
    'CHTR','CL','CMCSA','COF','COP','COST','CRM','CSCO','CVS','CVX',
    'DHR','DIS','DOW','DUK','EMR','EXC','F','FDX','FOX','FOXA',
    'GD','GE','GILD','GM','GOOG','GOOGL','GS','HD','HON','IBM',
    'INTC','JNJ','JPM','KHC','KMI','KO','LIN','LLY','LMT','LOW',
    'MA','MCD','MDLZ','MDT','MET','META','MMM','MO','MRK','MS',
    'MSFT','NEE','NFLX','NKE','NVDA','ORCL','PEP','PFE','PG','PM',
    'PYPL','QCOM','RTX','SBUX','SCHW','SO','SPG','T','TGT','TMO',
    'TMUS','TSLA','TXN','UNH','UNP','UPS','USB','V','VZ','WBA',
    'WFC','WMT','XOM'
]

# --------- Sidebar Controls ---------
st.sidebar.header("Filter Settings")
min_price = st.sidebar.number_input("Min Price ($)", value=10.0)
max_price = st.sidebar.number_input("Max Price ($)", value=2000.0)
min_vol = st.sidebar.number_input("Min Avg Vol (40 bars)", value=100000)
capital = st.sidebar.number_input("Capital per Trade ($)", value=1000.0, step=100.0)
show_all = st.sidebar.checkbox("Show All Signals (not just Top 3)", value=False)

# --------- Utility ---------
def normalize_cols(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            '_'.join([str(x) for x in col if x not in [None, '', 'nan']]).lower()
            for col in df.columns
        ]
    else:
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    return df

# --------- Strategies ---------

def mean_reversion_signal(df):
    # Mean Reversion (close > sma40 and rsi3 < 15): historically good for quick snap-backs
    if 'sma40' not in df.columns or 'rsi3' not in df.columns:
        return False, ""
    c, sma, rsi = df['close'].iloc[-1], df['sma40'].iloc[-1], df['rsi3'].iloc[-1]
    cond = (c > sma) and (rsi < 15)
    score = 80 if cond else 0
    return cond, f"MeanRev: close>{sma:.2f}, rsi3={rsi:.2f}", score

def ema40_breakout_signal(df):
    # EMA40 Breakout (close crosses above ema40 after being below): good for momentum
    if 'ema40' not in df.columns:
        return False, ""
    c, ema = df['close'].iloc[-1], df['ema40'].iloc[-1]
    prev_c, prev_ema = df['close'].iloc[-2], df['ema40'].iloc[-2]
    # Ensure it was below and now above (cross)
    cond = (prev_c < prev_ema) and (c > ema)
    score = 85 if cond else 0
    return cond, f"EMA40Breakout: prev_close<{prev_ema:.2f}, now_close>{ema:.2f}", score

def macd_ema_combo(df):
    # MACD cross up below 0 + EMA8>EMA21: high win rate in strong markets
    if 'macd' not in df.columns or 'macd_signal' not in df.columns or 'ema8' not in df.columns or 'ema21' not in df.columns:
        return False, ""
    macd, macd_sig = df['macd'].iloc[-1], df['macd_signal'].iloc[-1]
    macd_prev, macd_sig_prev = df['macd'].iloc[-2], df['macd_signal'].iloc[-2]
    ema8, ema21 = df['ema8'].iloc[-1], df['ema21'].iloc[-1]
    cross = (macd_prev < macd_sig_prev) and (macd > macd_sig) and (macd < 0)
    cond = cross and (ema8 > ema21)
    score = 78 if cond else 0
    return cond, f"MACD+EMA: macd_cross_up, ema8>{ema21:.2f}", score

# --------- Main Loop ---------
results = []

for ticker in sp100:
    df = yf.download(ticker, period='5d', interval='5m', progress=False, threads=False)
    if df.empty or len(df) < 41:
        continue
    df = normalize_cols(df)
    # Handle suffixed columns
    for base in ['close', 'open', 'high', 'low', 'volume']:
        possible_col = f"{base}_{ticker.lower()}"
        if possible_col in df.columns:
            df[base] = df[possible_col]
    # Indicators
    df['sma40'] = df['close'].rolling(40).mean()
    df['ema40'] = df['close'].ewm(span=40, min_periods=40).mean()
    df['ema8'] = df['close'].ewm(span=8, min_periods=8).mean()
    df['ema21'] = df['close'].ewm(span=21, min_periods=21).mean()
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(3).mean()
    roll_down = down.rolling(3).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['rsi3'] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = df['close'].ewm(span=12, min_periods=12).mean()
    ema26 = df['close'].ewm(span=26, min_periods=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, min_periods=9).mean()
    df['avgvol40'] = df['volume'].rolling(40, min_periods=1).mean()
    last = df.iloc[-1]
    close_price = last['close']
    avgvol40 = last['avgvol40']

    # Filters
    if not (min_price <= close_price <= max_price):
        continue
    if avgvol40 < min_vol:
        continue

    # Run all strategies, collect best signal
    picks = []
    for strat_func, strat_name in [
        (mean_reversion_signal, "Mean Reversion"),
        (ema40_breakout_signal, "EMA40 Breakout"),
        (macd_ema_combo, "MACD+EMA Combo")
    ]:
        hit, reason, score = strat_func(df)
        if hit:
            picks.append((score, strat_name, reason))
    if picks:
        picks.sort(reverse=True)
        best_score, best_strat, best_reason = picks[0]
        shares = int(capital // close_price)
        invested = shares * close_price
        results.append({
            "Ticker": ticker,
            "Strategy": best_strat,
            "AI Score": round(best_score, 2),
            "Entry Price": round(close_price, 2),
            "Capital Used": round(invested, 2),
            "Shares": shares,
            "Reason": best_reason,
            "AvgVol40": int(avgvol40)
        })
    elif show_all:
        # Show even if no signal (for debugging)
        results.append({
            "Ticker": ticker,
            "Strategy": "-",
            "AI Score": 0,
            "Entry Price": round(close_price, 2),
            "Capital Used": 0,
            "Shares": 0,
            "Reason": "-",
            "AvgVol40": int(avgvol40)
        })

# --------- Top 3 AI Picks ---------
if results:
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("AI Score", ascending=False).head(3).reset_index(drop=True)
    df_results.insert(0, "Rank", df_results.index+1)
    st.subheader("🚦 Top 3 AI-Powered Intraday Stock Picks (S&P 100, 5-min)")
    st.dataframe(df_results, use_container_width=True)
else:
    st.warning("No stocks met any high-conviction strategy signal right now (try adjusting filters or check when US market is open).")

# --------- Show All (optional) ---------
if show_all and results:
    st.subheader("All Signals (inc. non-picks)")
    st.dataframe(pd.DataFrame(results), use_container_width=True)

st.markdown("""
---
**Strategies used:**
- **Mean Reversion:** Close above SMA40 and RSI(3) < 15 (snapback setups)
- **EMA40 Breakout:** Fresh close crossing above EMA40 (momentum triggers)
- **MACD+EMA Combo:** MACD crosses up below zero, and EMA8 > EMA21 (strong trend momentum)
---
""")
