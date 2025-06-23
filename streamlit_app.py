import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# ---- TICKER LIST ----
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

# ---- SIDEBAR ----
st.sidebar.header("Trade Settings")
min_price = st.sidebar.number_input("Min Price ($)", value=10.0)
max_price = st.sidebar.number_input("Max Price ($)", value=2000.0)
min_volume = st.sidebar.number_input("Min Avg Vol (40 bars)", value=100_000)
capital_per_trade = st.sidebar.number_input("Capital per Trade ($)", value=1000.0, step=100.0)

st.title("🔍 Debug: US Stocks Screener - Always-On Signal Test")

# --- Utility Functions ---
def format_number(num, decimals=2):
    try:
        if num is None or num == "" or np.isnan(num): return "-"
        if isinstance(num, int) or decimals == 0:
            return f"{int(num):,}"
        return f"{num:,.{decimals}f}"
    except Exception:
        return str(num)

def safe_scalar(val):
    if isinstance(val, pd.Series) or isinstance(val, np.ndarray):
        if len(val) == 1:
            return float(val.item())
        elif len(val) > 0:
            return float(val[-1])
        else:
            return np.nan
    try:
        return float(val)
    except Exception:
        return np.nan

def normalize_df_cols(df):
    # Lowercase and simplify columns, especially with multi-index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[-1]).lower() if isinstance(c, tuple) else str(c).lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]
    return df

def calc_indicators(df):
    df['sma40'] = df['close'].rolling(window=40).mean()
    df['ema40'] = df['close'].ewm(span=40, min_periods=40).mean()
    df['ema8'] = df['close'].ewm(span=8, min_periods=8).mean()
    df['ema21'] = df['close'].ewm(span=21, min_periods=21).mean()
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(window=3).mean()
    roll_down = down.rolling(window=3).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['rsi3'] = 100 - (100 / (1 + rs))
    ema12 = df['close'].ewm(span=12, min_periods=12).mean()
    ema26 = df['close'].ewm(span=26, min_periods=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, min_periods=9).mean()
    df['avgvol40'] = df['volume'].replace(0, np.nan).rolling(window=40, min_periods=1).mean().fillna(0)
    return df

# --- The always-on debug strategy ---
def always_on_debug_signal(df):
    try:
        if len(df) < 3:
            return False, "Not enough data", 0
        sma3 = df['close'].tail(3).mean()
        c = df['close'].iloc[-1]
        cond = c > sma3
        score = 50 if cond else 0
        return bool(cond), "DEBUG: Close > SMA(3)", float(score)
    except Exception as e:
        return False, f"DEBUG ERROR: {e}", 0

# --- Original signals (can comment out for isolated debug) ---
def mean_reversion_signal(df):
    c = float(safe_scalar(df['close'].iloc[-1]))
    sma = float(safe_scalar(df['sma40'].iloc[-1]))
    rsi = float(safe_scalar(df['rsi3'].iloc[-1]))
    cond = (c > sma) and (rsi < 15)
    score = 75 + max(0, 15 - rsi) if cond else 0
    return bool(cond), "Mean Reversion", float(f"{score:.2f}")

def ema40_breakout_signal(df):
    c = float(safe_scalar(df['close'].iloc[-1]))
    ema = float(safe_scalar(df['ema40'].iloc[-1]))
    pc = float(safe_scalar(df['close'].iloc[-2]))
    pema = float(safe_scalar(df['ema40'].iloc[-2]))
    dipped = np.any(df['close'].iloc[-10:-1] < df['ema40'].iloc[-10:-1])
    cond = (c > ema) and ((pc < pema) or dipped)
    score = 70 + min(20, c - ema) if cond else 0
    return bool(cond), "EMA40 Breakout", float(f"{score:.2f}")

def macd_ema_signal(df):
    macd = float(safe_scalar(df['macd'].iloc[-1]))
    macd_signal = float(safe_scalar(df['macd_signal'].iloc[-1]))
    macd_prev = float(safe_scalar(df['macd'].iloc[-2]))
    macd_signal_prev = float(safe_scalar(df['macd_signal'].iloc[-2]))
    ema8 = float(safe_scalar(df['ema8'].iloc[-1]))
    ema21 = float(safe_scalar(df['ema21'].iloc[-1]))
    cross = (macd_prev < macd_signal_prev) and (macd > macd_signal) and (macd < 0)
    cond = cross and (ema8 > ema21)
    score = 65 + int(abs(macd)*5) if cond else 0
    return bool(cond), "MACD+EMA", float(f"{score:.2f}")

# ---- SCAN ALL TICKERS ----
scan_status = []
results_debug = []

for ticker in sp100:
    try:
        df = yf.download(ticker, period="5d", interval="5m", progress=False, threads=False)
        df = normalize_df_cols(df)
        if df.empty or len(df) < 50:
            scan_status.append({"Ticker": ticker, "Status": "Empty/Short", "Last Close": "-", "AvgVol40": "-"})
            continue
        if 'close' not in df.columns or 'volume' not in df.columns:
            scan_status.append({"Ticker": ticker, "Status": "Missing Columns", "Last Close": "-", "AvgVol40": "-"})
            continue
        df = calc_indicators(df)
        last = df.iloc[-1]
        close_price = float(safe_scalar(last['close']))
        avgvol40 = float(safe_scalar(last['avgvol40']))
        if not (min_price <= close_price <= max_price):
            scan_status.append({"Ticker": ticker, "Status": "Filtered: Price", "Last Close": close_price, "AvgVol40": avgvol40})
            continue
        if avgvol40 < min_volume:
            scan_status.append({"Ticker": ticker, "Status": "Filtered: Volume", "Last Close": close_price, "AvgVol40": avgvol40})
            continue
        picks = []
        for func, strat_name in [
            (always_on_debug_signal, "DEBUG: Close>SMA3"),
            (mean_reversion_signal, "Mean Reversion"),
            (ema40_breakout_signal, "EMA40 Breakout"),
            (macd_ema_signal, "MACD+EMA")
        ]:
            sig, reason, score = func(df)
            if sig:
                picks.append((strat_name, reason, score))
        if not picks:
            scan_status.append({"Ticker": ticker, "Status": "No Signal", "Last Close": close_price, "AvgVol40": avgvol40})
        else:
            strat_name, reason, score = picks[0]
            scan_status.append({"Ticker": ticker, "Status": f"SIGNAL: {strat_name}", "Last Close": close_price, "AvgVol40": avgvol40})
            results_debug.append({
                "Ticker": ticker,
                "Strategy": strat_name,
                "Reason": reason,
                "Score": score,
                "Last Close": format_number(close_price,2),
                "AvgVol40": format_number(avgvol40,0)
            })
    except Exception as e:
        scan_status.append({"Ticker": ticker, "Status": f"Error: {e}", "Last Close": "-", "AvgVol40": "-"})

# ---- OUTPUT ----
st.subheader("Full Scan Debug Table (see status for EVERY ticker)")
df_status = pd.DataFrame(scan_status)
st.dataframe(df_status)

st.subheader("Debug Strategy Signals Fired")
if results_debug:
    st.dataframe(pd.DataFrame(results_debug))
else:
    st.info("No tickers triggered even the debug strategy. If so, your code/data source is broken, not your strategies.")
