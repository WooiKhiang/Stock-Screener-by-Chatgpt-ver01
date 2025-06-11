import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

sp500 = [
    'AAPL','MSFT','GOOGL','AMZN','NVDA','META','BRK-B','JPM','UNH','XOM',
    'LLY','JNJ','V','PG','MA','AVGO','HD','MRK','COST','ADBE'
]

st.sidebar.header("Filter Settings")
min_price = st.sidebar.number_input("Min Price ($)", value=10.0, key="min_price")
max_price = st.sidebar.number_input("Max Price ($)", value=2000.0, key="max_price")
min_volume = st.sidebar.number_input("Min Avg Vol (40 bars)", value=50000, key="min_vol")
capital_per_trade = st.sidebar.number_input("Capital per Trade ($)", value=1000.0, step=100.0, key="capital_trade")

st.title("⚡ S&P 500 - 5-Minute Intraday Trade Screener (BULLETPROOF DEBUG MODE)")
st.caption(f"Last run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def safe_scalar(val):
    if isinstance(val, pd.Series) or isinstance(val, np.ndarray):
        if len(val) == 1:
            return float(val.item())
        else:
            return np.nan
    try:
        return float(val)
    except Exception:
        return np.nan

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
    c = safe_scalar(df['Close'].iloc[-1])
    sma = safe_scalar(df['SMA40'].iloc[-1])
    rsi = safe_scalar(df['RSI3'].iloc[-1])
    if np.isnan(c) or np.isnan(sma) or np.isnan(rsi):
        return False, None
    cond = (c > sma) and (rsi < 15)
    return cond, "Mean Reversion: Price > SMA40 & RSI(3)<15" if cond else (False, None)

def ema40_breakout_signal(df):
    c = safe_scalar(df['Close'].iloc[-1])
    ema = safe_scalar(df['EMA40'].iloc[-1])
    pc = safe_scalar(df['Close'].iloc[-2])
    pema = safe_scalar(df['EMA40'].iloc[-2])
    if np.isnan(c) or np.isnan(ema) or np.isnan(pc) or np.isnan(pema):
        return False, None
    left = df['Close'].iloc[-10:-1]
    right = df['EMA40'].iloc[-10:-1]
    left, right = left.align(right, join='inner', axis=0)  # <- fix axis=0!
    dipped = (left < right).any()
    cond = (c > ema) and ((pc < pema) or dipped)
    return cond, "EMA40 Breakout: Price reclaimed EMA40 (with shakeout)" if cond else (False, None)

def macd_ema_signal(df):
    macd = safe_scalar(df['MACD'].iloc[-1])
    macd_signal = safe_scalar(df['MACD_signal'].iloc[-1])
    macd_prev = safe_scalar(df['MACD'].iloc[-2])
    macd_signal_prev = safe_scalar(df['MACD_signal'].iloc[-2])
    ema8 = safe_scalar(df['EMA8'].iloc[-1])
    ema21 = safe_scalar(df['EMA21'].iloc[-1])
    if any(np.isnan(x) for x in [macd, macd_signal, macd_prev, macd_signal_prev, ema8, ema21]):
        return False, None
    cross = (macd_prev < macd_signal_prev) and (macd > macd_signal) and (macd < 0)
    cond = cross and (ema8 > ema21)
    return cond, "MACD+EMA: MACD cross up <0 & EMA8>EMA21" if cond else (False, None)

debug_rows = []
results = []

for ticker in sp500:
    debug_status = ""
    try:
        df = yf.download(ticker, period="5d", interval="5m", progress=False)
        # --- Normalize DataFrame, force single-column
        if isinstance(df, pd.DataFrame) and isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)
        if isinstance(df['Close'], pd.DataFrame):
            df['Close'] = df['Close'].iloc[:, 0]
        if isinstance(df['Volume'], pd.DataFrame):
            df['Volume'] = df['Volume'].iloc[:, 0]
        # ------------------------
        if df.empty or len(df) < 50:
            debug_status = f"{ticker}: Not enough data ({len(df)} bars)"
            debug_rows.append({'Ticker': ticker, 'Status': debug_status})
            continue
        df = calc_indicators(df)
        last = df.iloc[-1]
        close_price = safe_scalar(last['Close'])
        avgvol40 = safe_scalar(last['AvgVol40'])

        if any(np.isnan([close_price, avgvol40])):
            debug_status = f"{ticker}: NaNs in indicators or data"
            debug_rows.append({
                'Ticker': ticker,
                'Status': debug_status,
                'Close': close_price,
                'SMA40': safe_scalar(last['SMA40']),
                'RSI3': safe_scalar(last['RSI3']),
                'EMA40': safe_scalar(last['EMA40']),
                'MACD': safe_scalar(last['MACD']),
                'Volume': safe_scalar(last['Volume']),
                'AvgVol40': avgvol40,
            })
            continue
        if not (min_price <= close_price <= max_price):
            debug_status = f"{ticker}: Price {close_price} outside filter"
            debug_rows.append({'Ticker': ticker, 'Status': debug_status, 'Close': close_price})
            continue
        if avgvol40 < min_volume:
            debug_status = f"{ticker}: AvgVol40 {avgvol40} below filter"
            debug_rows.append({'Ticker': ticker, 'Status': debug_status, 'AvgVol40': avgvol40})
            continue

        mr, mr_reason = mean_reversion_signal(df)
        ema, ema_reason = ema40_breakout_signal(df)
        macdema, macdema_reason = macd_ema_signal(df)

        debug_rows.append({
            'Ticker': ticker,
            'Status': 'OK',
            'Close': close_price,
            'SMA40': safe_scalar(last['SMA40']),
            'RSI3': safe_scalar(last['RSI3']),
            'EMA40': safe_scalar(last['EMA40']),
            'MACD': safe_scalar(last['MACD']),
            'Volume': safe_scalar(last['Volume']),
            'AvgVol40': avgvol40,
            'MR?': mr,
            'EMA?': ema,
            'MACD?': macdema
        })

        strat, reason, rank = None, None, 0
        if mr:
            strat, reason, rank = "Mean Reversion", mr_reason, 1
        elif ema:
            strat, reason, rank = "EMA40 Breakout", ema_reason, 2
        elif macdema:
            strat, reason, rank = "MACD+EMA", macdema_reason, 3

        if strat:
            entry = close_price
            if strat == "Mean Reversion":
                tp = entry * 1.01
                sl = entry * 0.995
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
                "Volume": int(safe_scalar(last['Volume'])),
                "Avg Vol (40)": int(avgvol40),
                "RSI(3)": round(safe_scalar(last['RSI3']), 2),
                "EMA40": round(safe_scalar(last['EMA40']), 2),
                "SMA40": round(safe_scalar(last['SMA40']), 2)
            })
    except Exception as e:
        debug_status = f"{ticker}: Exception - {e}"
        debug_rows.append({'Ticker': ticker, 'Status': debug_status})

df_debug = pd.DataFrame(debug_rows)
if not df_debug.empty:
    st.subheader("DEBUG: Ticker Status & Indicator Values")
    st.dataframe(df_debug)

df_results = pd.DataFrame(results)

st.header("5-Minute Trade Recommendations (S&P 500 sample)")
if not df_results.empty:
    df_results = df_results.sort_values(["Rank", "Strategy", "RSI(3)"])
    st.dataframe(df_results.reset_index(drop=True), use_container_width=True)
else:
    st.info("No stocks meet your filter/strategy criteria right now.")

st.caption("This version is robust to all DataFrame/Series, index alignment, and axis errors. If you still get errors, show me the exact line and message!")
