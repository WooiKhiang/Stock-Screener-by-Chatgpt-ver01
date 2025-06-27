import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

def normalize_cols(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            '_'.join([str(x) for x in col if x not in [None, '', 'nan']]).lower()
            for col in df.columns
        ]
    else:
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    return df

def mean_reversion_signal(df):
    if 'sma40' not in df.columns or 'rsi3' not in df.columns:
        return False, ""
    c, sma, rsi = df['close'].iloc[-1], df['sma40'].iloc[-1], df['rsi3'].iloc[-1]
    cond = (c > sma) and (rsi < 15)
    return cond, f"MeanRev: close>{sma:.2f} rsi3={rsi:.2f}"

def ema40_breakout_signal(df):
    if 'ema40' not in df.columns:
        return False, ""
    c, ema = df['close'].iloc[-1], df['ema40'].iloc[-1]
    cond = c > ema
    return cond, f"EMA40: close>{ema:.2f}"

st.title("AI-Powered US Stock Screener")

tickers = st.multiselect(
    "Select tickers to scan", 
    options=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA', 'SPY', 'META'],
    default=['AAPL', 'MSFT', 'GOOGL']
)
results = []

for ticker in tickers:
    df = yf.download(ticker, period='5d', interval='5m', progress=False, threads=False)
    if df.empty or len(df) < 40:
        continue
    df = normalize_cols(df)
    # Handle suffixed columns
    for base in ['close', 'open', 'high', 'low', 'volume']:
        possible_col = f"{base}_{ticker.lower()}"
        if possible_col in df.columns:
            df[base] = df[possible_col]
    df['sma40'] = df['close'].rolling(40).mean()
    df['ema40'] = df['close'].ewm(span=40, min_periods=40).mean()
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(3).mean()
    roll_down = down.rolling(3).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['rsi3'] = 100 - (100 / (1 + rs))

    for strat_func, strat_name in [
        (mean_reversion_signal, "Mean Reversion"),
        (ema40_breakout_signal, "EMA40 Breakout")
    ]:
        hit, note = strat_func(df)
        if hit:
            results.append({
                'Ticker': ticker,
                'Strategy': strat_name,
                'Note': note,
                'Close': round(df['close'].iloc[-1],2)
            })

st.dataframe(pd.DataFrame(results))
