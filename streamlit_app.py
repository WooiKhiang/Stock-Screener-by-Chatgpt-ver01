import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, time as dt_time
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import time

# ---- Ticker List ----
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

nasdaq100 = [
    'AAPL','MSFT','AMZN','NVDA','GOOGL','GOOG','META','AVGO','ADBE','COST','TSLA','PEP',
    'AMD','NFLX','CSCO','INTC','CMCSA','TXN','QCOM','HON','INTU','SBUX','AMGN','GILD',
    'BKNG','ISRG','MDLZ','REGN','ADI','VRTX','ABNB','LRCX','MU','SNPS','ASML','CRWD',
    'MRNA','PDD','PYPL','ZM','KLAC','WDAY','MAR','PANW','DXCM','MNST','CDNS','ROST','KDP',
    'AEP','CSX','PCAR','CHTR','MELI','IDXX','EXC','CTAS','XEL','ORLY','PAYX','FAST',
    'TEAM','SGEN','ANSS','ALGN','VRSK','CTSH','SIRI'
]

all_tickers = sorted(set(sp100 + nasdaq100))

# ---- Helper Functions ----
def formatn(num, d=2, pct=False):
    try:
        if pct:
            return f"{num:.2f}%"
        if num is None or num == "" or np.isnan(num): return "-"
        if isinstance(num, int) or d == 0:
            return f"{int(num):,}"
        return f"{num:,.{d}f}"
    except Exception:
        return str(num)

def norm(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x).lower() for x in c if x]) for c in df.columns]
    else:
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    return df

def ensure_core_cols(df):
    req = ["close","open","high","low","volume"]
    col_map = {c: c for c in df.columns}
    for col in df.columns:
        cstr = str(col).lower()
        for r in req:
            if r in cstr and r not in col_map.values():
                col_map[col] = r
    df = df.rename(columns=col_map)
    missing = [x for x in req if x not in df.columns]
    if missing:
        raise Exception(f"Missing columns: {missing}")
    return df

def calc_indicators(df):
    df['ema10'] = df['close'].ewm(span=10, min_periods=10).mean()
    df['ema20'] = df['close'].ewm(span=20, min_periods=20).mean()
    # RSI
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(window=14).mean()
    roll_down = down.rolling(window=14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['rsi14'] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = df['close'].ewm(span=12, min_periods=12).mean()
    ema26 = df['close'].ewm(span=26, min_periods=26).mean()
    df['macd'] = ema12 - ema26
    df['macdsignal'] = df['macd'].ewm(span=9, min_periods=9).mean()
    df['hist'] = df['macd'] - df['macdsignal']
    return df

def local_time_str():
    return datetime.now(pytz.timezone("Asia/Singapore")).strftime('%Y-%m-%d %H:%M:%S')

def get_market_sentiment():
    try:
        spy = yf.download('SPY', period='1d', interval='5m', progress=False, threads=False)
        if spy.empty: return 0, "Sentiment: Unknown"
        spy = norm(spy)
        spy = ensure_core_cols(spy)
        open_, last = spy['open'].iloc[0], spy['close'].iloc[-1]
        pct = (last - open_) / open_ * 100
        if pct > 0.5: return pct, "🟢 Bullish"
        elif pct < -0.5: return pct, "🔴 Bearish"
        else: return pct, "🟡 Sideways"
    except Exception:
        return 0, "Sentiment: Unknown"

# ---- Sidebar (RSI Filter) ----
st.sidebar.header("KIV RSI Filter (applies to KIV signals only)")
min_rsi = st.sidebar.number_input("Min RSI", value=35, min_value=0, max_value=100)
max_rsi = st.sidebar.number_input("Max RSI", value=70, min_value=0, max_value=100)

# ---- Market Sentiment ----
sentiment_pct, sentiment_text = get_market_sentiment()
st.title("AI S&P100 + NASDAQ100: 1h MACD KIV Screener")
st.caption(f"Last run: {local_time_str()}")
st.caption(f"Market Sentiment: {sentiment_text} ({sentiment_pct:.2f}%)")

# ---- KIV List Finder ----
kiv_results = []
reference_results = []

for ticker in all_tickers:
    try:
        df = yf.download(ticker, period='60d', interval='1h', progress=False, threads=False)
        if df.empty or len(df) < 20: continue
        df = norm(df)
        df = ensure_core_cols(df)
        df = calc_indicators(df)

        macd = df['macd']
        macdsig = df['macdsignal']
        hist = df['hist']
        rsi = df['rsi14']
        close = df['close']
        ema10 = df['ema10']
        ema20 = df['ema20']
        time_idx = df.index

        # Check for MACD cross up zero, MACD signal below zero, reference table
        cross_idx = np.where((macd > 0) & (macd.shift(1) < 0) & (macdsig < 0))[0]
        for idx in cross_idx:
            row = {
                "Ticker": ticker,
                "Datetime": time_idx[idx].strftime('%Y-%m-%d %H:%M'),
                "MACD": formatn(macd.iloc[idx], 3),
                "MACDsig": formatn(macdsig.iloc[idx], 3),
                "RSI": formatn(rsi.iloc[idx], 2),
                "Close": formatn(close.iloc[idx]),
            }
            reference_results.append(row)

        # Now, apply KIV rules (only for most recent bar)
        idx = len(df)-1
        # 1. MACD cross up zero (now > 0, prev < 0), macdsig < 0 now
        cond_macd = (macd.iloc[idx] > 0 and macd.iloc[idx-1] < 0 and macdsig.iloc[idx] < 0)
        # 2. MACD 10 bars ago < MACD 5 bars ago < current MACD (rising)
        cond_macd_rising = (macd.iloc[idx-10] < macd.iloc[idx-5] < macd.iloc[idx])
        # 3. RSI in range
        rsi_now = rsi.iloc[idx]
        cond_rsi = (min_rsi <= rsi_now <= max_rsi)
        # 4. EMA10 > EMA20 and EMA10 rising
        cond_ema = (ema10.iloc[idx] > ema20.iloc[idx]) and (ema10.iloc[idx] > ema10.iloc[idx-1])
        # 5. Histogram momentum: now > 5 bars ago
        cond_hist = (hist.iloc[idx-5] <= hist.iloc[idx])
        # If all KIV conditions met:
        if cond_macd and cond_macd_rising and cond_rsi and cond_ema and cond_hist:
            # Entry/TP logic
            rsi_zone = "Low" if rsi_now < 60 else "High"
            entry_price = close.iloc[idx]
            tp_pct = 0.05
            cl_pct = 0.025
            if rsi_now < 60:
                # Immediate entry
                action = "Enter now (RSI 35–60)"
                buy_price = entry_price
            else:
                # Wait for 1% dip
                action = "Wait for 1% dip (RSI 60–70)"
                buy_price = entry_price * 0.99
            take_profit = buy_price * (1+tp_pct)
            cut_loss = buy_price * (1-cl_pct)
            kiv_results.append({
                "Ticker": ticker,
                "Datetime": time_idx[idx].strftime('%Y-%m-%d %H:%M'),
                "RSI": formatn(rsi_now, 2),
                "Entry Action": action,
                "Entry Price": formatn(buy_price, 2),
                "Take Profit": formatn(take_profit, 2),
                "Cut Loss": formatn(cut_loss, 2),
                "MACD": formatn(macd.iloc[idx], 3),
                "MACDsig": formatn(macdsig.iloc[idx], 3),
                "EMA10": formatn(ema10.iloc[idx], 2),
                "EMA20": formatn(ema20.iloc[idx], 2),
                "Histogram": formatn(hist.iloc[idx], 3),
            })

    except Exception as e:
        continue

# ---- OUTPUT ----
if kiv_results:
    st.subheader("🟢 1h KIV Buy List (Filtered, Ready to Watch/Enter)")
    df_kiv = pd.DataFrame(kiv_results)
    st.dataframe(df_kiv, use_container_width=True)
    st.markdown("**Entry/TP logic:**\n- RSI 35–60: Buy at close\n- RSI 60–70: Wait for 1% drop\n- TP: 5%, CL: 2.5%, Time stop: 5 bars")
else:
    st.info("No current 1h KIV signals found.")

if reference_results:
    st.subheader("📋 MACD Zero Cross Reference (Recent 1h Events, All Tickers)")
    df_ref = pd.DataFrame(reference_results)
    st.dataframe(df_ref, use_container_width=True)
    st.markdown("_Shows all events where MACD crosses zero upwards and signal < 0 (regardless of other filters, for research/historical context)._")
else:
    st.info("No MACD zero-cross reference events in latest bars.")

st.caption("© AI S&P100+NASDAQ100 1h KIV Screener. Rules & TP/CL as discussed. All times GMT+8.")
