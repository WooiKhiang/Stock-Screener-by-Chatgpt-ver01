import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Hybrid AI US Stock Screener", layout="wide")
st.title("🚦 Hybrid AI Stock Screener – Intraday/Swing + EMA200 Breakout (S&P 100)")

sp100 = [ # S&P 100 tickers...
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

st.sidebar.header("Trade Settings")
min_price = st.sidebar.number_input("Min Price ($)", value=10.0)
max_price = st.sidebar.number_input("Max Price ($)", value=2000.0)
min_vol = st.sidebar.number_input("Min Avg Vol (40)", value=100000)
capital = st.sidebar.number_input("Capital per Trade ($)", value=1000.0, step=100.0)

def norm(df):
    # Flatten MultiIndex and force all columns to lower case, no spaces
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [('_'.join([str(x) for x in col if x not in [None,'','nan']])).lower().replace(" ","_") for col in df.columns]
    else:
        df.columns = [str(c).lower().replace(" ","_") for c in df.columns]
    return df

def ensure_core_cols(df):
    # Rename columns to expected names
    col_map = {}
    for c in df.columns:
        if c == "close" or "close" in c: col_map[c] = "close"
        if c == "open" or "open" in c: col_map[c] = "open"
        if c == "high" or "high" in c: col_map[c] = "high"
        if c == "low" or "low" in c: col_map[c] = "low"
        if c == "volume" or "vol" in c: col_map[c] = "volume"
    df = df.rename(columns=col_map)
    # Now force all required columns to exist
    need = ["close", "open", "high", "low", "volume"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise Exception(f"Missing columns: {missing}")
    return df

def formatn(x, d=2):
    try: return f"{x:,.{d}f}"
    except: return x

def ema200_breakout_daily(df):
    # Use daily bars
    if len(df) < 210 or 'ema200' not in df.columns: return False, "", 0
    prev, curr = df.iloc[-2], df.iloc[-1]
    breakout = (prev['close'] < prev['ema200']) and (curr['close'] > curr['ema200'])
    vol_ok = curr['volume'] > 1.5 * df['volume'][-20:-1].mean()
    rsi_ok = curr['rsi14'] < 70
    cond = breakout and vol_ok and rsi_ok
    score = 99 if cond else 0
    return cond, "Daily EMA200 Breakout + volume/rsi", score

def hybrid_signal(df5, df1h):
    # 5m EMA40 breakout + confirm price > EMA200 on 1hr
    if len(df5) < 50 or len(df1h) < 50: return False, "", 0
    # 5m logic
    c, ema40 = df5['close'].iloc[-1], df5['ema40'].iloc[-1]
    prev_c, prev_ema = df5['close'].iloc[-2], df5['ema40'].iloc[-2]
    breakout = (prev_c < prev_ema) and (c > ema40)
    vol_spike = df5['volume'].iloc[-1] > 1.3 * df5['volume'][-40:-1].mean()
    # 1hr confirm
    curr_1h = df1h.iloc[-1]
    above_ema200 = curr_1h['close'] > curr_1h['ema200']
    cond = breakout and vol_spike and above_ema200
    score = 90 if cond else 0
    return cond, "5min EMA40 Breakout + 1hr EMA200 confirm", score

def calc_indicators(df):
    df['ema200'] = df['close'].ewm(span=200, min_periods=200).mean()
    df['ema40'] = df['close'].ewm(span=40, min_periods=40).mean()
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-8)
    df['rsi14'] = 100 - (100 / (1 + rs))
    return df

results = []
for ticker in sp100:
    try:
        # --- Daily chart for EMA200 breakout
        dfd = yf.download(ticker, period='1y', interval='1d', progress=False, threads=False)
        if dfd.empty or len(dfd) < 210: continue
        dfd = norm(dfd)
        dfd = ensure_core_cols(dfd)
        dfd = calc_indicators(dfd)
        hit, reason, score = ema200_breakout_daily(dfd)
        if hit:
            curr = dfd.iloc[-1]
            price = curr['close']
            if not (min_price <= price <= max_price): continue
            if curr['volume'] < min_vol: continue
            shares = int(capital // price)
            results.append({
                "Ticker": ticker,
                "Strategy": "EMA200 Breakout (Daily)",
                "Score": score,
                "Entry": formatn(price),
                "Shares": shares,
                "Reason": reason,
                "Type": "Swing"
            })
            continue  # if breakout fires, don't double-count for hybrid

        # --- Hybrid: 5-min entry + 1hr confirm
        df5 = yf.download(ticker, period='3d', interval='5m', progress=False, threads=False)
        df1h = yf.download(ticker, period='30d', interval='60m', progress=False, threads=False)
        if df5.empty or df1h.empty: continue
        df5, df1h = norm(df5), norm(df1h)
        df5, df1h = ensure_core_cols(df5), ensure_core_cols(df1h)
        df5 = calc_indicators(df5)
        df1h = calc_indicators(df1h)
        hit, reason, score = hybrid_signal(df5, df1h)
        if hit:
            price = df5['close'].iloc[-1]
            if not (min_price <= price <= max_price): continue
            if df5['volume'].iloc[-1] < min_vol: continue
            shares = int(capital // price)
            results.append({
                "Ticker": ticker,
                "Strategy": "Hybrid 5min+1hr Confirm",
                "Score": score,
                "Entry": formatn(price),
                "Shares": shares,
                "Reason": reason,
                "Type": "Hybrid"
            })
    except Exception as e:
        st.write(f"{ticker}: {e}")
        continue

if results:
    df_out = pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)
    st.subheader("⭐ AI-Picked Stock Setups (Top 5)")
    st.dataframe(df_out.head(5), use_container_width=True)
else:
    st.info("No stocks met high-quality hybrid or EMA200 breakout criteria right now.")

st.markdown("""
---
**Strategy Summary:**  
- **Hybrid:** Enter on 5min EMA40 breakout *only if* 1hr price > EMA200 (trend confirmed).  
- **EMA200 Breakout:** Daily close above EMA200 + volume + RSI filter (the classic, rare but high conviction).
""")
