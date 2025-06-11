import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# --- S&P 100 Tickers (abbreviated, add more if needed) ---
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

# --- Streamlit Sidebar ---
st.sidebar.header("Filter Settings")
min_price = st.sidebar.number_input("Min Price ($)", value=10.0, key="min_price")
max_price = st.sidebar.number_input("Max Price ($)", value=2000.0, key="max_price")
min_volume = st.sidebar.number_input("Min Avg Vol (40 bars)", value=100000, key="min_vol")
capital_per_trade = st.sidebar.number_input("Capital per Trade ($)", value=1000.0, step=100.0, key="capital_trade")

st.title("🔍 S&P 100 Intraday Screener & AI Top 3 Stock Picks")
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
    df['AvgVol40'] = df['Volume'].replace(0, np.nan).rolling(window=40, min_periods=1).mean().fillna(0)
    return df

# --- Strategies ---
def mean_reversion_signal(df):
    c = safe_scalar(df['Close'].iloc[-1])
    sma = safe_scalar(df['SMA40'].iloc[-1])
    rsi = safe_scalar(df['RSI3'].iloc[-1])
    if np.isnan(c) or np.isnan(sma) or np.isnan(rsi):
        return False, None, 0
    score = 0
    cond = (c > sma) and (rsi < 15)
    if cond:
        score = 75 + max(0, 15 - rsi)  # bonus for more oversold
    return cond, "Mean Reversion: Price > SMA40 & RSI(3)<15", score

def ema40_breakout_signal(df):
    c = safe_scalar(df['Close'].iloc[-1])
    ema = safe_scalar(df['EMA40'].iloc[-1])
    pc = safe_scalar(df['Close'].iloc[-2])
    pema = safe_scalar(df['EMA40'].iloc[-2])
    if np.isnan(c) or np.isnan(ema) or np.isnan(pc) or np.isnan(pema):
        return False, None, 0
    left_vals = df['Close'].iloc[-10:-1].values
    right_vals = df['EMA40'].iloc[-10:-1].values
    if len(left_vals) == len(right_vals) and len(left_vals) > 0:
        dipped = (left_vals < right_vals).any()
    else:
        dipped = False
    cond = (c > ema) and ((pc < pema) or dipped)
    score = 0
    if cond:
        # The bigger the break and shakeout, the better
        magnitude = max(0, c - ema)
        score = 70 + min(20, magnitude)  # Cap the bonus for safety
    return cond, "EMA40 Breakout: Price reclaimed EMA40 (with shakeout)", score

def macd_ema_signal(df):
    macd = safe_scalar(df['MACD'].iloc[-1])
    macd_signal = safe_scalar(df['MACD_signal'].iloc[-1])
    macd_prev = safe_scalar(df['MACD'].iloc[-2])
    macd_signal_prev = safe_scalar(df['MACD_signal'].iloc[-2])
    ema8 = safe_scalar(df['EMA8'].iloc[-1])
    ema21 = safe_scalar(df['EMA21'].iloc[-1])
    if any(np.isnan(x) for x in [macd, macd_signal, macd_prev, macd_signal_prev, ema8, ema21]):
        return False, None, 0
    cross = (macd_prev < macd_signal_prev) and (macd > macd_signal) and (macd < 0)
    cond = cross and (ema8 > ema21)
    score = 0
    if cond:
        # The closer to zero (MACD rising up), the better
        score = 65 + int(abs(macd)*5)
    return cond, "MACD+EMA: MACD cross up <0 & EMA8>EMA21", score

# --- Main Loop ---
debug_rows = []
results = []

for ticker in sp100:
    debug_status = ""
    try:
        df = yf.download(ticker, period="5d", interval="5m", progress=False)
        if df.empty or len(df) < 50 or 'Close' not in df.columns or 'Volume' not in df.columns:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(col).strip() for col in df.columns.values]
            if 'Close' not in df.columns:
                close_col = [c for c in df.columns if 'Close' in c]
                if close_col:
                    df['Close'] = df[close_col[0]]
            if 'Volume' not in df.columns:
                volume_col = [c for c in df.columns if 'Volume' in c]
                if volume_col:
                    df['Volume'] = df[volume_col[0]]
            if df.empty or len(df) < 50 or 'Close' not in df.columns or 'Volume' not in df.columns:
                debug_status = f"{ticker}: Not enough data or missing Close/Volume column"
                debug_rows.append({'Ticker': ticker, 'Status': debug_status})
                continue
        if isinstance(df['Close'], pd.DataFrame):
            df['Close'] = df['Close'].iloc[:, 0]
        if isinstance(df['Volume'], pd.DataFrame):
            df['Volume'] = df['Volume'].iloc[:, 0]
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

        # --- Apply all strategies, collect all triggers ---
        picks = []
        for func, strat_name in [
            (mean_reversion_signal, "Mean Reversion"),
            (ema40_breakout_signal, "EMA40 Breakout"),
            (macd_ema_signal, "MACD+EMA")
        ]:
            sig, reason, score = func(df)
            if sig:
                picks.append((strat_name, reason, score))

        debug_rows.append({
            'Ticker': ticker,
            'Status': 'OK',
            'Close': close_price,
            'SMA40': safe_scalar(last['SMA40']),
            'RSI3': safe_scalar(last['RSI3']),
            'EMA40': safe_scalar(last['EMA40']),
            'MACD': safe_scalar(last['MACD']),
            'Volume': int(safe_scalar(df['Volume'][df['Volume'] > 0].iloc[-1])) if (df['Volume'] > 0).any() else 0,
            'AvgVol40': int(avgvol40),
            'MR?': any(x[0] == "Mean Reversion" for x in picks),
            'EMA?': any(x[0] == "EMA40 Breakout" for x in picks),
            'MACD?': any(x[0] == "MACD+EMA" for x in picks)
        })

        # --- If at least one strategy triggers, add for ranking ---
        if picks:
            # Pick the highest scoring strategy for this ticker
            picks.sort(key=lambda x: -x[2])
            strat_name, reason, score = picks[0]
            entry = close_price
            shares = int(capital_per_trade // entry)
            invested = shares * entry
            results.append({
                "Ticker": ticker,
                "Strategy": strat_name,
                "AI Score": score,
                "Entry Price": round(entry, 2),
                "Capital Used": round(invested, 2),
                "Shares": shares,
                "Reason": reason,
                "Volume": int(safe_scalar(df['Volume'][df['Volume'] > 0].iloc[-1])) if (df['Volume'] > 0).any() else 0,
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

st.header("AI-Powered Top 3 Intraday Stock Picks (S&P 100)")

if not df_results.empty:
    # Rank by AI Score (confidence/potential)
    df_results = df_results.sort_values(["AI Score", "Strategy", "RSI(3)"], ascending=[False, True, True])
    st.dataframe(df_results.reset_index(drop=True), use_container_width=True)

    # --- Recommendation Section ---
    top3 = df_results.head(3)
    st.subheader("📈 Today's AI Stock Recommendations")
    for idx, row in top3.iterrows():
        rank_num = idx + 1
        rank_emoji = "🥇" if rank_num == 1 else ("🥈" if rank_num == 2 else "🥉")
        st.markdown(f"""
        {rank_emoji} **Rank #{rank_num}: {row['Ticker']}**  
        **Signal:** {row['Strategy']}  
        **AI Confidence Score:** {row['AI Score']}  
        **Reason:** {row['Reason']}  
        **Entry Price:** ${row['Entry Price']} | **Capital Used:** ${row['Capital Used']}  
        **RSI(3):** {row['RSI(3)']} | **EMA40:** {row['EMA40']} | **SMA40:** {row['SMA40']}  
        """)
        if rank_num == 1:
            st.info("Rank #1: Highest confidence and strongest signal based on all indicators and strategy score. Most potential for intraday growth today.")
        elif rank_num == 2:
            st.info("Rank #2: Good signal, but a bit less compelling than #1. Still has solid edge today.")
        elif rank_num == 3:
            st.info("Rank #3: Valid setup, but less optimal than #1 or #2 based on strategy and indicator strength.")
else:
    st.info("No stocks meet your filter/strategy criteria right now.")

st.caption("Each recommendation above is ranked by a composite confidence score. View the debug table for diagnostic info on all tickers.")
