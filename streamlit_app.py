import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, time as dt_time

# -- Ticker Universe --
SP100 = [
    'AAPL','ABBV','ABT','ACN','ADBE','AIG','AMGN','AMT','AMZN','AVGO','AXP','BA','BAC','BK','BKNG','BLK','BMY','BRK-B','C','CAT',
    'CHTR','CL','CMCSA','COF','COP','COST','CRM','CSCO','CVS','CVX','DHR','DIS','DOW','DUK','EMR','EXC','F','FDX','FOX','FOXA',
    'GD','GE','GILD','GM','GOOG','GOOGL','GS','HD','HON','IBM','INTC','JNJ','JPM','KHC','KMI','KO','LIN','LLY','LMT','LOW',
    'MA','MCD','MDLZ','MDT','MET','META','MMM','MO','MRK','MS','MSFT','NEE','NFLX','NKE','NVDA','ORCL','PEP','PFE','PG','PM',
    'PYPL','QCOM','RTX','SBUX','SCHW','SO','SPG','T','TGT','TMO','TMUS','TSLA','TXN','UNH','UNP','UPS','USB','V','VZ','WBA',
    'WFC','WMT','XOM'
]
NDX100 = [
    'AAPL','MSFT','GOOGL','GOOG','AMZN','NVDA','META','PEP','COST','AVGO','TMUS','CSCO','ADBE','CMCSA','TXN','AMGN','AMAT','INTC','QCOM','GILD',
    'REGN','LRCX','ADP','SBUX','MDLZ','VRTX','ISRG','INTU','MU','MAR','ADI','CTAS','ATVI','KLAC','MELI','IDXX','CDNS','ASML','DXCM','PANW',
    'CSX','SNPS','MNST','ORLY','EXC','AEP','KDP','ODFL','FAST','PCAR','XEL','SGEN','ROST','PAYX','BIIB','EA','DLTR','PDD','ILMN','SIRI',
    'ANSS','TEAM','MRVL','WDAY','CPRT','CTSH','JD','LCID','BKR','CHTR','ALGN','EBAY','SWKS','TTD','ZS','VRSK','NTES','OKTA','MTCH','CRWD',
    'DOCU','DDOG','FANG','PTON','VRSN','FOXA','FOX','ZM','BIDU','SPLK','UAL','LULU','EXPE','SGEN','NXPI','BMRN','WBD','LKQ'
]
TICKERS = sorted(set(SP100 + NDX100))

# -- Helper functions --
def formatn(num, d=2, pct=False):
    try:
        if num is None or num == "" or np.isnan(num): return "-"
        if pct:
            return f"{num:.{d}f}%"
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

# --- Sidebar ---
st.sidebar.header("KIV Signal Parameters")
min_vol = st.sidebar.number_input("Min Volume (last 1h bar)", value=100000, step=10000)
min_price = st.sidebar.number_input("Min Price ($)", value=10.0)
max_price = st.sidebar.number_input("Max Price ($)", value=2000.0)
min_rsi = st.sidebar.number_input("Min RSI", value=35, min_value=0, max_value=100)
max_rsi = st.sidebar.number_input("Max RSI", value=65, min_value=0, max_value=100)

# --- Market Sentiment ---
sentiment_pct, sentiment_text = get_market_sentiment()
st.title("AI-Powered US Stocks Screener: 1h KIV + MACD Reference Table")
st.caption(f"Last run: {local_time_str()}")
st.markdown(f"### Market Sentiment: {'🟢' if 'Bullish' in sentiment_text else '🔴' if 'Bearish' in sentiment_text else '🟡'} {sentiment_text} ({formatn(sentiment_pct,2,pct=True)})")

# --- Screener Logic ---
kiv_results = []
macd_crosses = []
for ticker in TICKERS:
    try:
        df = yf.download(ticker, period='30d', interval='1h', progress=False, threads=False)
        if df.empty or len(df) < 30: continue
        df = norm(df)
        df = ensure_core_cols(df)
        df = calc_indicators(df)
        macd = df['macd'].iloc[-1]
        macdsignal = df['macdsignal'].iloc[-1]
        close = df['close'].iloc[-1]
        vol = df['volume'].iloc[-1]
        rsi = df['rsi14'].iloc[-1]
        # --- Main KIV Signal Logic (strict) ---
        macd_prev_10 = df['macd'].iloc[-10] if len(df) > 10 else None
        macd_prev_5 = df['macd'].iloc[-5] if len(df) > 5 else None
        ema10 = df['ema10'].iloc[-1]
        ema20 = df['ema20'].iloc[-1]
        rsi_is_growing = rsi > df['rsi14'].iloc[-2] > df['rsi14'].iloc[-3]
        macd_uptrend = macd_prev_10 is not None and macd_prev_5 is not None and macd_prev_10 < macd_prev_5 < macd
        cond_kiv = (
            macd > 0 and macdsignal < 0 and
            macd_uptrend and
            min_rsi <= rsi <= max_rsi and rsi_is_growing and
            ema10 > ema20 and ema10 > df['ema10'].iloc[-2] and
            min_price <= close <= max_price and vol >= min_vol
        )
        if cond_kiv:
            kiv_results.append({
                "Ticker": ticker,
                "Signal Time": local_time_str(),
                "MACD": formatn(macd, 4),
                "MACD Signal": formatn(macdsignal, 4),
                "RSI": formatn(rsi, 2),
                "Price": formatn(close, 2),
                "Volume": f"{int(vol):,}"
            })
        # --- MACD Cross Reference Table (no RSI/EMA) ---
        macd_ref_cross = (macd > 0) and (macdsignal < 0)
        if macd_ref_cross:
            macd_crosses.append({
                "Ticker": ticker,
                "Signal Time": local_time_str(),
                "MACD": formatn(macd, 4),
                "MACD Signal": formatn(macdsignal, 4),
                "Price": formatn(close, 2),
                "Volume": f"{int(vol):,}"
            })
    except Exception as e:
        continue

# --- Output Main KIV Signal Table ---
st.markdown("#### 📝 1h KIV Signal Setups (Strict All-Criteria Matches)")
if kiv_results:
    df_kiv = pd.DataFrame(kiv_results)
    st.dataframe(df_kiv, use_container_width=True)
else:
    st.info("No current 1h KIV signals found.")

# --- Output Reference Table ---
st.markdown("#### 🗒️ Reference: MACD Crosses Zero, Signal Still Below Zero (No RSI/EMA Filter)")
if macd_crosses:
    df_ref = pd.DataFrame(macd_crosses)
    st.dataframe(df_ref, use_container_width=True)
else:
    st.info("No current MACD ref signals found.")

st.caption("© AI Screener | S&P 100 + NASDAQ 100. 1h chart. Market sentiment adjusts screening. RSI filter only applies to KIV Table.")

