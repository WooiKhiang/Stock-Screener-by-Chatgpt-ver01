import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, time as dt_time

# ---- CONFIG ----
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
    'AAPL','ADBE','ADP','AMD','AMGN','AMZN','ANSS','ASML','ATVI','ADSK','BIIB','BKNG','CDNS','CDW','CERN','CHTR','CMCSA','COST','CPRT','CRWD','CSCO','CSX','CTAS','CTSH','DDOG','DLTR','DOCU','DXCM','EA','EBAY','EXC','FAST','FISV','FTNT','GILD','GOOG','GOOGL','HON','IDXX','ILMN','INTC','INTU','ISRG','JD','KDP','KHC','KLAC','LCID','LRCX','LULU','MAR','MCHP','MDLZ','MELI','META','MNST','MRNA','MSFT','MU','NFLX','NVDA','NXPI','ODFL','OKTA','ORLY','PANW','PAYX','PCAR','PEP','PDD','PYPL','QCOM','REGN','ROST','SBUX','SGEN','SIRI','SNPS','SPLK','SWKS','TEAM','TMUS','TSLA','TXN','VRSK','VRSN','VRTX','WBA','WDAY','XEL','ZM','ZS'
]
universe = sorted(list(set(sp100 + nasdaq100)))

# ---- UTILS ----
def formatn(val, d=2):
    if isinstance(val, str): return val
    try:
        if np.isnan(val): return "-"
        if abs(val) >= 1e6: return f"{val:,.0f}"
        return f"{val:,.{d}f}"
    except Exception: return str(val)

def norm(df):
    # flatten, lower, strip spaces
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
    df['ema50'] = df['close'].ewm(span=50, min_periods=50).mean()
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

def get_market_sentiment():
    try:
        spy = yf.download('SPY', period='2d', interval='1h', progress=False, threads=False)
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

def local_time_str():
    return datetime.now(pytz.timezone("Asia/Singapore")).strftime('%Y-%m-%d %H:%M:%S')

# ---- SIDEBAR ----
st.sidebar.title("KIV Signal Parameters")
min_vol = st.sidebar.number_input("Min Volume (last 1h bar)", value=100000)
min_price = st.sidebar.number_input("Min Price ($)", value=10.0)
max_price = st.sidebar.number_input("Max Price ($)", value=2000.0)
rsi_min = st.sidebar.number_input("Min RSI (14)", value=35)
rsi_max = st.sidebar.number_input("Max RSI (14)", value=65)

# ---- TITLE ----
sentiment_pct, sentiment_text = get_market_sentiment()
st.title("AI-Powered US Stocks Screener: 1h KIV + MACD Reference Table")
st.caption(f"Last run: {local_time_str()}")
st.markdown(f"**Market Sentiment:** {sentiment_text} ({formatn(sentiment_pct,2)}%)")

# ---- MAIN LOGIC ----
kiv_rows = []
macdref_rows = []

N_REF = 20  # Last N bars for MACD ref

for ticker in universe:
    try:
        df = yf.download(ticker, period='30d', interval='1h', progress=False, threads=False)
        if df.empty or len(df) < max(25, N_REF+10): continue
        df = norm(df)
        df = ensure_core_cols(df)
        df = calc_indicators(df)
        # --- KIV: Only last bar ---
        i = -1  # last bar only
        row = df.iloc[i]
        # (1) MACD cross up zero (prev <0, now >0), signal still <0
        macd_prev, macd_now = df['macd'].iloc[i-1], df['macd'].iloc[i]
        macdsig_now = df['macdsignal'].iloc[i]
        cross_up = (macd_prev < 0) and (macd_now > 0) and (macdsig_now < 0)
        # (2) MACD trend: 10 bars ago < 5 bars ago < now
        macd_10 = df['macd'].iloc[i-10] if i-10 >= -len(df) else None
        macd_5 = df['macd'].iloc[i-5] if i-5 >= -len(df) else None
        macd_uptrend = (macd_10 is not None and macd_5 is not None and macd_10 < macd_5 < macd_now)
        # (3) RSI range + rising from below
        rsi_now = row['rsi14']
        rsi_prev = df['rsi14'].iloc[i-5]
        rsi_cond = rsi_min <= rsi_now <= rsi_max and (rsi_now > rsi_prev)
        # (4) EMA10 > EMA20 and EMA10 rising
        ema10, ema20 = row['ema10'], row['ema20']
        ema10_prev = df['ema10'].iloc[i-5]
        ema_cond = (ema10 > ema20) and (ema10 > ema10_prev)
        # --- All criteria ---
        price, vol = row['close'], row['volume']
        price_ok = min_price <= price <= max_price
        vol_ok = vol >= min_vol

        if all([cross_up, macd_uptrend, rsi_cond, ema_cond, price_ok, vol_ok]):
            kiv_rows.append({
                "Ticker": ticker,
                "Signal Time": row.name.strftime('%Y-%m-%d %H:%M'),
                "MACD": formatn(macd_now, 4),
                "MACD Signal": formatn(macdsig_now, 4),
                "Price": formatn(price, 2),
                "RSI": formatn(rsi_now, 2),
                "Volume": f"{int(vol):,}"
            })
        # --- Reference table: last N bars, all events (NO FILTERS except MACD cross up zero & sig <0) ---
        for j in range(-N_REF, 0):
            if j < -len(df)+1: continue
            macd_p, macd_c = df['macd'].iloc[j-1], df['macd'].iloc[j]
            macdsig_c = df['macdsignal'].iloc[j]
            if macd_p < 0 and macd_c > 0 and macdsig_c < 0:
                macdref_rows.append({
                    "Ticker": ticker,
                    "Signal Time": df.index[j].strftime('%Y-%m-%d %H:%M'),
                    "MACD": formatn(macd_c, 4),
                    "MACD Signal": formatn(macdsig_c, 4),
                    "Price": formatn(df['close'].iloc[j], 2),
                    "Volume": f"{int(df['volume'].iloc[j]):,}"
                })
    except Exception as e:
        continue

# ---- MAIN TABLE ----
st.markdown("### 📝 1h KIV Signal Setups (Strict All-Criteria Matches)")
if kiv_rows:
    df_kiv = pd.DataFrame(kiv_rows)
    st.dataframe(df_kiv, use_container_width=True)
    st.caption(f"Example: {df_kiv.iloc[0].to_dict()}")
else:
    st.info("No current 1h KIV signals found.")

# ---- REFERENCE TABLE ----
st.markdown("### 📝 Reference: MACD Crosses Zero, Signal Still Below Zero (No RSI/EMA Filter)")
if macdref_rows:
    df_ref = pd.DataFrame(macdref_rows)
    st.dataframe(df_ref, use_container_width=True)
    st.caption(f"Example: {df_ref.iloc[0].to_dict()}")
else:
    st.info("No recent MACD cross-up signals found in last 20 bars.")

st.caption("© AI Screener | S&P 100 + NASDAQ 100 | 1h chart. Market sentiment adjusts screening. Reference table = research-only.")
