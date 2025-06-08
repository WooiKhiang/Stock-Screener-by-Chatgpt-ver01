import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

st.title("Market Go/No-Go Dashboard")

# --- User Parameters ---
min_index_change = st.number_input("Min Index Change (%)", value=0.05)
max_atr_percent = st.number_input("Max ATR (%)", value=0.015)
volume_factor = st.number_input("Min Volume Factor", value=0.7)

# --- Data Fetching ---
def get_data(ticker):
    end = datetime.now()
    start = end - timedelta(days=5)
    df = yf.download(ticker, start=start, end=end, interval='5m', progress=False)
    return df

spy = get_data('SPY')
qqq = get_data('QQQ')
vxx = get_data('VXZ')  # VXZ as replacement for VXX

if not spy.empty and not qqq.empty and not vxx.empty:
    # --- Change Calculations ---
    if len(spy) > 2 and len(qqq) > 2:
        try:
            spy_change = (float(spy['Close'].iloc[-1]) - float(spy['Close'].iloc[-2])) / float(spy['Close'].iloc[-2]) * 100
        except Exception:
            spy_change = 0
        try:
            qqq_change = (float(qqq['Close'].iloc[-1]) - float(qqq['Close'].iloc[-2])) / float(qqq['Close'].iloc[-2]) * 100
        except Exception:
            qqq_change = 0
    else:
        spy_change = qqq_change = 0

    # --- Volume & ATR ---
    if len(spy) > 15:
        try:
            spy_volume = float(spy['Volume'].iloc[-1])
        except Exception:
            spy_volume = 0
        try:
            spy_avg_volume = float(spy['Volume'].rolling(window=10).mean().iloc[-1])
        except Exception:
            spy_avg_volume = 0
        try:
            spy_atr = float((spy['High'] - spy['Low']).rolling(window=14).mean().iloc[-1])
        except Exception:
            spy_atr = 0
        try:
            close_last = float(spy['Close'].iloc[-1])
        except Exception:
            close_last = 0
        if close_last != 0:
            atr_percent = spy_atr / close_last
        else:
            atr_percent = 0
        if pd.notna(spy_avg_volume) and spy_avg_volume != 0:
            vol_factor_val = spy_volume / spy_avg_volume
        else:
            vol_factor_val = 0
    else:
        spy_volume = spy_avg_volume = spy_atr = atr_percent = vol_factor_val = 0

    # --- Sentiment (VXZ) ---
    if len(vxx) > 20:
        try:
            vxx_fast = float(vxx['Close'].rolling(window=5).mean().iloc[-1])
        except Exception:
            vxx_fast = 0
        try:
            vxx_slow = float(vxx['Close'].rolling(window=20).mean().iloc[-1])
        except Exception:
            vxx_slow = 0
        if pd.notna(vxx_slow) and vxx_slow != 0:
            vxx_diff = (vxx_fast - vxx_slow) / vxx_slow
        else:
            vxx_diff = 0
    else:
        vxx_diff = 0

    # --- Go/No-Go Logic ---
    market_health = spy_change >= min_index_change and qqq_change >= min_index_change
    liquidity = vol_factor_val >= volume_factor
    volatility = atr_percent <= max_atr_percent
    sentiment = vxx_diff < 0.03

    go_day = market_health and liquidity and volatility and sentiment

    # --- Dashboard ---
    st.subheader("Today's Market Conditions")
    st.markdown(f"**SPY Change:** {spy_change:.2f}%")
    st.markdown(f"**QQQ Change:** {qqq_change:.2f}%")
    st.markdown(f"**Volume Factor:** {vol_factor_val:.2f}x")
    st.markdown(f"**ATR %:** {atr_percent:.3f}")
    st.markdown(f"**VXZ Fast/Slow Diff:** {vxx_diff:.3f}")

    st.subheader("GO/NO-GO Signal")
    if go_day:
        st.success("GO DAY! Risk-on opportunities detected.")
    else:
        st.error("NO-GO DAY. Consider capital protection strategies.")

else:
    st.warning("Could not load recent market data. Try again later.")

st.caption("Data source: Yahoo Finance")

# --- Step 2: Institutional Accumulation Screener (always runs for testing) ---

watchlist = ["SPY", "QQQ", "NVDA", "AAPL", "TSLA", "AMD", "META", "AMZN", "MSFT", "GOOGL"]

def get_intraday_data(ticker):
    try:
        df = yf.download(ticker, period="2d", interval="5m", progress=False)
    except Exception:
        return None
    return df

def scan_stock(ticker, spy_change):
    df = get_intraday_data(ticker)
    if df is None or len(df) < 22:
        return None

    today = df[df.index.date == df.index[-1].date()]
    if today.empty or len(today) < 10:
        return None

    # Today's % change
    try:
        today_change = (float(today["Close"].iloc[-1]) - float(today["Open"].iloc[0])) / float(today["Open"].iloc[0]) * 100
    except Exception:
        today_change = 0

    # Relative Strength Score
    rs_score = today_change - spy_change

    # Volume Spike (any 5m bar in last hour >2x average)
    try:
        last_hour = today.iloc[-12:]
        avg_vol = today["Volume"].rolling(10).mean().iloc[-1]
        volume_spike = any(last_hour["Volume"] > 2 * avg_vol)
    except Exception:
        volume_spike = False

    # Above VWAP
    vwap = (today["Volume"] * (today["High"] + today["Low"] + today["Close"]) / 3).cumsum() / today["Volume"].cumsum()
    price_above_vwap = float(today["Close"].iloc[-1]) > float(vwap.iloc[-1])

    # Above EMAs (10, 20, 50)
    ema10 = today["Close"].ewm(span=10).mean().iloc[-1]
    ema20 = today["Close"].ewm(span=20).mean().iloc[-1]
    ema50 = today["Close"].ewm(span=50).mean().iloc[-1]
    price_above_ema10 = float(today["Close"].iloc[-1]) > ema10
    price_above_ema20 = float(today["Close"].iloc[-1]) > ema20
    price_above_ema50 = float(today["Close"].iloc[-1]) > ema50

    # MACD (12,26,9) bullish?
    exp12 = today["Close"].ewm(span=12).mean()
    exp26 = today["Close"].ewm(span=26).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9).mean()
    macd_bullish = macd.iloc[-1] > signal.iloc[-1]

    # Reasoning
    reasons = []
    if rs_score > 0:
        reasons.append("Strong RS")
    if price_above_vwap:
        reasons.append("VWAP Reclaim")
    if price_above_ema10 and price_above_ema20:
        reasons.append("EMA Alignment")
    if volume_spike:
        reasons.append("Volume Spike")
    if macd_bullish:
        reasons.append("MACD Bullish")

    # Less strict for debug (only needs RS, EMA10, EMA20)
    all_criteria = (rs_score > 0 and price_above_ema10 and price_above_ema20)

    return {
        "Ticker": ticker,
        "Change %": f"{today_change:.2f}",
        "RS vs SPY": f"{rs_score:.2f}",
        "VWAP": "Yes" if price_above_vwap else "No",
        "EMA Align": "Yes" if price_above_ema10 and price_above_ema20 else "No",
        "Volume Spike": "Yes" if volume_spike else "No",
        "MACD Bullish": "Yes" if macd_bullish else "No",
        "Reasons": ", ".join(reasons),
        "Qualified": all_criteria
    }

# --- Run Screener (always, for now) ---
st.subheader("Institutional Accumulation Screener (DEBUG)")

results = []
all_results = []
for ticker in watchlist:
    result = scan_stock(ticker, spy_change)
    if result:
        all_results.append(result)
        if result["Qualified"]:
            results.append(result)

if results:
    st.success("Qualified Accumulation Setups:")
    st.dataframe(pd.DataFrame(results)[["Ticker", "Change %", "RS vs SPY", "VWAP", "EMA Align", "Volume Spike", "MACD Bullish", "Reasons"]])
else:
    st.info("No accumulation setups found in current watchlist with current criteria.")

# DEBUG: Show all scan results, not just qualified
if all_results:
    st.write("All scan results (for debugging):")
    st.dataframe(pd.DataFrame(all_results)[["Ticker", "Change %", "RS vs SPY", "VWAP", "EMA Align", "Volume Spike", "MACD Bullish", "Reasons"]])
