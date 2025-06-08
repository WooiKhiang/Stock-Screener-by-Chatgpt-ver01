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

def get_intraday_data(ticker):
    try:
        df = yf.download(ticker, period="2d", interval="5m", progress=False)
    except Exception:
        return None
    return df

def scan_stock_all(ticker, spy_change):
    df = get_intraday_data(ticker)
    if df is None or len(df) < 22:
        return None

    today = df[df.index.date == df.index[-1].date()]
    if today.empty or len(today) < 10:
        return None

    try:
        today_open = float(today["Open"].iloc[0])
        today_close = float(today["Close"].iloc[-1])
        today_change = (today_close - today_open) / today_open * 100
    except Exception:
        today_change = 0

    rs_score = today_change - spy_change

    try:
        last_hour = today.iloc[-12:]
        avg_vol = float(today["Volume"].rolling(10).mean().iloc[-1])
        volume_spike = any(last_hour["Volume"] > 2 * avg_vol)
    except Exception:
        volume_spike = False

    try:
        vwap = (today["Volume"] * (today["High"] + today["Low"] + today["Close"]) / 3).cumsum() / today["Volume"].cumsum()
        price_above_vwap = today_close > float(vwap.iloc[-1])
    except Exception:
        price_above_vwap = False

    try:
        ema10 = float(today["Close"].ewm(span=10).mean().iloc[-1])
        ema20 = float(today["Close"].ewm(span=20).mean().iloc[-1])
        ema50 = float(today["Close"].ewm(span=50).mean().iloc[-1])
        price_above_ema10 = today_close > ema10
        price_above_ema20 = today_close > ema20
        price_above_ema50 = today_close > ema50
    except Exception:
        price_above_ema10 = price_above_ema20 = price_above_ema50 = False

    try:
        exp12 = today["Close"].ewm(span=12).mean()
        exp26 = today["Close"].ewm(span=26).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9).mean()
        macd_bullish = float(macd.iloc[-1]) > float(signal.iloc[-1])
    except Exception:
        macd_bullish = False

    # Section 1: GO DAY
    go_criteria = (
        price_above_ema10 and price_above_ema20 and price_above_ema50 and
        macd_bullish and volume_spike and rs_score > 0
    )

    # Section 2: NO-GO DAY
    nogo_criteria = (
        price_above_ema50 and not (price_above_ema10 and price_above_ema20) and rs_score > -1
    )

    # Section 3: Institutional Accumulation (already have criteria)
    accumulation_criteria = (
        rs_score > 0 and price_above_ema10 and price_above_ema20
    )

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

    return {
        "Ticker": ticker,
        "Change %": f"{today_change:.2f}",
        "RS vs SPY": f"{rs_score:.2f}",
        "VWAP": "Yes" if price_above_vwap else "No",
        "EMA 10": "Yes" if price_above_ema10 else "No",
        "EMA 20": "Yes" if price_above_ema20 else "No",
        "EMA 50": "Yes" if price_above_ema50 else "No",
        "Volume Spike": "Yes" if volume_spike else "No",
        "MACD Bullish": "Yes" if macd_bullish else "No",
        "Reasons": ", ".join(reasons),
        "GO": go_criteria,
        "NO-GO": nogo_criteria,
        "Accumulation": accumulation_criteria,
    }

# --- Go/No-Go Dashboard Logic ---
spy = get_data('SPY')
qqq = get_data('QQQ')
vxx = get_data('VXZ')  # VXZ as replacement for VXX

if not spy.empty and not qqq.empty and not vxx.empty:
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

    market_health = spy_change >= min_index_change and qqq_change >= min_index_change
    liquidity = vol_factor_val >= volume_factor
    volatility = atr_percent <= max_atr_percent
    sentiment = vxx_diff < 0.03

    go_day = market_health and liquidity and volatility and sentiment

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

# --- Section: Stock Screens (for all three regimes) ---

watchlist = ["SPY", "QQQ", "NVDA", "AAPL", "TSLA", "AMD", "META", "AMZN", "MSFT", "GOOGL"]

results = []
for ticker in watchlist:
    result = scan_stock_all(ticker, spy_change if 'spy_change' in locals() else 0)
    if result:
        results.append(result)
df_results = pd.DataFrame(results) if results else pd.DataFrame()

# Section 1: Go Day Recommendations
st.subheader("Go Day Stock Recommendations (Momentum/Breakout)")
if not df_results.empty:
    go_df = df_results[df_results["GO"] == True]
    if not go_df.empty:
        st.dataframe(go_df[["Ticker", "Change %", "RS vs SPY", "VWAP", "EMA 10", "EMA 20", "EMA 50", "Volume Spike", "MACD Bullish", "Reasons"]])
    else:
        st.info("No stocks meet GO Day momentum/breakout criteria.")
else:
    st.info("No data for stock screening.")

# Section 2: No-Go Day Defensive/Resilient Picks
st.subheader("No-Go Day Stock Recommendations (Defensive/Resilient)")
if not df_results.empty:
    nogo_df = df_results[df_results["NO-GO"] == True]
    if not nogo_df.empty:
        st.dataframe(nogo_df[["Ticker", "Change %", "RS vs SPY", "EMA 50", "Reasons"]])
    else:
        st.info("No stocks meet NO-GO Day defensive/resilient criteria.")

# Section 3: Potential Institutional Accumulation
st.subheader("Potential Institutional Accumulation Screener")
if not df_results.empty:
    acc_df = df_results[df_results["Accumulation"] == True]
    if not acc_df.empty:
        st.dataframe(acc_df[["Ticker", "Change %", "RS vs SPY", "VWAP", "EMA 10", "EMA 20", "Reasons"]])
    else:
        st.info("No stocks meet institutional accumulation criteria.")
