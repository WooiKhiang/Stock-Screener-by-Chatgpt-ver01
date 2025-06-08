import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="Go/No-Go Screener", layout="wide")

st.title("Market Go/No-Go Dashboard")

# -------- Sidebar Parameters --------
market = st.sidebar.selectbox("Select Market", ["US Market", "Malaysia (Bursa)"])

if market == "US Market":
    watchlist = ["SPY", "QQQ", "NVDA", "AAPL", "TSLA", "AMD", "META", "AMZN", "MSFT", "GOOGL"]
    currency = "USD"
else:
    # Example active Bursa stocks—customize as you like
    watchlist = ["CIMB.KL", "MAYBANK.KL", "TENAGA.KL", "PCHEM.KL", "TOPGLOV.KL", "SIME.KL", "PETGAS.KL", "GENTING.KL", "AXIATA.KL", "KLCC.KL"]
    currency = "MYR"

min_price = st.sidebar.number_input(f"Min Price ({currency})", value=5.0)
max_price = st.sidebar.number_input(f"Max Price ({currency})", value=500.0)
min_avg_vol = st.sidebar.number_input("Min Average Volume", value=1_000_000)

min_index_change = st.sidebar.number_input("Min Index Change (%)", value=0.05)
max_atr_percent = st.sidebar.number_input("Max ATR (%)", value=0.015)
volume_factor = st.sidebar.number_input("Min Volume Factor", value=0.7)

# New: EMA200 breakout lookback window
ema200_lookback = st.sidebar.number_input("EMA200 Breakout Lookback Bars", value=6, min_value=1, max_value=48)

# -------- Data Fetching --------
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

def scan_stock_all(ticker, spy_change, min_price, max_price, min_avg_vol, ema200_lookback):
    df = get_intraday_data(ticker)
    if df is None or len(df) < 22:
        return None

    today = df[df.index.date == df.index[-1].date()]
    if today.empty or len(today) < 10:
        return None

    try:
        today_open = float(today["Open"].iloc[0])
        today_close = float(today["Close"].iloc[-1])
        today_high = float(today["High"].max())
        today_low = float(today["Low"].min())
        today_change = (today_close - today_open) / today_open * 100
        price = today_close
    except Exception:
        return None

    try:
        avg_vol = float(today["Volume"].rolling(10).mean().iloc[-1])
        volume = float(today["Volume"].iloc[-1])
        rel_vol = volume / avg_vol if avg_vol else 0
    except Exception:
        avg_vol = volume = rel_vol = 0

    # Price/Volume Filter
    if not (min_price <= price <= max_price and avg_vol >= min_avg_vol):
        return None

    rs_score = today_change - spy_change

    try:
        last_hour = today.iloc[-12:]
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
        ema200_series = today["Close"].ewm(span=200).mean()
        ema200 = float(ema200_series.iloc[-1])
        price_above_ema10 = today_close > ema10
        price_above_ema20 = today_close > ema20
        price_above_ema50 = today_close > ema50
        price_above_ema200 = today_close > ema200
    except Exception:
        price_above_ema10 = price_above_ema20 = price_above_ema50 = price_above_ema200 = False
        ema200 = 0

    # Improved EMA200 Breakout: crossed within last N bars and still above
    try:
        ema200_breakout = False
        if len(today) > ema200_lookback:
            for i in range(1, ema200_lookback + 1):
                prev_close = float(today["Close"].iloc[-i-1])
                prev_ema200 = float(ema200_series.iloc[-i-1])
                curr_close = float(today["Close"].iloc[-i])
                curr_ema200 = float(ema200_series.iloc[-i])
                if (prev_close < prev_ema200) and (curr_close > curr_ema200):
                    ema200_breakout = True
                    break
            # Must still be above EMA200 now
            ema200_breakout = ema200_breakout and (today_close > ema200)
    except Exception:
        ema200_breakout = False

    try:
        exp12 = today["Close"].ewm(span=12).mean()
        exp26 = today["Close"].ewm(span=26).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9).mean()
        macd_bullish = float(macd.iloc[-1]) > float(signal.iloc[-1])
    except Exception:
        macd_bullish = False

    go_criteria = (
        price_above_ema10 and price_above_ema20 and price_above_ema50 and
        macd_bullish and volume_spike and rs_score > 0
    )

    nogo_criteria = (
        price_above_ema50 and not (price_above_ema10 and price_above_ema20) and rs_score > -1
    )

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
    if ema200_breakout:
        reasons.append("EMA200 Breakout (last {} bars)".format(ema200_lookback))

    return {
        "Ticker": ticker,
        "Price": f"{price:.2f}",
        "Open": f"{today_open:.2f}",
        "Close": f"{today_close:.2f}",
        "High": f"{today_high:.2f}",
        "Low": f"{today_low:.2f}",
        "Change %": f"{today_change:.2f}",
        "Avg Volume": f"{avg_vol:,.0f}",
        "Volume": f"{volume:,.0f}",
        "Rel Vol": f"{rel_vol:.2f}",
        "RS vs Index": f"{rs_score:.2f}",
        "VWAP": "Yes" if price_above_vwap else "No",
        "EMA 10": "Yes" if price_above_ema10 else "No",
        "EMA 20": "Yes" if price_above_ema20 else "No",
        "EMA 50": "Yes" if price_above_ema50 else "No",
        "EMA 200": f"{ema200:.2f}",
        "Volume Spike": "Yes" if volume_spike else "No",
        "MACD Bullish": "Yes" if macd_bullish else "No",
        "Reasons": ", ".join(reasons),
        "GO": go_criteria,
        "NO-GO": nogo_criteria,
        "Accumulation": accumulation_criteria,
        "EMA200 Breakout": ema200_breakout,
    }

# ---- Go/No-Go Dashboard Logic (unchanged) ----
spy = get_data('SPY') if market == "US Market" else get_data("FBMKLCI.KL")
qqq = get_data('QQQ') if market == "US Market" else get_data("KLSE.KL")
vxx = get_data('VXZ') if market == "US Market" else get_data("KLSE.KL")  # Placeholder

if not spy.empty and not qqq.empty:
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

    # For Malaysia, VXZ not relevant, just use dummy
    vxx_diff = 0

    market_health = spy_change >= min_index_change and qqq_change >= min_index_change
    liquidity = vol_factor_val >= volume_factor
    volatility = atr_percent <= max_atr_percent
    sentiment = vxx_diff < 0.03

    go_day = market_health and liquidity and volatility and sentiment

    st.subheader("Today's Market Conditions")
    st.markdown(f"**Index 1 Change:** {spy_change:.2f}%")
    st.markdown(f"**Index 2 Change:** {qqq_change:.2f}%")
    st.markdown(f"**Volume Factor:** {vol_factor_val:.2f}x")
    st.markdown(f"**ATR %:** {atr_percent:.3f}")

    st.subheader("GO/NO-GO Signal")
    if go_day:
        st.success("GO DAY! Risk-on opportunities detected.")
    else:
        st.error("NO-GO DAY. Consider capital protection strategies.")

else:
    st.warning("Could not load recent market data. Try again later.")

st.caption("Data source: Yahoo Finance")

# ---- Section: Stock Screens (all four regimes) ----

results = []
for ticker in watchlist:
    result = scan_stock_all(
        ticker,
        spy_change if 'spy_change' in locals() else 0,
        min_price, max_price, min_avg_vol,
        ema200_lookback
    )
    if result:
        results.append(result)
df_results = pd.DataFrame(results) if results else pd.DataFrame()

def show_section(title, filter_column, columns):
    st.subheader(title)
    if not df_results.empty:
        df = df_results[df_results[filter_column] == True]
        if not df.empty:
            st.dataframe(df[columns])
        else:
            st.info(f"No stocks meet {title.lower()} criteria.")
    else:
        st.info("No data for stock screening.")

# Section 1: Go Day Recommendations
show_section(
    "Go Day Stock Recommendations (Momentum/Breakout)",
    "GO",
    ["Ticker", "Price", "Open", "Close", "High", "Low", "Change %", "Volume", "Avg Volume", "Rel Vol", "RS vs Index", "VWAP", "EMA 10", "EMA 20", "EMA 50", "MACD Bullish", "Reasons"]
)

# Section 2: No-Go Day Defensive/Resilient Picks
show_section(
    "No-Go Day Stock Recommendations (Defensive/Resilient)",
    "NO-GO",
    ["Ticker", "Price", "Open", "Close", "High", "Low", "Change %", "Volume", "Avg Volume", "Rel Vol", "RS vs Index", "EMA 50", "Reasons"]
)

# Section 3: Potential Institutional Accumulation
show_section(
    "Potential Institutional Accumulation Screener",
    "Accumulation",
    ["Ticker", "Price", "Open", "Close", "High", "Low", "Change %", "Volume", "Avg Volume", "Rel Vol", "RS vs Index", "VWAP", "EMA 10", "EMA 20", "Reasons"]
)

# Section 4: EMA200 Breakout Reversal
show_section(
    "EMA200 Breakout Reversal Screener",
    "EMA200 Breakout",
    ["Ticker", "Price", "Open", "Close", "High", "Low", "Change %", "Volume", "Avg Volume", "Rel Vol", "RS vs Index", "EMA 200", "VWAP", "Reasons"]
)
