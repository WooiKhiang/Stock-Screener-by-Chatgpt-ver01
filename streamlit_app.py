import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="US Day Trading Screener", layout="wide")
st.title("US Market Go/No-Go Dashboard")

# -------- Sidebar Parameters --------
currency = "USD"

min_price = st.sidebar.number_input(f"Min Price ({currency})", value=5.0)
max_price = st.sidebar.number_input(f"Max Price ({currency})", value=500.0)
min_avg_vol = st.sidebar.number_input("Min Average Volume", value=1_000_000)

min_index_change = st.sidebar.number_input("Min Index Change (%)", value=0.05)
max_atr_percent = st.sidebar.number_input("Max ATR (%)", value=0.015)
volume_factor = st.sidebar.number_input("Min Volume Factor", value=0.7)
ema200_lookback = st.sidebar.number_input("EMA200 Breakout Lookback Bars", value=6, min_value=1, max_value=48)
pullback_lookback = st.sidebar.number_input("VWAP/EMA Pullback Lookback Bars", value=6, min_value=2, max_value=20)

# --- Position Size Calculator (in sidebar) ---
st.sidebar.markdown("---")
st.sidebar.header("Position Size Calculator")

account_size = st.sidebar.number_input("Account Size ($)", value=10000.0, step=100.0)
risk_per_trade_pct = st.sidebar.number_input("Risk per Trade (%)", value=1.0, min_value=0.1, max_value=10.0, step=0.1)
entry_price = st.sidebar.number_input("Entry Price", value=100.0, min_value=0.01, step=0.01)
stop_loss_price = st.sidebar.number_input("Stop-Loss Price", value=99.0, min_value=0.01, step=0.01)

# Calculation
risk_dollars = account_size * (risk_per_trade_pct / 100)
stop_distance = abs(entry_price - stop_loss_price)
if stop_distance > 0:
    position_size = risk_dollars / stop_distance
else:
    position_size = 0

st.sidebar.markdown(f"**Max Risk per Trade:** ${risk_dollars:,.2f}")
st.sidebar.markdown(f"**Position Size (shares):** {position_size:,.0f}")

# Optional: Show max loss if stopped out
if position_size > 0:
    max_loss = position_size * stop_distance
    st.sidebar.markdown(f"**Max Loss at Stop:** ${max_loss:,.2f}")


# -------- S&P 100 Watchlist (super liquid, active stocks) --------
watchlist = [
    "AAPL","ABBV","ABT","ACN","ADBE","AIG","AMGN","AMT","AMZN","AVGO",
    "AXP","BA","BAC","BK","BKNG","BLK","BMY","BRK.B","C","CAT",
    "CHTR","CL","CMCSA","COF","COP","COST","CRM","CSCO","CVS","CVX",
    "DHR","DIS","DOW","DUK","EMR","EXC","F","FDX","FOX","FOXA",
    "GD","GE","GILD","GM","GOOG","GOOGL","GS","HD","HON","IBM",
    "INTC","JNJ","JPM","KHC","KMI","KO","LIN","LLY","LMT","LOW",
    "MA","MCD","MDLZ","MDT","MET","META","MMM","MO","MRK","MS",
    "MSFT","NEE","NFLX","NKE","NVDA","ORCL","PEP","PFE","PG","PM",
    "PYPL","QCOM","RTX","SBUX","SCHW","SO","SPG","T","TGT","TMO",
    "TMUS","TSLA","TXN","UNH","UNP","UPS","USB","V","VZ","WBA",
    "WFC","WMT","XOM"
]

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

def scan_stock_all(ticker, spy_change, min_price, max_price, min_avg_vol, ema200_lookback, pullback_lookback):
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

    # EMA200 Breakout: crossed within last N bars and still above
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

    # VWAP/EMA20 Pullback Bounce logic
    try:
        pullback = False
        if len(today) > pullback_lookback:
            for i in range(2, pullback_lookback + 1):
                prev_low = float(today["Low"].iloc[-i])
                prev_vwap = float(vwap.iloc[-i])
                prev_ema20 = float(today["Close"].ewm(span=20).mean().iloc[-i])
                if (
                    (abs(prev_low - prev_vwap) / prev_vwap < 0.0025 or abs(prev_low - prev_ema20) / prev_ema20 < 0.0025)
                    and (today_close > prev_vwap or today_close > prev_ema20)
                ):
                    pullback = True
                    break
    except Exception:
        pullback = False

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
        reasons.append(f"EMA200 Breakout ({ema200_lookback} bars)")
    if pullback:
        reasons.append(f"VWAP/EMA20 Pullback ({pullback_lookback} bars)")

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
        "RS vs SPY": f"{rs_score:.2f}",
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
        "Pullback Bounce": pullback,
    }

# ---- Go/No-Go Dashboard Logic ----
spy = get_data('SPY')
qqq = get_data('QQQ')

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

    st.subheader("Today's Market Conditions")
    st.markdown(f"**SPY Change:** {spy_change:.2f}%")
    st.markdown(f"**QQQ Change:** {qqq_change:.2f}%")
    st.markdown(f"**Volume Factor:** {vol_factor_val:.2f}x")
    st.markdown(f"**ATR %:** {atr_percent:.3f}")

    st.subheader("GO/NO-GO Signal")
    go_day = (
        spy_change >= min_index_change and
        qqq_change >= min_index_change and
        vol_factor_val >= volume_factor and
        atr_percent <= max_atr_percent
    )
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
        ema200_lookback, pullback_lookback
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
    ["Ticker", "Price", "Open", "Close", "High", "Low", "Change %", "Volume", "Avg Volume", "Rel Vol", "RS vs SPY", "VWAP", "EMA 10", "EMA 20", "EMA 50", "MACD Bullish", "Reasons"]
)

# Section 2: No-Go Day Defensive/Resilient Picks
show_section(
    "No-Go Day Stock Recommendations (Defensive/Resilient)",
    "NO-GO",
    ["Ticker", "Price", "Open", "Close", "High", "Low", "Change %", "Volume", "Avg Volume", "Rel Vol", "RS vs SPY", "EMA 50", "Reasons"]
)

# Section 3: Potential Institutional Accumulation
show_section(
    "Potential Institutional Accumulation Screener",
    "Accumulation",
    ["Ticker", "Price", "Open", "Close", "High", "Low", "Change %", "Volume", "Avg Volume", "Rel Vol", "RS vs SPY", "VWAP", "EMA 10", "EMA 20", "Reasons"]
)

# Section 4: EMA200 Breakout Reversal
show_section(
    "EMA200 Breakout Reversal Screener",
    "EMA200 Breakout",
    ["Ticker", "Price", "Open", "Close", "High", "Low", "Change %", "Volume", "Avg Volume", "Rel Vol", "RS vs SPY", "EMA 200", "VWAP", "Reasons"]
)

# Section 5: VWAP/EMA Pullback Bounce Screener
show_section(
    "VWAP/EMA20 Pullback Bounce Screener",
    "Pullback Bounce",
    ["Ticker", "Price", "Open", "Close", "High", "Low", "Change %", "Volume", "Avg Volume", "Rel Vol", "RS vs SPY", "VWAP", "EMA 20", "Reasons"]
)
