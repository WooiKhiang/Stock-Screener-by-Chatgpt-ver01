import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, date
import requests

# --- SETTINGS ---
TELEGRAM_BOT_TOKEN = "7280991990:AAEk5x4XFCW_sTohAQGUujy1ECAQHjSY_OU"
TELEGRAM_CHAT_ID = "713264762"
ALERT_LOG = "alerts_log.csv"

st.set_page_config(page_title="US Market Day Trading Screener", layout="wide")
st.title("US Market Go/No-Go Dashboard")

# --- Sidebar: Screener Criteria ---
with st.sidebar.expander("Screener Criteria", expanded=True):
    min_price = st.number_input("Min Price ($)", value=5.0)
    max_price = st.number_input("Max Price ($)", value=500.0)
    min_avg_vol = st.number_input("Min Avg Volume", value=1_000_000)
    min_index_change = st.number_input("Min Index Change (%)", value=0.05)
    max_atr_percent = st.number_input("Max ATR (%)", value=0.015)
    volume_factor = st.number_input("Min Volume Factor", value=0.7)
    ema200_lookback = st.number_input("EMA200 Breakout Lookback", value=6, min_value=1, max_value=48)
    pullback_lookback = st.number_input("VWAP/EMA Pullback Lookback", value=6, min_value=2, max_value=20)

with st.sidebar.expander("Profit & Risk Planner", expanded=True):
    capital = st.number_input("Capital per Trade ($)", value=1000.0, step=100.0)
    take_profit_pct = st.number_input("Take Profit (%)", value=2.0, min_value=0.5, max_value=10.0, step=0.1) / 100
    cut_loss_pct = st.number_input("Cut Loss (%)", value=1.0, min_value=0.2, max_value=5.0, step=0.1) / 100

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

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        st.warning(f"Failed to send Telegram alert: {e}")

def was_alerted_today(ticker, section_name):
    today_str = str(date.today())
    try:
        with open(ALERT_LOG, "r") as f:
            for row in f:
                cols = row.strip().split(",")
                if len(cols) == 3 and cols[0] == today_str and cols[1] == section_name and cols[2] == ticker:
                    return True
    except FileNotFoundError:
        return False
    return False

def add_to_alert_log(ticker, section_name):
    today_str = str(date.today())
    with open(ALERT_LOG, "a") as f:
        f.write(f"{today_str},{section_name},{ticker}\n")

def get_intraday_data(ticker):
    try:
        df = yf.download(ticker, period="2d", interval="5m", progress=False)
    except Exception:
        return None
    return df

def get_daily_data(ticker, days=15):
    try:
        df = yf.download(ticker, period=f"{days}d", interval="1d", progress=False)
    except Exception:
        return None
    return df

def calc_pivot_points(df_day):
    try:
        high = df_day["High"]
        low = df_day["Low"]
        close = df_day["Close"]
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        return f"{s1:.2f}", f"{s2:.2f}", f"{pivot:.2f}", f"{r1:.2f}", f"{r2:.2f}"
    except Exception:
        return "-", "-", "-", "-", "-"

def scan_stock_all(ticker, spy_change, min_price, max_price, min_avg_vol, ema200_lookback, pullback_lookback,
                   capital, take_profit_pct, cut_loss_pct):
    df = get_intraday_data(ticker)
    if df is None or len(df) < 22:
        return None

    today = df[df.index.date == df.index[-1].date()]
    if today.empty or len(today) < 10:
        return None

    try:
        today_open = float(today["Open"].iloc[0])
        today_close = float(today["Close"].iloc[-1])
        price = today_close
        avg_vol = float(today["Volume"].rolling(10).mean().iloc[-1])
        volume = float(today["Volume"].iloc[-1])
    except Exception:
        return None

    if not (min_price <= price <= max_price and avg_vol >= min_avg_vol):
        return None

    try:
        today_change = (today_close - today_open) / today_open * 100
        rs_score = today_change - spy_change
    except Exception:
        today_change = 0
        rs_score = 0

    # --- EMA/VWAP checks (for signal/notes only) ---
    try:
        ema200_series = today["Close"].ewm(span=200).mean()
        ema200 = float(ema200_series.iloc[-1])
        crossed = False
        for i in range(1, min(ema200_lookback, len(today)-1)):
            if today["Close"].iloc[-i-1] < ema200_series.iloc[-i-1] and today["Close"].iloc[-i] > ema200_series.iloc[-i]:
                crossed = True
                break
    except Exception:
        ema200, crossed = 0, False

    try:
        vwap = (today["Volume"] * (today["High"] + today["Low"] + today["Close"]) / 3).cumsum() / today["Volume"].cumsum()
        vwap_signal = today_close > float(vwap.iloc[-1])
    except Exception:
        vwap_signal = False

    # --- Trade Management ---
    target_price = price * (1 + take_profit_pct)
    cut_loss_price = price * (1 - cut_loss_pct)
    position_size = int(capital / price) if price > 0 else 0

    # --- Pivot/Sup/Res for Top5: need 2 days daily data
    df_day = get_daily_data(ticker, days=3)
    if df_day is not None and len(df_day) >= 2:
        prev_day = df_day.iloc[-2]
        s1, s2, pivot, r1, r2 = calc_pivot_points(prev_day)
        resistance = r1
        support = s1
    else:
        support, resistance, pivot = "-", "-", "-"

    # --- AI score: simple ranking by RS score + volume for demo
    ai_score = 0.6 * rs_score + 0.4 * (volume / avg_vol if avg_vol else 0)

    # --- Compose Reason ---
    notes = []
    if rs_score > 0: notes.append("Strong RS")
    if crossed: notes.append("EMA200 Crossover")
    if vwap_signal: notes.append("VWAP Above")
    notes = ", ".join(notes) if notes else "-"

    return {
        "Ticker": ticker,
        "Price": f"{price:.2f}",
        "Change %": f"{today_change:.2f}",
        "RS Score": f"{rs_score:.2f}",
        "Volume": f"{volume:,.0f}",
        "AI Score": ai_score,
        "Support (S1)": support,
        "Resistance (R1)": resistance,
        "Pivot": pivot,
        "GO": rs_score > 1,
        "NO-GO": rs_score <= 1,
        "Target Price": f"{target_price:.2f}",
        "Cut Loss Price": f"{cut_loss_price:.2f}",
        "Position Size": position_size,
        "AI Reasoning": notes
    }

# --- Market Sentiment Analysis (FIXED) ---
spy = get_intraday_data('SPY')
qqq = get_intraday_data('QQQ')
go_day = False

spy_change = 0
qqq_change = 0

if spy is not None and not spy.empty and qqq is not None and not qqq.empty:
    spy_today = spy[spy.index.date == spy.index[-1].date()]
    qqq_today = qqq[qqq.index.date == qqq.index[-1].date()]
    if len(spy_today) > 10 and len(qqq_today) > 10:
        try:
            spy_open = float(spy_today["Open"].iloc[0])
            spy_close = float(spy_today["Close"].iloc[-1])
            spy_change = (spy_close - spy_open) / spy_open * 100
        except Exception as e:
            st.warning(f"Error calculating SPY change: {e}")
            spy_change = 0
        try:
            qqq_open = float(qqq_today["Open"].iloc[0])
            qqq_close = float(qqq_today["Close"].iloc[-1])
            qqq_change = (qqq_close - qqq_open) / qqq_open * 100
        except Exception as e:
            st.warning(f"Error calculating QQQ change: {e}")
            qqq_change = 0

        st.subheader("Market Sentiment Panel")
        st.markdown(f"**SPY:** {spy_change:.2f}% | **QQQ:** {qqq_change:.2f}%")
        go_day = (spy_change > float(min_index_change)) and (qqq_change > float(min_index_change))
        st.markdown("**Trend:** " + ("🟢 GO Day – Favorable" if go_day else "🔴 NO-GO Day – Defensive"))
    else:
        st.warning("Could not load enough recent market data. Try again later.")
        spy_change = qqq_change = 0
else:
    st.warning("Could not load recent market data. Try again later.")
    spy_change = qqq_change = 0

# --- Screener ---
results = []
for ticker in watchlist:
    result = scan_stock_all(
        ticker, spy_change,
        min_price, max_price, min_avg_vol,
        ema200_lookback, pullback_lookback,
        capital, take_profit_pct, cut_loss_pct
    )
    if result:
        results.append(result)
df_results = pd.DataFrame(results) if results else pd.DataFrame()

# --- Show results, AI Picks, Alerts ---
trade_cols = [
    "Ticker", "Price", "Change %", "RS Score", "Volume",
    "Support (S1)", "Resistance (R1)", "Pivot",
    "Target Price", "Cut Loss Price", "Position Size", "AI Score", "AI Reasoning"
]

if not df_results.empty:
    # Show all that pass the screener (sorted)
    st.subheader("All Screener Results")
    st.dataframe(df_results[trade_cols].sort_values("AI Score", ascending=False).reset_index(drop=True), use_container_width=True)
    
    # AI Top 5 Picks
    df_ai_top = df_results.sort_values("AI Score", ascending=False).head(5).copy()
    st.subheader("⭐ Top 5 AI Picks of the Day")
    st.dataframe(df_ai_top[trade_cols].reset_index(drop=True), use_container_width=True)
else:
    st.warning("⚠️ No stocks meet your screener criteria. Try relaxing the filter settings.")

# --- Alerts (for Top 5 only, to avoid spam) ---
if 'alerted_tickers' not in st.session_state:
    st.session_state['alerted_tickers'] = set()
alerted_tickers = st.session_state['alerted_tickers']

for _, row in df_results.sort_values("AI Score", ascending=False).head(5).iterrows():
    section_name = "AI Top 5"
    ticker = row["Ticker"]
    if not was_alerted_today(ticker, section_name):
        msg = (
            f"📈 {section_name} ALERT!\n"
            f"Ticker: {ticker}\n"
            f"Price: ${row['Price']} | Target: ${row['Target Price']} | Cut Loss: ${row['Cut Loss Price']}\n"
            f"Support: {row['Support (S1)']} | Resistance: {row['Resistance (R1)']}\n"
            f"Reason: {row['AI Reasoning']}"
        )
        send_telegram_alert(msg)
        add_to_alert_log(ticker, section_name)
        alerted_tickers.add((ticker, section_name))
st.session_state['alerted_tickers'] = alerted_tickers

# --- END ---
