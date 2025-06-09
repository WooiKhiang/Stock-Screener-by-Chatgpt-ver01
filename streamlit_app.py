import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
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
    min_avg_vol = st.number_input("Min Avg Volume", value=100_000)  # Default 100,000
    min_index_change = st.number_input("Min Index Change (%)", value=0.05)
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

def scan_stock_all(
    ticker, spy_change,
    min_price, max_price, min_avg_vol,
    ema200_lookback, pullback_lookback,
    capital, take_profit_pct, cut_loss_pct
):
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

    notes = []
    score = 0
    conf = 50  # start at 50%

    # EMA200 Crossover
    try:
        ema200_series = today["Close"].ewm(span=200).mean()
        crossed = False
        for i in range(1, min(ema200_lookback, len(today)-1)):
            if today["Close"].iloc[-i-1] < ema200_series.iloc[-i-1] and today["Close"].iloc[-i] > ema200_series.iloc[-i]:
                crossed = True
                break
        if crossed:
            score += 25
            conf += 8
            notes.append("EMA200 Cross ↑")
    except Exception:
        crossed = False

    # VWAP
    try:
        vwap = (today["Volume"] * (today["High"] + today["Low"] + today["Close"]) / 3).cumsum() / today["Volume"].cumsum()
        vwap_signal = today_close > float(vwap.iloc[-1])
        if vwap_signal:
            score += 20
            conf += 7
            notes.append("VWAP Above")
    except Exception:
        vwap_signal = False

    # MACD
    try:
        exp12 = today["Close"].ewm(span=12).mean()
        exp26 = today["Close"].ewm(span=26).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9).mean()
        macd_bull = float(macd.iloc[-1]) > float(signal.iloc[-1])
        if macd_bull:
            score += 15
            conf += 6
            notes.append("MACD Bullish")
    except Exception:
        macd_bull = False

    # RSI
    try:
        delta = today["Close"].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        roll_up = up.rolling(14).mean()
        roll_down = down.rolling(14).mean()
        rs = roll_up / (roll_down + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        if rsi.iloc[-1] > 70:
            conf -= 6
            notes.append("RSI Overbought")
        elif rsi.iloc[-1] < 30:
            score += 5
            conf += 5
            notes.append("RSI Oversold")
    except Exception:
        pass

    # ATR
    try:
        high = today["High"]
        low = today["Low"]
        close = today["Close"]
        tr = pd.concat([
            (high - low),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        notes.append(f"ATR: {atr:.2f}")
    except Exception:
        atr = 0

    # High Liquidity (Order Flow)
    try:
        avg5m_vol = today["Volume"].rolling(10).mean()
        recent_bars = today.tail(12)
        flow_spike = (recent_bars["Volume"] > 2 * avg5m_vol).any()
        if flow_spike:
            score += 15
            conf += 5
            notes.append("Order Flow Spike")
    except Exception:
        flow_spike = False

    # 10-bar High/Low Breakout
    try:
        high_10 = today["High"].rolling(10).max()
        low_10 = today["Low"].rolling(10).min()
        if today_close >= high_10.iloc[-1]:
            score += 15
            conf += 6
            notes.append("Breakout: 10-bar High")
        elif today_close <= low_10.iloc[-1]:
            score -= 10
            conf -= 8
            notes.append("Breakdown: 10-bar Low")
    except Exception:
        pass

    # Range Contraction/Expansion
    try:
        range_10 = today["High"].rolling(10).max() - today["Low"].rolling(10).min()
        range_20 = today["High"].rolling(20).max() - today["Low"].rolling(20).min()
        if range_10.iloc[-1] < 0.7 * range_20.iloc[-1]:
            notes.append("Range Contraction")
        elif range_10.iloc[-1] > 1.5 * range_20.iloc[-1]:
            notes.append("Range Expansion")
    except Exception:
        pass

    # Relative Strength
    if rs_score > 0:
        score += 10
        conf += 5
        notes.append("Strong RS")

    # Trade Management
    target_price = price * (1 + take_profit_pct)
    cut_loss_price = price * (1 - cut_loss_pct)
    position_size = int(capital / price) if price > 0 else 0

    # Risk:Reward
    risk_reward_ratio = (take_profit_pct / cut_loss_pct) if cut_loss_pct > 0 else np.nan

    # Clip for 0-100 range
    score = min(max(score, 0), 100)
    conf = min(max(conf, 0), 100)

    # Classification
    is_go = rs_score > 1 and vwap_signal and crossed and volume > avg_vol
    is_nogo = rs_score < -1 or not vwap_signal or not crossed
    is_ema200 = crossed
    is_vwap = vwap_signal
    is_institutional = rs_score > 0 and volume > avg_vol

    return {
        "Ticker": ticker,
        "Price": f"{price:.2f}",
        "Change %": f"{today_change:.2f}",
        "RS Score": f"{rs_score:.2f}",
        "Volume": f"{volume:,.0f}",
        "Avg Volume": f"{avg_vol:,.0f}",
        "Target Price": f"{target_price:.2f}",
        "Cut Loss Price": f"{cut_loss_price:.2f}",
        "Position Size": position_size,
        "Risk:Reward": f"{risk_reward_ratio:.2f}" if not np.isnan(risk_reward_ratio) else "-",
        "AI Score": score,
        "Confidence Level": f"{conf:.0f}%",
        "AI Reasoning": ", ".join(notes) if notes else "-",
        "GO": is_go,
        "NO-GO": is_nogo,
        "EMA200": is_ema200,
        "VWAP": is_vwap,
        "Institutional": is_institutional
    }

# --- Market Sentiment Analysis ---
spy = get_intraday_data('SPY')
qqq = get_intraday_data('QQQ')
go_day = False

if spy is not None and not spy.empty and qqq is not None and not qqq.empty:
    spy_today = spy[spy.index.date == spy.index[-1].date()]
    qqq_today = qqq[qqq.index.date == qqq.index[-1].date()]
    if len(spy_today) > 10 and len(qqq_today) > 10:
        try:
            spy_change = float((spy_today["Close"].iloc[-1] - spy_today["Open"].iloc[0]) / spy_today["Open"].iloc[0] * 100)
            qqq_change = float((qqq_today["Close"].iloc[-1] - qqq_today["Open"].iloc[0]) / qqq_today["Open"].iloc[0] * 100)
        except Exception:
            spy_change = qqq_change = 0.0
        try:
            st.subheader("Market Sentiment Panel")
            st.markdown(f"**SPY:** {spy_change:.2f}% | **QQQ:** {qqq_change:.2f}%")
            go_day = spy_change > float(min_index_change) and qqq_change > float(min_index_change)
            st.markdown("**Trend:** " + ("🟢 GO Day – Favorable" if go_day else "🔴 NO-GO Day – Defensive"))
        except Exception:
            st.markdown("**SPY/QQQ Data Error**")
    else:
        st.warning("Could not load enough recent market data. Try again later.")
        spy_change = qqq_change = 0.0
else:
    st.warning("Could not load recent market data. Try again later.")
    spy_change = qqq_change = 0.0

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

trade_cols = [
    "Ticker", "Price", "Change %", "RS Score", "Volume", "Avg Volume",
    "Target Price", "Cut Loss Price", "Position Size", "Risk:Reward",
    "AI Score", "Confidence Level", "AI Reasoning"
]

# --- Section Tables ---
def show_section(title, filter_col):
    st.subheader(title)
    if not df_results.empty:
        df = df_results[df_results[filter_col] == True]
        if not df.empty:
            st.dataframe(df[trade_cols].sort_values("AI Score", ascending=False).reset_index(drop=True), use_container_width=True)
        else:
            st.info(f"No stocks meet {title.lower()} criteria.")
    else:
        st.info("No data for stock screening.")

show_section("Go Day Stock Recommendations (Momentum/Breakout)", "GO")
show_section("No-Go Day Stock Recommendations (Defensive)", "NO-GO")
show_section("EMA200 Crossover Screener", "EMA200")
show_section("VWAP Above Screener", "VWAP")
show_section("Potential Institutional Accumulation (RS > 0 & High Volume)", "Institutional")

# --- AI Top 5 Picks ---
if not df_results.empty:
    df_ai_top = df_results.sort_values("AI Score", ascending=False).head(5).copy()
    st.subheader("⭐ Top 5 AI Picks of the Day")
    st.dataframe(df_ai_top[trade_cols].reset_index(drop=True), use_container_width=True)
else:
    st.warning("⚠️ No stocks meet your screener criteria. Try relaxing the filter settings.")

# --- Alerts (for Top 5 only) ---
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
            f"Risk:Reward {row['Risk:Reward']} | AI Score: {row['AI Score']} | Conf: {row['Confidence Level']}\n"
            f"Reason: {row['AI Reasoning']}"
        )
        send_telegram_alert(msg)
        add_to_alert_log(ticker, section_name)
        alerted_tickers.add((ticker, section_name))
st.session_state['alerted_tickers'] = alerted_tickers

# --- Win Rate & RR Backtest Panel ---
def simulate_intraday_trades(ticker, days=10, take_profit_pct=0.01, cut_loss_pct=0.005):
    df = yf.download(ticker, period=f"{days+2}d", interval="5m", progress=False)
    if df.empty:
        return None
    df['Date'] = df.index.date
    grouped = df.groupby('Date')
    results = []
    for d, group in grouped:
        if len(group) < 20:
            continue
        open_price = group['Open'].iloc[0]
        tp = open_price * (1 + take_profit_pct)
        sl = open_price * (1 - cut_loss_pct)
        hit_tp = (group['High'] >= tp).any()
        hit_sl = (group['Low'] <= sl).any()
        win = None
        if hit_tp and hit_sl:
            tp_idx = group[group['High'] >= tp].index[0]
            sl_idx = group[group['Low'] <= sl].index[0]
            win = tp_idx < sl_idx
        elif hit_tp:
            win = True
        elif hit_sl:
            win = False
        else:
            win = group['Close'].iloc[-1] > open_price
        gain = take_profit_pct if win else -cut_loss_pct
        results.append({"Date": d, "Win": int(win), "Gain": gain})
    return results

with st.expander("🧪 Quick Win Rate & RR Estimator"):
    ticker_for_test = st.text_input("Ticker to Backtest", value="AAPL")
    backtest_days = st.number_input("Days to Backtest", value=10, min_value=2, max_value=30)
    backtest_tp = st.number_input("Take Profit (%)", value=1.0, min_value=0.2, max_value=5.0, step=0.1) / 100
    backtest_sl = st.number_input("Cut Loss (%)", value=0.5, min_value=0.1, max_value=3.0, step=0.1) / 100
    if st.button("Run Backtest"):
        st.write(f"Running backtest on {ticker_for_test} for {backtest_days} days...")
        bt = simulate_intraday_trades(
            ticker_for_test,
            days=int(backtest_days),
            take_profit_pct=backtest_tp,
            cut_loss_pct=backtest_sl
        )
        if bt:
            df_bt = pd.DataFrame(bt)
            win_rate = df_bt["Win"].mean() * 100
            rr = backtest_tp / backtest_sl if backtest_sl != 0 else np.nan
            avg_return = df_bt["Gain"].mean() * 100
            st.markdown(f"**Win Rate:** {win_rate:.2f}%")
            st.markdown(f"**Risk/Reward Ratio:** {rr:.2f}")
            st.markdown(f"**Average Return per Trade:** {avg_return:.2f}%")
            st.dataframe(df_bt)
        else:
            st.warning("Not enough data to simulate.")
