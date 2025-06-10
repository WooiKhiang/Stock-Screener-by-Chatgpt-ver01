import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date
import requests

# --- SETTINGS ---
TELEGRAM_BOT_TOKEN = "xxx"
TELEGRAM_CHAT_ID = "xxx"
ALERT_LOG = "alerts_log.csv"

st.set_page_config(page_title="US Market Day Trading Screener", layout="wide")
st.title("US Market Go/No-Go Dashboard")

# --- Refresh Button & Timestamp ---
if st.button("🔄 Refresh Data Now"):
    st.experimental_rerun()
st.caption(f"Last data refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- Sidebar: Screener Criteria ---
with st.sidebar.expander("Screener Criteria", expanded=True):
    min_price = st.number_input("Min Price ($)", value=5.0)
    max_price = st.number_input("Max Price ($)", value=500.0)
    min_avg_vol = st.number_input("Min Avg Volume", value=100_000)
    min_index_change = st.number_input("Min Index Change (%)", value=0.05)
    ema200_lookback = st.number_input("EMA200 Breakout Lookback", value=6, min_value=1, max_value=48)
    pullback_lookback = st.number_input("VWAP/EMA Pullback Lookback", value=6, min_value=2, max_value=20)

with st.sidebar.expander("Profit & Risk Planner", expanded=True):
    capital = st.number_input("Capital per Trade ($)", value=1000.0, step=100.0)
    take_profit_pct = st.number_input("Take Profit (%)", value=2.0, min_value=0.5, max_value=10.0, step=0.1) / 100
    cut_loss_pct = st.number_input("Cut Loss (%)", value=1.0, min_value=0.2, max_value=5.0, step=0.1) / 100
    commission_per_trade = st.number_input("Commission per trade ($)", value=1.0, step=0.1)
    slippage_pct = st.number_input("Slippage (%)", value=0.02, step=0.01) / 100

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

def has_column(df, col):
    return (not df.empty) and (col in df.columns)

def estimate_net_profit(price, shares, take_profit_pct, commission_per_trade, slippage_pct):
    gross_profit = price * take_profit_pct * shares
    est_fees = commission_per_trade * 2
    est_slippage = price * shares * slippage_pct * 2
    net_profit = gross_profit - est_fees - est_slippage
    return net_profit, est_fees, est_slippage

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

def dynamic_ai_score(today, spy_change, ema200_lookback):
    ema200_series = today["Close"].ewm(span=200).mean()
    close = today["Close"].iloc[-1]
    ema200 = ema200_series.iloc[-1]
    ema_diff = (close - ema200) / ema200 * 100
    if ema_diff > 2:
        ema_score = 20
        ema_label = "Strong EMA200 Breakout"
    elif ema_diff > 1:
        ema_score = 12
        ema_label = "Clean EMA200 Cross"
    elif ema_diff > 0:
        ema_score = 5
        ema_label = "Mild EMA200 Cross"
    else:
        ema_score = 0
        ema_label = None

    vwap = (today["Volume"] * (today["High"] + today["Low"] + today["Close"]) / 3).cumsum() / today["Volume"].cumsum()
    vwap_diff = (close - vwap.iloc[-1]) / vwap.iloc[-1] * 100
    if vwap_diff > 1:
        vwap_score = 10
        vwap_label = "Strong VWAP Above"
    elif vwap_diff > 0.2:
        vwap_score = 7
        vwap_label = "VWAP Above"
    elif vwap_diff > 0:
        vwap_score = 3
        vwap_label = "VWAP Just Above"
    else:
        vwap_score = 0
        vwap_label = None

    exp12 = today["Close"].ewm(span=12).mean()
    exp26 = today["Close"].ewm(span=26).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9).mean()
    macd_spread = macd.iloc[-1] - signal.iloc[-1]
    if macd_spread > 0.5:
        macd_score = 8
        macd_label = "Strong MACD Bullish"
    elif macd_spread > 0.1:
        macd_score = 4
        macd_label = "MACD Bullish"
    elif macd_spread > 0:
        macd_score = 2
        macd_label = "Weak MACD Bullish"
    else:
        macd_score = 0
        macd_label = None

    today_open = today["Open"].iloc[0]
    today_close = today["Close"].iloc[-1]
    today_change = (today_close - today_open) / today_open * 100
    rs_score = today_change - spy_change
    if rs_score > 2:
        rs_points = 12
        rs_label = "Very Strong RS"
    elif rs_score > 1:
        rs_points = 8
        rs_label = "Strong RS"
    elif rs_score > 0:
        rs_points = 4
        rs_label = "RS > 0"
    else:
        rs_points = 0
        rs_label = None

    avg_vol = today["Volume"].rolling(10).mean().iloc[-1]
    last_vol = today["Volume"].iloc[-1]
    vol_ratio = last_vol / avg_vol if avg_vol > 0 else 0
    if vol_ratio > 3:
        vol_score = 15
        vol_label = "3x Volume Spike"
    elif vol_ratio > 2:
        vol_score = 8
        vol_label = "2x Volume Spike"
    elif vol_ratio > 1.3:
        vol_score = 4
        vol_label = "Volume Above Average"
    else:
        vol_score = 0
        vol_label = None

    high_10 = today["High"].rolling(10).max().iloc[-1]
    if close >= high_10 and vol_ratio > 2:
        breakout_score = 15
        breakout_label = "Explosive 10-bar Breakout"
    elif close >= high_10:
        breakout_score = 5
        breakout_label = "10-bar High Breakout"
    else:
        breakout_score = 0
        breakout_label = None

    range_10 = today["High"].rolling(10).max() - today["Low"].rolling(10).min()
    range_20 = today["High"].rolling(20).max() - today["Low"].rolling(20).min()
    if range_10 > 1.5 * range_20:
        atr_score = 5
        atr_label = "Large ATR Expansion"
    elif range_10 > range_20:
        atr_score = 2
        atr_label = "ATR Expanding"
    else:
        atr_score = 0
        atr_label = None

    spy_down = spy_change < 0
    defensive_score = 12 if (spy_down and rs_score > 2) else 0
    defensive_label = "Defensive Play" if defensive_score else None

    delta = today["Close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs_rsi = roll_up / (roll_down + 1e-9)
    rsi = 100 - (100 / (1 + rs_rsi))
    rsi_note = None
    if rsi.iloc[-1] > 70:
        rsi_note = "RSI Overbought"
    elif rsi.iloc[-1] < 30:
        rsi_note = "RSI Oversold"

    ai_score = min(ema_score + vwap_score + macd_score + rs_points +
                   vol_score + breakout_score + atr_score + defensive_score, 100)
    signals = [ema_label, vwap_label, macd_label, rs_label, vol_label, breakout_label, atr_label, defensive_label]
    notes = [s for s in signals if s] + ([rsi_note] if rsi_note else [])

    all_scores = dict(
        ema_score=ema_score, vwap_score=vwap_score, macd_score=macd_score,
        rs_points=rs_points, vol_score=vol_score, breakout_score=breakout_score,
        atr_score=atr_score, defensive_score=defensive_score
    )
    return ai_score, notes, all_scores, rs_score, rsi.iloc[-1], avg_vol, last_vol

def calc_confidence(all_scores, go_day, avg_vol, last_vol, rsi_val):
    conf = 30
    for k in ["ema_score", "vwap_score", "rs_points", "vol_score"]:
        if all_scores[k] >= 8:
            conf += 8
        elif all_scores[k] >= 4:
            conf += 4
    for k in ["macd_score", "breakout_score", "atr_score"]:
        if all_scores[k] >= 5:
            conf += 4
        elif all_scores[k] > 0:
            conf += 2
    if not go_day and all_scores["defensive_score"] == 0:
        conf -= 8
    if last_vol < avg_vol:
        conf -= 10
    if rsi_val > 75:
        conf -= 6
    conf = min(max(conf, 0), 100)
    return conf

def scan_stock_all(
    ticker, spy_change, min_price, max_price, min_avg_vol,
    ema200_lookback, pullback_lookback, capital, take_profit_pct, cut_loss_pct, go_day
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

    ai_score, notes, all_scores, rs_score, rsi_val, avg_vol, last_vol = dynamic_ai_score(today, spy_change, ema200_lookback)
    conf = calc_confidence(all_scores, go_day, avg_vol, last_vol, rsi_val)

    today_change = (today_close - today_open) / today_open * 100
    target_price = price * (1 + take_profit_pct)
    cut_loss_price = price * (1 - cut_loss_pct)
    position_size = int(capital / price) if price > 0 else 0
    risk_reward_ratio = (take_profit_pct / cut_loss_pct) if cut_loss_pct > 0 else np.nan

    net_profit, est_fees, est_slippage = estimate_net_profit(
        price, position_size, take_profit_pct, commission_per_trade, slippage_pct)

    signal_strong_count = sum([all_scores[k] >= 8 for k in ["ema_score", "vwap_score", "rs_points", "vol_score"]])
    is_aplus = (signal_strong_count >= 3) and (ai_score >= 70) and go_day
    is_defensive = (rs_score > 1) and (today_change > 0) and (not go_day)
    is_go = is_aplus
    is_nogo = is_defensive

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
        "AI Score": ai_score,
        "Confidence Level": f"{conf:.0f}%",
        "AI Reasoning": ", ".join(notes) if notes else "-",
        "A+": is_aplus,
        "Defensive": is_defensive,
        "GO": is_go,
        "NO-GO": is_nogo,
        "Estimated Net Profit": f"{net_profit:.2f}",
        "Estimated Fees": f"{est_fees:.2f}",
        "Estimated Slippage": f"{est_slippage:.2f}"
    }

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

results = []
for ticker in watchlist:
    result = scan_stock_all(
        ticker, spy_change,
        min_price, max_price, min_avg_vol,
        ema200_lookback, pullback_lookback,
        capital, take_profit_pct, cut_loss_pct, go_day
    )
    if result:
        results.append(result)
df_results = pd.DataFrame(results) if results else pd.DataFrame()

trade_cols = [
    "Ticker", "Price", "Change %", "RS Score", "Volume", "Avg Volume",
    "Target Price", "Cut Loss Price", "Position Size", "Risk:Reward",
    "AI Score", "Confidence Level", "AI Reasoning",
    "Estimated Net Profit", "Estimated Fees", "Estimated Slippage"
]

def show_section(title, filter_col):
    st.subheader(title)
    if has_column(df_results, filter_col):
        df = df_results[df_results[filter_col] == True]
        if not df.empty:
            st.dataframe(df[trade_cols].sort_values("AI Score", ascending=False).reset_index(drop=True), use_container_width=True)
        else:
            st.info(f"No stocks meet {title.lower()} criteria.")
    else:
        st.info("No data for stock screening.")

show_section("🔥 A+ Setups (High Confluence, Go Day)", "A+")
show_section("🛡️ Defensive (Anti-Market) Picks (No-Go Day)", "Defensive")
show_section("Go Day Stock Recommendations (Momentum/Breakout)", "GO")
show_section("No-Go Day Stock Recommendations (Defensive)", "NO-GO")

if has_column(df_results, "A+"):
    df_ai_top = df_results[df_results["A+"] == True].sort_values(
        ["AI Score", "Confidence Level", "Risk:Reward"], ascending=False
    ).head(5)
    st.subheader("⭐ Top 5 AI Picks of the Day (A+ Net Profit After Fees)")
    st.dataframe(df_ai_top[trade_cols].reset_index(drop=True), use_container_width=True)
else:
    st.warning("⚠️ No stocks meet your screener criteria. Try relaxing the filter settings.")

if 'alerted_tickers' not in st.session_state:
    st.session_state['alerted_tickers'] = set()
alerted_tickers = st.session_state['alerted_tickers']

if has_column(df_results, "A+"):
    for _, row in df_results[df_results["A+"] == True].sort_values("AI Score", ascending=False).head(5).iterrows():
        section_name = "AI Top 5"
        ticker = row["Ticker"]
        if not was_alerted_today(ticker, section_name):
            msg = (
                f"📈 {section_name} ALERT!\n"
                f"Ticker: {ticker}\n"
                f"Price: ${row['Price']} | Target: ${row['Target Price']} | Cut Loss: ${row['Cut Loss Price']}\n"
                f"Risk:Reward {row['Risk:Reward']} | AI Score: {row['AI Score']} | Conf: {row['Confidence Level']}\n"
                f"Net Profit: ${row['Estimated Net Profit']} (after fees/slippage)\n"
                f"Reason: {row['AI Reasoning']}"
            )
            send_telegram_alert(msg)
            add_to_alert_log(ticker, section_name)
            alerted_tickers.add((ticker, section_name))
st.session_state['alerted_tickers'] = alerted_tickers

# Your backtest tool (unchanged)
