import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
import os
import csv
from datetime import date
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import numpy as np

st.set_page_config(page_title="US Market Day Trading Screener", layout="wide")
st.title("US Market Go/No-Go Dashboard")

# ---- Telegram Alert Settings ----
TELEGRAM_BOT_TOKEN = "7280991990:AAEk5x4XFCW_sTohAQGUujy1ECAQHjSY_OU"
TELEGRAM_CHAT_ID = "713264762"

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        st.warning(f"Failed to send Telegram alert: {e}")

# ------------------- SIDEBAR -------------------
# -- Screener Criteria Group --
with st.sidebar.expander("🔎 Screener Criteria", expanded=True):
    min_price = st.number_input("Min Price ($)", value=5.0)
    max_price = st.number_input("Max Price ($)", value=500.0)
    min_avg_vol = st.number_input("Min Average Volume", value=1_000_000)
    min_index_change = st.number_input("Min Index Change (%)", value=0.05)
    max_atr_percent = st.number_input("Max ATR (%)", value=0.015)
    volume_factor = st.number_input("Min Volume Factor", value=0.7)
    ema200_lookback = st.number_input("EMA200 Breakout Lookback Bars", value=6, min_value=1, max_value=48)
    pullback_lookback = st.number_input("VWAP/EMA Pullback Lookback Bars", value=6, min_value=2, max_value=20)

# -- Profit & Risk Planner Group --
with st.sidebar.expander("💰 Profit & Risk Planner", expanded=True):
    capital = st.number_input("Capital per Trade ($)", value=1000.0, step=100.0)
    take_profit_pct = st.number_input("Take Profit (%)", value=2.0, min_value=0.5, max_value=10.0, step=0.1) / 100
    cut_loss_pct = st.number_input("Cut Loss (%)", value=1.0, min_value=0.2, max_value=5.0, step=0.1) / 100

# -------------- S&P 100 Watchlist --------------
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

# -------------- Data Fetching --------------
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

    # --- Trade management values ---
    target_price = price * (1 + take_profit_pct)
    cut_loss_price = price * (1 - cut_loss_pct)
    risk_per_share = price - cut_loss_price
    position_size = int(capital / price) if price > 0 else 0
    max_loss = position_size * risk_per_share

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
        "Target Price": f"{target_price:.2f}",
        "Cut Loss Price": f"{cut_loss_price:.2f}",
        "Position Size": position_size,
        "Max Loss at Stop": f"{max_loss:.2f}",
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
        ema200_lookback, pullback_lookback,
        capital, take_profit_pct, cut_loss_pct
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

common_trade_cols = ["Target Price", "Cut Loss Price", "Position Size", "Max Loss at Stop"]

show_section(
    "Go Day Stock Recommendations (Momentum/Breakout)",
    "GO",
    [
        "Ticker", "Price", "Open", "Close", "High", "Low", "Change %", "Volume", "Avg Volume", "Rel Vol",
        "RS vs SPY", "VWAP", "EMA 10", "EMA 20", "EMA 50", "MACD Bullish", "Reasons"
    ] + common_trade_cols
)

show_section(
    "No-Go Day Stock Recommendations (Defensive/Resilient)",
    "NO-GO",
    [
        "Ticker", "Price", "Open", "Close", "High", "Low", "Change %", "Volume", "Avg Volume", "Rel Vol",
        "RS vs SPY", "EMA 50", "Reasons"
    ] + common_trade_cols
)

show_section(
    "Potential Institutional Accumulation Screener",
    "Accumulation",
    [
        "Ticker", "Price", "Open", "Close", "High", "Low", "Change %", "Volume", "Avg Volume", "Rel Vol",
        "RS vs SPY", "VWAP", "EMA 10", "EMA 20", "Reasons"
    ] + common_trade_cols
)

show_section(
    "EMA200 Breakout Reversal Screener",
    "EMA200 Breakout",
    [
        "Ticker", "Price", "Open", "Close", "High", "Low", "Change %", "Volume", "Avg Volume", "Rel Vol",
        "RS vs SPY", "EMA 200", "VWAP", "Reasons"
    ] + common_trade_cols
)

show_section(
    "VWAP/EMA20 Pullback Bounce Screener",
    "Pullback Bounce",
    [
        "Ticker", "Price", "Open", "Close", "High", "Low", "Change %", "Volume", "Avg Volume", "Rel Vol",
        "RS vs SPY", "VWAP", "EMA 20", "Reasons"
    ] + common_trade_cols
)

# ---------------- BACKTEST MODULE ----------------

st.sidebar.markdown("---")
with st.sidebar.expander("🧪 Backtest Module", expanded=False):
    run_backtest = st.checkbox("Run GO Day Backtest", value=False)
    lookback_days = st.number_input("Backtest Days", value=10, min_value=5, max_value=60)
    backtest_tp = st.number_input("Backtest Take Profit (%)", value=2.0, step=0.1) / 100
    backtest_sl = st.number_input("Backtest Stop Loss (%)", value=1.0, step=0.1) / 100

if run_backtest:
    st.subheader("Backtest Results (GO Day Strategy)")
    backtest_results = []
    for ticker in watchlist:
        df = yf.download(ticker, period=f"{lookback_days+2}d", interval="5m", progress=False)
        if df.empty or len(df) < 100:
            continue
        df['Date'] = df.index.date
        for date in sorted(df['Date'].unique())[-lookback_days:]:
            day = df[df['Date'] == date]
            if len(day) < 10:
                continue

            day['EMA10'] = day['Close'].ewm(span=10).mean()
            day['EMA20'] = day['Close'].ewm(span=20).mean()
            day['EMA50'] = day['Close'].ewm(span=50).mean()
            exp12 = day['Close'].ewm(span=12).mean()
            exp26 = day['Close'].ewm(span=26).mean()
            day['MACD'] = exp12 - exp26
            day['Signal'] = day['MACD'].ewm(span=9).mean()

            last = day.iloc[-1]
            go_signal = (
                last['Close'] > last['EMA10'] and
                last['Close'] > last['EMA20'] and
                last['Close'] > last['EMA50'] and
                last['MACD'] > last['Signal']
            )
            if not go_signal:
                continue

            entry = float(last['Close'])
            tp = entry * (1 + backtest_tp)
            sl = entry * (1 - backtest_sl)
            exit_price = entry
            hit_tp = hit_sl = False

            # Simulate TP/SL for the next X bars (first 5 bars of next day)
            next_days = sorted(df['Date'].unique())
            next_idx = next_days.index(date) + 1
            if next_idx < len(next_days):
                next_day = df[df['Date'] == next_days[next_idx]]
                for _, bar in next_day.iloc[:5].iterrows():
                    if bar['High'] >= tp:
                        exit_price = tp
                        hit_tp = True
                        break
                    if bar['Low'] <= sl:
                        exit_price = sl
                        hit_sl = True
                        break
                    exit_price = bar['Close']

            pnl = exit_price - entry
            pnl_pct = pnl / entry * 100

            backtest_results.append({
                "Ticker": ticker,
                "Date": date,
                "Entry": f"{entry:.2f}",
                "Exit": f"{exit_price:.2f}",
                "PnL %": f"{pnl_pct:.2f}",
                "TP?": "✅" if hit_tp else "",
                "SL?": "✅" if hit_sl else "",
            })

    if backtest_results:
        df_bt = pd.DataFrame(backtest_results)
        st.dataframe(df_bt)
        wins = df_bt[df_bt['PnL %'].astype(float) > 0]
        losses = df_bt[df_bt['PnL %'].astype(float) <= 0]
        st.markdown(f"**Total Trades:** {len(df_bt)}")
        st.markdown(f"**Win Rate:** {len(wins) / len(df_bt) * 100:.1f}%")
        st.markdown(f"**Avg Win:** {wins['PnL %'].astype(float).mean():.2f}%")
        st.markdown(f"**Avg Loss:** {losses['PnL %'].astype(float).mean():.2f}%")
    else:
        st.warning("No GO Day signals found in backtest window.")

# --- ALERTS ---
# Only send alerts for NEW screener hits each run (in-memory)
if 'alerted_tickers' not in st.session_state:
    st.session_state['alerted_tickers'] = set()
alerted_tickers = st.session_state['alerted_tickers']

def alert_for_section(section_name, filter_col):
    if not df_results.empty:
        new_hits = df_results[df_results[filter_col] == True]
        for _, row in new_hits.iterrows():
            ticker = row["Ticker"]
            # Only alert if not alerted today
            if not was_alerted_today(ticker, section_name):
                alert_msg = (
                    f"📈 {section_name} ALERT!\n"
                    f"Ticker: {ticker}\n"
                    f"Price: ${row['Price']} | Target: ${row['Target Price']} | Cut Loss: ${row['Cut Loss Price']}\n"
                    f"Position: {row['Position Size']} shares | Max Loss: ${row['Max Loss at Stop']}\n"
                    f"Reasons: {row['Reasons']}"
                )
                send_telegram_alert(alert_msg)
                add_to_alert_log(ticker, section_name)
                log_to_google_sheet(row, section_name)

def log_to_google_sheet(row_dict, section_name):
    # Sheet setup
    SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    SHEET_ID = "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4"
    SHEET_NAME = "Sheet1"  # Change if you rename your worksheet

    # Get credentials
    if "gcp_service_account" in st.secrets:
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, SCOPE)
    else:
        creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", SCOPE)

    gc = gspread.authorize(creds)
    worksheet = gc.open_by_key(SHEET_ID).worksheet(SHEET_NAME)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_row = [
        now,
        section_name,
        row_dict.get("Ticker"),
        row_dict.get("Price"),
        row_dict.get("Target Price"),
        row_dict.get("Cut Loss Price"),
        row_dict.get("Position Size"),
        row_dict.get("Max Loss at Stop"),
        row_dict.get("Reasons"),
    ]
    worksheet.append_row(log_row, value_input_option="USER_ENTERED")

alert_for_section("GO Day", "GO")
alert_for_section("No-Go Day", "NO-GO")
alert_for_section("Institutional Accumulation", "Accumulation")
alert_for_section("EMA200 Breakout", "EMA200 Breakout")
alert_for_section("VWAP/EMA Pullback", "Pullback Bounce")

st.session_state['alerted_tickers'] = alerted_tickers

ALERT_LOG = "alerts_log.csv"

def was_alerted_today(ticker, section_name):
    today_str = str(date.today())
    if os.path.exists(ALERT_LOG):
        with open(ALERT_LOG, "r") as f:
            for row in csv.reader(f):
                if len(row) == 3 and row[0] == today_str and row[1] == section_name and row[2] == ticker:
                    return True
    return False

def add_to_alert_log(ticker, section_name):
    today_str = str(date.today())
    with open(ALERT_LOG, "a", newline="") as f:
        csv.writer(f).writerow([today_str, section_name, ticker])
