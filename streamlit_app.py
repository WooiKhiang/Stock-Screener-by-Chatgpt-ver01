import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, date
import numpy as np
import requests
import os
import csv
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ----- CONFIG -----
st.set_page_config(page_title="US Market Go/No-Go Dashboard", layout="wide")
st.title("US Market Go/No-Go Dashboard")

TELEGRAM_BOT_TOKEN = "7280991990:AAEk5x4XFCW_sTohAQGUujy1ECAQHjSY_OU"
TELEGRAM_CHAT_ID = "713264762"
GOOGLE_SHEET_ID = "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4"
GOOGLE_SHEET_NAME = "Sheet1"
ALERT_LOG = "alerts_log.csv"

# ---- Sidebar: Screener Criteria ----
with st.sidebar.expander("🔎 Screener Criteria", expanded=True):
    min_price = st.number_input("Min Price ($)", value=5.0)
    max_price = st.number_input("Max Price ($)", value=500.0)
    min_avg_vol = st.number_input("Min Avg Volume", value=1_000_000)
    min_index_change = st.number_input("Min Index Change (%)", value=0.05)
    max_atr_percent = st.number_input("Max ATR (%)", value=0.015)
    volume_factor = st.number_input("Min Volume Factor", value=0.7)
    ema200_lookback = st.number_input("EMA200 Breakout Lookback Bars", value=6, min_value=1, max_value=48)
    pullback_lookback = st.number_input("VWAP/EMA Pullback Bars", value=6, min_value=2, max_value=20)

with st.sidebar.expander("💰 Profit & Risk Planner", expanded=True):
    capital = st.number_input("Capital per Trade ($)", value=1000.0, step=100.0)
    take_profit_pct = st.number_input("Take Profit (%)", value=2.0, min_value=0.5, max_value=10.0, step=0.1) / 100
    cut_loss_pct = st.number_input("Cut Loss (%)", value=1.0, min_value=0.2, max_value=5.0, step=0.1) / 100

with st.sidebar.expander("🕰️ Backtest (GO Day Open Breakout)", expanded=False):
    run_backtest = st.checkbox("Run Backtest Now")
    lookback_days = st.number_input("Backtest: Days", value=10, min_value=2, max_value=30, step=1)
    backtest_tp = st.number_input("Backtest: Take Profit %", value=2.0, min_value=0.5, max_value=10.0, step=0.1) / 100
    backtest_sl = st.number_input("Backtest: Stop Loss %", value=1.0, min_value=0.2, max_value=5.0, step=0.1) / 100

watchlist = [
    "AAPL","ABBV","ABT","ACN","ADBE","AIG","AMGN","AMT","AMZN","AVGO","AXP","BA","BAC","BK","BKNG","BLK","BMY","BRK.B",
    "C","CAT","CHTR","CL","CMCSA","COF","COP","COST","CRM","CSCO","CVS","CVX","DHR","DIS","DOW","DUK","EMR","EXC","F",
    "FDX","FOX","FOXA","GD","GE","GILD","GM","GOOG","GOOGL","GS","HD","HON","IBM","INTC","JNJ","JPM","KHC","KMI","KO",
    "LIN","LLY","LMT","LOW","MA","MCD","MDLZ","MDT","MET","META","MMM","MO","MRK","MS","MSFT","NEE","NFLX","NKE","NVDA",
    "ORCL","PEP","PFE","PG","PM","PYPL","QCOM","RTX","SBUX","SCHW","SO","SPG","T","TGT","TMO","TMUS","TSLA","TXN","UNH",
    "UNP","UPS","USB","V","VZ","WBA","WFC","WMT","XOM"
]

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        st.warning(f"Failed to send Telegram alert: {e}")

def log_to_google_sheet(row_dict, section_name):
    SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    try:
        if "gcp_service_account" in st.secrets:
            creds_dict = st.secrets["gcp_service_account"]
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, SCOPE)
        else:
            creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", SCOPE)
        gc = gspread.authorize(creds)
        worksheet = gc.open_by_key(GOOGLE_SHEET_ID).worksheet(GOOGLE_SHEET_NAME)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_row = [
            now, section_name,
            row_dict.get("Ticker"),
            row_dict.get("Price"),
            row_dict.get("Target Price"),
            row_dict.get("Cut Loss Price"),
            row_dict.get("Position Size"),
            row_dict.get("Max Loss at Stop"),
            row_dict.get("Reasons"),
        ]
        worksheet.append_row(log_row, value_input_option="USER_ENTERED")
    except Exception as e:
        st.warning(f"Failed to log to Google Sheet: {e}")

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
    # --- Support & Resistance ---
    try:
        support_lvl = float(today["Low"].rolling(12).min().iloc[-1])
    except:
        support_lvl = None
    try:
        resistance_lvl = float(today["High"].rolling(12).max().iloc[-1])
    except:
        resistance_lvl = None
    # --- Predictions ---
    pred_5b = today_close + np.std(today["Close"][-5:]) * 1.5
    pred_15b = today_close + np.std(today["Close"][-15:]) * 2
    # --- Trade management values ---
    target_price = price * (1 + take_profit_pct)
    cut_loss_price = price * (1 - cut_loss_pct)
    risk_per_share = price - cut_loss_price
    position_size = int(capital / price) if price > 0 else 0
    max_loss = position_size * risk_per_share
    # --- Logic/AI Picks Score ---
    ai_score = (
        2*(price_above_ema10 and price_above_ema20) +
        1.2*(macd_bullish) +
        1.3*(price_above_vwap) +
        1.5*(ema200_breakout) +
        0.5*(pullback) +
        1.0*(volume_spike)
    )
    go_criteria = price_above_ema10 and price_above_ema20 and price_above_ema50 and macd_bullish and volume_spike and rs_score > 0
    nogo_criteria = price_above_ema50 and not (price_above_ema10 and price_above_ema20) and rs_score > -1
    accumulation_criteria = rs_score > 0 and price_above_ema10 and price_above_ema20
    reasons = []
    if rs_score > 0: reasons.append("Strong RS")
    if price_above_vwap: reasons.append("VWAP Reclaim")
    if price_above_ema10 and price_above_ema20: reasons.append("EMA Alignment")
    if volume_spike: reasons.append("Volume Spike")
    if macd_bullish: reasons.append("MACD Bullish")
    if ema200_breakout: reasons.append(f"EMA200 Breakout ({ema200_lookback} bars)")
    if pullback: reasons.append(f"VWAP/EMA20 Pullback ({pullback_lookback} bars)")
    if support_lvl: reasons.append(f"Support {support_lvl:.2f}")
    if resistance_lvl: reasons.append(f"Resistance {resistance_lvl:.2f}")
    return {
        "Ticker": ticker,
        "Price": f"{price:.2f}",
        "Change %": f"{today_change:.2f}",
        "Volume": f"{volume:,.0f}",
        "Avg Volume": f"{avg_vol:,.0f}",
        "Rel Vol": f"{rel_vol:.2f}",
        "Support": f"{support_lvl:.2f}" if support_lvl else "",
        "Resistance": f"{resistance_lvl:.2f}" if resistance_lvl else "",
        "VWAP": "Yes" if price_above_vwap else "No",
        "EMA 10": "Yes" if price_above_ema10 else "No",
        "EMA 20": "Yes" if price_above_ema20 else "No",
        "EMA 50": "Yes" if price_above_ema50 else "No",
        "EMA 200": f"{ema200:.2f}",
        "MACD Bullish": "Yes" if macd_bullish else "No",
        "Volume Spike": "Yes" if volume_spike else "No",
        "RS vs SPY": f"{rs_score:.2f}",
        "GO": go_criteria,
        "NO-GO": nogo_criteria,
        "Accumulation": accumulation_criteria,
        "EMA200 Breakout": ema200_breakout,
        "VWAP/EMA Pullback": pullback,
        "Target Price": f"{target_price:.2f}",
        "Cut Loss Price": f"{cut_loss_price:.2f}",
        "Position Size": position_size,
        "Max Loss at Stop": f"{max_loss:.2f}",
        "Pred 5 Bars": f"{pred_5b:.2f}",
        "Pred 15 Bars": f"{pred_15b:.2f}",
        "Reasons": ", ".join(reasons),
        "AI Score": ai_score,
    }

spy = get_data('SPY')
qqq = get_data('QQQ')
spy_change = qqq_change = 0

if not spy.empty and not qqq.empty:
    try:
        spy_change = (float(spy['Close'].iloc[-1]) - float(spy['Close'].iloc[-2])) / float(spy['Close'].iloc[-2]) * 100
        qqq_change = (float(qqq['Close'].iloc[-1]) - float(qqq['Close'].iloc[-2]) * 100) / float(qqq['Close'].iloc[-2])
    except Exception:
        spy_change = qqq_change = 0
    try:
        spy_volume = float(spy['Volume'].iloc[-1])
        spy_avg_volume = float(spy['Volume'].rolling(window=10).mean().iloc[-1])
        spy_atr = float((spy['High'] - spy['Low']).rolling(window=14).mean().iloc[-1])
        close_last = float(spy['Close'].iloc[-1])
        atr_percent = spy_atr / close_last if close_last else 0
        vol_factor_val = spy_volume / spy_avg_volume if spy_avg_volume else 0
    except Exception:
        atr_percent = vol_factor_val = 0
    st.subheader("Market Sentiment Analysis")
    go_day = (
        spy_change >= min_index_change and
        qqq_change >= min_index_change and
        vol_factor_val >= volume_factor and
        atr_percent <= max_atr_percent
    )
    if go_day:
        st.success("GO DAY! Risk-on, liquidity, and trend detected.")
        st.markdown("**Market Trend:** Up  \n**Liquidity:** High  \n**Big Players:** Risk-on  \n**Sentiment:** Bullish  \n**Catalyst Risk:** None  \n**Trade Signal:** Aggressive OK")
    else:
        st.error("NO-GO DAY. Consider capital protection strategies.")
        st.markdown("**Market Trend:** Flat/Down  \n**Liquidity:** Low/Avg  \n**Big Players:** Defensive  \n**Sentiment:** Mixed or Bearish  \n**Catalyst Risk:** High/Unclear  \n**Trade Signal:** Cautious/No Entry")
else:
    st.warning("Could not load recent market data. Try again later.")
    go_day = False

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

# ---- Display Screener Results ----
def show_section(title, filter_column):
    st.subheader(title)
    display_cols = [
        "Ticker","Price","Change %","Volume","Avg Volume","Rel Vol",
        "Support","Resistance","VWAP","EMA 10","EMA 20","EMA 50","EMA 200",
        "MACD Bullish","Volume Spike","RS vs SPY","Target Price","Cut Loss Price","Position Size","Max Loss at Stop",
        "Pred 5 Bars","Pred 15 Bars","Reasons"
    ]
    if not df_results.empty:
        df = df_results[df_results[filter_column] == True]
        if not df.empty:
            st.dataframe(df[display_cols])
        else:
            st.info(f"⚠️ No stocks meet {title.lower()} criteria. Try relaxing the filter settings.")
    else:
        st.info("⚠️ No stocks meet your screener criteria. Try relaxing the filter settings.")

show_section("Go Day Stock Recommendations (Momentum/Breakout)", "GO")
show_section("No-Go Day Stock Recommendations (Defensive/Resilient)", "NO-GO")
show_section("Potential Institutional Accumulation Screener", "Accumulation")
show_section("EMA200 Breakout Reversal Screener", "EMA200 Breakout")
show_section("VWAP/EMA20 Pullback Scalping Screener", "VWAP/EMA Pullback")

# ---- AI Top 5 Picks ----
if not df_results.empty:
    top_picks = df_results.sort_values(by="AI Score", ascending=False).head(5)
    st.markdown("## 🤖 Top 5 AI Picks Today")
    st.dataframe(top_picks[["Ticker", "Price", "Change %", "Reasons", "AI Score"]])
    st.markdown("**Reasoning:** Picks are ranked by a weighted logic of EMA alignment, MACD, VWAP, volume spike, EMA200 breakout, and pullback. Higher score = stronger multi-factor setup.")

# ------------- ALERTS & GOOGLE LOGGING ---------------
if 'alerted_tickers' not in st.session_state:
    st.session_state['alerted_tickers'] = set()
alerted_tickers = st.session_state['alerted_tickers']
for _, row in df_results.iterrows():
    for section_name, filter_col in [
        ("GO Day", "GO"),
        ("No-Go Day", "NO-GO"),
        ("Institutional Accumulation", "Accumulation"),
        ("EMA200 Breakout", "EMA200 Breakout"),
        ("VWAP/EMA Pullback", "VWAP/EMA Pullback")
    ]:
        if row[filter_col]:
            ticker = row["Ticker"]
            if not was_alerted_today(ticker, section_name):
                msg = (
                    f"📈 {section_name} ALERT!\n"
                    f"Ticker: {ticker}\n"
                    f"Price: ${row['Price']} | Target: ${row['Target Price']} | Cut Loss: ${row['Cut Loss Price']}\n"
                    f"Position: {row['Position Size']} shares | Max Loss: ${row['Max Loss at Stop']}\n"
                    f"Reasons: {row['Reasons']}"
                )
                send_telegram_alert(msg)
                add_to_alert_log(ticker, section_name)
                log_to_google_sheet(row, section_name)
                alerted_tickers.add((ticker, section_name))
st.session_state['alerted_tickers'] = alerted_tickers

# ---- BACKTEST MODULE ----
if run_backtest:
    st.subheader(f"Backtest Results (last {lookback_days} days, GO Day strategy)")
    bt_results = []
    for ticker in watchlist[:30]: # limit for speed
        df = yf.download(ticker, period=f"{lookback_days+2}d", interval='5m', progress=False)
        if df.empty: continue
        df['Date'] = df.index.date
        day_groups = df.groupby('Date')
        for d, group in day_groups:
            if len(group) < 20: continue
            o = group['Open'].iloc[0]
            h = group['High'].max()
            l = group['Low'].min()
            c = group['Close'].iloc[-1]
            tp = o * (1 + backtest_tp)
            sl = o * (1 - backtest_sl)
            hit_tp = (group['High'] >= tp).any()
            hit_sl = (group['Low'] <= sl).any()
            win = None
            # fix: must use bool not Series
            if bool(hit_tp) and bool(hit_sl):
                tp_idx = group[group['High'] >= tp].index[0]
                sl_idx = group[group['Low'] <= sl].index[0]
                win = tp_idx < sl_idx
            elif bool(hit_tp):
                win = True
            elif bool(hit_sl):
                win = False
            else:
                win = c > o
            bt_results.append({'Ticker': ticker, 'Date': d, 'Open': o, 'Close': c, 'TP_hit': bool(hit_tp), 'SL_hit': bool(hit_sl), 'Win': win})
    if bt_results:
        df_bt = pd.DataFrame(bt_results)
        win_rate = df_bt['Win'].mean() * 100
        st.markdown(f"**Backtest Win Rate:** {win_rate:.2f}%  (sample size: {len(df_bt)})")
        st.dataframe(df_bt.tail(100))
    else:
        st.info("No backtest results found. Try a smaller watchlist or different settings.")
