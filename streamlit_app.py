import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, time as dt_time
import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import time

# ---- Config ----
TELEGRAM_BOT_TOKEN = "7280991990:AAEk5x4XFCW_sTohAQGUujy1ECAQHjSY_OU"
TELEGRAM_CHAT_ID = "713264762"
GOOGLE_SHEET_ID = "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4"
GOOGLE_SHEET_NAME = "SnP"
KLSE_GOOGLE_SHEET_NAME = "KLSE"

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

# ---- Helper Functions ----
def formatn(num, d=2):
    try:
        if num is None or num == "" or np.isnan(num): return "-"
        if isinstance(num, int) or d == 0:
            return f"{int(num):,}"
        return f"{num:,.{d}f}"
    except Exception:
        return str(num)

def norm(df):
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
    df['ema40'] = df['close'].ewm(span=40, min_periods=40).mean()
    df['ema200'] = df['close'].ewm(span=200, min_periods=200).mean()
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
    # ATR
    high_low = df['high'] - df['low']
    high_prevclose = np.abs(df['high'] - df['close'].shift(1))
    low_prevclose = np.abs(df['low'] - df['close'].shift(1))
    ranges = pd.concat([high_low, high_prevclose, low_prevclose], axis=1)
    df['atr14'] = ranges.max(axis=1).rolling(14).mean()
    return df

def local_time_str():
    return datetime.now(pytz.timezone("Asia/Singapore")).strftime('%Y-%m-%d %H:%M:%S')

def get_market_sentiment():
    try:
        spy = yf.download('SPY', period='1d', interval='5m', progress=False, threads=False)
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

def is_defensive_sector(ticker):
    defensive = {'MRK', 'JNJ', 'PFE', 'LLY', 'ABBV', 'ABT', 'MDT', 'WMT', 'KO', 'PEP', 'PG', 'CL', 'MO', 'WBA'}
    return ticker in defensive

def is_market_open():
    eastern = pytz.timezone("US/Eastern")
    now_et = datetime.now(pytz.utc).astimezone(eastern)
    t = now_et.time()
    return ((dt_time(4,0) <= t < dt_time(20,0)))

def send_telegram_alert(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})

def get_gspread_client_from_secrets():
    info = st.secrets["gcp_service_account"]
    creds_dict = {k: v for k, v in info.items()}
    if isinstance(creds_dict["private_key"], list):
        creds_dict["private_key"] = "\n".join(creds_dict["private_key"])
    creds_json = json.dumps(creds_dict)
    scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(creds_json), scope)
    return gspread.authorize(creds)

def append_to_gsheet(rows, sheet_name):
    try:
        client = get_gspread_client_from_secrets()
        sheet = client.open_by_key(GOOGLE_SHEET_ID).worksheet(sheet_name)
        for row in rows:
            sheet.append_row(row, value_input_option="USER_ENTERED")
    except Exception as e:
        st.warning(f"Google Sheet ({sheet_name}): {e}")

# ---- Sidebar ----
st.sidebar.header("Filter Settings")
min_price = st.sidebar.number_input("Min Price ($)", value=10.0)
max_price = st.sidebar.number_input("Max Price ($)", value=2000.0)
min_vol = st.sidebar.number_input("Min Volume (last bar)", value=100000)
capital_per_trade = st.sidebar.number_input("Capital per Trade ($)", value=1000.0, step=100.0)
min_target_pct = st.sidebar.number_input("Min Target (%)", value=1.0, min_value=0.1, step=0.1) / 100
min_cutloss_pct = st.sidebar.number_input("Min Cutloss (%)", value=0.7, min_value=0.1, step=0.1) / 100
macd_stack_on = st.sidebar.checkbox("MACD Stack (Momentum)", value=True)
hybrid_on = st.sidebar.checkbox("Hybrid 5min+1h Confirm", value=True)
ema200_on = st.sidebar.checkbox("EMA200 Breakout (Swing)", value=True)
show_debug = st.sidebar.checkbox("Show Debug Info", value=False)

# ---- Market Sentiment ----
sentiment_pct, sentiment_text = get_market_sentiment()
st.title("AI-Powered US Stocks Screener: Intraday, Swing & Pre/Post-Market Hybrid")
st.caption(f"Last run: {local_time_str()}")
st.caption(f"Market Sentiment: {sentiment_text} ({sentiment_pct:.2f}%)")

if "alerted_today" not in st.session_state: st.session_state["alerted_today"] = set()
debug_issues, results = [], []

for ticker in sp100:
    try:
        # Market Sentiment filter
        if sentiment_text == "🔴 Bearish" and not is_defensive_sector(ticker):
            continue

        # --- Hybrid 5m+1h Confirm ---
        if hybrid_on:
            df5 = yf.download(ticker, period='3d', interval='5m', progress=False, threads=False)
            df1h = yf.download(ticker, period='10d', interval='1h', progress=False, threads=False)
            if df5.empty or df1h.empty: continue
            df5 = norm(df5)
            df1h = norm(df1h)
            try:
                df5 = ensure_core_cols(df5)
                df1h = ensure_core_cols(df1h)
            except Exception as e:
                debug_issues.append({"Ticker": ticker, "Issue": str(e)})
                continue
            df5 = calc_indicators(df5)
            df1h = calc_indicators(df1h)
            c, ema40 = df5['close'].iloc[-1], df5['ema40'].iloc[-1]
            prev_c, prev_ema = df5['close'].iloc[-2], df5['ema40'].iloc[-2]
            breakout = (prev_c < prev_ema) and (c > ema40)
            vol_avg = df5['volume'][-40:-1].mean() + 1e-8
            vol_spike = df5['volume'].iloc[-1] / vol_avg
            curr_1h = df1h.iloc[-1]
            above_ema200 = curr_1h['close'] > curr_1h['ema200']
            pct_breakout = (c - ema40) / ema40 * 100
            pct_above_ema200 = (curr_1h['close'] - curr_1h['ema200']) / curr_1h['ema200'] * 100
            rsi_bonus = 7 if 55 <= curr_1h['rsi14'] <= 65 else 0
            score = 55 + min(40, pct_breakout*12 + (vol_spike-1)*15 + pct_above_ema200*4 + rsi_bonus)
            score = int(max(45, min(score, 100)))
            cond = breakout and (vol_spike > 1.2) and above_ema200
            if cond:
                price = c
                atr = df5['atr14'].iloc[-1]
                if not (min_price <= price <= max_price): continue
                if df5['volume'].iloc[-1] < min_vol: continue
                target_val = max(2*atr, price*min_target_pct)
                cut_val = max(1.5*atr, price*min_cutloss_pct)
                if target_val < price*min_target_pct: continue
                shares = int(capital_per_trade // price)
                target_price = round(price + target_val, 2)
                cut_loss_price = round(price - cut_val, 2)
                sigid = (ticker, "Hybrid 5min+1hr Confirm")
                results.append({
                    "Ticker": ticker,
                    "Strategy": "Hybrid 5min+1hr Confirm",
                    "Score": score,
                    "Entry": formatn(price),
                    "Target Price": formatn(target_price),
                    "Cut Loss Price": formatn(cut_loss_price),
                    "Shares": shares,
                    "ATR": formatn(atr,2),
                    "Reason": f"Breakout: {pct_breakout:.2f}%, Vol x{vol_spike:.2f}, EMA200 Confirm: {pct_above_ema200:.2f}%",
                    "Type": "Hybrid",
                    "Time Picked": local_time_str(),
                    "SigID": sigid
                })

        # --- MACD Stack (Momentum, 5m) ---
        if macd_stack_on:
            df = yf.download(ticker, period='3d', interval='5m', progress=False, threads=False)
            if df.empty: continue
            df = norm(df)
            try:
                df = ensure_core_cols(df)
            except Exception as e:
                debug_issues.append({"Ticker": ticker, "Issue": str(e)})
                continue
            df = calc_indicators(df)
            c, ema10, ema20, ema50 = df['close'].iloc[-1], df['ema10'].iloc[-1], df['ema20'].iloc[-1], df['ema50'].iloc[-1]
            macd, macdsignal = df['macd'].iloc[-1], df['macdsignal'].iloc[-1]
            macd_prev, macdsignal_prev = df['macd'].iloc[-2], df['macdsignal'].iloc[-2]
            cross_up = (macd_prev < macdsignal_prev) and (macd > macdsignal)
            rising = (macd > macd_prev) and (macd < 0)
            stacked = (c > ema10 > ema20 > ema50)
            vol_avg = df['volume'][-20:-1].mean() + 1e-8
            vol_spike = df['volume'].iloc[-1] / vol_avg
            macd_dist = abs(macd - macdsignal)
            score = 50 + min(40, (macd_dist*60) + (vol_spike-1)*12 + (ema10-ema50)/ema50*20)
            score = int(max(40, min(score, 99)))
            cond = cross_up and rising and stacked and (vol_spike > 1.1)
            if cond:
                price = c
                atr = df['atr14'].iloc[-1]
                if not (min_price <= price <= max_price): continue
                if df['volume'].iloc[-1] < min_vol: continue
                target_val = max(2*atr, price*min_target_pct)
                cut_val = max(1.5*atr, price*min_cutloss_pct)
                if target_val < price*min_target_pct: continue
                shares = int(capital_per_trade // price)
                target_price = round(price + target_val, 2)
                cut_loss_price = round(price - cut_val, 2)
                sigid = (ticker, "MACD Stack")
                results.append({
                    "Ticker": ticker,
                    "Strategy": "MACD Stack",
                    "Score": score,
                    "Entry": formatn(price),
                    "Target Price": formatn(target_price),
                    "Cut Loss Price": formatn(cut_loss_price),
                    "Shares": shares,
                    "ATR": formatn(atr,2),
                    "Reason": f"MACD Up, Stack, Vol x{vol_spike:.2f}, dist {macd_dist:.4f}",
                    "Type": "Momentum",
                    "Time Picked": local_time_str(),
                    "SigID": sigid
                })

        # --- EMA200 Breakout (Swing) ---
        if ema200_on:
            dfd = yf.download(ticker, period="1y", interval="1d", progress=False, threads=False)
            if dfd.empty or len(dfd) < 210: continue
            dfd = norm(dfd)
            try:
                dfd = ensure_core_cols(dfd)
            except Exception as e:
                debug_issues.append({"Ticker": ticker, "Issue": str(e)})
                continue
            dfd = calc_indicators(dfd)
            prev, curr = dfd.iloc[-2], dfd.iloc[-1]
            breakout = (prev['close'] < prev['ema200']) and (curr['close'] > curr['ema200'])
            vol_avg = dfd['volume'][-20:-1].mean() + 1e-8
            vol_mult = curr['volume'] / vol_avg
            rsi_ok = curr['rsi14'] < 70
            pct_above_ema = (curr['close'] - curr['ema200']) / curr['ema200'] * 100
            score = 60 + min(40, pct_above_ema*6 + (vol_mult-1)*12)
            score = int(max(50, min(score, 100)))
            cond = breakout and (vol_mult > 1.5) and rsi_ok
            if cond:
                entry = curr['close']
                atr = dfd['atr14'].iloc[-1]
                target_val = max(2*atr, entry*min_target_pct)
                cut_val = max(1.5*atr, entry*min_cutloss_pct)
                if target_val < entry*min_target_pct: continue
                shares = int(capital_per_trade // entry)
                target_price = round(entry + target_val, 2)
                cut_loss_price = round(entry - cut_val, 2)
                sigid = (ticker, "EMA200 Breakout")
                results.append({
                    "Ticker": ticker,
                    "Strategy": "EMA200 Breakout",
                    "Score": score,
                    "Entry": formatn(entry),
                    "Target Price": formatn(target_price),
                    "Cut Loss Price": formatn(cut_loss_price),
                    "Shares": shares,
                    "ATR": formatn(atr,2),
                    "Reason": f"Breakout +{pct_above_ema:.2f}%, Vol x{vol_mult:.2f}",
                    "Type": "Swing",
                    "Time Picked": local_time_str(),
                    "SigID": sigid
                })
        time.sleep(0.07)
    except Exception as e:
        debug_issues.append({"Ticker": ticker, "Issue": str(e)})
        continue

# ---- OUTPUT TABLE & ALERTS ----
if results:
    df_out = pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)
    st.subheader("⭐ AI-Picked Stock Setups (Top 5)")
    st.dataframe(df_out.head(5), use_container_width=True)
    
    # ---- TELEGRAM/GOOGLE SHEETS: Top 3 only ----
    if is_market_open():
        telegram_msgs, gsheet_rows = [], []
        for idx, row in df_out.head(3).iterrows():
            if row['SigID'] in st.session_state["alerted_today"]:
                continue
            msg = (
                f"#{row['Strategy']} #{row['Type']}\n"
                f"Ticker: {row['Ticker']} | Score: {row['Score']}\n"
                f"Entry: ${row['Entry']} | Target: ${row['Target Price']} | Stop: ${row['Cut Loss Price']}\n"
                f"ATR: {row['ATR']}\n"
                f"Reason: {row['Reason']}\n"
                f"Time: {row['Time Picked']} (GMT+8)"
            )
            telegram_msgs.append(msg)
            gsheet_rows.append([
                row['Time Picked'], row['Ticker'], row['Entry'], row['Target Price'], row['Cut Loss Price'],
                row['Strategy'], row['Score'], row['Reason']
            ])
            st.session_state["alerted_today"].add(row['SigID'])
        for msg in telegram_msgs: send_telegram_alert(msg)
        if gsheet_rows: append_to_gsheet(gsheet_rows, GOOGLE_SHEET_NAME)
else:
    st.warning("No current signals found.")

# ---- Last Signals Table (Google Sheet, last 10 rows) ----
try:
    client = get_gspread_client_from_secrets()
    sheet = client.open_by_key(GOOGLE_SHEET_ID).worksheet(GOOGLE_SHEET_NAME)
    data = sheet.get_all_values()
    if data and len(data) > 1:
        headers, rows = data[0], data[-10:]
        df_last = pd.DataFrame(rows, columns=headers)
        st.subheader("🕒 Last 10 US Screener Signals")
        st.dataframe(df_last)
except Exception as e:
    st.warning(f"Error loading last signals: {e}")

# ---- KLSE Screener (EOD only) ----
klse_list = ["MAYBANK.KL", "PBBANK.KL", "CIMB.KL", "SIME.KL", "PETGAS.KL", "AXIATA.KL", "GENTING.KL", "TENAGA.KL",
    "DIGI.KL", "MAXIS.KL", "HARTA.KL", "TOPGLOV.KL", "IHH.KL", "KLK.KL", "SIMEPLT.KL", "MRDIY.KL",
    "TM.KL", "PCHEM.KL", "PETDAG.KL", "HLBANK.KL", "HLFG.KL", "RHBBANK.KL", "AMBANK.KL", "BURSA.KL",
    "INARI.KL", "SUPERMX.KL", "FGV.KL", "AIRPORT.KL", "YTLPOWR.KL", "YTL.KL", "DIALOG.KL", "HEIM.KL",
    "CARLSBG.KL", "PPB.KL", "GASMSIA.KL", "PETRONM.KL", "GENM.KL", "MISC.KL", "WESTPORTS.KL", "QES.KL",
    "SCGM.KL", "FRONTKN.KL", "MYEG.KL", "DNEX.KL", "REVENUE.KL", "HIBISCS.KL", "UEMS.KL", "ECOWLD.KL",
    "MAHSING.KL", "SPSETIA.KL", "SUNWAY.KL", "IJM.KL", "MRCB.KL", "AME.KL", "KERJAYA.KL", "TROP.KL",
    "SIMEPROP.KL", "UOADEV.KL", "LCTITAN.KL", "KOSSAN.KL", "SCIENTX.KL", "FARMFESH.KL", "QL.KL", "BPLANT.KL",
    "MBSB.KL", "UWC.KL", "CTOS.KL", "NESTLE.KL", "VITROX.KL", "KESM.KL", "GREATEC.KL", "GENETEC.KL",
    "PMETAL.KL", "ANNJOO.KL", "LIONIND.KL", "ATAIMS.KL", "VS.KL", "KOBAY.KL", "MASTEEL.KL", "PENTA.KL",
    "PIE.KL", "TGUAN.KL", "WELLCAL.KL", "KAREX.KL", "COMCORP.KL", "JHM.KL", "ENGTEX.KL", "TASCO.KL",
    "LATITUD.KL", "TEXCHEM.KL", "SAM.KL", "CCK.KL", "HENGYUAN.KL", "UMW.KL", "TDM.KL", "FPGROUP.KL",
    "UNISEM.KL", "UCHITEC.KL", "RESINTC.KL", "PIB.KL"]
klse_results = []
for ticker in klse_list:
    try:
        df = yf.download(ticker, period="90d", interval="1d", progress=False, threads=False)
        if df.empty or len(df) < 30: continue
        df = norm(df)
        df = ensure_core_cols(df)
        df = calc_indicators(df)
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        breakout = (prev['close'] < prev['ema200']) and (curr['close'] > curr['ema200'])
        vol_avg = df['volume'][-20:-1].mean() + 1e-8
        vol_mult = curr['volume'] / vol_avg
        pct_above_ema = (curr['close'] - curr['ema200']) / curr['ema200'] * 100
        cond = breakout and (vol_mult > 1.5)
        if cond:
            price = curr['close']
            atr = curr['atr14']
            target_val = max(2*atr, price*min_target_pct)
            cut_val = max(1.5*atr, price*min_cutloss_pct)
            shares = int(capital_per_trade // price)
            target_price = round(price + target_val, 2)
            cut_loss_price = round(price - cut_val, 2)
            klse_results.append({
                "Ticker": ticker,
                "Entry": formatn(price),
                "Target Price": formatn(target_price),
                "Cut Loss Price": formatn(cut_loss_price),
                "Shares": shares,
                "ATR": formatn(atr,2),
                "Volume": int(curr['volume']),
                "Reason": f"Breakout +{pct_above_ema:.2f}%, Vol x{vol_mult:.2f}",
                "Time Picked": local_time_str()
            })
    except Exception as e:
        continue

st.subheader("🇲🇾 KLSE: EMA200 Breakout (Daily)")
if klse_results:
    df_klse = pd.DataFrame(klse_results)
    st.dataframe(df_klse, use_container_width=True)
    try:
        append_to_gsheet(df_klse.values.tolist(), KLSE_GOOGLE_SHEET_NAME)
    except Exception as e:
        st.warning(f"KLSE sheet log error: {e}")
else:
    st.info("No KLSE EMA200 breakouts found today.")
    
if show_debug:
    st.subheader("Debug: Issues Encountered")
    if debug_issues:
        st.dataframe(pd.DataFrame(debug_issues))
    else:
        st.info("No issues.")

st.caption("© AI Screener | S&P 100 + KLSE. Sentiment adapts universe. Signals include pre-market, regular, and after-hours. Confidence scoring reflects signal strength.")
