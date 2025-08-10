import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yfinance as yf
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, time as dtime, timedelta
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import math
import io

"""
US Intraday Screener + Auto-Trader (PDT-aware)
-------------------------------------------------
Implements your requested changes:
1) **Auto-build universe ($5–$100) without hardcoding** using NASDAQ Trader symbol files (fallbacks included) + multi-ticker chunking.
2) **Keeps MACD + RSI look-back** fast via caching and incremental buffers (no full re-download each refresh).
3) **Displays US/Eastern time** consistently; optional MYT clock for reference.
4) **Adds light backtest pane** (last N sessions) with fees/slippage + PDT cap to estimate true expectancy.
5) Regime-gated two setups (ORB–VWAP Continuation, RSI(2) VWAP Snapback) retained, plus your 1H MACD cross as reference.

Notes:
- If external symbol files are blocked, app falls back to your prior S&P-like list and/or a manual paste box.
- For live trading, wire your broker in `route_order()`.
"""

# -------------- Config --------------
GOOGLE_SHEET_ID = "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4"
SHEET_SIGNALS = "Signals"
SHEET_UNIVERSE = "Universe"

ET_TZ = pytz.timezone("US/Eastern")
MY_TZ = pytz.timezone("Asia/Kuala_Lumpur")

# -------------- Utilities --------------
def et_now():
    return datetime.now(ET_TZ)

def et_now_str():
    return et_now().strftime('%Y-%m-%d %H:%M:%S')

def my_now_str():
    return datetime.now(MY_TZ).strftime('%Y-%m-%d %H:%M:%S')

def formatn(x, d=2):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "-"
        if isinstance(x, (int, np.integer)) or d == 0:
            return f"{int(x):,}"
        return f"{float(x):,.{d}f}"
    except Exception:
        return str(x)

# ---- Google Sheets helpers ----
def get_gspread_client_from_secrets():
    info = st.secrets.get("gcp_service_account", {})
    if not info:
        raise RuntimeError("Missing gcp_service_account in st.secrets")
    creds_dict = {k: v for k, v in info.items()}
    if isinstance(creds_dict.get("private_key"), list):
        creds_dict["private_key"] = "\n".join(creds_dict["private_key"])  # fix multiline key issue  # fix multiline key issue
    creds_json = json.dumps(creds_dict)
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(creds_json), scope)
    return gspread.authorize(creds)

def append_to_gsheet(rows, sheet_name):
    try:
        client = get_gspread_client_from_secrets()
        sheet = client.open_by_key(GOOGLE_SHEET_ID).worksheet(sheet_name)
        if isinstance(rows, pd.DataFrame):
            rows = rows.values.tolist()
        for row in rows:
            sheet.append_row(row, value_input_option="USER_ENTERED")
    except Exception as e:
        st.warning(f"Google Sheet ({sheet_name}): {e}")

# ---- Data fetch & caching ----
@st.cache_data(ttl=300)
def fetch_intraday(tickers, interval="5m", period="5d", prepost=True):
    """Multi-ticker intraday fetch with caching. Returns dict[ticker]->DataFrame.
    Ensures lowercase OHLCV columns and ET timezone for both single and multi ticker paths."""
    if isinstance(tickers, str):
        tickers = [tickers]
    data = {}
    CHUNK = 150
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i:i+CHUNK]
        try:
            raw = yf.download(" ".join(chunk), interval=interval, period=period, prepost=prepost,
                              group_by='ticker', auto_adjust=False, progress=False, threads=False)
            # Normalize to dict[ticker]->DataFrame with lowercase cols + ET tz
            if len(chunk) == 1:
                sub = raw.copy()
                sub = sub.rename(columns={c: c.split(" ")[0] for c in sub.columns})
                if sub.index.tz is None:
                    sub.index = sub.index.tz_localize('UTC').tz_convert(ET_TZ)
                else:
                    sub.index = sub.index.tz_convert(ET_TZ)
                sub = sub[['Open','High','Low','Close','Volume']].rename(columns=str.lower).dropna()
                data[chunk[0]] = sub
            else:
                dd = {}
                for t in chunk:
                    try:
                        sub = raw[t]
                    except Exception:
                        # Some tickers may be missing
                        continue
                    sub = sub.rename(columns={c: c.split(" ")[0] for c in sub.columns})
                    if sub.index.tz is None:
                        sub.index = sub.index.tz_localize('UTC').tz_convert(ET_TZ)
                    else:
                        sub.index = sub.index.tz_convert(ET_TZ)
                    dd[t] = sub[['Open','High','Low','Close','Volume']].rename(columns=str.lower).dropna()
                for t, d in dd.items():
                    data[t] = d
        except Exception:
            continue
    return data

@st.cache_data(ttl=900)
def fetch_daily(tickers, period="6mo"):
    """Multi-ticker daily fetch returning dict[ticker]->DataFrame with lowercase OHLCV (ET tz)."""
    if isinstance(tickers, str):
        tickers = [tickers]
    data = {}
    CHUNK = 200
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i:i+CHUNK]
        try:
            raw = yf.download(" ".join(chunk), interval="1d", period=period, group_by='ticker', auto_adjust=False, progress=False, threads=False)
            if len(chunk) == 1:
                sub = raw.copy()
                sub = sub.rename(columns={c: c.split(" ")[0] for c in sub.columns})
                if sub.index.tz is None:
                    sub.index = sub.index.tz_localize('UTC').tz_convert(ET_TZ)
                else:
                    sub.index = sub.index.tz_convert(ET_TZ)
                sub = sub[['Open','High','Low','Close','Volume']].rename(columns=str.lower).dropna()
                data[chunk[0]] = sub
            else:
                dd = {}
                for t in chunk:
                    try:
                        sub = raw[t]
                    except Exception:
                        continue
                    sub = sub.rename(columns={c: c.split(" ")[0] for c in sub.columns})
                    if sub.index.tz is None:
                        sub.index = sub.index.tz_localize('UTC').tz_convert(ET_TZ)
                    else:
                        sub.index = sub.index.tz_convert(ET_TZ)
                    dd[t] = sub[['Open','High','Low','Close','Volume']].rename(columns=str.lower).dropna()
                for t, d in dd.items():
                    data[t] = d
        except Exception:
            continue
    return data

# ---- Indicators ----
def rsi(series: pd.Series, length=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(length).mean()
    roll_down = down.rolling(length).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, length=14):
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def vwap_session(df: pd.DataFrame):
    """Session VWAP from today's first bar (04:00 ET including premarket)."""
    if df.empty:
        return pd.Series(dtype=float)
    today = df.index[-1].date()
    sess_start = datetime.combine(today, dtime(4, 0), tzinfo=ET_TZ)
    intr = df[df.index >= sess_start]
    tp = (intr['high'] + intr['low'] + intr['close']) / 3
    cum_tp_vol = (tp * intr['volume']).cumsum()
    cum_vol = intr['volume'].cumsum().replace(0, np.nan)
    v = cum_tp_vol / cum_vol
    out = pd.Series(index=df.index, dtype=float)
    out.loc[intr.index] = v
    return out

# ---- Opening Range ----
def opening_range(df: pd.DataFrame, start=dtime(9,30), end=dtime(10,0)):
    if df.empty: return np.nan, np.nan
    dti = df.between_time(start_time=start, end_time=end)
    if dti.empty: return np.nan, np.nan
    return float(dti['high'].max()), float(dti['low'].min())

# ---- Regime ----
@st.cache_data(ttl=600)
def market_regime(sample_tickers):
    spy = fetch_daily(["SPY"], period="12mo").get("SPY", pd.DataFrame())
    if spy.empty:
        return {"go": False, "why": "SPY data unavailable", "trend": False, "vol_ok": False, "breadth_ok": False, "atrp5": None}
    spy['sma50'] = spy['close'].rolling(50).mean()
    spy['sma200'] = spy['close'].rolling(200).mean()
    spy['atrp5'] = (atr(spy, 5) / spy['close']) * 100
    last = spy.iloc[-1]
    trend = (last['close'] > last['sma50']) and (last['sma50'] > last['sma200'])
    vol_ok = last['atrp5'] < 2.2
    # Breadth proxy: % of sample above 20SMA
    daily = fetch_daily(sample_tickers, period="3mo")
    up = 0; n = 0
    for t, d in daily.items():
        if d.empty or len(d) < 25: continue
        n += 1
        if d['close'].iloc[-1] > d['close'].rolling(20).mean().iloc[-1]:
            up += 1
    breadth_ok = (n > 0) and ((up / n) > 0.55)
    go = bool(trend and vol_ok and breadth_ok)
    return {"go": go, "trend": trend, "vol_ok": vol_ok, "breadth_ok": breadth_ok, "atrp5": float(last['atrp5'])}

# ---- Universe Builder ----
NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

@st.cache_data(ttl=60*60*6)
def download_symbol_files():
    """Download symbol tables; return list of tickers. Fallback if blocked."""
    syms = []
    try:
        nasdaq = pd.read_csv(NASDAQ_URL, sep='|')
        nasdaq = nasdaq[~nasdaq['Symbol'].str.contains("\^|\$|\.|~", na=False)]
        syms.extend(nasdaq['Symbol'].dropna().tolist())
    except Exception:
        pass
    try:
        other = pd.read_csv(OTHER_URL, sep='|')
        other = other[~other['ACT Symbol'].str.contains("\^|\$|\.|~", na=False)]
        syms.extend(other['ACT Symbol'].dropna().tolist())
    except Exception:
        pass
    # Fallback minimal set if both failed
    if not syms:
        syms = [
            "AAPL","MSFT","AMZN","NVDA","META","GOOGL","TSLA","AMD","NFLX","QCOM","COIN","SQ","TTD","F","GM",
            "ON","MPWR","KLAC","LRCX","MU","CRM","ORCL","INTC","CAT","MCD","CMG","ROKU","UBER","LYFT"
        ]
    # Deduplicate & clean
    syms = sorted(list({s.strip().upper(): None for s in syms}.keys()))
    return syms

@st.cache_data(ttl=60*60)
def build_universe(min_price=5.0, max_price=100.0, min_avg_dollar_vol=20_000_000, min_atr_pct=2.0, max_atr_pct=12.0, max_symbols=800):
    """Return filtered tickers meeting liquidity/vol/price constraints."""
    raw_syms = download_symbol_files()
    # Limit the number of symbols for performance
    raw_syms = raw_syms[:5000]
    # Fetch daily snapshots in chunks
    daily = fetch_daily(raw_syms, period="3mo")
    selected = []
    for t, d in daily.items():
        if d.empty or len(d) < 25:
            continue
        last = d.iloc[-1]
        price_ok = (min_price <= last['close'] <= max_price)
        d['atrp14'] = (atr(d, 14) / d['close']) * 100
        atrp = d['atrp14'].iloc[-1]
        atr_ok = (min_atr_pct <= atrp <= max_atr_pct)
        # 20d avg dollar volume
        adv = (d['close'] * d['volume']).rolling(20).mean().iloc[-1]
        liquid = adv is not None and not np.isnan(adv) and (adv >= min_avg_dollar_vol)
        if price_ok and atr_ok and liquid:
            selected.append(t)
        if len(selected) >= max_symbols:
            break
    return selected

# -------------- Streamlit UI --------------
st.set_page_config(page_title="US Screener + Auto-Trader", layout="wide")
st_autorefresh(interval=5*60*1000, key="autorefresh")  # every 5 min

st.title("US Intraday Screener + Auto-Trader (PDT-aware)")
colt1, colt2 = st.columns(2)
with colt1:
    st.caption(f"Last run (US/Eastern): {et_now_str()}")
with colt2:
    st.caption(f"Local (MYT): {my_now_str()}")

with st.sidebar:
    st.header("Universe & Filters")
    min_price = st.number_input("Min Price ($)", value=5.0)
    max_price = st.number_input("Max Price ($)", value=100.0)
    min_avg_dollar_vol = st.number_input("Min Avg Dollar Volume (20d)", value=20_000_000)
    max_atr_pct = st.number_input("Max 14d ATR%", value=12.0)
    min_atr_pct = st.number_input("Min 14d ATR%", value=2.0)
    st.markdown("Manual tickers (optional, comma/space-separated):")
    manual_syms = st.text_area("Symbols override", value="", height=60)

    st.divider()
    st.subheader("Risk & PDT")
    acct_equity = st.number_input("Account Equity ($)", value=10_000)
    risk_pct = st.slider("Risk % per trade", 0.1, 1.0, 0.75, step=0.05)
    max_new_trades_today = st.number_input("Max NEW trades today", value=3, min_value=1)
    fees_per_trade = st.number_input("Fees per trade ($)", value=0.50, min_value=0.0, step=0.05)
    slip_bps = st.number_input("Slippage (bps)", value=5, min_value=0)

    st.divider()
    st.subheader("Strategy Windows")
    rvol_threshold = st.number_input("Min 5m RVOL (signal bar)", value=1.5)
    enable_orb = st.checkbox("Enable ORB-VWAP Continuation", value=True)
    enable_snap = st.checkbox("Enable RSI(2) VWAP Snapback", value=True)

# Session state buffers (incremental updates)
if "buffers" not in st.session_state:
    st.session_state.buffers = {}  # ticker -> DataFrame (5m intraday)
if "ledger" not in st.session_state:
    st.session_state.ledger = []
if "today_trades" not in st.session_state:
    st.session_state.today_trades = 0

# -------------- Build Universe --------------
if manual_syms.strip():
    base_universe = [s.strip().upper() for s in manual_syms.replace('\n',' ').replace(',', ' ').split() if s.strip()]
else:
    base_universe = build_universe(min_price, max_price, min_avg_dollar_vol, min_atr_pct, max_atr_pct)

st.success(f"Universe size: {len(base_universe)} (auto-built)")

# Export universe to GSheet (optional, non-fatal)
try:
    # Write universe vertically: Timestamp, Ticker per row
try:
    uni_rows = [[et_now_str(), t] for t in base_universe]
    append_to_gsheet(uni_rows, SHEET_UNIVERSE)
except Exception:
    pass
except Exception:
    pass

# -------------- Regime Panel --------------
# Use a small sample for breadth calc to reduce calls
sample_for_regime = base_universe[:120] if base_universe else ["AAPL","MSFT","NVDA","SPY"]
reg = market_regime(sample_for_regime)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Regime", "GO" if reg["go"] else "CHOP", help=f"Trend={reg['trend']} / VolOK={reg['vol_ok']} / BreadthOK={reg['breadth_ok']}")
col2.metric("Trend", "OK" if reg["trend"] else "Weak")
col3.metric("Vol Proxy (SPY ATR% 5d)", formatn(reg.get("atrp5"),2))
col4.metric("Breadth>20SMA", "OK" if reg["breadth_ok"] else "Weak")

st.divider()

# -------------- Strategy Logic --------------
def position_size(equity, risk_pct, entry, stop):
    risk_amt = equity * (risk_pct/100)
    per_sh = max(0.01, abs(entry - stop))
    qty = math.floor(risk_amt / per_sh)
    return max(0, qty)


def within_regular_hours(ts):
    t = ts.tz_convert(ET_TZ).time()
    return dtime(9,30) <= t <= dtime(16,0)


def strategy_signals_for_df(ticker: str, df5: pd.DataFrame, go_regime: bool):
    if df5.empty or len(df5) < 60:
        return []
    # indicators
    df5 = df5.copy()
    df5['ema9'] = df5['close'].ewm(span=9, min_periods=9).mean()
    df5['rsi2'] = rsi(df5['close'], 2)
    df5['vwap'] = vwap_session(df5)
    orh, orl = opening_range(df5)
    last = df5.iloc[-1]
    if not within_regular_hours(df5.index[-1]):
        return []
    out = []
    # ORB-VWAP (10:00–12:00)
    if go_regime and enable_orb and dtime(10,0) <= df5.index[-1].time() <= dtime(12,0):
        vol20 = df5['volume'].rolling(20).mean().iloc[-1]
        rvol = last['volume'] / max(1, vol20)
        trigger = (not np.isnan(orh)) and (last['close'] > orh) and (last['close'] > last['vwap']) and (rvol >= rvol_threshold)
        if trigger:
            atr5 = atr(df5, 14).iloc[-1]
            stop = min(last['vwap'] - 0.25*atr5, df5['low'].iloc[-3:-1].min())
            entry = last['close'] + 0.01
            rr = (entry - stop)
            out.append({
                'ticker': ticker, 'setup': 'ORB_VWAP', 'time': df5.index[-1], 'price': float(last['close']),
                'entry': float(entry), 'stop': float(stop), 'target1': float(entry + 1.5*rr), 'rvol': float(rvol),
                'orh': float(orh), 'orl': float(orl)
            })
    # RSI(2) VWAP Snapback (10:00–15:30)
    if (not go_regime) and enable_snap and dtime(10,0) <= df5.index[-1].time() <= dtime(15,30):
        rsi2v = last['rsi2']
        dipped = (df5['close'].iloc[-2] < df5['vwap'].iloc[-2]) or (df5['low'].iloc[-1] < last['vwap'])
        reclaimed = (last['close'] > last['vwap']) or (last['close'] > last['ema9'] and df5['close'].iloc[-2] <= df5['ema9'].iloc[-2])
        if rsi2v <= 5 and dipped and reclaimed:
            swing_low = min(df5['low'].iloc[-3:-1])
            entry = last['close'] + 0.01
            stop = swing_low - 0.01
            atr5 = atr(df5, 14).iloc[-1]
            out.append({
                'ticker': ticker, 'setup': 'RSI2_VWAP', 'time': df5.index[-1], 'price': float(last['close']),
                'entry': float(entry), 'stop': float(stop), 'target1': float(last['vwap'] + 0.5*atr5), 'rsi2': float(rsi2v)
            })
    return out

# -------------- Incremental Buffers & Scan --------------
# Load latest 5m for universe (buffers updated incrementally)
existing = list(st.session_state.buffers.keys())
# First pass: for tickers we don't have, fetch 5d; for existing, fetch 1d and append
new_needed = [t for t in base_universe if t not in st.session_state.buffers]
if new_needed:
    intraday_new = fetch_intraday(new_needed, interval="5m", period="5d", prepost=True)
    for t, df in intraday_new.items():
        st.session_state.buffers[t] = df
# Update existing with just recent bars
intraday_recent = fetch_intraday(base_universe, interval="5m", period="1d", prepost=True)
for t, df in intraday_recent.items():
    if t not in st.session_state.buffers:
        st.session_state.buffers[t] = df
    else:
        old = st.session_state.buffers[t]
        # append rows newer than last index in old
        newer = df[df.index > (old.index.max() if not old.empty else df.index.min())]
        if not newer.empty:
            st.session_state.buffers[t] = pd.concat([old, newer]).drop_duplicates().sort_index()

progress = st.progress(0.0)
signals = []
for i, t in enumerate(base_universe[:400]):  # safety cap for UI perf
    try:
        df5 = st.session_state.buffers.get(t, pd.DataFrame())
        sigs = strategy_signals_for_df(t, df5, reg['go'])
        signals.extend(sigs)
    except Exception:
        pass
    progress.progress((i+1)/max(1, len(base_universe[:400])))

# PDT-aware throttle
remaining = max(0, max_new_trades_today - st.session_state.today_trades)
if remaining and signals:
    def score(s):
        base = 100 if (s['setup'] == 'ORB_VWAP' and reg['go']) else 80
        dist = abs(s['entry'] - s['stop'])
        return base - (dist * 1000)
    signals = sorted(signals, key=score, reverse=True)[:remaining]
else:
    signals = []

if signals:
    st.subheader("Actionable Signals (PDT-aware)")
    rows = []
    for s in signals:
        qty = position_size(acct_equity, risk_pct, s['entry'], s['stop'])
        rows.append({
            'Ticker': s['ticker'], 'Setup': s['setup'], 'Time (ET)': s['time'].strftime('%Y-%m-%d %H:%M'),
            'Price': formatn(s['price']), 'Entry': formatn(s['entry']), 'Stop': formatn(s['stop']),
            'Qty@Risk%': qty, 'Target1': formatn(s['target1'])
        })
    df_sig = pd.DataFrame(rows)
    st.dataframe(df_sig, use_container_width=True)
    try:
        append_to_gsheet(df_sig, SHEET_SIGNALS)
    except Exception:
        pass
else:
    st.info("No actionable signals (PDT slots used / outside RTH / no triggers).")

# -------------- Paper Trade Controls --------------
st.divider()
st.subheader("Paper Trades")
colA, colB = st.columns(2)
with colA:
    if st.button("Record above signals as paper trades") and signals:
        for s in signals:
            qty = position_size(acct_equity, risk_pct, s['entry'], s['stop'])
            st.session_state.ledger.append({
                'time': et_now_str(), 'ticker': s['ticker'], 'setup': s['setup'],
                'entry': s['entry'], 'stop': s['stop'], 'qty': qty, 'status': 'OPEN'
            })
        st.session_state.today_trades += len(signals)
        st.success(f"Recorded {len(signals)} paper trades. Today trades = {st.session_state.today_trades}")
with colB:
    if st.button("Flatten All (paper)"):
        for tr in st.session_state.ledger:
            if tr['status'] == 'OPEN':
                tr['status'] = 'CLOSED'
        st.success("All paper trades marked CLOSED.")

if st.session_state.ledger:
    st.dataframe(pd.DataFrame(st.session_state.ledger), use_container_width=True)

# -------------- Backtest (light) --------------
st.divider()
st.subheader("Light Backtest: last N sessions, with fees & slippage & PDT")
N_sessions = st.slider("Sessions (days)", 3, 15, 5)

@st.cache_data(ttl=900)
def backtest_last_sessions(tickers, sessions=5, go_regime_hint=True, fees=0.5, slippage_bps=5, risk_pct=0.75, pdt_cap=3):
    # Use last `sessions` days of 5m data; simulate entries at signal close + slippage; exits at target1 or EOD
    res = []
    # Fetch 5d+ data once
    all_data = fetch_intraday(tickers, interval="5m", period=f"{max(5, sessions*2)}d", prepost=True)
    days = sorted({d.index[-1].date() for d in all_data.values() if not d.empty})[-sessions:]
    trades_count = 0
    for t in tickers:
        df = all_data.get(t, pd.DataFrame())
        if df.empty: continue
        df['ema9'] = df['close'].ewm(span=9, min_periods=9).mean()
        df['rsi2'] = rsi(df['close'], 2)
        df['vwap'] = vwap_session(df)
        atr5 = atr(df, 14)
        for day in days:
            dfd = df[df.index.date == day]
            if dfd.empty: continue
            # opening range for the day
            orh = dfd.between_time('09:30','10:00')['high'].max() if not dfd.empty else np.nan
            # simple PDT cap per day
            used_today = 0
            for idx in dfd.index:
                if used_today >= pdt_cap:
                    break
                if not (dtime(10,0) <= idx.time() <= dtime(15,30)):
                    continue
                row = dfd.loc[idx]
                # Decide which setup based on go_regime_hint
                if go_regime_hint and enable_orb and (dtime(10,0) <= idx.time() <= dtime(12,0)):
                    vol20 = dfd['volume'].rolling(20).mean().loc[idx]
                    rvol = row['volume']/max(1, vol20) if pd.notna(vol20) else 0
                    if (pd.notna(orh) and row['close']>orh and row['close']>row['vwap'] and rvol>=rvol_threshold):
                        entry = row['close']*(1+slippage_bps/10000)
                        stop = min(row['vwap'] - 0.25*atr5.loc[idx], dfd['low'].loc[:idx].tail(3).min())
                        if pd.isna(stop) or stop>=entry: continue
                        rr = entry - stop
                        target = entry + 1.5*rr
                        # simulate forward until hit target or 15:55
                        fwd = dfd[(dfd.index>idx) & (dfd.index.time <= dtime(15,55))]
                        hit = fwd[fwd['high']>=target]
                        exit_price = (target if not hit.empty else fwd['close'].iloc[-1])
                        pnl = (exit_price - entry) - fees
                        res.append({"ticker":t,"day":str(day),"setup":"ORB_VWAP","pnl":pnl,"R":(exit_price-entry)/rr if rr>0 else 0})
                        used_today += 1; trades_count += 1
                if (not go_regime_hint) and enable_snap:
                    # RSI2 <=5 dip + reclaim
                    prev = dfd.shift(1).loc[idx]
                    dipped = (prev['close'] < prev['vwap']) if pd.notna(prev['vwap']) else False
                    reclaimed = (row['close'] > row['vwap']) or (row['close']>row['ema9'] and prev['close']<=prev['ema9'])
                    if (row['rsi2']<=5) and dipped and reclaimed:
                        entry = row['close']*(1+slippage_bps/10000)
                        swing_low = dfd['low'].loc[:idx].tail(3).min()
                        stop = swing_low - 0.01
                        target = row['vwap'] + 0.5*atr5.loc[idx]
                        if pd.isna(stop) or pd.isna(target) or stop>=entry: continue
                        fwd = dfd[(dfd.index>idx) & (dfd.index.time <= dtime(15,55))]
                        hit = fwd[fwd['high']>=target]
                        exit_price = (target if not hit.empty else fwd['close'].iloc[-1])
                        pnl = (exit_price - entry) - fees
                        rr = entry - stop
                        res.append({"ticker":t,"day":str(day),"setup":"RSI2_VWAP","pnl":pnl,"R":(exit_price-entry)/rr if rr>0 else 0})
                        used_today += 1; trades_count += 1
    if not res:
        return pd.DataFrame(), {"trades":0, "win%":None, "avg_pnl":None, "sum_pnl":0}
    dfres = pd.DataFrame(res)
    win = (dfres['pnl']>0).mean()*100
    return dfres, {"trades":len(dfres), "win%":round(win,1), "avg_pnl":round(dfres['pnl'].mean(),2), "sum_pnl":round(dfres['pnl'].sum(),2)}

bt_df, bt_stats = backtest_last_sessions(base_universe[:120], sessions=N_sessions, go_regime_hint=reg['go'], fees=fees_per_trade, slippage_bps=slip_bps, risk_pct=risk_pct, pdt_cap=max_new_trades_today)
if not bt_df.empty:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Backtest Trades", bt_stats['trades'])
    c2.metric("Win %", f"{formatn(bt_stats['win%'],1)}%")
    c3.metric("Avg PnL ($)", formatn(bt_stats['avg_pnl'],2))
    c4.metric("Total PnL ($)", formatn(bt_stats['sum_pnl'],2))
    st.dataframe(bt_df.tail(200), use_container_width=True)
else:
    st.info("Backtest: no trades generated under current filters/period.")

# -------------- Reference: MACD Cross (1H) --------------
st.divider()
with st.expander("Reference: MACD Cross Zero (1H)", expanded=False):
    # Use the same auto-universe but cap to keep it light
    cap_list = base_universe[:150] if base_universe else ["AAPL","MSFT","NVDA","AMD","SPY"]
    ref_rows, filt_rows = [], []
    # bulk fetch hourly in chunks
    CHUNK=120
    for i in range(0, len(cap_list), CHUNK):
        chunk = cap_list[i:i+CHUNK]
        try:
            hraw = yf.download(" ".join(chunk), period="10d", interval="1h", progress=False, threads=False, group_by='ticker')
        except Exception:
            continue
        if len(chunk) == 1:
            hdict = {chunk[0]: hraw}
        else:
            hdict = {t: hraw.get(t, pd.DataFrame()) for t in chunk}
        for t, h in hdict.items():
            if h is None or h.empty or len(h) < 40:
                continue
            try:
                h.index = h.index.tz_localize('UTC').tz_convert(ET_TZ)
            except Exception:
                h.index = pd.to_datetime(h.index).tz_convert(ET_TZ)
            h = h.rename(columns={c: c.split(" ")[0] for c in h.columns})
            h = h[['Open','High','Low','Close','Volume']].rename(columns=str.lower)
            ema10 = h['close'].ewm(span=10, min_periods=10).mean()
            ema20 = h['close'].ewm(span=20, min_periods=20).mean()
            delta = h['close'].diff(); up = delta.clip(lower=0); down = -delta.clip(upper=0)
            rsi14 = 100 - (100 / (1 + (up.rolling(14).mean() / (down.rolling(14).mean()+1e-9))))
            ema12 = h['close'].ewm(span=12, min_periods=12).mean(); ema26 = h['close'].ewm(span=26, min_periods=26).mean()
            macd = ema12 - ema26; macds = macd.ewm(span=9, min_periods=9).mean(); hist = macd - macds
            last = h.iloc[-1]
            if (macd.iloc[-1] > 0) and (macds.iloc[-1] < 0):
                ref_rows.append({"Ticker": t, "RSI": formatn(rsi14.iloc[-1],2), "MACD": formatn(macd.iloc[-1],4),
                                  "Signal": formatn(macds.iloc[-1],4), "Hist": formatn(hist.iloc[-1],4), "Price": formatn(last['close'])})
            if ((macd.iloc[-1] > 0) and (macds.iloc[-1] < 0) and (rsi14.iloc[-1] <= 60) and (ema10.iloc[-1] > ema20.iloc[-1]) and (hist.iloc[-1] > 0)):
                filt_rows.append({"Ticker": t, "RSI": formatn(rsi14.iloc[-1],2), "MACD": formatn(macd.iloc[-1],4),
                                  "Signal": formatn(macds.iloc[-1],4), "Hist": formatn(hist.iloc[-1],4), "Price": formatn(last['close'])})
    st.write("Filtered:")
    if filt_rows:
        st.dataframe(pd.DataFrame(filt_rows), use_container_width=True)
    else:
        st.info("No filtered 1H MACD cross signals.")
    st.write("Reference (any cross):")
    if ref_rows:
        st.dataframe(pd.DataFrame(ref_rows), use_container_width=True)
    else:
        st.info("No reference 1H crosses.")

st.caption("Auto-built universe in the $5–$100 range with liquidity/vol filters. Signals use US/Eastern time; MYT shown for convenience. Backtest includes fees/slippage & PDT cap.")
