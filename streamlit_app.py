import streamlit as st
# Optional autorefresh (safe fallback if not available)
try:
    from streamlit_autorefresh import st_autorefresh as _st_autorefresh
    _AUTOREFRESH_OK = True
except Exception:
    _AUTOREFRESH_OK = False
    def _st_autorefresh(*args, **kwargs):
        return None

import yfinance as yf
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
import math, re, json, time
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# =========================
# Config & Constants
# =========================
GOOGLE_SHEET_ID = "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4"
SHEET_UNIVERSE  = "Universe"
SHEET_SIGNALS   = "Signals"

ET_TZ = pytz.timezone("US/Eastern")
MY_TZ = pytz.timezone("Asia/Kuala_Lumpur")

st.set_page_config(page_title="Retail 1H Screener (MACD + Wife)", layout="wide")
_st_autorefresh(interval=5*60*1000, key="autorefresh")  # every 5 min

# =========================
# Utility helpers
# =========================
def et_now():
    return datetime.now(ET_TZ)

def et_now_str():
    return et_now().strftime("%Y-%m-%d %H:%M")

def my_now_str():
    return datetime.now(MY_TZ).strftime("%Y-%m-%d %H:%M")

def formatn(num, d=2):
    try:
        if num is None or (isinstance(num, float) and math.isnan(num)):
            return "-"
        return f"{float(num):,.{d}f}"
    except Exception:
        return str(num)

def to_yahoo_symbol(sym: str) -> str:
    """Convert NASDAQ dot class symbols to Yahoo dash class (e.g., BRK.B->BRK-B)."""
    if not sym:
        return sym
    s = sym.strip().upper()
    # Common fix: dot class to dash class
    s = s.replace(".", "-")
    return s

# =========================
# Google Sheets helpers
# =========================
def get_gspread_client_from_secrets():
    info = st.secrets["gcp_service_account"]
    creds_dict = dict(info)
    # If private_key arrived as list of lines, join it
    if isinstance(creds_dict.get("private_key"), list):
        creds_dict["private_key"] = "\n".join(creds_dict["private_key"])
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    return gspread.authorize(creds)

@st.cache_data(ttl=600)
def _get_sheet(ss_id, sheet_name):
    client = get_gspread_client_from_secrets()
    ss = client.open_by_key(ss_id)
    try:
        ws = ss.worksheet(sheet_name)
    except Exception:
        ws = ss.add_worksheet(title=sheet_name, rows="50000", cols="24")
    return ws

def append_to_gsheet(rows, sheet_name):
    """Append rows (list[list]) to a sheet. Auto-create if missing."""
    try:
        ws = _get_sheet(GOOGLE_SHEET_ID, sheet_name)
        if isinstance(rows, pd.DataFrame):
            rows = rows.values.tolist()
        if not rows:
            return
        if hasattr(ws, "append_rows"):
            ws.append_rows(rows, value_input_option="USER_ENTERED")
        else:
            for r in rows:
                ws.append_row(r, value_input_option="USER_ENTERED")
    except Exception as e:
        st.warning(f"Google Sheet ({sheet_name}): {e}")

def write_universe_sheet(tickers):
    """Bulk replace Universe sheet: header + rows; write Yahoo-normalized symbols."""
    try:
        ws = _get_sheet(GOOGLE_SHEET_ID, SHEET_UNIVERSE)
        rows = [["Timestamp (ET)", "Ticker"]] + [[et_now_str(), t] for t in tickers]
        ws.clear()
        ws.update("A1", rows, value_input_option="USER_ENTERED")
        return True, f"Wrote {len(tickers)} rows"
    except Exception as e:
        st.warning(f"Google Sheet (Universe): {e}")
        return False, str(e)

# =========================
# Symbol universe (NASDAQ files → Q/N only)
# =========================
NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_URL  = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

@st.cache_data(ttl=60*60*6)
def download_symbol_files_qn_only(exclude_funds=True, exclude_derivs=True):
    def clean_nasdaq(df):
        df = df.copy()
        df = df[df["Symbol"].notna() & (df["Symbol"] != "Symbol")]
        return pd.DataFrame({
            "symbol": df["Symbol"].astype(str).str.upper().str.strip(),
            "security_name": df.get("Security Name", pd.Series([None]*len(df))),
            "exchange": "Q",
            "etf": df.get("ETF", "N"),
            "nextshares": df.get("NextShares", "N"),
            "test_issue": df.get("Test Issue", "N"),
        })

    def clean_other(df):
        df = df.copy()
        df = df[df["ACT Symbol"].notna()]
        # Keep only NYSE (N)
        df = df[df.get("Exchange", "").astype(str).str.upper().eq("N")]
        return pd.DataFrame({
            "symbol": df["ACT Symbol"].astype(str).str.upper().str.strip(),
            "security_name": df.get("Security Name", pd.Series([None]*len(df))),
            "exchange": "N",
            "etf": df.get("ETF", "N"),
            "nextshares": pd.Series(["N"]*len(df)),
            "test_issue": df.get("Test Issue", "N"),
        })

    frames = []
    try:
        nasdaq = pd.read_csv(NASDAQ_URL, sep="|")
        frames.append(clean_nasdaq(nasdaq))
    except Exception:
        pass
    try:
        other = pd.read_csv(OTHER_URL, sep="|")
        frames.append(clean_other(other))
    except Exception:
        pass

    if not frames:
        # Fallback minimal list
        frames = [pd.DataFrame({"symbol": ["AAPL","MSFT","AMZN","NVDA","META","GOOGL","TSLA","AMD","NFLX","QCOM"],
                                "security_name": None, "exchange": "Q",
                                "etf":"N","nextshares":"N","test_issue":"N"})]
    df = pd.concat(frames, ignore_index=True)
    df.drop_duplicates(subset=["symbol"], inplace=True)

    if exclude_funds:
        df = df[(df["etf"].astype(str) != "Y") & (df["test_issue"].astype(str) != "Y")]
        if "nextshares" in df.columns:
            df = df[df["nextshares"].astype(str) != "Y"]

    if exclude_derivs and "security_name" in df.columns:
        pat = re.compile(r"(WARRANT|RIGHTS?|UNITS?|PREF|PREFERRED|NOTE|BOND|TRUST|FUND|ETF|ETN|DEPOSITARY|SPAC)", re.IGNORECASE)
        df = df[~df["security_name"].astype(str).str.contains(pat, na=False)]

    # Return Yahoo-normalized unique symbols
    syms = sorted({to_yahoo_symbol(s) for s in df["symbol"].dropna().unique().tolist()})
    return syms

# =========================
# Data fetchers (Daily / 1H)
# =========================
@st.cache_data(ttl=900)
def fetch_daily(tickers, period="6mo"):
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = [to_yahoo_symbol(t) for t in tickers]
    data = {}
    CHUNK = 200
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i:i+CHUNK]
        try:
            raw = yf.download(" ".join(chunk), interval="1d", period=period,
                              group_by="ticker", auto_adjust=False, progress=False, threads=False)
        except Exception:
            continue
        if len(chunk) == 1:
            sub = raw.rename(columns={c: c.split(" ")[0] for c in raw.columns})
            if sub.index.tz is None:
                sub.index = sub.index.tz_localize("UTC").tz_convert(ET_TZ)
            else:
                sub.index = sub.index.tz_convert(ET_TZ)
            sub = sub[["Open","High","Low","Close","Volume"]].rename(columns=str.lower).dropna()
            data[chunk[0]] = sub
        else:
            for t in chunk:
                try:
                    sub = raw[t]
                except Exception:
                    continue
                sub = sub.rename(columns={c: c.split(" ")[0] for c in sub.columns})
                if sub.index.tz is None:
                    sub.index = sub.index.tz_localize("UTC").tz_convert(ET_TZ)
                else:
                    sub.index = sub.index.tz_convert(ET_TZ)
                sub = sub[["Open","High","Low","Close","Volume"]].rename(columns=str.lower).dropna()
                data[t] = sub
    return data

@st.cache_data(ttl=300)
def fetch_1h_chunks(tickers, period="60d"):
    tickers = [to_yahoo_symbol(t) for t in tickers]
    out = {}
    CHUNK = 60
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i:i+CHUNK]
        try:
            raw = yf.download(" ".join(chunk), interval="1h", period=period, prepost=False,
                              progress=False, threads=False, group_by="ticker")
        except Exception:
            continue
        if len(chunk) == 1:
            dd = {chunk[0]: raw}
        else:
            dd = {}
            for t in chunk:
                try:
                    dd[t] = raw[t]
                except Exception:
                    continue
        for t, h in dd.items():
            if h is None or h.empty or len(h) < 40:
                continue
            h = h.rename(columns={c: c.split(" ")[0] for c in h.columns})
            try:
                h = h[["Open","High","Low","Close","Volume"]].rename(columns=str.lower)
            except Exception:
                cols = [c for c in ["Open","High","Low","Close","Volume"] if c in h.columns]
                if len(cols) < 5:
                    continue
                h = h[cols].rename(columns=str.lower)
            try:
                if h.index.tz is None:
                    h.index = h.index.tz_localize("UTC").tz_convert(ET_TZ)
                else:
                    h.index = h.index.tz_convert(ET_TZ)
            except Exception:
                h.index = pd.to_datetime(h.index).tz_localize("UTC").tz_convert(ET_TZ)
            h = h.between_time("09:30", "16:00")  # RTH only
            out[t] = h
    return out

# =========================
# Indicators
# =========================
def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr1 = (high - low).abs()
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr1 = (high - low)
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_n = tr.rolling(n).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(n).mean() / (atr_n + 1e-9)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(n).mean() / (atr_n + 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9)
    return dx.rolling(n).mean()

def same_hour_rvol(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return np.nan
    last_ts = df.index[-1]
    hr = last_ts.hour
    prev = df.iloc[:-1]
    prev_same_hr = prev[prev.index.hour == hr].tail(20)
    if prev_same_hr.empty:
        return np.nan
    return float(df["volume"].iloc[-1] / max(1.0, prev_same_hr["volume"].median()))

def ema(series: pd.Series, span: int, minp: int = None) -> pd.Series:
    return series.ewm(span=span, min_periods=minp or span).mean()

def kdj(df: pd.DataFrame, n: int = 9, k_smooth: int = 3, d_smooth: int = 3):
    """KDJ (stochastic variant): returns K, D, J."""
    high, low, close = df["high"], df["low"], df["close"]
    low_n = low.rolling(n).min()
    high_n = high.rolling(n).max()
    rsv = (close - low_n) / (high_n - low_n + 1e-9) * 100.0
    K = rsv.ewm(alpha=1.0/k_smooth, adjust=False).mean()
    D = K.ewm(alpha=1.0/d_smooth, adjust=False).mean()
    J = 3 * K - 2 * D
    return K, D, J

# =========================
# Universe Builder
# =========================
@st.cache_data(ttl=60*60)
def build_universe(min_price=5.0, max_price=100.0, min_avg_dollar_vol=30_000_000,
                   min_atr_pct=1.5, max_atr_pct=12.0, max_symbols=800,
                   return_stats=False):
    raw_syms = download_symbol_files_qn_only()
    raw_syms = raw_syms[:6000]  # safety cap
    daily = fetch_daily(raw_syms, period="3mo")
    evaluated = 0
    picked = []
    for t, d in daily.items():
        if d.empty or len(d) < 25:
            continue
        evaluated += 1
        last = d.iloc[-1]
        price_ok = (min_price <= last["close"] <= max_price)
        atrp14 = (atr(d, 14) / d["close"] * 100).iloc[-1]
        atr_ok = (atrp14 >= min_atr_pct) and (atrp14 <= max_atr_pct)
        adv = (d["close"] * d["volume"]).rolling(20).mean().iloc[-1]
        liquid = adv is not None and not math.isnan(float(adv)) and adv >= min_avg_dollar_vol
        if price_ok and atr_ok and liquid:
            picked.append((t, float(adv)))
    picked.sort(key=lambda x: x[1], reverse=True)
    selected = [t for t, _ in picked[:max_symbols]]
    if return_stats:
        return selected, {
            "evaluated": evaluated,
            "selected": len(selected),
            "adv_threshold": int(min_avg_dollar_vol),
            "atr_band": (min_atr_pct, max_atr_pct),
        }
    return selected

# =========================
# Strategy A: MACD_1H (filtered)
# =========================
def macd_1h_screener(tickers: list[str],
                     rsi_max: float = 60.0,
                     adx_min: float = 20.0,
                     require_ema_trend: bool = True,
                     rvol_min: float = 1.2,
                     daily_bias: bool = False) -> pd.DataFrame:
    data = fetch_1h_chunks(tickers, period="60d")
    out = []

    # Optional daily bias: EMA20 > EMA50
    daily_ok = set()
    if daily_bias and data:
        daily_data = fetch_daily(list(data.keys()), period="6mo")
        for t, d in daily_data.items():
            if d is None or d.empty or len(d) < 50: 
                continue
            if ema(d["close"], 20).iloc[-1] > ema(d["close"], 50).iloc[-1]:
                daily_ok.add(t)

    for t, h in data.items():
        if h is None or h.empty or len(h) < 40:
            continue
        ema10 = ema(h["close"], 10)
        ema20 = ema(h["close"], 20)
        delta = h["close"].diff(); up = delta.clip(lower=0); down = -delta.clip(upper=0)
        rsi14 = 100 - (100 / (1 + (up.rolling(14).mean() / (down.rolling(14).mean() + 1e-9))))
        ema12 = ema(h["close"], 12)
        ema26 = ema(h["close"], 26)
        macd = ema12 - ema26
        macds = ema(macd, 9)
        hist = macd - macds
        adx14 = adx(h, 14)
        rvol = same_hour_rvol(h)

        last = pd.DataFrame({
            "close": h["close"], "ema10": ema10, "ema20": ema20,
            "rsi14": rsi14, "macd": macd, "macds": macds, "hist": hist, "adx14": adx14
        }).iloc[-1]

        ok = (
            (last["macd"] > 0) and (last["macds"] < 0) and (last["hist"] > 0) and
            (last["rsi14"] <= rsi_max) and (rvol >= rvol_min) and (last["adx14"] >= adx_min)
        )
        if require_ema_trend:
            ok = ok and (last["ema10"] > last["ema20"])
        if daily_bias:
            ok = ok and (t in daily_ok)
        if not ok:
            continue

        out.append({
            "Strategy": "MACD_1H",
            "BarTime (ET)": h.index[-1].strftime("%Y-%m-%d %H:%M"),
            "Ticker": t,
            "Price": float(last["close"]),
            "RSI14": float(last["rsi14"]),
            "MACD": float(last["macd"]),
            "Signal": float(last["macds"]),
            "Hist": float(last["hist"]),
            "EMA10": float(last["ema10"]),
            "EMA20": float(last["ema20"]),
            "ADX14": float(last["adx14"]),
            "RVOL_same_hour": float(rvol),
            "DailyBias": bool(t in daily_ok) if daily_bias else None,
        })
    return pd.DataFrame(out)

# =========================
# Strategy B: WIFE_1H (EMA5/20/50 + MACD leadership + KDJ + vol + mcap)
# =========================
@st.cache_data(ttl=60*60)
def get_market_caps(tickers: list[str]) -> dict:
    """Fast market caps via yfinance fast_info; fallback to info."""
    caps = {}
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            cap = None
            # fast_info (newer yfinance) may be faster
            fi = getattr(tk, "fast_info", None)
            if fi and "market_cap" in fi and fi["market_cap"]:
                cap = fi["market_cap"]
            if cap is None:
                info = tk.info or {}
                cap = info.get("marketCap")
            caps[t] = cap if cap is not None else np.nan
        except Exception:
            caps[t] = np.nan
    return caps

def consec_green_count(close: pd.Series) -> int:
    """Consecutive green bars (close > prev close) ending at last bar."""
    if len(close) < 2:
        return 0
    up = (close > close.shift(1)).astype(int)
    # count run-length at the end
    cnt = 0
    for val in reversed(up.tolist()):
        if val == 1:
            cnt += 1
        else:
            break
    return cnt

def wife_1h_screener(tickers: list[str],
                     min_1h_volume: int = 1_000_000,
                     max_rvol_after_cross: float = 2.0,
                     min_market_cap: float = 1_000_000_000.0) -> pd.DataFrame:
    """
    Implements wife's rules on 1H:
      ⿡ EMA5 > EMA20 > EMA50 and each rising
      ⿢ MACD leadership: 
          (Hist<0 and Hist_t>Hist_{t-1}) OR (Hist>0 with <=2 consecutive green bars and RVOL<=cap)
          AND MACD>Signal AND MACD_t>MACD_{t-1}
      ⿣ KDJ bullish not overheated: K>D, K↑, D↑, J↑, J<80
      ⿤ Volume filter: 1H Volume >= min_1h_volume
      ⿥ Market Cap >= min_market_cap
    """
    data = fetch_1h_chunks(tickers, period="60d")
    out = []

    # Precompute market caps only for candidates later (to save bandwidth)
    # We'll gather tickers that pass all OHLCV rules first, then filter by mcap.

    proto_rows = []
    for t, h in data.items():
        if h is None or h.empty or len(h) < 60:
            continue

        # EMAs
        ema5  = ema(h["close"], 5)
        ema20 = ema(h["close"], 20)
        ema50 = ema(h["close"], 50)

        last = h.index[-1]
        # Short EMAs rising condition
        short_ok = (
            ema5.iloc[-1] > ema20.iloc[-1] > ema50.iloc[-1] and
            ema5.iloc[-1]  > ema5.iloc[-2]  and
            ema20.iloc[-1] > ema20.iloc[-2] and
            ema50.iloc[-1] > ema50.iloc[-2]
        )

        # MACD leadership
        macd_line = ema(h["close"], 12) - ema(h["close"], 26)
        signal    = ema(macd_line, 9)
        hist      = macd_line - signal

        macd_lead = (macd_line.iloc[-1] > signal.iloc[-1]) and (macd_line.iloc[-1] > macd_line.iloc[-2])

        # Hist patterns
        hist_neg_shrinking = (hist.iloc[-1] < 0) and (hist.iloc[-1] > hist.iloc[-2])
        hist_pos_early = False
        if hist.iloc[-1] > 0:
            cg = consec_green_count(h["close"])
            rvol = same_hour_rvol(h)
            hist_pos_early = (cg <= 2) and (rvol <= max_rvol_after_cross)

        macd_ok = macd_lead and (hist_neg_shrinking or hist_pos_early)

        # KDJ bullish not overheated
        K, D, J = kdj(h, n=9, k_smooth=3, d_smooth=3)
        kdj_ok = (
            (K.iloc[-1] > D.iloc[-1]) and
            (K.iloc[-1] > K.iloc[-2]) and (D.iloc[-1] > D.iloc[-2]) and
            (J.iloc[-1] > J.iloc[-2]) and (J.iloc[-1] < 80)
        )

        # Volume rule (1H)
        vol_ok = (h["volume"].iloc[-1] >= min_1h_volume)

        if short_ok and macd_ok and kdj_ok and vol_ok:
            rvol = same_hour_rvol(h)
            proto_rows.append({
                "Strategy": "WIFE_1H",
                "BarTime (ET)": last.strftime("%Y-%m-%d %H:%M"),
                "Ticker": t,
                "Price": float(h["close"].iloc[-1]),
                "EMA5":  float(ema5.iloc[-1]),
                "EMA20": float(ema20.iloc[-1]),
                "EMA50": float(ema50.iloc[-1]),
                "MACD":  float(macd_line.iloc[-1]),
                "Signal": float(signal.iloc[-1]),
                "Hist":  float(hist.iloc[-1]),
                "K": float(K.iloc[-1]),
                "D": float(D.iloc[-1]),
                "J": float(J.iloc[-1]),
                "RVOL_same_hour": float(rvol) if rvol is not None else np.nan,
                "Volume": int(h["volume"].iloc[-1]),
            })

    # Market cap filter (only on candidates)
    if proto_rows:
        caps = get_market_caps([r["Ticker"] for r in proto_rows])
        for r in proto_rows:
            cap = caps.get(r["Ticker"], np.nan)
            if pd.notna(cap) and cap >= min_market_cap:
                r["MarketCap"] = float(cap)
                out.append(r)

    return pd.DataFrame(out)

# =========================
# UI — Header / Health
# =========================
st.title("Retail 1H Screener — MACD + Wife Strategy")
c0, c1 = st.columns(2)
with c0: st.caption(f"Local (MYT): {my_now_str()}")
with c1: st.caption(f"US/Eastern: {et_now_str()}")

# Quick health proxy for last 1H bar time
last_bar_str, last_bar_sym = "n/a", None
try:
    for sym in ["SPY","QQQ","AAPL","MSFT"]:
        dfp = fetch_1h_chunks([sym], period="5d").get(sym, pd.DataFrame())
        if dfp is not None and not dfp.empty:
            last_bar_str = dfp.index[-1].strftime("%Y-%m-%d %H:%M")
            last_bar_sym = sym
            break
except Exception:
    pass
hA, hB = st.columns(2)
hA.metric("Last 1H bar", last_bar_str)
if last_bar_sym: hA.caption(f"from {last_bar_sym}")

# =========================
# Sidebar — Universe & Strategy Filters
# =========================
with st.sidebar:
    if not _AUTOREFRESH_OK:
        st.info("Auto-refresh module unavailable. Use Manual refresh.")
        if st.button("Manual refresh"):
            st.experimental_rerun()

    st.header("Universe (NASDAQ + NYSE only)")
    min_price = st.number_input("Min Price ($)", value=5.0)
    max_price = st.number_input("Max Price ($)", value=100.0)
    min_avg_dollar_vol = st.number_input("Min Avg Dollar Volume (20d)", value=30_000_000)
    max_atr_pct = st.number_input("Max 14d ATR%", value=12.0)
    min_atr_pct = st.number_input("Min 14d ATR%", value=1.5)
    max_symbols = st.slider("Max Universe Size", 200, 1200, 800, step=50)

    st.markdown("Manual tickers (optional, comma/space):")
    manual_syms = st.text_area("Symbols override", value="", height=60)

    rebuild_now = st.button("Rebuild Universe now")
    st.caption(f"Last rebuild (ET): {st.session_state.get('universe_built_at', 'never')}")

    st.divider()
    st.subheader("Strategy A — MACD 1H")
    rsi_max = st.number_input("RSI(14) max", value=60.0)
    adx_min = st.number_input("ADX(14) min", value=20.0)
    rvol_min = st.number_input("Same-hour RVOL min", value=1.2)
    require_ema_trend = st.checkbox("Require EMA10 > EMA20", value=True)
    daily_bias = st.checkbox("Require Daily EMA20 > EMA50", value=False)

    st.divider()
    st.subheader("Strategy B — Wife 1H")
    min_1h_volume = st.number_input("Min 1H Volume (shares)", value=1_000_000, step=50_000)
    max_rvol_after_cross = st.number_input("Max RVOL after Hist>0", value=2.0, step=0.1)
    min_market_cap = st.number_input("Min Market Cap ($)", value=1_000_000_000)

    st.divider()
    scan_top_n = st.slider("Scan top N by ADV", 50, 800, 400, step=50)

    st.divider()
    st.subheader("Output")
    auto_write_signals = st.checkbox("Auto-write Signals to Google Sheet", value=True)
    force_universe_sync = st.button("Sync Universe to Google Sheet now")

# =========================
# Build Universe (daily)
# =========================
et_today = et_now().date()
if manual_syms.strip():
    base_universe = [to_yahoo_symbol(s) for s in manual_syms.replace("\n"," ").replace(","," ").split() if s.strip()]
    source_symbols = base_universe
else:
    need_rebuild = (
        st.session_state.get("universe_built_date") != et_today
        or st.session_state.get("universe_cache") is None
        or rebuild_now
    )
    if need_rebuild:
        source_symbols = download_symbol_files_qn_only()
        try:
            base_universe, uni_stats = build_universe(
                min_price, max_price, min_avg_dollar_vol, min_atr_pct, max_atr_pct,
                max_symbols=max_symbols, return_stats=True
            )
        except TypeError:
            base_universe = build_universe(
                min_price, max_price, min_avg_dollar_vol, min_atr_pct, max_atr_pct,
                max_symbols=max_symbols, return_stats=False
            )
            uni_stats = None
        st.session_state["universe_cache"] = base_universe
        st.session_state["source_symbols_cache"] = source_symbols
        st.session_state["universe_built_date"] = et_today
        st.session_state["universe_built_at"] = et_now_str()
    else:
        base_universe = st.session_state.get("universe_cache", [])
        source_symbols = st.session_state.get("source_symbols_cache", [])

_src_count = len(source_symbols)
st.success(f"Universe size: {len(base_universe)} (from {_src_count} filtered Q/N symbols)")
if "uni_stats" in locals() and uni_stats:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Evaluated", uni_stats.get("evaluated", 0))
    sel = uni_stats.get("selected", 0); ev = max(1, uni_stats.get("evaluated", 1))
    c2.metric("Selected", f"{sel} ({100*sel/ev:.1f}%)")
    c3.metric("ADV ≥", f"${uni_stats.get('adv_threshold', 0):,}")
    lo, hi = uni_stats.get("atr_band", (min_atr_pct, max_atr_pct))
    c4.metric("ATR% band", f"{lo}–{hi}")

# Sync Universe (bulk, gated)
cur_hash = hash(tuple(base_universe))
now_et = et_now()
last_hash = st.session_state.get("universe_written_hash")
last_at   = st.session_state.get("universe_written_at")
ok_to_write_uni = (last_hash != cur_hash) and (last_at is None or (now_et - last_at).total_seconds() > 600)
if ((ok_to_write_uni and base_universe) or force_universe_sync) and not manual_syms.strip():
    ok, msg = write_universe_sheet(base_universe)
    if ok:
        st.session_state["universe_written_hash"] = cur_hash
        st.session_state["universe_written_at"]   = now_et
        st.caption(f"Universe synced to Google Sheet (bulk). {msg}")
else:
    st.caption("Universe not synced (unchanged/recently synced or manual symbols mode).")

# =========================
# Run Strategies
# =========================
st.divider()
st.subheader("Signals — latest 1H bar")

scan_list = base_universe[:scan_top_n] if base_universe else ["AAPL","MSFT","NVDA","AMD","SPY"]

# MACD_1H
with st.spinner("Scanning MACD 1H…"):
    macd_df = macd_1h_screener(
        scan_list, rsi_max=float(rsi_max), adx_min=float(adx_min),
        require_ema_trend=bool(require_ema_trend), rvol_min=float(rvol_min),
        daily_bias=bool(daily_bias)
    )
if not macd_df.empty:
    show = macd_df.copy()
    for col, d in [("Price",2),("RSI14",1),("MACD",4),("Signal",4),("Hist",4),("EMA10",2),("EMA20",2),("ADX14",1),("RVOL_same_hour",2)]:
        show[col] = show[col].apply(lambda x: formatn(x, d))
    st.markdown("**Strategy A — MACD_1H**")
    st.dataframe(show, use_container_width=True)
else:
    st.info("Strategy A — No MACD_1H signals on the latest bar.")

# Wife_1H
with st.spinner("Scanning Wife 1H…"):
    wife_df = wife_1h_screener(
        scan_list,
        min_1h_volume=int(min_1h_volume),
        max_rvol_after_cross=float(max_rvol_after_cross),
        min_market_cap=float(min_market_cap)
    )
if not wife_df.empty:
    show2 = wife_df.copy()
    for col, d in [("Price",2),("EMA5",2),("EMA20",2),("EMA50",2),("MACD",4),("Signal",4),("Hist",4),("K",1),("D",1),("J",1),("RVOL_same_hour",2)]:
        show2[col] = show2[col].apply(lambda x: formatn(x, d))
    st.markdown("**Strategy B — WIFE_1H**")
    st.dataframe(show2, use_container_width=True)
else:
    st.info("Strategy B — No WIFE_1H signals on the latest bar.")

# =========================
# Auto-write Signals (de-dup per bar)
# =========================
if auto_write_signals:
    # Determine last bar time string from header health probe (or any df)
    bar_time_str = last_bar_str if last_bar_str != "n/a" else (
        (macd_df["BarTime (ET)"].iloc[0] if not macd_df.empty else (wife_df["BarTime (ET)"].iloc[0] if not wife_df.empty else None))
    )
    combined = pd.concat([macd_df, wife_df], ignore_index=True) if not macd_df.empty or not wife_df.empty else pd.DataFrame()
    if bar_time_str and not combined.empty:
        last_written_bar = st.session_state.get("signals_last_bar_time")
        last_written_count = st.session_state.get("signals_last_count", 0)
        # Only write once per bar time and only if count changed
        if (bar_time_str != last_written_bar) or (len(combined) != last_written_count):
            # Order columns for sheet
            cols = [
                "BarTime (ET)", "Strategy", "Ticker", "Price",
                "RSI14", "MACD", "Signal", "Hist", "EMA10", "EMA20", "ADX14", "RVOL_same_hour", "DailyBias",
                "EMA5","EMA20","EMA50","K","D","J","Volume","MarketCap"
            ]
            for c in cols:
                if c not in combined.columns:
                    combined[c] = None
            combined = combined[cols]

            rows = combined.values.tolist()
            try:
                append_to_gsheet(rows, SHEET_SIGNALS)
                st.caption(f"Wrote {len(rows)} signal rows to Google Sheet for bar {bar_time_str}.")
                st.session_state["signals_last_bar_time"] = bar_time_str
                st.session_state["signals_last_count"] = len(rows)
            except Exception as e:
                st.warning(f"Signals write failed: {e}")
    else:
        st.caption("No signals to write for this bar.")

st.caption("End · 1H bars · Universe rebuilt daily (ET) · NASDAQ+NYSE only · Dot→Dash symbol fix for Yahoo")
