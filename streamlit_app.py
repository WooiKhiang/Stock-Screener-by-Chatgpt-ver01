# streamlit_app.py — Alpaca-only SCREEN • Universe from Google Sheet • Ranked + Push back to Sheet
import os
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Dict

import numpy as np
import pandas as pd
import requests
import streamlit as st

# === Google Sheets ===
import gspread
from google.oauth2.service_account import Credentials

# =========================
# Config: API keys & bases
# =========================
ALPACA_KEY = st.secrets.get("ALPACA_KEY") or os.getenv("ALPACA_KEY", "")
ALPACA_SECRET = st.secrets.get("ALPACA_SECRET") or os.getenv("ALPACA_SECRET", "")
ALPACA_BASE = st.secrets.get("ALPACA_BASE") or os.getenv("ALPACA_BASE", "https://paper-api.alpaca.markets")
ALPACA_HEADERS = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}
ALPACA_DATA_BASE = "https://data.alpaca.markets"  # market data host

# === Google Sheets IDs (as requested) ===
GOOGLE_SHEET_ID = "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4"
UNIVERSE_SHEET_NAME = "Universe"
RESULT_SHEET_NAME = "Scanned Result"

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")

def _alpaca_tf(resolution: str) -> str:
    return {"5":"5Min","15":"15Min","30":"30Min","60":"1Hour","D":"1Day"}.get(resolution, "1Hour")

# =========================
# Google Sheets helpers (hardened)
# =========================
def _normalize_service_account_info(info_obj):
    """
    Accepts either a dict or JSON string. Normalizes private_key newlines.
    Raises a descriptive RuntimeError on common formatting problems.
    """
    # Convert JSON string to dict if needed
    if isinstance(info_obj, str):
        try:
            info = json.loads(info_obj)
        except Exception as e:
            raise RuntimeError(f"Service account JSON is not valid JSON. ({e})")
    else:
        info = dict(info_obj)  # shallow copy

    if "private_key" not in info or not info["private_key"]:
        raise RuntimeError("Service account is missing 'private_key'.")

    pk = info["private_key"]

    # If the key contains literal backslash-n (\\n), convert to real newlines
    if "\\n" in pk and "\n" not in pk:
        pk = pk.replace("\\n", "\n")

    # Ensure it has proper header/footer and trailing newline
    if "BEGIN PRIVATE KEY" not in pk or "END PRIVATE KEY" not in pk:
        raise RuntimeError(
            "private_key appears truncated or missing BEGIN/END lines. "
            "Please copy it exactly from Google Cloud without extra escaping."
        )
    if not pk.endswith("\n"):
        pk += "\n"

    info["private_key"] = pk
    return info

def _get_gspread_client():
    """
    Expects service account in either:
      - st.secrets['gcp_service_account'] (preferred; dict-like section in secrets.toml)
      - env var GOOGLE_APPLICATION_CREDENTIALS_JSON (full JSON string)
    Normalizes newlines and builds Credentials.
    """
    raw = st.secrets.get("gcp_service_account")
    if not raw:
        raw = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "")
        if not raw:
            raise RuntimeError(
                "Missing Google credentials. Add [gcp_service_account] in .streamlit/secrets.toml "
                "or set GOOGLE_APPLICATION_CREDENTIALS_JSON env var."
            )

    info = _normalize_service_account_info(raw)

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    try:
        credentials = Credentials.from_service_account_info(info, scopes=scopes)
        gc = gspread.authorize(credentials)
    except Exception as e:
        raise RuntimeError(
            f"Failed to build Google credentials. Root cause: {e}\n"
            "Fix tips:\n"
            "- Ensure secrets.toml uses escaped newlines: private_key = \"-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n\"\n"
            "- Or paste exact JSON into GOOGLE_APPLICATION_CREDENTIALS_JSON env var.\n"
            "- Share the Sheet with the service account email."
        )
    return gc

@st.cache_data(show_spinner=False)
def load_universe_from_sheet(sheet_id: str, tab_name: str) -> List[str]:
    """
    Reads tickers from the given Google Sheet tab.
    Collects all non-empty cells, uppercases, filters to ASCII+[-._].
    """
    gc = _get_gspread_client()
    ws = gc.open_by_key(sheet_id).worksheet(tab_name)
    values = ws.get_all_values()
    tickers = []
    for row in values:
        for cell in row:
            sym = cell.strip().upper()
            if sym and sym.isascii():
                if all(ch.isalnum() or ch in (".", "-", "_") for ch in sym):
                    tickers.append(sym)
    # Deduplicate preserving order
    seen, uniq = set(), []
    for t in tickers:
        if t not in seen:
            uniq.append(t); seen.add(t)
    return uniq

def write_results_to_sheet(sheet_id: str, tab_name: str, df: pd.DataFrame):
    """
    Overwrites the entire target tab with the current scan result, including a timestamp column.
    If the tab doesn't exist, it will be created.
    """
    gc = _get_gspread_client()
    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(tab_name)
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=tab_name, rows=str(len(df) + 10), cols=str(len(df.columns) + 5))

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_out = df.copy()
    df_out.insert(0, "timestamp", timestamp)

    sheet_values = [list(df_out.columns)] + df_out.astype(object).where(pd.notnull(df_out), "").values.tolist()
    ws.update("A1", sheet_values, value_input_option="RAW")

# =========================
# Alpaca market data (bars)
# =========================
def get_candles_alpaca(symbol: str, resolution: str = "60", lookback_days: int = 60) -> pd.DataFrame:
    if not (ALPACA_KEY and ALPACA_SECRET):
        raise RuntimeError("Alpaca keys missing; cannot fetch Alpaca data.")
    tf = _alpaca_tf(resolution)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    url = f"{ALPACA_DATA_BASE}/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": tf,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "limit": 10000,
        "feed": "iex",
        "adjustment": "all",
    }
    r = requests.get(url, headers=ALPACA_HEADERS, params=params, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"{symbol} bars error {r.status_code}: {r.text}")
    bars = r.json().get("bars", [])
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(
        [{"t": pd.to_datetime(b["t"], utc=True), "open": b["o"], "high": b["h"],
          "low": b["l"], "close": b["c"], "volume": b["v"]} for b in bars]
    )
    return df.set_index("t").sort_index()

# =========================
# Indicators & rules (same as before)
# =========================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(close, fast); ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def stochastic_kd(df: pd.DataFrame, period: int = 14, k_smooth: int = 3, d_smooth: int = 3):
    lowest_low = df["low"].rolling(window=period).min()
    highest_high = df["high"].rolling(window=period).max()
    fast_k = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low)
    k = fast_k.rolling(window=k_smooth).mean()
    d = k.rolling(window=d_smooth).mean()
    return k, d

def kdj(df: pd.DataFrame, period: int = 14):
    k, d = stochastic_kd(df, period=period)
    j = 3 * k - 2 * d
    return k, d, j

@dataclass
class Rules:
    ema_fast: int = 5
    ema_mid: int = 20
    ema_slow: int = 50
    require_ema_rising: bool = True
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    macd_hist_must_shrink_neg: bool = True
    macd_line_gt_signal: bool = True
    kdj_period: int = 14
    kdj_j_max: float = 80.0
    min_volume: int = 1_000_000
    rsi_min: float = 40.0
    rsi_max: float = 60.0

def apply_rules(df: pd.DataFrame, rules: Rules) -> Dict[str, any]:
    result = {
        "passes": False,
        "vol_ok": False,
        "ema_bullish": False,
        "ema_rising": False,
        "macd_ok": False,
        "kdj_ok": False,
        "rsi_ok": False,
        "why": [],
    }
    need = max(rules.ema_slow, rules.kdj_period, rules.macd_slow) + 5
    if df.empty or len(df) < need:
        result["why"].append("insufficient data")
        return result

    df = df.copy()
    df["ema_fast"] = ema(df["close"], rules.ema_fast)
    df["ema_mid"]  = ema(df["close"], rules.ema_mid)
    df["ema_slow"] = ema(df["close"], rules.ema_slow)

    macd_line, macd_signal, hist = macd(df["close"], rules.macd_fast, rules.macd_slow, rules.macd_signal)
    df["macd_line"], df["macd_signal"], df["macd_hist"] = macd_line, macd_signal, hist

    df["rsi14"] = rsi(df["close"], 14)

    k, d = stochastic_kd(df, period=rules.kdj_period)
    j = 3 * k - 2 * d
    df["K"], df["D"], df["J"] = k, d, j

    last, prev = df.iloc[-1], df.iloc[-2]

    # Volume
    result["vol_ok"] = bool(last["volume"] >= rules.min_volume)
    if not result["vol_ok"]:
        result["why"].append(f"volume {int(last['volume'])} < {rules.min_volume}")

    # EMA stack + rising
    result["ema_bullish"] = bool(last["ema_fast"] > last["ema_mid"] > last["ema_slow"])
    if not result["ema_bullish"]:
        result["why"].append("EMA order not bullish (fast>mid>slow)")
    if rules.require_ema_rising:
        result["ema_rising"] = bool(last["ema_fast"] > prev["ema_fast"] and
                                    last["ema_mid"]  > prev["ema_mid"]  and
                                    last["ema_slow"] > prev["ema_slow"])
        if not result["ema_rising"]:
            result["why"].append("EMAs not rising vs previous bar")
    else:
        result["ema_rising"] = True

    # MACD
    macd_hist_shrinking_neg = (last["macd_hist"] > prev["macd_hist"] and last["macd_hist"] < 0) if rules.macd_hist_must_shrink_neg else True
    macd_line_rising_gt_sig = (last["macd_line"] > last["macd_signal"] and last["macd_line"] > prev["macd_line"]) if rules.macd_line_gt_signal else True
    result["macd_ok"] = bool(macd_hist_shrinking_neg and macd_line_rising_gt_sig)
    if not result["macd_ok"]:
        result["why"].append("MACD not in desired state")

    # KDJ
    kdj_bullish_not_hot = (last["K"] > last["D"] and last["K"] > prev["K"] and last["D"] > prev["D"] and (3*last["K"]-2*last["D"]) > (3*prev["K"]-2*prev["D"]) and last["J"] < rules.kdj_j_max)
    result["kdj_ok"] = bool(kdj_bullish_not_hot)
    if not result["kdj_ok"]:
        result["why"].append("KDJ not bullish-but-not-hot")

    # RSI
    result["rsi_ok"] = bool(rules.rsi_min <= last["rsi14"] <= rules.rsi_max)
    if not result["rsi_ok"]:
        result["why"].append(f"RSI {last['rsi14']:.1f} outside [{rules.rsi_min},{rules.rsi_max}]")

    result["passes"] = all([result["vol_ok"], result["ema_bullish"], result["ema_rising"], result["macd_ok"], result["kdj_ok"], result["rsi_ok"]])
    result["last_close"] = float(last["close"])
    result["last_rsi"] = float(last["rsi14"])
    result["last_j"] = float(last["J"])
    result["macd_hist_delta"] = float(last["macd_hist"] - prev["macd_hist"])
    result["volume"] = int(last["volume"])
    return result

def confidence_score(res: Dict[str, any], rules: Rules) -> float:
    base = (1*res["vol_ok"] + 2*res["ema_bullish"] + 1*res["ema_rising"] +
            2*res["macd_ok"] + 2*res["kdj_ok"] + 1*res["rsi_ok"])
    rsi = res.get("last_rsi", 50.0)
    rsi_bonus = 1.0 - min(abs(rsi - 50.0), 50.0)/50.0  # 0..1
    delta = res.get("macd_hist_delta", 0.0)
    macd_bonus = 1.0/(1.0 + np.exp(-10.0*delta))       # 0..1
    j = res.get("last_j", rules.kdj_j_max)
    j_bonus = max(0.0, min(1.0, (rules.kdj_j_max - j)/rules.kdj_j_max))
    return float(base + 0.6*rsi_bonus + 0.3*macd_bonus + 0.1*j_bonus)

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Screener — Google Sheet Universe", layout="wide")
st.title("📈 Screener — Google Sheet Universe (Alpaca data)")
st.caption("Reads tickers from Google Sheets → applies EMA/MACD/KDJ/RSI+Volume → ranks by confidence → writes results back to Sheets.")

with st.sidebar:
    st.header("Bars & Lookback")
    resolution = st.selectbox("Bar timeframe", ["60","30","15","5","D"], index=0)
    lookback_days = st.slider("Lookback days", 20, 180, 60)

    st.header("Rules")
    ema_fast = st.number_input("EMA fast", 3, 50, 5)
    ema_mid  = st.number_input("EMA mid", 5, 100, 20)
    ema_slow = st.number_input("EMA slow", 10, 200, 50)
    require_rising = st.checkbox("Require EMAs rising", True)

    macd_fast = st.number_input("MACD fast", 5, 50, 12)
    macd_slow = st.number_input("MACD slow", 10, 100, 26)
    macd_signal = st.number_input("MACD signal", 5, 50, 9)
    macd_shrink = st.checkbox("MACD hist negative but shrinking", True)

    kdj_period = st.number_input("KDJ period", 5, 50, 14)
    kdj_j_max = st.number_input("KDJ J max", 50.0, 100.0, 80.0)

    rsi_min = st.number_input("RSI min", 0.0, 100.0, 40.0)
    rsi_max = st.number_input("RSI max", 0.0, 100.0, 60.0)

    min_volume = st.number_input("Min Volume (last bar)", 0, 50_000_000, 1_000_000, step=100_000)

    st.divider()
    st.header("Actions")
    run_scan = st.button("🔍 Load From Sheet & Run Screener", type="primary")

st.markdown("### Strategy Criteria")
st.markdown("""
- **Volume**: Last bar ≥ *Min Volume* (default **1,000,000**)
- **EMAs**: Bullish stack (**EMA fast > EMA mid > EMA slow**); **rising vs previous bar** (toggle)
- **MACD**: Histogram **negative but shrinking** *(hist < 0 and hist increases)*; **MACD line > signal** and **rising**
- **KDJ**: **K > D**, K & D rising, **J rising** but **J < J max** (default **80**)
- **RSI(14)**: within **[RSI min, RSI max]** (default **40–60**)
""")

rules = Rules(
    ema_fast=ema_fast, ema_mid=ema_mid, ema_slow=ema_slow, require_ema_rising=require_rising,
    macd_fast=macd_fast, macd_slow=macd_slow, macd_signal=macd_signal,
    macd_hist_must_shrink_neg=macd_shrink, macd_line_gt_signal=True,
    kdj_period=kdj_period, kdj_j_max=kdj_j_max,
    min_volume=min_volume, rsi_min=rsi_min, rsi_max=rsi_max
)

st.markdown("---")

if run_scan:
    # Quick diagnostics to help with auth issues
    with st.expander("Diagnostics (auth)"):
        svc = st.secrets.get("gcp_service_account")
        email = None
        try:
            if svc:
                if isinstance(svc, str):
                    svc_obj = json.loads(svc)
                else:
                    svc_obj = dict(svc)
                email = svc_obj.get("client_email")
            else:
                env_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "")
                if env_json:
                    svc_obj = json.loads(env_json)
                    email = svc_obj.get("client_email")
            st.write({"service_account_email": email or "(not found)"})
            st.caption("Make sure this email has at least Viewer access to the Sheet.")
        except Exception as e:
            st.write({"diagnostic_error": str(e)})

    if not (ALPACA_KEY and ALPACA_SECRET):
        st.error("Alpaca keys missing; set ALPACA_KEY/ALPACA_SECRET/ALPACA_BASE in Secrets."); st.stop()

    try:
        with st.spinner("Loading tickers from Google Sheet…"):
            tickers = load_universe_from_sheet(GOOGLE_SHEET_ID, UNIVERSE_SHEET_NAME)
        if not tickers:
            st.warning("No tickers found in the Universe sheet. Please populate it and rerun.")
            st.stop()

        st.success(f"Loaded {len(tickers)} tickers from Google Sheet.")
        results_rows = []
        progress = st.progress(0.0)

        for i, sym in enumerate(tickers, start=1):
            try:
                df = get_candles_alpaca(sym, resolution=resolution, lookback_days=lookback_days)
                res = apply_rules(df, rules)
                row = {
                    "symbol": sym,
                    "passes": res["passes"],
                    "confidence": round(confidence_score(res, rules), 3),
                    "last_close": res.get("last_close", float("nan")),
                    "volume": res.get("volume", 0),
                    "vol_ok": res["vol_ok"],
                    "ema_bullish": res["ema_bullish"],
                    "ema_rising": res["ema_rising"],
                    "macd_ok": res["macd_ok"],
                    "kdj_ok": res["kdj_ok"],
                    "rsi_ok": res["rsi_ok"],
                    "rsi": round(res.get("last_rsi", np.nan), 2) if not np.isnan(res.get("last_rsi", np.nan)) else "",
                    "J": round(res.get("last_j", np.nan), 2) if not np.isnan(res.get("last_j", np.nan)) else "",
                    "reasons": "; ".join(res["why"]),
                }
                results_rows.append(row)
                time.sleep(0.08)
            except Exception as e:
                results_rows.append({
                    "symbol": sym, "passes": False, "confidence": 0.0, "last_close": float("nan"),
                    "volume": 0, "vol_ok": False, "ema_bullish": False, "ema_rising": False,
                    "macd_ok": False, "kdj_ok": False, "rsi_ok": False, "rsi": "", "J": "",
                    "reasons": f"error: {e}"
                })
            progress.progress(i / max(len(tickers), 1))

        df_res = pd.DataFrame(results_rows)
        df_res = df_res.sort_values(by=["passes", "confidence", "symbol"], ascending=[False, False, True]).reset_index(drop=True)

        st.subheader("Ranked Results (highest confidence first)")
        st.dataframe(df_res, use_container_width=True)

        csv_bytes = df_res.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results as CSV",
            data=csv_bytes,
            file_name=f"scan_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

        with st.spinner("Writing results to Google Sheet…"):
            write_results_to_sheet(GOOGLE_SHEET_ID, RESULT_SHEET_NAME, df_res)
        st.success(f"Results written to sheet '{RESULT_SHEET_NAME}' with timestamp.")

    except Exception as e:
        st.error(f"Scan failed: {e}")

else:
    st.info("Click **Load From Sheet & Run Screener** to scan the tickers listed in your Google Sheet.")

st.markdown("---")
st.caption(f"Build finished at {utcnow_iso()} — Data via Alpaca Market Data (IEX feed).")
