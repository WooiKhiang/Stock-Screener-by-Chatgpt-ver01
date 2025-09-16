# worker_universe.py
# Recompute dynamic US tickers and overwrite a Google Sheet (2 columns: timestamp (ET), tickers)
# Runs cleanly in GitHub Actions on an hourly schedule.

import os
import json
import time
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import requests
import pytz
import gspread
from google.oauth2.service_account import Credentials

# -------------------------
# Config via env variables
# -------------------------
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID", "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4")
GOOGLE_SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME", "Universe")

ALPACA_KEY = os.getenv("ALPACA_KEY", "")
ALPACA_SECRET = os.getenv("ALPACA_SECRET", "")
ALPACA_BASE = (os.getenv("ALPACA_BASE", "https://paper-api.alpaca.markets")).rstrip("/")
TRADING_V2 = ALPACA_BASE if ALPACA_BASE.endswith("/v2") else (ALPACA_BASE + "/v2")
ALPACA_DATA_BASE = "https://data.alpaca.markets"

# Universe parameters (tunable via env)
N_TOP = int(os.getenv("N_TOP", "100"))
MIN_PRICE = float(os.getenv("MIN_PRICE", "5"))
MAX_PRICE = float(os.getenv("MAX_PRICE", "100"))
INTRADAY_RANK = os.getenv("INTRADAY_RANK", "false").lower() == "true"
EXCLUDE_TOP_VOLUME_PCT = float(os.getenv("EXCLUDE_TOP_VOLUME_PCT", "1.5"))  # exclude top X% by raw volume
GENTLE_MOMENTUM = os.getenv("GENTLE_MOMENTUM", "true").lower() == "true"
START_DAYS_DAILY = int(os.getenv("START_DAYS_DAILY", "5"))  # lookback days for daily mode

IEX_OR_SIP = os.getenv("ALPACA_FEED", "iex")  # 'iex' is fine for most

HEADERS = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}

if not (ALPACA_KEY and ALPACA_SECRET):
    raise SystemExit("Missing ALPACA_KEY / ALPACA_SECRET")

# -------------------------
# Google auth
# -------------------------
def gs_client_from_env():
    sa_json = os.getenv("GCP_SERVICE_ACCOUNT", "")
    if not sa_json:
        raise SystemExit("Missing GCP_SERVICE_ACCOUNT (JSON) in env")
    data = json.loads(sa_json)
    creds = Credentials.from_service_account_info(
        data,
        scopes=["https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"]
    )
    return gspread.authorize(creds)

# -------------------------
# Helpers
# -------------------------
def _chunk(lst: List[str], size: int):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

def list_active_us_equities(limit_to: int = 5000, require_tradable: bool = True, require_etb: bool = False) -> List[str]:
    """Pull active assets and return symbols."""
    params = {"status": "active", "asset_class": "us_equity"}
    for attempt in range(2):
        r = requests.get(f"{TRADING_V2}/assets", headers=HEADERS, params=params, timeout=30)
        if r.status_code == 200:
            assets = r.json()
            syms = []
            for a in assets:
                if require_tradable and not a.get("tradable", False):
                    continue
                if require_etb and not a.get("easy_to_borrow", False):
                    continue
                sym = a.get("symbol")
                if not sym:
                    continue
                # keep "BRK.B" (Alpaca uses dot classes); skip other dotted except .B
                if "." in sym and not sym.endswith(".B"):
                    continue
                syms.append(sym)
                if len(syms) >= limit_to:
                    break
            return syms
        time.sleep(0.7)
    raise RuntimeError(f"Assets error {r.status_code}: {r.text}")

def top_n_by_dollar_volume(
    n: int = 100,
    min_price: float = 5.0,
    max_price: float = 100.0,
    start_days: int = 5,
    use_intraday: bool = False,
    exclude_top_volume_pct: float = 0.0,
    require_gentle_momentum: bool = True,
    feed: str = "iex",
) -> List[str]:
    """
    Rank symbols by recent dollar volume share:
      - Filters by price band
      - Optional exclusion of top X% by raw volume (noise)
      - Optional gentle momentum filter (last close >= prev close or close >= EMA20)
    Returns Top N symbols.
    """
    symbols = list_active_us_equities(limit_to=5000, require_tradable=True, require_etb=False)
    if not symbols:
        return []

    tf = "1Hour" if use_intraday else "1Day"
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=(3 if use_intraday else start_days))

    ranked: List[Tuple[str, float, float, float]] = []  # (sym, adv, last_price, last_raw_vol)
    momentum_pass: Dict[str, bool] = {}

    for batch in _chunk(symbols, 200):
        params = {
            "timeframe": tf,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": 1000,
            "feed": feed,
            "adjustment": "all",
            "symbols": ",".join(batch),
        }
        # small retry for 429/5xx
        ok = False
        for attempt in range(2):
            r = requests.get(f"{ALPACA_DATA_BASE}/v2/stocks/bars", headers=HEADERS, params=params, timeout=40)
            if r.status_code == 200:
                ok = True
                break
            time.sleep(0.8)
        if not ok:
            continue

        payload = r.json()
        bars_by_sym = payload.get("bars", {})
        for sym, bars in bars_by_sym.items():
            if not bars:
                continue
            df = pd.DataFrame(bars)
            if df.empty or "c" not in df or "v" not in df:
                continue
            df["t"] = pd.to_datetime(df["t"], utc=True)
            df = df.sort_values("t")

            last_price = float(df["c"].iloc[-1])
            if last_price < min_price or last_price > max_price:
                continue

            adv = float((df["c"] * df["v"]).mean())  # average dollar volume
            last_vol = float(df["v"].iloc[-1])

            if require_gentle_momentum and len(df) >= 21:
                ema20 = df["c"].ewm(span=20, adjust=False).mean()
                last_c = float(df["c"].iloc[-1])
                prev_c = float(df["c"].iloc[-2])
                momentum_pass[sym] = (last_c >= prev_c) or (last_c >= float(ema20.iloc[-1]))
            else:
                momentum_pass[sym] = True

            ranked.append((sym, adv, last_price, last_vol))

        time.sleep(0.20)
        if len(ranked) > 8000:
            break

    if not ranked:
        return []

    # Exclude top X% raw volume (noise magnets)
    if exclude_top_volume_pct and exclude_top_volume_pct > 0:
        vols = np.array([r[3] for r in ranked], dtype=float)
        cutoff = np.quantile(vols, 1.0 - (exclude_top_volume_pct / 100.0))
        ranked = [r for r in ranked if r[3] < cutoff]

    total_adv = float(sum(r[1] for r in ranked)) or 1.0
    scored = []
    for sym, adv, _px, _vol in ranked:
        if not momentum_pass.get(sym, True):
            continue
        share = adv / total_adv
        scored.append((sym, adv, share))

    scored.sort(key=lambda x: (x[2], x[1]), reverse=True)

    # Fail-soft: if too few after momentum, relax momentum once
    if len(scored) < max(20, int(0.3 * n)):
        scored = []
        for sym, adv, _px, _vol in ranked:
            share = adv / total_adv
            scored.append((sym, adv, share))
        scored.sort(key=lambda x: (x[2], x[1]), reverse=True)

    return [s for s, _adv, _share in scored[:n]]

def now_et_str() -> str:
    tz = pytz.timezone("US/Eastern")
    return datetime.now(timezone.utc).astimezone(tz).strftime("%Y-%m-%d %H:%M:%S %Z")

def write_sheet_two_cols(timestamp_et: str, tickers: List[str]):
    client = gs_client_from_env()
    sh = client.open_by_key(GOOGLE_SHEET_ID)
    try:
        ws = sh.worksheet(GOOGLE_SHEET_NAME)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=GOOGLE_SHEET_NAME, rows=1, cols=2)

    # Erase and write a single row (A1:B1)
    ws.clear()
    csv_line = ",".join(tickers)
    ws.update("A1:B1", [[timestamp_et, csv_line]])

def main():
    print(f"[worker] Building dynamic universe: N_TOP={N_TOP}, price=[{MIN_PRICE},{MAX_PRICE}], "
          f"intraday={INTRADAY_RANK}, excl_top_vol%={EXCLUDE_TOP_VOLUME_PCT}, gentle_momo={GENTLE_MOMENTUM}")

    syms = top_n_by_dollar_volume(
        n=N_TOP,
        min_price=MIN_PRICE,
        max_price=MAX_PRICE,
        start_days=START_DAYS_DAILY,
        use_intraday=INTRADAY_RANK,
        exclude_top_volume_pct=EXCLUDE_TOP_VOLUME_PCT,
        require_gentle_momentum=GENTLE_MOMENTUM,
        feed=IEX_OR_SIP,
    )

    if not syms:
        # We still write, but mark empty to be explicit
        print("[worker] WARNING: No symbols selected. Writing empty list to sheet.")
    ts = now_et_str()
    write_sheet_two_cols(ts, syms)
    print(f"[worker] Wrote {len(syms)} tickers at {ts}")

if __name__ == "__main__":
    main()
