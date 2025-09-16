# streamlit_app.py
# Minimal dashboard that reads the latest universe from Google Sheet and explains the criteria.

import os
import json
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID", "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4")
GOOGLE_SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME", "Universe")

# (Display) same knobs as worker (read-only)
N_TOP = int(os.getenv("N_TOP", "100"))
MIN_PRICE = float(os.getenv("MIN_PRICE", "5"))
MAX_PRICE = float(os.getenv("MAX_PRICE", "100"))
INTRADAY_RANK = os.getenv("INTRADAY_RANK", "false").lower() == "true"
EXCLUDE_TOP_VOLUME_PCT = float(os.getenv("EXCLUDE_TOP_VOLUME_PCT", "1.5"))
GENTLE_MOMENTUM = os.getenv("GENTLE_MOMENTUM", "true").lower() == "true"
ALPACA_FEED = os.getenv("ALPACA_FEED", "iex")

st.set_page_config(page_title="Dynamic Universe (Alpaca) — Status", layout="centered")

st.title("🗂️ Dynamic Tickers — Hourly Universe")
st.caption("This page is read-only. A headless worker recomputes the tickers hourly and overwrites the sheet.")

# -------------------------
# Google auth (from secrets/env)
# -------------------------
def gs_client_from_env():
    # Prefer Streamlit Secrets; fallback to env var
    sa_json = st.secrets.get("GCP_SERVICE_ACCOUNT") or os.getenv("GCP_SERVICE_ACCOUNT", "")
    if not sa_json:
        st.error("Missing GCP_SERVICE_ACCOUNT in Streamlit secrets or env.")
        st.stop()
    data = json.loads(sa_json)
    creds = Credentials.from_service_account_info(
        data,
        scopes=["https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"]
    )
    return gspread.authorize(creds)

@st.cache_data(ttl=60, show_spinner=False)
def load_latest():
    client = gs_client_from_env()
    sh = client.open_by_key(GOOGLE_SHEET_ID)
    try:
        ws = sh.worksheet(GOOGLE_SHEET_NAME)
    except gspread.WorksheetNotFound:
        return {"timestamp": None, "tickers": []}
    vals = ws.get("A1:B1")
    if not vals or not vals[0]:
        return {"timestamp": None, "tickers": []}
    ts = vals[0][0] if len(vals[0]) > 0 else None
    tickers_csv = vals[0][1] if len(vals[0]) > 1 else ""
    tickers = [t.strip().upper() for t in tickers_csv.split(",") if t.strip()]
    return {"timestamp": ts, "tickers": tickers}

with st.expander("Selection Criteria (explained)", expanded=True):
    st.markdown(
        """
**Goal:** Focus on the “hidden picture”—skip the obvious/noisy extremes and keep liquid, actionable names.

**Filters used by the worker:**
- **Price band:** `$5 – $100`
- **Activity:** rank by **$-flow share** (avg `close × volume` vs all candidates)
- **Noise control (optional):** exclude the **top X% by raw volume** (defaults to 1.5%)
- **Gentle momentum:** last close ≥ previous close **OR** last close ≥ EMA20
- **Timeframe for ranking:** **Daily** over ~5 bars *(faster & stable)*, or optional 1-hour intraday *(slower)*
        """
    )

st.markdown("### Current Settings")
c1, c2, c3 = st.columns(3)
c1.metric("Top N", N_TOP)
c2.metric("Price Min ($)", MIN_PRICE)
c3.metric("Price Max ($)", MAX_PRICE)
c4, c5, c6 = st.columns(3)
c4.metric("Exclude Top Vol %", EXCLUDE_TOP_VOLUME_PCT)
c5.metric("Gentle Momentum", "On" if GENTLE_MOMENTUM else "Off")
c6.metric("Ranking TF", "1h" if INTRADAY_RANK else "1d")

st.divider()

data = load_latest()
ts = data["timestamp"]
tickers = data["tickers"]

st.markdown("### Latest Universe")
if ts:
    st.write(f"**Last run (ET):** {ts}")
else:
    st.warning("No timestamp found.")

st.write(f"**Tickers selected:** {len(tickers)}")
if tickers:
    preview = ", ".join(tickers[:50])
    st.code(preview, language="text")

# Simple download
if tickers:
    csv_line = ",".join(tickers)
    st.download_button("Download tickers (CSV row)", data=csv_line, file_name="tickers.csv", mime="text/csv")

st.caption("Updates hourly by GitHub Actions. This dashboard only reads from Google Sheets.")
