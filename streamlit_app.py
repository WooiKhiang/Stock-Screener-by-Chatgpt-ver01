import os, json, time
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Dynamic Tickers — Status", layout="centered")
st.title("🗂️ Dynamic Tickers — Hourly Universe")

GOOGLE_SHEET_ID = st.secrets.get("GOOGLE_SHEET_ID") or os.getenv("GOOGLE_SHEET_ID")
GOOGLE_SHEET_NAME = st.secrets.get("GOOGLE_SHEET_NAME") or os.getenv("GOOGLE_SHEET_NAME")

def _fail(msg: str):
    st.error(msg)
    st.stop()

# 1) Secrets presence checks (fail-fast)
sa_json = st.secrets.get("GCP_SERVICE_ACCOUNT") or os.getenv("GCP_SERVICE_ACCOUNT")
if not sa_json:
    _fail("Missing **GCP_SERVICE_ACCOUNT** in Streamlit secrets. Paste the entire service-account JSON on one line.")
if not GOOGLE_SHEET_ID:
    _fail("Missing **GOOGLE_SHEET_ID** in secrets.")
if not GOOGLE_SHEET_NAME:
    _fail("Missing **GOOGLE_SHEET_NAME** in secrets.")

# 2) Parse service-account JSON (catch formatting errors)
try:
    sa = json.loads(sa_json)
except Exception as e:
    _fail(f"Invalid GCP_SERVICE_ACCOUNT JSON: {e}")

# 3) Build client
try:
    creds = Credentials.from_service_account_info(
        sa,
        scopes=["https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"]
    )
    client = gspread.authorize(creds)
except Exception as e:
    _fail(f"Google auth error: {e}")

@st.cache_data(ttl=60, show_spinner=False)
def load_latest():
    sh = client.open_by_key(GOOGLE_SHEET_ID)
    try:
        ws = sh.worksheet(GOOGLE_SHEET_NAME)
    except gspread.WorksheetNotFound:
        return {"timestamp": None, "tickers": [], "note": "Worksheet not found"}
    vals = ws.get("A1:B1")
    if not vals or not vals[0]:
        return {"timestamp": None, "tickers": [], "note": "No data in A1:B1"}
    ts = vals[0][0] if len(vals[0]) > 0 else None
    tickers_csv = vals[0][1] if len(vals[0]) > 1 else ""
    tickers = [t.strip().upper() for t in tickers_csv.split(",") if t.strip()]
    return {"timestamp": ts, "tickers": tickers, "note": ""}

with st.expander("Selection Criteria (summary)", expanded=True):
    st.markdown("""
- **Price band:** $5–$100  
- **Rank:** by $-flow share = avg(close×volume) vs pool  
- **Noise control:** optionally exclude top ~1.5% by raw volume  
- **Gentle momentum:** last close ≥ prev close **or** ≥ EMA20  
- **Ranking TF:** daily (~5 bars) by default (intraday optional in worker)
    """)

colx, coly = st.columns([1,1])
with colx:
    if st.button("↻ Force refresh"):
        load_latest.clear()
        time.sleep(0.2)

data = load_latest()
ts = data["timestamp"]; tickers = data["tickers"]
note = data.get("note","")

st.markdown("### Latest Universe")
if note:
    st.info(note)
st.write(f"**Last run (ET):** {ts or '—'}")
st.write(f"**Tickers selected:** {len(tickers)}")
if tickers:
    st.code(", ".join(tickers[:50]), language="text")
    st.download_button("Download tickers (CSV row)", data=",".join(tickers), file_name="tickers.csv", mime="text/csv")
else:
    st.warning("Empty universe so far. Once the GitHub Action writes A1:B1, it will appear here.")
st.caption("Reads from Google Sheets. The hourly writer runs in GitHub Actions.")
