# streamlit_app.py — Dynamic Tickers (read-only dashboard, fast boot, lazy Google auth)
import os, json, time
import streamlit as st

st.set_page_config(page_title="Dynamic Tickers — Status", layout="centered")
st.title("🗂️ Dynamic Tickers — Hourly Universe")

# -----------------------
# Config from secrets/env
# -----------------------
GOOGLE_SHEET_ID = st.secrets.get("GOOGLE_SHEET_ID") or os.getenv("GOOGLE_SHEET_ID")
GOOGLE_SHEET_NAME = st.secrets.get("GOOGLE_SHEET_NAME") or os.getenv("GOOGLE_SHEET_NAME")
SA_RAW = st.secrets.get("GCP_SERVICE_ACCOUNT") or os.getenv("GCP_SERVICE_ACCOUNT")
CSV_FALLBACK_URL = st.secrets.get("CSV_URL") or os.getenv("CSV_URL")  # optional published CSV fallback

def fail(msg: str):
    st.error(msg)
    st.stop()

with st.expander("Selection Criteria (summary)", expanded=True):
    st.markdown("""
- **Price band:** $5–$100  
- **Rank:** by **$-flow share** (avg close×volume relative to pool)  
- **Noise control:** optionally exclude top ~1.5% by raw volume  
- **Gentle momentum:** last close ≥ prev close **or** ≥ EMA20  
- **Ranking TF:** daily (~5 bars) by default (intraday optional in worker)
    """)

# -----------------------
# Minimal sanity section
# -----------------------
colA, colB = st.columns(2)
colA.write(f"**Sheet ID set?** {'✅' if bool(GOOGLE_SHEET_ID) else '❌'}")
colB.write(f"**Worksheet name:** `{GOOGLE_SHEET_NAME or '—'}`")

if not (GOOGLE_SHEET_ID and GOOGLE_SHEET_NAME):
    fail("Missing **GOOGLE_SHEET_ID** or **GOOGLE_SHEET_NAME** in secrets.")

# -----------------------
# Lazy Google auth helpers
# -----------------------
def parse_service_account(raw: str):
    if not raw:
        fail("Missing **GCP_SERVICE_ACCOUNT** in secrets.")
    # Try parse; if it fails due to raw newlines inside the private key, auto-repair once
    def _repair_private_key(json_str: str) -> str:
        if "-----BEGIN PRIVATE KEY-----" in json_str and "\\n" not in json_str:
            block_start = json_str.find("-----BEGIN PRIVATE KEY-----")
            block_end = json_str.find("-----END PRIVATE KEY-----", block_start)
            if block_start != -1 and block_end != -1:
                block_end += len("-----END PRIVATE KEY-----")
                block = json_str[block_start:block_end]
                fixed = block.replace("\r\n", "\n").replace("\n", "\\n")
                json_str = json_str.replace(block, fixed)
        return json_str

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(_repair_private_key(raw))

@st.cache_resource(show_spinner=False)
def get_gs_client(sa: dict):
    # lazy imports to keep boot instant
    import gspread
    from google.oauth2.service_account import Credentials
    creds = Credentials.from_service_account_info(
        sa,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    return gspread.authorize(creds)

@st.cache_data(ttl=60, show_spinner=False)
def read_universe_from_sheet(sheet_id: str, sheet_name: str):
    client = get_gs_client(parse_service_account(SA_RAW))
    sh = client.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(sheet_name)
    except Exception as e:
        return {"ok": False, "err": f"Worksheet error: {e}", "ts": None, "tickers": []}
    try:
        vals = ws.get("A1:B1")
    except Exception as e:
        return {"ok": False, "err": f"Read error A1:B1: {e}", "ts": None, "tickers": []}
    if not vals or not vals[0] or len(vals[0]) < 2:
        return {"ok": True, "err": "", "ts": None, "tickers": []}
    ts = vals[0][0]
    csv_line = vals[0][1] or ""
    tickers = [t.strip().upper() for t in csv_line.split(",") if t.strip()]
    return {"ok": True, "err": "", "ts": ts, "tickers": tickers}

@st.cache_data(ttl=60, show_spinner=False)
def read_universe_from_csv(url: str):
    import pandas as pd
    try:
        df = pd.read_csv(url, header=None, nrows=1)
        ts = str(df.iloc[0, 0]) if df.shape[1] >= 1 else None
        csv_line = str(df.iloc[0, 1]) if df.shape[1] >= 2 else ""
        tickers = [t.strip().upper() for t in csv_line.split(",") if t.strip()]
        return {"ok": True, "err": "", "ts": ts, "tickers": tickers}
    except Exception as e:
        return {"ok": False, "err": f"CSV fallback error: {e}", "ts": None, "tickers": []}

# -----------------------
# Controls & data loading
# -----------------------
left, right = st.columns(2)
with left:
    if st.button("↻ Force refresh", help="Clear cache and reload from Google Sheet"):
        read_universe_from_sheet.clear()
        read_universe_from_csv.clear()
        time.sleep(0.15)

with st.spinner("Loading latest universe…"):
    data = read_universe_from_sheet(GOOGLE_SHEET_ID, GOOGLE_SHEET_NAME)

if not data["ok"]:
    st.error(data["err"])
    if CSV_FALLBACK_URL:
        st.info("Trying CSV fallback (published Sheet)…")
        data = read_universe_from_csv(CSV_FALLBACK_URL)

# -----------------------
# Display
# -----------------------
st.markdown("### Latest Universe")
st.write(f"**Last run (ET):** {data['ts'] or '—'}")
st.write(f"**Tickers selected:** {len(data['tickers'])}")

if data["tickers"]:
    st.code(", ".join(data["tickers"][:50]), language="text")
    st.download_button(
        "Download tickers (CSV row)",
        data=",".join(data["tickers"]),
        file_name="tickers.csv",
        mime="text/csv",
    )
else:
    if data["ok"]:
        st.warning("Sheet is empty. Once the GitHub Action writes A1:B1, results will appear here.")
    else:
        st.error("Could not load data from Sheet or CSV.")

st.caption("Reads from Google Sheets (worker writes hourly via GitHub Actions).")
