# streamlit_app.py — Dynamic Tickers (read-only; hard-coded Sheet)
import os, json, time
import streamlit as st

st.set_page_config(page_title="Dynamic Tickers — Status", layout="centered")
st.title("🗂️ Dynamic Tickers — Hourly Universe")

# -----------------------
# Hard-coded Google Sheet
# -----------------------
GOOGLE_SHEET_ID = "1zg3_-xhLi9KCetsA1KV0Zs7IRVIcwzWJ_s15CT2_eA4"
GOOGLE_SHEET_NAME = "Universe"

# Optional fallback (publish Sheet -> CSV and paste URL here if you want it)
CSV_FALLBACK_URL = st.secrets.get("CSV_URL") or os.getenv("CSV_URL")

# Service account JSON (keep in secrets or env ONLY)
SA_RAW = st.secrets.get("GCP_SERVICE_ACCOUNT") or os.getenv("GCP_SERVICE_ACCOUNT")

with st.expander("Selection Criteria (summary)", expanded=True):
    st.markdown("""
- **Price band:** $5–$100  
- **Rank:** by **$-flow share** (avg close×volume relative to pool)  
- **Noise control:** optionally exclude top ~1.5% by raw volume  
- **Gentle momentum:** last close ≥ prev close **or** ≥ EMA20  
- **Ranking TF:** daily (~5 bars) by default (intraday optional in worker)
    """)

# -----------------------
# Helpers
# -----------------------
def fail(msg: str):
    st.error(msg)
    st.stop()

def parse_service_account(raw: str):
    if not raw:
        fail("Missing **GCP_SERVICE_ACCOUNT** in secrets/env.")
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
        rows = ws.get("A:B")  # all rows (timestamp, csv tickers)
    except Exception as e:
        return {"ok": False, "err": f"Read error A:B: {e}", "ts": None, "tickers": []}

    # Find last non-empty row in col B, skipping header rows like "TICKER"
    last = None
    for r in reversed(rows):
        if len(r) >= 2 and r[1].strip():
            if r[1].strip().upper() == "TICKER":
                continue
            last = r
            break

    if not last:
        return {"ok": True, "err": "", "ts": None, "tickers": []}

    ts = last[0].strip() if last[0].strip() else None
    csv_line = last[1].strip()
    tickers = [t.strip().upper() for t in csv_line.split(",") if t.strip()]
    return {"ok": True, "err": "", "ts": ts, "tickers": tickers}

@st.cache_data(ttl=60, show_spinner=False)
def read_universe_from_csv(url: str):
    import pandas as pd
    try:
        df = pd.read_csv(url, header=None)
        # valid rows: non-empty col1 and not header label
        ser = df.iloc[:, 1].astype(str)
        mask = ser.str.strip().ne("").values & ser.str.upper().ne("TICKER").values
        valid = df[mask]
        if valid.empty:
            return {"ok": True, "err": "", "ts": None, "tickers": []}
        row = valid.tail(1).iloc[0]
        ts = str(row.iloc[0]) if len(row) >= 1 else None
        csv_line = str(row.iloc[1]) if len(row) >= 2 else ""
        tickers = [t.strip().upper() for t in csv_line.split(",") if t.strip()]
        return {"ok": True, "err": "", "ts": ts, "tickers": tickers}
    except Exception as e:
        return {"ok": False, "err": f"CSV fallback error: {e}", "ts": None, "tickers": []}

# -----------------------
# Minimal sanity
# -----------------------
colA, colB = st.columns(2)
colA.write("**Sheet ID set?** ✅")
colB.write(f"**Worksheet name:** `{GOOGLE_SHEET_NAME}`")

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
    # show up to 100 for quick copy
    st.code(", ".join(data["tickers"][:100]), language="text")

    st.download_button(
        "Download tickers (CSV row)",
        data=",".join(data["tickers"]),
        file_name="tickers.csv",
        mime="text/csv",
    )
else:
    if data["ok"]:
        st.warning("No tickers yet. Once the worker writes a row (A: timestamp, B: 'AAPL, MSFT, …'), results will appear.")
    else:
        st.error("Could not load data from Sheet or CSV.")

st.caption("Reads the last non-empty row from Google Sheets (worker writes hourly via GitHub Actions).")
