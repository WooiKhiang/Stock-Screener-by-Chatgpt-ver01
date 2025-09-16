# --- DEBUG: Finnhub key quick test ---
import requests
from datetime import datetime, timedelta, timezone

with st.sidebar:
    st.markdown("### 🔑 Debug: Finnhub Key Test")

    if st.button("Run Finnhub Test"):
        # Set time range (last 30 days)
        to_ts = int(datetime.now(timezone.utc).timestamp())
        from_ts = to_ts - 30 * 24 * 3600

        url = "https://finnhub.io/api/v1/stock/candle"
        params = {
            "symbol": "AAPL",          # US ticker (free plan should allow this)
            "resolution": "D",         # Daily bars (allowed on free plan)
            "from": from_ts,
            "to": to_ts,
            "token": FINNHUB_KEY,
        }

        st.write("📡 Requesting AAPL daily candles from Finnhub...")
        resp = requests.get(url, params=params, timeout=20)
        st.write("HTTP status:", resp.status_code)

        try:
            data = resp.json()
            st.json(data)
        except Exception as e:
            st.error(f"Could not parse JSON: {e}")
            st.text(resp.text[:500])
