import requests
import time

if st.button("Test Finnhub Key"):
    url = "https://finnhub.io/api/v1/stock/candle"
    params = {
        "symbol": "AAPL",
        "resolution": "D",
        "from": 1700000000,
        "to": 1700500000,
        "token": FINNHUB_KEY
    }
    resp = requests.get(url, params=params)
    st.write("Status:", resp.status_code)
    st.json(resp.json())
