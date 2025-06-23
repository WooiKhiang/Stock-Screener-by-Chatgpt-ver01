import streamlit as st
import yfinance as yf

st.title("🔎 YFinance Intraday Data Debugger")

ticker = st.text_input("Ticker (e.g., AAPL, MSFT, SPY)", value="AAPL")
interval = st.selectbox("Interval", ["1d", "5m", "15m", "1h"], index=1)
period = st.selectbox("Period", ["5d", "1d", "1mo"], index=0)
go = st.button("Download Data")

if go:
    st.write(f"Fetching {ticker} for period={period}, interval={interval} ...")
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
        st.write("DataFrame shape:", df.shape)
        st.write("DataFrame columns:", df.columns.tolist())
        if df.empty:
            st.warning("Returned DataFrame is EMPTY! No data from yfinance.")
        else:
            st.dataframe(df.head(20))
            st.success("Data loaded successfully.")
    except Exception as e:
        st.error(f"Exception fetching data: {e}")
else:
    st.info("Select a ticker and press Download Data to test.")
