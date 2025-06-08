# --- Step 2: Institutional Accumulation Screener (always runs for testing) ---

watchlist = ["SPY", "QQQ", "NVDA", "AAPL", "TSLA", "AMD", "META", "AMZN", "MSFT", "GOOGL"]

def get_intraday_data(ticker):
    try:
        df = yf.download(ticker, period="2d", interval="5m", progress=False)
    except Exception:
        return None
    return df

def scan_stock(ticker, spy_change):
    df = get_intraday_data(ticker)
    if df is None or len(df) < 22:
        return None

    today = df[df.index.date == df.index[-1].date()]
    if today.empty or len(today) < 10:
        return None

    # Today's % change
    try:
        today_open = float(today["Open"].iloc[0])
        today_close = float(today["Close"].iloc[-1])
        today_change = (today_close - today_open) / today_open * 100
    except Exception:
        today_change = 0

    # Relative Strength Score
    rs_score = today_change - spy_change

    # Volume Spike (any 5m bar in last hour >2x average)
    try:
        last_hour = today.iloc[-12:]
        avg_vol = float(today["Volume"].rolling(10).mean().iloc[-1])
        volume_spike = any(last_hour["Volume"] > 2 * avg_vol)
    except Exception:
        volume_spike = False

    # Above VWAP
    try:
        vwap = (today["Volume"] * (today["High"] + today["Low"] + today["Close"]) / 3).cumsum() / today["Volume"].cumsum()
        price_above_vwap = today_close > float(vwap.iloc[-1])
    except Exception:
        price_above_vwap = False

    # Above EMAs (10, 20, 50)
    try:
        ema10 = float(today["Close"].ewm(span=10).mean().iloc[-1])
        ema20 = float(today["Close"].ewm(span=20).mean().iloc[-1])
        ema50 = float(today["Close"].ewm(span=50).mean().iloc[-1])
        price_above_ema10 = today_close > ema10
        price_above_ema20 = today_close > ema20
        price_above_ema50 = today_close > ema50
    except Exception:
        price_above_ema10 = price_above_ema20 = price_above_ema50 = False

    # MACD (12,26,9) bullish?
    try:
        exp12 = today["Close"].ewm(span=12).mean()
        exp26 = today["Close"].ewm(span=26).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9).mean()
        macd_bullish = float(macd.iloc[-1]) > float(signal.iloc[-1])
    except Exception:
        macd_bullish = False

    # Reasoning
    reasons = []
    if rs_score > 0:
        reasons.append("Strong RS")
    if price_above_vwap:
        reasons.append("VWAP Reclaim")
    if price_above_ema10 and price_above_ema20:
        reasons.append("EMA Alignment")
    if volume_spike:
        reasons.append("Volume Spike")
    if macd_bullish:
        reasons.append("MACD Bullish")

    # Less strict for debug (only needs RS, EMA10, EMA20)
    all_criteria = (rs_score > 0 and price_above_ema10 and price_above_ema20)

    return {
        "Ticker": ticker,
        "Change %": f"{today_change:.2f}",
        "RS vs SPY": f"{rs_score:.2f}",
        "VWAP": "Yes" if price_above_vwap else "No",
        "EMA Align": "Yes" if price_above_ema10 and price_above_ema20 else "No",
        "Volume Spike": "Yes" if volume_spike else "No",
        "MACD Bullish": "Yes" if macd_bullish else "No",
        "Reasons": ", ".join(reasons),
        "Qualified": all_criteria
    }

st.subheader("Institutional Accumulation Screener (DEBUG)")

results = []
all_results = []
for ticker in watchlist:
    result = scan_stock(ticker, spy_change)
    if result:
        all_results.append(result)
        if result["Qualified"]:
            results.append(result)

if results:
    st.success("Qualified Accumulation Setups:")
    st.dataframe(pd.DataFrame(results)[["Ticker", "Change %", "RS vs SPY", "VWAP", "EMA Align", "Volume Spike", "MACD Bullish", "Reasons"]])
else:
    st.info("No accumulation setups found in current watchlist with current criteria.")

# DEBUG: Show all scan results, not just qualified
if all_results:
    st.write("All scan results (for debugging):")
    st.dataframe(pd.DataFrame(all_results)[["Ticker", "Change %", "RS vs SPY", "VWAP", "EMA Align", "Volume Spike", "MACD Bullish", "Reasons"]])
