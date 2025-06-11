for ticker in sp100:
    debug_status = ""
    try:
        df = yf.download(ticker, period="5d", interval="5m", progress=False)
        if df.empty or len(df) < 50:
            debug_status = f"{ticker}: Not enough data"
            debug_rows.append({'Ticker': ticker, 'Status': debug_status})
            continue

        # --- FLATTEN all multi-index columns (always) ---
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join([str(x) for x in c]).lower() for c in df.columns]
        else:
            df.columns = [str(c).lower() for c in df.columns]

        # --- Universal "close" finder ---
        close_col = None
        candidates = ['close', 'adjclose', 'adj close']
        for cand in candidates:
            for col in df.columns:
                if cand == col.strip().replace(' ', '').lower():
                    close_col = col
                    break
            if close_col: break
        # Try partial match as fallback
        if not close_col:
            for col in df.columns:
                if 'close' in col.replace(' ', '').lower():
                    close_col = col
                    break
        if not close_col:
            debug_status = f"{ticker}: 'Close' column missing ({df.columns})"
            debug_rows.append({'Ticker': ticker, 'Status': debug_status})
            continue
        if isinstance(df[close_col], pd.DataFrame):
            df['Close'] = df[close_col].iloc[:,0]
        else:
            df['Close'] = df[close_col]

        # --- Universal "volume" finder ---
        vol_col = None
        candidates = ['volume', 'regularmarketvolume']
        for cand in candidates:
            for col in df.columns:
                if cand == col.strip().replace(' ', '').lower():
                    vol_col = col
                    break
            if vol_col: break
        # Try partial match as fallback
        if not vol_col:
            for col in df.columns:
                if 'vol' in col.replace(' ', '').lower():
                    vol_col = col
                    break
        if not vol_col:
            debug_status = f"{ticker}: 'Volume' column missing ({df.columns})"
            debug_rows.append({'Ticker': ticker, 'Status': debug_status})
            continue
        if isinstance(df[vol_col], pd.DataFrame):
            df['Volume'] = df[vol_col].iloc[:,0]
        else:
            df['Volume'] = df[vol_col]

        df = calc_indicators(df)
        last = df.iloc[-1]
        close_price = float(safe_scalar(last['Close']))
        avgvol40 = float(safe_scalar(last['AvgVol40']))

        if any(np.isnan([close_price, avgvol40])):
            debug_status = f"{ticker}: NaNs in indicators or data"
            debug_rows.append({'Ticker': ticker, 'Status': debug_status})
            continue
        if not (min_price <= close_price <= max_price):
            debug_status = f"{ticker}: Price {close_price} outside filter"
            debug_rows.append({'Ticker': ticker, 'Status': debug_status, 'Close': close_price})
            continue
        if avgvol40 < min_volume:
            debug_status = f"{ticker}: AvgVol40 {avgvol40} below filter"
            debug_rows.append({'Ticker': ticker, 'Status': debug_status, 'AvgVol40': avgvol40})
            continue

        # --- Apply all strategies, collect all triggers ---
        picks = []
        for func, strat_name in [
            (mean_reversion_signal, "Mean Reversion"),
            (ema40_breakout_signal, "EMA40 Breakout"),
            (macd_ema_signal, "MACD+EMA")
        ]:
            sig, reason, score = func(df)
            if sig:
                picks.append((strat_name, reason, score))

        debug_rows.append({
            'Ticker': ticker,
            'Status': 'OK',
            'Close': close_price,
            'SMA40': float(safe_scalar(last['SMA40'])),
            'RSI3': float(safe_scalar(last['RSI3'])),
            'EMA40': float(safe_scalar(last['EMA40'])),
            'MACD': float(safe_scalar(last['MACD'])),
            'Volume': int(safe_scalar(df['Volume'][df['Volume'] > 0].iloc[-1])) if (df['Volume'] > 0).any() else 0,
            'AvgVol40': int(avgvol40),
            'MR?': any(x[0] == "Mean Reversion" for x in picks),
            'EMA?': any(x[0] == "EMA40 Breakout" for x in picks),
            'MACD?': any(x[0] == "MACD+EMA" for x in picks)
        })

        # --- If at least one strategy triggers, add for ranking ---
        if picks:
            picks.sort(key=lambda x: -x[2])
            strat_name, reason, score = picks[0]
            entry = close_price
            shares = int(capital_per_trade // entry)
            invested = shares * entry
            results.append({
                "Ticker": ticker,
                "Strategy": strat_name,
                "AI Score": score,
                "Entry Price": round(entry, 2),
                "Capital Used": round(invested, 2),
                "Shares": shares,
                "Reason": reason,
                "Volume": int(safe_scalar(df['Volume'][df['Volume'] > 0].iloc[-1])) if (df['Volume'] > 0).any() else 0,
                "Avg Vol (40)": int(avgvol40),
                "RSI(3)": round(float(safe_scalar(last['RSI3'])), 2),
                "EMA40": round(float(safe_scalar(last['EMA40'])), 2),
                "SMA40": round(float(safe_scalar(last['SMA40'])), 2)
            })
    except Exception as e:
        debug_status = f"{ticker}: Exception - {e}"
        debug_rows.append({'Ticker': ticker, 'Status': debug_status})
