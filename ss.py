import datetime
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(layout="wide", page_title="BTC & Asset 15-Strategy Tester")
st.title("🎯 15-Strategy Trade Signal System + Advanced Backtest Engine")

# ---------------------------------------------------------
# 0. Helper Functions (ADX, ATR, StochRSI, VWAP, PSAR, Line)
# ---------------------------------------------------------
def calculate_adx(df, period=14):
    df = df.copy()
    df["UpMove"] = df["High"] - df["High"].shift(1)
    df["DownMove"] = df["Low"].shift(1) - df["Low"]
    df["+DM"] = np.where((df["UpMove"] > df["DownMove"]) & (df["UpMove"] > 0), df["UpMove"], 0)
    df["-DM"] = np.where((df["DownMove"] > df["UpMove"]) & (df["DownMove"] > 0), df["DownMove"], 0)
    df["TR"] = np.maximum(
        df["High"] - df["Low"],
        np.maximum(abs(df["High"] - df["Close"].shift(1)), abs(df["Low"] - df["Close"].shift(1)))
    )
    tr_smooth = df["TR"].ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (df["+DM"].ewm(alpha=1/period, adjust=False).mean() / (tr_smooth + 1e-10))
    minus_di = 100 * (df["-DM"].ewm(alpha=1/period, adjust=False).mean() / (tr_smooth + 1e-10))
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
    return dx.ewm(alpha=1/period, adjust=False).mean()

def calculate_atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def calculate_stoch_rsi(series, period=14, smoothK=3, smoothD=3):
    rsi_min = series.rolling(period).min()
    rsi_max = series.rolling(period).max()
    stoch_rsi = (series - rsi_min) / (rsi_max - rsi_min + 1e-10)
    k = stoch_rsi.rolling(smoothK).mean() * 100
    d = k.rolling(smoothD).mean()
    return k, d

def calculate_vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).cumsum() / (df["Volume"].cumsum() + 1e-10)

def calculate_psar(df, af_start=0.02, af_step=0.02, af_max=0.2):
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    psar = close.copy()
    psarbull = [True] * len(close)
    
    af = af_start
    hp = high[0]
    lp = low[0]
    
    for i in range(2, len(close)):
        if psarbull[i-1]:
            psar[i] = psar[i-1] + af * (hp - psar[i-1])
            psarbull[i] = True
            if low[i] < psar[i]:
                psarbull[i] = False
                psar[i] = hp
                lp = low[i]
                af = af_start
            else:
                if high[i] > hp:
                    hp = high[i]
                    af = min(af + af_step, af_max)
                if low[i-1] < psar[i]:
                    psar[i] = low[i-1]
                if low[i-2] < psar[i]:
                    psar[i] = low[i-2]
        else:
            psar[i] = psar[i-1] + af * (lp - psar[i-1])
            psarbull[i] = False
            if high[i] > psar[i]:
                psarbull[i] = True
                psar[i] = lp
                hp = high[i]
                af = af_start
            else:
                if low[i] < lp:
                    lp = low[i]
                    af = min(af + af_step, af_max)
                if high[i-1] > psar[i]:
                    psar[i] = high[i-1]
                if high[i-2] > psar[i]:
                    psar[i] = high[i-2]
    return pd.Series(psar, index=df.index), pd.Series(psarbull, index=df.index)

def send_line_alert(token, message):
    if token and token.strip() != "":
        url = "https://notify-api.line.me/api/notify"
        headers = {"Authorization": f"Bearer {token}"}
        try:
            requests.post(url, headers=headers, data={"message": message}, timeout=5)
        except Exception as e:
            st.error(f"Line Alert Error: {e}")

# ---------------------------------------------------------
# 1. Asset Selection & Sidebar Setup
# ---------------------------------------------------------
ALL_TICKERS = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
    "PTT.BK", "AOT.BK", "CPALL.BK", "DELTA.BK", "ADVANC.BK", "KBANK.BK",
    "NVDA", "AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "META"
]

st.sidebar.header("⚙️ ตั้งค่าระบบและ Money Management")

selected_asset = st.sidebar.selectbox("🔍 เลือก/พิมพ์ค้นหา Symbol:", options=ALL_TICKERS, index=0)
custom_ticker = st.sidebar.text_input("หรือพิมพ์ Symbol อื่นๆ:", "").upper()
final_symbol = custom_ticker if custom_ticker.strip() != "" else selected_asset

strategy_choice = st.sidebar.selectbox(
    "📊 เลือกกลยุทธ์การเทรด:",
    options=[
        "1. Strict Trend Dip Buy (EMA200 + RSI + ADX)",
        "2. Volatility Squeeze Breakout (Bollinger + Vol)",
        "3. Trend Following Cross (MACD + EMA20)",
        "4. Mean Reversion Rebound (RSI Oversold + Engulfing)",
        "5. Key Level Support Bounce (Support 20)",
        "6. EMA Ribbon Trend Alignment (EMA 20/50/200)",
        "7. Volume Weighted Breakout (OBV Trend + Volume Surge)",
        "8. ATR Dynamic Volatility Channel",
        "9. Triple Indicator Confluence (EMA + RSI + MACD Signal)",
        "10. Counter-Trend Oversold Scalp (High-WinRate Mean Reversion)",
        "11. High-Frequency StochRSI Scalper (Scalping ไว ซื้อขายบ่อย)",
        "12. Micro-Breakout Momentum Scalp (ทะลุ High 5 แท่ง + Volume)",
        "13. Fast EMA Crossover Scalper (EMA 5/13 Cross + RSI)",
        "14. VWAP Intra-Day Trend Scalper (แนะนำ: ตามต้นทุนสถาบัน)",
        "15. Parabolic SAR Momentum Trend-Follow (แนะนำ: จุดเปลี่ยนเทรนด์ไว)"
    ],
    index=13
)

timeframe_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "1d": "1d", "1wk": "1wk"}
timeframe_user = st.sidebar.selectbox("Timeframe:", options=list(timeframe_map.keys()), index=4)
timeframe = timeframe_map[timeframe_user]

start_date = st.sidebar.date_input("วันที่เริ่ม Backtest:", datetime.date(2023, 1, 1))
rr_ratio = st.sidebar.slider("Risk : Reward Ratio (R:R):", 1.0, 4.0, 2.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("💰 บริหารเงินทุน & ค่าธรรมเนียม")
trade_capital = st.sidebar.number_input("จำนวนเงินลงทุนต่อไม้ (บาท/USD):", min_value=100.0, value=2000.0, step=500.0)
trading_fee_pct = st.sidebar.number_input("ค่าธรรมเนียม + Slippage ต่อขา (%):", min_value=0.0, max_value=10.0, value=1.5, step=0.1)

st.sidebar.markdown("---")
line_token = st.sidebar.text_input("ใส่ Line Notify Token (ถ้ามี):", type="password")

# ---------------------------------------------------------
# 2. Fetch Data (Up to Present)
# ---------------------------------------------------------
fetch_period = "7d" if timeframe in ["1m", "5m"] else ("60d" if timeframe in ["15m", "1h"] else "max")

@st.cache_data(ttl=60)
def load_data(symbol, period, interval):
    ticker = yf.Ticker(symbol)
    df_fetched = ticker.history(period=period, interval=interval)
    return df_fetched

try:
    df_raw = load_data(final_symbol, fetch_period, timeframe)
except Exception as e:
    df_raw = pd.DataFrame()

if not df_raw.empty:
    df = df_raw.copy()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    # Moving Averages
    df["EMA5"] = df["Close"].ewm(span=5, adjust=False).mean()
    df["EMA13"] = df["Close"].ewm(span=13, adjust=False).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    
    # Technical Indicators
    df["ADX"] = calculate_adx(df, 14)
    df["ATR"] = calculate_atr(df, 14)
    df["VWAP"] = calculate_vwap(df)
    df["PSAR"], df["PSAR_Bull"] = calculate_psar(df)

    # RSI & StochRSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["StochK"], df["StochD"] = calculate_stoch_rsi(df["RSI"])

    # MACD & BB
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df["BB_Middle"] = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
    df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]

    df["High_5"] = df["High"].rolling(5).max()
    df["Support_20"] = df["Low"].rolling(20).min()
    df["Resistance_20"] = df["High"].rolling(20).max()
    df["Vol_SMA20"] = df["Volume"].rolling(20).mean()
    df["Vol_SMA5"] = df["Volume"].rolling(5).mean()

    # OBV
    obv_change = np.where(df["Close"] > df["Close"].shift(1), df["Volume"], 
                 np.where(df["Close"] < df["Close"].shift(1), -df["Volume"], 0))
    df["OBV"] = obv_change.cumsum()
    df["OBV_EMA20"] = df["OBV"].ewm(span=20, adjust=False).mean()

    # ---------------------------------------------------------
    # 3. STRATEGY ROUTING (15 Strategies)
    # ---------------------------------------------------------
    buy_signals = pd.Series(False, index=df.index)
    sell_signals = pd.Series(False, index=df.index)

    uptrend_major = df["Close"] > df["EMA200"]
    trending_market = df["ADX"] >= 20

    if "1. Strict Trend Dip Buy" in strategy_choice:
        buy_signals = uptrend_major & trending_market & (df["RSI"] < 45) & (df["Close"] > df["Open"])
        sell_signals = (df["Close"] < df["EMA20"]) | (df["RSI"] > 70)

    elif "2. Volatility Squeeze Breakout" in strategy_choice:
        squeeze = df["BB_Width"] < df["BB_Width"].rolling(20).quantile(0.25)
        vol_surge = df["Volume"] > (df["Vol_SMA20"] * 1.5)
        buy_signals = uptrend_major & squeeze.shift(1) & (df["Close"] > df["BB_Upper"]) & vol_surge & (df["ADX"] > 18)
        sell_signals = df["Close"] < df["BB_Middle"]

    elif "3. Trend Following Cross" in strategy_choice:
        macd_bull = (df["MACD"] > df["MACD_Signal"]) & (df["MACD"] > 0)
        touch_ema = (df["Low"] <= df["EMA20"]) & (df["Close"] >= df["EMA20"])
        buy_signals = uptrend_major & macd_bull & touch_ema & trending_market
        sell_signals = (df["MACD"] < df["MACD_Signal"])

    elif "4. Mean Reversion Rebound" in strategy_choice:
        bullish_bar = (df["Close"] > df["Open"]) & (df["Close"] > df["High"].shift(1))
        buy_signals = (df["RSI"] < 30) & bullish_bar
        sell_signals = (df["RSI"] > 65)

    elif "5. Key Level Support Bounce" in strategy_choice:
        near_support = (df["Low"] <= df["Support_20"] * 1.005)
        buy_signals = uptrend_major & near_support & (df["Close"] > df["Open"]) & (df["ADX"] > 15)
        sell_signals = (df["High"] >= df["Resistance_20"] * 0.995)

    elif "6. EMA Ribbon Trend Alignment" in strategy_choice:
        ribbon_bullish = (df["EMA20"] > df["EMA50"]) & (df["EMA50"] > df["EMA200"])
        pullback_ema50 = (df["Low"] <= df["EMA50"]) & (df["Close"] > df["EMA50"])
        buy_signals = ribbon_bullish & pullback_ema50 & (df["Close"] > df["Open"])
        sell_signals = df["Close"] < df["EMA50"]

    elif "7. Volume Weighted Breakout" in strategy_choice:
        obv_bull = df["OBV"] > df["OBV_EMA20"]
        break_res = df["Close"] >= df["Resistance_20"].shift(1)
        vol_extreme = df["Volume"] > (df["Vol_SMA20"] * 2.0)
        buy_signals = obv_bull & break_res & vol_extreme
        sell_signals = df["Close"] < df["EMA20"]

    elif "8. ATR Dynamic Volatility Channel" in strategy_choice:
        upper_atr = df["EMA20"] + (df["ATR"] * 2.0)
        buy_signals = (df["Close"] > upper_atr) & (df["ADX"] > 22)
        sell_signals = df["Close"] < df["EMA20"]

    elif "9. Triple Indicator Confluence" in strategy_choice:
        macd_cross = (df["MACD"] > df["MACD_Signal"]) & (df["MACD"].shift(1) <= df["MACD_Signal"].shift(1))
        rsi_bull = (df["RSI"] > 50) & (df["RSI"] < 65)
        buy_signals = uptrend_major & macd_cross & rsi_bull
        sell_signals = (df["MACD"] < df["MACD_Signal"]) | (df["RSI"] > 75)

    elif "10. Counter-Trend Oversold Scalp" in strategy_choice:
        extreme_oversold = (df["RSI"] < 25) & (df["Low"] < df["BB_Lower"])
        green_reversal = df["Close"] > df["Open"]
        buy_signals = extreme_oversold & green_reversal
        sell_signals = (df["RSI"] > 50) | (df["Close"] >= df["BB_Middle"])

    elif "11. High-Frequency StochRSI Scalper" in strategy_choice:
        stoch_cross_up = (df["StochK"] > df["StochD"]) & (df["StochK"].shift(1) <= df["StochD"].shift(1))
        buy_signals = stoch_cross_up & (df["StochK"] < 25)
        sell_signals = (df["StochK"] > 80) | ((df["StochK"] < df["StochD"]) & (df["StochK"].shift(1) >= df["StochD"].shift(1)))

    elif "12. Micro-Breakout Momentum Scalp" in strategy_choice:
        break_5bar = df["Close"] > df["High_5"].shift(1)
        vol_bump = df["Volume"] > df["Vol_SMA5"] * 1.2
        buy_signals = break_5bar & vol_bump & (df["Close"] > df["EMA13"])
        sell_signals = df["Close"] < df["EMA5"]

    elif "13. Fast EMA Crossover Scalper" in strategy_choice:
        ema_cross_up = (df["EMA5"] > df["EMA13"]) & (df["EMA5"].shift(1) <= df["EMA13"].shift(1))
        buy_signals = ema_cross_up & (df["RSI"] > 48)
        sell_signals = (df["EMA5"] < df["EMA13"]) & (df["EMA5"].shift(1) >= df["EMA13"].shift(1))

    elif "14. VWAP Intra-Day Trend Scalper" in strategy_choice:
        vwap_cross_up = (df["Close"] > df["VWAP"]) & (df["Close"].shift(1) <= df["VWAP"].shift(1))
        vol_valid = df["Volume"] > df["Vol_SMA20"]
        buy_signals = vwap_cross_up & (df["RSI"] > 50) & vol_valid
        sell_signals = df["Close"] < df["VWAP"]

    elif "15. Parabolic SAR Momentum Trend-Follow" in strategy_choice:
        psar_flip_bull = df["PSAR_Bull"] & (~df["PSAR_Bull"].shift(1).fillna(False))
        buy_signals = psar_flip_bull & (df["ADX"] > 20) & (df["Close"] > df["EMA50"])
        sell_signals = ~df["PSAR_Bull"]

    # Dynamic SL Calculation
    df["SL"] = df["Close"] - (df["ATR"] * 1.5)
    df["Risk"] = df["Close"] - df["SL"]
    df["TP"] = df["Close"] + (df["Risk"] * rr_ratio)

    # ---------------------------------------------------------
    # Filter Data from Start Date to Present
    # ---------------------------------------------------------
    if df.index.tz is not None:
        start_dt = pd.to_datetime(start_date).tz_localize(df.index.tz)
    else:
        start_dt = pd.to_datetime(start_date)

    valid_mask = df.index >= start_dt
    df_filtered = df[valid_mask]
    buy_filtered = buy_signals[valid_mask]
    sell_filtered = sell_signals[valid_mask]

    if not df_filtered.empty:
        # ---------------------------------------------------------
        # 4. Chart Render
        # ---------------------------------------------------------
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df_filtered.index, open=df_filtered["Open"], high=df_filtered["High"],
            low=df_filtered["Low"], close=df_filtered["Close"], name="Price"
        ))

        fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered["EMA20"], line=dict(color="yellow", width=1), name="EMA 20"))
        fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered["VWAP"], line=dict(color="cyan", width=1.5, dash="dot"), name="VWAP"))
        fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered["EMA200"], line=dict(color="purple", width=1.5), name="EMA 200"))

        fig.add_trace(go.Scatter(
            x=df_filtered[buy_filtered].index, y=df_filtered[buy_filtered]["Low"] * 0.99,
            mode="markers", marker=dict(symbol="triangle-up", size=9, color="#00FF00"), name="Buy Signal"
        ))
        fig.add_trace(go.Scatter(
            x=df_filtered[sell_filtered].index, y=df_filtered[sell_filtered]["High"] * 1.01,
            mode="markers", marker=dict(symbol="triangle-down", size=9, color="#FF0000"), name="Sell Signal"
        ))

        fig.update_layout(
            title=f"วิเคราะห์ราคา: {final_symbol} | กลยุทธ์: {strategy_choice}",
            xaxis_rangeslider_visible=False, height=550, template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ---------------------------------------------------------
        # 5. Live / Closed Candle Signal Status
        # ---------------------------------------------------------
        st.subheader("📌 สถานะสัญญาณล่าสุด (ยืนยันเฉพาะแท่งที่ปิดแล้ว)")

        current_candle = df_filtered.iloc[-1]
        closed_candle = df_filtered.iloc[-2] if len(df_filtered) > 1 else current_candle

        is_buy_closed = buy_filtered.iloc[-2] if len(buy_filtered) > 1 else False
        is_sell_closed = sell_filtered.iloc[-2] if len(sell_filtered) > 1 else False

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("ราคาปัจจุบัน", f"{current_candle['Close']:,.2f}")
        col_b.metric("RSI (14)", f"{current_candle['RSI']:.2f}")
        col_c.metric("ADX Trend Strength", f"{current_candle['ADX']:.1f}")
        col_d.metric("VWAP Baseline", f"{current_candle['VWAP']:,.2f}")

        if is_buy_closed:
            entry_p = current_candle["Close"]
            sl_p = closed_candle["SL"]
            tp_p = entry_p + ((entry_p - sl_p) * rr_ratio)
            st.success(f"✅ **BUY SIGNAL (ยืนยันจบแท่ง)** | ราคาเข้า: {entry_p:,.2f} | TP: {tp_p:,.2f} | SL: {sl_p:,.2f}")
        elif is_sell_closed:
            st.error("🚨 **SELL SIGNAL (ยืนยันจบแท่ง)** | เกิดสัญญาณขายทำกำไรหรือตัดขาดทุน")
        else:
            st.info("⏳ แท่งล่าสุดที่เพิ่งปิดยังไม่มีสัญญาณซื้อขาย (สถานะปกติ)")

        # ---------------------------------------------------------
        # 6. BACKTEST ENGINE WITH FEE & POSITION SIZING
        # ---------------------------------------------------------
        st.markdown("---")
        latest_time_str = df_filtered.index[-1].strftime("%Y-%m-%d %H:%M")
        st.subheader(f"📊 ผลการทดสอบกลยุทธ์ย้อนหลัง (เงินทุนไม้ละ {trade_capital:,.2f} | ค่าธรรมเนียม {trading_fee_pct}%)")

        trades = []
        in_position = False
        buy_price = 0
        entry_time = None
        sl_price = 0
        tp_price = 0
        units_bought = 0
        fee_rate = trading_fee_pct / 100.0

        for i in range(len(df_filtered)):
            current_time = df_filtered.index[i]
            price = df_filtered["Close"].iloc[i]
            low_p = df_filtered["Low"].iloc[i]
            high_p = df_filtered["High"].iloc[i]

            # 1. Check Exit condition first (if holding position)
            if in_position:
                exit_reason = None
                exit_price = price

                if low_p <= sl_price:
                    exit_price = sl_price
                    exit_reason = "STOP LOSS"
                elif high_p >= tp_price:
                    exit_price = tp_price
                    exit_reason = "TAKE PROFIT"
                elif sell_filtered.iloc[i]:
                    exit_price = price
                    exit_reason = "SELL SIGNAL"

                if exit_reason:
                    gross_return = units_bought * exit_price
                    exit_fee = gross_return * fee_rate
                    net_return = gross_return - exit_fee
                    
                    net_pnl_cash = net_return - trade_capital
                    net_pnl_pct = (net_pnl_cash / trade_capital) * 100

                    trades.append({
                        "Trade #": len(trades) + 1,
                        "Entry Time": entry_time.strftime("%Y-%m-%d %H:%M"),
                        "Exit Time": current_time.strftime("%Y-%m-%d %H:%M"),
                        "Entry Price": buy_price,
                        "Exit Price": exit_price,
                        "Capital Input": trade_capital,
                        "Net Return": net_return,
                        "PnL (บาท/USD)": net_pnl_cash,
                        "Net PnL (%)": net_pnl_pct,
                        "Result": "WIN" if net_pnl_cash > 0 else "LOSS",
                        "Exit Reason": exit_reason
                    })
                    in_position = False

            # 2. Check Entry condition (if not holding position)
            if not in_position and buy_filtered.iloc[i]:
                in_position = True
                buy_price = price
                entry_time = current_time
                sl_price = df_filtered["SL"].iloc[i]
                tp_price = df_filtered["TP"].iloc[i]

                # คำนวณการซื้อแบบหักค่าธรรมเนียมขาเข้า
                entry_fee = trade_capital * fee_rate
                net_capital_for_buy = trade_capital - entry_fee
                units_bought = net_capital_for_buy / buy_price

        if in_position:
            current_price = df_filtered["Close"].iloc[-1]
            unrealized_gross = units_bought * current_price
            unrealized_net = unrealized_gross * (1 - fee_rate)
            unrealized_pnl = unrealized_net - trade_capital
            unrealized_pnl_pct = (unrealized_pnl / trade_capital) * 100
            st.warning(f"🔔 **หมายเหตุ:** มี 1 ออเดอร์เปิดค้างอยู่ | เข้าเมื่อ: {entry_time.strftime('%Y-%m-%d %H:%M')} | ราคาเข้า: ${buy_price:,.2f} | PnL ปัจจุบัน (สุทธิหลังหัก Fee): {unrealized_pnl:+=,.2f} ({unrealized_pnl_pct:+.2f}%)")

        if trades:
            trade_df = pd.DataFrame(trades)
            total_trades = len(trade_df)
            wins = len(trade_df[trade_df["Result"] == "WIN"])
            win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
            
            total_net_pnl_cash = trade_df["PnL (บาท/USD)"].sum()
            total_net_pnl_pct = trade_df["Net PnL (%)"].sum()

            col_bt1, col_bt2, col_bt3, col_bt4 = st.columns(4)
            col_bt1.metric("จำนวนไม้ที่ปิดแล้ว", f"{total_trades} ไม้")
            col_bt2.metric("อัตราการชนะ (Win Rate)", f"{win_rate:.1f}%")
            col_bt3.metric("กำไรสุทธิรวม (บาท/USD)", f"{total_net_pnl_cash:+,.2f}")
            col_bt4.metric("ผลตอบแทนสะสมสุทธิ (%)", f"{total_net_pnl_pct:+.2f}%")

            st.markdown("### 📝 ประวัติการซื้อขายรายไม้ (คำนวณหัก ค่าธรรมเนียมแล้ว)")
            
            formatted_trade_df = trade_df.copy()
            formatted_trade_df["Entry Price"] = formatted_trade_df["Entry Price"].apply(lambda x: f"${x:,.2f}" if x > 10 else f"${x:,.4f}")
            formatted_trade_df["Exit Price"] = formatted_trade_df["Exit Price"].apply(lambda x: f"${x:,.2f}" if x > 10 else f"${x:,.4f}")
            formatted_trade_df["Capital Input"] = formatted_trade_df["Capital Input"].apply(lambda x: f"{x:,.2f}")
            formatted_trade_df["Net Return"] = formatted_trade_df["Net Return"].apply(lambda x: f"{x:,.2f}")
            formatted_trade_df["PnL (บาท/USD)"] = formatted_trade_df["PnL (บาท/USD)"].apply(lambda x: f"{x:+,.2f}")
            formatted_trade_df["Net PnL (%)"] = formatted_trade_df["Net PnL (%)"].apply(lambda x: f"{x:+.2f}%")

            st.dataframe(
                formatted_trade_df,
                use_container_width=True,
                column_config={
                    "Result": st.column_config.TextColumn("Result", help="ผลลัพธ์การเทรด"),
                    "Exit Reason": st.column_config.TextColumn("สาเหตุที่ปิดไม้")
                },
                hide_index=True
            )

            csv_data = trade_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 ดาวน์โหลดประวัติการเทรด (CSV)",
                data=csv_data,
                file_name=f"trade_history_{final_symbol}_{strategy_choice[:2].strip()}_with_fees.csv",
                mime="text/csv"
            )

        else:
            st.warning("ไม่พบประวัติการเทรดที่เข้าเงื่อนไขในช่วงเวลาที่เลือกจนถึงปัจจุบัน")

    else:
        st.warning("ไม่มีข้อมูลตามช่วงวันที่กำหนด")
else:
    st.error(f"ไม่พบข้อมูลของ Symbol '{final_symbol}' หรือบริการ Yahoo Finance ดึงข้อมูลไม่สำเร็จในขณะนี้")
