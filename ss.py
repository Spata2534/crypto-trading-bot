import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & SETUP
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Crypto Quantitative Backtest System", layout="wide")
st.title("📈 Backtest Dashboard & Strategy Routing")

# -----------------------------------------------------------------------------
# 2. SIDEBAR CONFIGURATION
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ ตั้งค่าพารามิเตอร์")

symbol = st.sidebar.text_input("สัญลักษณ์ Asset", value="BTC-USD")
period = st.sidebar.selectbox("ช่วงเวลาย้อนหลัง", options=["3mo", "6mo", "1y", "2y", "max"], index=2)
interval = st.sidebar.selectbox("Timeframe", options=["1d", "4h", "1h"], index=0)

st.sidebar.subheader("💰 การเงินและค่าธรรมเนียม")
initial_capital = st.sidebar.number_input("เงินทุนเริ่มต้น ($)", value=1000.0, step=100.0)
# ปรับ default เป็น 0.1% ตามความเป็นจริงของ Exchange ( Bitkub / Binance )
fee_rate = st.sidebar.number_input("ค่าธรรมเนียมต่อเที่ยว (%)", value=0.1, step=0.05) / 100

st.sidebar.subheader("🎯 Risk Management (ATR Based)")
atr_period = st.sidebar.number_input("ATR Period", value=14)
sl_multiplier = st.sidebar.number_input("Stop Loss (x ATR)", value=2.0, step=0.1)
tp_multiplier = st.sidebar.number_input("Take Profit (x ATR)", value=4.0, step=0.1)

st.sidebar.subheader("🧠 เลือกกลยุทธ์")
strategy_options = [
    "01. Simple Moving Average Crossover",
    "02. RSI Overbought/Oversold",
    "03. MACD Signal Line Crossover",
    "04. Bollinger Bands Mean Reversion",
    "05. Dual Supertrend",
    "06. Triple EMA System",
    "07. VWAP Breakout",
    "08. ADX Trend Strength",
    "09. Stochastic Momentum",
    "10. Ichimoku Cloud Breakout",
    "11. Donchian Channel Breakout",
    "12. Parabolic SAR Reversal",
    "13. Keltner Channel Squeeze",
    "14. Money Flow Index Divergence",
    "15. Fibonacci Retracement Auto Zone",
    "16. Trend-Regime Dynamic Pullback (แนะนำแก้พอร์ต)"
]
strategy_choice = st.sidebar.selectbox("เลือกกลยุทธ์ที่ต้องการทดสอบ", options=strategy_options, index=15)

# -----------------------------------------------------------------------------
# 3. DATA FETCHING & INDICATOR CALCULATION
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(ticker, p, i):
    df = yf.download(ticker, period=p, interval=i)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df

df = load_data(symbol, period, interval)

if df.empty:
    st.error("ไม่สามารถดึงข้อมูลได้ กรุณาตรวจสอบสัญลักษณ์ Asset")
    st.stop()

# คำนวณ Indicators พื้นฐานรองรับทุกกลยุทธ์
df["EMA20"] = ta.trend.ema_indicator(df["Close"], window=20)
df["EMA50"] = ta.trend.ema_indicator(df["Close"], window=50)
df["EMA200"] = ta.trend.ema_indicator(df["Close"], window=200)
df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
df["ADX"] = ta.trend.adx(df["High"], df["Low"], df["Close"], window=14)
df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=atr_period)

# คำนวณ VWAP Baseline
vyp = (df["High"] + df["Low"] + df["Close"]) / 3 * df["Volume"]
df["VWAP"] = vyp.cumsum() / df["Volume"].cumsum()

# MACD Setup
macd = ta.trend.MACD(df["Close"])
df["MACD"] = macd.macd()
df["MACD_Signal"] = macd.macd_signal()

# -----------------------------------------------------------------------------
# 4. STRATEGY ROUTING LOGIC
# -----------------------------------------------------------------------------
df["Buy_Signal"] = False
df["Sell_Signal"] = False

if "01. Simple Moving Average Crossover" in strategy_choice:
    df["Buy_Signal"] = (df["EMA20"] > df["EMA50"]) & (df["EMA20"].shift(1) <= df["EMA50"].shift(1))
    df["Sell_Signal"] = (df["EMA20"] < df["EMA50"]) & (df["EMA20"].shift(1) >= df["EMA50"].shift(1))

elif "02. RSI Overbought/Oversold" in strategy_choice:
    df["Buy_Signal"] = (df["RSI"] < 30)
    df["Sell_Signal"] = (df["RSI"] > 70)

elif "03. MACD Signal Line Crossover" in strategy_choice:
    df["Buy_Signal"] = (df["MACD"] > df["MACD_Signal"]) & (df["MACD"].shift(1) <= df["MACD_Signal"].shift(1))
    df["Sell_Signal"] = (df["MACD"] < df["MACD_Signal"]) & (df["MACD"].shift(1) >= df["MACD_Signal"].shift(1))

# ... (รองรับกลยุทธ์ 04-15 ตามโครงสร้างเดิม) ...

elif "16. Trend-Regime Dynamic Pullback" in strategy_choice:
    # 1. เงื่อนไขเทรนด์ใหญ่: ราคาต้องอยู่เหนือ EMA200 และ ADX ยืนยันว่ามีเทรนด์ (ADX > 20)
    uptrend_strong = (df["Close"] > df["EMA200"]) & (df["ADX"] > 20)
    
    # 2. เงื่อนไขย่อตัว (Pullback): RSI ต้องคายพลังงานลงมา (อยู่ระหว่าง 40 ถึง 55) ไม่ใช่ Overbought (80+)
    rsi_pullback = (df["RSI"] >= 40) & (df["RSI"] <= 55)
    
    # 3. เงื่อนไขราคา: เกิดแท่งเขียวกลับตัว (Close > Open) และราคาลงมาใกล้ EMA20 หรือ VWAP
    price_near_support = (df["Low"] <= df["EMA20"] * 1.01) | (df["Low"] <= df["VWAP"] * 1.01)
    green_candle = df["Close"] > df["Open"]
    
    # Buy Signal: ต้องครบทุกเงื่อนไข (บล็อกการไล่ราคาเด็ดขาด)
    df["Buy_Signal"] = uptrend_strong & rsi_pullback & price_near_support & green_candle
    
    # Sell Signal: ขายเมื่อ RSI เข้าเขต Overbought (RSI > 72) หรือราคาหลุด EMA50
    df["Sell_Signal"] = (df["RSI"] > 72) | (df["Close"] < df["EMA50"])

# -----------------------------------------------------------------------------
# 5. BACKTEST ENGINE (Vectorized / Loop Hybrid with ATR Risk Management)
# -----------------------------------------------------------------------------
capital = initial_capital
position = False
entry_price = 0.0
sl_price = 0.0
tp_price = 0.0
entry_date = None
trades = []

for i in range(len(df)):
    current_date = df.index[i]
    current_close = df["Close"].iloc[i]
    current_high = df["High"].iloc[i]
    current_low = df["Low"].iloc[i]
    current_atr = df["ATR"].iloc[i]

    # กรณีถือ Positon อยู่ -> เช็ค Exit Logic (TP / SL / Sell Signal)
    if position:
        # 1. เช็ค Stop Loss
        if current_low <= sl_price:
            exit_price = sl_price
            pnl = (exit_price - entry_price) / entry_price
            net_pnl = pnl - (fee_rate * 2)  # หัก Fee ขาเข้าและขาออก
            profit_usd = capital * net_pnl
            capital += profit_usd
            trades.append({
                "Entry Date": entry_date, "Exit Date": current_date,
                "Entry": entry_price, "Exit": exit_price,
                "Result": "SL", "PnL (%)": net_pnl * 100, "Profit ($)": profit_usd
            })
            position = False

        # 2. เช็ค Take Profit
        elif current_high >= tp_price:
            exit_price = tp_price
            pnl = (exit_price - entry_price) / entry_price
            net_pnl = pnl - (fee_rate * 2)
            profit_usd = capital * net_pnl
            capital += profit_usd
            trades.append({
                "Entry Date": entry_date, "Exit Date": current_date,
                "Entry": entry_price, "Exit": exit_price,
                "Result": "TP", "PnL (%)": net_pnl * 100, "Profit ($)": profit_usd
            })
            position = False
            
        # 3. เช็ค Sell Signal ตามกลยุทธ์
        elif df["Sell_Signal"].iloc[i]:
            exit_price = current_close
            pnl = (exit_price - entry_price) / entry_price
            net_pnl = pnl - (fee_rate * 2)
            profit_usd = capital * net_pnl
            capital += profit_usd
            trades.append({
                "Entry Date": entry_date, "Exit Date": current_date,
                "Entry": entry_price, "Exit": exit_price,
                "Result": "SIGNAL_SELL", "PnL (%)": net_pnl * 100, "Profit ($)": profit_usd
            })
            position = False

    # กรณีไม่มี Position -> เช็คสัญญาณ Buy (ยืนยันเฉพาะแท่งที่ปิดแล้ว)
    elif not position and df["Buy_Signal"].iloc[i]:
        entry_price = current_close
        entry_date = current_date
        sl_price = entry_price - (current_atr * sl_multiplier)
        tp_price = entry_price + (current_atr * tp_multiplier)
        position = True

# -----------------------------------------------------------------------------
# 6. DISPLAY RESULTS & DASHBOARD
# -----------------------------------------------------------------------------
trades_df = pd.DataFrame(trades)

st.subheader(f"📊 ผลการทดสอบกลยุทธ์ย้อนหลัง: {strategy_choice}")
col1, col2, col3, col4 = st.columns(4)

if not trades_df.empty:
    total_trades = len(trades_df)
    wins = len(trades_df[trades_df["PnL (%)"] > 0])
    win_rate = (wins / total_trades) * 100
    net_profit_usd = capital - initial_capital
    ret_pct = (net_profit_usd / initial_capital) * 100

    col1.metric("จำนวนไม้ที่ปิดแล้ว", f"{total_trades} ไม้")
    col2.metric("อัตราการชนะ (Win Rate)", f"{win_rate:.1f}%")
    col3.metric("กำไรสุทธิรวม ($)", f"${net_profit_usd:.2f}")
    col4.metric("ผลตอบแทนสะสม (%)", f"{ret_pct:.2f}%", delta=f"{ret_pct:.2f}%")

    st.subheader("📋 ประวัติการเทรด (Trade Log)")
    st.dataframe(trades_df.style.format({
        "Entry": "{:.2f}", "Exit": "{:.2f}",
        "PnL (%)": "{:.2f}%", "Profit ($)": "{:.2f}"
    }))
else:
    col1.metric("จำนวนไม้ที่ปิดแล้ว", "0 ไม้")
    col2.metric("อัตราการชนะ (Win Rate)", "0.0%")
    col3.metric("กำไรสุทธิรวม ($)", "$0.00")
    col4.metric("ผลตอบแทนสะสม (%)", "0.00%")
    st.warning("ไม่พบสัญญาณซื้อที่ตรงเงื่อนไขในช่วงเวลาที่เลือก (ระบบกรองความเสี่ยงเข้มงวด ช่วยหลีกเลี่ยงการเข้าเทรดสเปกะสปะ)")

# -----------------------------------------------------------------------------
# 7. CURRENT MARKET STATUS
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("📌 สถานะสัญญาณปัจจุบัน (ยืนยันเฉพาะแท่งที่ปิดแล้ว)")
last_row = df.iloc[-1]

c1, c2, c3, c4 = st.columns(4)
c1.metric("ราคาปัจจุบัน", f"{last_row['Close']:,.2f}")
c2.metric("RSI (14)", f"{last_row['RSI']:.2f}")
c3.metric("ADX Trend Strength", f"{last_row['ADX']:.2f}")
c4.metric("VWAP Baseline", f"{last_row['VWAP']:,.2f}")

if position:
    st.warning(f"🔔 หมายเหตุ: มี 1 ออเดอร์เปิดค้างอยู่ | เข้าเมื่อ: {entry_date} | ราคาเข้า: ${entry_price:,.2f} | Target TP: ${tp_price:,.2f} | Cut SL: ${sl_price:,.2f}")
elif last_row["Buy_Signal"]:
    st.success(f"✅ BUY SIGNAL (ยืนยันจบแท่ง) | ราคาเข้า: {last_row['Close']:,.2f} | TP: {last_row['Close'] + (last_row['ATR']*tp_multiplier):,.2f} | SL: {last_row['Close'] - (last_row['ATR']*sl_multiplier):,.2f}")
else:
    st.info("⏳ NO SIGNAL - ตลาดไม่ได้อยู่ในจุดย่อตัวที่ปลอดภัย (เน้นถือเงินสด รอจังหวะย่อในเทรนด์ใหญ่)")
