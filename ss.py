import datetime
import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ==========================================
# 1. Helper Functions & Technical Indicators
# ==========================================


def send_line_alert(token, message):
    """ส่งการแจ้งเตือนผ่าน LINE Notify / Webhook"""
    if not token:
        return
    url = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {token}"}
    data = {"message": message}
    try:
        requests.post(url, headers=headers, data=data, timeout=5)
    except Exception as e:
        st.warning(f"ไม่สามารถส่ง LINE Alert ได้: {e}")


def calculate_rsi(series, period=14):
    """คำนวณ Relative Strength Index (RSI)"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_atr(df, period=14):
    """คำนวณ Average True Range (ATR)"""
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift(1))
    low_close = np.abs(df["Low"] - df["Close"].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_adx(df, period=14):
    """คำนวณ Average Directional Index (ADX)"""
    up = df["High"].diff()
    down = -df["Low"].diff()

    plus_di_raw = np.where((up > down) & (up > 0), up, 0)
    minus_di_raw = np.where((down > up) & (down > 0), down, 0)

    tr = calculate_atr(df, period=period)
    plus_di = 100 * (
        pd.Series(plus_di_raw, index=df.index).rolling(period).mean() / tr
    )
    minus_di = 100 * (
        pd.Series(minus_di_raw, index=df.index).rolling(period).mean() / tr
    )

    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=period).mean()
    return adx


def calculate_vwap(df):
    """คำนวณ Volume Weighted Average Price (VWAP)"""
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    tp_v = typical_price * df["Volume"]
    vwap = tp_v.cumsum() / df["Volume"].cumsum()
    return vwap


# ==========================================
# 2. Streamlit UI Dashboard Setup
# ==========================================
st.set_page_config(
    page_title="Trading Strategy & Backtest Dashboard", layout="wide"
)
st.title("📈 System Trading & Backtest Engine")

# Sidebar Configurations
st.sidebar.header("⚙️ การตั้งค่าพารามิเตอร์")
symbol = st.sidebar.text_input("สัญลักษณ์หุ้น / สินทรัพย์ (เช่น BTC-USD, AAPL)", "BTC-USD")
timeframe = st.sidebar.selectbox("Timeframe", ["1d", "4h", "1h", "15m"], index=0)
period_range = st.sidebar.selectbox(
    "ย้อนหลัง (Period)", ["1y", "2y", "6m", "3m"], index=0
)

st.sidebar.subheader("🎯 Risk Management & Strategy")
rr_ratio = st.sidebar.number_input(
    "Risk:Reward Ratio (R:R)", min_value=1.0, max_value=5.0, value=2.0, step=0.1
)
atr_multiplier = st.sidebar.number_input(
    "ATR Stop Loss Multiplier", min_value=0.5, max_value=4.0, value=1.5, step=0.1
)
initial_capital = st.sidebar.number_input(
    "ทุนเริ่มต้น ($)", min_value=100.0, value=10000.0, step=500.0
)
line_token = st.sidebar.text_input("LINE Notify Token (ระบุเพื่อรับการแจ้งเตือน)", type="password")

# ==========================================
# 3. Data Fetching & Technical Calculation
# ==========================================
final_symbol = symbol.strip().upper()

try:
    df = yf.download(
        final_symbol, period=period_range, interval=timeframe, progress=False
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
except Exception as err:
    st.error(f"ไม่สามารถดึงข้อมูลได้: {err}")
    st.stop()

if df.empty or len(df) < 50:
    st.warning("ข้อมูลไม่เพียงพอในการคำนวณตัวชี้วัดทางเทคนิค")
    st.stop()

# คำนวณ Indicators
df["RSI"] = calculate_rsi(df["Close"])
df["ATR"] = calculate_atr(df)
df["ADX"] = calculate_adx(df)
df["VWAP"] = calculate_vwap(df)
df["EMA_Fast"] = df["Close"].ewm(span=12, adjust=False).mean()
df["EMA_Slow"] = df["Close"].ewm(span=26, adjust=False).mean()

# เงื่อนไขการเกิดสัญญาณ
buy_raw = (
    (df["EMA_Fast"] > df["EMA_Slow"])
    & (df["EMA_Fast"].shift(1) <= df["EMA_Slow"].shift(1))
    & (df["RSI"] > 45)
    & (df["ADX"] > 20)
)
sell_raw = (df["EMA_Fast"] < df["EMA_Slow"]) & (
    df["EMA_Fast"].shift(1) >= df["EMA_Slow"].shift(1)
)

# คำนวณ SL/TP แบบเวกเตอร์บน DataFrame
df["Risk"] = df["ATR"] * atr_multiplier
df["SL"] = df["Close"] - df["Risk"]
df["TP"] = df["Close"] + (df["Risk"] * rr_ratio)

df_filtered = df.dropna().copy()
buy_filtered = buy_raw.reindex(df_filtered.index, fill_value=False)
sell_filtered = sell_raw.reindex(df_filtered.index, fill_value=False)

# ==========================================
# 4. Live / Closed Candle Signal Status (แก้ไขแล้ว)
# ==========================================
st.subheader("📌 สถานะสัญญาณล่าสุด (ยืนยันเฉพาะแท่งที่ปิดแล้ว)")

current_candle = df_filtered.iloc[-1]
closed_candle = df_filtered.iloc[-2]  # แท่งที่เพิ่งปิดยืนยันสัญญาณ

is_buy_closed = buy_filtered.iloc[-2]
is_sell_closed = sell_filtered.iloc[-2]

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("ราคาปัจจุบัน", f"{current_candle['Close']:,.2f}")
col_b.metric("RSI (14)", f"{current_candle['RSI']:.2f}")
col_c.metric("ADX Trend Strength", f"{current_candle['ADX']:.1f}")
col_d.metric("VWAP Baseline", f"{current_candle['VWAP']:,.2f}")

if is_buy_closed:
    # ใช้วิธีอ้างอิง Entry, SL, TP จากแท่งที่เกิดสัญญาณซื้อโดยตรง (closed_candle)
    entry_p = closed_candle["Close"]
    sl_p = closed_candle["SL"]
    tp_p = closed_candle["TP"]

    # ตรวจสอบความถูกต้องของตรรกะ (BUY: TP ต้อง > Entry และ SL ต้อง < Entry)
    if tp_p > entry_p and sl_p < entry_p:
        st.success(
            f"✅ **BUY SIGNAL (ยืนยันจบแท่ง)** | ราคาเข้า: {entry_p:,.2f} | TP: {tp_p:,.2f} | SL: {sl_p:,.2f}"
        )
        send_line_alert(
            line_token,
            f"\n🚨 BUY SIGNAL: {final_symbol}\nEntry: {entry_p:,.2f}\nTP: {tp_p:,.2f}\nSL: {sl_p:,.2f}",
        )
    else:
        st.error(
            "⚠️ สัญญาณซื้อขัดแย้ง: การคำนวณ ATR ผิดปกติ ทำให้ระยะ TP/SL ไม่สัมพันธ์กับราคาเข้า"
        )

elif is_sell_closed:
    st.error(
        f"🚨 **SELL SIGNAL (ยืนยันจบแท่ง)** | เกิดสัญญาณขายที่ราคา: {closed_candle['Close']:,.2f}"
    )
    send_line_alert(
        line_token,
        f"\n🚨 SELL SIGNAL: {final_symbol}\nPrice: {closed_candle['Close']:,.2f}",
    )
else:
    st.info("⏳ แท่งล่าสุดที่เพิ่งปิดยังไม่มีสัญญาณซื้อขาย (สถานะปกติ)")

st.divider()

# ==========================================
# 5. Backtest Simulation Engine
# ==========================================
st.subheader("📊 ผลการทดสอบกลยุทธ์ย้อนหลัง (Backtest Performance)")

capital = initial_capital
position = None
trades = []

for i in range(len(df_filtered)):
    date = df_filtered.index[i]
    close_price = df_filtered["Close"].iloc[i]
    high_price = df_filtered["High"].iloc[i]
    low_price = df_filtered["Low"].iloc[i]

    # ถ้าถือออเดอร์อยู่ เช็ก TP / SL / Sell Signal
    if position is not None:
        # Check Stop Loss
        if low_price <= position["SL"]:
            exit_price = position["SL"]
            pnl = (exit_price - position["Entry"]) * position["Units"]
            capital += position["Units"] * exit_price
            trades.append(
                {
                    "Entry Date": position["Date"],
                    "Exit Date": date,
                    "Type": "BUY",
                    "Entry": position["Entry"],
                    "Exit": exit_price,
                    "PnL": pnl,
                    "Return (%)": (
                        (exit_price - position["Entry"]) / position["Entry"]
                    )
                    * 100,
                    "Reason": "Stop Loss",
                }
            )
            position = None

        # Check Take Profit
        elif high_price >= position["TP"]:
            exit_price = position["TP"]
            pnl = (exit_price - position["Entry"]) * position["Units"]
            capital += position["Units"] * exit_price
            trades.append(
                {
                    "Entry Date": position["Date"],
                    "Exit Date": date,
                    "Type": "BUY",
                    "Entry": position["Entry"],
                    "Exit": exit_price,
                    "PnL": pnl,
                    "Return (%)": (
                        (exit_price - position["Entry"]) / position["Entry"]
                    )
                    * 100,
                    "Reason": "Take Profit",
                }
            )
            position = None

        # Check Exit Signal (EMA Cross Sell)
        elif sell_filtered.iloc[i]:
            exit_price = close_price
            pnl = (exit_price - position["Entry"]) * position["Units"]
            capital += position["Units"] * exit_price
            trades.append(
                {
                    "Entry Date": position["Date"],
                    "Exit Date": date,
                    "Type": "BUY",
                    "Entry": position["Entry"],
                    "Exit": exit_price,
                    "PnL": pnl,
                    "Return (%)": (
                        (exit_price - position["Entry"]) / position["Entry"]
                    )
                    * 100,
                    "Reason": "Signal Exit",
                }
            )
            position = None

    # ถ้ายังไม่มีออเดอร์ และเกิด Buy Signal ให้เข้าซื้อ
    if position is None and buy_filtered.iloc[i]:
        entry_price = close_price
        sl_price = df_filtered["SL"].iloc[i]
        tp_price = df_filtered["TP"].iloc[i]

        # เข้าเทรดเต็มจำนวนเงินทุนที่มี (Full Position Size)
        units = capital / entry_price
        position = {
            "Date": date,
            "Entry": entry_price,
            "SL": sl_price,
            "TP": tp_price,
            "Units": units,
        }
        capital = 0

# สรุปภาพรวม Backtest
trades_df = pd.DataFrame(trades)

if not trades_df.empty:
    total_trades = len(trades_df)
    win_trades = len(trades_df[trades_df["PnL"] > 0])
    loss_trades = len(trades_df[trades_df["PnL"] <= 0])
    win_rate = (win_trades / total_trades) * 100
    total_pnl = trades_df["PnL"].sum()
    final_equity = initial_capital + total_pnl
    total_return = (
        (final_equity - initial_capital) / initial_capital
    ) * 100

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("จำนวนเทรดทั้งหมด", f"{total_trades} ครั้ง")
    m2.metric("Win Rate", f"{win_rate:.1f}%")
    m3.metric("กำไรสุทธิ ($)", f"{total_pnl:,.2f}")
    m4.metric("ผลตอบแทนรวม", f"{total_return:.2f}%")
    m5.metric("เงินทุนสุทธิ์", f"{final_equity:,.2f}")

    # แสดงประวัติการเทรด
    st.write("📋 **ประวัติการเทรดทั้งหมด (Trade Logs)**")
    st.dataframe(
        trades_df.style.format(
            {
                "Entry": "{:,.2f}",
                "Exit": "{:,.2f}",
                "PnL": "{:,.2f}",
                "Return (%)": "{:,.2f}%",
            }
        ),
        use_container_width=True,
    )
else:
    st.info("ไม่พบประวัติการเปิด-ปิดออเดอร์ในช่วงเวลาที่เลือก")

# ==========================================
# 6. Price & Strategy Charting
# ==========================================
st.subheader("📉 กราฟราคาและอินดิเคเตอร์")
st.line_chart(df_filtered[["Close", "EMA_Fast", "EMA_Slow", "VWAP"]])
