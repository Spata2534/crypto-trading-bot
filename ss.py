import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & SETUP
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Crypto Quantitative Backtest System", layout="wide")
st.title("📈 Backtest Dashboard & Strategy Routing")

# -----------------------------------------------------------------------------
# 2. SIDEBAR CONFIGURATION (BASIC INPUTS)
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ 1. เลือกสินทรัพย์และไทม์เฟรม")

crypto_presets = {
    "Bitcoin (BTC/USD)": "BTC-USD",
    "Ethereum (ETH/USD)": "ETH-USD",
    "Solana (SOL/USD)": "SOL-USD",
    "Binance Coin (BNB/USD)": "BNB-USD",
    "Ripple (XRP/USD)": "XRP-USD",
    "Cardano (ADA/USD)": "ADA-USD",
    "Dogecoin (DOGE/USD)": "DOGE-USD",
    "Avalanche (AVAX/USD)": "AVAX-USD",
    "Custom (ระบุเอง)": "CUSTOM"
}

selected_preset = st.sidebar.selectbox("เลือกเหรียญยอดนิยม", options=list(crypto_presets.keys()), index=0)

if crypto_presets[selected_preset] == "CUSTOM":
    symbol = st.sidebar.text_input("พิมพ์สัญลักษณ์ Ticker (เช่น NEAR-USD)", value="NEAR-USD").upper()
else:
    symbol = crypto_presets[selected_preset]

period = st.sidebar.selectbox("ช่วงเวลาย้อนหลัง", options=["3mo", "6mo", "1y", "2y", "5y"], index=2)
interval = st.sidebar.selectbox("Timeframe", options=["1d", "4h", "1h"], index=0)

st.sidebar.subheader("💰 2. บริหารเงินทุน (Money Management)")
initial_capital = st.sidebar.number_input("เงินทุนเริ่มต้น ($)", value=1000.0, step=100.0)
risk_per_trade_pct = st.sidebar.number_input("ความเสี่ยงต่อไม้ (%)", value=2.0, step=0.5) / 100
fee_rate = st.sidebar.number_input("ค่าธรรมเนียมต่อเที่ยว (%)", value=0.1, step=0.01) / 100

st.sidebar.subheader("🎯 3. การจัดการความเสี่ยง (ATR Risk)")
atr_period = st.sidebar.number_input("ATR Period", value=14)
sl_multiplier = st.sidebar.number_input("Stop Loss (x ATR)", value=2.0, step=0.1)
tp_multiplier = st.sidebar.number_input("Take Profit (x ATR)", value=4.0, step=0.1)

# -----------------------------------------------------------------------------
# 3. DATA FETCHING & INDICATORS
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(ticker, p, i):
    try:
        df = yf.download(ticker, period=p, interval=i)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

df = load_data(symbol, period, interval)

if df.empty or len(df) < 50:
    st.error(f"❌ ไม่สามารถโหลดข้อมูลสำหรับ {symbol} ได้ หรือข้อมูลมีน้อยเกินไป กรุณาตรวจสอบสัญลักษณ์ Asset")
    st.stop()

# คำนวณ Indicators
df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

delta = df["Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df["RSI"] = 100 - (100 / (1 + rs))

high_low = df["High"] - df["Low"]
high_close = (df["High"] - df["Close"].shift()).abs()
low_close = (df["Low"] - df["Close"].shift()).abs()
ranges = pd.concat([high_low, high_close, low_close], axis=1)
true_range = ranges.max(axis=1)
df["ATR"] = true_range.rolling(atr_period).mean()

up_move = df["High"].diff()
down_move = -df["Low"].diff()
plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
tr14 = true_range.rolling(14).sum()
plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(14).sum() / tr14)
minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(14).sum() / tr14)
dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
df["ADX"] = dx.rolling(14).mean()

vyp = (df["High"] + df["Low"] + df["Close"]) / 3 * df["Volume"]
df["VWAP"] = vyp.cumsum() / df["Volume"].cumsum()

ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = ema12 - ema26
df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

# -----------------------------------------------------------------------------
# 4. BACKTEST FUNCTION FOR STRATEGY EVALUATION
# -----------------------------------------------------------------------------
raw_strategies = [
    "01. Simple Moving Average Crossover",
    "02. RSI Overbought/Oversold",
    "03. MACD Signal Line Crossover",
    "16. Trend-Regime Dynamic Pullback (ปรับปรุงสูตร)"
]

def run_backtest(df_input, strat_name):
    df_temp = df_input.copy()
    df_temp["Buy_Signal"] = False
    df_temp["Sell_Signal"] = False

    if "01." in strat_name:
        df_temp["Buy_Signal"] = (df_temp["EMA20"] > df_temp["EMA50"]) & (df_temp["EMA20"].shift(1) <= df_temp["EMA50"].shift(1))
        df_temp["Sell_Signal"] = (df_temp["EMA20"] < df_temp["EMA50"]) & (df_temp["EMA20"].shift(1) >= df_temp["EMA50"].shift(1))
    elif "02." in strat_name:
        df_temp["Buy_Signal"] = (df_temp["RSI"] < 30)
        df_temp["Sell_Signal"] = (df_temp["RSI"] > 70)
    elif "03." in strat_name:
        df_temp["Buy_Signal"] = (df_temp["MACD"] > df_temp["MACD_Signal"]) & (df_temp["MACD"].shift(1) <= df_temp["MACD_Signal"].shift(1))
        df_temp["Sell_Signal"] = (df_temp["MACD"] < df_temp["MACD_Signal"]) & (df_temp["MACD"].shift(1) >= df_temp["MACD_Signal"].shift(1))
    elif "16." in strat_name:
        uptrend_strong = (df_temp["Close"] > df_temp["EMA200"]) & (df_temp["ADX"] > 18)
        rsi_pullback = (df_temp["RSI"] >= 38) & (df_temp["RSI"] <= 58)
        price_near_support = (df_temp["Low"] <= df_temp["EMA20"] * 1.015) | (df_temp["Low"] <= df_temp["VWAP"] * 1.015)
        green_candle = df_temp["Close"] > df_temp["Open"]
        df_temp["Buy_Signal"] = uptrend_strong & rsi_pullback & price_near_support & green_candle
        df_temp["Sell_Signal"] = (df_temp["RSI"] > 75) | (df_temp["Close"] < df_temp["EMA50"])

    capital = initial_capital
    position = False
    entry_price, sl_price, tp_price, entry_date, position_size = 0.0, 0.0, 0.0, None, 0.0
    trades = []
    equity_curve = [initial_capital]
    equity_dates = [df_temp.index[0]]

    for i in range(len(df_temp)):
        current_date = df_temp.index[i]
        current_close = df_temp["Close"].iloc[i]
        current_high = df_temp["High"].iloc[i]
        current_low = df_temp["Low"].iloc[i]
        current_atr = df_temp["ATR"].iloc[i]

        if position:
            is_exit, exit_price, reason = False, 0.0, ""
            if current_low <= sl_price:
                exit_price, reason, is_exit = sl_price, "SL", True
            elif current_high >= tp_price:
                exit_price, reason, is_exit = tp_price, "TP", True
            elif df_temp["Sell_Signal"].iloc[i]:
                exit_price, reason, is_exit = current_close, "SIGNAL", True

            if is_exit:
                raw_pnl_usd = (exit_price - entry_price) * position_size
                total_fee = (entry_price * position_size * fee_rate) + (exit_price * position_size * fee_rate)
                net_profit_usd = raw_pnl_usd - total_fee
                capital += net_profit_usd
                pnl_pct = (net_profit_usd / (entry_price * position_size)) * 100
                trades.append({
                    "Entry Date": entry_date, "Exit Date": current_date,
                    "Entry": entry_price, "Exit": exit_price,
                    "Size (Units)": position_size, "Result": reason,
                    "PnL (%)": pnl_pct, "Profit ($)": net_profit_usd,
                    "Capital After Trade": capital
                })
                position = False

        elif not position and df_temp["Buy_Signal"].iloc[i] and not np.isnan(current_atr):
            entry_price, entry_date = current_close, current_date
            sl_price = entry_price - (current_atr * sl_multiplier)
            tp_price = entry_price + (current_atr * tp_multiplier)
            risk_amount = capital * risk_per_trade_pct
            risk_per_unit = entry_price - sl_price
            if risk_per_unit > 0:
                position_size = min(risk_amount / risk_per_unit, (capital * 0.95) / entry_price)
                position = True

        equity_curve.append(capital)
        equity_dates.append(current_date)

    return df_temp, pd.DataFrame(trades), pd.DataFrame({"Date": equity_dates, "Equity": equity_curve}).set_index("Date"), capital - initial_capital, position, entry_price, sl_price, tp_price, entry_date

# -----------------------------------------------------------------------------
# 5. STRATEGY EVALUATION & DYNAMIC DROPDOWN WITH ICONS
# -----------------------------------------------------------------------------
strategy_map = {}
dropdown_options = []

for strat in raw_strategies:
    _, _, _, net_pnl, _, _, _, _, _ = run_backtest(df, strat)
    if net_pnl > 0:
        label = f"✅ {strat} (+$ {net_pnl:.2f})"
    else:
        label = f"❌ {strat} (-$ {abs(net_pnl):.2f})"
    strategy_map[label] = strat
    dropdown_options.append(label)

st.sidebar.subheader("🧠 4. เลือกระบบเทรด (Strategy)")
selected_label = st.sidebar.selectbox("เลือกกลยุทธ์ (✅ = กำไร / ❌ = ขาดทุน)", options=dropdown_options, index=len(dropdown_options)-1)
strategy_choice = strategy_map[selected_label]

# Run final backtest for selected strategy
df, trades_df, equity_df, net_profit_usd, position, entry_price, sl_price, tp_price, entry_date = run_backtest(df, strategy_choice)

# -----------------------------------------------------------------------------
# 6. TOP SECTION: CURRENT MARKET STATUS (ย้ายขึ้นมาบนสุดตามสั่ง)
# -----------------------------------------------------------------------------
st.subheader("📌 สถานะสัญญาณปัจจุบัน (การวิเคราะห์แท่งล่าสุด)")
last_row = df.iloc[-1]

c1, c2, c3, c4 = st.columns(4)
c1.metric("ราคาปัจจุบัน", f"${last_row['Close']:,.2f}")
c2.metric("RSI (14)", f"{last_row['RSI']:.2f}")
c3.metric("ADX Trend", f"{last_row['ADX']:.2f}")
c4.metric("VWAP Level", f"${last_row['VWAP']:,.2f}")

if position:
    st.warning(f"🔔 มีสถานะค้างอยู่ 1 ออเดอร์ | เข้าซื้อเมื่อ: {entry_date} | ราคาเข้า: ${entry_price:,.2f} | Target TP: ${tp_price:,.2f} | Cut SL: ${sl_price:,.2f}")
elif last_row["Buy_Signal"]:
    st.success(f"✅ BUY SIGNAL CONFIRMED | ราคาปัจจุบัน: ${last_row['Close']:,.2f} | TP แนะนำ: ${last_row['Close'] + (last_row['ATR']*tp_multiplier):,.2f} | SL แนะนำ: ${last_row['Close'] - (last_row['ATR']*sl_multiplier):,.2f}")
else:
    st.info("⏳ NO SIGNAL - ไม่อยู่ในจุดย่อตัวที่ได้เปรียบ แนะนำถือเงินสด (Cash Position) ไว้ก่อน")

st.markdown("---")

# -----------------------------------------------------------------------------
# 7. INTERACTIVE PLOTLY CHARTING
# -----------------------------------------------------------------------------
st.subheader(f"📉 กราฟราคาสินทรัพย์ & จุดเข้าทำกำไร/ตัดขาดทุน ({symbol})")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.03, row_heights=[0.75, 0.25])

fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
    name="Price"
), row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], line=dict(color='orange', width=1), name="EMA 20"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["EMA200"], line=dict(color='purple', width=1.5), name="EMA 200"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"], line=dict(color='blue', width=1, dash='dash'), name="VWAP"), row=1, col=1)

if not trades_df.empty:
    fig.add_trace(go.Scatter(
        x=trades_df["Entry Date"], y=trades_df["Entry"],
        mode="markers", marker=dict(symbol="triangle-up", size=12, color="green"),
        name="Buy Entry"
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=trades_df["Exit Date"], y=trades_df["Exit"],
        mode="markers", marker=dict(symbol="triangle-down", size=12, color="red"),
        name="Exit Order"
    ), row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color='green', width=1), name="RSI"), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

fig.update_layout(
    height=600,
    margin=dict(l=10, r=10, t=20, b=10),
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 8. PERFORMANCE DASHBOARD METRICS
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader(f"📊 สรุปประสิทธิภาพกลยุทธ์: {strategy_choice}")

col1, col2, col3, col4, col5, col6 = st.columns(6)

if not trades_df.empty:
    total_trades = len(trades_df)
    wins = len(trades_df[trades_df["Profit ($)"] > 0])
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    ret_pct = (net_profit_usd / initial_capital) * 100
    
    gross_profit = trades_df[trades_df["Profit ($)"] > 0]["Profit ($)"].sum()
    gross_loss = abs(trades_df[trades_df["Profit ($)"] < 0]["Profit ($)"].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0)
    
    equity_df["Peak"] = equity_df["Equity"].cummax()
    equity_df["Drawdown"] = (equity_df["Equity"] - equity_df["Peak"]) / equity_df["Peak"]
    max_drawdown = equity_df["Drawdown"].min() * 100

    col1.metric("จำนวนไม้ทั้งหมด", f"{total_trades} ไม้")
    col2.metric("Win Rate", f"{win_rate:.1f}%")
    col3.metric("กำไรสุทธิ ($)", f"${net_profit_usd:.2f}")
    col4.metric("ผลตอบแทนสะสม (%)", f"{ret_pct:.2f}%", delta=f"{ret_pct:.2f}%")
    col5.metric("Profit Factor", f"{profit_factor:.2f}")
    col6.metric("Max Drawdown", f"{max_drawdown:.2f}%", delta_color="inverse")
else:
    col1.metric("จำนวนไม้ทั้งหมด", "0 ไม้")
    col2.metric("Win Rate", "0.0%")
    col3.metric("กำไรสุทธิ ($)", "$0.00")
    col4.metric("ผลตอบแทนสะสม (%)", "0.00%")
    col5.metric("Profit Factor", "0.00")
    col6.metric("Max Drawdown", "0.00%")
    st.warning("⚠️ ไม่พบสัญญาณซื้อที่ตรงตามเงื่อนไขในช่วงเวลาที่เลือก")

# -----------------------------------------------------------------------------
# 9. TRADE LOG
# -----------------------------------------------------------------------------
if not trades_df.empty:
    st.markdown("---")
    st.subheader("📋 ตารางบันทึกการเทรด (Trade Log Detailed)")
    st.dataframe(trades_df.style.format({
        "Entry": "{:.2f}", "Exit": "{:.2f}", "Size (Units)": "{:.4f}",
        "PnL (%)": "{:.2f}%", "Profit ($)": "{:.2f}", "Capital After Trade": "{:.2f}"
    }), use_container_width=True)
