import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import nltk
import os

# -----------------------------------------------------------------------------
# 0. NLTK DATA SETUP & INITIALIZATION
# -----------------------------------------------------------------------------
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

@st.cache_resource
def init_nltk():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', download_dir=nltk_data_dir)

init_nltk()
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & SETUP
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Multi-Asset Quant & AI News Analytics", layout="wide")
st.title("📈 Multi-Asset Quant Backtest & AI News Dashboard")

# -----------------------------------------------------------------------------
# 2. SIDEBAR CONFIGURATION
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ 1. เลือกสินทรัพย์และไทม์เฟรม")

asset_categories = {
    "🪙 Crypto Top Staking": {
        "Bitcoin (BTC/USD)": ("BTC-USD", "crypto"),
        "Ethereum (ETH/USD)": ("ETH-USD", "crypto"),
        "Solana (SOL/USD)": ("SOL-USD", "crypto"),
        "Binance Coin (BNB/USD)": ("BNB-USD", "crypto"),
        "Ripple (XRP/USD)": ("XRP-USD", "crypto")
    },
    "🇹🇭 หุ้นไทย (SET Top Market Cap)": {
        "DELTA (บมจ.เดลต้า อีเลคโทรนิคส์)": ("DELTA.BK", "stock"),
        "PTT (บมจ.ปตท.)": ("PTT.BK", "stock"),
        "AOT (บมจ.ท่าอากาศยานไทย)": ("AOT.BK", "stock"),
        "CPALL (บมจ.ซีพี ออลล์)": ("CPALL.BK", "stock"),
        "ADVANC (บมจ.แอดวานซ์ อินโฟร์ เซอร์วิส)": ("ADVANC.BK", "stock"),
        "BDMS (บมจ.กรุงเทพดุสิตเวชการ)": ("BDMS.BK", "stock"),
        "KBANK (ธนาคารกสิกรไทย)": ("KBANK.BK", "stock"),
        "SCB (เอสซีบี เอกซ์)": ("SCB.BK", "stock")
    },
    "🇺🇸 หุ้นต่างประเทศ / US Tech & Index": {
        "NVIDIA (NVDA)": ("NVDA", "stock"),
        "Apple (AAPL)": ("AAPL", "stock"),
        "Microsoft (MSFT)": ("MSFT", "stock"),
        "Tesla (TSLA)": ("TSLA", "stock"),
        "Amazon (AMZN)": ("AMZN", "stock"),
        "Meta Platforms (META)": ("META", "stock"),
        "Alphabet / Google (GOOGL)": ("GOOGL", "stock"),
        "S&P 500 Index (SPY)": ("SPY", "stock")
    },
    "🌐 Custom (ระบุ Ticker เอง)": {
        "ระบุ Ticker สัญลักษณ์เอง": ("CUSTOM", "custom")
    }
}

category_choice = st.sidebar.selectbox("เลือกหมวดหมู่สินทรัพย์", options=list(asset_categories.keys()))
selected_asset_label = st.sidebar.selectbox("เลือกชื่อหุ้น/เหรียญ", options=list(asset_categories[category_choice].keys()))

symbol_info = asset_categories[category_choice][selected_asset_label]

if symbol_info[0] == "CUSTOM":
    symbol = st.sidebar.text_input("พิมพ์ Ticker (เช่น PTT.BK, TSLA, ETH-USD)", value="TSLA").upper()
    asset_type = "stock" if not symbol.endswith("-USD") else "crypto"
else:
    symbol = symbol_info[0]
    asset_type = symbol_info[1]

period = st.sidebar.selectbox("ช่วงเวลาย้อนหลัง", options=["1y", "2y", "5y", "max"], index=1)
interval = st.sidebar.selectbox("Timeframe", options=["1d", "1wk"], index=0)

st.sidebar.subheader("💰 2. บริหารเงินทุน (Money Management)")
initial_capital = st.sidebar.number_input("เงินทุนเริ่มต้น ($/฿)", value=100000.0, step=10000.0)
risk_per_trade_pct = st.sidebar.number_input("ความเสี่ยงต่อไม้ (%)", value=2.0, step=0.5) / 100
fee_rate = st.sidebar.number_input("ค่าธรรมเนียมต่อเที่ยว (%)", value=0.15, step=0.01) / 100

st.sidebar.subheader("🎯 3. การจัดการความเสี่ยง (ATR Risk)")
atr_period = st.sidebar.number_input("ATR Period", value=14)
sl_multiplier = st.sidebar.number_input("Stop Loss (x ATR)", value=2.0, step=0.1)
tp_multiplier = st.sidebar.number_input("Take Profit (x ATR)", value=4.0, step=0.1)

# -----------------------------------------------------------------------------
# 3. NEWS & SENTIMENT ENGINE
# -----------------------------------------------------------------------------
@st.cache_data(ttl=900)
def fetch_news_and_sentiment(ticker_symbol, a_type):
    processed_news = []
    sia = SentimentIntensityAnalyzer()
    total_compound = 0.0

    try:
        if a_type == "crypto":
            coin_clean = ticker_symbol.split("-")[0]
            url = f"https://min-api.cryptocompare.com/data/v2/news/?categories={coin_clean}&excludeCategories=Sponsored"
            res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5).json()
            articles = res.get("Data", [])[:6]
            for item in articles:
                processed_news.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", "#"),
                    "source": item.get("source_info", {}).get("name", "CryptoNews"),
                    "text": f"{item.get('title', '')}. {item.get('body', '')}"
                })
        else:
            ticker_obj = yf.Ticker(ticker_symbol)
            raw_news = ticker_obj.news[:6] if ticker_obj.news else []
            for item in raw_news:
                content = item.get("content", {})
                processed_news.append({
                    "title": content.get("title", item.get("title", "")),
                    "url": content.get("canonicalUrl", {}).get("url", item.get("link", "#")),
                    "source": content.get("provider", {}).get("displayName", "Yahoo Finance"),
                    "text": f"{content.get('title', '')}. {content.get('summary', '')}"
                })

        final_news = []
        for n in processed_news:
            scores = sia.polarity_scores(n["text"])
            c = scores["compound"]
            total_compound += c
            s_label = "🟢 BULLISH" if c >= 0.05 else ("🔴 BEARISH" if c <= -0.05 else "⚪ NEUTRAL")
            final_news.append({
                "title": n["title"], "url": n["url"], "source": n["source"],
                "sentiment": s_label, "score": c, "summary": n["text"][:140] + "..."
            })

        avg_score = total_compound / len(final_news) if final_news else 0.0
        return final_news, avg_score

    except Exception:
        return [], 0.0

news_data, overall_sentiment_score = fetch_news_and_sentiment(symbol, asset_type)

# -----------------------------------------------------------------------------
# 4. DATA FETCHING & INDICATORS
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

df_raw = load_data(symbol, period, interval)

if df_raw.empty or len(df_raw) < 50:
    st.error(f"❌ ไม่สามารถโหลดข้อมูลสำหรับ {symbol} ได้ กรุณาตรวจสอบสัญลักษณ์ Ticker")
    st.stop()

@st.cache_data
def compute_all_indicators(df_input, atr_p):
    df = df_input.copy()
    
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df["SMA20"] = df["Close"].rolling(20).mean()

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(atr_p).mean()

    vyp = (df["High"] + df["Low"] + df["Close"]) / 3 * df["Volume"]
    df["VWAP"] = vyp.cumsum() / df["Volume"].cumsum()
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    std20 = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["SMA20"] + (std20 * 2)
    df["BB_Lower"] = df["SMA20"] - (std20 * 2)

    return df

df = compute_all_indicators(df_raw, atr_period)

# -----------------------------------------------------------------------------
# 5. BACKTEST ENGINE & STRATEGIES
# -----------------------------------------------------------------------------
strategies_list = [
    "01. Golden Cross (EMA20/50)", "02. RSI Oversold Rebound (<30)", "03. MACD Zero-Line Cross",
    "04. Bollinger Band Mean Reversion", "05. VWAP Pullback Strategy", "06. EMA200 Institutional Rebound",
    "07. RSI Momentum Breakout (>60)", "08. MACD + RSI Confluence"
]

def evaluate_signals(df_input, strat_name):
    df_temp = df_input.copy()
    df_temp["Buy_Signal"] = False
    df_temp["Sell_Signal"] = False

    if "01." in strat_name:
        df_temp["Buy_Signal"] = (df_temp["EMA20"] > df_temp["EMA50"]) & (df_temp["EMA20"].shift(1) <= df_temp["EMA50"].shift(1))
        df_temp["Sell_Signal"] = (df_temp["EMA20"] < df_temp["EMA50"]) & (df_temp["EMA20"].shift(1) >= df_temp["EMA50"].shift(1))
    elif "02." in strat_name:
        df_temp["Buy_Signal"] = (df_temp["RSI"] < 35)
        df_temp["Sell_Signal"] = (df_temp["RSI"] > 65)
    elif "03." in strat_name:
        df_temp["Buy_Signal"] = (df_temp["MACD"] > 0) & (df_temp["MACD"].shift(1) <= 0)
        df_temp["Sell_Signal"] = (df_temp["MACD"] < 0) & (df_temp["MACD"].shift(1) >= 0)
    elif "04." in strat_name:
        df_temp["Buy_Signal"] = df_temp["Close"] <= df_temp["BB_Lower"]
        df_temp["Sell_Signal"] = df_temp["Close"] >= df_temp["SMA20"]
    elif "05." in strat_name:
        df_temp["Buy_Signal"] = (df_temp["Low"] <= df_temp["VWAP"]) & (df_temp["Close"] > df_temp["VWAP"])
        df_temp["Sell_Signal"] = df_temp["Close"] < df_temp["EMA20"]
    elif "06." in strat_name:
        near_ema200 = (df_temp["Low"] <= df_temp["EMA200"] * 1.01) & (df_temp["High"] >= df_temp["EMA200"])
        df_temp["Buy_Signal"] = near_ema200 & (df_temp["Close"] > df_temp["EMA200"])
        df_temp["Sell_Signal"] = df_temp["Close"] < df_temp["EMA200"] * 0.98
    elif "07." in strat_name:
        df_temp["Buy_Signal"] = (df_temp["RSI"] > 55) & (df_temp["RSI"].shift(1) <= 55)
        df_temp["Sell_Signal"] = df_temp["RSI"] < 45
    elif "08." in strat_name:
        df_temp["Buy_Signal"] = (df_temp["MACD"] > df_temp["MACD_Signal"]) & (df_temp["RSI"] > 50)
        df_temp["Sell_Signal"] = (df_temp["MACD"] < df_temp["MACD_Signal"]) & (df_temp["RSI"] < 45)

    return df_temp

@st.cache_data
def run_fast_backtest(df_input, strat_name, init_cap, risk_pct, fee, atr_m_sl, atr_m_tp):
    df_temp = evaluate_signals(df_input, strat_name)
    capital = init_cap
    position = False
    entry_price, sl_price, tp_price, entry_date, position_size = 0.0, 0.0, 0.0, None, 0.0
    trades = []
    equity_curve = [init_cap]
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
                raw_pnl = (exit_price - entry_price) * position_size
                total_fee = (entry_price * position_size * fee) + (exit_price * position_size * fee)
                net_profit = raw_pnl - total_fee
                capital += net_profit
                pnl_pct = (net_profit / (entry_price * position_size)) * 100
                trades.append({
                    "Entry Date": entry_date, "Exit Date": current_date,
                    "Entry": entry_price, "Exit": exit_price,
                    "Size": position_size, "Result": reason,
                    "PnL (%)": pnl_pct, "Profit": net_profit,
                    "Capital": capital
                })
                position = False

        elif not position and df_temp["Buy_Signal"].iloc[i] and not np.isnan(current_atr):
            entry_price, entry_date = current_close, current_date
            sl_price = entry_price - (current_atr * atr_m_sl)
            tp_price = entry_price + (current_atr * atr_m_tp)
            risk_amount = capital * risk_pct
            risk_per_unit = entry_price - sl_price
            if risk_per_unit > 0:
                position_size = min(risk_amount / risk_per_unit, (capital * 0.95) / entry_price)
                position = True

        equity_curve.append(capital)
        equity_dates.append(current_date)

    net_profit = capital - init_cap
    return df_temp, pd.DataFrame(trades), pd.DataFrame({"Date": equity_dates, "Equity": equity_curve}).set_index("Date"), net_profit, position, entry_price, sl_price, tp_price, entry_date

# -----------------------------------------------------------------------------
# 6. DYNAMIC STRATEGY DROPDOWN
# -----------------------------------------------------------------------------
strategy_map = {}
dropdown_options = []

for strat in strategies_list:
    _, _, _, net_pnl, _, _, _, _, _ = run_fast_backtest(
        df, strat, initial_capital, risk_per_trade_pct, fee_rate, sl_multiplier, tp_multiplier
    )
    label = f"✅ {strat} (+{net_pnl:,.2f})" if net_pnl > 0 else f"❌ {strat} (-{abs(net_pnl):,.2f})"
    strategy_map[label] = strat
    dropdown_options.append(label)

st.sidebar.subheader("🧠 4. เลือกระบบเทรด")
selected_label = st.sidebar.selectbox("เลือกกลยุทธ์", options=dropdown_options, index=0)
strategy_choice = strategy_map[selected_label]

df, trades_df, equity_df, net_profit_val, position, entry_price, sl_price, tp_price, entry_date = run_fast_backtest(
    df, strategy_choice, initial_capital, risk_per_trade_pct, fee_rate, sl_multiplier, tp_multiplier
)

# -----------------------------------------------------------------------------
# 7. DASHBOARD DISPLAY
# -----------------------------------------------------------------------------
st.subheader(f"📌 สัญญาณปัจจุบัน & ข่าวสาร: {symbol}")

last_row = df.iloc[-1]
c_col1, c_col2 = st.columns(2)

with c_col1:
    st.markdown("##### 🤖 Quantitative Signal")
    st.metric("ราคาล่าสุด", f"{last_row['Close']:,.2f}")
    if position:
        st.warning(f"🔔 ถือออเดอร์ค้างไว้ | Entry: {entry_price:,.2f} | TP: {tp_price:,.2f} | SL: {sl_price:,.2f}")
    elif last_row["Buy_Signal"]:
        st.success(f"✅ BUY SIGNAL AT {last_row['Close']:,.2f}")
    else:
        st.info("⏳ NO SIGNAL - Cash Position")

with c_col2:
    st.markdown("##### 📰 AI Sentiment Score")
    s_text = "🟢 BULLISH" if overall_sentiment_score >= 0.05 else ("🔴 BEARISH" if overall_sentiment_score <= -0.05 else "⚪ NEUTRAL")
    st.metric(label="ข่าวสาร AI สรุป", value=s_text, delta=f"Score: {overall_sentiment_score:.2f}")

st.markdown("---")

# -----------------------------------------------------------------------------
# 8. CHART DISPLAY
# -----------------------------------------------------------------------------
st.subheader(f"📉 กราฟราคา & จุดเข้าออกออเดอร์ ({symbol})")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], line=dict(color='orange', width=1), name="EMA 20"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["EMA200"], line=dict(color='purple', width=1.5), name="EMA 200"), row=1, col=1)

if not trades_df.empty:
    fig.add_trace(go.Scatter(x=trades_df["Entry Date"], y=trades_df["Entry"], mode="markers", marker=dict(symbol="triangle-up", size=12, color="green"), name="Buy"), row=1, col=1)
    fig.add_trace(go.Scatter(x=trades_df["Exit Date"], y=trades_df["Exit"], mode="markers", marker=dict(symbol="triangle-down", size=12, color="red"), name="Exit"), row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color='green', width=1), name="RSI"), row=2, col=1)
fig.update_layout(height=500, margin=dict(l=10, r=10, t=20, b=10))
fig.update_xaxes(rangeslider_visible=False)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# 9. PERFORMANCE METRICS DASHBOARD (ย้ายลงมาไว้ใต้กราฟที่นี่)
# -----------------------------------------------------------------------------
st.subheader(f"📊 สรุปผล Backtest: {strategy_choice}")
col1, col2, col3, col4, col5, col6 = st.columns(6)

if not trades_df.empty:
    total_trades = len(trades_df)
    wins = len(trades_df[trades_df["Profit"] > 0])
    win_rate = (wins / total_trades) * 100
    ret_pct = (net_profit_val / initial_capital) * 100
    gross_profit = trades_df[trades_df["Profit"] > 0]["Profit"].sum()
    gross_loss = abs(trades_df[trades_df["Profit"] < 0]["Profit"].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else gross_profit
    
    equity_df["Peak"] = equity_df["Equity"].cummax()
    equity_df["Drawdown"] = (equity_df["Equity"] - equity_df["Peak"]) / equity_df["Peak"]
    max_drawdown = equity_df["Drawdown"].min() * 100

    col1.metric("จำนวนไม้ที่เทรด", f"{total_trades} ไม้")
    col2.metric("Win Rate", f"{win_rate:.1f}%")
    col3.metric("กำไรสุทธิ", f"{net_profit_val:,.2f}")
    col4.metric("ผลตอบแทน (%)", f"{ret_pct:.2f}%")
    col5.metric("Profit Factor", f"{profit_factor:.2f}")
    col6.metric("Max Drawdown", f"{max_drawdown:.2f}%")
else:
    col1.metric("จำนวนไม้ที่เทรด", "0 ไม้")
    col2.metric("Win Rate", "0.0%")
    col3.metric("กำไรสุทธิ", "0.00")
    col4.metric("ผลตอบแทน (%)", "0.00%")
    col5.metric("Profit Factor", "0.00")
    col6.metric("Max Drawdown", "0.00%")

st.markdown("---")

# -----------------------------------------------------------------------------
# 10. TRADE LOG & EQUITY CURVE
# -----------------------------------------------------------------------------
t_col1, t_col2 = st.columns([6, 4])

with t_col1:
    st.subheader("📋 ประวัติรายการเทรด (Trade Log)")
    if not trades_df.empty:
        formatted_trades = trades_df.copy()
        formatted_trades["Entry Date"] = formatted_trades["Entry Date"].dt.strftime('%Y-%m-%d')
        formatted_trades["Exit Date"] = formatted_trades["Exit Date"].dt.strftime('%Y-%m-%d')
        st.dataframe(
            formatted_trades.style.format({
                "Entry": "{:,.2f}", "Exit": "{:,.2f}", "Size": "{:,.2f}",
                "PnL (%)": "{:+.2f}%", "Profit": "{:+.2f}", "Capital": "{:,.2f}"
            }),
            use_container_width=True,
            height=300
        )
    else:
        st.info("ℹ️ ไม่มีรายการเทรดเกิดขึ้นตามเงื่อนไขที่เลือก (ลองปรับช่วงเวลาย้อนหลังเพิ่มขึ้นที่ Sidebar)")

with t_col2:
    st.subheader("📈 การเติบโตของเงินทุน (Equity Curve)")
    if not equity_df.empty:
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=equity_df.index, y=equity_df["Equity"], mode='lines', line=dict(color='#00CC96', width=2)))
        fig_eq.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
        fig_eq.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig_eq, use_container_width=True)
