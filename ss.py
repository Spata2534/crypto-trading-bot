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
st.set_page_config(page_title="Crypto Quant + AI News Analytics", layout="wide")
st.title("📈 Backtest Dashboard & AI News Sentiment Analyzer")

# -----------------------------------------------------------------------------
# 2. SIDEBAR CONFIGURATION
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ 1. เลือกสินทรัพย์และไทม์เฟรม")

crypto_presets = {
    "Bitcoin (BTC/USD)": ("BTC-USD", "BTC"),
    "Ethereum (ETH/USD)": ("ETH-USD", "ETH"),
    "Solana (SOL/USD)": ("SOL-USD", "SOL"),
    "Binance Coin (BNB/USD)": ("BNB-USD", "BNB"),
    "Ripple (XRP/USD)": ("XRP-USD", "XRP"),
    "Cardano (ADA/USD)": ("ADA-USD", "ADA"),
    "Dogecoin (DOGE/USD)": ("DOGE-USD", "DOGE"),
    "Avalanche (AVAX/USD)": ("AVAX-USD", "AVAX"),
    "Custom (ระบุเอง)": ("CUSTOM", "CUSTOM")
}

selected_preset = st.sidebar.selectbox("เลือกเหรียญยอดนิยม", options=list(crypto_presets.keys()), index=0)

if crypto_presets[selected_preset][0] == "CUSTOM":
    symbol = st.sidebar.text_input("พิมพ์สัญลักษณ์ Ticker (เช่น NEAR-USD)", value="NEAR-USD").upper()
    news_symbol = symbol.split("-")[0]
else:
    symbol, news_symbol = crypto_presets[selected_preset]

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
# 3. AI NEWS FETCHING & SENTIMENT ENGINE
# -----------------------------------------------------------------------------
@st.cache_data(ttl=900)
def fetch_crypto_news_and_sentiment(coin_ticker):
    url = f"https://min-api.cryptocompare.com/data/v2/news/?categories={coin_ticker}&excludeCategories=Sponsored"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        res_json = response.json()
        articles = res_json.get("Data", [])
        
        if not articles:
            fallback_url = "https://min-api.cryptocompare.com/data/v2/news/?categories=Market&excludeCategories=Sponsored"
            res_json = requests.get(fallback_url, headers=headers, timeout=5).json()
            articles = res_json.get("Data", [])

        sia = SentimentIntensityAnalyzer()
        processed_news = []
        total_compound = 0.0

        for item in articles[:6]:
            title = item.get("title", "")
            body = item.get("body", "")
            full_text = f"{title}. {body}"
            
            scores = sia.polarity_scores(full_text)
            compound = scores["compound"]
            total_compound += compound
            
            if compound >= 0.05:
                sentiment_label = "🟢 BULLISH"
            elif compound <= -0.05:
                sentiment_label = "🔴 BEARISH"
            else:
                sentiment_label = "⚪ NEUTRAL"

            processed_news.append({
                "title": title,
                "url": item.get("url", "#"),
                "source": item.get("source_info", {}).get("name", "CryptoNews"),
                "sentiment": sentiment_label,
                "score": compound,
                "summary": body[:140] + "..."
            })
            
        avg_score = total_compound / len(processed_news) if processed_news else 0.0
        return processed_news, avg_score

    except Exception:
        return [], 0.0

news_data, overall_sentiment_score = fetch_crypto_news_and_sentiment(news_symbol)

# -----------------------------------------------------------------------------
# 4. DATA FETCHING & INDICATORS CALCULATION
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
    st.error(f"❌ ไม่สามารถโหลดข้อมูลสำหรับ {symbol} ได้ กรุณาตรวจสอบสัญลักษณ์ Asset")
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

    up_move = df["High"].diff()
    down_move = -df["Low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    tr14 = true_range.rolling(14).sum()
    df["Plus_DI"] = 100 * (pd.Series(plus_dm, index=df.index).rolling(14).sum() / tr14)
    df["Minus_DI"] = 100 * (pd.Series(minus_dm, index=df.index).rolling(14).sum() / tr14)
    dx = 100 * (abs(df["Plus_DI"] - df["Minus_DI"]) / (df["Plus_DI"] + df["Minus_DI"]))
    df["ADX"] = dx.rolling(14).mean()

    vyp = (df["High"] + df["Low"] + df["Close"]) / 3 * df["Volume"]
    df["VWAP"] = vyp.cumsum() / df["Volume"].cumsum()
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    std20 = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["SMA20"] + (std20 * 2)
    df["BB_Lower"] = df["SMA20"] - (std20 * 2)

    low14 = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    df["Stoch_K"] = 100 * ((df["Close"] - low14) / (high14 - low14))
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    df["Donchian_High"] = df["High"].rolling(20).max()
    df["Donchian_Low"] = df["Low"].rolling(20).min()
    df["Volume_MA20"] = df["Volume"].rolling(20).mean()

    hl2 = (df["High"] + df["Low"]) / 2
    df["ST_Upper"] = hl2 + (1.5 * df["ATR"])
    df["ST_Lower"] = hl2 - (1.5 * df["ATR"])

    return df

df = compute_all_indicators(df_raw, atr_period)

# -----------------------------------------------------------------------------
# 5. BACKTEST ENGINE & 20 STRATEGIES
# -----------------------------------------------------------------------------
strategies_list = [
    "01. Golden Cross (EMA20/50)", "02. RSI Oversold Rebound (<30)", "03. MACD Zero-Line Cross",
    "04. Bollinger Band Mean Reversion", "05. Donchian Channel Breakout", "06. Supertrend Trend Following",
    "07. Stochastic Crossover (<20)", "08. VWAP Pullback Strategy", "09. Volume Breakout (>2x MA20)",
    "10. Triple EMA System (9/20/50)", "11. ADX Strong Trend Rider", "12. RSI Momentum Breakout (>60)",
    "13. Multi-Timeframe Alignment", "14. ATR Volatility Expansion", "15. Bollinger Squeeze Breakout",
    "16. Trend-Regime Dynamic Pullback", "17. Counter-Trend Exhaustion", "18. Dual Thrust System",
    "19. MACD + RSI Confluence", "20. EMA200 Institutional Rebound"
]

def evaluate_signals(df_input, strat_name):
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
        df_temp["Buy_Signal"] = (df_temp["MACD"] > 0) & (df_temp["MACD"].shift(1) <= 0)
        df_temp["Sell_Signal"] = (df_temp["MACD"] < 0) & (df_temp["MACD"].shift(1) >= 0)
    elif "04." in strat_name:
        df_temp["Buy_Signal"] = df_temp["Close"] <= df_temp["BB_Lower"]
        df_temp["Sell_Signal"] = df_temp["Close"] >= df_temp["SMA20"]
    elif "05." in strat_name:
        df_temp["Buy_Signal"] = df_temp["Close"] > df_temp["Donchian_High"].shift(1)
        df_temp["Sell_Signal"] = df_temp["Close"] < df_temp["Donchian_Low"].shift(1)
    elif "06." in strat_name:
        df_temp["Buy_Signal"] = (df_temp["Close"] > df_temp["ST_Upper"].shift(1))
        df_temp["Sell_Signal"] = (df_temp["Close"] < df_temp["ST_Lower"].shift(1))
    elif "07." in strat_name:
        df_temp["Buy_Signal"] = (df_temp["Stoch_K"] < 20) & (df_temp["Stoch_K"] > df_temp["Stoch_D"])
        df_temp["Sell_Signal"] = (df_temp["Stoch_K"] > 80)
    elif "08." in strat_name:
        df_temp["Buy_Signal"] = (df_temp["Close"] > df_temp["EMA200"]) & (df_temp["Low"] <= df_temp["VWAP"]) & (df_temp["Close"] > df_temp["VWAP"])
        df_temp["Sell_Signal"] = df_temp["Close"] < df_temp["EMA20"]
    elif "09." in strat_name:
        df_temp["Buy_Signal"] = (df_temp["Volume"] > df_temp["Volume_MA20"] * 2.0) & (df_temp["Close"] > df_temp["Open"])
        df_temp["Sell_Signal"] = df_temp["Close"] < df_temp["EMA20"]
    elif "10." in strat_name:
        df_temp["Buy_Signal"] = (df_temp["EMA9"] > df_temp["EMA20"]) & (df_temp["EMA20"] > df_temp["EMA50"])
        df_temp["Sell_Signal"] = df_temp["EMA9"] < df_temp["EMA20"]
    elif "11." in strat_name:
        df_temp["Buy_Signal"] = (df_temp["ADX"] > 25) & (df_temp["Plus_DI"] > df_temp["Minus_DI"])
        df_temp["Sell_Signal"] = df_temp["Minus_DI"] > df_temp["Plus_DI"]
    elif "12." in strat_name:
        df_temp["Buy_Signal"] = (df_temp["RSI"] > 60) & (df_temp["RSI"].shift(1) <= 60)
        df_temp["Sell_Signal"] = df_temp["RSI"] < 50
    elif "13." in strat_name:
        df_temp["Buy_Signal"] = (df_temp["Close"] > df_temp["EMA200"]) & (df_temp["RSI"] > 50) & (df_temp["MACD"] > df_temp["MACD_Signal"])
        df_temp["Sell_Signal"] = (df_temp["RSI"] < 45)
    elif "14." in strat_name:
        atr_breakout = (df_temp["High"] - df_temp["Low"]) > (df_temp["ATR"] * 1.8)
        df_temp["Buy_Signal"] = atr_breakout & (df_temp["Close"] > df_temp["Open"])
        df_temp["Sell_Signal"] = df_temp["Close"] < df_temp["EMA20"]
    elif "15." in strat_name:
        bb_width = (df_temp["BB_Upper"] - df_temp["BB_Lower"]) / df_temp["SMA20"]
        squeeze = bb_width < bb_width.rolling(50).min() * 1.15
        df_temp["Buy_Signal"] = squeeze & (df_temp["Close"] > df_temp["BB_Upper"])
        df_temp["Sell_Signal"] = df_temp["Close"] < df_temp["SMA20"]
    elif "16." in strat_name:
        uptrend = (df_temp["Close"] > df_temp["EMA200"]) & (df_temp["ADX"] > 18)
        rsi_pullback = (df_temp["RSI"] >= 38) & (df_temp["RSI"] <= 58)
        price_support = (df_temp["Low"] <= df_temp["EMA20"] * 1.015) | (df_temp["Low"] <= df_temp["VWAP"] * 1.015)
        df_temp["Buy_Signal"] = uptrend & rsi_pullback & price_support & (df_temp["Close"] > df_temp["Open"])
        df_temp["Sell_Signal"] = (df_temp["RSI"] > 75) | (df_temp["Close"] < df_temp["EMA50"])
    elif "17." in strat_name:
        df_temp["Buy_Signal"] = (df_temp["RSI"] < 25) & (df_temp["Low"] <= df_temp["BB_Lower"])
        df_temp["Sell_Signal"] = df_temp["RSI"] > 50
    elif "18." in strat_name:
        range_val = (df_temp["High"].shift(1) - df_temp["Low"].shift(1)) * 0.7
        df_temp["Buy_Signal"] = df_temp["Close"] > (df_temp["Open"] + range_val)
        df_temp["Sell_Signal"] = df_temp["Close"] < (df_temp["Open"] - range_val)
    elif "19." in strat_name:
        df_temp["Buy_Signal"] = (df_temp["MACD"] > df_temp["MACD_Signal"]) & (df_temp["RSI"] > 55) & (df_temp["RSI"].shift(1) <= 55)
        df_temp["Sell_Signal"] = (df_temp["MACD"] < df_temp["MACD_Signal"]) & (df_temp["RSI"] < 45)
    elif "20." in strat_name:
        near_ema200 = (df_temp["Low"] <= df_temp["EMA200"] * 1.01) & (df_temp["High"] >= df_temp["EMA200"])
        df_temp["Buy_Signal"] = near_ema200 & (df_temp["Close"] > df_temp["EMA200"]) & (df_temp["Close"] > df_temp["Open"])
        df_temp["Sell_Signal"] = df_temp["Close"] < df_temp["EMA200"] * 0.98

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
                raw_pnl_usd = (exit_price - entry_price) * position_size
                total_fee = (entry_price * position_size * fee) + (exit_price * position_size * fee)
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
    label = f"✅ {strat} (+$ {net_pnl:.2f})" if net_pnl > 0 else f"❌ {strat} (-$ {abs(net_pnl):.2f})"
    strategy_map[label] = strat
    dropdown_options.append(label)

st.sidebar.subheader("🧠 4. เลือกระบบเทรด (20 Strategies)")
selected_label = st.sidebar.selectbox("เลือกกลยุทธ์", options=dropdown_options, index=15)
strategy_choice = strategy_map[selected_label]

df, trades_df, equity_df, net_profit_usd, position, entry_price, sl_price, tp_price, entry_date = run_fast_backtest(
    df, strategy_choice, initial_capital, risk_per_trade_pct, fee_rate, sl_multiplier, tp_multiplier
)

# -----------------------------------------------------------------------------
# 7. TOP SECTION: SIGNAL + AI SENTIMENT DASHBOARD
# -----------------------------------------------------------------------------
st.subheader("📌 1. สถานะสัญญาณเทรด & AI Market Sentiment")

last_row = df.iloc[-1]
col_sys1, col_sys2 = st.columns([1, 1])

with col_sys1:
    st.markdown("##### 🤖 Quantitative Signal Result")
    c1, c2 = st.columns(2)
    c1.metric("ราคาปัจจุบัน", f"${last_row['Close']:,.2f}")
    c2.metric("RSI (14)", f"{last_row['RSI']:.2f}")

    if position:
        st.warning(f"🔔 มีสถานะค้างอยู่ | Entry: ${entry_price:,.2f} | TP: ${tp_price:,.2f} | SL: ${sl_price:,.2f}")
    elif last_row["Buy_Signal"]:
        st.success(f"✅ BUY SIGNAL CONFIRMED | ราคา: ${last_row['Close']:,.2f}")
    else:
        st.info("⏳ NO SIGNAL - ถือเงินสด (Cash Position)")

with col_sys2:
    st.markdown("##### 📰 AI Sentiment Score (ข่าวเรียลไทม์)")
    
    if overall_sentiment_score >= 0.05:
        sentiment_status = "🟢 BULLISH (ข่าวเชิงบวก)"
    elif overall_sentiment_score <= -0.05:
        sentiment_status = "🔴 BEARISH (ข่าวเชิงลบ)"
    else:
        sentiment_status = "⚪ NEUTRAL (ข่าวเป็นกลาง)"

    st.metric(label=f"สรุปข่าวสาร {news_symbol}", value=sentiment_status, delta=f"Score: {overall_sentiment_score:.2f}")
    
    if last_row["Buy_Signal"] and overall_sentiment_score >= 0.05:
        st.success("🔥 High Confluence: สัญญาณเทรดทางเทคนิคและข่าวมองไปในทางเดียวกัน (Strong Buy)")
    elif last_row["Buy_Signal"] and overall_sentiment_score < -0.05:
        st.error("⚠️ Divergence Warning: สัญญาณเทรดเป็น BUY แต่ข่าวยังเป็น BEARISH ระวัง False Break")
    else:
        st.caption("ℹ️ ตลาดอยู่ในสภาวะปกติตามสัญญาณ Quantitative Model")

st.markdown("---")

# -----------------------------------------------------------------------------
# 8. LIVE NEWS FEED & DETAILS
# -----------------------------------------------------------------------------
st.subheader(f"🌐 2. อัปเดตข่าวสารล่าสุดเกี่ยวกับ {news_symbol} (AI Analyzed)")

if news_data:
    news_cols = st.columns(3)
    for idx, item in enumerate(news_data[:6]):
        col_target = news_cols[idx % 3]
        with col_target:
            with st.container(border=True):
                st.markdown(f"**[{item['title']}]({item['url']})**")
                st.caption(f"สำนักข่าว: {item['source']} | Sentiment: **{item['sentiment']}** (Score: {item['score']:.2f})")
                st.write(item['summary'])
else:
    st.warning("⚠️ ไม่พบข้อมูลข่าวสารล่าสุด หรือ API มีปัญหากับการดึงข้อมูล")

st.markdown("---")

# -----------------------------------------------------------------------------
# 9. INTERACTIVE CHARTING & BACKTEST RESULTS
# -----------------------------------------------------------------------------
st.subheader(f"📉 3. กราฟราคา & จุดเข้าออกออเดอร์ ({symbol})")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], line=dict(color='orange', width=1), name="EMA 20"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["EMA200"], line=dict(color='purple', width=1.5), name="EMA 200"), row=1, col=1)

if not trades_df.empty:
    fig.add_trace(go.Scatter(x=trades_df["Entry Date"], y=trades_df["Entry"], mode="markers", marker=dict(symbol="triangle-up", size=12, color="green"), name="Buy"), row=1, col=1)
    fig.add_trace(go.Scatter(x=trades_df["Exit Date"], y=trades_df["Exit"], mode="markers", marker=dict(symbol="triangle-down", size=12, color="red"), name="Exit"), row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color='green', width=1), name="RSI"), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

fig.update_layout(height=550, margin=dict(l=10, r=10, t=20, b=10))
fig.update_xaxes(rangeslider_visible=False)

st.plotly_chart(fig, use_container_width=True)

# Performance Metrics Dashboard
st.subheader(f"📊 สรุปผล Backtest: {strategy_choice}")
col1, col2, col3, col4, col5, col6 = st.columns(6)

if not trades_df.empty:
    total_trades = len(trades_df)
    wins = len(trades_df[trades_df["Profit ($)"] > 0])
    win_rate = (wins / total_trades) * 100
    ret_pct = (net_profit_usd / initial_capital) * 100
    gross_profit = trades_df[trades_df["Profit ($)"] > 0]["Profit ($)"].sum()
    gross_loss = abs(trades_df[trades_df["Profit ($)"] < 0]["Profit ($)"].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else gross_profit
    
    equity_df["Peak"] = equity_df["Equity"].cummax()
    equity_df["Drawdown"] = (equity_df["Equity"] - equity_df["Peak"]) / equity_df["Peak"]
    max_drawdown = equity_df["Drawdown"].min() * 100

    col1.metric("จำนวนไม้", f"{total_trades}")
    col2.metric("Win Rate", f"{win_rate:.1f}%")
    col3.metric("กำไรสุทธิ ($)", f"${net_profit_usd:.2f}")
    col4.metric("ผลตอบแทน (%)", f"{ret_pct:.2f}%")
    col5.metric("Profit Factor", f"{profit_factor:.2f}")
    col6.metric("Max Drawdown", f"{max_drawdown:.2f}%")
else:
    col1.metric("จำนวนไม้", "0")
    col2.metric("Win Rate", "0.0%")
    col3.metric("กำไรสุทธิ ($)", "$0.00")
    col4.metric("ผลตอบแทน (%)", "0.00%")
    col5.metric("Profit Factor", "0.00")
    col6.metric("Max Drawdown", "0.00%")

st.markdown("---")

# -----------------------------------------------------------------------------
# 10. TRADE HISTORY TABLE & EQUITY CURVE (เติมส่วนนี้ที่หายไป)
# -----------------------------------------------------------------------------
col_tab1, col_tab2 = st.columns([6, 4])

with col_tab1:
    st.subheader("📋 ประวัติรายการเทรดทั้งหมด (Trade Log History)")
    if not trades_df.empty:
        # จัดรูปแบบตัวเลขให้อ่านง่าย
        formatted_trades = trades_df.copy()
        formatted_trades["Entry Date"] = formatted_trades["Entry Date"].dt.strftime('%Y-%m-%d %H:%M')
        formatted_trades["Exit Date"] = formatted_trades["Exit Date"].dt.strftime('%Y-%m-%d %H:%M')
        
        st.dataframe(
            formatted_trades.style.format({
                "Entry": "${:,.2f}",
                "Exit": "${:,.2f}",
                "Size (Units)": "{:,.4f}",
                "PnL (%)": "{:+.2f}%",
                "Profit ($)": "${:+.2f}",
                "Capital After Trade": "${:,.2f}"
            }),
            use_container_width=True,
            height=300
        )
    else:
        st.info("ℹ️ ไม่มีประวัติการเทรดเกิดขึ้นในช่วงเวลาและเงื่อนไขกลยุทธ์ที่เลือก")

with col_tab2:
    st.subheader("📈 การเติบโตของพอร์ต (Equity Curve)")
    if not equity_df.empty:
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(x=equity_df.index, y=equity_df["Equity"], mode='lines', name='Account Balance', line=dict(color='#00CC96', width=2)))
        fig_equity.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
        fig_equity.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig_equity, use_container_width=True)
