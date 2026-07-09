import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Swing Scanner", layout="wide")

st.title("📈 Swing Trading Scanner")
st.write("Paso 6: EMAs + RSI 14")

ticker = st.text_input("Ticker", "AAPL").upper().strip()

df = yf.download(
    ticker,
    period="1y",
    interval="1d",
    auto_adjust=True,
    progress=False
)

if df.empty:
    st.error("No se pudieron obtener datos para este ticker.")
    st.stop()

# Corregir columnas MultiIndex de yfinance
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.reset_index()

# =========================
# INDICADORES
# =========================

# EMAs
df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()
df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

# RSI 14
delta = df["Close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()

rs = avg_gain / avg_loss
df["RSI14"] = 100 - (100 / (1 + rs))

# =========================
# ÚLTIMOS VALORES
# =========================

precio_actual = float(df["Close"].iloc[-1])

ema10 = float(df["EMA10"].iloc[-1])
ema20 = float(df["EMA20"].iloc[-1])
ema50 = float(df["EMA50"].iloc[-1])

rsi14 = float(df["RSI14"].iloc[-1])

# =========================
# CONDICIONES EMA
# =========================

condicion_precio = precio_actual > ema50
condicion_ema10 = ema10 > ema20
condicion_ema20 = ema20 > ema50

score_ema = 0

if condicion_precio:
    score_ema += 10

if condicion_ema10:
    score_ema += 10

if condicion_ema20:
    score_ema += 10

# =========================
# CONDICIONES RSI
# =========================

score_rsi = 0

condicion_rsi_ideal = 55 <= rsi14 <= 68
condicion_rsi_aceptable = 50 <= rsi14 < 55 or 68 < rsi14 <= 75

if condicion_rsi_ideal:
    score_rsi += 15
elif condicion_rsi_aceptable:
    score_rsi += 8

# =========================
# SCORE TOTAL PARCIAL
# =========================

score_total = score_ema + score_rsi

# =========================
# TABLA
# =========================

st.subheader(f"Datos diarios para {ticker}")

st.dataframe(
    df[
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "EMA10",
            "EMA20",
            "EMA50",
            "RSI14"
        ]
    ].tail(30),
    use_container_width=True
)

# =========================
# VALIDACIÓN DE TESIS
# =========================

st.subheader("📋 Validación de la tesis")

st.write(f"Precio ({precio_actual:.2f}) > EMA50 ({ema50:.2f}) : {'✅' if condicion_precio else '❌'}")
st.write(f"EMA10 ({ema10:.2f}) > EMA20 ({ema20:.2f}) : {'✅' if condicion_ema10 else '❌'}")
st.write(f"EMA20 ({ema20:.2f}) > EMA50 ({ema50:.2f}) : {'✅' if condicion_ema20 else '❌'}")

st.write(f"RSI14 ({rsi14:.2f}) entre 55 y 68 : {'✅' if condicion_rsi_ideal else '❌'}")

# =========================
# SCORE
# =========================

st.subheader("🎯 Score parcial")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Score EMA", f"{score_ema}/30")

with col2:
    st.metric("Score RSI", f"{score_rsi}/15")

with col3:
    st.metric("Score parcial", f"{score_total}/45")
