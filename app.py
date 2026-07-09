import streamlit as st
import yfinance as yf
import pandas as pd

# ============================================
# CONFIGURACIÓN DE LA APP
# ============================================

st.set_page_config(
    page_title="Swing Scanner",
    layout="wide"
)

st.title("📈 Swing Trading Scanner")
st.write("Paso 5: Descarga de datos y cálculo de EMAs")

# ============================================
# INPUT DEL USUARIO
# ============================================

ticker = st.text_input(
    "Ticker",
    "AAPL"
).upper().strip()

# ============================================
# DESCARGA DE DATOS
# ============================================

df = yf.download(
    ticker,
    period="1y",
    interval="1d",
    auto_adjust=True,
    progress=False
)

# ============================================
# VALIDAR DATOS
# ============================================

if df.empty:
    st.error("No se pudieron obtener datos para este ticker.")
    st.stop()

# ============================================
# CALCULAR EMAs
# ============================================

df["EMA10"] = df["Close"].ewm(
    span=10,
    adjust=False
).mean()

df["EMA20"] = df["Close"].ewm(
    span=20,
    adjust=False
).mean()

df["EMA50"] = df["Close"].ewm(
    span=50,
    adjust=False
).mean()

# ============================================
# EVALUAR CONDICIONES DE LA TESIS
# ============================================

precio_actual = df["Close"].iloc[-1]
ema10 = df["EMA10"].iloc[-1]
ema20 = df["EMA20"].iloc[-1]
ema50 = df["EMA50"].iloc[-1]

condicion_precio = precio_actual > ema50
condicion_ema10 = ema10 > ema20
condicion_ema20 = ema20 > ema50

# ============================================
# MOSTRAR RESULTADOS
# ============================================

st.subheader(f"Datos diarios para {ticker}")

st.dataframe(
    df[
        [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "EMA10",
            "EMA20",
            "EMA50"
        ]
    ].tail(30),
    use_container_width=True
)

# ============================================
# ESTADO DE LAS CONDICIONES
# ============================================

st.subheader("📋 Validación de la tesis")

st.write(
    f"Precio ({precio_actual:.2f}) > EMA50 ({ema50:.2f}) : "
    f"{'✅' if condicion_precio else '❌'}"
)

st.write(
    f"EMA10 ({ema10:.2f}) > EMA20 ({ema20:.2f}) : "
    f"{'✅' if condicion_ema10 else '❌'}"
)

st.write(
    f"EMA20 ({ema20:.2f}) > EMA50 ({ema50:.2f}) : "
    f"{'✅' if condicion_ema20 else '❌'}"
)

# ============================================
# SCORE PARCIAL
# ============================================

score_ema = 0

if condicion_precio:
    score_ema += 10

if condicion_ema10:
    score_ema += 10

if condicion_ema20:
    score_ema += 10

st.subheader("🎯 Score de tendencia")

st.metric(
    label="Score EMA",
    value=f"{score_ema}/30"
)
