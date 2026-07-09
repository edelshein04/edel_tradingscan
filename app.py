import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Swing Scanner", layout="wide")

st.title("📈 Swing Trading Scanner")
st.write("Paso 5: Descarga de datos y cálculo de EMAs")

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

df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()
df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

precio_actual = float(df["Close"].iloc[-1])
ema10 = float(df["EMA10"].iloc[-1])
ema20 = float(df["EMA20"].iloc[-1])
ema50 = float(df["EMA50"].iloc[-1])

condicion_precio = precio_actual > ema50
condicion_ema10 = ema10 > ema20
condicion_ema20 = ema20 > ema50

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
            "EMA50"
        ]
    ].tail(30),
    use_container_width=True
)

st.subheader("📋 Validación de la tesis")

st.write(f"Precio ({precio_actual:.2f}) > EMA50 ({ema50:.2f}) : {'✅' if condicion_precio else '❌'}")
st.write(f"EMA10 ({ema10:.2f}) > EMA20 ({ema20:.2f}) : {'✅' if condicion_ema10 else '❌'}")
st.write(f"EMA20 ({ema20:.2f}) > EMA50 ({ema50:.2f}) : {'✅' if condicion_ema20 else '❌'}")

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
