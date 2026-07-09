import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Swing Scanner", layout="wide")

st.title("📈 Swing Trading Scanner")
st.write("Paso 8: EMAs + RSI 14 + MACD + Volumen")

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

# MACD 12,26,9
df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = df["EMA12"] - df["EMA26"]
df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

# Volumen relativo
df["VOL_AVG20"] = df["Volume"].rolling(20).mean()
df["VOL_REL"] = df["Volume"] / df["VOL_AVG20"]

# =========================
# ÚLTIMOS VALORES
# =========================

precio_actual = float(df["Close"].iloc[-1])

open_actual = float(df["Open"].iloc[-1])
close_actual = float(df["Close"].iloc[-1])

ema10 = float(df["EMA10"].iloc[-1])
ema20 = float(df["EMA20"].iloc[-1])
ema50 = float(df["EMA50"].iloc[-1])

rsi14 = float(df["RSI14"].iloc[-1])

macd = float(df["MACD"].iloc[-1])
macd_signal = float(df["MACD_SIGNAL"].iloc[-1])
macd_hist = float(df["MACD_HIST"].iloc[-1])
macd_hist_prev = float(df["MACD_HIST"].iloc[-2])
macd_hist_prev2 = float(df["MACD_HIST"].iloc[-3])

volumen_actual = float(df["Volume"].iloc[-1])
volumen_prev = float(df["Volume"].iloc[-2])
vol_avg20 = float(df["VOL_AVG20"].iloc[-1])
vol_rel = float(df["VOL_REL"].iloc[-1])

# =========================
# SCORE EMA
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
# SCORE RSI
# =========================

score_rsi = 0

condicion_rsi_ideal = 55 <= rsi14 <= 68
condicion_rsi_aceptable = 50 <= rsi14 < 55 or 68 < rsi14 <= 75

if condicion_rsi_ideal:
    score_rsi += 15
elif condicion_rsi_aceptable:
    score_rsi += 8

# =========================
# SCORE MACD
# =========================

score_macd = 0

condicion_macd_signal = macd > macd_signal
condicion_macd_hist_positivo = macd_hist > 0
condicion_macd_hist_creciente = macd_hist > macd_hist_prev > macd_hist_prev2

if condicion_macd_signal:
    score_macd += 8

if condicion_macd_hist_positivo:
    score_macd += 8

if condicion_macd_hist_creciente:
    score_macd += 4

# =========================
# SCORE VOLUMEN
# =========================

score_volumen = 0

condicion_vol_promedio = volumen_actual > vol_avg20
condicion_vol_15x = volumen_actual > 1.5 * vol_avg20
condicion_vela_alcista_volumen = close_actual > open_actual and volumen_actual > volumen_prev

if condicion_vol_promedio:
    score_volumen += 5

if condicion_vol_15x:
    score_volumen += 5

if condicion_vela_alcista_volumen:
    score_volumen += 5

# =========================
# SCORE TOTAL PARCIAL
# =========================

score_total = score_ema + score_rsi + score_macd + score_volumen

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
            "VOL_AVG20",
            "VOL_REL",
            "EMA10",
            "EMA20",
            "EMA50",
            "RSI14",
            "MACD",
            "MACD_SIGNAL",
            "MACD_HIST"
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

st.write(f"MACD ({macd:.4f}) > Signal ({macd_signal:.4f}) : {'✅' if condicion_macd_signal else '❌'}")
st.write(f"Hist MACD ({macd_hist:.4f}) > 0 : {'✅' if condicion_macd_hist_positivo else '❌'}")
st.write(f"Hist MACD creciente 3 días : {'✅' if condicion_macd_hist_creciente else '❌'}")

st.write(f"Volumen actual ({volumen_actual:,.0f}) > promedio 20D ({vol_avg20:,.0f}) : {'✅' if condicion_vol_promedio else '❌'}")
st.write(f"Volumen relativo ({vol_rel:.2f}x) > 1.5x : {'✅' if condicion_vol_15x else '❌'}")
st.write(f"Vela alcista con volumen mayor al día anterior : {'✅' if condicion_vela_alcista_volumen else '❌'}")

# =========================
# SCORE
# =========================

st.subheader("🎯 Score parcial")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Score EMA", f"{score_ema}/30")

with col2:
    st.metric("Score RSI", f"{score_rsi}/15")

with col3:
    st.metric("Score MACD", f"{score_macd}/20")

with col4:
    st.metric("Score Volumen", f"{score_volumen}/15")

with col5:
    st.metric("Score parcial", f"{score_total}/80")
