import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Swing Scanner", layout="wide")

st.title("📈 Swing Trading Scanner")
st.write("Scanner completo diario para swing trading 3 a 10 días")

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
rs = avg_gain / avg_loss.replace(0, np.nan)
df["RSI14"] = 100 - (100 / (1 + rs))
df["RSI14"] = df["RSI14"].fillna(50)

# MACD 12,26,9
df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = df["EMA12"] - df["EMA26"]
df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

# Volumen
df["VOL_AVG20"] = df["Volume"].rolling(20).mean()
df["VOL_REL"] = df["Volume"] / df["VOL_AVG20"]

# Acción del precio
df["HIGH_10D"] = df["High"].rolling(10).max().shift(1)
df["LOW_10D"] = df["Low"].rolling(10).min().shift(1)

# ATR14
high_low = df["High"] - df["Low"]
high_close_prev = (df["High"] - df["Close"].shift(1)).abs()
low_close_prev = (df["Low"] - df["Close"].shift(1)).abs()

df["TRUE_RANGE"] = pd.concat(
    [high_low, high_close_prev, low_close_prev],
    axis=1
).max(axis=1)

df["ATR14"] = df["TRUE_RANGE"].rolling(14).mean()
df["ATR_PCT"] = (df["ATR14"] / df["Close"]) * 100

df = df.dropna().reset_index(drop=True)

if len(df) < 60:
    st.error("No hay suficientes datos para calcular todos los indicadores.")
    st.stop()

# =========================
# ÚLTIMOS VALORES
# =========================

last = df.iloc[-1]
prev = df.iloc[-2]
prev2 = df.iloc[-3]

precio_actual = float(last["Close"])
open_actual = float(last["Open"])
close_actual = float(last["Close"])
high_actual = float(last["High"])
low_actual = float(last["Low"])

ema10 = float(last["EMA10"])
ema20 = float(last["EMA20"])
ema50 = float(last["EMA50"])

rsi14 = float(last["RSI14"])

macd = float(last["MACD"])
macd_signal = float(last["MACD_SIGNAL"])
macd_hist = float(last["MACD_HIST"])
macd_hist_prev = float(prev["MACD_HIST"])
macd_hist_prev2 = float(prev2["MACD_HIST"])

volumen_actual = float(last["Volume"])
volumen_prev = float(prev["Volume"])
vol_avg20 = float(last["VOL_AVG20"])
vol_rel = float(last["VOL_REL"])

high_10d = float(last["HIGH_10D"])
low_10d = float(last["LOW_10D"])

atr14 = float(last["ATR14"])
atr_pct = float(last["ATR_PCT"])

# =========================
# SCORE EMA — 30
# =========================

score_ema = 0

condicion_precio = precio_actual > ema50
condicion_ema10 = ema10 > ema20
condicion_ema20 = ema20 > ema50

if condicion_precio:
    score_ema += 10

if condicion_ema10:
    score_ema += 10

if condicion_ema20:
    score_ema += 10

# =========================
# SCORE RSI — 15
# =========================

score_rsi = 0

condicion_rsi_ideal = 55 <= rsi14 <= 68
condicion_rsi_aceptable = 50 <= rsi14 < 55 or 68 < rsi14 <= 75

if condicion_rsi_ideal:
    score_rsi += 15
elif condicion_rsi_aceptable:
    score_rsi += 8

# =========================
# SCORE MACD — 20
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
# SCORE VOLUMEN — 15
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
# SCORE ACCIÓN DEL PRECIO — 10
# =========================

score_precio = 0

condicion_breakout_10d = precio_actual > high_10d

rango_diario = high_actual - low_actual
posicion_cierre = 0

if rango_diario > 0:
    posicion_cierre = (close_actual - low_actual) / rango_diario

condicion_cierre_superior = posicion_cierre >= 0.66

if condicion_breakout_10d:
    score_precio += 5

if condicion_cierre_superior:
    score_precio += 5

# =========================
# SCORE TOTAL
# =========================

score_total = score_ema + score_rsi + score_macd + score_volumen + score_precio

# =========================
# ENTRADA, STOP Y TARGETS
# =========================

entrada_sugerida = max(precio_actual, high_10d)

stop_atr = entrada_sugerida - (2 * atr14)
stop_5pct = entrada_sugerida * 0.95

# El stop sugerido usa el nivel más cercano que no exceda 5% de riesgo
stop_loss = max(stop_atr, stop_5pct, low_10d if low_10d < entrada_sugerida else stop_atr)

riesgo = entrada_sugerida - stop_loss

if riesgo > 0:
    riesgo_pct = (riesgo / entrada_sugerida) * 100
    tp_2r = entrada_sugerida + (2 * riesgo)
    tp_3r = entrada_sugerida + (3 * riesgo)
else:
    riesgo_pct = np.nan
    tp_2r = np.nan
    tp_3r = np.nan

# =========================
# DECISIÓN
# =========================

if score_total >= 90:
    decision = "COMPRA AGRESIVA"
elif score_total >= 80:
    decision = "COMPRAR"
elif score_total >= 70:
    decision = "MANTENER / WATCHLIST"
else:
    decision = "NO COMPRAR"

# =========================
# DASHBOARD
# =========================

st.subheader(f"Resumen para {ticker}")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Decisión", decision)

with col2:
    st.metric("Score Total", f"{score_total}/90")

with col3:
    st.metric("Precio", f"${precio_actual:.2f}")

with col4:
    st.metric("Entrada sugerida", f"${entrada_sugerida:.2f}")

with col5:
    st.metric("Stop Loss", f"${stop_loss:.2f}")

col6, col7, col8, col9 = st.columns(4)

with col6:
    st.metric("Riesgo", f"{riesgo_pct:.2f}%")

with col7:
    st.metric("TP 2R", f"${tp_2r:.2f}")

with col8:
    st.metric("TP 3R", f"${tp_3r:.2f}")

with col9:
    st.metric("ATR %", f"{atr_pct:.2f}%")

# =========================
# SCORE POR MÓDULO
# =========================

st.subheader("🎯 Score por módulo")

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    st.metric("EMA", f"{score_ema}/30")

with c2:
    st.metric("RSI", f"{score_rsi}/15")

with c3:
    st.metric("MACD", f"{score_macd}/20")

with c4:
    st.metric("Volumen", f"{score_volumen}/15")

with c5:
    st.metric("Precio", f"{score_precio}/10")

with c6:
    st.metric("Total", f"{score_total}/90")

# =========================
# VALIDACIÓN DE TESIS
# =========================

st.subheader("📋 Validación de condiciones")

st.write(f"Precio ({precio_actual:.2f}) > EMA50 ({ema50:.2f}) : {'✅' if condicion_precio else '❌'}")
st.write(f"EMA10 ({ema10:.2f}) > EMA20 ({ema20:.2f}) : {'✅' if condicion_ema10 else '❌'}")
st.write(f"EMA20 ({ema20:.2f}) > EMA50 ({ema50:.2f}) : {'✅' if condicion_ema20 else '❌'}")

st.write(f"RSI14 ({rsi14:.2f}) ideal 55–68 : {'✅' if condicion_rsi_ideal else '❌'}")

st.write(f"MACD ({macd:.4f}) > Signal ({macd_signal:.4f}) : {'✅' if condicion_macd_signal else '❌'}")
st.write(f"Hist MACD ({macd_hist:.4f}) > 0 : {'✅' if condicion_macd_hist_positivo else '❌'}")
st.write(f"Hist MACD creciente 3 días : {'✅' if condicion_macd_hist_creciente else '❌'}")

st.write(f"Volumen actual ({volumen_actual:,.0f}) > promedio 20D ({vol_avg20:,.0f}) : {'✅' if condicion_vol_promedio else '❌'}")
st.write(f"Volumen relativo ({vol_rel:.2f}x) > 1.5x : {'✅' if condicion_vol_15x else '❌'}")
st.write(f"Vela alcista con volumen mayor al día anterior : {'✅' if condicion_vela_alcista_volumen else '❌'}")

st.write(f"Breakout sobre máximo 10D ({high_10d:.2f}) : {'✅' if condicion_breakout_10d else '❌'}")
st.write(f"Cierre en tercio superior de la vela ({posicion_cierre:.2f}) : {'✅' if condicion_cierre_superior else '❌'}")

# =========================
# TABLA DE INDICADORES
# =========================

st.subheader("📊 Últimas 30 velas con indicadores")

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
            "MACD_HIST",
            "HIGH_10D",
            "LOW_10D",
            "ATR14",
            "ATR_PCT"
        ]
    ].tail(30),
    use_container_width=True
)
