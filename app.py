import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Swing Trading Scanner",
    layout="wide"
)

st.title("📈 Swing Trading Scanner")
st.caption(
    "Estrategia de 3 a 10 días con análisis diario "
    "y confirmaciones intradía de 4H y 1H."
)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def limpiar_columnas_yfinance(df):
    """
    Corrige las columnas MultiIndex que yfinance puede devolver.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df.copy()


def calcular_rsi(close, periodo=14):
    """
    Calcula RSI utilizando medias exponenciales tipo Wilder.
    """
    delta = close.diff()

    ganancias = delta.clip(lower=0)
    perdidas = -delta.clip(upper=0)

    media_ganancias = ganancias.ewm(
        alpha=1 / periodo,
        adjust=False
    ).mean()

    media_perdidas = perdidas.ewm(
        alpha=1 / periodo,
        adjust=False
    ).mean()

    rs = media_ganancias / media_perdidas.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)


def calcular_indicadores(df):
    """
    Calcula EMAs, RSI, MACD, volumen relativo, ATR
    y niveles de máximos y mínimos recientes.
    """
    df = df.copy()

    # EMAs
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

    # RSI
    df["RSI14"] = calcular_rsi(
        df["Close"],
        periodo=14
    )

    # MACD 12, 26, 9
    ema12 = df["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = df["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_SIGNAL"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["MACD_HIST"] = (
        df["MACD"] - df["MACD_SIGNAL"]
    )

    # Volumen relativo
    df["VOL_AVG20"] = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    df["VOL_REL"] = (
        df["Volume"] / df["VOL_AVG20"]
    )

    # Máximos y mínimos anteriores
    df["HIGH_10D"] = (
        df["High"]
        .rolling(10)
        .max()
        .shift(1)
    )

    df["LOW_10D"] = (
        df["Low"]
        .rolling(10)
        .min()
        .shift(1)
    )

    # ATR14
    high_low = df["High"] - df["Low"]

    high_close_prev = (
        df["High"] - df["Close"].shift(1)
    ).abs()

    low_close_prev = (
        df["Low"] - df["Close"].shift(1)
    ).abs()

    df["TRUE_RANGE"] = pd.concat(
        [
            high_low,
            high_close_prev,
            low_close_prev
        ],
        axis=1
    ).max(axis=1)

    df["ATR14"] = (
        df["TRUE_RANGE"]
        .rolling(14)
        .mean()
    )

    df["ATR_PCT"] = (
        df["ATR14"] / df["Close"]
    ) * 100

    return df


def convertir_1h_a_4h(df_1h):
    """
    Agrupa velas de una hora en velas aproximadas de cuatro horas.

    Cada sesión bursátil se divide en:
    - Primer bloque: primeras cuatro velas de 1H.
    - Segundo bloque: velas restantes de la sesión.
    """
    df = df_1h.copy()

    if "Datetime" in df.columns:
        columna_fecha = "Datetime"
    elif "Date" in df.columns:
        columna_fecha = "Date"
    else:
        raise ValueError(
            "No se encontró columna Date o Datetime."
        )

    df[columna_fecha] = pd.to_datetime(
        df[columna_fecha]
    )

    df["SESSION_DATE"] = (
        df[columna_fecha]
        .dt.date
    )

    df["BLOQUE_4H"] = (
        df.groupby("SESSION_DATE")
        .cumcount() // 4
    )

    df_4h = (
        df.groupby(
            ["SESSION_DATE", "BLOQUE_4H"],
            as_index=False
        )
        .agg(
            {
                columna_fecha: "first",
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum"
            }
        )
    )

    df_4h = df_4h.rename(
        columns={columna_fecha: "Datetime"}
    )

    return df_4h


# ============================================================
# DESCARGA DE DATOS
# ============================================================

ticker = st.text_input(
    "Ticker",
    value="AAPL"
).upper().strip()

if not ticker:
    st.warning("Escribe un ticker.")
    st.stop()

with st.spinner(
    f"Descargando datos de {ticker}..."
):
    df_diario = yf.download(
        ticker,
        period="1y",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    df_1h = yf.download(
        ticker,
        period="60d",
        interval="1h",
        auto_adjust=True,
        progress=False,
        prepost=False
    )


# ============================================================
# VALIDACIÓN DE DATOS
# ============================================================

if df_diario.empty:
    st.error(
        "No se pudieron obtener datos diarios."
    )
    st.stop()

if df_1h.empty:
    st.error(
        "No se pudieron obtener datos intradía de 1H."
    )
    st.stop()

df_diario = limpiar_columnas_yfinance(
    df_diario
).reset_index()

df_1h = limpiar_columnas_yfinance(
    df_1h
).reset_index()

df_4h = convertir_1h_a_4h(
    df_1h
)


# ============================================================
# CALCULAR INDICADORES
# ============================================================

df_diario = calcular_indicadores(
    df_diario
)

df_1h = calcular_indicadores(
    df_1h
)

df_4h = calcular_indicadores(
    df_4h
)

df_diario = (
    df_diario
    .dropna()
    .reset_index(drop=True)
)

df_1h = (
    df_1h
    .dropna()
    .reset_index(drop=True)
)

df_4h = (
    df_4h
    .dropna()
    .reset_index(drop=True)
)

if len(df_diario) < 60:
    st.error(
        "No hay suficientes datos diarios."
    )
    st.stop()

if len(df_1h) < 50:
    st.error(
        "No hay suficientes velas de 1H."
    )
    st.stop()

if len(df_4h) < 50:
    st.error(
        "No hay suficientes velas de 4H."
    )
    st.stop()


# ============================================================
# VALORES ACTUALES
# ============================================================

daily = df_diario.iloc[-1]
daily_prev = df_diario.iloc[-2]
daily_prev2 = df_diario.iloc[-3]

hour_1 = df_1h.iloc[-1]
hour_4 = df_4h.iloc[-1]

precio_actual = float(
    daily["Close"]
)

ema10_d = float(
    daily["EMA10"]
)

ema20_d = float(
    daily["EMA20"]
)

ema50_d = float(
    daily["EMA50"]
)

rsi_d = float(
    daily["RSI14"]
)

macd_d = float(
    daily["MACD"]
)

macd_signal_d = float(
    daily["MACD_SIGNAL"]
)

macd_hist_d = float(
    daily["MACD_HIST"]
)

macd_hist_d_prev = float(
    daily_prev["MACD_HIST"]
)

macd_hist_d_prev2 = float(
    daily_prev2["MACD_HIST"]
)

volumen_actual = float(
    daily["Volume"]
)

volumen_anterior = float(
    daily_prev["Volume"]
)

volumen_promedio = float(
    daily["VOL_AVG20"]
)

volumen_relativo = float(
    daily["VOL_REL"]
)

high_10d = float(
    daily["HIGH_10D"]
)

low_10d = float(
    daily["LOW_10D"]
)

atr14 = float(
    daily["ATR14"]
)

atr_pct = float(
    daily["ATR_PCT"]
)


# ============================================================
# 1. SCORE DE TENDENCIA DIARIA — 25 PUNTOS
# ============================================================

score_tendencia = 0

cond_precio_sobre_ema50 = (
    precio_actual > ema50_d
)

cond_ema10_sobre_ema20 = (
    ema10_d > ema20_d
)

cond_ema20_sobre_ema50 = (
    ema20_d > ema50_d
)

if cond_precio_sobre_ema50:
    score_tendencia += 9

if cond_ema10_sobre_ema20:
    score_tendencia += 8

if cond_ema20_sobre_ema50:
    score_tendencia += 8


# ============================================================
# 2. SCORE MACD DIARIO — 20 PUNTOS
# ============================================================

score_macd = 0

cond_macd_sobre_signal = (
    macd_d > macd_signal_d
)

cond_histograma_positivo = (
    macd_hist_d > 0
)

cond_histograma_creciente = (
    macd_hist_d
    > macd_hist_d_prev
    > macd_hist_d_prev2
)

if cond_macd_sobre_signal:
    score_macd += 8

if cond_histograma_positivo:
    score_macd += 8

if cond_histograma_creciente:
    score_macd += 4


# ============================================================
# 3. SCORE RSI DIARIO — 15 PUNTOS
# ============================================================

score_rsi = 0

cond_rsi_ideal = (
    55 <= rsi_d <= 68
)

cond_rsi_aceptable = (
    50 <= rsi_d < 55
    or
    68 < rsi_d <= 75
)

if cond_rsi_ideal:
    score_rsi = 15

elif cond_rsi_aceptable:
    score_rsi = 8


# ============================================================
# 4. SCORE VOLUMEN DIARIO — 15 PUNTOS
# ============================================================

score_volumen = 0

cond_volumen_sobre_promedio = (
    volumen_actual > volumen_promedio
)

cond_volumen_15x = (
    volumen_actual
    > 1.5 * volumen_promedio
)

cond_vela_alcista_volumen = (
    daily["Close"] > daily["Open"]
    and
    volumen_actual > volumen_anterior
)

if cond_volumen_sobre_promedio:
    score_volumen += 5

if cond_volumen_15x:
    score_volumen += 5

if cond_vela_alcista_volumen:
    score_volumen += 5


# ============================================================
# 5. ACCIÓN DEL PRECIO — 10 PUNTOS
# ============================================================

score_precio = 0

cond_breakout_10d = (
    precio_actual > high_10d
)

rango_diario = float(
    daily["High"] - daily["Low"]
)

if rango_diario > 0:
    posicion_cierre = (
        float(daily["Close"] - daily["Low"])
        / rango_diario
    )
else:
    posicion_cierre = 0

cond_cierre_tercio_superior = (
    posicion_cierre >= 0.66
)

if cond_breakout_10d:
    score_precio += 5

if cond_cierre_tercio_superior:
    score_precio += 5


# ============================================================
# 6. CONFIRMACIÓN 4H — 10 PUNTOS
# ============================================================

score_4h = 0

precio_4h = float(
    hour_4["Close"]
)

ema10_4h = float(
    hour_4["EMA10"]
)

ema20_4h = float(
    hour_4["EMA20"]
)

macd_hist_4h = float(
    hour_4["MACD_HIST"]
)

cond_precio_sobre_ema20_4h = (
    precio_4h > ema20_4h
)

cond_ema10_sobre_ema20_4h = (
    ema10_4h > ema20_4h
)

cond_macd_hist_positivo_4h = (
    macd_hist_4h > 0
)

if cond_precio_sobre_ema20_4h:
    score_4h += 4

if cond_ema10_sobre_ema20_4h:
    score_4h += 3

if cond_macd_hist_positivo_4h:
    score_4h += 3


# ============================================================
# 7. CONFIRMACIÓN 1H — 5 PUNTOS
# ============================================================

score_1h = 0

precio_1h = float(
    hour_1["Close"]
)

ema20_1h = float(
    hour_1["EMA20"]
)

rsi_1h = float(
    hour_1["RSI14"]
)

macd_hist_1h = float(
    hour_1["MACD_HIST"]
)

cond_precio_sobre_ema20_1h = (
    precio_1h > ema20_1h
)

cond_rsi_1h = (
    50 <= rsi_1h <= 70
)

cond_macd_hist_positivo_1h = (
    macd_hist_1h > 0
)

if cond_precio_sobre_ema20_1h:
    score_1h += 2

if cond_rsi_1h:
    score_1h += 2

if cond_macd_hist_positivo_1h:
    score_1h += 1


# ============================================================
# SCORE TOTAL
# ============================================================

score_diario = (
    score_tendencia
    + score_macd
    + score_rsi
    + score_volumen
    + score_precio
)

score_total = (
    score_diario
    + score_4h
    + score_1h
)


# ============================================================
# ENTRADA, STOP LOSS Y OBJETIVOS
# ============================================================

entrada_sugerida = max(
    precio_actual,
    high_10d
)

stop_por_atr = (
    entrada_sugerida
    - 2 * atr14
)

stop_maximo_5pct = (
    entrada_sugerida
    * 0.95
)

niveles_stop = [
    stop_por_atr,
    stop_maximo_5pct
]

if (
    low_10d < entrada_sugerida
    and
    low_10d >= entrada_sugerida * 0.90
):
    niveles_stop.append(
        low_10d
    )

stop_loss = max(
    niveles_stop
)

riesgo_unitario = (
    entrada_sugerida
    - stop_loss
)

if riesgo_unitario > 0:
    riesgo_pct = (
        riesgo_unitario
        / entrada_sugerida
    ) * 100

    tp_2r = (
        entrada_sugerida
        + 2 * riesgo_unitario
    )

    tp_3r = (
        entrada_sugerida
        + 3 * riesgo_unitario
    )

else:
    riesgo_pct = np.nan
    tp_2r = np.nan
    tp_3r = np.nan


# ============================================================
# DECISIÓN FINAL
# ============================================================

if (
    score_total >= 90
    and
    score_4h >= 8
    and
    score_1h >= 4
):
    decision = "COMPRA AGRESIVA"

elif (
    score_total >= 80
    and
    score_4h >= 7
    and
    score_1h >= 3
):
    decision = "COMPRAR"

elif score_total >= 70:
    decision = "MANTENER / WATCHLIST"

else:
    decision = "NO COMPRAR"


# ============================================================
# RESUMEN PRINCIPAL
# ============================================================

st.subheader(
    f"Resumen de {ticker}"
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Decisión",
        decision
    )

with col2:
    st.metric(
        "Score total",
        f"{score_total}/100"
    )

with col3:
    st.metric(
        "Score diario",
        f"{score_diario}/85"
    )

with col4:
    st.metric(
        "Precio actual",
        f"${precio_actual:.2f}"
    )

col5, col6, col7, col8 = st.columns(4)

with col5:
    st.metric(
        "Entrada sugerida",
        f"${entrada_sugerida:.2f}"
    )

with col6:
    st.metric(
        "Stop loss",
        f"${stop_loss:.2f}"
    )

with col7:
    st.metric(
        "Riesgo",
        f"{riesgo_pct:.2f}%"
    )

with col8:
    st.metric(
        "ATR diario",
        f"{atr_pct:.2f}%"
    )

col9, col10, col11, col12 = st.columns(4)

with col9:
    st.metric(
        "Objetivo 2R",
        f"${tp_2r:.2f}"
    )

with col10:
    st.metric(
        "Objetivo 3R",
        f"${tp_3r:.2f}"
    )

with col11:
    st.metric(
        "Confirmación 4H",
        f"{score_4h}/10"
    )

with col12:
    st.metric(
        "Confirmación 1H",
        f"{score_1h}/5"
    )


# ============================================================
# SCORE POR MÓDULO
# ============================================================

st.subheader(
    "🎯 Puntuación por módulo"
)

s1, s2, s3, s4 = st.columns(4)

with s1:
    st.metric(
        "Tendencia diaria",
        f"{score_tendencia}/25"
    )

with s2:
    st.metric(
        "MACD diario",
        f"{score_macd}/20"
    )

with s3:
    st.metric(
        "RSI diario",
        f"{score_rsi}/15"
    )

with s4:
    st.metric(
        "Volumen diario",
        f"{score_volumen}/15"
    )

s5, s6, s7, s8 = st.columns(4)

with s5:
    st.metric(
        "Acción del precio",
        f"{score_precio}/10"
    )

with s6:
    st.metric(
        "Confirmación 4H",
        f"{score_4h}/10"
    )

with s7:
    st.metric(
        "Confirmación 1H",
        f"{score_1h}/5"
    )

with s8:
    st.metric(
        "Total",
        f"{score_total}/100"
    )


# ============================================================
# VALIDACIÓN DIARIA
# ============================================================

st.subheader(
    "📋 Validación de la tesis diaria"
)

st.write(
    f"Precio ${precio_actual:.2f} > "
    f"EMA50 ${ema50_d:.2f}: "
    f"{'✅' if cond_precio_sobre_ema50 else '❌'}"
)

st.write(
    f"EMA10 ${ema10_d:.2f} > "
    f"EMA20 ${ema20_d:.2f}: "
    f"{'✅' if cond_ema10_sobre_ema20 else '❌'}"
)

st.write(
    f"EMA20 ${ema20_d:.2f} > "
    f"EMA50 ${ema50_d:.2f}: "
    f"{'✅' if cond_ema20_sobre_ema50 else '❌'}"
)

st.write(
    f"RSI diario {rsi_d:.2f} "
    f"en zona ideal 55–68: "
    f"{'✅' if cond_rsi_ideal else '❌'}"
)

st.write(
    f"MACD {macd_d:.4f} > "
    f"Signal {macd_signal_d:.4f}: "
    f"{'✅' if cond_macd_sobre_signal else '❌'}"
)

st.write(
    f"Histograma MACD diario positivo: "
    f"{'✅' if cond_histograma_positivo else '❌'}"
)

st.write(
    f"Histograma MACD creciente durante 3 sesiones: "
    f"{'✅' if cond_histograma_creciente else '❌'}"
)

st.write(
    f"Volumen relativo {volumen_relativo:.2f}x "
    f"> 1.5x: "
    f"{'✅' if cond_volumen_15x else '❌'}"
)

st.write(
    f"Breakout sobre máximo de 10 días "
    f"${high_10d:.2f}: "
    f"{'✅' if cond_breakout_10d else '❌'}"
)

st.write(
    f"Cierre en tercio superior de la vela: "
    f"{'✅' if cond_cierre_tercio_superior else '❌'}"
)


# ============================================================
# VALIDACIÓN 4H
# ============================================================

st.subheader(
    "🕓 Confirmación intradía 4H"
)

st.write(
    f"Precio 4H ${precio_4h:.2f} > "
    f"EMA20 4H ${ema20_4h:.2f}: "
    f"{'✅' if cond_precio_sobre_ema20_4h else '❌'}"
)

st.write(
    f"EMA10 4H ${ema10_4h:.2f} > "
    f"EMA20 4H ${ema20_4h:.2f}: "
    f"{'✅' if cond_ema10_sobre_ema20_4h else '❌'}"
)

st.write(
    f"Histograma MACD 4H "
    f"{macd_hist_4h:.4f} > 0: "
    f"{'✅' if cond_macd_hist_positivo_4h else '❌'}"
)


# ============================================================
# VALIDACIÓN 1H
# ============================================================

st.subheader(
    "🕐 Confirmación intradía 1H"
)

st.write(
    f"Precio 1H ${precio_1h:.2f} > "
    f"EMA20 1H ${ema20_1h:.2f}: "
    f"{'✅' if cond_precio_sobre_ema20_1h else '❌'}"
)

st.write(
    f"RSI 1H {rsi_1h:.2f} "
    f"entre 50 y 70: "
    f"{'✅' if cond_rsi_1h else '❌'}"
)

st.write(
    f"Histograma MACD 1H "
    f"{macd_hist_1h:.4f} > 0: "
    f"{'✅' if cond_macd_hist_positivo_1h else '❌'}"
)


# ============================================================
# TABLAS DE DATOS
# ============================================================

with st.expander(
    "Ver últimas velas diarias"
):
    st.dataframe(
        df_diario[
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
                "RSI14",
                "MACD",
                "MACD_SIGNAL",
                "MACD_HIST",
                "VOL_REL",
                "ATR14",
                "ATR_PCT"
            ]
        ].tail(30),
        use_container_width=True
    )

with st.expander(
    "Ver últimas velas 4H"
):
    st.dataframe(
        df_4h[
            [
                "Datetime",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "EMA10",
                "EMA20",
                "RSI14",
                "MACD",
                "MACD_SIGNAL",
                "MACD_HIST"
            ]
        ].tail(30),
        use_container_width=True
    )

with st.expander(
    "Ver últimas velas 1H"
):
    columna_fecha_1h = (
        "Datetime"
        if "Datetime" in df_1h.columns
        else "Date"
    )

    st.dataframe(
        df_1h[
            [
                columna_fecha_1h,
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "EMA10",
                "EMA20",
                "RSI14",
                "MACD",
                "MACD_SIGNAL",
                "MACD_HIST"
            ]
        ].tail(30),
        use_container_width=True
    )


# ============================================================
# AVISO
# ============================================================

st.divider()

st.caption(
    "Herramienta educativa de análisis técnico. "
    "La confirmación intradía de yfinance puede tener retraso "
    "y no debe considerarse una señal de ejecución en tiempo real."
)
