import time

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Swing Trading Scanner",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Swing Trading Scanner")
st.caption(
    "Scanner para operaciones de 3 a 10 días con análisis diario "
    "y confirmaciones intradía de 4H y 1H."
)


# ============================================================
# CONFIGURACIÓN DE LA ESTRATEGIA
# ============================================================

PESO_TENDENCIA = 25
PESO_MACD = 20
PESO_RSI = 15
PESO_VOLUMEN = 15
PESO_PRECIO = 10
PESO_4H = 10
PESO_1H = 5

TICKERS_PREDETERMINADOS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMD",
    "META",
    "AMZN",
    "GOOGL",
    "TSLA",
    "NFLX",
    "AVGO",
    "PLTR",
    "COIN",
    "CRM",
    "ORCL",
    "QCOM",
    "MU"
]


# ============================================================
# FUNCIONES DE DATOS
# ============================================================

def limpiar_columnas_yfinance(df):
    """
    Corrige las columnas MultiIndex que puede devolver yfinance.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


@st.cache_data(ttl=900, show_spinner=False)
def descargar_datos(ticker):
    """
    Descarga datos diarios y datos de una hora.
    El caché dura 15 minutos.
    """

    diario = yf.download(
        ticker,
        period="1y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False
    )

    una_hora = yf.download(
        ticker,
        period="60d",
        interval="1h",
        auto_adjust=True,
        prepost=False,
        progress=False,
        threads=False
    )

    diario = limpiar_columnas_yfinance(diario)
    una_hora = limpiar_columnas_yfinance(una_hora)

    if not diario.empty:
        diario = diario.reset_index()

    if not una_hora.empty:
        una_hora = una_hora.reset_index()

    return diario, una_hora


# ============================================================
# FUNCIONES DE INDICADORES
# ============================================================

def calcular_rsi(close, periodo=14):
    """
    Calcula RSI utilizando suavizado exponencial tipo Wilder.
    """

    delta = close.diff()

    ganancias = delta.clip(lower=0)
    perdidas = -delta.clip(upper=0)

    promedio_ganancias = ganancias.ewm(
        alpha=1 / periodo,
        adjust=False
    ).mean()

    promedio_perdidas = perdidas.ewm(
        alpha=1 / periodo,
        adjust=False
    ).mean()

    rs = promedio_ganancias / promedio_perdidas.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)


def calcular_indicadores(df):
    """
    Calcula todos los indicadores utilizados por la estrategia.
    """

    if df.empty:
        return pd.DataFrame()

    df = df.copy()

    columnas_requeridas = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for columna in columnas_requeridas:
        df[columna] = pd.to_numeric(
            df[columna],
            errors="coerce"
        )

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
    df["VOL_AVG20"] = df["Volume"].rolling(20).mean()

    df["VOL_REL"] = (
        df["Volume"] / df["VOL_AVG20"]
    )

    # Máximos y mínimos previos de 10 velas
    df["HIGH_10"] = (
        df["High"]
        .rolling(10)
        .max()
        .shift(1)
    )

    df["LOW_10"] = (
        df["Low"]
        .rolling(10)
        .min()
        .shift(1)
    )

    # True Range y ATR14
    rango_max_min = df["High"] - df["Low"]

    max_cierre_anterior = (
        df["High"] - df["Close"].shift(1)
    ).abs()

    min_cierre_anterior = (
        df["Low"] - df["Close"].shift(1)
    ).abs()

    df["TRUE_RANGE"] = pd.concat(
        [
            rango_max_min,
            max_cierre_anterior,
            min_cierre_anterior
        ],
        axis=1
    ).max(axis=1)

    df["ATR14"] = df["TRUE_RANGE"].rolling(14).mean()

    df["ATR_PCT"] = (
        df["ATR14"] / df["Close"]
    ) * 100

    return df


def convertir_1h_a_4h(df_1h):
    """
    Agrupa las velas de una hora en bloques de aproximadamente
    cuatro horas dentro de cada sesión bursátil.
    """

    if df_1h.empty:
        return pd.DataFrame()

    df = df_1h.copy()

    if "Datetime" in df.columns:
        columna_fecha = "Datetime"

    elif "Date" in df.columns:
        columna_fecha = "Date"

    else:
        return pd.DataFrame()

    df[columna_fecha] = pd.to_datetime(
        df[columna_fecha],
        errors="coerce"
    )

    df = df.dropna(
        subset=[columna_fecha]
    )

    df["SESSION_DATE"] = df[columna_fecha].dt.date

    df["ORDEN_SESION"] = (
        df.groupby("SESSION_DATE")
        .cumcount()
    )

    df["BLOQUE_4H"] = (
        df["ORDEN_SESION"] // 4
    )

    df_4h = (
        df.groupby(
            [
                "SESSION_DATE",
                "BLOQUE_4H"
            ],
            as_index=False
        )
        .agg(
            Datetime=(columna_fecha, "first"),
            Open=("Open", "first"),
            High=("High", "max"),
            Low=("Low", "min"),
            Close=("Close", "last"),
            Volume=("Volume", "sum")
        )
    )

    return df_4h


# ============================================================
# FUNCIONES DE PUNTUACIÓN
# ============================================================

def analizar_ticker(ticker):
    """
    Analiza un ticker y devuelve una fila con sus resultados.
    """

    df_diario, df_1h = descargar_datos(ticker)

    if df_diario.empty:
        raise ValueError("Sin datos diarios")

    if df_1h.empty:
        raise ValueError("Sin datos de 1H")

    df_4h = convertir_1h_a_4h(df_1h)

    if df_4h.empty:
        raise ValueError("No fue posible calcular las velas 4H")

    df_diario = calcular_indicadores(df_diario)
    df_1h = calcular_indicadores(df_1h)
    df_4h = calcular_indicadores(df_4h)

    df_diario = df_diario.dropna().reset_index(drop=True)
    df_1h = df_1h.dropna().reset_index(drop=True)
    df_4h = df_4h.dropna().reset_index(drop=True)

    if len(df_diario) < 60:
        raise ValueError("Datos diarios insuficientes")

    if len(df_1h) < 50:
        raise ValueError("Datos 1H insuficientes")

    if len(df_4h) < 50:
        raise ValueError("Datos 4H insuficientes")

    actual_d = df_diario.iloc[-1]
    anterior_d = df_diario.iloc[-2]
    anterior_2d = df_diario.iloc[-3]

    actual_1h = df_1h.iloc[-1]
    actual_4h = df_4h.iloc[-1]

    # --------------------------------------------------------
    # VALORES DIARIOS
    # --------------------------------------------------------

    precio = float(actual_d["Close"])

    ema10_d = float(actual_d["EMA10"])
    ema20_d = float(actual_d["EMA20"])
    ema50_d = float(actual_d["EMA50"])

    rsi_d = float(actual_d["RSI14"])

    macd_d = float(actual_d["MACD"])
    signal_d = float(actual_d["MACD_SIGNAL"])
    hist_d = float(actual_d["MACD_HIST"])

    hist_d_anterior = float(
        anterior_d["MACD_HIST"]
    )

    hist_d_anterior_2 = float(
        anterior_2d["MACD_HIST"]
    )

    volumen = float(actual_d["Volume"])
    volumen_anterior = float(anterior_d["Volume"])
    volumen_promedio = float(actual_d["VOL_AVG20"])
    volumen_relativo = float(actual_d["VOL_REL"])

    high_10 = float(actual_d["HIGH_10"])
    low_10 = float(actual_d["LOW_10"])

    atr14 = float(actual_d["ATR14"])
    atr_pct = float(actual_d["ATR_PCT"])

    # --------------------------------------------------------
    # SCORE DE TENDENCIA — 25
    # --------------------------------------------------------

    score_tendencia = 0

    cond_precio_ema50 = precio > ema50_d
    cond_ema10_ema20 = ema10_d > ema20_d
    cond_ema20_ema50 = ema20_d > ema50_d

    if cond_precio_ema50:
        score_tendencia += 9

    if cond_ema10_ema20:
        score_tendencia += 8

    if cond_ema20_ema50:
        score_tendencia += 8

    # --------------------------------------------------------
    # SCORE MACD — 20
    # --------------------------------------------------------

    score_macd = 0

    cond_macd_signal = macd_d > signal_d
    cond_hist_positivo = hist_d > 0

    cond_hist_creciente = (
        hist_d
        > hist_d_anterior
        > hist_d_anterior_2
    )

    if cond_macd_signal:
        score_macd += 8

    if cond_hist_positivo:
        score_macd += 8

    if cond_hist_creciente:
        score_macd += 4

    # --------------------------------------------------------
    # SCORE RSI — 15
    # --------------------------------------------------------

    score_rsi = 0

    if 55 <= rsi_d <= 68:
        score_rsi = 15

    elif (
        50 <= rsi_d < 55
        or
        68 < rsi_d <= 75
    ):
        score_rsi = 8

    # --------------------------------------------------------
    # SCORE VOLUMEN — 15
    # --------------------------------------------------------

    score_volumen = 0

    cond_volumen_promedio = (
        volumen > volumen_promedio
    )

    cond_volumen_15 = (
        volumen > 1.5 * volumen_promedio
    )

    cond_vela_volumen = (
        actual_d["Close"] > actual_d["Open"]
        and
        volumen > volumen_anterior
    )

    if cond_volumen_promedio:
        score_volumen += 5

    if cond_volumen_15:
        score_volumen += 5

    if cond_vela_volumen:
        score_volumen += 5

    # --------------------------------------------------------
    # SCORE ACCIÓN DEL PRECIO — 10
    # --------------------------------------------------------

    score_precio = 0

    cond_breakout = precio > high_10

    rango_diario = float(
        actual_d["High"] - actual_d["Low"]
    )

    if rango_diario > 0:
        posicion_cierre = (
            float(
                actual_d["Close"]
                - actual_d["Low"]
            )
            / rango_diario
        )
    else:
        posicion_cierre = 0

    cond_cierre_superior = (
        posicion_cierre >= 0.66
    )

    if cond_breakout:
        score_precio += 5

    if cond_cierre_superior:
        score_precio += 5

    # --------------------------------------------------------
    # CONFIRMACIÓN 4H — 10
    # --------------------------------------------------------

    precio_4h = float(actual_4h["Close"])
    ema10_4h = float(actual_4h["EMA10"])
    ema20_4h = float(actual_4h["EMA20"])
    hist_4h = float(actual_4h["MACD_HIST"])
    rsi_4h = float(actual_4h["RSI14"])

    score_4h = 0

    cond_precio_ema20_4h = (
        precio_4h > ema20_4h
    )

    cond_ema10_ema20_4h = (
        ema10_4h > ema20_4h
    )

    cond_hist_positivo_4h = (
        hist_4h > 0
    )

    if cond_precio_ema20_4h:
        score_4h += 4

    if cond_ema10_ema20_4h:
        score_4h += 3

    if cond_hist_positivo_4h:
        score_4h += 3

    # --------------------------------------------------------
    # CONFIRMACIÓN 1H — 5
    # --------------------------------------------------------

    precio_1h = float(actual_1h["Close"])
    ema20_1h = float(actual_1h["EMA20"])
    rsi_1h = float(actual_1h["RSI14"])
    hist_1h = float(actual_1h["MACD_HIST"])

    score_1h = 0

    cond_precio_ema20_1h = (
        precio_1h > ema20_1h
    )

    cond_rsi_1h = (
        50 <= rsi_1h <= 70
    )

    cond_hist_positivo_1h = (
        hist_1h > 0
    )

    if cond_precio_ema20_1h:
        score_1h += 2

    if cond_rsi_1h:
        score_1h += 2

    if cond_hist_positivo_1h:
        score_1h += 1

    # --------------------------------------------------------
    # SCORE TOTAL
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # ENTRADA, STOP LOSS Y OBJETIVOS
    # --------------------------------------------------------

    entrada = max(
        precio,
        high_10
    )

    stop_atr = (
        entrada - 2 * atr14
    )

    stop_5pct = (
        entrada * 0.95
    )

    stops_validos = [
        stop_atr,
        stop_5pct
    ]

    if (
        low_10 < entrada
        and
        low_10 >= entrada * 0.90
    ):
        stops_validos.append(low_10)

    stop_loss = max(stops_validos)

    riesgo_unitario = (
        entrada - stop_loss
    )

    if riesgo_unitario > 0:
        riesgo_pct = (
            riesgo_unitario / entrada
        ) * 100

        objetivo_2r = (
            entrada
            + 2 * riesgo_unitario
        )

        objetivo_3r = (
            entrada
            + 3 * riesgo_unitario
        )

    else:
        riesgo_pct = np.nan
        objetivo_2r = np.nan
        objetivo_3r = np.nan

    # --------------------------------------------------------
    # DECISIÓN
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # NOTAS
    # --------------------------------------------------------

    notas = []

    if not cond_precio_ema50:
        notas.append("Precio bajo EMA50")

    if not cond_ema10_ema20:
        notas.append("EMA10 bajo EMA20")

    if not cond_macd_signal:
        notas.append("MACD bajo Signal")

    if not cond_hist_positivo:
        notas.append("Histograma diario negativo")

    if volumen_relativo < 1:
        notas.append("Volumen bajo")

    if not cond_breakout:
        notas.append("Sin breakout 10D")

    if score_4h < 7:
        notas.append("Confirmación 4H débil")

    if score_1h < 3:
        notas.append("Confirmación 1H débil")

    if not notas:
        texto_notas = "Setup completo"

    else:
        texto_notas = "; ".join(
            notas[:4]
        )

    return {
        "Ticker": ticker,
        "Decisión": decision,
        "Score total": int(score_total),
        "Score diario": int(score_diario),
        "Score 4H": int(score_4h),
        "Score 1H": int(score_1h),
        "Precio": round(precio, 2),
        "Entrada": round(entrada, 2),
        "Stop loss": round(stop_loss, 2),
        "Riesgo %": round(riesgo_pct, 2),
        "TP 2R": round(objetivo_2r, 2),
        "TP 3R": round(objetivo_3r, 2),
        "EMA10 D": round(ema10_d, 2),
        "EMA20 D": round(ema20_d, 2),
        "EMA50 D": round(ema50_d, 2),
        "RSI D": round(rsi_d, 2),
        "MACD D": round(macd_d, 4),
        "Signal D": round(signal_d, 4),
        "Hist MACD D": round(hist_d, 4),
        "Vol/Avg20": round(volumen_relativo, 2),
        "ATR14": round(atr14, 2),
        "ATR %": round(atr_pct, 2),
        "RSI 4H": round(rsi_4h, 2),
        "Hist MACD 4H": round(hist_4h, 4),
        "RSI 1H": round(rsi_1h, 2),
        "Hist MACD 1H": round(hist_1h, 4),
        "Notas": texto_notas
    }


# ============================================================
# INTERFAZ DEL SCANNER
# ============================================================

with st.sidebar:

    st.header("Configuración")

    tickers_texto = st.text_area(
        "Tickers separados por coma",
        value=", ".join(TICKERS_PREDETERMINADOS),
        height=180
    )

    score_minimo = st.slider(
        "Score mínimo para candidatos",
        min_value=0,
        max_value=100,
        value=70
    )

    mostrar_no_compra = st.checkbox(
        "Mostrar acciones con score bajo",
        value=True
    )

    pausa_llamadas = st.number_input(
        "Pausa entre tickers",
        min_value=0.0,
        max_value=5.0,
        value=0.3,
        step=0.1,
        help="Ayuda a evitar bloqueos temporales de Yahoo Finance."
    )

    ejecutar = st.button(
        "🔍 Ejecutar scanner",
        type="primary",
        use_container_width=True
    )


# ============================================================
# FILA DE CONDICIONES IDEALES
# ============================================================

fila_ideal = {
    "Rank": "IDEAL",
    "Ticker": "—",
    "Decisión": "COMPRAR",
    "Score total": "90+",
    "Score diario": "80+",
    "Score 4H": "8–10",
    "Score 1H": "4–5",
    "Precio": "—",
    "Entrada": "Breakout 10D",
    "Stop loss": "2 ATR / swing",
    "Riesgo %": "3–5%",
    "TP 2R": "2× riesgo",
    "TP 3R": "3× riesgo",
    "EMA10 D": "> EMA20",
    "EMA20 D": "> EMA50",
    "EMA50 D": "Precio > EMA50",
    "RSI D": "55–68",
    "MACD D": "> Signal",
    "Signal D": "< MACD",
    "Hist MACD D": ">0 creciente",
    "Vol/Avg20": ">1.5x",
    "ATR14": "—",
    "ATR %": "2–5%",
    "RSI 4H": "50–70",
    "Hist MACD 4H": ">0",
    "RSI 1H": "50–70",
    "Hist MACD 1H": ">0",
    "Notas": "Condiciones ideales"
}


# ============================================================
# EJECUCIÓN
# ============================================================

if ejecutar:

    tickers = [
        ticker.strip().upper()
        for ticker in tickers_texto.split(",")
        if ticker.strip()
    ]

    tickers = list(dict.fromkeys(tickers))

    if not tickers:
        st.warning(
            "Agrega al menos un ticker."
        )
        st.stop()

    resultados = []
    errores = []

    barra = st.progress(0)
    mensaje = st.empty()

    total_tickers = len(tickers)

    for indice, ticker in enumerate(tickers):

        mensaje.write(
            f"Analizando {ticker} "
            f"({indice + 1}/{total_tickers})..."
        )

        try:
            resultado = analizar_ticker(ticker)

            resultados.append(resultado)

        except Exception as error:
            errores.append(
                {
                    "Ticker": ticker,
                    "Error": str(error)
                }
            )

        barra.progress(
            (indice + 1) / total_tickers
        )

        if pausa_llamadas > 0:
            time.sleep(pausa_llamadas)

    mensaje.success(
        "Scanner terminado."
    )

    if not resultados:
        st.error(
            "No fue posible analizar ningún ticker."
        )

        if errores:
            st.dataframe(
                pd.DataFrame(errores),
                use_container_width=True
            )

        st.stop()

    df_resultados = pd.DataFrame(
        resultados
    )

    df_resultados = df_resultados.sort_values(
        by=[
            "Score total",
            "Score 4H",
            "Score 1H",
            "Vol/Avg20"
        ],
        ascending=[
            False,
            False,
            False,
            False
        ]
    ).reset_index(drop=True)

    df_resultados.insert(
        0,
        "Rank",
        range(1, len(df_resultados) + 1)
    )

    if not mostrar_no_compra:
        df_resultados = df_resultados[
            df_resultados["Score total"]
            >= score_minimo
        ].copy()

    candidatos = df_resultados[
        df_resultados["Score total"]
        >= score_minimo
    ].copy()

    # --------------------------------------------------------
    # MÉTRICAS GENERALES
    # --------------------------------------------------------

    st.subheader(
        "Resumen del scanner"
    )

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "Acciones analizadas",
            len(resultados)
        )

    with m2:
        st.metric(
            "Candidatos",
            len(candidatos)
        )

    with m3:
        compras = df_resultados[
            df_resultados["Decisión"].isin(
                [
                    "COMPRA AGRESIVA",
                    "COMPRAR"
                ]
            )
        ]

        st.metric(
            "Señales de compra",
            len(compras)
        )

    with m4:
        mejor_score = (
            int(df_resultados["Score total"].max())
            if not df_resultados.empty
            else 0
        )

        st.metric(
            "Mejor score",
            f"{mejor_score}/100"
        )

    # --------------------------------------------------------
    # TABLA DE CANDIDATOS
    # --------------------------------------------------------

    st.subheader(
        "✅ Candidatos por score"
    )

    if candidatos.empty:
        st.warning(
            "No hay acciones que superen "
            "el score mínimo seleccionado."
        )

    else:
        tabla_candidatos = pd.concat(
            [
                pd.DataFrame([fila_ideal]),
                candidatos
            ],
            ignore_index=True
        )

        st.dataframe(
            tabla_candidatos,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # TABLA COMPLETA
    # --------------------------------------------------------

    st.subheader(
        "📋 Ranking completo"
    )

    tabla_completa = pd.concat(
        [
            pd.DataFrame([fila_ideal]),
            df_resultados
        ],
        ignore_index=True
    )

    st.dataframe(
        tabla_completa,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # DESCARGA CSV
    # --------------------------------------------------------

    csv = df_resultados.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Descargar resultados CSV",
        data=csv,
        file_name="swing_scanner_resultados.csv",
        mime="text/csv"
    )

    # --------------------------------------------------------
    # ERRORES
    # --------------------------------------------------------

    if errores:

        with st.expander(
            "Ver tickers con errores"
        ):
            st.dataframe(
                pd.DataFrame(errores),
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# INFORMACIÓN DE LA ESTRATEGIA
# ============================================================

else:

    st.info(
        "Configura la lista de tickers y presiona "
        "“Ejecutar scanner”."
    )

    st.subheader(
        "Sistema de puntuación"
    )

    tabla_pesos = pd.DataFrame(
        {
            "Módulo": [
                "Tendencia diaria",
                "MACD diario",
                "RSI diario",
                "Volumen diario",
                "Acción del precio",
                "Confirmación 4H",
                "Confirmación 1H"
            ],
            "Puntos": [
                PESO_TENDENCIA,
                PESO_MACD,
                PESO_RSI,
                PESO_VOLUMEN,
                PESO_PRECIO,
                PESO_4H,
                PESO_1H
            ]
        }
    )

    st.dataframe(
        tabla_pesos,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# AVISO
# ============================================================

st.divider()

st.caption(
    "Herramienta educativa de análisis técnico. "
    "Los datos de Yahoo Finance pueden tener retraso. "
    "Una puntuación alta no garantiza beneficios. "
    "Revisa noticias, resultados trimestrales, liquidez "
    "y riesgo antes de ejecutar una operación."
)
