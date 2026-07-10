import io
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Nasdaq + S&P 500 Swing Scanner",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Nasdaq + S&P 500 Swing Scanner")

st.caption(
    "Scanner para swing trading de 3 a 10 días. "
    "Utiliza análisis diario y confirmaciones intradía 4H y 1H."
)


# ============================================================
# FUENTES DEL UNIVERSO
# ============================================================

NASDAQ_SYMBOL_URL = (
    "https://ftp.nasdaqtrader.com/"
    "SymbolDirectory/nasdaqlisted.txt"
)

SP500_URL = (
    "https://en.wikipedia.org/wiki/"
    "List_of_S%26P_500_companies"
)


# ============================================================
# PONDERACIONES
# ============================================================

PESO_TENDENCIA = 25
PESO_MACD = 20
PESO_RSI = 15
PESO_VOLUMEN = 15
PESO_PRECIO = 10
PESO_4H = 10
PESO_1H = 5


# ============================================================
# FUNCIONES GENERALES
# ============================================================

def dividir_lista(lista, tamano):
    """
    Divide una lista en grupos más pequeños.
    """

    for posicion in range(0, len(lista), tamano):
        yield lista[posicion:posicion + tamano]


def normalizar_ticker_yahoo(ticker):
    """
    Yahoo Finance utiliza guion para ciertas clases de acciones.

    Ejemplo:
    BRK.B -> BRK-B
    """

    return str(ticker).strip().upper().replace(".", "-")


def limpiar_columnas_yfinance(df):
    """
    Corrige columnas y convierte OHLCV a valores numéricos.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    columnas_numericas = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for columna in columnas_numericas:
        if columna in df.columns:
            df[columna] = pd.to_numeric(
                df[columna],
                errors="coerce"
            )

    return df


# ============================================================
# UNIVERSO NASDAQ
# ============================================================

@st.cache_data(ttl=21600, show_spinner=False)
def obtener_nasdaq():
    """
    Obtiene el directorio oficial de valores listados en Nasdaq.
    Excluye ETF y símbolos de prueba.
    """

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        NASDAQ_SYMBOL_URL,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    contenido = response.text

    nasdaq = pd.read_csv(
        io.StringIO(contenido),
        sep="|"
    )

    nasdaq["Symbol"] = (
        nasdaq["Symbol"]
        .astype(str)
        .str.strip()
    )

    # Eliminar línea final del archivo
    nasdaq = nasdaq[
        ~nasdaq["Symbol"].str.contains(
            "File Creation Time",
            na=False
        )
    ].copy()

    # Excluir ETF
    if "ETF" in nasdaq.columns:
        nasdaq = nasdaq[
            nasdaq["ETF"].astype(str).str.upper() == "N"
        ].copy()

    # Excluir símbolos de prueba
    if "Test Issue" in nasdaq.columns:
        nasdaq = nasdaq[
            nasdaq["Test Issue"].astype(str).str.upper() == "N"
        ].copy()

    # Excluir símbolos incompatibles o especiales
    nasdaq = nasdaq[
        nasdaq["Symbol"].str.match(
            r"^[A-Z]{1,5}$",
            na=False
        )
    ].copy()

    resultado = pd.DataFrame({
        "Ticker original": nasdaq["Symbol"],
        "Ticker": nasdaq["Symbol"].apply(
            normalizar_ticker_yahoo
        ),
        "Empresa": nasdaq["Security Name"],
        "Origen": "Nasdaq"
    })

    return resultado.drop_duplicates(
        subset=["Ticker"]
    ).reset_index(drop=True)


# ============================================================
# UNIVERSO S&P 500
# ============================================================

@st.cache_data(ttl=21600, show_spinner=False)
def obtener_sp500():
    """
    Obtiene la tabla pública de constituyentes del S&P 500.
    """

    tablas = pd.read_html(
        SP500_URL
    )

    sp500 = tablas[0].copy()

    resultado = pd.DataFrame({
        "Ticker original": sp500["Symbol"],
        "Ticker": sp500["Symbol"].apply(
            normalizar_ticker_yahoo
        ),
        "Empresa": sp500["Security"],
        "Origen": "S&P 500"
    })

    return resultado.drop_duplicates(
        subset=["Ticker"]
    ).reset_index(drop=True)


# ============================================================
# UNIVERSO COMBINADO
# ============================================================

@st.cache_data(ttl=21600, show_spinner=False)
def obtener_universo():
    """
    Combina Nasdaq y S&P 500 y conserva el origen de cada ticker.
    """

    nasdaq = obtener_nasdaq()
    sp500 = obtener_sp500()

    universo = pd.concat(
        [nasdaq, sp500],
        ignore_index=True
    )

    universo = (
        universo.groupby(
            "Ticker",
            as_index=False
        )
        .agg({
            "Ticker original": "first",
            "Empresa": "first",
            "Origen": lambda valores: " + ".join(
                sorted(set(valores))
            )
        })
    )

    universo = universo.sort_values(
        "Ticker"
    ).reset_index(drop=True)

    return universo


# ============================================================
# INDICADORES
# ============================================================

def calcular_rsi(close, periodo=14):
    """
    RSI con suavizado exponencial tipo Wilder.
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

    rs = (
        promedio_ganancias
        / promedio_perdidas.replace(0, np.nan)
    )

    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)


def calcular_indicadores(df):
    """
    Calcula todos los indicadores de la tesis.
    """

    if df.empty:
        return pd.DataFrame()

    df = limpiar_columnas_yfinance(df)

    df = df.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]
    ).copy()

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

    # Volumen
    df["VOL_AVG20"] = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    df["VOL_REL"] = (
        df["Volume"]
        / df["VOL_AVG20"]
    )

    df["DOLLAR_VOL_AVG20"] = (
        df["Close"] * df["Volume"]
    ).rolling(20).mean()

    # Máximos y mínimos previos
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

    # ATR
    rango_max_min = (
        df["High"] - df["Low"]
    )

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

    df["ATR14"] = (
        df["TRUE_RANGE"]
        .rolling(14)
        .mean()
    )

    df["ATR_PCT"] = (
        df["ATR14"]
        / df["Close"]
    ) * 100

    return df


# ============================================================
# EXTRAER TICKER DE DESCARGA MÚLTIPLE
# ============================================================

def extraer_ticker_del_lote(datos_lote, ticker, total_tickers):
    """
    Extrae el DataFrame correspondiente a un ticker.
    """

    if datos_lote is None or datos_lote.empty:
        return pd.DataFrame()

    try:

        if isinstance(
            datos_lote.columns,
            pd.MultiIndex
        ):

            nivel_cero = list(
                datos_lote.columns.get_level_values(0)
            )

            nivel_uno = list(
                datos_lote.columns.get_level_values(1)
            )

            if ticker in nivel_cero:
                df = datos_lote[ticker].copy()

            elif ticker in nivel_uno:
                df = datos_lote.xs(
                    ticker,
                    axis=1,
                    level=1
                ).copy()

            else:
                return pd.DataFrame()

        elif total_tickers == 1:
            df = datos_lote.copy()

        else:
            return pd.DataFrame()

        df = limpiar_columnas_yfinance(df)

        return df.dropna(
            how="all"
        )

    except Exception:
        return pd.DataFrame()


# ============================================================
# DESCARGA DIARIA EN LOTES
# ============================================================

@st.cache_data(ttl=21600, show_spinner=False)
def descargar_lote_diario(tickers):
    """
    Descarga un lote de tickers con velas diarias.
    """

    tickers = list(tickers)

    datos = yf.download(
        tickers=tickers,
        period="1y",
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
        timeout=30
    )

    return datos


# ============================================================
# SCORE DIARIO
# ============================================================

def analizar_diario(
    ticker,
    empresa,
    origen,
    df,
    precio_minimo,
    volumen_minimo,
    dollar_volume_minimo
):
    """
    Aplica filtros de liquidez y calcula score diario /85.
    """

    if df.empty or len(df) < 100:
        return None

    df = calcular_indicadores(df)

    df = df.dropna().copy()

    if len(df) < 60:
        return None

    actual = df.iloc[-1]
    anterior = df.iloc[-2]
    anterior_2 = df.iloc[-3]

    precio = float(actual["Close"])
    volumen_promedio = float(actual["VOL_AVG20"])
    dollar_volume = float(
        actual["DOLLAR_VOL_AVG20"]
    )

    # Filtros iniciales
    if precio < precio_minimo:
        return None

    if volumen_promedio < volumen_minimo:
        return None

    if dollar_volume < dollar_volume_minimo:
        return None

    ema10 = float(actual["EMA10"])
    ema20 = float(actual["EMA20"])
    ema50 = float(actual["EMA50"])

    rsi = float(actual["RSI14"])

    macd = float(actual["MACD"])
    signal = float(actual["MACD_SIGNAL"])
    hist = float(actual["MACD_HIST"])

    hist_anterior = float(
        anterior["MACD_HIST"]
    )

    hist_anterior_2 = float(
        anterior_2["MACD_HIST"]
    )

    volumen_actual = float(actual["Volume"])
    volumen_anterior = float(anterior["Volume"])
    volumen_relativo = float(actual["VOL_REL"])

    high_10 = float(actual["HIGH_10"])
    low_10 = float(actual["LOW_10"])

    atr14 = float(actual["ATR14"])
    atr_pct = float(actual["ATR_PCT"])

    # --------------------------------------------------------
    # TENDENCIA — 25
    # --------------------------------------------------------

    score_tendencia = 0

    if precio > ema50:
        score_tendencia += 9

    if ema10 > ema20:
        score_tendencia += 8

    if ema20 > ema50:
        score_tendencia += 8

    # --------------------------------------------------------
    # MACD — 20
    # --------------------------------------------------------

    score_macd = 0

    if macd > signal:
        score_macd += 8

    if hist > 0:
        score_macd += 8

    if hist > hist_anterior > hist_anterior_2:
        score_macd += 4

    # --------------------------------------------------------
    # RSI — 15
    # --------------------------------------------------------

    score_rsi = 0

    if 55 <= rsi <= 68:
        score_rsi = 15

    elif (
        50 <= rsi < 55
        or
        68 < rsi <= 75
    ):
        score_rsi = 8

    # --------------------------------------------------------
    # VOLUMEN — 15
    # --------------------------------------------------------

    score_volumen = 0

    if volumen_actual > volumen_promedio:
        score_volumen += 5

    if volumen_actual > 1.5 * volumen_promedio:
        score_volumen += 5

    if (
        actual["Close"] > actual["Open"]
        and
        volumen_actual > volumen_anterior
    ):
        score_volumen += 5

    # --------------------------------------------------------
    # ACCIÓN DEL PRECIO — 10
    # --------------------------------------------------------

    score_precio = 0

    breakout = precio > high_10

    rango_diario = float(
        actual["High"] - actual["Low"]
    )

    if rango_diario > 0:
        posicion_cierre = (
            float(actual["Close"] - actual["Low"])
            / rango_diario
        )
    else:
        posicion_cierre = 0

    if breakout:
        score_precio += 5

    if posicion_cierre >= 0.66:
        score_precio += 5

    score_diario = (
        score_tendencia
        + score_macd
        + score_rsi
        + score_volumen
        + score_precio
    )

    return {
        "Ticker": ticker,
        "Empresa": empresa,
        "Origen": origen,
        "Score diario": int(score_diario),
        "Score tendencia": int(score_tendencia),
        "Score MACD": int(score_macd),
        "Score RSI": int(score_rsi),
        "Score volumen": int(score_volumen),
        "Score precio": int(score_precio),
        "Precio": precio,
        "EMA10 D": ema10,
        "EMA20 D": ema20,
        "EMA50 D": ema50,
        "RSI D": rsi,
        "MACD D": macd,
        "Signal D": signal,
        "Hist MACD D": hist,
        "Vol/Avg20": volumen_relativo,
        "Volumen promedio": volumen_promedio,
        "Dollar Volume": dollar_volume,
        "ATR14": atr14,
        "ATR %": atr_pct,
        "High 10D": high_10,
        "Low 10D": low_10
    }


# ============================================================
# CONVERTIR 1H A 4H
# ============================================================

def convertir_1h_a_4h(df_1h):
    """
    Construye velas 4H agrupando barras dentro de cada sesión.
    """

    if df_1h.empty:
        return pd.DataFrame()

    df = df_1h.copy()

    df.index = pd.to_datetime(
        df.index,
        errors="coerce"
    )

    df = df[
        ~df.index.isna()
    ].copy()

    df["SESSION_DATE"] = df.index.date

    df["ORDEN"] = (
        df.groupby("SESSION_DATE")
        .cumcount()
    )

    df["BLOQUE_4H"] = (
        df["ORDEN"] // 4
    )

    df_4h = (
        df.groupby(
            ["SESSION_DATE", "BLOQUE_4H"]
        )
        .agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        })
    )

    df_4h = df_4h.reset_index(
        drop=True
    )

    return df_4h


# ============================================================
# DESCARGA INTRADÍA
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def descargar_intradia(ticker):
    """
    Descarga 60 días de velas 1H.
    """

    df = yf.download(
        ticker,
        period="60d",
        interval="1h",
        auto_adjust=True,
        prepost=False,
        progress=False,
        threads=False,
        timeout=30
    )

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return limpiar_columnas_yfinance(df)


# ============================================================
# CONFIRMACIONES INTRADÍA
# ============================================================

def agregar_confirmacion_intradia(fila):
    """
    Agrega score 4H, score 1H, entrada, stop y objetivos.
    """

    ticker = fila["Ticker"]

    df_1h = descargar_intradia(ticker)

    if df_1h.empty:
        fila["Score 4H"] = 0
        fila["Score 1H"] = 0
        fila["Score total"] = fila["Score diario"]
        fila["Estado intradía"] = "Sin datos"
        return fila

    df_4h = convertir_1h_a_4h(
        df_1h
    )

    df_1h = calcular_indicadores(
        df_1h
    ).dropna()

    df_4h = calcular_indicadores(
        df_4h
    ).dropna()

    if len(df_1h) < 50 or len(df_4h) < 50:
        fila["Score 4H"] = 0
        fila["Score 1H"] = 0
        fila["Score total"] = fila["Score diario"]
        fila["Estado intradía"] = "Datos insuficientes"
        return fila

    actual_1h = df_1h.iloc[-1]
    actual_4h = df_4h.iloc[-1]

    # --------------------------------------------------------
    # SCORE 4H — 10
    # --------------------------------------------------------

    score_4h = 0

    precio_4h = float(actual_4h["Close"])
    ema10_4h = float(actual_4h["EMA10"])
    ema20_4h = float(actual_4h["EMA20"])
    rsi_4h = float(actual_4h["RSI14"])
    hist_4h = float(actual_4h["MACD_HIST"])

    if precio_4h > ema20_4h:
        score_4h += 4

    if ema10_4h > ema20_4h:
        score_4h += 3

    if hist_4h > 0:
        score_4h += 3

    # --------------------------------------------------------
    # SCORE 1H — 5
    # --------------------------------------------------------

    score_1h = 0

    precio_1h = float(actual_1h["Close"])
    ema20_1h = float(actual_1h["EMA20"])
    rsi_1h = float(actual_1h["RSI14"])
    hist_1h = float(actual_1h["MACD_HIST"])

    if precio_1h > ema20_1h:
        score_1h += 2

    if 50 <= rsi_1h <= 70:
        score_1h += 2

    if hist_1h > 0:
        score_1h += 1

    score_total = (
        int(fila["Score diario"])
        + score_4h
        + score_1h
    )

    # --------------------------------------------------------
    # ENTRADA Y STOP
    # --------------------------------------------------------

    precio = float(fila["Precio"])
    high_10 = float(fila["High 10D"])
    low_10 = float(fila["Low 10D"])
    atr14 = float(fila["ATR14"])

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

    candidatos_stop = [
        stop_atr,
        stop_5pct
    ]

    if (
        low_10 < entrada
        and
        low_10 >= entrada * 0.90
    ):
        candidatos_stop.append(
            low_10
        )

    stop_loss = max(
        candidatos_stop
    )

    riesgo_unitario = (
        entrada - stop_loss
    )

    if riesgo_unitario > 0:
        riesgo_pct = (
            riesgo_unitario / entrada
        ) * 100

        tp_2r = (
            entrada
            + 2 * riesgo_unitario
        )

        tp_3r = (
            entrada
            + 3 * riesgo_unitario
        )

    else:
        riesgo_pct = np.nan
        tp_2r = np.nan
        tp_3r = np.nan

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

    fila["Score 4H"] = score_4h
    fila["Score 1H"] = score_1h
    fila["Score total"] = score_total
    fila["Decisión"] = decision

    fila["Entrada"] = round(
        entrada,
        2
    )

    fila["Stop loss"] = round(
        stop_loss,
        2
    )

    fila["Riesgo %"] = round(
        riesgo_pct,
        2
    )

    fila["TP 2R"] = round(
        tp_2r,
        2
    )

    fila["TP 3R"] = round(
        tp_3r,
        2
    )

    fila["EMA10 4H"] = round(
        ema10_4h,
        2
    )

    fila["EMA20 4H"] = round(
        ema20_4h,
        2
    )

    fila["RSI 4H"] = round(
        rsi_4h,
        2
    )

    fila["Hist MACD 4H"] = round(
        hist_4h,
        4
    )

    fila["EMA20 1H"] = round(
        ema20_1h,
        2
    )

    fila["RSI 1H"] = round(
        rsi_1h,
        2
    )

    fila["Hist MACD 1H"] = round(
        hist_1h,
        4
    )

    fila["Estado intradía"] = "Completo"

    return fila


# ============================================================
# INTERFAZ LATERAL
# ============================================================

with st.sidebar:

    st.header("Configuración")

    precio_minimo = st.number_input(
        "Precio mínimo",
        min_value=0.0,
        value=5.0,
        step=1.0
    )

    volumen_minimo = st.number_input(
        "Volumen promedio mínimo 20D",
        min_value=0,
        value=500000,
        step=100000
    )

    dollar_volume_minimo = st.number_input(
        "Valor negociado promedio mínimo",
        min_value=0,
        value=10000000,
        step=1000000
    )

    score_diario_minimo = st.slider(
        "Score diario mínimo para análisis intradía",
        min_value=0,
        max_value=85,
        value=55
    )

    top_intradia = st.slider(
        "Número máximo para confirmación intradía",
        min_value=10,
        max_value=100,
        value=40,
        step=5
    )

    tamano_lote = st.selectbox(
        "Tickers por lote",
        options=[25, 50, 75, 100],
        index=2
    )

    pausa_lotes = st.number_input(
        "Pausa entre lotes",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.5
    )

    ejecutar = st.button(
        "🔍 Escanear Nasdaq + S&P 500",
        type="primary",
        use_container_width=True
    )


# ============================================================
# INFORMACIÓN INICIAL
# ============================================================

if not ejecutar:

    try:
        universo_preview = obtener_universo()

        st.info(
            f"Universo disponible: "
            f"{len(universo_preview):,} símbolos únicos."
        )

        resumen_origen = (
            universo_preview["Origen"]
            .value_counts()
            .reset_index()
        )

        resumen_origen.columns = [
            "Origen",
            "Símbolos"
        ]

        st.dataframe(
            resumen_origen,
            use_container_width=True,
            hide_index=True
        )

    except Exception as error:
        st.warning(
            f"No fue posible cargar la lista: {error}"
        )

    st.subheader("Flujo del scanner")

    st.code(
        """
Nasdaq + S&P 500
        ↓
Filtro de precio y liquidez
        ↓
Indicadores diarios
        ↓
Score diario /85
        ↓
Mejores candidatos
        ↓
Confirmación 4H y 1H
        ↓
Score total /100
        ↓
Entrada, stop, TP 2R y TP 3R
        """
    )


# ============================================================
# EJECUCIÓN DEL SCANNER
# ============================================================

if ejecutar:

    tiempo_inicio = time.time()

    with st.spinner(
        "Cargando universo Nasdaq + S&P 500..."
    ):
        universo = obtener_universo()

    st.success(
        f"Universo cargado: "
        f"{len(universo):,} símbolos únicos."
    )

    mapa_empresas = universo.set_index(
        "Ticker"
    ).to_dict("index")

    tickers = universo["Ticker"].tolist()

    lotes = list(
        dividir_lista(
            tickers,
            tamano_lote
        )
    )

    resultados_diarios = []
    errores = []

    barra_diaria = st.progress(0)
    mensaje_diario = st.empty()

    for numero_lote, lote in enumerate(lotes):

        mensaje_diario.write(
            f"Descargando lote "
            f"{numero_lote + 1}/{len(lotes)} "
            f"— {len(lote)} símbolos..."
        )

        try:
            datos_lote = descargar_lote_diario(
                tuple(lote)
            )

            for ticker in lote:

                df_ticker = extraer_ticker_del_lote(
                    datos_lote,
                    ticker,
                    len(lote)
                )

                if df_ticker.empty:
                    continue

                metadata = mapa_empresas.get(
                    ticker,
                    {}
                )

                resultado = analizar_diario(
                    ticker=ticker,
                    empresa=metadata.get(
                        "Empresa",
                        ""
                    ),
                    origen=metadata.get(
                        "Origen",
                        ""
                    ),
                    df=df_ticker,
                    precio_minimo=precio_minimo,
                    volumen_minimo=volumen_minimo,
                    dollar_volume_minimo=dollar_volume_minimo
                )

                if resultado is not None:
                    resultados_diarios.append(
                        resultado
                    )

        except Exception as error:
            errores.append({
                "Lote": numero_lote + 1,
                "Error": str(error)
            })

        barra_diaria.progress(
            (numero_lote + 1)
            / len(lotes)
        )

        if pausa_lotes > 0:
            time.sleep(
                pausa_lotes
            )

    mensaje_diario.success(
        "Análisis diario terminado."
    )

    if not resultados_diarios:
        st.error(
            "Ninguna acción superó los filtros "
            "o Yahoo Finance bloqueó las descargas."
        )

        if errores:
            st.dataframe(
                pd.DataFrame(errores)
            )

        st.stop()

    df_diario = pd.DataFrame(
        resultados_diarios
    )

    df_diario = df_diario.sort_values(
        by=[
            "Score diario",
            "Vol/Avg20",
            "Dollar Volume"
        ],
        ascending=[
            False,
            False,
            False
        ]
    ).reset_index(drop=True)

    candidatos_intradia = df_diario[
        df_diario["Score diario"]
        >= score_diario_minimo
    ].head(
        top_intradia
    ).copy()

    st.subheader(
        "Resultado del prefiltro diario"
    )

    r1, r2, r3, r4 = st.columns(4)

    with r1:
        st.metric(
            "Universo",
            f"{len(universo):,}"
        )

    with r2:
        st.metric(
            "Superaron liquidez",
            f"{len(df_diario):,}"
        )

    with r3:
        st.metric(
            "Para intradía",
            len(candidatos_intradia)
        )

    with r4:
        mejor_diario = int(
            df_diario["Score diario"].max()
        )

        st.metric(
            "Mejor score diario",
            f"{mejor_diario}/85"
        )

    # --------------------------------------------------------
    # CONFIRMACIÓN INTRADÍA
    # --------------------------------------------------------

    resultados_finales = []

    barra_intradia = st.progress(0)
    mensaje_intradia = st.empty()

    total_intradia = len(
        candidatos_intradia
    )

    for indice, (_, fila) in enumerate(
        candidatos_intradia.iterrows()
    ):

        ticker = fila["Ticker"]

        mensaje_intradia.write(
            f"Confirmación intradía "
            f"{ticker} "
            f"({indice + 1}/{total_intradia})..."
        )

        try:
            resultado_final = (
                agregar_confirmacion_intradia(
                    fila.to_dict()
                )
            )

            resultados_finales.append(
                resultado_final
            )

        except Exception as error:
            fila_error = fila.to_dict()

            fila_error["Score 4H"] = 0
            fila_error["Score 1H"] = 0
            fila_error["Score total"] = (
                fila_error["Score diario"]
            )

            fila_error["Decisión"] = (
                "ERROR INTRADÍA"
            )

            fila_error["Estado intradía"] = str(
                error
            )

            resultados_finales.append(
                fila_error
            )

        barra_intradia.progress(
            (indice + 1)
            / max(total_intradia, 1)
        )

        time.sleep(0.25)

    mensaje_intradia.success(
        "Confirmación intradía terminada."
    )

    df_final = pd.DataFrame(
        resultados_finales
    )

    df_final = df_final.sort_values(
        by=[
            "Score total",
            "Score diario",
            "Score 4H",
            "Score 1H"
        ],
        ascending=[
            False,
            False,
            False,
            False
        ]
    ).reset_index(drop=True)

    df_final.insert(
        0,
        "Rank",
        range(1, len(df_final) + 1)
    )

    # --------------------------------------------------------
    # FILA IDEAL
    # --------------------------------------------------------

    fila_ideal = {
        "Rank": "IDEAL",
        "Ticker": "—",
        "Empresa": "Condiciones ideales",
        "Origen": "Nasdaq / S&P 500",
        "Decisión": "COMPRAR",
        "Score total": "90+",
        "Score diario": "80–85",
        "Score 4H": "8–10",
        "Score 1H": "4–5",
        "Precio": "≥ $5",
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
        "ATR %": "2–5%",
        "RSI 4H": "50–70",
        "Hist MACD 4H": ">0",
        "RSI 1H": "50–70",
        "Hist MACD 1H": ">0"
    }

    # --------------------------------------------------------
    # TABLA PRINCIPAL
    # --------------------------------------------------------

    st.subheader(
        "🏆 Ranking final"
    )

    columnas_principales = [
        "Rank",
        "Ticker",
        "Empresa",
        "Origen",
        "Decisión",
        "Score total",
        "Score diario",
        "Score 4H",
        "Score 1H",
        "Precio",
        "Entrada",
        "Stop loss",
        "Riesgo %",
        "TP 2R",
        "TP 3R",
        "EMA10 D",
        "EMA20 D",
        "EMA50 D",
        "RSI D",
        "MACD D",
        "Signal D",
        "Hist MACD D",
        "Vol/Avg20",
        "ATR %",
        "RSI 4H",
        "Hist MACD 4H",
        "RSI 1H",
        "Hist MACD 1H",
        "Estado intradía"
    ]

    columnas_disponibles = [
        columna
        for columna in columnas_principales
        if columna in df_final.columns
    ]

    tabla_final = df_final[
        columnas_disponibles
    ].copy()

    tabla_con_ideal = pd.concat(
        [
            pd.DataFrame([fila_ideal]),
            tabla_final
        ],
        ignore_index=True
    )

    st.dataframe(
        tabla_con_ideal,
        use_container_width=True,
        hide_index=True,
        height=650
    )

    # --------------------------------------------------------
    # SEÑALES DE COMPRA
    # --------------------------------------------------------

    compras = df_final[
        df_final["Decisión"].isin([
            "COMPRA AGRESIVA",
            "COMPRAR"
        ])
    ].copy()

    st.subheader(
        "✅ Señales de compra"
    )

    if compras.empty:
        st.warning(
            "No se encontraron señales de compra "
            "con confirmación suficiente."
        )

    else:
        st.dataframe(
            compras[
                [
                    columna
                    for columna in columnas_principales
                    if columna in compras.columns
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # DESCARGAS
    # --------------------------------------------------------

    csv_final = df_final.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Descargar ranking final CSV",
        data=csv_final,
        file_name=(
            "nasdaq_sp500_swing_scanner_"
            f"{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        ),
        mime="text/csv"
    )

    with st.expander(
        "Ver ranking diario completo"
    ):
        st.dataframe(
            df_diario,
            use_container_width=True,
            hide_index=True
        )

    if errores:
        with st.expander(
            "Ver errores de descarga"
        ):
            st.dataframe(
                pd.DataFrame(errores),
                use_container_width=True,
                hide_index=True
            )

    tiempo_total = (
        time.time() - tiempo_inicio
    )

    st.success(
        f"Scanner completado en "
        f"{tiempo_total / 60:.1f} minutos."
    )


# ============================================================
# AVISO
# ============================================================

st.divider()

st.caption(
    "Herramienta educativa de análisis técnico. "
    "Yahoo Finance puede retrasar, limitar o interrumpir "
    "descargas masivas. Una puntuación alta no garantiza "
    "beneficios. Revisa noticias, earnings, liquidez, spreads "
    "y riesgo antes de ejecutar una operación."
)
