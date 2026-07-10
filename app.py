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
    layout="wide",
)

st.title("📈 Nasdaq + S&P 500 Swing Scanner")
st.caption(
    "Scanner para swing trading de 3 a 10 días. "
    "Filtra acciones líquidas y confirma la tesis en diario, 4H y 1H."
)

NASDAQ_SYMBOL_URLS = [
    "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt",
    "https://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt",
]
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

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
    for posicion in range(0, len(lista), tamano):
        yield lista[posicion:posicion + tamano]


def normalizar_ticker_yahoo(ticker):
    return str(ticker).strip().upper().replace(".", "-")


def limpiar_columnas_yfinance(df):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    for columna in ["Open", "High", "Low", "Close", "Volume"]:
        if columna in df.columns:
            df[columna] = pd.to_numeric(df[columna], errors="coerce")

    return df


def parece_adr(nombre):
    texto = str(nombre).upper()
    patrones = [
        " ADR",
        "ADR ",
        "ADS",
        "AMERICAN DEPOSITARY",
        "DEPOSITARY SHARE",
        "DEPOSITARY SHARES",
    ]
    return any(patron in texto for patron in patrones)


# ============================================================
# UNIVERSO NASDAQ
# ============================================================

@st.cache_data(ttl=21600, show_spinner=False)
def obtener_nasdaq(excluir_adr=True):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
        ),
        "Accept": "text/plain,text/html,*/*",
        "Connection": "close",
    }

    contenido = None
    errores = []

    with requests.Session() as session:
        for url in NASDAQ_SYMBOL_URLS:
            for intento in range(3):
                try:
                    response = session.get(url, headers=headers, timeout=(15, 90))
                    response.raise_for_status()
                    texto = response.text.strip()

                    if "Symbol|Security Name" in texto and len(texto) > 1000:
                        contenido = texto
                        break
                    errores.append(f"{url}: contenido inválido")
                except requests.RequestException as error:
                    errores.append(
                        f"{url}: intento {intento + 1}: "
                        f"{type(error).__name__}: {error}"
                    )
                time.sleep(2 * (intento + 1))

            if contenido is not None:
                break

    if contenido is None:
        raise RuntimeError(
            "No fue posible descargar el directorio Nasdaq. "
            + " | ".join(errores[-6:])
        )

    nasdaq = pd.read_csv(io.StringIO(contenido), sep="|", dtype=str)

    if "Symbol" not in nasdaq.columns:
        raise RuntimeError("El archivo Nasdaq no contiene la columna Symbol.")

    nasdaq["Symbol"] = (
        nasdaq["Symbol"].astype(str).str.strip().str.upper()
    )

    nasdaq = nasdaq[
        ~nasdaq["Symbol"].str.contains("FILE CREATION TIME", case=False, na=False)
    ].copy()

    if "ETF" in nasdaq.columns:
        nasdaq = nasdaq[
            nasdaq["ETF"].fillna("").str.strip().str.upper().eq("N")
        ].copy()

    if "Test Issue" in nasdaq.columns:
        nasdaq = nasdaq[
            nasdaq["Test Issue"].fillna("").str.strip().str.upper().eq("N")
        ].copy()

    nasdaq = nasdaq[
        nasdaq["Symbol"].str.match(r"^[A-Z]{1,5}$", na=False)
    ].copy()

    columna_empresa = "Security Name" if "Security Name" in nasdaq.columns else "Symbol"

    if excluir_adr:
        nasdaq = nasdaq[~nasdaq[columna_empresa].apply(parece_adr)].copy()

    resultado = pd.DataFrame(
        {
            "Ticker original": nasdaq["Symbol"],
            "Ticker": nasdaq["Symbol"].apply(normalizar_ticker_yahoo),
            "Empresa": nasdaq[columna_empresa],
            "Origen": "Nasdaq",
        }
    )

    return (
        resultado.dropna(subset=["Ticker"])
        .query("Ticker != ''")
        .drop_duplicates(subset=["Ticker"])
        .sort_values("Ticker")
        .reset_index(drop=True)
    )


# ============================================================
# UNIVERSO S&P 500
# ============================================================

@st.cache_data(ttl=21600, show_spinner=False)
def obtener_sp500():
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(SP500_URL, headers=headers, timeout=(15, 60))
        response.raise_for_status()
        tablas = pd.read_html(io.StringIO(response.text))
    except Exception as error:
        raise RuntimeError(
            f"No fue posible descargar la lista del S&P 500: {error}"
        ) from error

    if not tablas:
        raise RuntimeError("Wikipedia no devolvió tablas para el S&P 500.")

    sp500 = tablas[0].copy()
    if not {"Symbol", "Security"}.issubset(sp500.columns):
        raise RuntimeError("La tabla del S&P 500 cambió de formato.")

    return pd.DataFrame(
        {
            "Ticker original": sp500["Symbol"],
            "Ticker": sp500["Symbol"].apply(normalizar_ticker_yahoo),
            "Empresa": sp500["Security"],
            "Origen": "S&P 500",
        }
    ).drop_duplicates(subset=["Ticker"]).reset_index(drop=True)


# ============================================================
# UNIVERSO COMBINADO
# ============================================================

@st.cache_data(ttl=21600, show_spinner=False)
def obtener_universo(excluir_adr=True):
    fuentes = []
    errores = []

    try:
        fuentes.append(obtener_nasdaq(excluir_adr=excluir_adr))
    except Exception as error:
        errores.append(f"Nasdaq: {error}")

    try:
        fuentes.append(obtener_sp500())
    except Exception as error:
        errores.append(f"S&P 500: {error}")

    fuentes_validas = [df for df in fuentes if df is not None and not df.empty]
    if not fuentes_validas:
        raise RuntimeError(
            "No fue posible cargar ninguna fuente del universo. " + " | ".join(errores)
        )

    universo = pd.concat(fuentes_validas, ignore_index=True)
    universo = (
        universo.groupby("Ticker", as_index=False)
        .agg(
            {
                "Ticker original": "first",
                "Empresa": "first",
                "Origen": lambda valores: " + ".join(sorted(set(valores))),
            }
        )
        .sort_values("Ticker")
        .reset_index(drop=True)
    )
    universo.attrs["errores_fuentes"] = errores
    return universo


# ============================================================
# DATOS FUNDAMENTALES / CAPITALIZACIÓN
# ============================================================

@st.cache_data(ttl=21600, show_spinner=False)
def obtener_market_cap(ticker):
    """Obtiene market cap. Devuelve NaN si Yahoo no lo publica."""
    try:
        objeto = yf.Ticker(ticker)

        try:
            fast_info = objeto.fast_info
            market_cap = fast_info.get("market_cap")
            if market_cap is not None and np.isfinite(float(market_cap)):
                return float(market_cap)
        except Exception:
            pass

        try:
            info = objeto.info
            market_cap = info.get("marketCap")
            if market_cap is not None and np.isfinite(float(market_cap)):
                return float(market_cap)
        except Exception:
            pass

    except Exception:
        pass

    return np.nan


# ============================================================
# INDICADORES
# ============================================================

def calcular_rsi(close, periodo=14):
    delta = close.diff()
    ganancias = delta.clip(lower=0)
    perdidas = -delta.clip(upper=0)
    promedio_ganancias = ganancias.ewm(alpha=1 / periodo, adjust=False).mean()
    promedio_perdidas = perdidas.ewm(alpha=1 / periodo, adjust=False).mean()
    rs = promedio_ganancias / promedio_perdidas.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def calcular_indicadores(df):
    if df.empty:
        return pd.DataFrame()

    df = limpiar_columnas_yfinance(df)
    requeridas = ["Open", "High", "Low", "Close", "Volume"]
    if not set(requeridas).issubset(df.columns):
        return pd.DataFrame()

    df = df.dropna(subset=requeridas).copy()

    df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["RSI14"] = calcular_rsi(df["Close"], periodo=14)

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    df["VOL_AVG20"] = df["Volume"].rolling(20).mean()
    df["VOL_REL"] = df["Volume"] / df["VOL_AVG20"]
    df["DOLLAR_VOL_AVG20"] = (df["Close"] * df["Volume"]).rolling(20).mean()

    df["HIGH_10"] = df["High"].rolling(10).max().shift(1)
    df["LOW_10"] = df["Low"].rolling(10).min().shift(1)

    rango_max_min = df["High"] - df["Low"]
    max_cierre_anterior = (df["High"] - df["Close"].shift(1)).abs()
    min_cierre_anterior = (df["Low"] - df["Close"].shift(1)).abs()
    df["TRUE_RANGE"] = pd.concat(
        [rango_max_min, max_cierre_anterior, min_cierre_anterior], axis=1
    ).max(axis=1)
    df["ATR14"] = df["TRUE_RANGE"].rolling(14).mean()
    df["ATR_PCT"] = (df["ATR14"] / df["Close"]) * 100

    return df


# ============================================================
# DESCARGAS
# ============================================================

def extraer_ticker_del_lote(datos_lote, ticker, total_tickers):
    if datos_lote is None or datos_lote.empty:
        return pd.DataFrame()

    try:
        if isinstance(datos_lote.columns, pd.MultiIndex):
            nivel_cero = list(datos_lote.columns.get_level_values(0))
            nivel_uno = list(datos_lote.columns.get_level_values(1))

            if ticker in nivel_cero:
                df = datos_lote[ticker].copy()
            elif ticker in nivel_uno:
                df = datos_lote.xs(ticker, axis=1, level=1).copy()
            else:
                return pd.DataFrame()
        elif total_tickers == 1:
            df = datos_lote.copy()
        else:
            return pd.DataFrame()

        return limpiar_columnas_yfinance(df).dropna(how="all")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=21600, show_spinner=False)
def descargar_lote_diario(tickers):
    return yf.download(
        tickers=list(tickers),
        period="1y",
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
        timeout=30,
    )


@st.cache_data(ttl=1800, show_spinner=False)
def descargar_intradia(ticker):
    df = yf.download(
        ticker,
        period="60d",
        interval="1h",
        auto_adjust=True,
        prepost=False,
        progress=False,
        threads=False,
        timeout=30,
    )
    return limpiar_columnas_yfinance(df)


# ============================================================
# ANÁLISIS DIARIO
# ============================================================

def analizar_diario(
    ticker,
    empresa,
    origen,
    df,
    precio_minimo,
    volumen_minimo,
    dollar_volume_minimo,
):
    if df.empty or len(df) < 100:
        return None

    df = calcular_indicadores(df).dropna().copy()
    if len(df) < 60:
        return None

    actual = df.iloc[-1]
    anterior = df.iloc[-2]
    anterior_2 = df.iloc[-3]

    precio = float(actual["Close"])
    volumen_promedio = float(actual["VOL_AVG20"])
    dollar_volume = float(actual["DOLLAR_VOL_AVG20"])

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
    hist_anterior = float(anterior["MACD_HIST"])
    hist_anterior_2 = float(anterior_2["MACD_HIST"])
    volumen_actual = float(actual["Volume"])
    volumen_anterior = float(anterior["Volume"])
    volumen_relativo = float(actual["VOL_REL"])
    high_10 = float(actual["HIGH_10"])
    low_10 = float(actual["LOW_10"])
    atr14 = float(actual["ATR14"])
    atr_pct = float(actual["ATR_PCT"])

    score_tendencia = 0
    if precio > ema50:
        score_tendencia += 9
    if ema10 > ema20:
        score_tendencia += 8
    if ema20 > ema50:
        score_tendencia += 8

    score_macd = 0
    if macd > signal:
        score_macd += 8
    if hist > 0:
        score_macd += 8
    if hist > hist_anterior > hist_anterior_2:
        score_macd += 4

    score_rsi = 0
    if 55 <= rsi <= 68:
        score_rsi = 15
    elif 50 <= rsi < 55 or 68 < rsi <= 75:
        score_rsi = 8

    score_volumen = 0
    if volumen_actual > volumen_promedio:
        score_volumen += 5
    if volumen_actual > 1.5 * volumen_promedio:
        score_volumen += 5
    if actual["Close"] > actual["Open"] and volumen_actual > volumen_anterior:
        score_volumen += 5

    score_precio = 0
    breakout = precio > high_10
    rango_diario = float(actual["High"] - actual["Low"])
    posicion_cierre = (
        float(actual["Close"] - actual["Low"]) / rango_diario
        if rango_diario > 0
        else 0
    )
    if breakout:
        score_precio += 5
    if posicion_cierre >= 0.66:
        score_precio += 5

    score_diario = (
        score_tendencia + score_macd + score_rsi + score_volumen + score_precio
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
        "Low 10D": low_10,
    }


# ============================================================
# FILTRO DE MARKET CAP
# ============================================================

def aplicar_filtro_market_cap(df, market_cap_minimo, exigir_market_cap):
    if df.empty:
        return df, 0

    filas = []
    sin_dato = 0

    barra = st.progress(0)
    mensaje = st.empty()

    total = len(df)
    for indice, (_, fila) in enumerate(df.iterrows()):
        ticker = fila["Ticker"]
        mensaje.write(
            f"Validando capitalización {ticker} ({indice + 1}/{total})..."
        )

        market_cap = obtener_market_cap(ticker)
        fila = fila.copy()
        fila["Market Cap"] = market_cap

        if np.isnan(market_cap):
            sin_dato += 1
            if not exigir_market_cap:
                filas.append(fila)
        elif market_cap >= market_cap_minimo:
            filas.append(fila)

        barra.progress((indice + 1) / max(total, 1))

    mensaje.success("Validación de capitalización terminada.")
    return pd.DataFrame(filas), sin_dato


# ============================================================
# CONFIRMACIÓN INTRADÍA
# ============================================================

def convertir_1h_a_4h(df_1h):
    if df_1h.empty:
        return pd.DataFrame()

    df = df_1h.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].copy()
    df["SESSION_DATE"] = df.index.date
    df["ORDEN"] = df.groupby("SESSION_DATE").cumcount()
    df["BLOQUE_4H"] = df["ORDEN"] // 4

    return (
        df.groupby(["SESSION_DATE", "BLOQUE_4H"])
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .reset_index(drop=True)
    )


def agregar_confirmacion_intradia(fila):
    ticker = fila["Ticker"]
    df_1h = descargar_intradia(ticker)

    if df_1h.empty:
        fila.update(
            {
                "Score 4H": 0,
                "Score 1H": 0,
                "Score total": fila["Score diario"],
                "Decisión": "SIN DATOS INTRADÍA",
                "Estado intradía": "Sin datos",
            }
        )
        return fila

    df_4h = convertir_1h_a_4h(df_1h)
    df_1h = calcular_indicadores(df_1h).dropna()
    df_4h = calcular_indicadores(df_4h).dropna()

    if len(df_1h) < 50 or len(df_4h) < 50:
        fila.update(
            {
                "Score 4H": 0,
                "Score 1H": 0,
                "Score total": fila["Score diario"],
                "Decisión": "DATOS INSUFICIENTES",
                "Estado intradía": "Datos insuficientes",
            }
        )
        return fila

    actual_1h = df_1h.iloc[-1]
    actual_4h = df_4h.iloc[-1]

    precio_4h = float(actual_4h["Close"])
    ema10_4h = float(actual_4h["EMA10"])
    ema20_4h = float(actual_4h["EMA20"])
    rsi_4h = float(actual_4h["RSI14"])
    hist_4h = float(actual_4h["MACD_HIST"])

    score_4h = 0
    if precio_4h > ema20_4h:
        score_4h += 4
    if ema10_4h > ema20_4h:
        score_4h += 3
    if hist_4h > 0:
        score_4h += 3

    precio_1h = float(actual_1h["Close"])
    ema20_1h = float(actual_1h["EMA20"])
    rsi_1h = float(actual_1h["RSI14"])
    hist_1h = float(actual_1h["MACD_HIST"])

    score_1h = 0
    if precio_1h > ema20_1h:
        score_1h += 2
    if 50 <= rsi_1h <= 70:
        score_1h += 2
    if hist_1h > 0:
        score_1h += 1

    score_total = int(fila["Score diario"]) + score_4h + score_1h

    precio = float(fila["Precio"])
    high_10 = float(fila["High 10D"])
    low_10 = float(fila["Low 10D"])
    atr14 = float(fila["ATR14"])

    entrada = max(precio, high_10)
    candidatos_stop = [entrada - 2 * atr14, entrada * 0.95]
    if low_10 < entrada and low_10 >= entrada * 0.90:
        candidatos_stop.append(low_10)
    stop_loss = max(candidatos_stop)
    riesgo_unitario = entrada - stop_loss

    if riesgo_unitario > 0:
        riesgo_pct = riesgo_unitario / entrada * 100
        tp_2r = entrada + 2 * riesgo_unitario
        tp_3r = entrada + 3 * riesgo_unitario
    else:
        riesgo_pct = np.nan
        tp_2r = np.nan
        tp_3r = np.nan

    if score_total >= 90 and score_4h >= 8 and score_1h >= 4:
        decision = "COMPRA AGRESIVA"
    elif score_total >= 80 and score_4h >= 7 and score_1h >= 3:
        decision = "COMPRAR"
    elif score_total >= 70:
        decision = "MANTENER / WATCHLIST"
    else:
        decision = "NO COMPRAR"

    fila.update(
        {
            "Score 4H": score_4h,
            "Score 1H": score_1h,
            "Score total": score_total,
            "Decisión": decision,
            "Entrada": round(entrada, 2),
            "Stop loss": round(stop_loss, 2),
            "Riesgo %": round(riesgo_pct, 2),
            "TP 2R": round(tp_2r, 2),
            "TP 3R": round(tp_3r, 2),
            "EMA10 4H": round(ema10_4h, 2),
            "EMA20 4H": round(ema20_4h, 2),
            "RSI 4H": round(rsi_4h, 2),
            "Hist MACD 4H": round(hist_4h, 4),
            "EMA20 1H": round(ema20_1h, 2),
            "RSI 1H": round(rsi_1h, 2),
            "Hist MACD 1H": round(hist_1h, 4),
            "Estado intradía": "Completo",
        }
    )
    return fila


# ============================================================
# INTERFAZ LATERAL
# ============================================================

with st.sidebar:
    st.header("Configuración")

    st.subheader("Universo")
    excluir_adr = st.checkbox("Excluir ADR / ADS", value=True)

    st.subheader("Filtros institucionales")
    precio_minimo = st.number_input(
        "Precio mínimo (USD)", min_value=0.0, value=10.0, step=1.0
    )
    volumen_minimo = st.number_input(
        "Volumen promedio mínimo 20D",
        min_value=0,
        value=500_000,
        step=100_000,
    )
    dollar_volume_minimo = st.number_input(
        "Dollar volume promedio mínimo 20D (USD)",
        min_value=0,
        value=20_000_000,
        step=5_000_000,
    )
    market_cap_minimo = st.number_input(
        "Market cap mínimo (USD)",
        min_value=0,
        value=2_000_000_000,
        step=500_000_000,
    )
    exigir_market_cap = st.checkbox(
        "Excluir acciones sin dato de market cap", value=True
    )

    st.subheader("Análisis")
    score_diario_minimo = st.slider(
        "Score diario mínimo para análisis intradía",
        min_value=0,
        max_value=85,
        value=55,
    )
    top_intradia = st.slider(
        "Número máximo para confirmación intradía",
        min_value=10,
        max_value=100,
        value=40,
        step=5,
    )
    tamano_lote = st.selectbox(
        "Tickers por lote", options=[25, 50, 75, 100], index=2
    )
    pausa_lotes = st.number_input(
        "Pausa entre lotes (segundos)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.5,
    )

    ejecutar = st.button(
        "🔍 Escanear Nasdaq + S&P 500",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# INFORMACIÓN INICIAL
# ============================================================

if not ejecutar:
    try:
        universo_preview = obtener_universo(excluir_adr=excluir_adr)
        st.info(f"Universo disponible: {len(universo_preview):,} símbolos únicos.")

        resumen_origen = universo_preview["Origen"].value_counts().reset_index()
        resumen_origen.columns = ["Origen", "Símbolos"]
        st.dataframe(resumen_origen, use_container_width=True, hide_index=True)
    except Exception as error:
        st.warning(f"No fue posible cargar la lista: {error}")

    st.subheader("Flujo del scanner")
    st.code(
        """
Nasdaq + S&P 500
        ↓
Excluir ETF, test issues y ADR/ADS
        ↓
Precio ≥ $10
Volumen promedio 20D ≥ 500,000
Dollar volume promedio 20D ≥ $20M
Market cap ≥ $2B
        ↓
Indicadores diarios y score /85
        ↓
Confirmación 4H y 1H
        ↓
Score total /100
        ↓
Entrada, stop, TP 2R y TP 3R
        """
    )
    st.stop()


# ============================================================
# EJECUCIÓN DEL SCANNER
# ============================================================

tiempo_inicio = time.time()

with st.spinner("Cargando universo Nasdaq + S&P 500..."):
    try:
        universo = obtener_universo(excluir_adr=excluir_adr)
    except Exception as error:
        st.error("No fue posible cargar el universo de acciones.")
        st.code(str(error))
        st.stop()

st.success(f"Universo cargado: {len(universo):,} símbolos únicos.")

errores_fuentes = universo.attrs.get("errores_fuentes", [])
if errores_fuentes:
    st.warning("El scanner continuará con una fuente parcial: " + " | ".join(errores_fuentes))

mapa_empresas = universo.set_index("Ticker").to_dict("index")
tickers = universo["Ticker"].tolist()
lotes = list(dividir_lista(tickers, tamano_lote))

if not lotes:
    st.error("El universo no contiene símbolos válidos.")
    st.stop()

resultados_diarios = []
errores = []
barra_diaria = st.progress(0)
mensaje_diario = st.empty()

for numero_lote, lote in enumerate(lotes):
    mensaje_diario.write(
        f"Descargando lote {numero_lote + 1}/{len(lotes)} — {len(lote)} símbolos..."
    )

    try:
        datos_lote = descargar_lote_diario(tuple(lote))

        for ticker in lote:
            df_ticker = extraer_ticker_del_lote(datos_lote, ticker, len(lote))
            if df_ticker.empty:
                continue

            metadata = mapa_empresas.get(ticker, {})
            resultado = analizar_diario(
                ticker=ticker,
                empresa=metadata.get("Empresa", ""),
                origen=metadata.get("Origen", ""),
                df=df_ticker,
                precio_minimo=precio_minimo,
                volumen_minimo=volumen_minimo,
                dollar_volume_minimo=dollar_volume_minimo,
            )
            if resultado is not None:
                resultados_diarios.append(resultado)

    except Exception as error:
        errores.append({"Lote": numero_lote + 1, "Error": str(error)})

    barra_diaria.progress((numero_lote + 1) / len(lotes))
    if pausa_lotes > 0:
        time.sleep(pausa_lotes)

mensaje_diario.success("Prefiltro de precio y liquidez terminado.")

if not resultados_diarios:
    st.error(
        "Ninguna acción superó los filtros iniciales o Yahoo Finance bloqueó las descargas."
    )
    if errores:
        st.dataframe(pd.DataFrame(errores), use_container_width=True)
    st.stop()

# Primero ordenamos para validar market cap en los candidatos de mayor calidad.
df_prefiltro = pd.DataFrame(resultados_diarios).sort_values(
    by=["Score diario", "Vol/Avg20", "Dollar Volume"],
    ascending=[False, False, False],
).reset_index(drop=True)

st.info(
    f"Superaron precio y liquidez: {len(df_prefiltro):,}. "
    "Ahora se valida market cap."
)

df_diario, market_cap_sin_dato = aplicar_filtro_market_cap(
    df_prefiltro,
    market_cap_minimo=market_cap_minimo,
    exigir_market_cap=exigir_market_cap,
)

if df_diario.empty:
    st.error("Ninguna acción superó el filtro de capitalización bursátil.")
    st.stop()

df_diario = df_diario.sort_values(
    by=["Score diario", "Vol/Avg20", "Dollar Volume"],
    ascending=[False, False, False],
).reset_index(drop=True)

candidatos_intradia = df_diario[
    df_diario["Score diario"] >= score_diario_minimo
].head(top_intradia).copy()

st.subheader("Resultado del prefiltro institucional")
r1, r2, r3, r4, r5 = st.columns(5)
r1.metric("Universo", f"{len(universo):,}")
r2.metric("Precio + liquidez", f"{len(df_prefiltro):,}")
r3.metric("Market cap", f"{len(df_diario):,}")
r4.metric("Para intradía", f"{len(candidatos_intradia):,}")
r5.metric("Mejor score diario", f"{int(df_diario['Score diario'].max())}/85")

if market_cap_sin_dato:
    st.caption(
        f"Yahoo Finance no devolvió market cap para {market_cap_sin_dato} símbolos."
    )

columnas_prefiltro = [
    "Ticker",
    "Empresa",
    "Origen",
    "Score diario",
    "Precio",
    "Volumen promedio",
    "Dollar Volume",
    "Market Cap",
    "RSI D",
    "Vol/Avg20",
    "ATR %",
]
st.dataframe(
    df_diario[[c for c in columnas_prefiltro if c in df_diario.columns]],
    use_container_width=True,
    hide_index=True,
)

if candidatos_intradia.empty:
    st.warning("Ninguna acción alcanzó el score diario mínimo para confirmación intradía.")
    st.stop()

resultados_finales = []
barra_intradia = st.progress(0)
mensaje_intradia = st.empty()
total_intradia = len(candidatos_intradia)

for indice, (_, fila) in enumerate(candidatos_intradia.iterrows()):
    ticker = fila["Ticker"]
    mensaje_intradia.write(
        f"Confirmación intradía {ticker} ({indice + 1}/{total_intradia})..."
    )

    try:
        resultados_finales.append(agregar_confirmacion_intradia(fila.to_dict()))
    except Exception as error:
        fila_error = fila.to_dict()
        fila_error.update(
            {
                "Score 4H": 0,
                "Score 1H": 0,
                "Score total": fila_error["Score diario"],
                "Decisión": "ERROR INTRADÍA",
                "Estado intradía": str(error),
            }
        )
        resultados_finales.append(fila_error)

    barra_intradia.progress((indice + 1) / max(total_intradia, 1))
    time.sleep(0.25)

mensaje_intradia.success("Confirmación intradía terminada.")

df_final = pd.DataFrame(resultados_finales).sort_values(
    by=["Score total", "Score diario", "Score 4H", "Score 1H"],
    ascending=[False, False, False, False],
).reset_index(drop=True)

st.subheader("Resultados finales")
columnas_finales = [
    "Ticker",
    "Empresa",
    "Origen",
    "Decisión",
    "Score total",
    "Score diario",
    "Score 4H",
    "Score 1H",
    "Precio",
    "Market Cap",
    "Entrada",
    "Stop loss",
    "Riesgo %",
    "TP 2R",
    "TP 3R",
    "RSI D",
    "RSI 4H",
    "RSI 1H",
    "Vol/Avg20",
    "Dollar Volume",
    "ATR %",
    "Estado intradía",
]
st.dataframe(
    df_final[[c for c in columnas_finales if c in df_final.columns]],
    use_container_width=True,
    hide_index=True,
)

csv = df_final.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "⬇️ Descargar resultados CSV",
    data=csv,
    file_name=f"swing_scanner_{datetime.now():%Y%m%d_%H%M}.csv",
    mime="text/csv",
    use_container_width=True,
)

if errores:
    with st.expander("Errores de descarga por lote"):
        st.dataframe(pd.DataFrame(errores), use_container_width=True, hide_index=True)

minutos = (time.time() - tiempo_inicio) / 60
st.caption(
    f"Proceso terminado en {minutos:.1f} minutos. "
    "Los resultados son un filtro cuantitativo, no una recomendación financiera."
)
