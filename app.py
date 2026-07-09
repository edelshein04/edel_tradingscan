import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

st.set_page_config(page_title="Finnhub Swing Scanner", layout="wide")

st.title("Finnhub Swing Scanner")
st.write("Prueba de conexión Finnhub + descarga OHLCV")

api_key = st.secrets["FINNHUB_API_KEY"]

ticker = st.text_input("Ticker", "AAPL").upper().strip()

# 1. Probar conexión con quote
quote_url = "https://finnhub.io/api/v1/quote"
quote_params = {
    "symbol": ticker,
    "token": api_key
}

quote_response = requests.get(quote_url, params=quote_params)
quote_data = quote_response.json()

st.subheader(f"1. Quote actual: {ticker}")
st.write("Status code:", quote_response.status_code)
st.json(quote_data)

# 2. Descargar velas OHLCV
to_ts = int(datetime.now(timezone.utc).timestamp())
from_ts = int((datetime.now(timezone.utc) - timedelta(days=365)).timestamp())

candle_url = "https://finnhub.io/api/v1/stock/candle"
candle_params = {
    "symbol": ticker,
    "resolution": "D",
    "from": from_ts,
    "to": to_ts,
    "token": api_key
}

candle_response = requests.get(candle_url, params=candle_params)
candle_data = candle_response.json()

st.subheader(f"2. Velas diarias OHLCV: {ticker}")
st.write("Status code:", candle_response.status_code)
st.write("Respuesta cruda de Finnhub:")
st.json(candle_data)

if candle_data.get("s") == "ok":
    df = pd.DataFrame({
        "Date": pd.to_datetime(candle_data["t"], unit="s"),
        "Open": candle_data["o"],
        "High": candle_data["h"],
        "Low": candle_data["l"],
        "Close": candle_data["c"],
        "Volume": candle_data["v"]
    })

    st.success("Datos OHLCV descargados correctamente.")
    st.dataframe(df.tail(30), use_container_width=True)

elif candle_data.get("s") == "no_data":
    st.warning("Finnhub respondió 'no_data'. Prueba con AAPL, MSFT, NVDA o TSLA.")

elif "error" in candle_data:
    st.error(f"Error de Finnhub: {candle_data['error']}")

else:
    st.error("No se pudieron obtener datos de velas.")
