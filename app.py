import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

st.title("Finnhub Swing Scanner")

api_key = st.secrets["FINNHUB_API_KEY"]

ticker = st.text_input("Ticker", "NVDA").upper()

to_ts = int(datetime.now(timezone.utc).timestamp())
from_ts = int((datetime.now(timezone.utc) - timedelta(days=220)).timestamp())

url = "https://finnhub.io/api/v1/stock/candle"
params = {
    "symbol": ticker,
    "resolution": "D",
    "from": from_ts,
    "to": to_ts,
    "token": api_key
}

response = requests.get(url, params=params)
data = response.json()

st.subheader(f"Velas diarias OHLCV: {ticker}")

if data.get("s") == "ok":
    df = pd.DataFrame({
        "Date": pd.to_datetime(data["t"], unit="s"),
        "Open": data["o"],
        "High": data["h"],
        "Low": data["l"],
        "Close": data["c"],
        "Volume": data["v"]
    })

    st.dataframe(df.tail(20), use_container_width=True)
else:
    st.error("No se pudieron obtener datos.")
    st.json(data)
