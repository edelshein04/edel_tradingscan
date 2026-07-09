import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Swing Scanner", layout="wide")

st.title("Swing Scanner")
st.write("Descarga de velas diarias OHLCV usando yfinance")

ticker = st.text_input("Ticker", "AAPL").upper().strip()

df = yf.download(
    ticker,
    period="1y",
    interval="1d",
    auto_adjust=True,
    progress=False
)

st.subheader(f"Velas diarias OHLCV: {ticker}")

if df.empty:
    st.error("No se pudieron obtener datos.")
else:
    st.success("Datos descargados correctamente.")
    st.dataframe(df.tail(30), use_container_width=True)
