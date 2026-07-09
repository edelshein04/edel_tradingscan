import streamlit as st
import requests

st.title("Finnhub Swing Scanner")

api_key = st.secrets["FINNHUB_API_KEY"]

ticker = st.text_input("Ticker", "NVDA").upper()

url = "https://finnhub.io/api/v1/quote"
params = {
    "symbol": ticker,
    "token": api_key
}

response = requests.get(url, params=params)
data = response.json()

st.subheader(f"Precio actual: {ticker}")
st.json(data)
