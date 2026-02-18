# app.py - Streamlit Live Demo for Monte Carlo Port Risk Simulator
# Deploy: https://streamlit.io/cloud

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Page config
st.set_page_config(page_title="Monte Carlo Risk Simulator", page_icon="📊", layout="wide")

# Title + Description
st.title("🇲🇦→🇸🇦 Morocco-Gulf Risk Simulator")
st.markdown("*Monte Carlo Value-at-Risk modeling for strategic trade exposures*")

# Sidebar controls
st.sidebar.header("⚙️ Parameters")
ticker = st.sidebar.selectbox(
    "Ticker Symbol",
    [
        "^GSPC",      # S&P 500 ✅ Best for testing
        "MAD=X",      # Moroccan Dirham / USD 🇲🇦
        "SAR=X",      # Saudi Riyal / USD 🇸🇦
        "AAPL",       # Apple Inc. (test)
        "^DFMGI",     # Dubai Financial Market
        "EURUSD=X",   # Euro / USD
    ],
    index=0
)
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
horizon_days = st.sidebar.slider("Forecast Horizon (days)", 7, 90, 30)
n_sims = st.sidebar.selectbox("Simulations", [1000, 5000, 10000, 50000], index=2)
confidence = st.sidebar.slider("Confidence Level", 90, 99, 95) / 100

# Load data function
@st.cache_data
def load_returns(ticker, start):
    try:
        data = yf.download(ticker, start=start)['Close'].pct_change().dropna()
        return data
    except:
        return pd.Series()

# Monte Carlo engine
def simulate_gbm(returns, n_sims, horizon):
    mu = float(returns.mean())      # ← Convert to float
    sigma = float(returns.std())    # ← Convert to float
    paths = np.zeros((n_sims, horizon+1))
    paths[:, 0] = 100
    for t in range(1, horizon+1):
        z = np.random.normal(0, 1, n_sims)
        paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5*sigma**2) + sigma*z)
    return paths

# Main app
if st.button("🚀 Run Simulation", type="primary"):
    with st.spinner(f"Loading {ticker} data..."):
        returns = load_returns(ticker, start_date)
    
    if len(returns) < 30:
        st.error("❌ Not enough data for this ticker.")
        st.info("💡 **Try**: ^GSPC (S&P 500), MAD=X (Morocco USD), or SAR=X (Saudi USD)")
        st.stop()
    else:
        # Run simulation
        sims
