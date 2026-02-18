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
ticker = st.sidebar.text_input("Ticker Symbol", value="TASI.SR")
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
    mu, sigma = returns.mean(), returns.std()
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
        st.error("❌ Not enough data. Try a different ticker or earlier start date.")
    else:
        # Run simulation
        sims = simulate_gbm(returns, n_sims, horizon_days)
        final_values = sims[:, -1]
        var_value = np.percentile(final_values, (1-confidence)*100)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("95% VaR", f"{var_value:.2f}", f"{abs(var_value-100):.2f}% potential loss")
        col2.metric("Mean Outcome", f"{np.mean(final_values):.2f}")
        col3.metric("Simulations", f"{n_sims:,}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(final_values, bins=40, alpha=0.7, edgecolor='black', color='#2E86AB')
        ax.axvline(var_value, color='#A23B72', linestyle='--', linewidth=2, label=f'{int(confidence*100)}% VaR: {var_value:.1f}')
        ax.set_xlabel('Normalized Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{ticker}: {horizon_days}-Day Risk Distribution ({n_sims:,} sims)')
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        
        # Plain-English interpretation
        st.info(f"💡 **Interpretation**: There's a {int((1-confidence)*100)}% chance that {ticker} could lose more than **{abs(var_value-100):.2f}%** over {horizon_days} days.")
else:
    st.info("👈 Adjust parameters in the sidebar, then click **Run Simulation**")

# Footer
st.markdown("---")
st.caption("By Jihan Lahmar | [View Code on GitHub](https://github.com/jihanlahmar/jihanlahmar/tree/main/projects/monte-carlo-port-risk) | [Portfolio](https://jihanlahmar.github.io/jihanlahmar/)")
