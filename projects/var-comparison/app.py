# app.py - Streamlit Live Demo for VaR Comparison Engine
# Deploy: https://streamlit.io/cloud

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm

# Page config
st.set_page_config(page_title="VaR Comparison Engine", page_icon="📊", layout="wide")

# Title + Description
st.title("📊 VaR Comparison Engine")
st.markdown("*Compare Historical, Parametric & Monte Carlo VaR methods for Morocco-Gulf exposures*")

# Sidebar controls
st.sidebar.header("⚙️ Parameters")
ticker = st.sidebar.selectbox(
    "Ticker Symbol",
    ["^GSPC", "MAD=X", "SAR=X", "AAPL", "EURUSD=X"],
    index=0
)
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
confidence = st.sidebar.slider("Confidence Level", 90, 99, 95) / 100
n_sims = st.sidebar.selectbox("Monte Carlo Simulations", [5000, 10000, 20000], index=1)

# Load data function
@st.cache_data
def load_returns(ticker, start):
    try:
        data = yf.download(ticker, start=start, progress=False)
        if len(data) == 0:
            return np.array([])
        close_prices = data['Close'].dropna()
        if len(close_prices) < 30:
            return np.array([])
        returns = close_prices.pct_change().dropna().values
        return returns.astype(float)
    except Exception as e:
        return np.array([])

# VaR Methods
def historical_var(returns, confidence):
    """Historical VaR: percentile of actual returns"""
    return np.percentile(returns, (1 - confidence) * 100)

def parametric_var(returns, confidence):
    """Parametric VaR: assumes normal distribution"""
    mu, sigma = np.mean(returns), np.std(returns)
    return mu + sigma * norm.ppf(1 - confidence)

def monte_carlo_var(returns, confidence, n_sims=10000, horizon=1):
    """Monte Carlo VaR: simulate future paths"""
    mu, sigma = np.mean(returns), np.std(returns)
    simulated_returns = np.random.normal(mu * horizon, sigma * np.sqrt(horizon), n_sims)
    return np.percentile(simulated_returns, (1 - confidence) * 100)

# Main app
st.markdown("---")

if st.button("🚀 Compare VaR Methods", type="primary"):
    with st.spinner(f"Loading {ticker} data..."):
        returns = load_returns(ticker, start_date)
    
    if len(returns) < 30:
        st.error("❌ Not enough data for this ticker.")
        st.info("💡 **Try**: ^GSPC (S&P 500) or AAPL (Apple)")
        st.stop()
    
    # Calculate VaR for all 3 methods
    var_historical = historical_var(returns, confidence)
    var_parametric = parametric_var(returns, confidence)
    var_monte_carlo = monte_carlo_var(returns, confidence, n_sims)
    
    # Display metrics
    st.subheader(f"{int(confidence*100)}% VaR Comparison for {ticker}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Historical VaR", f"{var_historical:.4f}", "No assumptions")
    col2.metric("Parametric VaR", f"{var_parametric:.4f}", "Normal distribution")
    col3.metric("Monte Carlo VaR", f"{var_monte_carlo:.4f}", f"{n_sims:,} simulations")
    
    # Bar chart comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    methods = ['Historical', 'Parametric', 'Monte Carlo']
    values = [var_historical, var_parametric, var_monte_carlo]
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    bars = ax.bar(methods, values, color=colors, alpha=0.8, edgecolor='black')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_ylabel('Return')
    ax.set_title(f'{ticker}: {int(confidence*100)}% VaR Method Comparison')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val, f'{val:.4f}', 
               ha='center', va='bottom' if val < 0 else 'top', fontsize=10, fontweight='bold')
    
    st.pyplot(fig)
    
    # Recommendation logic
    st.markdown("---")
    st.subheader("💡 Recommendation")
    spread = max(values) - min(values)
    if spread > 0.02:
        st.warning(f"**Methods diverge** (spread: {spread:.4f}) → Prefer **Historical VaR** (no distribution assumptions)")
    else:
        st.success(f"**Methods agree** (spread: {spread:.4f}) → **Parametric VaR** is efficient for quick estimates")
    
    # Plain-English interpretation
    st.info(f"**Interpretation**: There's a {int((1-confidence)*100)}% chance that {ticker} could lose more than **{abs(min(values))*100:.2f}%** in one day.")
else:
    st.info("👈 Adjust parameters in the sidebar, then click **Compare VaR Methods**")

# Footer
st.markdown("---")
st.caption("By Jihan Lahmar | [View Code on GitHub](https://github.com/jihanlahmar/jihanlahmar/tree/main/projects/var-comparison) | [Portfolio](https://jihanlahmar.github.io/jihanlahmar/)")
