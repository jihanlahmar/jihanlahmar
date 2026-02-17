# monte_carlo_port.py
# Morocco-Gulf Exposure: Monte Carlo Risk Simulator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

def load_returns(ticker="TASI.SR", start="2023-01-01"):
    """Load daily returns for Morocco-Gulf exposure proxy"""
    data = yf.download(ticker, start=start)['Close'].pct_change().dropna()
    return data

def simulate_gbm(returns, n_sims=10000, horizon_days=30):
    """Geometric Brownian Motion simulator"""
    mu, sigma = returns.mean(), returns.std()
    paths = np.zeros((n_sims, horizon_days+1))
    paths[:, 0] = 100
    for t in range(1, horizon_days+1):
        z = np.random.normal(0, 1, n_sims)
        paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5*sigma**2) + sigma*z)
    return paths

def run_brief(ticker="TASI.SR"):
    """Run simulation + calculate 95% VaR"""
    returns = load_returns(ticker)
    sims = simulate_gbm(returns)
    final_values = sims[:, -1]
    var_95 = np.percentile(final_values, 5)
    return var_95, final_values

if __name__ == "__main__":
    var, dist = run_brief()
    plt.figure(figsize=(10,6))
    plt.hist(dist, bins=40, alpha=0.7, edgecolor='black', color='#2E86AB')
    plt.axvline(var, color='#A23B72', linestyle='--', linewidth=2, label=f'95% VaR: {var:.1f}')
    plt.title('Morocco-Gulf Exposure: 30-Day Risk Distribution (Monte Carlo, 10k sims)')
    plt.xlabel('Normalized Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('risk_brief_chart.png', dpi=300, bbox_inches='tight')
    print(f'✅ Saved: risk_brief_chart.png')
    print(f'📊 95% VaR (30-day): {var:.1f} | 5% chance of loss > {abs(var-100):.1f}%')
