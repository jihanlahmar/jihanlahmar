
# var_comparison.py
# Morocco-Gulf Exposure: VaR Comparison Engine
# Run: python var_comparison.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm

def load_returns(ticker="TASI.SR", start="2023-01-01"):
    """Load daily returns for Morocco-Gulf exposure proxy"""
    data = yf.download(ticker, start=start)['Close'].pct_change().dropna()
    return data

def historical_var(returns, confidence=0.95):
    """Historical VaR: percentile of actual returns"""
    return np.percentile(returns, (1 - confidence) * 100)

def parametric_var(returns, confidence=0.95):
    """Parametric VaR: assumes normal distribution"""
    mu, sigma = returns.mean(), returns.std()
    return mu + sigma * norm.ppf(1 - confidence)

def monte_carlo_var(returns, confidence=0.95, n_sims=10000, horizon=1):
    """Monte Carlo VaR: simulate future paths"""
    mu, sigma = returns.mean(), returns.std()
    # Simulate horizon-day returns
    simulated_returns = np.random.normal(mu * horizon, sigma * np.sqrt(horizon), n_sims)
    return np.percentile(simulated_returns, (1 - confidence) * 100)

def compare_var(returns, confidence_levels=[0.95, 0.99]):
    """Compare all 3 methods across confidence levels"""
    results = {}
    for conf in confidence_levels:
        results[conf] = {
            'Historical': historical_var(returns, conf),
            'Parametric': parametric_var(returns, conf),
            'Monte Carlo': monte_carlo_var(returns, conf)
        }
    return results
ef plot_comparison(returns, results):
    """Create side-by-side comparison chart"""
    methods = ['Historical', 'Parametric', 'Monte Carlo']
    conf_levels = list(results.keys())
    
    fig, axes = plt.subplots(1, len(conf_levels), figsize=(6*len(conf_levels), 4))
    if len(conf_levels) == 1:
        axes = [axes]
    
    for ax, conf in zip(axes, conf_levels):
        values = [results[conf][m] for m in methods]
        bars = ax.bar(methods, values, color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.8)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_title(f'{int(conf*100)}% Confidence VaR')
        ax.set_ylabel('Return')
        ax.grid(axis='y', alpha=0.3)
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, val, f'{val:.3f}', 
                   ha='center', va='bottom' if val < 0 else 'top', fontsize=9)
    
    plt.suptitle('VaR Method Comparison: Morocco-Gulf Exposure', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('var_comparison_chart.png', dpi=300, bbox_inches='tight')
    print(f'✅ Saved: var_comparison_chart.png')

def generate_recommendation(results):
    """Simple logic to recommend a method"""
    # If methods disagree significantly, prefer Historical (no assumptions)
    conf_95 = results[0.95]
    spread = max(conf_95.values()) - min(conf_95.values())
    if spread > 0.02:  # 2% difference is significant
        return "Methods diverge → Prefer Historical VaR (no distribution assumptions)"
    else:
        return "Methods agree → Parametric VaR is efficient for quick estimates"

if __name__ == "__main__":
    # Load data
    ticker = "TASI.SR"  # Morocco-Gulf exposure proxy
    returns = load_returns(ticker)
    
    # Run comparison
    results = compare_var(returns)
    
    # Print results
    print(f"\n📊 VaR Comparison for {ticker}")
    print(f"{'Confidence':<12} {'Historical':<12} {'Parametric':<12} {'Monte Carlo':<12}")
    print("-" * 48)
    for conf, methods in results.items():
        print(f"{int(conf*100):<12}% {methods['Historical']:<12.4f} {methods['Parametric']:<12.4f} {methods['Monte Carlo']:<12.4f}")
      # Generate recommendation
    rec = generate_recommendation(results)
    print(f"\n💡 Recommendation: {rec}")
    
    # Plot
    plot_comparison(returns, results)
