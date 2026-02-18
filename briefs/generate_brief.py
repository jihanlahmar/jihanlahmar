# generate_brief.py - Auto-generate 1-page PDF briefs
# Run: python generate_brief.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import yfinance as yf
from datetime import datetime

def load_returns(ticker="^GSPC", start="2023-01-01"):
    """Load daily returns"""
    data = yf.download(ticker, start=start, progress=False)
    return data['Close'].pct_change().dropna().values.astype(float)

def simulate_gbm(returns, n_sims=10000, horizon=30):
    """Monte Carlo simulator"""
    mu, sigma = np.mean(returns), np.std(returns)
    paths = np.zeros((n_sims, horizon+1))
    paths[:, 0] = 100
    for t in range(1, horizon+1):
        z = np.random.normal(0, 1, n_sims)
        paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5*sigma**2) + sigma*z)
    return paths

def generate_monte_carlo_brief(ticker="^GSPC", output_path="briefs/tanger-med-var-2026.pdf"):
    """Generate 1-page PDF brief for Monte Carlo project"""
    
    # Run simulation
    returns = load_returns(ticker)
    sims = simulate_gbm(returns)
    final_values = sims[:, -1]
    var_95 = np.percentile(final_values, 5)
    var_99 = np.percentile(final_values, 1)
    
    # Create PDF
    with PdfPages(output_path) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 11))  # Letter size
        fig.patch.set_facecolor('white')
        
        # Header
        ax.text(0.5, 0.95, 'DEAL RISK BRIEF', ha='center', va='top', 
               fontsize=16, fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, 0.92, f'{ticker} | Monte Carlo VaR Analysis', ha='center', va='top',
               fontsize=12, style='italic', transform=ax.transAxes)
        ax.text(0.5, 0.89, f'Generated: {datetime.now().strftime("%Y-%m-%d")}', 
               ha='center', va='top', fontsize=9, transform=ax.transAxes)
        
        # Key metrics box
        metrics_text = f"""
KEY METRICS (30-Day Horizon)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
95% VaR:     {var_95:.2f}  →  {abs(var_95-100):.2f}% potential loss
99% VaR:     {var_99:.2f}  →  {abs(var_99-100):.2f}% potential loss
Mean Outcome: {np.mean(final_values):.2f}
Simulations:  10,000
Method:       Geometric Brownian Motion
        """
        ax.text(0.05, 0.78, metrics_text, fontsize=10, family='monospace',
               bbox=dict(boxstyle='round', facecolor='#f5f5f5', edgecolor='#ccc'),
               transform=ax.transAxes, va='top')
        
        # Chart
        ax.hist(final_values, bins=40, alpha=0.7, edgecolor='black', color='#2E86AB')
        ax.axvline(var_95, color='#A23B72', linestyle='--', linewidth=2, label=f'95% VaR: {var_95:.1f}')
        ax.axvline(var_99, color='#F18F01', linestyle=':', linewidth=2, label=f'99% VaR: {var_99:.1f}')
        ax.set_xlabel('Normalized Value', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.set_title(f'Risk Distribution: {ticker}', fontsize=11, pad=20)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        
        # Position chart
        ax.set_position([0.1, 0.25, 0.8, 0.5])
        
        # Interpretation
        interpretation = f"""
INTERPRETATION
━━━━━━━━━━━━━━
• There is a 5% probability that {ticker} could decline by more than {abs(var_95-100):.2f}% 
  over a 30-day period under current market conditions.

• This metric supports strategic decision-making for Morocco-Gulf trade exposures, 
  port logistics planning, and commodity risk hedging.

• Methodology: Geometric Brownian Motion with 10,000 Monte Carlo simulations. 
  Historical volatility and drift estimated from daily returns since 2023-01-01.
        """
        ax.text(0.05, 0.18, interpretation, fontsize=9, transform=ax.transAxes,
               va='top', linespacing=1.5)
        
        # Footer
        ax.text(0.5, 0.03, 'By Jihan Lahmar | Morocco-Gulf Analytics', 
               ha='center', va='bottom', fontsize=8, transform=ax.transAxes)
        ax.text(0.5, 0.01, 'jihan.lahmar@protonmail.com | Portfolio: jihanlahmar.github.io/jihanlahmar',
               ha='center', va='bottom', fontsize=7, transform=ax.transAxes)
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print(f'✅ Saved: {output_path}')
    return output_path

if __name__ == "__main__":
    # Generate brief for Monte Carlo project
    generate_monte_carlo_brief(ticker="^GSPC", output_path="briefs/tanger-med-var-2026.pdf")
    
    # To generate VaR Comparison brief, create similar function
    print('💡 To generate VaR Comparison brief, run with: ticker="MAD=X"')
