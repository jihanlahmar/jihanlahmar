# VaR Comparison Engine

> Morocco-Gulf exposure risk modeling | 3 methods | Python-automated

## 🎯 Purpose
Compare three Value-at-Risk (VaR) estimation methods for Morocco-Gulf trade exposures:
- **Historical VaR**: Non-parametric, uses actual past returns
- **Parametric VaR**: Assumes normal distribution, uses mean + volatility
- **Monte Carlo VaR**: Simulates 10,000 paths using Geometric Brownian Motion

## 📊 Output
- Console print: VaR values for all 3 methods at 95% and 99% confidence
- `var_comparison_chart.png`: Side-by-side histogram comparison
- Plain-English recommendation: "Which method to trust for your exposure?"

## 🚀 Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the script
python var_comparison.py

# 3. Check output
✅ var_comparison_chart.png saved
✅ Console shows VaR values + recommendation
