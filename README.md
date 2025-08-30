# 📈 Statistical Arbitrage — Pairs Trading with Cointegration

This project implements a **statistical arbitrage strategy** using the **Engle–Granger cointegration test** to identify mean-reverting stock pairs and trade their spread.  

It demonstrates skills in **time-series econometrics, trading strategy design, backtesting, and reporting**.

---

## 🔍 Project Overview

- **Data Source**: Daily adjusted close prices pulled via yfinance api
- **Methodology**:
  1. Download sector stocks (banks in this demo).
  2. Test for cointegration using `statsmodels.tsa.stattools.coint`.
  3. Construct spread for pairs, compute z‑score.
  4. Trading rules:
     - Enter **long spread** if z < -2.
     - Enter **short spread** if z > +2.
     - Exit when z ≈ 0.
  5. Backtest on single pairs and as a multi‑pair portfolio.
- **Evaluation Metrics**:
  - CAGR
  - Annualised Mean
  - Annualised Volatility
  - Sharpe Ratio
  - Sortino Ratio
  - Max Drawdown
  - Win Rate

---

## 📊 Results & Insights

**Backtest Period:** 2018‑01‑02 to 2025‑07‑31  
**Universe:** 10 U.S. bank stocks - ["JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "TFC", "BK"] 

### Z‑Score Example (BAC-PNC)
![Z-Score](images/equity_and_zscore_BAC_PNC.png)

### Z‑Score Example (TFC–USB)
![Z-Score](images/equity_and_zscore_TFC_USB.png)

### Cointegration Heatmap
![Heatmap](images/heatmap_new.png)

---

## 📑 Full Research Report
For a detailed breakdown of the methodology, performance metrics, and full set of plots, see the complete report:  
[📄 Quant Research Report (PDF)](Quant_Research_Report.pdf)
### Key Findings
- **Cointegrated Pairs Found:** 2 (TFC‑USB, BAC‑PNC)  
- **Best Performing Pair:** TFC‑USB with a Sharpe ratio of **0.61** and CAGR of **6.3%**  
- **Second Pair:** BAC‑PNC achieved a Sharpe ratio of **0.56** and CAGR of **8.1%**
- **Portfolio Performance (Equal‑weighted):**
  - CAGR: **7.8%**
  - Annualised Mean: **~7.9%**
  - Annualised Volatility: **9.3%**
  - Sharpe Ratio: **0.85**
  - Sortino Ratio: **1.46**
  - Max Drawdown: **‑12.1%**
  - Win Rate: **28.1%**


### Interpretation
- The TFC-USB pair provided the most stable mean-reversion performance with moderate drawdowns.
- The BAC-PNC pair, while statistically cointegrated, exhibited higher volatility and drawdowns but still delivered positive returns.
- At the portfolio level, performance was more attractive than the previous iteration, with a Sharpe ratio of 0.85. However, the relatively low win rate suggests that gains were achieved through a small number of strong trades rather than frequent small wins.


### Limitations
- Transaction Costs: Not included in this backtest; actual implementation would likely reduce performance.
- Limited Universe: Only 2 cointegrated pairs were identified among 10 bank tickers; expanding to a larger universe could uncover stronger opportunities.
- Instability of Cointegration: Relationships may break down in stress periods, making ongoing recalibration essential.

### Future Work
- Incorporate more **sectors/universes** for robust pair detection.    
- Include **transaction costs and slippage** for realistic PnL estimation.

