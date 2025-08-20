# utils.py
"""
Utility functions for:
- Data fetching from Yahoo Finance
- Cointegration testing
- Pairs trading backtest
- Performance metrics
- Plotting helpers
"""

import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt


# --------------------------------------
# 1. Fetch Adjusted Close Prices
# --------------------------------------
def fetch_data(tickers, start, end, save_path=None):
    """
    Download *adjusted* close prices from Yahoo Finance for given tickers.
    Uses auto_adjust=True to ensure prices are split/dividend adjusted.
    Returns a clean DataFrame with tickers as columns.
    """
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,  # Adjusts OHLC automatically
        progress=False
    )

    if data.empty:
        raise ValueError(f"No data fetched for tickers: {tickers}. Check symbols or date range.")

    # Only keep the Close column (already adjusted)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]] if "Close" in data.columns else data

    prices = prices.dropna(how='all').fillna(method='ffill')

    # Save to CSV if requested
    if save_path:
        prices.to_csv(save_path)

    return prices


# --------------------------------------
# 2. Find Cointegrated Pairs
# --------------------------------------
def find_cointegrated_pairs(data, significance=0.05):
    """
    Tests all pairs for cointegration using Engle–Granger two-step method.
    Returns p-value matrix and a list of cointegrated pairs.
    """
    n = data.shape[1]
    keys = data.columns
    pval_matrix = np.ones((n, n))
    pairs = []

    for i in range(n):
        for j in range(i+1, n):
            _, pval, _ = coint(data[keys[i]], data[keys[j]])
            pval_matrix[i, j] = pval
            if pval < significance:
                pairs.append((keys[i], keys[j], pval))

    return pval_matrix, pairs


# --------------------------------------
# 3. Backtest Pairs Trading
# --------------------------------------
def backtest_pairs(prices, pair, entry_z=2, exit_z=0):
    """
    Simple pairs trading backtest:
    - Entry: z > +entry_z → Short spread
    - Entry: z < -entry_z → Long spread
    - Exit: |z| < exit_z → Close position
    """
    s1, s2 = prices[pair[0]], prices[pair[1]]

    # Hedge ratio via OLS
    beta = np.polyfit(s2, s1, 1)[0]

    # Spread & Z-score
    spread = s1 - beta * s2
    zscore = (spread - spread.mean()) / spread.std()

    # Trading signals
    position = np.zeros(len(spread))
    position[zscore < -entry_z] = 1      # Long spread
    position[zscore > entry_z] = -1      # Short spread
    position = pd.Series(position, index=spread.index).ffill()

    # Daily returns
    ret = position.shift(1) * (s1.pct_change() - beta * s2.pct_change())
    equity = (1 + ret.fillna(0)).cumprod()

    return equity, ret, zscore


# --------------------------------------
# 4. Performance Metrics
# --------------------------------------
def performance_metrics(returns, freq=252):
    """
    Compute Annualized Return, Volatility, Sharpe Ratio, Max Drawdown, Win Rate.
    """
    ann_return = (1 + returns.mean())**freq - 1
    ann_vol = returns.std() * np.sqrt(freq)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0

    cum_returns = (1 + returns.fillna(0)).cumprod()
    peak = cum_returns.cummax()
    max_dd = ((cum_returns - peak) / peak).min()

    win_rate = (returns > 0).sum() / len(returns)

    return {
        "Annualized Return": ann_return,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Win Rate": win_rate
    }


# --------------------------------------
# 5. Plot Z-Score Bands
# --------------------------------------
def plot_zscore(zscore, entry_z=2, exit_z=0, title="Z-score Spread"):
    plt.figure(figsize=(12,5))
    plt.plot(zscore, label="Z-score")
    plt.axhline(entry_z, color='red', linestyle='--', label='Upper Threshold')
    plt.axhline(-entry_z, color='green', linestyle='--', label='Lower Threshold')
    plt.axhline(exit_z, color='grey', linestyle='--', label='Exit Threshold')
    plt.axhline(-exit_z, color='grey', linestyle='--')
    plt.legend()
    plt.title(title)
    plt.show()