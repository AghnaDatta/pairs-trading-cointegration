# Interview revision card

Result: 0 completed trades across 12 windows;
net CAGR 0.00%. No pair passed the fixed screening policy.
Sharpe, Sortino and win rate are undefined in this all-cash baseline.

## Explain in order
1. Economic idea: relative bank valuations might revert, but common sector exposure is not proof.
2. Statistics: correlation is co-movement; cointegration is a stationary combination of I(1) levels.
3. Chronology: fit on 504 preceding observations, trade the next 126; prior full-sample exposure remains.
4. Hedge: price-level beta is X units per Y unit; P&L is qY*dY + qX*dX.
5. Timing: decide at t close, fill at t+1 close, earn price P&L only afterwards.
6. Account: short proceeds create a liability; equity is cash plus signed holdings.
7. Costs: 5 bps each traded leg, 1% borrow, 5% debit funding; illustrative, not observed.
8. Evidence: 15 tests exercise active trading; the historical baseline itself never trades.
9. Conclusion: report insufficient evidence under this policy, not a successful zero-risk strategy.

## Likely follow-ups
- Why no trades? Minimum formation p-value 0.004335 exceeds 0.05/45 = 0.001111.
- Is Bonferroni too strict? It is conservative but fixed; it controls a per-window family under valid tests.
- Is it out of sample? Each decision uses prior data, but the researcher already inspected the whole history.
- Is it market-neutral? No; a long/short hedge can retain dollar, factor, sector and structural risk.
- Does zero net value mean no capital? No; both gross legs, margin and losses need capital.
- Why adjusted prices? Preserved source availability; synthetic units avoid mixing dividend conventions.
- Why are Sharpe and win rate blank? Zero volatility and no completed trades make them undefined.
- What remains unresolved? Point-in-time universe/actions, true fills, locates, recalls and untouched evaluation data.

## Honest ownership
The original project and this rebuild used AI assistance. Explain and defend the logic
you understand; do not claim independent implementation expertise until you can trace it.
The replacement CV bullets are in notebook section 9.

## Practice
Start with: what does cointegration establish that high return correlation does not?
Answer one question at a time, then connect your answer to one equation and one code function.
