# Pairs trading: chronological research backtest

Open **`pairs_trading.ipynb`** for the research walkthrough, equations,
assumptions, validation, results and replacement CV bullets. The notebook is the
main deliverable; `strategy.py` holds the reusable functions. No GitHub changes
have been published.

## Run

Extract the complete project ZIP and work in its `pairs_trading_research` directory.
Use **Python 3.12** (tested with 3.12.14). A virtual environment is recommended:

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on macOS/Linux, or
`.venv\Scripts\activate` in Windows Command Prompt. Install once:

```bash
python -m pip install -r requirements.txt
```

Then one command runs the whole analysis, tests and output saving:

```bash
python -m nbconvert --to notebook --execute --inplace pairs_trading.ipynb --ExecutePreprocessor.timeout=600
```

Alternatively open the notebook in a Jupyter-compatible editor, select this Python
environment, restart the kernel and run all cells. Dependencies need an initial
installation, but the analysis itself requires **no network or new market data**.
To run only the focused tests: `python -m pytest -q`.

## What running it produces

The notebook creates `results/` with screening tables, diagnostics, an account
ledger, charts, validation results and an interview revision card. This ZIP has
no generated results or notebook outputs. Use the tables to determine eligibility
and interpret performance only after running all cells.

## Fixed baseline

- Fit on 504 preceding closes, then trade 126; include the final 15-close period.
- Test 45 alphabetically oriented pairs with Engle–Granger and Bonferroni at 5%.
  ADF level/difference diagnostics assess plausible I(1); both tests use a constant,
  AIC lag selection and maximum 10 lags. Require positive beta.
- At most two disjoint pairs, ranked by p-value, in two equal capital slots.
  Empty slots hold cash. Models stay fixed within each window.
- Use 60-close z-scores; enter between 2 and 3.5 in absolute value; direction-aware
  convergence at 0.5, adverse stop at 3.5, maximum 20 holding intervals, reset before
  re-entry. Liquidate at each window end with an order scheduled one close earlier.
- Size fixed synthetic quantities at the decision close; fill at the next close.
  Target 80% gross exposure per slot; an observed 100% breach triggers next-close
  liquidation. Gaps can exceed this threshold; it is not an intraday hard cap.
- Reconcile cash plus signed holdings to equity. Reserve short proceeds; they are
  not profit. Charge 5 bps on both traded legs, 1% annual short borrowing and 5%
  debit funding, actual calendar days/365. Cash and collateral receive no interest.
  These costs are illustrative. Gross comparison keeps identical fills and units.

All parameters are in the notebook's configuration cell. The rules were specified
before the corrected baseline; the notebook gives their precise timing and priority.

## Files and evidence

| File | Purpose |
|---|---|
| `pairs_trading.ipynb` | Main blank notebook; complete workflow |
| `strategy.py` | Screening, signals, execution, accounting and metrics |
| `tests/test_strategy.py` | 15 economic and timing checks, including active synthetic trades |
| `requirements.txt` | Pinned dependencies |
| `data/prices.csv` | Byte-preserved original data |
| `data/provenance.json` | Source, hash and data limitations |
| `original_snapshot.zip` | Complete unchanged commit `4cae5e7cbce6ad2bd851693fc9e0db6c1387922e` |
| `results/` | Created on the first run: screening, ledgers, metrics, plots and interview guide |

Blank numeric CSV fields mean undefined metrics, not zero. Empty ledgers retain
column headings. Saved screen tables explain rejections; no trades are fabricated.

## Limits

Decisions are chronological on the supplied series, but the full sample was already
inspected in earlier research. This is **not a pristine holdout**. Bonferroni controls
a family within each window under valid p-values, not all repeated research over time.

The retrospective universe may have selection/survivorship bias. Adjusted prices
are used consistently as synthetic total-return units, with no added dividend cash
flow; raw fills and corporate actions cannot be reconstructed. The original downloader
forward-filled before saving, and TFC includes predecessor history around the 2019
BB&T/SunTrust merger. No independent vendor/calendar audit, borrow locates, recalls,
market impact or live margin simulation is claimed. Passing tests verifies the
specified examples and accounting, not profitability or live readiness.
