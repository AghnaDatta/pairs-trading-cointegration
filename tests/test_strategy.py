"""Small economic examples, not tests that copy the backtest's implementation."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from strategy import (check_prices, cost_scenarios, make_schedule, performance_metrics,
                      run_walk_forward, screen_pairs, simulate_pair, spread_zscore)


def example(z, y=None, x=None, **parameters):
    dates = pd.bdate_range("2020-01-02", periods=len(z))
    p = pd.DataFrame({"Y": y if y is not None else np.full(len(z), 40.0),
                      "X": x if x is not None else np.full(len(z), 160.0)}, index=dates)
    defaults = dict(beta=0.25, capital=10_000, cost_bps=0, borrow_rate=0, finance_rate=0)
    defaults.update(parameters)
    return simulate_pair(p, pd.Series(z, index=dates), **defaults)


def test_share_hedge_zero_pnl_when_both_legs_rise_one_percent():
    daily, trades, _, _ = example([-2.5, -2.5, 0, 0],
                                  y=[40, 40, 40.4, 40.4], x=[160, 160, 161.6, 161.6])
    assert trades.gross_pnl.iloc[0] == pytest.approx(0, abs=1e-9)
    assert daily.equity.iloc[-1] == pytest.approx(10_000)
    assert 0.01 - 0.25 * 0.01 == pytest.approx(0.0075)  # the original wrong return


def test_price_units_and_inverse_holdings_preserve_economics():
    z = [-2.5, -2.5, 0, 0]
    a, ta, _, _ = example(z, y=[40, 41, 42, 43], x=[160, 161, 162, 163],
                         cost_bps=5, borrow_rate=0.01)
    b, tb, _, _ = example(z, y=np.array([40, 41, 42, 43]) * 100,
                         x=np.array([160, 161, 162, 163]) * 10, beta=2.5,
                         cost_bps=5, borrow_rate=0.01)
    np.testing.assert_allclose(a.equity, b.equity)
    np.testing.assert_allclose(a.q_y, b.q_y * 100)
    np.testing.assert_allclose(a.q_x, b.q_x * 10)
    assert ta.net_pnl.iloc[0] == pytest.approx(tb.net_pnl.iloc[0])


def test_no_price_profit_before_fill_and_exit_earns_last_interval():
    daily, trades, fills, _ = example([-2.5, -2.5, 0, 0],
                                     y=[40, 50, 51, 52], max_gross=2)
    assert daily.gross_pnl.iloc[1] == 0  # the move 40 -> 50 occurred before entry
    assert daily.gross_pnl.iloc[2] == pytest.approx(100)  # 100 units * $1
    assert daily.gross_pnl.iloc[3] == pytest.approx(100)  # held until exit close
    assert trades.gross_pnl.iloc[0] == pytest.approx(200)
    assert (fills.date > fills.signal_date).all()


@pytest.mark.parametrize("sign", [1, -1])
def test_convergence_jump_across_entire_band(sign):
    _, trades, _, _ = example(np.array([-2.5, -2.5, 1.5, 1.5, 1.5]) * sign)
    assert trades.exit_reason.tolist() == ["convergence"]
    assert trades.holding_intervals.tolist() == [2]


def test_stop_is_next_close_and_requires_reset_before_reentry():
    daily, trades, fills, _ = example([-2.5, -2.5, -4, -4, -2.5, -2.5, 0,
                                     -2.5, -2.5, 0, 0, 0])
    assert trades.exit_reason.iloc[0] == "stop"
    assert trades.exit_signal_date.iloc[0] == daily.index[2]
    assert trades.exit_date.iloc[0] == daily.index[3]
    assert fills.loc[fills.kind == "entry", "date"].tolist() == [daily.index[1], daily.index[8]]


def test_extreme_flat_signal_cannot_trigger_entry_until_reset():
    daily, _, fills, _ = example([-4, -2.5, -2.5, 0, -2.5, -2.5, 0, 0])
    assert fills.loc[fills.kind == "entry", "date"].tolist() == [daily.index[5]]


def test_entry_order_is_not_cancelled_by_unknown_fill_close():
    daily, trades, fills, _ = example([-2.5, -4, -4, -4, -4])
    assert fills.kind.tolist() == ["entry", "exit"]
    assert fills.date.tolist() == [daily.index[1], daily.index[2]]
    assert trades.exit_reason.tolist() == ["stop"]


def test_holding_limit_and_prescheduled_boundary():
    daily, trades, _, _ = example([-2.5] * 8, max_hold=3)
    assert trades.holding_intervals.tolist() == [3]
    assert trades.exit_signal_date.iloc[0] == daily.index[3]
    assert trades.exit_date.iloc[0] == daily.index[4]
    _, boundary, _, _ = example([-2.5] * 5, max_hold=20)
    assert boundary.exit_reason.tolist() == ["window_end"]
    assert boundary.holding_intervals.tolist() == [3]


def test_cash_short_proceeds_costs_borrow_and_equity_reconcile():
    daily, trades, fills, _ = example([-2.5, -2.5, 0, 0], cost_bps=10, borrow_rate=0.01)
    # Entry: +100 Y at $40, -25 X at $160. $4k short cash is a liability too.
    assert daily.cash.iloc[1] == pytest.approx(9992)
    assert daily.equity.iloc[1] == pytest.approx(9992)
    assert fills.notional.tolist() == pytest.approx([8000, 8000])
    assert fills.cost.sum() == pytest.approx(16)
    # Fri -> Mon -> Tue: four calendar days, charged on $4,000 short notional.
    assert daily.borrow_cost.sum() == pytest.approx(4000 * .01 * 4 / 365)
    np.testing.assert_allclose(daily.equity, daily.cash + daily.q_y*daily.price_y + daily.q_x*daily.price_x)
    np.testing.assert_allclose(daily.equity.diff().iloc[1:],
                              (daily.gross_pnl-daily.transaction_cost-daily.borrow_cost-daily.finance_cost).iloc[1:])
    assert daily.equity.iloc[-1] - 10_000 == pytest.approx(trades.net_pnl.sum())
    assert trades.gross_pnl.sum() == pytest.approx(daily.gross_pnl.sum())


def test_debit_financing_and_exposure_liquidation_after_gap():
    # A short X leg jumps to $480 at entry. The next close observes leverage >1.
    daily, trades, _, _ = example([-2.5] * 5, y=[40, 40, 40, 40, 40],
                                  x=[160, 480, 480, 480, 480], finance_rate=.05)
    assert trades.exit_reason.tolist() == ["leverage"]
    assert daily.gross_exposure.iloc[1] > 1
    # A 160% target in this stress fixture creates debit funding; never baseline.
    funded, _, _, _ = example([-2.5, -2.5, 0, 0], beta=.01, gross_target=1.6,
                              max_gross=3, finance_rate=.05)
    debit = funded.long_value.iloc[1] - funded.equity.iloc[1]
    assert funded.finance_cost.iloc[2] == pytest.approx(debit * .05 * 3 / 365)


def test_no_pairs_and_no_signals_hold_cash():
    dates = pd.bdate_range("2020-01-02", periods=65)
    p = pd.DataFrame({"A": np.arange(65)+100, "B": np.arange(65)+200}, index=dates)
    schedule = [dict(window=1, start=60, end=65, formation_start=dates[0],
                     formation_end=dates[59], trading_end=dates[-1], models=[])]
    result = run_walk_forward(p, schedule)
    assert (result["daily"].equity == 100_000).all()
    assert result["trades"].empty and result["fills"].empty
    metrics = performance_metrics(result["daily"], result["trades"])
    assert metrics["cagr"] == 0 and metrics["time_invested"] == 0
    assert np.isnan(metrics["sharpe"]) and np.isnan(metrics["win_rate"])
    daily, trades, fills, _ = example([0] * 6, cost_bps=5, borrow_rate=.01)
    assert trades.empty and fills.empty and (daily.equity == 10_000).all()


def test_metrics_against_hand_calculation_including_initial_drawdown():
    r = np.array([-.1, .2, -.05])
    equity = 100 * np.cumprod(np.r_[1, 1+r])
    m = performance_metrics(pd.DataFrame({"equity": equity}), frequency=3)
    assert m["cagr"] == pytest.approx(.026)
    assert m["annualised_mean"] == pytest.approx(.05)
    assert m["annualised_volatility"] == pytest.approx(np.std(r, ddof=1)*np.sqrt(3))
    assert m["sortino"] == pytest.approx(.05/np.sqrt(.1**2+.05**2))
    assert m["max_drawdown"] == pytest.approx(-.1)
    ledger = pd.DataFrame({"net_pnl": [100, -50, 0], "holding_intervals": [1, 3, 2]})
    trade_metrics = performance_metrics(pd.DataFrame({"equity": equity}), ledger, frequency=3)
    assert trade_metrics["win_rate"] == pytest.approx(1/3)
    assert trade_metrics["average_winner_usd"] == 100
    assert trade_metrics["average_loser_usd"] == -50
    assert trade_metrics["average_holding_intervals"] == 2


def test_future_prices_cannot_change_earlier_models_decisions_or_fills():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2019-01-02", periods=300)
    x = 100 + np.cumsum(rng.normal(0, 1, len(dates)))
    p = pd.DataFrame({"A": 20 + .6*x + rng.normal(0, .7, len(dates)), "B": x}, index=dates)
    # Make the family size the actual ten-stock, 45-pair setting.
    for j in range(8):
        p[f"C{j}"] = 100 + np.cumsum(rng.normal(0, 1, len(dates)))
    cutoff = dates[201]
    future = p.copy()
    future.loc[future.index > cutoff] *= rng.uniform(.7, 1.3, (98, 10))
    s1, a1, _ = make_schedule(p, 120, 60)
    s2, a2, _ = make_schedule(future, 120, 60)
    assert_frame_equal(a1.loc[a1.formation_end <= cutoff], a2.loc[a2.formation_end <= cutoff])
    b1 = run_walk_forward(p, s1, z_window=20)
    b2 = run_walk_forward(future, s2, z_window=20)
    daily = b1["daily"]
    np.testing.assert_allclose(daily.equity, daily.cash + daily.long_value - daily.short_value)
    assert daily.equity.iloc[-1] - daily.equity.iloc[0] == pytest.approx(b1["trades"].net_pnl.sum())
    assert len(b1["fills"].loc[b1["fills"].date <= cutoff]) > 0
    assert_frame_equal(b1["daily"].loc[:cutoff], b2["daily"].loc[:cutoff])
    for name in ["fills", "decisions"]:
        assert_frame_equal(b1[name].loc[b1[name].date <= cutoff], b2[name].loc[b2[name].date <= cutoff])
    for _, frame in a1.groupby("window"):
        assert len(frame) == 45
        np.testing.assert_allclose(frame.adjusted_p, np.minimum(frame.p_value * 45, 1))
        selected = frame.loc[frame.selected]
        assert len(set(selected.y) | set(selected.x)) == 2 * len(selected)
    scenarios = cost_scenarios(b1["pair_daily"], b1["daily"], [("baseline", 5, .01, .05), ("gross", 0, 0, 0)])
    assert scenarios.final_equity.iloc[0] == pytest.approx(b1["daily"].equity.iloc[-1])
    assert scenarios.final_equity.iloc[1] == pytest.approx(b1["daily"].gross_equity.iloc[-1])


def test_missing_and_duplicate_prices_fail_loudly():
    p = pd.DataFrame({"A": [1., np.nan]}, index=pd.date_range("2020-01-01", periods=2))
    with pytest.raises(ValueError, match="no filling"):
        check_prices(p)
    p.iloc[1] = 1
    p.index = [p.index[0], p.index[0]]
    with pytest.raises(ValueError, match="unique"):
        check_prices(p)
