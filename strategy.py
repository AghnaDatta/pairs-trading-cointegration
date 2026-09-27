"""One research strategy: chronological screening and explicit synthetic accounts.

Amounts are USD. Prices are adjusted-price units, not historical raw share prices.
The notebook owns parameters, explanations, plots and saving; this file has no I/O.
"""

from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint


def check_prices(prices):
    """Fail visibly rather than inventing prices or silently changing the sample."""
    if prices.empty or not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Prices need a non-empty DatetimeIndex.")
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Dates must be unique and increasing.")
    if not prices.columns.is_unique:
        raise ValueError("Duplicate symbols.")
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("Missing, non-finite or non-positive prices; no filling allowed.")


def screen_pairs(formation, family_alpha=0.05, max_pairs=2):
    """Bonferroni family includes every pair, even when the I(1) gate fails."""
    check_prices(formation)
    symbols = sorted(formation.columns)
    family_size = len(symbols) * (len(symbols) - 1) // 2
    diagnostics = []
    for symbol in symbols:
        level_p = adfuller(formation[symbol], maxlag=10,
                          regression="c", autolag="AIC")[1]
        difference_p = adfuller(formation[symbol].diff().dropna(), maxlag=10,
                               regression="c", autolag="AIC")[1]
        diagnostics.append(dict(symbol=symbol, level_p=level_p,
                                difference_p=difference_p,
                                plausible_i1=level_p > 0.05 and difference_p < 0.05))
    i1 = pd.DataFrame(diagnostics).set_index("symbol")
    rows = []
    for y, x in combinations(symbols, 2):
        # One direction chosen in advance; do not test both and keep the better p.
        model = sm.OLS(formation[y], sm.add_constant(formation[x])).fit()
        alpha, beta = model.params.iloc[0], model.params.iloc[1]
        statistic, p, _ = coint(formation[y], formation[x], trend="c",
                                maxlag=10, autolag="aic")
        adjusted_p = min(float(p) * family_size, 1.0)
        plausible_i1 = bool(i1.loc[y, "plausible_i1"] and i1.loc[x, "plausible_i1"])
        # Perfectly collinear / degenerate inputs are not accepted as evidence.
        eligible = (np.isfinite(statistic) and np.isfinite(beta) and beta > 0
                    and plausible_i1 and adjusted_p <= family_alpha)
        rows.append(dict(y=y, x=x, alpha=float(alpha), beta=float(beta),
                         p_value=p, adjusted_p=adjusted_p, plausible_i1=plausible_i1,
                         eligible=eligible, selected=False))
    table = pd.DataFrame(rows)
    used = set()
    selected = 0
    for idx, row in table.sort_values(["p_value", "y", "x"]).iterrows():
        if row.eligible and selected < max_pairs and not ({row.y, row.x} & used):
            table.loc[idx, "selected"] = True
            used.update([row.y, row.x])
            selected += 1
    return table, i1.reset_index()


def make_schedule(prices, formation_days=504, trading_days=126, max_pairs=2):
    """Each fit sees only its preceding formation slice; retain an audit table."""
    check_prices(prices)
    if formation_days < 60 or trading_days < 2 or len(prices) <= formation_days:
        raise ValueError("Need >=60 formation rows and at least one evaluation row.")
    schedule, screens, diagnostics = [], [], []
    for window, start in enumerate(range(formation_days, len(prices), trading_days), 1):
        end = min(start + trading_days, len(prices))  # exclusive
        formation = prices.iloc[start - formation_days:start]
        screen, i1 = screen_pairs(formation, max_pairs=max_pairs)
        metadata = dict(window=window, formation_start=formation.index[0],
                        formation_end=formation.index[-1],
                        trading_start=prices.index[start], trading_end=prices.index[end - 1])
        screens.append(screen.assign(**metadata))
        diagnostics.append(i1.assign(**metadata))
        schedule.append(dict(**metadata, start=start, end=end,
                             models=screen.loc[screen.selected].sort_values("p_value").to_dict("records")))
    return schedule, pd.concat(screens, ignore_index=True), pd.concat(diagnostics, ignore_index=True)


def spread_zscore(prices, y, x, alpha, beta, window=60):
    """Current close is available at decision time; the order fills tomorrow."""
    spread = prices[y] - alpha - beta * prices[x]
    std = spread.rolling(window).std(ddof=1)
    z = (spread - spread.rolling(window).mean()) / std.replace(0.0, np.nan)
    return spread, z


def size_position(y_price, x_price, beta, direction, equity, gross_target=0.8):
    """Size in synthetic shares, using prices/equity known when the order is sent."""
    scale = gross_target * equity / (y_price + beta * x_price)
    return direction * scale, -direction * beta * scale


def exit_reason(direction, z, age, max_hold, exit_z, stop_z, leverage, max_gross,
                boundary=False):
    """Reason for a next-close order; age counts intervals already held."""
    if boundary:
        return "window_end"
    if leverage > max_gross:
        return "leverage"
    if not np.isfinite(z):
        return "invalid_signal"
    if (direction == 1 and z >= -exit_z) or (direction == -1 and z <= exit_z):
        return "convergence"
    if (direction == 1 and z <= -stop_z) or (direction == -1 and z >= stop_z):
        return "stop"
    if age >= max_hold - 1:
        return "holding_limit"
    return None


TRADE_COLUMNS = ["trade_id", "window", "pair", "direction", "entry_signal_date",
                 "entry_date", "exit_signal_date", "exit_date", "entry_z", "exit_z",
                 "q_y", "q_x", "entry_y", "entry_x", "exit_y", "exit_x",
                 "entry_equity", "holding_intervals", "calendar_days", "exit_reason",
                 "gross_pnl", "transaction_cost", "borrow_cost", "finance_cost", "net_pnl"]
FILL_COLUMNS = ["trade_id", "window", "pair", "signal_date", "date", "kind", "reason",
                "delta_y", "delta_x", "price_y", "price_x", "notional", "cost"]
DECISION_COLUMNS = ["window", "pair", "date", "kind", "reason", "z", "q_y", "q_x"]


def simulate_pair(prices, z, beta, capital, window=1, entry_z=2.0, exit_z=0.5,
                  stop_z=3.5, max_hold=20, gross_target=0.8, max_gross=1.0,
                  cost_bps=5.0, borrow_rate=0.01, finance_rate=0.05):
    """One segregated pair account. First row is an anchor, not a return day.

    Daily order: mark yesterday's holdings; charge carry; fill yesterday's order;
    calculate equity; make today's decision. Units stay fixed between fills.
    """
    check_prices(prices)
    if prices.shape[1] != 2 or not z.index.equals(prices.index):
        raise ValueError("Need exactly two legs and aligned z-scores.")
    if not (0 <= exit_z < entry_z < stop_z and 0 < gross_target < max_gross):
        raise ValueError("Invalid thresholds or exposure limits.")
    if min(cost_bps, borrow_rate, finance_rate) < 0 or beta <= 0 or capital <= 0 or max_hold < 1:
        raise ValueError("Invalid costs, beta, capital or holding limit.")
    y, x = prices.columns
    pair = f"{y}-{x}"
    cash, q_y, q_x = float(capital), 0.0, 0.0
    previous_equity, previous_long, previous_short = float(capital), 0.0, 0.0
    cumulative_cost = 0.0
    pending, trade = None, None
    armed, trade_number = True, 0
    daily, trades, fills, decisions = [], [], [], []
    last = len(prices) - 1

    for i, (date, row) in enumerate(prices.iterrows()):
        py, px = float(row[y]), float(row[x])
        z_now = float(z.loc[date])
        was_invested = q_y != 0 or q_x != 0
        days = (date - prices.index[i - 1]).days if i else 0
        gross_pnl = (q_y * (py - prices.iloc[i - 1, 0])
                     + q_x * (px - prices.iloc[i - 1, 1])) if i else 0.0
        borrow = previous_short * borrow_rate * days / 365.0
        # Cash includes restricted short proceeds. Free funding = equity - long.
        finance = max(previous_long - previous_equity, 0.0) * finance_rate * days / 365.0
        cash -= borrow + finance
        if trade is not None:
            trade["borrow_cost"] += borrow
            trade["finance_cost"] += finance
        turnover, transaction_cost, filled_exit = 0.0, 0.0, False

        if pending is not None:
            order = pending
            pending = None
            is_entry = order["kind"] == "entry"
            dy = order["q_y"] if is_entry else -q_y
            dx = order["q_x"] if is_entry else -q_x
            turnover = abs(dy * py) + abs(dx * px)
            transaction_cost = turnover * cost_bps / 10000.0
            # Purchases reduce cash; short sales raise cash AND create a liability.
            cash -= dy * py + dx * px + transaction_cost
            if is_entry:
                trade_number += 1
                trade = dict(trade_id=f"{window}:{pair}:{trade_number}", window=window,
                             pair=pair, direction=order["direction"],
                             entry_signal_date=order["date"], entry_date=date,
                             entry_z=order["z"], q_y=dy, q_x=dx, entry_y=py, entry_x=px,
                             entry_equity=order["equity"], transaction_cost=transaction_cost,
                             borrow_cost=0.0, finance_cost=0.0, entry_index=i)
                trade_id = trade["trade_id"]
            else:
                trade_id = trade["trade_id"]
                trade["transaction_cost"] += transaction_cost
                trade["gross_pnl"] = q_y * (py - trade["entry_y"]) + q_x * (px - trade["entry_x"])
                trade.update(exit_signal_date=order["date"], exit_date=date,
                             exit_z=order["z"], exit_y=py, exit_x=px,
                             holding_intervals=i - trade.pop("entry_index"),
                             calendar_days=(date - trade["entry_date"]).days,
                             exit_reason=order["reason"])
                trade["net_pnl"] = (trade["gross_pnl"] - trade["transaction_cost"]
                                    - trade["borrow_cost"] - trade["finance_cost"])
                trades.append(trade)
                trade, filled_exit = None, True
            q_y, q_x = q_y + dy, q_x + dx
            fills.append(dict(trade_id=trade_id, window=window, pair=pair,
                              signal_date=order["date"], date=date, kind=order["kind"],
                              reason=order["reason"], delta_y=dy, delta_x=dx,
                              price_y=py, price_x=px, notional=turnover, cost=transaction_cost))

        long_value = max(q_y * py, 0.0) + max(q_x * px, 0.0)
        short_value = -min(q_y * py, 0.0) - min(q_x * px, 0.0)
        equity = cash + long_value - short_value
        if equity <= 0:
            raise ValueError("Account insolvent; this limited research model cannot continue.")
        cumulative_cost += transaction_cost + borrow + finance
        leverage = (long_value + short_value) / equity
        daily.append(dict(date=date, window=window, pair=pair, z=z_now,
                          price_y=py, price_x=px, q_y=q_y, q_x=q_x, cash=cash,
                          long_value=long_value, short_value=short_value, equity=equity,
                          gross_equity=equity + cumulative_cost, gross_pnl=gross_pnl,
                          transaction_cost=transaction_cost, borrow_cost=borrow,
                          finance_cost=finance, turnover=turnover,
                          invested=(q_y != 0), interval_invested=was_invested,
                          gross_exposure=leverage, net_exposure=(long_value-short_value)/equity))

        # Decisions are deliberately after fills: no newly filled position earns
        # the price move into that same close. A gapped entry cannot be cancelled.
        if i < last:
            if trade is not None:
                reason = exit_reason(trade["direction"], z_now, i - trade["entry_index"],
                                     max_hold, exit_z, stop_z, leverage, max_gross,
                                     boundary=i == last - 1)
                if reason:
                    pending = dict(kind="exit", reason=reason, date=date, z=z_now,
                                   q_y=0.0, q_x=0.0)
            elif np.isfinite(z_now):
                if abs(z_now) < entry_z:
                    armed = True
                if abs(z_now) >= stop_z:
                    armed = False
                if armed and not filled_exit and entry_z <= abs(z_now) < stop_z and i < last - 1:
                    direction = 1 if z_now < 0 else -1
                    dy, dx = size_position(py, px, beta, direction, equity, gross_target)
                    pending = dict(kind="entry", reason="entry", date=date, z=z_now,
                                   direction=direction, equity=equity, q_y=dy, q_x=dx)
                    armed = False
            if pending is not None:
                decisions.append(dict(window=window, pair=pair, date=date,
                                      kind=pending["kind"], reason=pending["reason"],
                                      z=z_now, q_y=pending["q_y"], q_x=pending["q_x"]))
        previous_equity, previous_long, previous_short = equity, long_value, short_value

    assert trade is None and pending is None and q_y == 0 and q_x == 0
    return (pd.DataFrame(daily).set_index("date"), pd.DataFrame(trades, columns=TRADE_COLUMNS),
            pd.DataFrame(fills, columns=FILL_COLUMNS), pd.DataFrame(decisions, columns=DECISION_COLUMNS))


def run_walk_forward(prices, schedule, initial_capital=100_000.0, slots=2,
                     z_window=60, **trading_parameters):
    """Rebalance the flat portfolio into equal slots at each formation boundary."""
    check_prices(prices)
    capital = initial_capital
    portfolio_days, pair_days, trades, fills, decisions, periods = [], [], [], [], [], []
    sums = ["cash", "long_value", "short_value", "equity", "gross_pnl",
            "transaction_cost", "borrow_cost", "finance_cost", "turnover"]
    for period in schedule:
        start, end = period["start"], period["end"]
        if len(period["models"]) > slots:
            raise ValueError("More selected pairs than capital slots.")
        dates = prices.index[start - 1:end]
        aggregate = pd.DataFrame(0.0, index=dates, columns=sums)
        aggregate["invested"] = False
        aggregate["interval_invested"] = False
        unused = capital * (slots - len(period["models"])) / slots
        aggregate["cash"] = unused
        aggregate["equity"] = unused
        for model in period["models"]:
            y, x = model["y"], model["x"]
            history = prices.loc[period["formation_start"]:period["trading_end"], [y, x]]
            _, z = spread_zscore(history, y, x, model["alpha"], model["beta"], z_window)
            daily, ledger, orders, actions = simulate_pair(
                prices.loc[dates, [y, x]], z.loc[dates], model["beta"], capital / slots,
                window=period["window"], **trading_parameters)
            aggregate[sums] += daily[sums]
            aggregate["invested"] |= daily["invested"]
            aggregate["interval_invested"] |= daily["interval_invested"]
            pair_days.append(daily.reset_index())
            if not ledger.empty:
                trades.append(ledger)
            if not orders.empty:
                fills.append(orders)
            if not actions.empty:
                decisions.append(actions)
        aggregate["window"] = period["window"]
        active_days = aggregate.iloc[1:]
        periods.append(dict(window=period["window"], formation_end=period["formation_end"],
                            start=dates[1], end=dates[-1], observations=len(active_days),
                            pairs=", ".join(f"{m['y']}-{m['x']}" for m in period["models"]) or "Cash",
                            start_equity=capital, end_equity=aggregate.equity.iloc[-1],
                            net_return=aggregate.equity.iloc[-1] / capital - 1,
                            gross_pnl=active_days.gross_pnl.sum(),
                            turnover=active_days.turnover.sum(),
                            invested_fraction=active_days.interval_invested.mean()))
        portfolio_days.append(aggregate if not portfolio_days else active_days)
        capital = aggregate.equity.iloc[-1]
    daily = pd.concat(portfolio_days)
    daily.index.name = "date"
    daily["gross_equity"] = initial_capital + daily.gross_pnl.cumsum()
    daily["gross_exposure"] = (daily.long_value + daily.short_value) / daily.equity
    daily["net_exposure"] = (daily.long_value - daily.short_value) / daily.equity
    daily["net_return"] = daily.equity.pct_change(fill_method=None).fillna(0.0)
    daily["drawdown"] = daily.equity / daily.equity.cummax() - 1.0
    return dict(daily=daily, pair_daily=pd.concat(pair_days, ignore_index=True) if pair_days else pd.DataFrame(),
                trades=pd.concat(trades, ignore_index=True) if trades else pd.DataFrame(columns=TRADE_COLUMNS),
                fills=pd.concat(fills, ignore_index=True) if fills else pd.DataFrame(columns=FILL_COLUMNS),
                decisions=pd.concat(decisions, ignore_index=True) if decisions else pd.DataFrame(columns=DECISION_COLUMNS),
                periods=pd.DataFrame(periods))


def performance_metrics(daily, trades=None, equity_column="equity", frequency=252):
    """Equity includes one starting anchor. Benchmark and downside target are 0.

    Sharpe / Sortino are NaN if their denominator is zero; no-trade win rate is
    undefined, not zero. Monetary trade statistics use net USD P&L.
    """
    equity = daily[equity_column]
    r = equity.pct_change(fill_method=None).iloc[1:]
    n = len(r)
    if n == 0:
        raise ValueError("Need at least one return interval.")
    mean = float(r.mean() * frequency)
    volatility = float(r.std(ddof=1) * np.sqrt(frequency))
    downside = float(np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)) * np.sqrt(frequency))
    metrics = dict(total_return=float(equity.iloc[-1] / equity.iloc[0] - 1),
                   cagr=float((equity.iloc[-1] / equity.iloc[0]) ** (frequency / n) - 1),
                   annualised_mean=mean, annualised_volatility=volatility,
                   sharpe=mean / volatility if volatility > 1e-14 else np.nan,
                   sortino=mean / downside if downside > 1e-14 else np.nan,
                   max_drawdown=float((equity / equity.cummax() - 1).min()))
    if "turnover" in daily:
        metrics.update(annualised_turnover=float((daily.turnover.iloc[1:] / equity.shift(1).iloc[1:]).sum() * frequency / n),
                       traded_notional=float(daily.turnover.sum()),
                       average_gross_exposure=float(((daily.long_value + daily.short_value) / equity).iloc[1:].mean()),
                       max_gross_exposure=float(((daily.long_value + daily.short_value) / equity).iloc[1:].max()),
                       average_net_exposure=float(((daily.long_value - daily.short_value) / equity).iloc[1:].mean()),
                       time_invested=float(daily.interval_invested.iloc[1:].mean()))
    if trades is not None:
        pnl = trades.net_pnl.astype(float)
        metrics.update(completed_trades=len(trades), win_rate=float((pnl > 0).mean()),
                       average_winner_usd=float(pnl[pnl > 0].mean()),
                       average_loser_usd=float(pnl[pnl < 0].mean()),
                       average_holding_intervals=float(trades.holding_intervals.astype(float).mean()))
    return metrics


def cost_scenarios(pair_daily, portfolio_daily, scenarios):
    """Reprice the SAME baseline holdings; no resizing, new screening or retiming.

    Carry is charged within each segregated slot; its cash balance changes with
    the scenario. This is a diagnostic, not an alternative deployable strategy.
    """
    rows = []
    for name, bps, borrow_rate, finance_rate in scenarios:
        charges = pd.Series(0.0, index=portfolio_daily.index)
        if not pair_daily.empty:
            for _, group in pair_daily.groupby(["window", "pair"], sort=False):
                group = group.sort_values("date")
                anchor = group.date.iloc[0]
                scenario_capital = portfolio_daily.gross_equity.loc[anchor] - charges.loc[:anchor].sum()
                slot_fraction = group.equity.iloc[0] / portfolio_daily.equity.loc[anchor]
                equity = float(scenario_capital * slot_fraction)
                previous_long = previous_short = 0.0
                previous_date = group.date.iloc[0]
                for row in group.iloc[1:].itertuples():
                    days = (row.date - previous_date).days
                    cost = (row.turnover * bps / 10000 + previous_short * borrow_rate * days / 365
                            + max(previous_long - equity, 0) * finance_rate * days / 365)
                    equity += row.gross_pnl - cost
                    charges.loc[row.date] += cost
                    previous_long, previous_short, previous_date = row.long_value, row.short_value, row.date
        diagnostic = portfolio_daily.copy()
        diagnostic["equity"] = diagnostic.gross_equity - charges.cumsum()
        m = performance_metrics(diagnostic)
        rows.append(dict(scenario=name, cost_bps=bps, borrow_rate=borrow_rate,
                         finance_rate=finance_rate, total_cost_usd=float(charges.sum()),
                         final_equity=float(diagnostic.equity.iloc[-1]), **m))
    return pd.DataFrame(rows)
