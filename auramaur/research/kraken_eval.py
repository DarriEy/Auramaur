"""Deterministic, portfolio-faithful Kraken research primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import sqrt
from statistics import mean, median, pstdev
from typing import Callable


@dataclass(frozen=True)
class Bar:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    bid: float | None = None
    ask: float | None = None
    event_score: float = 0.0
    order_imbalance: float = 0.0


@dataclass(frozen=True)
class CostModel:
    fee_bps: float = 26.0
    slippage_bps: float = 5.0
    spread_bps: float = 6.0
    stress_multiplier: float = 1.0

    def executable(self, bar: Bar, side: str) -> float:
        half = self.spread_bps / 20_000
        px = bar.ask if side == "buy" else bar.bid
        if px is None:
            px = bar.close * (1 + half if side == "buy" else 1 - half)
        slip = self.slippage_bps / 10_000 * self.stress_multiplier
        return px * (1 + slip if side == "buy" else 1 - slip)

    def fee(self, notional: float) -> float:
        return notional * self.fee_bps / 10_000 * self.stress_multiplier


@dataclass
class Decision:
    enter: bool = False
    exit: bool = False
    score: float = 0.0
    reason: str = ""


@dataclass
class Position:
    pair: str
    qty: float
    entry: float
    entry_fee: float
    opened: int
    peak: float


Signal = Callable[[str, list[Bar], int, Position | None], Decision]
SignalFactory = Callable[[dict[str, list[Bar]], dict[str, list[Bar]]], Signal]


@dataclass
class Trade:
    pair: str
    opened: int
    closed: int
    entry: float
    exit: float
    pnl: float
    return_pct: float
    reason: str


@dataclass
class Metrics:
    strategy: str
    trades: int
    win_rate: float
    net_pnl: float
    return_pct: float
    expectancy: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe: float
    turnover: float
    time_in_market_pct: float
    benchmark_return_pct: float
    excess_return_pct: float
    worst_trade: float
    by_pair: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class PortfolioEvaluator:
    def __init__(self, initial_cash=60.0, slot_usd=30.0, costs: CostModel | None = None):
        self.initial_cash = initial_cash
        self.slot_usd = slot_usd
        self.costs = costs or CostModel()

    def run(self, strategy: str, data: dict[str, list[Bar]], signal: Signal,
            trade_start_ts: int | None = None) -> Metrics:
        self._validate(data)
        if not data:
            return self._metrics(strategy, [], [self.initial_cash], 0, 0, data, [])
        timeline = sorted({b.ts for bars in data.values() for b in bars})
        indexes = {p: {b.ts: i for i, b in enumerate(bars)} for p, bars in data.items()}
        start = timeline[0] if trade_start_ts is None else trade_start_ts
        cash, turnover, occupied = self.initial_cash, 0.0, 0
        positions: dict[str, Position] = {}
        trades: list[Trade] = []
        equity = [cash]
        last_marks: dict[str, float] = {}

        for ts in timeline:
            trading = ts >= start
            for pair, idx in indexes.items():
                if ts in idx:
                    last_marks[pair] = data[pair][idx[ts]].close
            if trading:
                for pair in list(positions):
                    i = indexes[pair].get(ts)
                    if i is None:
                        continue
                    bar, pos = data[pair][i], positions[pair]
                    pos.peak = max(pos.peak, bar.high)
                    decision = signal(pair, data[pair], i, pos)
                    if decision.exit:
                        cash, trade, value = self._close(cash, pos, bar,
                                                         decision.reason or "signal")
                        turnover += value
                        trades.append(trade)
                        del positions[pair]

                candidates = []
                for pair, bars in data.items():
                    i = indexes[pair].get(ts)
                    if i is None or pair in positions:
                        continue
                    decision = signal(pair, bars, i, None)
                    if decision.enter:
                        candidates.append((decision.score, pair, i))
                for _, pair, i in sorted(candidates, reverse=True):
                    if cash < min(self.slot_usd, 5.0):
                        break
                    bar, spend = data[pair][i], min(self.slot_usd, cash)
                    px, fee = self.costs.executable(bar, "buy"), self.costs.fee(spend)
                    qty = max(0.0, (spend - fee) / px)
                    if qty:
                        cash -= spend
                        turnover += spend
                        positions[pair] = Position(pair, qty, px, fee, ts, bar.high)

            if trading:
                marked = cash + sum(pos.qty * last_marks.get(pair, pos.entry)
                                    for pair, pos in positions.items())
                equity.append(marked)
                occupied += bool(positions)

        for pair, pos in list(positions.items()):
            cash, trade, value = self._close(cash, pos, data[pair][-1], "end_of_window")
            turnover += value
            trades.append(trade)
        equity.append(cash)
        trading_periods = sum(ts >= start for ts in timeline)
        trading_timeline = [ts for ts in timeline if ts >= start]
        return self._metrics(strategy, trades, equity, turnover, occupied, data,
                             trading_timeline, trading_periods, start)

    @staticmethod
    def _validate(data):
        for pair, bars in data.items():
            if not bars:
                raise ValueError(f"{pair}: empty bar series")
            if any(b.close <= 0 or b.high < b.low for b in bars):
                raise ValueError(f"{pair}: invalid OHLC data")
            if any(bars[i].ts >= bars[i + 1].ts for i in range(len(bars) - 1)):
                raise ValueError(f"{pair}: timestamps must be strictly increasing")

    def _close(self, cash, pos, bar, reason):
        px = self.costs.executable(bar, "sell")
        gross, basis = pos.qty * px, pos.qty * pos.entry + pos.entry_fee
        proceeds = gross - self.costs.fee(gross)
        pnl = proceeds - basis
        return cash + proceeds, Trade(pos.pair, pos.opened, bar.ts, pos.entry, px, pnl,
                                      pnl / basis * 100 if basis else 0, reason), gross

    def _benchmark(self, data, start_ts=None):
        """Equal-dollar buy-and-hold, charged the same entry/exit costs."""
        returns = []
        for bars in data.values():
            eligible = [b for b in bars if start_ts is None or b.ts >= start_ts]
            if not eligible:
                continue
            buy = self.costs.executable(eligible[0], "buy")
            allocation = 1.0
            qty = (allocation - self.costs.fee(allocation)) / buy
            gross = qty * self.costs.executable(eligible[-1], "sell")
            returns.append(gross - self.costs.fee(gross) - allocation)
        return mean(returns) * 100 if returns else 0.0

    def _metrics(self, strategy, trades, equity, turnover, occupied, data,
                 timeline, trading_periods=0, start_ts=None):
        pnls = [t.pnl for t in trades]
        wins, losses = [x for x in pnls if x > 0], [-x for x in pnls if x < 0]
        peak, max_dd = equity[0], 0.0
        for value in equity:
            peak = max(peak, value)
            max_dd = max(max_dd, (peak - value) / peak * 100 if peak else 0)
        rets = [equity[i] / equity[i-1] - 1 for i in range(1, len(equity)) if equity[i-1]]
        intervals = [timeline[i+1] - timeline[i] for i in range(len(timeline)-1)]
        seconds = median(intervals) if intervals else 3600
        annual = 365.25 * 86400 / max(seconds, 1)
        sharpe = mean(rets) / pstdev(rets) * sqrt(annual) if len(rets) > 1 and pstdev(rets) else 0
        net, benchmark = sum(pnls), self._benchmark(data, start_ts)
        by_pair = {p: {"trades": len(xs := [t for t in trades if t.pair == p]),
                       "pnl": round(sum(t.pnl for t in xs), 4)} for p in data}
        ret = net / self.initial_cash * 100
        return Metrics(strategy, len(trades), len(wins)/len(trades)*100 if trades else 0,
                       net, ret, mean(pnls) if pnls else 0,
                       sum(wins)/sum(losses) if losses else (float("inf") if wins else 0),
                       max_dd, sharpe, turnover,
                       occupied/trading_periods*100 if trading_periods else 0,
                       benchmark, ret-benchmark, min(pnls) if pnls else 0, by_pair)


def _ret(bars, i, n):
    return bars[i].close / bars[i-n].close - 1 if i >= n and bars[i-n].close else 0.0


def llm_trend_signal(probabilities, minimum=.60, exit_below=.45, trend=72):
    def signal(pair, bars, i, position):
        p = probabilities.get((pair, bars[i].ts), .5)
        sma = mean(b.close for b in bars[max(0, i-trend+1):i+1])
        return (Decision(exit=p < exit_below, reason="llm_bearish") if position else
                Decision(enter=p >= minimum and bars[i].close > sma, score=p,
                         reason="llm_trend"))
    return signal


def relative_strength_signal(universe, lookback=72):
    maps = {p: {b.ts: i for i, b in enumerate(xs)} for p, xs in universe.items()}
    def signal(pair, bars, i, position):
        ts, ranked = bars[i].ts, []
        for candidate, xs in universe.items():
            j = maps[candidate].get(ts)
            if j is not None and j >= lookback:
                ranked.append((_ret(xs, j, lookback), candidate))
        leader = max(ranked)[1] if ranked else None
        strength = _ret(bars, i, lookback)
        return Decision(exit=position is not None and leader != pair,
                        enter=position is None and leader == pair and strength > 0,
                        score=strength, reason="rank_change")
    return signal


def fit_residual_betas(training, benchmark_pair="XBTUSDC", lookback=1):
    benchmark = training.get(benchmark_pair, [])
    benchmark_returns = {benchmark[i].ts: _ret(benchmark, i, lookback)
                         for i in range(lookback, len(benchmark))}
    betas = {}
    for pair, bars in training.items():
        observations = [(_ret(bars, i, lookback), benchmark_returns[bars[i].ts])
                        for i in range(lookback, len(bars))
                        if bars[i].ts in benchmark_returns]
        y = [pair_ret for pair_ret, _ in observations]
        x = [market_ret for _, market_ret in observations]
        mx, my = (mean(x), mean(y)) if x else (0, 0)
        var = sum((v-mx)**2 for v in x)
        betas[pair] = sum((a-mx)*(b-my) for a, b in zip(x, y)) / var if var else 1.0
    return betas


def residual_mean_reversion_signal(universe, betas, benchmark_pair="XBTUSDC",
                                   lookback=24, entry=.035, exit_band=.005):
    benchmark = universe[benchmark_pair]
    bench_idx = {b.ts: i for i, b in enumerate(benchmark)}
    def signal(pair, bars, i, position):
        j = bench_idx.get(bars[i].ts)
        if pair == benchmark_pair or j is None or min(i, j) < lookback:
            return Decision()
        residual = _ret(bars, i, lookback) - betas.get(pair, 1.0) * _ret(benchmark, j, lookback)
        return Decision(enter=position is None and residual <= -entry, score=-residual,
                        exit=position is not None and residual >= -exit_band,
                        reason="residual_normalized")
    return signal


def volatility_breakout_signal(lookback=48, atr_n=24, stop_atr=2.0):
    def signal(pair, bars, i, position):
        if i < max(lookback, atr_n):
            return Decision()
        atr = mean(b.high-b.low for b in bars[i-atr_n:i])
        if position:
            return Decision(exit=bars[i].low <= position.peak-stop_atr*atr, reason="atr_trail")
        breakout = bars[i].close > max(b.high for b in bars[i-lookback:i])
        return Decision(enter=breakout, score=atr/bars[i].close, reason="breakout")
    return signal


def event_confirmation_signal(min_event=.7, confirm_return=.005):
    def signal(pair, bars, i, position):
        confirmed = i > 0 and _ret(bars, i, 1) >= confirm_return
        return Decision(enter=position is None and bars[i].event_score >= min_event and confirmed,
                        score=bars[i].event_score,
                        exit=position is not None and bars[i].event_score < 0,
                        reason="event_reversed")
    return signal


def passive_liquidity_signal(min_imbalance=.25):
    """Research hook only; production evaluation requires queue-aware replay."""
    def signal(pair, bars, i, position):
        x = bars[i].order_imbalance
        return Decision(enter=position is None and x >= min_imbalance, score=x,
                        exit=position is not None and x <= 0, reason="imbalance_reversed")
    return signal


def graduation(metrics, elapsed_days, *, folds=None, stressed=None, regimes=None,
               holdout=None, min_trades=50, min_days=90):
    pair_pnls = [abs(v["pnl"]) for v in metrics.by_pair.values()]
    concentration = max(pair_pnls)/sum(pair_pnls) if sum(pair_pnls) else 1.0
    folds, regimes = folds or [], regimes or {}
    checks = {
        "sample": metrics.trades >= min_trades,
        "duration": elapsed_days >= min_days,
        "positive_expectancy": metrics.expectancy > 0,
        "profitable_after_costs": metrics.net_pnl > 0,
        "beats_benchmark": metrics.excess_return_pct > 0,
        "drawdown": metrics.max_drawdown_pct <= 20,
        "pair_concentration": concentration <= .70,
        "walk_forward_stability": len(folds) >= 3 and sum(m.net_pnl > 0 for m in folds) > len(folds)/2,
        "stressed_costs": stressed is not None and stressed.net_pnl > 0,
        "regime_breadth": len(regimes) >= 2 and all(m.net_pnl > 0 for m in regimes.values()),
        "untouched_holdout": holdout is not None and holdout.net_pnl > 0 and holdout.excess_return_pct > 0,
    }
    eligible = all(checks.values())
    return {"eligible": eligible, "checks": checks,
            "next_allocation_usd": 5.0 if eligible else 0.0}


def walk_forward(data, signal_factory: SignalFactory, train_bars, test_bars,
                 evaluator, name, warmup_bars=72):
    """Fit on train only; evaluate test with non-tradable historical warm-up."""
    n = min(len(v) for v in data.values())
    out, start = [], train_bars
    while start+test_bars <= n:
        train = {p: bars[start-train_bars:start] for p, bars in data.items()}
        context = {p: bars[max(0, start-warmup_bars):start+test_bars]
                   for p, bars in data.items()}
        test_start = min(xs[start].ts for xs in data.values())
        signal = signal_factory(train, context)
        out.append(evaluator.run(name, context, signal, trade_start_ts=test_start))
        start += test_bars
    return out
