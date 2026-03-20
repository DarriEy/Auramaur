"""Backtesting engine — replays historical signals to evaluate strategy performance."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import structlog

from auramaur.db.database import Database
from auramaur.risk.kelly import KellySizer
from auramaur.strategy.signals import POLYMARKET_FEE_PCT
from config.settings import Settings

log = structlog.get_logger()


@dataclass
class BacktestResult:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    brier_score: float = 0.0
    accuracy: float = 0.0  # % of directional calls correct
    avg_edge: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    by_category: dict[str, dict] = field(default_factory=dict)
    pnl_curve: list[float] = field(default_factory=list)
    trade_details: list[dict] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades * 100

    @property
    def avg_pnl_per_trade(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl / self.total_trades


class BacktestEngine:
    """Replays historical signals against actual market resolutions."""

    def __init__(self, db: Database, settings: Settings):
        self.db = db
        self.settings = settings
        self.kelly = KellySizer(fraction=settings.kelly.fraction)

    async def run(
        self,
        days: int = 30,
        min_edge_pct: float | None = None,
        kelly_fraction: float | None = None,
        bankroll: float | None = None,
        max_stake: float | None = None,
    ) -> BacktestResult:
        """Simulate trading on historical signals.

        1. Load all signals from the last N days
        2. Load calibration resolutions (actual outcomes)
        3. For each resolved signal:
           - What did Claude predict? (claude_prob)
           - What was the market price? (market_prob)
           - What was the edge? (edge)
           - Did the market resolve YES or NO?
           - Would we have made money?
        4. Simulate Kelly-sized positions
        5. Track cumulative PnL, drawdown, Brier score
        6. Break down by category
        """
        edge_threshold = min_edge_pct if min_edge_pct is not None else self.settings.risk.min_edge_pct
        kf = kelly_fraction if kelly_fraction is not None else self.settings.kelly.fraction
        bank = bankroll if bankroll is not None else self.settings.execution.paper_initial_balance
        stake_cap = max_stake if max_stake is not None else self.settings.risk.max_stake_per_market

        sizer = KellySizer(fraction=kf)

        # Load resolved signals with their calibration outcomes
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        rows = await self.db.fetchall(
            """
            SELECT
                s.id AS signal_id,
                s.market_id,
                s.claude_prob,
                s.market_prob,
                s.edge,
                s.timestamp,
                s.action,
                c.actual_outcome,
                c.predicted_prob,
                c.resolved_at,
                c.category,
                m.question
            FROM signals s
            INNER JOIN calibration c ON s.market_id = c.market_id
            LEFT JOIN markets m ON s.market_id = m.id
            WHERE c.actual_outcome IS NOT NULL
              AND s.timestamp >= ?
            ORDER BY s.timestamp ASC
            """,
            (cutoff,),
        )

        if not rows:
            log.warning("backtest.no_data", days=days)
            return BacktestResult()

        result = BacktestResult()
        cumulative_pnl = 0.0
        peak_pnl = 0.0
        max_dd = 0.0
        pnl_returns: list[float] = []
        brier_sum = 0.0
        brier_count = 0
        correct_calls = 0
        total_edge = 0.0
        category_stats: dict[str, dict] = {}

        seen_signals: set[int] = set()

        for row in rows:
            signal_id = row["signal_id"]
            if signal_id in seen_signals:
                continue
            seen_signals.add(signal_id)

            claude_prob = row["claude_prob"]
            market_prob = row["market_prob"]
            edge_pct = row["edge"]  # stored as percentage
            actual_outcome = row["actual_outcome"]
            category = row["category"] or "unknown"
            question = row["question"] or row["market_id"]

            # Brier score: (predicted - actual)^2
            brier_sum += (claude_prob - actual_outcome) ** 2
            brier_count += 1

            # Directional accuracy: did Claude correctly predict the direction?
            predicted_yes = claude_prob > 0.5
            actual_yes = actual_outcome == 1
            if predicted_yes == actual_yes:
                correct_calls += 1

            # Check edge threshold
            if abs(edge_pct) < edge_threshold:
                continue

            # Simulate the trade
            trade_pnl = self._simulate_trade(
                claude_prob=claude_prob,
                market_prob=market_prob,
                actual_outcome=actual_outcome,
                sizer=sizer,
                bankroll=bank,
                max_stake=stake_cap,
            )

            result.total_trades += 1
            total_edge += abs(edge_pct)

            if trade_pnl > 0:
                result.winning_trades += 1
            elif trade_pnl < 0:
                result.losing_trades += 1

            cumulative_pnl += trade_pnl
            result.pnl_curve.append(cumulative_pnl)
            pnl_returns.append(trade_pnl)

            # Track best/worst
            if trade_pnl > result.best_trade:
                result.best_trade = trade_pnl
            if trade_pnl < result.worst_trade:
                result.worst_trade = trade_pnl

            # Drawdown tracking
            if cumulative_pnl > peak_pnl:
                peak_pnl = cumulative_pnl
            if peak_pnl > 0:
                dd = (peak_pnl - cumulative_pnl) / peak_pnl * 100
                if dd > max_dd:
                    max_dd = dd

            # Category breakdown
            if category not in category_stats:
                category_stats[category] = {
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "pnl": 0.0,
                    "total_edge": 0.0,
                    "brier_sum": 0.0,
                    "brier_count": 0,
                }
            cat = category_stats[category]
            cat["trades"] += 1
            cat["pnl"] += trade_pnl
            cat["total_edge"] += abs(edge_pct)
            cat["brier_sum"] += (claude_prob - actual_outcome) ** 2
            cat["brier_count"] += 1
            if trade_pnl > 0:
                cat["wins"] += 1
            elif trade_pnl < 0:
                cat["losses"] += 1

            result.trade_details.append({
                "signal_id": signal_id,
                "market_id": row["market_id"],
                "question": question[:60],
                "claude_prob": claude_prob,
                "market_prob": market_prob,
                "edge_pct": edge_pct,
                "actual_outcome": actual_outcome,
                "pnl": trade_pnl,
                "category": category,
            })

        # Finalize metrics
        result.total_pnl = cumulative_pnl
        result.max_drawdown_pct = max_dd

        if brier_count > 0:
            result.brier_score = brier_sum / brier_count
            result.accuracy = correct_calls / brier_count * 100

        if result.total_trades > 0:
            result.avg_edge = total_edge / result.total_trades

        # Sharpe ratio (annualized, assuming ~1 trade per day)
        if len(pnl_returns) >= 2:
            mean_ret = sum(pnl_returns) / len(pnl_returns)
            variance = sum((r - mean_ret) ** 2 for r in pnl_returns) / (len(pnl_returns) - 1)
            std_ret = math.sqrt(variance) if variance > 0 else 0
            if std_ret > 0:
                result.sharpe_ratio = (mean_ret / std_ret) * math.sqrt(252)

        # Category breakdown — compute averages
        for cat_name, cat in category_stats.items():
            result.by_category[cat_name] = {
                "trades": cat["trades"],
                "wins": cat["wins"],
                "losses": cat["losses"],
                "pnl": round(cat["pnl"], 2),
                "win_rate": round(cat["wins"] / cat["trades"] * 100, 1) if cat["trades"] > 0 else 0,
                "avg_edge": round(cat["total_edge"] / cat["trades"], 1) if cat["trades"] > 0 else 0,
                "brier_score": round(cat["brier_sum"] / cat["brier_count"], 4) if cat["brier_count"] > 0 else 0,
            }

        log.info(
            "backtest.complete",
            days=days,
            trades=result.total_trades,
            pnl=f"${result.total_pnl:.2f}",
            sharpe=f"{result.sharpe_ratio:.2f}",
        )

        return result

    async def compare_strategies(
        self,
        params_a: dict,
        params_b: dict,
        days: int = 30,
    ) -> dict:
        """A/B test two parameter sets on the same historical data.

        Example: compare min_edge_pct=3% vs 5%, or kelly_fraction=0.25 vs 0.40
        Returns comparison metrics for both strategies.
        """
        result_a = await self.run(days=days, **params_a)
        result_b = await self.run(days=days, **params_b)

        return {
            "strategy_a": {
                "params": params_a,
                "total_trades": result_a.total_trades,
                "total_pnl": round(result_a.total_pnl, 2),
                "win_rate": round(result_a.win_rate, 1),
                "sharpe_ratio": round(result_a.sharpe_ratio, 2),
                "max_drawdown_pct": round(result_a.max_drawdown_pct, 1),
                "brier_score": round(result_a.brier_score, 4),
                "avg_edge": round(result_a.avg_edge, 1),
                "best_trade": round(result_a.best_trade, 2),
                "worst_trade": round(result_a.worst_trade, 2),
            },
            "strategy_b": {
                "params": params_b,
                "total_trades": result_b.total_trades,
                "total_pnl": round(result_b.total_pnl, 2),
                "win_rate": round(result_b.win_rate, 1),
                "sharpe_ratio": round(result_b.sharpe_ratio, 2),
                "max_drawdown_pct": round(result_b.max_drawdown_pct, 1),
                "brier_score": round(result_b.brier_score, 4),
                "avg_edge": round(result_b.avg_edge, 1),
                "best_trade": round(result_b.best_trade, 2),
                "worst_trade": round(result_b.worst_trade, 2),
            },
            "pnl_diff": round(result_a.total_pnl - result_b.total_pnl, 2),
            "sharpe_diff": round(result_a.sharpe_ratio - result_b.sharpe_ratio, 2),
            "winner": "A" if result_a.sharpe_ratio > result_b.sharpe_ratio else "B",
        }

    def _simulate_trade(
        self,
        claude_prob: float,
        market_prob: float,
        actual_outcome: int,
        sizer: KellySizer | None = None,
        bankroll: float = 1000.0,
        max_stake: float = 25.0,
    ) -> float:
        """Simulate a single trade and return PnL.

        Logic:
        - If claude_prob > market_prob: BUY YES at market_prob price
          - Payout if YES resolves: (1 - market_prob) * size
          - Loss if NO resolves: -market_prob * size (lose the stake)
        - If claude_prob < market_prob: BUY NO at (1 - market_prob) price
          - Payout if NO resolves: market_prob * size
          - Loss if YES resolves: -(1 - market_prob) * size
        """
        if sizer is None:
            sizer = self.kelly

        edge = claude_prob - market_prob

        # Size the position using Kelly
        stake = sizer.calculate(
            claude_prob=claude_prob,
            market_prob=market_prob,
            bankroll=bankroll,
            max_stake=max_stake,
        )

        if stake <= 0:
            return 0.0

        # Apply fee
        fee = stake * (POLYMARKET_FEE_PCT / 100)

        # Model slippage: larger orders move price more
        # Assume ~0.5% slippage per $10 of order size (based on typical PM liquidity)
        slippage_pct = min(0.03, stake * 0.0005)  # Cap at 3%

        # Model partial fills: larger orders have lower fill probability
        # Orders >$15 may only partially fill
        fill_rate = min(1.0, 15.0 / max(stake, 1.0))
        effective_stake = stake * fill_rate

        if effective_stake < 0.50:
            return 0.0

        if edge > 0:
            # BUY YES — slippage worsens our entry price
            entry_price = min(0.99, market_prob * (1 + slippage_pct))
            shares = effective_stake / entry_price
            if actual_outcome == 1:
                pnl = shares * (1.0 - entry_price) - fee
            else:
                pnl = -effective_stake - fee
        else:
            # BUY NO
            no_price = 1.0 - market_prob
            if no_price <= 0:
                return 0.0
            entry_price = min(0.99, no_price * (1 + slippage_pct))
            shares = effective_stake / entry_price
            if actual_outcome == 0:
                pnl = shares * (1.0 - entry_price) - fee
            else:
                pnl = -effective_stake - fee

        return round(pnl, 4)
