"""Walk-forward replay of the IBKR multi-asset book over historical bars.

Why this exists. `IBKRMultiAssetExecution.graduated()` needs 120 daily marks and
30 round trips over **180 elapsed days**, so the forward paper record cannot
answer "does this strategy have any edge?" until 2027. That bar is
pre-registered and is not moving. But the book's decision logic is entirely
deterministic — `normalized_momentum` over daily closes, `propose_multiasset_entry`
for sizing, a stop/target/momentum exit — so the same rules can be replayed over
years of history *today* and give an honest early read.

What this is NOT. It does not write `ibkr_paper_round_trips` or
`ibkr_paper_daily_marks`. Those tables are the pre-registered forward evidence
the live gate reads, and filling them with replayed history would manufacture
exactly the evidence the contract exists to demand. This module returns results
to a report and nothing else.

Honesty constraints, all load-bearing:

* **No lookahead.** Every decision on session *t* uses closes strictly before
  *t*. The momentum that triggers an entry cannot see the bar it trades on.
* **Costs are charged both ways** — the book's own `adverse_fill` slippage plus
  its commission schedule — because an uncosted momentum backtest on trending
  ETFs will always look profitable.
* **Parameters are read from the deployed config**, not fitted here. A replay
  that tunes its own thresholds measures the tuner, not the strategy.
* Daily bars carry no intraday BBO, so a spread is assumed and applied on both
  sides. It defaults to the book's own `max_spread_bps` — the worst spread the
  live book would accept — so the result errs pessimistic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from auramaur.experiments.strategies.ibkr_multiasset import (
    MultiAssetEntryInputs, MultiAssetEntryRules, propose_multiasset_entry,
)
from auramaur.risk.ibkr_math import (
    adverse_fill, annualized_volatility, normalized_momentum,
)


@dataclass(frozen=True)
class ReplayTrip:
    """One completed round trip, net of both commissions and slippage."""
    key: str
    asset_class: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    quantity: float
    gross_usd: float
    commission_usd: float
    reason: str

    @property
    def net_usd(self) -> float:
        return self.gross_usd - self.commission_usd

    @property
    def held_sessions(self) -> int:
        return 0  # filled in by the caller when session indices are known


@dataclass(frozen=True)
class ReplayResult:
    trips: tuple[ReplayTrip, ...]
    sessions: int
    first_date: str
    last_date: str
    open_at_end: int

    @property
    def net_pnls(self) -> list[float]:
        return [t.net_usd for t in self.trips]


def _equity_commission(notional_usd: float) -> float:
    """The book's STOCK/ETF schedule (IBKRMultiAssetPaperBook._commission)."""
    return max(1.0, min(notional_usd * 0.001, 10.0))


def replay_momentum_book(
    bars_by_key: dict[str, list[tuple[str, float]]],
    asset_class_by_key: dict[str, str],
    *,
    budget_usd: float,
    max_positions: int,
    max_position_pct: float,
    max_deployment_pct: float,
    risk_per_position_pct: float,
    max_asset_class_risk_pct: float,
    stop_vol_multiple: float,
    min_stop_pct: float,
    slippage_bps: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    min_norm_momentum: float,
    exit_norm_momentum: float,
    assumed_spread_bps: float,
    warmup: int = 130,
) -> ReplayResult:
    """Replay the deployed entry/exit rules session by session.

    ``bars_by_key`` maps instrument key -> ascending (date, close). Only closes
    strictly BEFORE the decision session are ever passed to the signal
    functions, which is what keeps the momentum honest.
    """
    dates = sorted({day for bars in bars_by_key.values() for day, _ in bars})
    if len(dates) <= warmup:
        return ReplayResult((), len(dates), dates[0] if dates else "",
                            dates[-1] if dates else "", 0)
    closes_by_key = {k: {d: c for d, c in v} for k, v in bars_by_key.items()}
    history: dict[str, list[float]] = {k: [] for k in bars_by_key}
    rules = MultiAssetEntryRules(
        budget_usd=budget_usd, max_position_pct=max_position_pct,
        max_deployment_pct=max_deployment_pct,
        risk_per_position_pct=risk_per_position_pct,
        max_asset_class_risk_pct=max_asset_class_risk_pct,
        stop_vol_multiple=stop_vol_multiple, min_stop_pct=min_stop_pct,
        slippage_bps=slippage_bps)

    open_pos: dict[str, dict] = {}
    trips: list[ReplayTrip] = []
    half = assumed_spread_bps / 2 / 10_000

    for index, day in enumerate(dates):
        # --- advance history with the PREVIOUS session, never today's ---
        if index:
            prior = dates[index - 1]
            for key in bars_by_key:
                close = closes_by_key[key].get(prior)
                if close:
                    history[key].append(float(close))
        if index < warmup:
            continue

        # --- exits first, on today's close, using history that excludes it ---
        for key in list(open_pos):
            close = closes_by_key[key].get(day)
            if not close:
                continue
            pos = open_pos[key]
            bid = close * (1 - half)
            gain_pct = (bid / pos["entry"] - 1) * 100
            hard = (bid <= pos["stop"] or gain_pct <= -stop_loss_pct
                    or gain_pct >= take_profit_pct)
            reason = ("stop" if bid <= pos["stop"] else
                      "stop_loss_pct" if gain_pct <= -stop_loss_pct else
                      "take_profit" if gain_pct >= take_profit_pct else "")
            if not hard:
                momentum = normalized_momentum(history[key])
                if momentum is not None and momentum <= exit_norm_momentum:
                    hard, reason = True, "momentum"
            if not hard:
                continue
            exit_price = adverse_fill(bid, close * (1 + half), "SELL", slippage_bps)
            gross = (exit_price - pos["entry"]) * pos["quantity"]
            commission = _equity_commission(exit_price * pos["quantity"])
            trips.append(ReplayTrip(
                key=key, asset_class=asset_class_by_key.get(key, "unknown"),
                entry_date=pos["date"], exit_date=day, entry_price=pos["entry"],
                exit_price=exit_price, quantity=pos["quantity"], gross_usd=gross,
                commission_usd=pos["entry_commission"] + commission,
                reason=reason))
            open_pos.pop(key)

        # --- entries, strongest momentum first, exactly as the book does ---
        if len(open_pos) >= max_positions:
            continue
        candidates = []
        for key in bars_by_key:
            if key in open_pos:
                continue
            close = closes_by_key[key].get(day)
            if not close:
                continue
            momentum = normalized_momentum(history[key])
            vol = annualized_volatility(history[key])
            if momentum is None or vol is None or momentum < min_norm_momentum:
                continue
            candidates.append((momentum, key, close, vol))
        candidates.sort(key=lambda item: -item[0])

        for _, key, close, vol in candidates:
            if len(open_pos) >= max_positions:
                break
            deployed = sum(p["entry"] * p["quantity"] for p in open_pos.values())
            asset_class = asset_class_by_key.get(key, "unknown")
            class_risk = sum(p["risk"] for p in open_pos.values()
                             if p["asset_class"] == asset_class)
            proposal = propose_multiasset_entry(MultiAssetEntryInputs(
                instrument_key=key, asset_class=asset_class,
                ask=close * (1 + half), bid=close * (1 - half),
                fx_to_usd=1.0, multiplier=1.0, annual_volatility=vol,
                deployed_usd=deployed, asset_class_risk_usd=class_risk,
                capital_per_unit_usd=close, fractional=True), rules)
            if proposal is None or proposal.quantity <= 0:
                continue
            notional = proposal.reference_price * proposal.quantity
            open_pos[key] = {
                "entry": proposal.reference_price, "quantity": proposal.quantity,
                "stop": proposal.stop_price, "date": day,
                "asset_class": asset_class, "risk": proposal.initial_risk_usd,
                "entry_commission": _equity_commission(notional),
            }

    return ReplayResult(tuple(trips), len(dates), dates[0], dates[-1],
                        len(open_pos))
