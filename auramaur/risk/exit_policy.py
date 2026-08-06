"""Pure exit-policy economics shared by runtime and calibration code."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExitEconomics:
    gross_pnl_pct: float
    net_pnl_pct: float
    estimated_fees: float
    # Reported, never charged — see binary_exit_economics. Kept so calibration
    # can measure the entry leg without the exit gate paying for it.
    entry_fee: float = 0.0


def binary_exit_economics(
    *, entry_price: float, exit_price: float, size: float, fee_coefficient: float,
    is_long: bool = True,
) -> ExitEconomics:
    """Return executable SELL economics for a long binary-token position.

    Only the EXIT fee is charged, because only the exit fee is ever realized.
    ``broker/pnl.py`` books ``pnl = (fill.price - old_avg_cost) * size -
    fill.fee`` — the exit leg alone — and builds the ``pnl_ledger`` event only
    on the SELL branch, so the entry fee never reaches the ledger at all. The
    BUY branch stores ``total_cost = price * size``, making ``avg_cost``
    fee-EXCLUSIVE; nothing downstream ever recovers the entry fee.

    Charging both legs here therefore made the gate demand a cost the
    accounting does not record, holding every winner past its target by the
    entry leg's drag. The entry fee is still computed and reported, so the
    diagnostic survives without binding the decision.
    """
    cost = entry_price * size
    if cost <= 0:
        return ExitEconomics(0.0, 0.0, 0.0, 0.0)
    direction = 1.0 if is_long else -1.0
    gross = direction * (exit_price - entry_price) * size
    coefficient = max(0.0, fee_coefficient)
    # Conservative when maker/taker ancestry is unavailable: reserve the
    # executable side at the taker schedule. Calibration sees the same
    # economics.
    entry_fee = coefficient * entry_price * (1.0 - entry_price) * size
    exit_fee = coefficient * exit_price * (1.0 - exit_price) * size
    return ExitEconomics(gross / cost * 100.0, (gross - exit_fee) / cost * 100.0,
                         exit_fee, entry_fee)


def lifecycle_profit_target(
    *, base_pct: float, early_pct: float, late_pct: float,
    fraction_remaining: float | None, early_fraction: float, late_fraction: float,
) -> float:
    if fraction_remaining is None:
        return base_pct
    if fraction_remaining > early_fraction:
        return early_pct
    if fraction_remaining < late_fraction:
        return late_pct
    return base_pct


def trailing_stop_triggered(
    *, peak_pct: float, current_pct: float, activation_pct: float,
    giveback_fraction: float,
) -> bool:
    """Trailing stop, where zero in EITHER knob means DISABLED.

    Zero used to mean the opposite of off. ``activation_pct = 0`` arms the
    stop against every non-negative peak, and ``giveback_fraction = 0``
    reduces the second test to ``peak > current`` — together they sell every
    position on its first adverse tick. ``Field(gt=0)`` only converted that
    footgun into a startup crash, which is stricter than the neighbouring
    ``stop_loss_pct`` / ``profit_target_pct`` and left an operator no way to
    turn the tier off. Honour the intent instead: 0 disables.
    """
    if activation_pct <= 0 or giveback_fraction <= 0:
        return False
    return peak_pct >= activation_pct and peak_pct - current_pct > peak_pct * giveback_fraction
