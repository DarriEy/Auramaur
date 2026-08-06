"""Pure exit-policy economics shared by runtime and calibration code."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExitEconomics:
    gross_pnl_pct: float
    net_pnl_pct: float
    estimated_fees: float


def binary_exit_economics(
    *, entry_price: float, exit_price: float, size: float, fee_coefficient: float,
    is_long: bool = True,
) -> ExitEconomics:
    """Return executable SELL economics for a long binary-token position."""
    cost = entry_price * size
    if cost <= 0:
        return ExitEconomics(0.0, 0.0, 0.0)
    direction = 1.0 if is_long else -1.0
    gross = direction * (exit_price - entry_price) * size
    coefficient = max(0.0, fee_coefficient)
    # Conservative when maker/taker ancestry is unavailable: reserve both
    # sides at the taker schedule. Calibration sees the same economics.
    entry_fee = coefficient * entry_price * (1.0 - entry_price) * size
    exit_fee = coefficient * exit_price * (1.0 - exit_price) * size
    fee = entry_fee + exit_fee
    return ExitEconomics(gross / cost * 100.0, (gross - fee) / cost * 100.0, fee)


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
    return peak_pct >= activation_pct and peak_pct - current_pct > peak_pct * giveback_fraction
