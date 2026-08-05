"""Generative invariants for the capital-sizing boundary.

Example tests pin known cases.  These tests explore the space between them and
sequences of allocations, where an unsafe sizing regression is more likely to
hide.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from auramaur.risk.kelly import KellySizer


probabilities = st.floats(min_value=0.011, max_value=0.989,
                          allow_nan=False, allow_infinity=False)
positive_money = st.floats(min_value=1.0, max_value=1_000_000.0,
                           allow_nan=False, allow_infinity=False)
multipliers = st.floats(min_value=0.0, max_value=1.0,
                        allow_nan=False, allow_infinity=False)


@given(
    model_prob=probabilities,
    market_prob=probabilities,
    bankroll=positive_money,
    max_stake=positive_money,
    fraction=st.floats(min_value=0.0, max_value=1.0,
                       allow_nan=False, allow_infinity=False),
    confidence=multipliers,
    category=multipliers,
)
def test_kelly_size_is_finite_non_negative_and_capped(
    model_prob, market_prob, bankroll, max_stake, fraction, confidence, category,
):
    size = KellySizer(fraction=fraction).calculate(
        model_prob, market_prob, bankroll,
        confidence_mult=confidence,
        category_mult=category,
        max_stake=max_stake,
    )

    assert math.isfinite(size)
    assert 0.0 <= size <= min(bankroll, max_stake)


@given(model_prob=probabilities, market_prob=probabilities,
       bankroll=positive_money, max_stake=positive_money)
def test_yes_no_probability_mirroring_preserves_size(
    model_prob, market_prob, bankroll, max_stake,
):
    sizer = KellySizer()
    yes = sizer.calculate(model_prob, market_prob, bankroll, max_stake=max_stake)
    no = sizer.calculate(
        1.0 - model_prob, 1.0 - market_prob, bankroll, max_stake=max_stake,
    )

    assert yes == pytest.approx(no)


class KellyAllocationMachine(RuleBasedStateMachine):
    """Exercise repeated allocations as a small capital state machine."""

    def __init__(self):
        super().__init__()
        self.initial_bankroll = 10_000.0
        self.cash = self.initial_bankroll
        self.committed = 0.0
        self.sizer = KellySizer()

    @rule(model_prob=probabilities, market_prob=probabilities,
          requested_cap=st.floats(min_value=1.0, max_value=250.0,
                                  allow_nan=False, allow_infinity=False))
    def allocate(self, model_prob, market_prob, requested_cap):
        cap = min(requested_cap, self.cash)
        size = self.sizer.calculate(
            model_prob, market_prob, self.cash, max_stake=cap,
        )
        assert 0.0 <= size <= cap
        self.cash -= size
        self.committed += size

    @rule()
    def release_all_capital(self):
        self.cash += self.committed
        self.committed = 0.0

    @invariant()
    def capital_is_conserved_and_never_negative(self):
        assert self.cash >= 0.0
        assert self.committed >= 0.0
        assert self.cash + self.committed == pytest.approx(self.initial_bankroll)


TestKellyAllocation = KellyAllocationMachine.TestCase
TestKellyAllocation.settings = settings(max_examples=75, stateful_step_count=40)
