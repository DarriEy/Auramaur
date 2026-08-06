"""An internal (YES+NO) arb is only a hedge at EQUAL SHARE COUNTS.

YES+NO of one market pays exactly $1 per SHARE held on the winning side.
Sizing both legs to the same DOLLARS buys unequal shares, so the "risk-free"
package has a losing branch whenever either leg prices above 0.50 — and it was
booked and alerted as risk-free. _execute_negrisk_arb got this right and says
so ("Size all legs to a common SHARE quantity"); this path did not.
"""

import pytest


def _qty(position_size: float, yes_px: float, no_px: float) -> float:
    """The sizing rule as implemented in _execute_internal_arb."""
    return min(position_size / yes_px, position_size / no_px)


def _worst_case_pnl(qty_yes: float, qty_no: float, yes_px: float, no_px: float) -> float:
    """A binary market pays $1 per share on exactly one side."""
    outlay = qty_yes * yes_px + qty_no * no_px
    return min(qty_yes, qty_no) * 1.0 - outlay


@pytest.mark.parametrize("yes_px,no_px", [
    (0.60, 0.35),   # the reported case: sum 0.95, one leg above 0.50
    (0.70, 0.25),
    (0.55, 0.40),
    (0.45, 0.45),   # both under 0.50 — the only region equal-notional survived
])
def test_equal_share_sizing_is_never_a_guaranteed_loss(yes_px, no_px):
    stake = 10.0
    q = _qty(stake, yes_px, no_px)
    assert _worst_case_pnl(q, q, yes_px, no_px) > 0, "a real arb cannot lose"


def test_equal_notional_sizing_would_have_lost_on_the_reported_case():
    """Pin the defect this fixes: same stake, dollar-sized legs, guaranteed loss."""
    stake, yes_px, no_px = 10.0, 0.60, 0.35
    q_yes, q_no = stake / yes_px, stake / no_px      # the old behaviour
    assert q_yes != q_no
    assert _worst_case_pnl(q_yes, q_no, yes_px, no_px) < 0

    q = _qty(stake, yes_px, no_px)                    # the new behaviour
    assert _worst_case_pnl(q, q, yes_px, no_px) > 0


def test_common_quantity_never_exceeds_the_approved_stake_on_either_leg():
    stake, yes_px, no_px = 10.0, 0.60, 0.35
    q = _qty(stake, yes_px, no_px)
    assert q * yes_px <= stake + 1e-9
    assert q * no_px <= stake + 1e-9


def test_no_leg_carries_the_no_token():
    """Order.token defaults to YES. Kalshi derives the book side and the price
    inversion from order.token, not token_id, so an unset token posted two YES
    bids instead of a hedge; the gateway also writes token=order.token into
    cost_basis/fills, booking a NO holding under 'YES'."""
    import inspect
    from auramaur import bot_arb_execute

    src = inspect.getsource(bot_arb_execute.ArbTradeExecutionMixin._execute_internal_arb)
    no_leg = src.split("# NO leg", 1)[1]
    assert "token=TokenType.NO" in no_leg
