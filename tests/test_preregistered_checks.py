"""The pre-registered check counter must fire exactly when the operator's
pre-committed triggers say — a counter that under-fires quietly voids the
whole pre-registration discipline."""

from auramaur.cli.reporting import (
    _PREREGISTERED_CHECKS,
    preregistered_check_status,
)


def test_status_fires_on_either_trigger():
    st = preregistered_check_status
    assert st(30, 0.0, 30, -50.0) == "FIRED"    # settle bar reached
    assert st(45, +90.0, 30, -50.0) == "FIRED"  # bar reached while winning
    assert st(0, -50.0, 30, -50.0) == "FIRED"   # dollar floor reached
    assert st(2, -61.2, 30, -50.0) == "FIRED"   # floor overshot


def test_status_warns_near_either_trigger():
    st = preregistered_check_status
    assert st(24, 0.0, 30, -50.0) == "NEAR"     # 80% of settle bar
    assert st(0, -40.0, 30, -50.0) == "NEAR"    # 80% of the floor
    assert st(23, -39.9, 30, -50.0) == ""       # inside both
    assert st(0, 0.0, 30, -50.0) == ""
    assert st(0, +25.0, 30, -50.0) == ""        # profit never warns


def test_registry_rows_are_well_formed():
    """Epochs must parse, bars must be positive counts, floors negative
    dollars — a malformed row would make its check silently uncheckable."""
    from datetime import date

    assert _PREREGISTERED_CHECKS
    for src, venue, cat, epoch, bar, floor in _PREREGISTERED_CHECKS:
        assert src and venue and cat
        date.fromisoformat(epoch)
        assert bar > 0
        assert floor < 0
