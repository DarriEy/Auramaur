"""The pre-registered check counter must fire exactly when the operator's
pre-committed triggers say — a counter that under-fires quietly voids the
whole pre-registration discipline."""

from auramaur.cli.reporting import (
    _CALENDAR_REVIEWS,
    _PREREGISTERED_CHECKS,
    preregistered_check_status,
)


def test_calendar_reviews_are_well_formed():
    """A malformed review date would make its judgment day silently
    unreachable — the exact slippage the calendar exists to prevent."""
    from datetime import date

    assert _CALENDAR_REVIEWS
    for due, what in _CALENDAR_REVIEWS:
        assert date.fromisoformat(due) > date(2026, 8, 1)
        assert what


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


def test_calendar_checkpoint_outranks_near_but_not_fired():
    """A cell whose entries settle years out is adjudicated by the calendar
    checkpoint, not the settlement counter — but a crossed numeric trigger
    still outranks the calendar."""
    st = preregistered_check_status
    assert st(0, 0.0, 30, -50.0, review_due=True) == "REVIEW"
    assert st(24, 0.0, 30, -50.0, review_due=True) == "REVIEW"   # beats NEAR
    assert st(0, -50.0, 30, -50.0, review_due=True) == "FIRED"   # loses to FIRED
    assert st(0, 0.0, 30, -50.0, review_due=False) == ""


def test_registry_rows_are_well_formed():
    """Epochs and review dates must parse, bars must be positive counts,
    floors negative dollars — a malformed row would make its check silently
    uncheckable."""
    from datetime import date

    assert _PREREGISTERED_CHECKS
    for src, venue, cat, epoch, bar, floor, review_by in _PREREGISTERED_CHECKS:
        assert src and venue and cat
        date.fromisoformat(epoch)
        assert bar > 0
        assert floor < 0
        if review_by is not None:
            assert date.fromisoformat(review_by) > date.fromisoformat(epoch)
