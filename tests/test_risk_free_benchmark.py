"""The risk-free hurdle: every judgement in this system was scored against
zero until 2026-08-01, which flatters anything that merely avoids losing."""
from types import SimpleNamespace

from auramaur.monitoring.ledger_report import risk_free_benchmark


def _s(rate=0.045, capital=2340.0):
    return SimpleNamespace(benchmark=SimpleNamespace(
        risk_free_annual_rate=rate, book_capital_usd=capital))


def test_benchmark_reproduces_the_measured_live_book():
    """Pins the 2026-08-01 measurement: $8.36 net over 114 days on a $2,340
    book is +1.14%/yr against 4.5% cash — behind cash, which no report said."""
    b = risk_free_benchmark(net_pnl=8.36, first_at="2026-04-08",
                            last_at="2026-07-31", settings=_s())
    assert b["available"] is True
    assert b["days"] == 114
    assert round(b["annualised"], 2) == 26.77
    assert round(b["return_pct"], 2) == 1.14
    assert b["excess_pct"] < 0                       # behind cash
    assert round(b["excess_dollars"], 0) == -79.0


def test_no_denominator_is_invented_when_book_size_is_unknown():
    """capital=0 must suppress the percentage rather than guess it.

    A wrong book size yields a confidently wrong verdict, which is worse than
    no verdict. Annualised dollars stay available — those need no denominator.
    """
    b = risk_free_benchmark(net_pnl=8.36, first_at="2026-04-08",
                            last_at="2026-07-31", settings=_s(capital=0.0))
    assert b["available"] is True
    assert "annualised" in b
    assert "return_pct" not in b
    assert "excess_pct" not in b


def test_sub_day_span_does_not_annualise():
    """Under a day of record, annualising multiplies noise by ~365."""
    b = risk_free_benchmark(net_pnl=5.0, first_at="2026-08-01",
                            last_at="2026-08-01", settings=_s())
    assert b["available"] is False


def test_a_book_beating_cash_reads_positive():
    b = risk_free_benchmark(net_pnl=500.0, first_at="2026-04-08",
                            last_at="2026-07-31", settings=_s())
    assert b["excess_pct"] > 0
    assert b["excess_dollars"] > 0
