"""Pure parsing + bin-pricing for Polymarket city-temperature markets."""

from __future__ import annotations

from datetime import date

from auramaur.strategy.weather_pricing import (
    bin_probability,
    parse_temp_market,
)


def test_parse_exact_celsius():
    s = parse_temp_market("Will the highest temperature in Tokyo be 23°C on June 20?", date(2026, 6, 20))
    assert s is not None
    assert s.kind == "high" and s.city == "tokyo" and s.unit == "C"
    assert (s.lo, s.hi) == (22.5, 23.5)
    assert abs(s.lat - 35.68) < 0.01


def test_parse_range_fahrenheit():
    s = parse_temp_market("Will the highest temperature in Atlanta be between 82-83°F on June 19?", date(2026, 6, 19))
    assert s is not None
    assert s.unit == "F" and (s.lo, s.hi) == (81.5, 83.5)


def test_parse_lowest():
    s = parse_temp_market("Will the lowest temperature in Miami be between 74-75°F on June 18?", date(2026, 6, 18))
    assert s is not None and s.kind == "low" and (s.lo, s.hi) == (73.5, 75.5)


def test_parse_unknown_city_returns_none():
    assert parse_temp_market("Will the highest temperature in Reykjavik be 12°C on June 19?", date(2026, 6, 19)) is None


def test_parse_non_temp_returns_none():
    assert parse_temp_market("Will Bitcoin be above 70000 on June 19?", date(2026, 6, 19)) is None


def test_bin_probability():
    members = [22.0, 22.8, 23.0, 23.2, 24.0, 29.0]  # 3 of 6 in [22.5, 23.5)
    assert abs(bin_probability(members, 22.5, 23.5) - 0.5) < 1e-9
    assert bin_probability([], 22.5, 23.5) is None
    assert bin_probability([30.0, 31.0], 22.5, 23.5) == 0.0
