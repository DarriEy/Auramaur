"""Pure pricing for Polymarket daily city-temperature bins (#weather strategy).

Parses "Will the highest/lowest temperature in {City} be {X}° / between {A}-{B}°
on {date}?" into a target bin, and prices it from an Open-Meteo ensemble: the
bin probability is just the fraction of members that land in the bin. Network
fetch + order placement live in the pillar; everything here is deterministic.

Rounding convention (a calibration risk to verify against real resolutions):
the stated whole-degree number is the *rounded* high/low the market resolves on,
so "be X°" => integer reading == X => [X-0.5, X+0.5); "between A-B°" => integer
reading in [A, B] => [A-0.5, B+0.5).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date


# Static geocode for the cities Polymarket runs temp markets on. Unknown city ->
# can't price (skip) rather than guess a location.
CITY_COORDS: dict[str, tuple[float, float]] = {
    "atlanta": (33.749, -84.388), "miami": (25.774, -80.194),
    "new york": (40.713, -74.006), "new york city": (40.713, -74.006),
    "nyc": (40.713, -74.006), "los angeles": (34.052, -118.244),
    "chicago": (41.878, -87.630), "houston": (29.760, -95.370),
    "london": (51.507, -0.128), "paris": (48.857, 2.353),
    "warsaw": (52.230, 21.012), "berlin": (52.520, 13.405),
    "moscow": (55.756, 37.617), "tokyo": (35.680, 139.690),
    "shanghai": (31.230, 121.474), "beijing": (39.904, 116.407),
    "guangzhou": (23.129, 113.264), "seoul": (37.567, 126.978),
    "cape town": (-33.925, 18.424), "johannesburg": (-26.205, 28.050),
    "sydney": (-33.868, 151.209), "mumbai": (19.076, 72.878),
    "delhi": (28.704, 77.102), "dubai": (25.205, 55.271),
    "singapore": (1.352, 103.820), "toronto": (43.651, -79.347),
    "mexico city": (19.433, -99.133), "são paulo": (-23.551, -46.633),
    "sao paulo": (-23.551, -46.633),
}

_RE_KIND = re.compile(r"\b(highest|lowest)\s+temperature\b", re.I)
_RE_CITY = re.compile(r"\btemperature in\s+(.+?)\s+be\b", re.I)
_RE_RANGE = re.compile(r"\bbe between\s+(-?\d+)\s*-\s*(-?\d+)\s*°?\s*([CF])\b", re.I)
_RE_EXACT = re.compile(r"\bbe\s+(-?\d+)\s*°?\s*([CF])\b", re.I)


@dataclass(frozen=True)
class TempSpec:
    kind: str          # "high" | "low"
    city: str          # normalized lower-case
    lat: float
    lon: float
    unit: str          # "C" | "F"
    lo: float          # inclusive bin lower bound (in `unit`)
    hi: float          # exclusive bin upper bound
    target: date


def parse_temp_market(question: str, target: date) -> TempSpec | None:
    """Parse a Poly temperature question into a TempSpec, or None if it isn't a
    recognised temp-bin market / the city isn't geocoded."""
    if not question:
        return None
    mk = _RE_KIND.search(question)
    mc = _RE_CITY.search(question)
    if not (mk and mc):
        return None
    kind = "high" if mk.group(1).lower() == "highest" else "low"
    city = mc.group(1).strip().lower().rstrip("?.,")
    coords = CITY_COORDS.get(city)
    if coords is None:
        return None

    rng = _RE_RANGE.search(question)
    if rng:
        a, b, unit = float(rng.group(1)), float(rng.group(2)), rng.group(3).upper()
        lo, hi = min(a, b) - 0.5, max(a, b) + 0.5
    else:
        ex = _RE_EXACT.search(question)
        if not ex:
            return None
        x, unit = float(ex.group(1)), ex.group(2).upper()
        lo, hi = x - 0.5, x + 0.5
    return TempSpec(kind=kind, city=city, lat=coords[0], lon=coords[1],
                   unit=unit, lo=lo, hi=hi, target=target)


def bin_probability(members: list[float], lo: float, hi: float) -> float | None:
    """Fraction of ensemble members landing in [lo, hi). None if no members."""
    if not members:
        return None
    return sum(1 for m in members if lo <= m < hi) / len(members)
