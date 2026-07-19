"""Leakage-resistant walk-forward helpers for strategy research."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class WalkForwardFold:
    train: tuple
    test: tuple


def event_unique(rows: Sequence[T], event_key: Callable[[T], str]) -> list[T]:
    """Keep the earliest observation per independent event."""
    seen: set[str] = set()
    out: list[T] = []
    for row in rows:
        key = event_key(row)
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def expanding_walk_forward(
    rows: Sequence[T], *, timestamp: Callable[[T], datetime],
    event_key: Callable[[T], str], min_train: int = 100,
    test_size: int = 25,
) -> list[WalkForwardFold]:
    """Chronological expanding-window folds with event-disjoint test sets."""
    if min_train < 1 or test_size < 1:
        raise ValueError("min_train and test_size must be positive")
    ordered = event_unique(sorted(rows, key=timestamp), event_key)
    folds: list[WalkForwardFold] = []
    cursor = min_train
    while cursor < len(ordered):
        test = ordered[cursor:cursor + test_size]
        if not test:
            break
        train = ordered[:cursor]
        train_events = {event_key(x) for x in train}
        test = [x for x in test if event_key(x) not in train_events]
        if test:
            folds.append(WalkForwardFold(tuple(train), tuple(test)))
        cursor += test_size
    return folds
