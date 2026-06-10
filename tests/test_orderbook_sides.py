"""OrderBook best-level selection must be ordering-agnostic.

The Polymarket CLOB /book endpoint returns levels WORST-first — bids
ascending ($0.01 stub first) and asks descending ($0.99 stub first) — so
indexing [0] read a fictional 98-cent-wide book. Live evidence: the exit
pricer logged ``bid: 0.01`` for a market whose true book was 0.07/0.14.
"""

from __future__ import annotations

from auramaur.exchange.models import OrderBook, OrderBookLevel


def _levels(*prices: float) -> list[OrderBookLevel]:
    return [OrderBookLevel(price=p, size=100.0) for p in prices]


def test_best_levels_with_clob_worst_first_ordering():
    # Exactly how the CLOB returns the Obama "Something" book.
    book = OrderBook(
        bids=_levels(0.01, 0.02, 0.03, 0.06, 0.07),
        asks=_levels(0.99, 0.98, 0.97, 0.15, 0.14),
    )
    assert book.best_bid == 0.07
    assert book.best_ask == 0.14
    assert abs(book.spread - 0.07) < 1e-9
    assert abs(book.midpoint - 0.105) < 1e-9


def test_best_levels_with_best_first_ordering():
    # A venue that returns best-first must yield the same answer.
    book = OrderBook(
        bids=_levels(0.07, 0.06, 0.03),
        asks=_levels(0.14, 0.15, 0.97),
    )
    assert book.best_bid == 0.07
    assert book.best_ask == 0.14


def test_empty_book_sides():
    book = OrderBook()
    assert book.best_bid is None
    assert book.best_ask is None
    assert book.midpoint is None
    assert book.spread is None
