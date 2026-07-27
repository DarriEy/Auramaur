"""Balance recorder tests — the bot-side rows the read-only dashboard reads.

The contract: one upserted ``venue_balances`` row per enabled venue; a fetch
failure keeps the last good row (its age tells the story) instead of
overwriting it with a dash; disabled venues are never touched; the IBKR fetch
honors the slower-cadence flag.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from auramaur.db.database import Database
from auramaur.monitoring import balances


def _settings(*, kalshi=False, kraken=False, ibkr=False) -> SimpleNamespace:
    # The recorder only reads the enabled flags; a namespace keeps the test
    # independent of Settings validation.
    return SimpleNamespace(
        kalshi=SimpleNamespace(enabled=kalshi),
        kraken=SimpleNamespace(enabled=kraken),
        ibkr=SimpleNamespace(enabled=ibkr),
    )


async def _rows(db) -> dict[str, dict]:
    return {r["venue"]: dict(r)
            for r in await db.fetchall("SELECT * FROM venue_balances")}


@pytest.mark.asyncio
async def test_upserts_and_keeps_last_good_row_on_failure(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "auramaur.db"))
    await db.connect()
    try:
        settings = _settings(kalshi=True, kraken=True)

        async def kalshi_ok(_s):
            return "$12.34"

        async def kraken_ok(_s):
            return "$500 USDC + 100 CAD"

        monkeypatch.setattr(balances, "kalshi_balance", kalshi_ok)
        monkeypatch.setattr(balances, "kraken_balance", kraken_ok)
        await balances.record_venue_balances(db, settings)

        rows = await _rows(db)
        assert rows["kalshi"]["detail"] == "$12.34"
        assert rows["kraken"]["detail"] == "$500 USDC + 100 CAD"
        kraken_ts = rows["kraken"]["fetched_at"]

        # Second cycle: kalshi refreshes, kraken FAILS — its last good row
        # (value and timestamp) must survive untouched.
        async def kalshi_ok2(_s):
            return "$56.78"

        async def kraken_boom(_s):
            raise RuntimeError("api down")

        monkeypatch.setattr(balances, "kalshi_balance", kalshi_ok2)
        monkeypatch.setattr(balances, "kraken_balance", kraken_boom)
        await balances.record_venue_balances(db, settings)

        rows = await _rows(db)
        assert len(rows) == 2  # upsert, never append
        assert rows["kalshi"]["detail"] == "$56.78"
        assert rows["kraken"]["detail"] == "$500 USDC + 100 CAD"
        assert rows["kraken"]["fetched_at"] == kraken_ts
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_disabled_venues_and_ibkr_cadence_flag(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "auramaur.db"))
    await db.connect()
    try:
        async def ibkr_ok(_s):
            return "$1,000.00 avail | $2,000.00 net"

        monkeypatch.setattr(balances, "ibkr_balance", ibkr_ok)

        # Nothing enabled → nothing recorded.
        await balances.record_venue_balances(db, _settings())
        assert await _rows(db) == {}

        # Enabled but off-cadence → still skipped.
        settings = _settings(ibkr=True)
        await balances.record_venue_balances(db, settings, include_ibkr=False)
        assert await _rows(db) == {}

        # On-cadence → recorded.
        await balances.record_venue_balances(db, settings, include_ibkr=True)
        rows = await _rows(db)
        assert rows["ibkr"]["detail"] == "$1,000.00 avail | $2,000.00 net"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_timeout_failure_names_its_type_and_escalates_when_stale(tmp_path):
    """A fetch failure must say WHAT failed, and shout when the row is fiction.

    Every fetch is wrapped in asyncio.wait_for, and TimeoutError stringifies to
    "" — so the warning added after the 2026-07-21 outage emitted `error: ""`
    153 times on 2026-07-27 while the IBKR balance sat two days stale. The
    operator could see that something failed but never what, and the displayed
    balance silently stood in for a live one.
    """
    from structlog.testing import capture_logs

    from auramaur.db.database import Database
    from auramaur.monitoring import balances as bal

    db = Database(str(tmp_path / "bal.db"))
    await db.connect()
    try:
        stale = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        await db.execute(
            "INSERT INTO venue_balances (venue, detail, fetched_at) "
            "VALUES ('ibkr', '$299.03 avail', ?)", (stale,))
        await db.commit()

        async def boom(_settings):
            raise asyncio.TimeoutError()

        settings = SimpleNamespace(
            kalshi=SimpleNamespace(enabled=False),
            kraken=SimpleNamespace(enabled=False),
            polymarket=SimpleNamespace(enabled=False),
            ibkr=SimpleNamespace(enabled=True))
        monkey = bal.ibkr_balance
        bal.ibkr_balance = boom
        try:
            with capture_logs() as events:
                await bal.record_venue_balances(db, settings, include_ibkr=True)
        finally:
            bal.ibkr_balance = monkey

        text = repr(events)
        # The type is named even though str(TimeoutError()) == "".
        assert "TimeoutError" in text
        # And a 48h-old row escalates past a routine warning.
        assert "stale_balance" in text

        # The last good row is preserved, never overwritten with a failure.
        row = await db.fetchone("SELECT detail FROM venue_balances WHERE venue='ibkr'")
        assert row["detail"] == "$299.03 avail"
    finally:
        await db.close()
