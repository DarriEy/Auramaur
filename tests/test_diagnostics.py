"""Diagnostics: error digest + attribution/error rendering."""

import asyncio
from types import SimpleNamespace

from auramaur.db.database import Database
from auramaur.monitoring.diagnostics import (
    error_panel_compact,
    gather_doctor,
    render_attribution,
    render_doctor,
    render_error_digest,
    summarize_errors,
)


def _rec(level, event, **extra):
    return {"level": level, "event": event, "timestamp": "2026-06-01T01:00:00Z", **extra}


def test_summarize_counts_and_excludes_info():
    records = [
        _rec("info", "cycle.done"),          # excluded
        _rec("debug", "cache.hit"),          # excluded
        _rec("warning", "newsapi_rate_limited"),
        _rec("warning", "newsapi_rate_limited"),
        _rec("error", "order_cancel.error", error="no attribute 'cancel'"),
        _rec("critical", "kill_switch"),
    ]
    s = summarize_errors(records)
    assert s["warnings"] == 2
    assert s["errors"] == 2          # error + critical
    assert s["total"] == 4
    # Most frequent first.
    assert s["top"][0]["event"] == "newsapi_rate_limited"
    assert s["top"][0]["count"] == 2
    # Latest message threaded through from the `error` field.
    cancel = next(e for e in s["top"] if e["event"] == "order_cancel.error")
    assert "cancel" in cancel["last_msg"]


def test_summarize_empty():
    s = summarize_errors([])
    assert s == {"errors": 0, "warnings": 0, "total": 0, "suppressed": 0, "top": []}


def test_top_is_bounded():
    records = [_rec("error", f"ev{i}") for i in range(20)]
    s = summarize_errors(records, top=5)
    assert len(s["top"]) == 5
    assert s["errors"] == 20


def test_renderers_build():
    s = summarize_errors([_rec("error", "boom", error="kaboom")])
    assert render_error_digest({**s, "scanned_mb": 6.0, "records": 1}) is not None
    assert error_panel_compact(s) is not None
    # Clean state renders too.
    assert render_error_digest({"errors": 0, "warnings": 0, "top": []}) is not None

    cats = [{"category": "crypto", "positions": 22, "exposure": 192.0,
             "realized_pnl": -11.64, "unrealized_pnl": 71.01,
             "accuracy": 0.73, "kelly_multiplier": 0.36}]
    strats = [{"strategy_source": "llm", "trade_count": 565, "wins": 10, "total_pnl": 3.2}]
    assert render_attribution(cats, strats, mode="live") is not None


def test_benign_events_filtered_by_default():
    records = [
        _rec("warning", "order.live"), _rec("warning", "order.live"),
        _rec("warning", "kraken.order"),
        _rec("error", "real.bug", error="boom"),
    ]
    s = summarize_errors(records)
    assert s["suppressed"] == 3            # 2x order.live + 1x kraken.order
    assert s["warnings"] == 0 and s["errors"] == 1
    assert [e["event"] for e in s["top"]] == ["real.bug"]
    # --all surfaces them again.
    s2 = summarize_errors(records, include_benign=True)
    assert s2["suppressed"] == 0 and s2["warnings"] == 3


def test_doctor_killswitch_drives_verdict():
    async def run():
        db = Database(":memory:")
        await db.connect()
        settings = SimpleNamespace(
            is_live=False, kill_switch_active=False,
            logging=SimpleNamespace(file="/nonexistent/auramaur.log"),
        )
        s = await gather_doctor(settings, db)
        names = {c["name"]: c for c in s["checks"]}
        assert names["mode"]["detail"] == "PAPER"
        assert names["kill switch"]["status"] == "ok"
        assert "positions" in names
        assert render_doctor(s) is not None

        # An active kill switch forces a PROBLEM verdict.
        settings.kill_switch_active = True
        s2 = await gather_doctor(settings, db)
        assert {c["name"]: c for c in s2["checks"]}["kill switch"]["status"] == "fail"
        assert s2["verdict"] == "fail"
        await db.close()

    asyncio.run(run())


def test_doctor_distinguishes_stale_from_dormant(tmp_path):
    """A pillar that went silent (stale) warns; one never seen (dormant) is just
    informational context (e.g. IBKR pre-funding) and does not warn."""
    import json
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    recent = (now - timedelta(seconds=30)).isoformat()
    old = (now - timedelta(minutes=40)).isoformat()
    logf = tmp_path / "auramaur.log"
    rows = [
        {"event": "boot", "level": "info", "timestamp": old},          # sacrificial 1st line
        {"event": "kalshi.scan", "level": "info", "timestamp": recent},   # alive
        {"event": "polymarket.cycle", "level": "info", "timestamp": old},  # went stale
        # ibkr never appears -> dormant
    ]
    logf.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    async def run():
        db = Database(":memory:")
        await db.connect()
        settings = SimpleNamespace(is_live=False, kill_switch_active=False,
                                   logging=SimpleNamespace(file=str(logf)))
        s = await gather_doctor(settings, db)
        pillars = next(c for c in s["checks"] if c["name"] == "pillars")
        assert pillars["status"] == "warn"          # polymarket went silent
        assert "polymarket" in pillars["detail"]
        assert "ibkr" in pillars["detail"]           # dormant context, not a fault
        assert "dormant" in pillars["detail"]
        await db.close()

    asyncio.run(run())


def test_doctor_fails_when_zero_expected_pillars_are_alive(tmp_path):
    async def run():
        db = Database(":memory:")
        await db.connect()
        settings = SimpleNamespace(
            is_live=False, kill_switch_active=False,
            logging=SimpleNamespace(file=str(tmp_path / "missing.log")),
            monitoring=SimpleNamespace(
                expected_pillars=["polymarket", "news"], pillar_stale_seconds=900),
        )
        state = await gather_doctor(settings, db)
        pillars = next(c for c in state["checks"] if c["name"] == "pillars")
        assert pillars["status"] == "fail"
        assert "ZERO expected pillars alive" in pillars["detail"]
        assert state["verdict"] == "fail"
        await db.close()
    asyncio.run(run())


def test_doctor_exit_lifecycle_warns_only_on_actionable_rows():
    """The blocked-exit check must be falsifiable: routine backoff and rows
    whose position already left the book are NOT blockages, or the check
    degrades to a permanently-warning counter the operator learns to ignore
    (the alarm-fatigue mode the 12-day exit outage postmortem warned about).
    """
    async def run():
        db = Database(":memory:")
        await db.connect()
        settings = SimpleNamespace(
            is_live=False, kill_switch_active=False,
            logging=SimpleNamespace(file="/nonexistent/auramaur.log"),
        )

        async def seed_row(market, state, retry_mod):
            await db.execute(
                """INSERT INTO exit_lifecycle
                       (exchange, market_id, token, is_paper, state,
                        next_retry_at)
                     VALUES ('polymarket', ?, 'YES', 1, ?,
                             datetime('now', ?))""",
                (market, state, retry_mod))

        async def seed_position(market):
            await db.execute(
                """INSERT INTO portfolio
                       (market_id, exchange, side, size, avg_price,
                        current_price, category, token, token_id, is_paper)
                     VALUES (?, 'polymarket', 'BUY', 10, 0.5, 0.5,
                             'politics', 'YES', 't', 1)""",
                (market,))

        # Routine backoff: future wall, live position -> not blocked.
        await seed_position("M-backoff")
        await seed_row("M-backoff", "RETRYABLE", "+10 minutes")
        s = await gather_doctor(settings, db)
        check = next(c for c in s["checks"] if c["name"] == "exit lifecycle")
        assert check["status"] == "ok"

        # Past-due wall with NO surviving position (settled out-of-band):
        # invisible, never an eternal warning.
        await seed_row("M-ghost", "RETRYABLE", "-30 minutes")
        s = await gather_doctor(settings, db)
        check = next(c for c in s["checks"] if c["name"] == "exit lifecycle")
        assert check["status"] == "ok"

        # Past-due wall WITH a live position: genuinely stuck -> warn.
        await seed_position("M-stuck")
        await seed_row("M-stuck", "RETRYABLE", "-30 minutes")
        s = await gather_doctor(settings, db)
        check = next(c for c in s["checks"] if c["name"] == "exit lifecycle")
        assert check["status"] == "warn"
        assert "retryable=1" in check["detail"]

        # Held dust is a known book state with its own remedy: named in the
        # detail, but it alone must not degrade the verdict.
        await db.execute("DELETE FROM exit_lifecycle")
        await db.execute("DELETE FROM portfolio")
        await seed_position("M-dust")
        await seed_row("M-dust", "UNSALEABLE_DUST", "+15 minutes")
        s = await gather_doctor(settings, db)
        check = next(c for c in s["checks"] if c["name"] == "exit lifecycle")
        assert check["status"] == "ok"
        assert "dust=1" in check["detail"]
        await db.close()

    asyncio.run(run())


def test_doctor_attribution_is_windowed_and_skips_sentinels():
    """One pre-attribution '' row from history must not pin the check at warn
    forever, and deliberately-bucketed rows (venue_/phantom_/legacy_
    unattributed) are labeled outcomes, not leaks."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        settings = SimpleNamespace(
            is_live=False, kill_switch_active=False,
            logging=SimpleNamespace(file="/nonexistent/auramaur.log"),
        )

        async def seed_ledger(ref, source, age_mod):
            await db.execute(
                """INSERT INTO pnl_ledger
                       (market_id, strategy_source, kind, pnl, is_paper,
                        source_ref, realized_at)
                     VALUES ('M1', ?, 'sell', 1.0, 1, ?,
                             datetime('now', ?))""",
                (source, ref, age_mod))

        # Ancient empty-source row: historical, outside the window -> ok.
        await seed_ledger("old-empty", "", "-30 days")
        # Recent sentinel-bucket row: a labeled outcome, not a leak -> ok.
        await seed_ledger("recent-bucket", "venue_unattributed", "-1 hours")
        s = await gather_doctor(settings, db)
        check = next(c for c in s["checks"] if c["name"] == "P&L attribution")
        assert check["status"] == "ok"

        # Recent empty-source row: a live leak -> warn.
        await seed_ledger("recent-empty", "", "-1 hours")
        s = await gather_doctor(settings, db)
        check = next(c for c in s["checks"] if c["name"] == "P&L attribution")
        assert check["status"] == "warn"
        assert "1 unattributed" in check["detail"]
        await db.close()

    asyncio.run(run())


if __name__ == "__main__":
    test_summarize_counts_and_excludes_info()
    test_summarize_empty()
    test_top_is_bounded()
    test_renderers_build()
    print("ok")
