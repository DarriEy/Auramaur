"""Diagnostics: error digest + attribution/error rendering."""

from auramaur.monitoring.diagnostics import (
    error_panel_compact,
    render_attribution,
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
    assert s == {"errors": 0, "warnings": 0, "total": 0, "top": []}


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


if __name__ == "__main__":
    test_summarize_counts_and_excludes_info()
    test_summarize_empty()
    test_top_is_bounded()
    test_renderers_build()
    print("ok")
