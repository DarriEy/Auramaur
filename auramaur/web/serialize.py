"""JSON-safe view of the cockpit state (datetimes → ISO strings/ages)."""

from __future__ import annotations

from datetime import datetime


def serialize_state(s: dict) -> dict:
    now: datetime = s["now"]

    def iso(ts: datetime | None) -> str | None:
        return ts.isoformat() if ts else None

    return {
        "now": iso(now),
        "is_live": s["is_live"],
        "transfers_armed": s["transfers_armed"],
        "kill_switch": s["kill_switch"],
        "venues": s["venues"],
        "pillars": [
            {
                "name": name,
                "last_seen": iso(ts),
                "age_seconds": None if ts is None else max(0.0, (now - ts).total_seconds()),
            }
            for name, ts in s["pillars"].items()
        ],
        "activity": [{"time": hhmm, "text": txt} for hhmm, txt in s["activity"]],
        "health": s["health"],
        "positions": s["positions"],
        "position_count": s["position_count"],
        "position_value": s["position_value"],
        "signals": s["signals"],
        "trade_count": s["trade_count"],
        "total_pnl": s["total_pnl"],
        "drawdown": s["drawdown"],
        "balance": s["balance"],
    }
