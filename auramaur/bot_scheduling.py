"""Adaptive scheduling policy for the main bot orchestrator."""

from __future__ import annotations

from datetime import datetime, timezone


class BotSchedulingMixin:
    """Keep time-of-day and cash-pressure scheduling out of orchestration code."""

    def _get_schedule_mode(self) -> str:
        """Return the current adaptive schedule mode."""
        cfg = self.settings.intervals
        if not cfg.adaptive_enabled:
            return ""

        if self._is_cash_starved():
            return "starved"

        hour_utc = datetime.now(timezone.utc).hour
        if hour_utc in cfg.quiet_hours_utc:
            return "quiet"
        if hour_utc not in cfg.peak_hours_utc:
            return "off_peak"
        return "peak"

    def _adaptive_interval(self, base_seconds: int) -> int:
        """Scale an interval based on time of day and available capital."""
        cfg = self.settings.intervals
        mode = self._get_schedule_mode()

        if mode == "quiet":
            multiplier = cfg.quiet_multiplier
        elif mode == "off_peak":
            multiplier = cfg.off_peak_multiplier
        else:
            multiplier = 1.0

        if self._is_cash_starved():
            multiplier *= 5.0

        return int(base_seconds * multiplier)

    def _is_cash_starved(self) -> bool:
        """Return whether available cash is below the minimum useful balance."""
        return getattr(self, "_last_known_cash", 0.0) < 5.0
