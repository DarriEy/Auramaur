"""Tests for hybrid mode configuration."""

from config.settings import HybridConfig, Settings


class TestHybridConfig:
    def test_defaults(self):
        cfg = HybridConfig()
        assert cfg.arb_scan_seconds == 60
        assert cfg.news_fast_analysis is True
        assert cfg.news_cycle_seconds == 30
        assert cfg.llm_domain_filter is True
        assert cfg.llm_whitelist_min_accuracy == 0.50
        assert cfg.llm_whitelist_min_trades == 5

    def test_settings_has_hybrid(self):
        s = Settings()
        assert hasattr(s, "hybrid")
        assert isinstance(s.hybrid, HybridConfig)

    def test_hybrid_defaults_from_yaml(self):
        s = Settings()
        assert s.hybrid.arb_scan_seconds == 60
        assert s.hybrid.news_cycle_seconds == 30

    def test_hybrid_does_not_override_book_enablement(self):
        """Regression (2026-06-12): --hybrid force-set market_maker.enabled
        = True via market_maker_auto_enable, silently reverting an explicit
        config disable on every restart. Book enablement now lives in exactly
        one place; the field is gone (and stale yaml keys are ignored)."""
        s = Settings()
        assert not hasattr(HybridConfig(), "market_maker_auto_enable")
        s.market_maker.enabled = False
        assert s.market_maker.enabled is False

    def test_hybrid_coexists_with_analysis_mode(self):
        s = Settings()
        s.analysis.mode = "agent"
        assert s.hybrid.arb_scan_seconds == 60
        assert s.analysis.mode == "agent"
