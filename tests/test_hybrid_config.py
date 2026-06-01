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
        assert cfg.market_maker_auto_enable is True

    def test_settings_has_hybrid(self):
        s = Settings()
        assert hasattr(s, "hybrid")
        assert isinstance(s.hybrid, HybridConfig)

    def test_hybrid_defaults_from_yaml(self):
        s = Settings()
        assert s.hybrid.arb_scan_seconds == 60
        assert s.hybrid.news_cycle_seconds == 30

    def test_market_maker_auto_enable(self):
        s = Settings()
        s.market_maker.enabled = False
        assert s.hybrid.market_maker_auto_enable is True
        # Simulating CLI --hybrid behavior
        if s.hybrid.market_maker_auto_enable:
            s.market_maker.enabled = True
        assert s.market_maker.enabled is True

    def test_hybrid_coexists_with_analysis_mode(self):
        s = Settings()
        s.analysis.mode = "agent"
        assert s.hybrid.arb_scan_seconds == 60
        assert s.analysis.mode == "agent"
