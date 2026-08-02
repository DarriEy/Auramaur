

def test_kalshi_markets_are_attributed_to_the_llm_kalshi_lane():
    """Every other multi-venue strategy carries a per-venue strategy_source
    (agent_trader_opus / agent_trader_opus_kalshi) so each venue is adjudicated
    on its own evidence. The engine ran BOTH venues under a single "llm",
    which pooled their cells — and once llm was ladder-exempt, let a
    Polymarket record authorise Kalshi trading.

    Only the DEFAULT splits; an explicit strategy_source is respected.
    """
    import inspect

    from auramaur.strategy.engine import TradingEngine

    src = inspect.getsource(TradingEngine.analyze_market)
    assert 'strategy_source == "llm"' in src
    assert '"kalshi"' in src and 'llm_kalshi' in src

    # The guard is on the default only, so a caller-supplied source survives.
    body = src[src.index('strategy_source == "llm"'):]
    assert 'strategy_source = "llm_kalshi"' in body[:400]


def test_live_allowlist_applies_the_strategys_own_narrowing():
    """The engine's live pre-filter must mirror the gateway's per-strategy
    narrowing, or a venue lane bounded by live_categories_only spends every
    analysis slot on categories its signals can never trade: measured
    2026-07-19..08-01, 14 days of kalshi engine cycles analyzed zero
    politics_us markets — llm_kalshi's only live-eligible category — while
    all 58 gateway rejections died on category."""
    from config.settings import RiskConfig

    from auramaur.strategy.engine_cycle import live_allowed_categories

    risk = RiskConfig(
        allowed_categories_live=["economics", "politics_us", "science"],
        live_categories_only={"llm_kalshi": ["politics_us"]},
    )
    assert live_allowed_categories(risk, "llm_kalshi") == {"politics_us"}
    # A source absent from the narrowing map keeps the full global allowlist.
    assert live_allowed_categories(risk, "llm") == {
        "economics", "politics_us", "science"}


def test_live_allowlist_narrowing_intersects_the_global_list():
    """Narrowing only: a category named in live_categories_only but absent
    from the global allowlist stays out — the restriction can never grant."""
    from config.settings import RiskConfig

    from auramaur.strategy.engine_cycle import live_allowed_categories

    risk = RiskConfig(
        allowed_categories_live=["economics"],
        live_categories_only={"llm_kalshi": ["politics_us"]},
    )
    assert live_allowed_categories(risk, "llm_kalshi") == set()


def test_live_allowlist_extension_widens_named_source_then_narrowing_wins():
    """allowed_categories_live_extra widens ONLY the named source, and
    live_categories_only applies after it, so an extension cannot widen
    around a restriction."""
    from config.settings import RiskConfig

    from auramaur.strategy.engine_cycle import live_allowed_categories

    risk = RiskConfig(
        allowed_categories_live=["economics"],
        allowed_categories_live_extra={"agent_trader_opus": ["other"]},
    )
    assert live_allowed_categories(risk, "agent_trader_opus") == {
        "economics", "other"}
    assert live_allowed_categories(risk, "llm") == {"economics"}

    restricted = RiskConfig(
        allowed_categories_live=["economics"],
        allowed_categories_live_extra={"agent_trader_opus": ["other"]},
        live_categories_only={"agent_trader_opus": ["economics"]},
    )
    assert live_allowed_categories(restricted, "agent_trader_opus") == {
        "economics"}


def test_live_allowlist_exempts_structural_strategies_entirely():
    """A category-gate-exempt source gets None — no filter — matching the
    gateway, where the whole allowlist check is skipped for it."""
    from config.settings import RiskConfig

    from auramaur.strategy.engine_cycle import live_allowed_categories

    risk = RiskConfig()
    assert live_allowed_categories(risk, "market_maker") is None


def test_run_cycle_prefilters_on_the_venue_lanes_own_allowlist():
    """run_cycle derives the engine's per-venue source (llm / llm_kalshi,
    same split analyze_market applies) and feeds it to the helper, so the
    kalshi engine pre-filters on llm_kalshi's own live categories."""
    import inspect

    from auramaur.strategy.engine_cycle import CycleOrchestrationMixin

    src = inspect.getsource(CycleOrchestrationMixin.run_cycle)
    assert "live_allowed_categories" in src
    derivation = src[src.index("engine_source"):]
    assert '"llm_kalshi" if self.exchange_name == "kalshi"' in derivation[:200]
