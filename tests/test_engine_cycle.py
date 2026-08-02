import pytest


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


def test_close_window_tiers_cover_exactly_the_junk_filters_tradeable_band():
    """The tiers must be contiguous and jointly span the 2h..10y window
    _is_junk_market accepts — a gap means a horizon whose markets can pass
    every filter yet never enter discovery (the Speaker-ladder failure mode:
    3 analyzable politics_us markets unscanned since June 2026)."""
    from auramaur.strategy.engine import _CLOSE_WINDOW_TIERS

    assert _CLOSE_WINDOW_TIERS[0][0] == 2 * 3600
    assert _CLOSE_WINDOW_TIERS[-1][1] == 10 * 365 * 86400
    for (_, hi, _), (lo, _, _) in zip(
            _CLOSE_WINDOW_TIERS, _CLOSE_WINDOW_TIERS[1:]):
        assert hi == lo
    # The venue caps a page at 1000; a larger limit would silently truncate.
    assert all(lim <= 1000 for _, _, lim in _CLOSE_WINDOW_TIERS)


def test_merge_markets_by_id_dedupes_and_keeps_first():
    from auramaur.exchange.models import Market

    from auramaur.strategy.engine import merge_markets_by_id

    base = [Market(id="a", question="A?"), Market(id="b", question="B?")]
    extra = [Market(id="b", question="B-dup?"), Market(id="c", question="C?")]
    merged = merge_markets_by_id(base, extra)
    assert [m.id for m in merged] == ["a", "b", "c"]
    assert merged[1].question == "B?"
    assert [m.id for m in base] == ["a", "b"]  # input not mutated


@pytest.mark.asyncio
async def test_augment_rotates_tiers_and_dedupes_against_the_generic_scan():
    from types import SimpleNamespace

    from auramaur.exchange.models import Market

    from auramaur.strategy.engine import _CLOSE_WINDOW_TIERS, TradingEngine

    calls: list[tuple[int, int, int]] = []

    class _Discovery:
        async def get_markets_by_close_window(self, min_ts, max_ts, limit=200):
            calls.append((min_ts, max_ts, limit))
            return [Market(id="window-only", question="W?"),
                    Market(id="generic", question="G-dup?")]

    self = SimpleNamespace(discovery=_Discovery())
    base = [Market(id="generic", question="G?")]

    for expected_tier in list(_CLOSE_WINDOW_TIERS) + [_CLOSE_WINDOW_TIERS[0]]:
        merged = await TradingEngine._augment_with_close_window(self, base)
        assert [m.id for m in merged] == ["generic", "window-only"]
        min_ts, max_ts, limit = calls[-1]
        min_off, max_off, tier_limit = expected_tier
        assert max_ts - min_ts == max_off - min_off
        assert limit == tier_limit
    assert len(calls) == len(_CLOSE_WINDOW_TIERS) + 1  # wrapped around


@pytest.mark.asyncio
async def test_augment_is_a_noop_without_the_windowed_fetch_and_on_error():
    from types import SimpleNamespace

    from auramaur.exchange.models import Market

    from auramaur.strategy.engine import TradingEngine

    base = [Market(id="a", question="A?")]

    plain = SimpleNamespace(discovery=object())
    assert await TradingEngine._augment_with_close_window(plain, base) is base

    class _Broken:
        async def get_markets_by_close_window(self, *a, **k):
            raise RuntimeError("venue down")

    broken = SimpleNamespace(discovery=_Broken())
    assert await TradingEngine._augment_with_close_window(broken, base) == base


def test_strategic_signals_carry_the_venue_lane_source():
    """The 2026-07-28 venue split retagged analyze_market but missed the
    strategic path: its Signal(...) fell back to the model default "llm",
    whose live_venues_only is polymarket-only, so the gateway emptied the
    allowlist and rejected every strategic kalshi entry regardless of
    category (observed 2026-08-02) — a lane that could never fire. The
    strategic construction must stamp the per-venue source and persist
    that same source, never a literal."""
    import inspect

    from auramaur.strategy.engine_cycle import CycleOrchestrationMixin

    src = inspect.getsource(CycleOrchestrationMixin._run_cycle_strategic)
    construction = src[src.index("signal = Signal("):]
    assert "strategy_source=" in construction[:900]
    assert '"llm_kalshi"' in construction[:900]
    # The signals INSERT must persist the signal's own source — a hardcoded
    # "llm" literal in its parameters is exactly the misattribution this
    # test pins down.
    insert_region = src[src.index("INSERT INTO signals"):][:900]
    assert '"llm"' not in insert_region
    assert "signal.strategy_source)," in insert_region


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
