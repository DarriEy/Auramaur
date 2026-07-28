

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
