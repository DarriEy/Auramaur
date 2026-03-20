"""Tests for smart market selection."""

from datetime import datetime, timezone, timedelta

from auramaur.exchange.models import Market
from auramaur.strategy.market_selector import score_market, rank_markets


def _make_market(**kwargs) -> Market:
    defaults = {
        "id": "test",
        "condition_id": "cond",
        "question": "Test?",
        "outcome_yes_price": 0.5,
        "liquidity": 5000.0,
        "volume": 10000.0,
        "spread": 0.02,
    }
    defaults.update(kwargs)
    return Market(**defaults)


class TestMarketScoring:
    def test_sweet_spot_price_scores_higher(self):
        """Markets at 30% or 70% should score higher than 50%."""
        m50 = _make_market(outcome_yes_price=0.50)
        m30 = _make_market(outcome_yes_price=0.30)
        assert score_market(m30) > score_market(m50)

    def test_medium_liquidity_preferred(self):
        """$5k liquidity should score higher than $500k."""
        m_low = _make_market(liquidity=5000.0)
        m_high = _make_market(liquidity=500000.0)
        assert score_market(m_low) > score_market(m_high)

    def test_price_movement_boosts_score(self):
        """Markets with recent price moves should score higher."""
        m = _make_market(id="moving")
        history = {"moving": [0.50, 0.58]}  # 8% move
        score_with_move = score_market(m, price_history=history)
        score_without = score_market(m)
        assert score_with_move > score_without

    def test_near_resolution_boost(self):
        """Markets 2 days from resolution with non-extreme prices get boosted."""
        m_near = _make_market(
            outcome_yes_price=0.65,
            end_date=datetime.now(timezone.utc) + timedelta(days=2),
        )
        m_far = _make_market(
            outcome_yes_price=0.65,
            end_date=datetime.now(timezone.utc) + timedelta(days=30),
        )
        assert score_market(m_near) > score_market(m_far)

    def test_rank_markets_sorts_descending(self):
        """rank_markets should return highest-scored first."""
        markets = [
            _make_market(id="boring", outcome_yes_price=0.50, liquidity=500000.0, spread=0.001),
            _make_market(id="juicy", outcome_yes_price=0.30, liquidity=5000.0, spread=0.03),
        ]
        ranked = rank_markets(markets)
        assert ranked[0][0].id == "juicy"
        assert ranked[0][1] > ranked[1][1]

    def test_wide_spread_bonus(self):
        """Wider spreads should get a bonus (limit order opportunity)."""
        m_tight = _make_market(spread=0.005)
        m_wide = _make_market(spread=0.03)
        assert score_market(m_wide) > score_market(m_tight)

    def test_momentum_boosts_score(self):
        """Markets with 3+ price snapshots and large momentum should score higher."""
        m = _make_market(id="momentum_market")
        # 50% price increase over the window
        history = {"momentum_market": [0.40, 0.45, 0.60]}
        score_with = score_market(m, price_history=history)
        score_without = score_market(m)
        assert score_with > score_without

    def test_momentum_negative_also_boosts(self):
        """Negative momentum (price dropping) should also boost score."""
        m = _make_market(id="dropping")
        history = {"dropping": [0.60, 0.50, 0.40]}
        score_with = score_market(m, price_history=history)
        score_without = score_market(m)
        assert score_with > score_without

    def test_momentum_needs_3_snapshots(self):
        """With fewer than 3 snapshots, momentum scoring should not apply."""
        m = _make_market(id="short_history")
        history_2 = {"short_history": [0.40, 0.60]}
        history_3 = {"short_history": [0.40, 0.50, 0.60]}
        score_2 = score_market(m, price_history=history_2)
        score_3 = score_market(m, price_history=history_3)
        # The 3-snapshot version gets momentum bonus on top of the movement bonus
        assert score_3 > score_2

    def test_momentum_zero_when_flat(self):
        """Flat prices should add zero momentum bonus."""
        m = _make_market(id="flat")
        history = {"flat": [0.50, 0.50, 0.50, 0.50]}
        score_flat = score_market(m, price_history=history)
        score_none = score_market(m)
        assert score_flat == score_none
