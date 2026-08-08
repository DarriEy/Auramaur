from unittest.mock import patch

from auramaur.exchange.models import Position, TokenType, OrderSide
from auramaur.risk.exit_calibration import (
    ExitCandidate,
    ExitObservation,
    calibrate,
    candidate_exit,
    completed_episodes,
)
from auramaur.risk.portfolio import PortfolioTracker


BASELINE = ExitCandidate("deployed", 1.0, 12.0, 0.45)
BANK = ExitCandidate("bank", 0.75, 12.0, 0.20)


def _observation(
    market: str, second: int, action: str, net: float, peak: float = 50.0,
) -> ExitObservation:
    return ExitObservation(
        gross_pnl_pct=net,
        observed_at=f"2026-01-01T00:00:{second:02d}+00:00",
        exchange="polymarket", market_id=market, token="YES", is_paper=1,
        policy_action=action, net_pnl_pct=net, peak_pnl_pct=peak,
        target_pct=100.0, entry_price=0.5, size=20.0,
    )


def _episode(index: int, *, early: float = 40.0, terminal: float = 20.0,
             terminal_peak: float = 50.0):
    market = f"m{index:03d}"
    return [
        _observation(market, 0, "HOLD", early),
        _observation(market, 1, "TRAILING_STOP", terminal, terminal_peak),
    ]


def test_completed_episodes_split_reentries_and_drop_open_paths():
    rows = [
        _observation("m1", 0, "HOLD", 10),
        _observation("m2", 1, "HOLD", 5),
        _observation("m1", 2, "TRAILING_STOP", 1),
        _observation("m1", 3, "HOLD", 8),
        _observation("m1", 4, "PROFIT_TARGET", 12),
    ]
    episodes = completed_episodes(rows)
    assert [len(episode) for episode in episodes] == [2, 2]
    assert all(episode[-1].policy_action != "HOLD" for episode in episodes)
    assert all(episode[0].market_id == "m1" for episode in episodes)


def test_candidate_uses_first_observed_trigger_without_future_prices():
    episode = _episode(1)
    assert candidate_exit(episode, BANK) is episode[0]
    assert candidate_exit(episode, BASELINE) is episode[1]


def test_trailing_replay_uses_gross_mark_like_runtime_policy():
    item = _observation("m1", 0, "HOLD", 5.0, peak=50.0)
    item = ExitObservation(**{
        **item.__dict__, "gross_pnl_pct": 30.0,
    })
    assert candidate_exit([item], BANK) is item


def test_calibration_counts_one_independent_episode_per_position():
    duplicate_market = [_episode(1), _episode(1)]
    report = calibrate(duplicate_market, [BASELINE, BANK], BASELINE,
                       min_train=1, min_holdout=1)
    assert report.episodes == 1


def test_calibration_selects_on_train_and_requires_positive_holdout_lcb():
    episodes = [_episode(i) for i in range(50)]
    report = calibrate(episodes, [BASELINE, BANK], BASELINE)
    assert report.selected == BANK
    assert report.holdout_score is not None
    assert report.holdout_score.n == 15
    assert report.holdout_score.delta_lcb95_usd > 0
    assert report.recommendation is not None


def test_train_winner_is_rejected_when_holdout_direction_reverses():
    episodes = [_episode(i) for i in range(35)]
    episodes += [_episode(i + 35, early=30.0, terminal=40.0, terminal_peak=80.0)
                 for i in range(15)]
    report = calibrate(episodes, [BASELINE, BANK], BASELINE)
    assert report.selected == BANK
    assert report.holdout_score is not None
    assert report.holdout_score.mean_delta_usd < 0
    assert report.recommendation is None


def test_insufficient_sample_never_recommends():
    report = calibrate([_episode(i) for i in range(20)],
                       [BASELINE, BANK], BASELINE)
    assert report.recommendation is None
    assert "insufficient" in report.reason


def test_hold_sampling_keeps_first_then_one_per_interval():
    tracker = PortfolioTracker(db=None)
    position = Position(
        market_id="m1", exchange="polymarket", side=OrderSide.BUY,
        size=10, avg_price=0.5, current_price=0.6,
        token=TokenType.YES, is_paper=True,
    )
    with patch("auramaur.risk.portfolio.monotonic",
               side_effect=[0.0, 10.0, 3600.0]):
        assert tracker._should_sample_hold(position, 1, 3600)
        assert not tracker._should_sample_hold(position, 1, 3600)
        assert tracker._should_sample_hold(position, 1, 3600)


def test_terminal_action_resets_sampling_for_reentry():
    tracker = PortfolioTracker(db=None)
    position = Position(
        market_id="m1", exchange="polymarket", side=OrderSide.BUY,
        size=10, avg_price=0.5, current_price=0.6,
        token=TokenType.YES, is_paper=True,
    )
    with patch("auramaur.risk.portfolio.monotonic", side_effect=[1.0, 2.0]):
        assert tracker._should_sample_hold(position, 1, 3600)
        tracker._forget_hold_sample(position, 1)
        assert tracker._should_sample_hold(position, 1, 3600)
