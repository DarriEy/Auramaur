from unittest.mock import patch

from auramaur.exchange.models import OrderSide, Position, TokenType
from auramaur.risk.exit_calibration import (
    ExitCandidate,
    ExitObservation,
    calibrate,
    candidate_exit,
    completed_episodes,
)
from auramaur.risk.portfolio import PortfolioTracker


BANK = ExitCandidate("bank", 0.75)


def _observation(
    market: str, second: int, action: str, net: float, *, peak: float = 50.0,
    entry: float = 0.5, size: float = 20.0, exchange: str = "polymarket",
    is_paper: int = 1,
) -> ExitObservation:
    return ExitObservation(
        gross_pnl_pct=net,
        observed_at=f"2026-01-01T00:00:{second:02d}+00:00",
        exchange=exchange, market_id=market, token="YES", is_paper=is_paper,
        policy_action=action, net_pnl_pct=net, peak_pnl_pct=peak,
        target_pct=50.0, entry_price=entry, size=size,
    )


def _episode(index: int, *, early: float = 40.0, terminal: float = 20.0,
             reason: str = "STOP_LOSS"):
    market = f"m{index:03d}"
    return [
        _observation(market, 0, "EPISODE_START", early),
        _observation(market, 1, reason, terminal),
    ]


def _position(**overrides) -> Position:
    values = dict(
        market_id="m1", exchange="polymarket", side=OrderSide.BUY,
        size=10, avg_price=0.5, current_price=0.6,
        token=TokenType.YES, is_paper=True,
    )
    values.update(overrides)
    return Position(**values)


def test_completed_episodes_include_losses_and_split_reentries():
    rows = [
        _observation("m1", 0, "EPISODE_START", 35),
        _observation("m1", 1, "STOP_LOSS", -20),
        _observation("m1", 2, "EPISODE_START", 10),
        _observation("m1", 3, "PROFIT_TARGET", 55),
    ]
    episodes = completed_episodes(rows)
    assert [episode[-1].policy_action for episode in episodes] == [
        "STOP_LOSS", "PROFIT_TARGET"]


def test_left_truncated_path_without_start_marker_is_excluded():
    rows = [
        _observation("m1", 0, "HOLD", 35),
        _observation("m1", 1, "STOP_LOSS", -20),
    ]
    assert completed_episodes(rows) == []

def test_inventory_change_censors_pre_scale_path():
    rows = [
        _observation("m1", 0, "EPISODE_START", 35, size=10),
        _observation("m1", 1, "EPISODE_START", 5, size=20),
        _observation("m1", 2, "STOP_LOSS", -10, size=20),
    ]
    episodes = completed_episodes(rows)
    assert len(episodes) == 1
    assert [item.size for item in episodes[0]] == [20, 20]


def test_candidate_banks_first_fee_net_target_mark():
    episode = _episode(1)
    assert candidate_exit(episode, BANK) is episode[0]


def test_candidate_falls_back_to_actual_terminal_without_future_prices():
    episode = _episode(1, early=10, terminal=-20)
    assert candidate_exit(episode, BANK) is episode[-1]


def test_reentries_are_clustered_as_one_independent_position():
    episodes = [_episode(1), _episode(1)]
    report = calibrate(episodes, [BANK], min_train=1, min_holdout=1)
    assert report.episodes == 2
    assert report.train_n + report.holdout_n == 1


def test_calibration_selects_on_train_and_requires_positive_holdout_lcb():
    report = calibrate([_episode(i) for i in range(50)], [BANK])
    assert report.selected == BANK
    assert report.holdout_score is not None
    assert report.holdout_score.n == 15
    assert report.holdout_score.delta_lcb95_usd > 0
    assert report.recommendation == "target_scale=0.75"


def test_train_winner_is_rejected_when_holdout_direction_reverses():
    episodes = [_episode(i) for i in range(35)]
    episodes += [_episode(i + 35, early=40.0, terminal=60.0)
                 for i in range(15)]
    report = calibrate(episodes, [BANK])
    assert report.selected == BANK
    assert report.holdout_score is not None
    assert report.holdout_score.mean_delta_usd < 0
    assert report.recommendation is None


def test_hold_sampling_key_separates_venue_book_and_inventory():
    tracker = PortfolioTracker(db=None)
    paper = _position()
    live = _position(is_paper=False)
    kalshi = _position(exchange="kalshi")
    scaled = _position(size=12)
    assert len({
        tracker._hold_sample_key(paper, None),
        tracker._hold_sample_key(live, None),
        tracker._hold_sample_key(kalshi, None),
        tracker._hold_sample_key(scaled, None),
    }) == 4


def test_hold_sampling_keeps_first_then_one_per_interval():
    tracker = PortfolioTracker(db=None)
    position = _position()
    with patch("auramaur.risk.portfolio.monotonic",
               side_effect=[0.0, 10.0, 3600.0]):
        assert tracker._should_sample_hold(position, 1, 3600)
        assert not tracker._should_sample_hold(position, 1, 3600)
        assert tracker._should_sample_hold(position, 1, 3600)


def test_terminal_observation_is_written_once_while_position_is_stuck():
    tracker = PortfolioTracker(db=None)
    position = _position()
    rows = []
    tracker._record_terminal_once(
        rows, position, 1, "STOP_LOSS", -20, -21, 0, None, 0.1)
    tracker._record_terminal_once(
        rows, position, 1, "STOP_LOSS", -22, -23, 0, None, 0.1)
    assert len(rows) == 1
    assert rows[0][4] == "STOP_LOSS"
