from datetime import datetime, timedelta, timezone
import math
import pytest
from auramaur.backtest.walk_forward import (
    StrategyObservation,
    evaluate_strategy,
    expanding_walk_forward,
)


def test_purge_overlap_and_embargo():
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = [
        {
            "id": str(i),
            "start": start + timedelta(days=i),
            "end": start + timedelta(days=i + (5 if i == 2 else 0)),
        }
        for i in range(8)
    ]
    folds = expanding_walk_forward(
        rows,
        timestamp=lambda x: x["start"],
        event_end=lambda x: x["end"],
        event_key=lambda x: x["id"],
        min_train=4,
        test_size=2,
        embargo=timedelta(days=1),
    )
    assert [x["id"] for x in folds[0].train] == ["0", "1"]
    assert [x["id"] for x in folds[0].test] == ["4", "5"]


def test_negative_embargo_rejected():
    with pytest.raises(ValueError, match="embargo"):
        expanding_walk_forward(
            [], timestamp=lambda x: x, event_key=str, embargo=timedelta(seconds=-1)
        )


def test_strategy_metrics_and_edges():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    points = [
        StrategyObservation(start, 100, 0),
        StrategyObservation(start + timedelta(days=1), 110, 10, 100, 50),
        StrategyObservation(start + timedelta(days=2), 99, -11, 50, 99),
        StrategyObservation(start + timedelta(days=3), 108.9, 9.9, 25, 54.45),
    ]
    result = evaluate_strategy(points, tail_probability=0.34)
    assert result.max_drawdown == pytest.approx(0.1)
    assert result.profit_factor == pytest.approx(19.9 / 11)
    assert result.hit_rate == pytest.approx(0.5)
    assert result.tail_loss == pytest.approx(0.1)
    assert result.annualized_volatility > 0 and result.cagr is not None
    one = evaluate_strategy([StrategyObservation(start, 100, 2)])
    assert one.cagr is None and math.isinf(one.profit_factor)
    with pytest.raises(ValueError, match="equity"):
        evaluate_strategy([StrategyObservation(start, 0)])
