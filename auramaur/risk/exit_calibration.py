"""Holdout-safe replay of earlier-banking exit policies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping


TERMINAL_ACTIONS = frozenset({"PROFIT_TARGET", "TRAILING_STOP"})


@dataclass(frozen=True)
class ExitObservation:
    observed_at: str
    exchange: str
    market_id: str
    token: str
    is_paper: int
    policy_action: str
    net_pnl_pct: float
    gross_pnl_pct: float
    peak_pnl_pct: float
    target_pct: float | None
    entry_price: float
    size: float

    @property
    def position_key(self) -> tuple[str, str, str, int]:
        return (self.exchange, self.market_id, self.token, self.is_paper)

    @property
    def estimated_net_usd(self) -> float:
        return self.entry_price * self.size * self.net_pnl_pct / 100.0


@dataclass(frozen=True)
class ExitCandidate:
    name: str
    target_scale: float
    trailing_activation_pct: float
    trailing_giveback_fraction: float


@dataclass(frozen=True)
class CandidateScore:
    candidate: ExitCandidate
    n: int
    mean_usd: float
    total_usd: float
    mean_delta_usd: float
    delta_lcb95_usd: float


@dataclass(frozen=True)
class CalibrationReport:
    episodes: int
    train_n: int
    holdout_n: int
    train_scores: tuple[CandidateScore, ...]
    holdout_score: CandidateScore | None
    selected: ExitCandidate | None
    recommendation: str | None
    reason: str


def observation(row: Mapping) -> ExitObservation:
    return ExitObservation(
        observed_at=str(row["observed_at"]),
        exchange=str(row["exchange"] or ""),
        market_id=str(row["market_id"]), token=str(row["token"]),
        is_paper=int(row["is_paper"]), policy_action=str(row["policy_action"]),
        net_pnl_pct=float(row["net_pnl_pct"]),
        gross_pnl_pct=float(row["gross_pnl_pct"]),
        peak_pnl_pct=float(row["peak_pnl_pct"]),
        target_pct=(None if row["target_pct"] is None else float(row["target_pct"])),
        entry_price=float(row["entry_price"]), size=float(row["size"]),
    )


def completed_episodes(
    observations: Iterable[ExitObservation],
) -> list[list[ExitObservation]]:
    """Split each position stream at a recorded target/trailing exit.

    Open episodes are right-censored and excluded: scoring a policy after the
    last observed mark would invent a price path. Re-entry starts a new episode.
    """
    active: dict[tuple[str, str, str, int], list[ExitObservation]] = {}
    completed: list[list[ExitObservation]] = []
    for item in observations:
        episode = active.setdefault(item.position_key, [])
        episode.append(item)
        if item.policy_action in TERMINAL_ACTIONS:
            completed.append(episode)
            active.pop(item.position_key, None)
    return completed


def candidate_exit(
    episode: list[ExitObservation], candidate: ExitCandidate,
) -> ExitObservation | None:
    for item in episode:
        trailing = (
            candidate.trailing_activation_pct > 0
            and candidate.trailing_giveback_fraction > 0
            and item.peak_pnl_pct >= candidate.trailing_activation_pct
            and item.gross_pnl_pct <= item.peak_pnl_pct
            * (1.0 - candidate.trailing_giveback_fraction)
        )
        target = (item.target_pct is not None
                  and item.net_pnl_pct >= item.target_pct * candidate.target_scale)
        if trailing or target:
            return item
    return None


def _paired_score(
    candidate: ExitCandidate, baseline: ExitCandidate,
    episodes: list[list[ExitObservation]],
) -> CandidateScore:
    outcomes: list[float] = []
    deltas: list[float] = []
    for episode in episodes:
        proposed = candidate_exit(episode, candidate)
        deployed = candidate_exit(episode, baseline)
        if proposed is None or deployed is None:
            continue
        outcomes.append(proposed.estimated_net_usd)
        deltas.append(proposed.estimated_net_usd - deployed.estimated_net_usd)
    n = len(deltas)
    mean = sum(outcomes) / n if n else 0.0
    delta = sum(deltas) / n if n else 0.0
    if n < 2:
        lcb = float("-inf")
    else:
        variance = sum((value - delta) ** 2 for value in deltas) / (n - 1)
        lcb = delta - 1.96 * math.sqrt(variance / n)
    return CandidateScore(candidate, n, mean, sum(outcomes), delta, lcb)


def calibrate(
    episodes: list[list[ExitObservation]], candidates: list[ExitCandidate],
    baseline: ExitCandidate, *, min_train: int = 30, min_holdout: int = 15,
) -> CalibrationReport:
    """Select on the oldest 70%; validate the winner once on newest 30%."""
    # Re-entries in one market share resolution risk and are not independent
    # evidence. Keep the first completed episode per position identity.
    independent: list[list[ExitObservation]] = []
    seen: set[tuple[str, str, str, int]] = set()
    for episode in episodes:
        if episode[0].position_key not in seen:
            independent.append(episode)
            seen.add(episode[0].position_key)
    episodes = independent
    cut = int(len(episodes) * 0.7)
    train, holdout = episodes[:cut], episodes[cut:]
    train_scores = tuple(
        _paired_score(candidate, baseline, train) for candidate in candidates)
    if len(train) < min_train or len(holdout) < min_holdout:
        return CalibrationReport(
            len(episodes), len(train), len(holdout), train_scores, None, None,
            None, "insufficient completed episodes")
    eligible = [score for score in train_scores
                if score.n >= min_train and score.candidate != baseline]
    if not eligible:
        return CalibrationReport(
            len(episodes), len(train), len(holdout), train_scores, None, None,
            None, "no candidate has enough paired training observations")
    winner = max(eligible, key=lambda score: score.delta_lcb95_usd)
    if winner.delta_lcb95_usd <= 0:
        return CalibrationReport(
            len(episodes), len(train), len(holdout), train_scores, None, None,
            None, "no candidate improves the deployed policy in training")
    holdout_score = _paired_score(winner.candidate, baseline, holdout)
    if holdout_score.n < min_holdout or holdout_score.delta_lcb95_usd <= 0:
        return CalibrationReport(
            len(episodes), len(train), len(holdout), train_scores,
            holdout_score, winner.candidate, None,
            "train winner did not clear the paired holdout lower bound")
    recommendation = (
        f"target_scale={winner.candidate.target_scale:g}, "
        f"trailing_giveback_fraction="
        f"{winner.candidate.trailing_giveback_fraction:g}")
    return CalibrationReport(
        len(episodes), len(train), len(holdout), train_scores,
        holdout_score, winner.candidate, recommendation,
        "paired holdout lower bound is positive")
