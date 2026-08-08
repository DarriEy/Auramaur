"""Holdout-safe replay of earlier profit-target policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from auramaur.backtest.ibkr_replay import mean_lcb

NONTERMINAL_ACTIONS = frozenset({"EPISODE_START", "HOLD"})


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
    def episode_key(self) -> tuple[str, str, str, int, float, float]:
        return (*self.position_key, self.entry_price, self.size)

    @property
    def estimated_net_usd(self) -> float:
        return self.entry_price * self.size * self.net_pnl_pct / 100.0


@dataclass(frozen=True)
class ExitCandidate:
    name: str
    target_scale: float


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
        market_id=str(row["market_id"]),
        token=str(row["token"]),
        is_paper=int(row["is_paper"]),
        policy_action=str(row["policy_action"]),
        net_pnl_pct=float(row["net_pnl_pct"]),
        gross_pnl_pct=float(row["gross_pnl_pct"]),
        peak_pnl_pct=float(row["peak_pnl_pct"]),
        target_pct=None if row["target_pct"] is None else float(row["target_pct"]),
        entry_price=float(row["entry_price"]),
        size=float(row["size"]),
    )


def completed_episodes(
    observations: Iterable[ExitObservation],
) -> list[list[ExitObservation]]:
    """Return complete, inventory-coherent episodes.

    Any non-HOLD action is terminal. A changed entry price or size starts a new
    inventory cohort and censors the older path instead of mixing scale-ins.
    """
    active: dict[tuple[str, str, str, int], list[ExitObservation]] = {}
    completed: list[list[ExitObservation]] = []
    for item in observations:
        prior = active.get(item.position_key)
        if prior is not None and prior[-1].episode_key != item.episode_key:
            active.pop(item.position_key, None)
            prior = None
        if prior is None:
            if item.policy_action != "EPISODE_START":
                continue
            prior = []
            active[item.position_key] = prior
        prior.append(item)
        if item.policy_action not in NONTERMINAL_ACTIONS:
            completed.append(prior)
            active.pop(item.position_key, None)
    return completed


def candidate_exit(
    episode: list[ExitObservation], candidate: ExitCandidate,
) -> ExitObservation:
    """Replay an earlier target; otherwise use the actually observed terminal."""
    for item in episode:
        if (
            item.policy_action in NONTERMINAL_ACTIONS
            and item.target_pct is not None
            and item.net_pnl_pct >= item.target_pct * candidate.target_scale
        ):
            return item
        if item.policy_action not in NONTERMINAL_ACTIONS:
            return item
    raise ValueError("candidate_exit requires a completed episode")


def _paired_score(
    candidate: ExitCandidate, episodes: list[list[ExitObservation]],
) -> CandidateScore:
    # Re-entries in one market share resolution risk. Average them into one
    # cluster before computing the evidence bound.
    clustered: dict[tuple[str, str, str, int], list[tuple[float, float]]] = {}
    for episode in episodes:
        proposed = candidate_exit(episode, candidate)
        deployed = episode[-1]
        clustered.setdefault(episode[0].position_key, []).append(
            (proposed.estimated_net_usd, proposed.estimated_net_usd - deployed.estimated_net_usd)
        )
    outcomes = [sum(v[0] for v in values) / len(values) for values in clustered.values()]
    deltas = [sum(v[1] for v in values) / len(values) for values in clustered.values()]
    n = len(deltas)
    mean = sum(outcomes) / n if n else 0.0
    delta = sum(deltas) / n if n else 0.0
    return CandidateScore(
        candidate, n, mean, sum(outcomes), delta, mean_lcb(deltas))


def calibrate(
    episodes: list[list[ExitObservation]], candidates: list[ExitCandidate],
    *, min_train: int = 30, min_holdout: int = 15,
) -> CalibrationReport:
    """Select on oldest market clusters; validate once on newest clusters."""
    clusters: dict[tuple[str, str, str, int], list[list[ExitObservation]]] = {}
    for episode in episodes:
        clusters.setdefault(episode[0].position_key, []).append(episode)
    ordered = list(clusters.values())
    cut = int(len(ordered) * 0.7)
    train = [episode for group in ordered[:cut] for episode in group]
    holdout = [episode for group in ordered[cut:] for episode in group]
    train_scores = tuple(_paired_score(candidate, train) for candidate in candidates)
    if cut < min_train or len(ordered) - cut < min_holdout:
        return CalibrationReport(
            len(episodes), cut, len(ordered) - cut, train_scores, None, None,
            None, "insufficient independent completed position clusters")
    eligible = [score for score in train_scores if score.n >= min_train]
    if not eligible:
        return CalibrationReport(
            len(episodes), cut, len(ordered) - cut, train_scores, None, None,
            None, "no candidate has enough paired training observations")
    winner = max(eligible, key=lambda score: score.delta_lcb95_usd)
    if winner.delta_lcb95_usd <= 0:
        return CalibrationReport(
            len(episodes), cut, len(ordered) - cut, train_scores, None, None,
            None, "no candidate improves deployed exits in training")
    holdout_score = _paired_score(winner.candidate, holdout)
    if holdout_score.n < min_holdout or holdout_score.delta_lcb95_usd <= 0:
        return CalibrationReport(
            len(episodes), cut, len(ordered) - cut, train_scores,
            holdout_score, winner.candidate, None,
            "train winner did not clear the paired holdout lower bound")
    recommendation = f"target_scale={winner.candidate.target_scale:g}"
    return CalibrationReport(
        len(episodes), cut, len(ordered) - cut, train_scores,
        holdout_score, winner.candidate, recommendation,
        "paired holdout lower bound is positive")
