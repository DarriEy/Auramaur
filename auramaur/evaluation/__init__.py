"""Shadow evaluation primitives for intelligence and exploration experiments."""

from .runner import (
    AdapterResponse,
    ArmSpec,
    Episode,
    ExplorationPolicy,
    Forecast,
    ForecastAttempt,
    GenerationRequest,
    IntelligenceEvalRunner,
    TreatmentResult,
    TreatmentSpec,
)
from .domain import (
    EpisodeSnapshot, EvaluationForecast, EvaluationOutcome, EvaluationRun, RunStatus,
)
from .scoring import ForecastScore, score_forecast
from .store import EvaluationStore

__all__ = [
    "AdapterResponse",
    "ArmSpec",
    "Episode",
    "ExplorationPolicy",
    "Forecast",
    "ForecastAttempt",
    "GenerationRequest",
    "IntelligenceEvalRunner",
    "TreatmentResult",
    "TreatmentSpec",
    "EpisodeSnapshot",
    "EvaluationForecast",
    "EvaluationOutcome",
    "EvaluationRun",
    "RunStatus",
    "ForecastScore",
    "score_forecast",
    "EvaluationStore",
]
