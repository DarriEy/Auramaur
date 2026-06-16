"""Shared NLP/LLM error types."""

from __future__ import annotations


class BudgetExhausted(RuntimeError):
    """Raised when a daily LLM/agent call budget is exhausted.

    Subclasses RuntimeError so existing ``except RuntimeError`` / ``except
    Exception`` handlers keep catching it unchanged. Callers that treat budget
    throttling as an *expected* condition (not a failure) can catch this
    specifically and log it quietly instead of at error level — the daily
    budget cap is a deliberate conservation control, not a fault.
    """
