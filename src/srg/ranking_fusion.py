"""Intent-conditioned fusion — SRG Lite v2.1 §6."""

from __future__ import annotations

from dataclasses import dataclass

from .intent_classifier import QueryIntent


@dataclass(frozen=True)
class QueryContextV2:
    intent: str
    confidence: float


def normalize_query_intent(intent: str) -> str:
    """Map legacy v2.0 intent strings to v2.1 taxonomy."""
    legacy = {
        QueryIntent.CANONICAL_LOOKUP: QueryIntent.METHOD_LOOKUP,
        "canonical_lookup": QueryIntent.METHOD_LOOKUP,
        QueryIntent.SURVEY_LEGACY: QueryIntent.SURVEY,
        "survey_mode": QueryIntent.SURVEY,
        QueryIntent.EXPLORATION_LEGACY: QueryIntent.EXPLORATORY,
        "exploratory_mode": QueryIntent.EXPLORATORY,
    }
    return legacy.get(intent, intent)


def fusion_weights_for_intent(intent: str) -> tuple[float, float, float]:
    """Returns (semantic, graph, canonicality) weights summing to 1 (§6.2)."""
    i = normalize_query_intent(intent)
    if i in (QueryIntent.DEFINITION, QueryIntent.METHOD_LOOKUP):
        return (0.4, 0.2, 0.4)
    if i in (QueryIntent.SURVEY, QueryIntent.COMPARISON):
        return (0.4, 0.4, 0.2)
    return (0.6, 0.3, 0.1)


def compute_final_score(
    semantic: float,
    graph: float,
    canonical: float,
    intent: str,
) -> float:
    w = fusion_weights_for_intent(intent)
    return w[0] * semantic + w[1] * graph + w[2] * canonical
