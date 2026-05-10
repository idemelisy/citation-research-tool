"""Adaptive fusion reranker — replaces static semantic/graph weights (SRG Lite v2)."""

from __future__ import annotations

from dataclasses import dataclass

from .intent_classifier import QueryIntent


@dataclass(frozen=True)
class QueryContextV2:
    intent: str
    confidence: float


def fusion_weights_for_intent(intent: str) -> tuple[float, float, float]:
    """Returns (semantic, graph, canonicality) weights summing to 1."""
    if intent == QueryIntent.CANONICAL_LOOKUP:
        return (0.4, 0.2, 0.4)
    if intent == QueryIntent.SURVEY:
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
