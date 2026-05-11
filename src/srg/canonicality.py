"""Canonicality scoring — definition-level sources vs graph-central noise."""

from __future__ import annotations

from typing import Any

from .intent_classifier import QueryIntent
from .ranking_fusion import normalize_query_intent


def compute_canonicality(
    node: dict[str, Any],
    query: str,
    intent: str,
    *,
    inbound_citations: int = 0,
    is_in_survey_context: bool = False,
) -> float:
    """
    Measures whether a paper is the *definition-level source*
    rather than just graph-central.
    """
    q = (query or "").strip().lower()
    title = (node.get("title") or "").strip().lower()
    score = 0.0

    if q and q in title:
        score += 0.6
    elif q:
        qtok = [t for t in q.replace(",", " ").split() if len(t) > 2]
        if qtok:
            hits = sum(1 for t in qtok if t in title)
            score += 0.45 * min(1.0, hits / max(len(qtok), 1))

    score += min(float(inbound_citations) / 100.0, 0.3)

    if is_in_survey_context:
        score += 0.2

    ni = normalize_query_intent(intent)
    if ni in (QueryIntent.METHOD_LOOKUP, QueryIntent.DEFINITION):
        score *= 1.5

    return max(0.0, min(1.0, score))


def infer_survey_context_flag(node: dict[str, Any]) -> bool:
    """Best-effort proxy for 'survey / foundational neighborhood' without hardcoded paper lists."""
    if node.get("is_foundational_hub"):
        return True
    if float(node.get("relation_expand_score", 0.0) or 0.0) >= 0.14:
        return True
    origin = (node.get("seed_origin") or "").strip()
    if origin in ("uploaded_pdf", "api_search"):
        return True
    return False
