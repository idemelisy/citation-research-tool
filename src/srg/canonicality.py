"""Canonicality scoring — definition-level sources vs graph-central noise."""

from __future__ import annotations

from datetime import datetime

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


def anchor_proximity_bonus_hops(anchor_hops: int, *, unreachable: int = 98) -> float:
    """
    SRG Lite v2.2 PR A2 — discrete hop bonus: 1→1.0, 2→0.75, 3→0.45, >3→0.10, unknown→0.
    """
    if anchor_hops >= unreachable:
        return 0.0
    h = int(anchor_hops)
    if h <= 1:
        return 1.0
    if h == 2:
        return 0.75
    if h == 3:
        return 0.45
    return 0.10


def anchor_proximity_strength_normalized(anchor_hops: int, *, unreachable: int = 98) -> float:
    """Same scale as ``anchor_proximity_bonus_hops`` for ordering / anchor-diff rules (PR A2)."""
    return anchor_proximity_bonus_hops(anchor_hops, unreachable=unreachable)


def recency_penalty_decay(node: dict[str, Any], intent: str) -> float:
    """Small penalty when the query type expects modern work but the paper is old."""
    ni = normalize_query_intent(intent)
    if ni not in (
        QueryIntent.HARDWARE_SYSTEM,
        QueryIntent.RECENT_PROGRESS,
        QueryIntent.TECHNICAL_METHOD,
    ):
        return 0.0
    y = node.get("year")
    if not isinstance(y, int):
        return 0.04
    cy = datetime.utcnow().year
    age = max(0, cy - y)
    if age <= 4:
        return 0.0
    if age <= 10:
        return 0.06
    if age <= 18:
        return 0.10
    return 0.14


def compute_query_conditioned_canonicality(
    node: dict[str, Any],
    query: str,
    intent: str,
    *,
    inbound_citations: int = 0,
    is_in_survey_context: bool = False,
    anchor_hops: int = 99,
    semantic_alignment: float = 0.5,
    vocabulary_coherence: float = 0.5,
) -> float:
    """
    Roadmap Phase 1.1 — additive blend: base + anchor proximity bonus + semantic +
    vocabulary − recency penalty, squashed to [0, 1] so anchor proximity can lift
    query-near papers without citation count alone dominating.
    """
    base = compute_canonicality(
        node,
        query,
        intent,
        inbound_citations=inbound_citations,
        is_in_survey_context=is_in_survey_context,
    )
    prox = anchor_proximity_bonus_hops(anchor_hops)
    sem_a = max(0.0, min(1.0, float(semantic_alignment)))
    voc_a = max(0.0, min(1.0, float(vocabulary_coherence)))
    pen = recency_penalty_decay(node, intent)
    combined = 0.32 * base + 0.28 * prox + 0.22 * sem_a + 0.18 * voc_a - pen
    # PR A3 — citation-age guardrail (global citations + age vs fresh semantic fit)
    y = node.get("year")
    cy = datetime.utcnow().year
    age = max(0, cy - int(y)) if isinstance(y, int) else 12
    total_cit_signal = max(int(inbound_citations), int(node.get("citation_count", 0) or 0))
    if total_cit_signal > 500 and age > 10:
        combined *= 0.85
    if isinstance(y, int) and (cy - y) <= 7 and sem_a >= 0.6:
        combined += 0.10
    return max(0.0, min(1.0, combined))


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
