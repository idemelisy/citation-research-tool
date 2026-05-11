"""
SRG Lite roadmap — Phase 3 scaffolding: failure taxonomy and evaluation hooks.

Full benchmarks (Phase 3.1) require curated queries + human labels; this module
centralizes category names and simple offline metrics for future use.
"""

from __future__ import annotations

from typing import Any

# Phase 3.2 — failure taxonomy (roadmap)
FAILURE_SEMANTIC_DRIFT = "semantic_drift"
FAILURE_FOUNDATIONAL_OVERSHADOWING = "foundational_overshadowing"
FAILURE_CLUSTER_LEAKAGE = "cluster_leakage"
FAILURE_RECENCY_FAILURE = "recency_failure"
FAILURE_OVER_SPECIALIZATION = "over_specialization"

FAILURE_TAXONOMY: tuple[str, ...] = (
    FAILURE_SEMANTIC_DRIFT,
    FAILURE_FOUNDATIONAL_OVERSHADOWING,
    FAILURE_CLUSTER_LEAKAGE,
    FAILURE_RECENCY_FAILURE,
    FAILURE_OVER_SPECIALIZATION,
)


def precision_at_k(
    ranked_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Standard Precision@k for offline eval (Phase 3.1)."""
    if k <= 0 or not ranked_ids:
        return 0.0
    top = ranked_ids[:k]
    hits = sum(1 for x in top if x in relevant_ids)
    return hits / float(min(k, len(top)))


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """MRR contribution for a single query (first relevant rank)."""
    for i, rid in enumerate(ranked_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / float(i)
    return 0.0


def precision_at_5(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """PR F2 — Precision@5."""
    return precision_at_k(ranked_ids, relevant_ids, 5)


def recall_at_10(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """PR F2 — recall mass captured in the top-10 ranked slice."""
    if not relevant_ids:
        return 0.0
    top = set(ranked_ids[:10])
    hits = sum(1 for rid in relevant_ids if rid in top)
    return hits / float(len(relevant_ids))


def intent_accuracy(predicted_intent: str, gold_intent: str) -> float:
    """PR F2 — 1.0 when normalized intents match."""
    from .ranking_fusion import normalize_query_intent

    return 1.0 if normalize_query_intent(predicted_intent) == normalize_query_intent(gold_intent) else 0.0


def hardware_recency_bias_score(
    ranked_nodes: list[dict[str, Any]],
    *,
    query_intent: str,
    top_k: int = 5,
) -> float:
    """
    PR F2 — for hardware queries, mean publication-recency alignment of top-k nodes
    (proxy for recency-aware ranking quality; requires ``year`` on nodes).
    """
    from .intent_classifier import QueryIntent
    from .ranking_fusion import normalize_query_intent

    if normalize_query_intent(query_intent) != QueryIntent.HARDWARE_SYSTEM:
        return 0.0
    slice_n = ranked_nodes[: max(1, top_k)]

    def recency_alignment(node: dict[str, Any]) -> float:
        y = node.get("year")
        if not isinstance(y, int):
            return 0.55
        from datetime import datetime

        cy = datetime.utcnow().year
        age = max(0, cy - y)
        if age <= 2:
            return 1.0
        if age <= 5:
            return 0.82
        if age <= 10:
            return 0.58
        if age <= 18:
            return 0.35
        return 0.18

    vals = [recency_alignment(n) for n in slice_n]
    return sum(vals) / float(len(vals))


def summarize_payload_for_eval(payload: dict[str, Any]) -> dict[str, Any]:
    """Lightweight snapshot for logging / future regression tests."""
    papers = [p for p in (payload.get("papers") or []) if not p.get("graph_noise")]
    ranked = sorted(
        papers,
        key=lambda p: float(p.get("relevance_diverse_norm", p.get("relevance_norm", 0.0)) or 0.0),
        reverse=True,
    )
    return {
        "n_papers": len(papers),
        "top10_ids": [str(p.get("id")) for p in ranked[:10]],
        "query_profile": payload.get("query_profile") or {},
        "intent_mode_v2": (payload.get("query_profile") or {}).get("intent_mode_v2"),
    }
