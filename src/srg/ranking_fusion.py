"""Intent-conditioned fusion — SRG Lite v2.1–v2.2 (distribution + recency triple)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .intent_classifier import QueryIntent


@dataclass(frozen=True)
class QueryContextV2:
    intent: str
    confidence: float


RECENCY_TRIPLE_INTENTS = frozenset(
    {
        QueryIntent.HARDWARE_SYSTEM,
        QueryIntent.TECHNICAL_METHOD,
        QueryIntent.RECENT_PROGRESS,
    }
)

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


def distribution_entropy(dist: dict[str, float]) -> float:
    """Shannon entropy of the intent distribution (natural log); higher ⇒ more mixed query."""
    if not dist:
        return 0.0
    vals = [float(v) for v in dist.values() if float(v) > 0]
    if not vals:
        return 0.0
    s = sum(vals)
    if s <= 0:
        return 0.0
    h = 0.0
    for v in vals:
        p = float(v) / s
        h -= p * math.log(p + 1e-15)
    return h


def distribution_uncertain(dist: dict[str, float]) -> bool:
    """PR B3 — blend fusion when no single intent dominates (max mass < 0.52)."""
    if not dist:
        return False
    mx = max(float(v) for v in dist.values())
    return mx < 0.52


def fusion_weights_for_intent(intent: str) -> tuple[float, float, float]:
    """Returns (semantic, graph, canonicality) for legacy paths (non–recency-triple)."""
    i = normalize_query_intent(intent)
    if i == QueryIntent.HISTORICAL_FOUNDATION:
        return (0.20, 0.50, 0.30)
    if i in (QueryIntent.DEFINITION, QueryIntent.METHOD_LOOKUP):
        return (0.4, 0.2, 0.4)
    if i in (QueryIntent.SURVEY, QueryIntent.COMPARISON):
        return (0.4, 0.4, 0.2)
    if i in RECENCY_TRIPLE_INTENTS:
        # Canonicality still used inside ``compute_query_conditioned_canonicality``; fusion uses triple elsewhere.
        return (0.45, 0.20, 0.35)
    return (0.6, 0.3, 0.1)


def _recency_triple_fused(
    semantic: float,
    graph: float,
    recency: float,
    *,
    hardware_recency_boost: bool,
) -> float:
    """PR A1 / B2 — fixed (semantic, recency, graph) for hardware / technical / recent_progress."""
    sem = max(0.0, min(1.0, float(semantic)))
    g = max(0.0, min(1.0, float(graph)))
    r = max(0.0, min(1.0, float(recency)))
    if hardware_recency_boost:
        return 0.40 * sem + 0.50 * r + 0.10 * g
    return 0.45 * sem + 0.35 * r + 0.20 * g


def _legacy_fused(
    semantic: float,
    graph: float,
    canonical: float,
    intent: str,
) -> float:
    w = fusion_weights_for_intent(intent)
    return w[0] * semantic + w[1] * graph + w[2] * canonical


def _fused_for_labeled_intent(
    semantic: float,
    graph: float,
    canonical: float,
    intent_label: str,
    recency: float,
    *,
    hardware_recency_boost: bool,
) -> float:
    i = normalize_query_intent(intent_label)
    if i in RECENCY_TRIPLE_INTENTS:
        return _recency_triple_fused(semantic, graph, recency, hardware_recency_boost=hardware_recency_boost)
    return _legacy_fused(semantic, graph, canonical, intent_label)


def compute_final_score(
    semantic: float,
    graph: float,
    canonical: float,
    intent: str,
    *,
    recency: float | None = None,
    intent_distribution: dict[str, float] | None = None,
    hardware_recency_boost: bool = False,
) -> float:
    """
    PR A1 — hardware / technical / recent_progress use (0.45, 0.35, 0.20) on (semantic, recency, graph).
    PR B3 — high-entropy ``intent_distribution`` blends fusion across labeled intents.
    PR A1 — if recency_alignment < 0.25 for a recency-triple primary intent, multiply fusion by 0.6.
    """
    primary = normalize_query_intent(intent)
    r_val = 0.55 if recency is None else float(recency)
    r_val = max(0.0, min(1.0, r_val))

    dist = intent_distribution or {}
    use_mix = bool(dist) and distribution_uncertain(dist)
    if use_mix:
        acc = 0.0
        tw = 0.0
        for k, w in dist.items():
            wf = float(w)
            if wf < 0.03:
                continue
            boost_k = hardware_recency_boost and normalize_query_intent(str(k)) == QueryIntent.HARDWARE_SYSTEM
            acc += wf * _fused_for_labeled_intent(
                semantic, graph, canonical, str(k), r_val, hardware_recency_boost=boost_k
            )
            tw += wf
        fused = acc / tw if tw > 0 else _fused_for_labeled_intent(
            semantic, graph, canonical, intent, r_val, hardware_recency_boost=hardware_recency_boost
        )
    else:
        fused = _fused_for_labeled_intent(
            semantic, graph, canonical, intent, r_val, hardware_recency_boost=hardware_recency_boost
        )

    if primary in RECENCY_TRIPLE_INTENTS and recency is not None and float(recency) < 0.25:
        fused *= 0.6
    return float(fused)


def primary_sem_rec_graph_key(
    semantic: float,
    graph: float,
    recency: float,
    *,
    hardware_recency_boost: bool,
) -> float:
    """PR G1 — dominance key for hardware top-10 stabilization (semantic + recency first)."""
    return _recency_triple_fused(semantic, graph, recency, hardware_recency_boost=hardware_recency_boost)


def apply_hardware_top10_sem_rec_graph_lock(
    nodes: list[dict[str, Any]],
    intent: str,
    *,
    hardware_recency_boost: bool = False,
) -> None:
    """
    PR G1 — for ``hardware_system``, ensure top-10 by ``relevance_raw`` does not invert
    ``primary_sem_rec_graph_key`` by more than two positions vs that key's ordering.
    """
    if normalize_query_intent(intent) != QueryIntent.HARDWARE_SYSTEM or not nodes:
        return

    def rec(n: dict[str, Any]) -> float:
        y = n.get("year")
        if not isinstance(y, int):
            return 0.55
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

    def pkey(n: dict[str, Any]) -> float:
        return primary_sem_rec_graph_key(
            float(n.get("semantic_score", 0.0) or 0.0),
            float(n.get("graph_score", 0.0) or 0.0),
            rec(n),
            hardware_recency_boost=hardware_recency_boost,
        )

    by_raw = sorted(nodes, key=lambda n: float(n.get("relevance_raw", 0.0)), reverse=True)
    by_primary = sorted(nodes, key=pkey, reverse=True)
    pos_p = {id(n): i for i, n in enumerate(by_primary)}
    for i, n in enumerate(by_raw[:12]):
        pi = pos_p[id(n)]
        # Graph ranked this node ``i`` places too high vs semantic+recency order ``pi``.
        if i < pi - 2:
            n["relevance_raw"] = float(n.get("relevance_raw", 0.0)) * 0.88
            n["hardware_top10_order_lock_applied"] = True
