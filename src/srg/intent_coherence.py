"""
Intent-coherence scoring for SRG Lite — precision-focused retrieval.

Composable signals: lexical overlap, semantic fit, domain/venue/topic coherence,
publication era, anchor proximity, and subfield cluster penalties.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from .query_intent_normalization import (
    SubfieldCluster,
    all_clusters_for_entry,
    registry_entry_by_id,
)
from .semantic_control import domain_consistency_multiplier, vocabulary_coherence_score

_STOPWORDS = frozenset(
    "the a an of for to in on and or with by as at from into over per via using "
    "method methods approach model models learning data based new using study "
    "analysis work results between among".split()
)

# Era bands for alignment queries (LLM era ~2018+)
_ALIGNMENT_MODERN_YEAR = 2018
_EXPLORATORY_BRIDGE_EXPAND_MAX = 0.14
_EXPLORATORY_BRIDGE_SEMANTIC_MAX = 0.38
_COHERENCE_GATE_DEFAULT = 0.32
_COHERENCE_GATE_STRICT = 0.42


def _blob(node: dict[str, Any]) -> str:
    return f"{node.get('title', '')} {str(node.get('abstract') or '')[:2800]}".lower()


def _cluster_hit(blob: str, cluster: SubfieldCluster) -> float:
    if not cluster.markers:
        return 0.0
    hits = sum(1 for m in cluster.markers if m in blob)
    return min(1.0, hits / max(1, min(3, len(cluster.markers))))


def subfield_cluster_scores(
    node: dict[str, Any],
    boost_clusters: tuple[SubfieldCluster, ...],
    penalty_clusters: tuple[SubfieldCluster, ...],
) -> tuple[float, float, str | None]:
    """
    Returns (boost_strength, penalty_strength, dominant_penalty_cluster_name).
    """
    b = _blob(node)
    boost = max((_cluster_hit(b, c) for c in boost_clusters), default=0.0)
    pen = 0.0
    pen_name: str | None = None
    for c in penalty_clusters:
        h = _cluster_hit(b, c)
        if h > pen:
            pen = h
            pen_name = c.name
    return boost, pen, pen_name


def lexical_similarity(query: str, node: dict[str, Any]) -> float:
    q = (query or "").lower()
    title = (node.get("title") or "").lower()
    if not q or not title:
        return 0.0
    qtok = [t for t in re.split(r"[^\w]+", q) if len(t) > 2 and t not in _STOPWORDS]
    if not qtok:
        return 0.0
    hits = sum(1 for t in set(qtok) if t in title)
    return min(1.0, hits / max(len(set(qtok)), 1))


def era_coherence_score(node: dict[str, Any], *, modern_floor_year: int = _ALIGNMENT_MODERN_YEAR) -> float:
    y = node.get("year")
    if not isinstance(y, int):
        return 0.55
    cy = datetime.utcnow().year
    if y >= modern_floor_year:
        return 1.0
    age = cy - y
    if age <= 25:
        return 0.72
    if age <= 35:
        return 0.5
    return 0.35


def anchor_consistency_score(hops: int | None, *, unreachable: int = 98) -> float:
    if hops is None:
        return 0.45
    h = int(hops)
    if h >= unreachable:
        return 0.25
    if h <= 1:
        return 1.0
    if h == 2:
        return 0.82
    if h == 3:
        return 0.62
    if h <= 5:
        return 0.45
    return 0.28


def venue_topic_coherence(node: dict[str, Any], enrichment_terms: list[str]) -> float:
    venue = (node.get("venue") or "").lower()
    blob = _blob(node)
    if not enrichment_terms:
        return 0.5
    hits = sum(1 for t in enrichment_terms if t.lower() in blob or t.lower() in venue)
    return min(1.0, hits / max(1, min(4, len(enrichment_terms))))


def is_exploratory_bridge(
    node: dict[str, Any],
    *,
    semantic_score: float,
    expand_score: float,
    hops: int | None,
    penalty_cluster_hit: float,
) -> bool:
    """Citation-adjacent nodes with weak semantic/intent fit — keep in graph, downrank in UX."""
    if node.get("seed_origin") in ("uploaded_pdf", "api_search"):
        return False
    if node.get("is_foundational_hub") and node.get("foundational_eligible", True):
        return False
    rel_exp = float(node.get("relation_expand_score", 0.0) or 0.0)
    if rel_exp >= 0.22 and semantic_score >= 0.42:
        return False
    if penalty_cluster_hit >= 0.34:
        return True
    if (
        float(expand_score or rel_exp) >= 0.08
        and semantic_score < _EXPLORATORY_BRIDGE_SEMANTIC_MAX
        and (hops is None or int(hops) >= 3)
    ):
        return True
    if float(expand_score or rel_exp) < _EXPLORATORY_BRIDGE_EXPAND_MAX and semantic_score < 0.32:
        if hops is not None and int(hops) >= 4:
            return True
    return False


def explanation_confidence_tier(
    intent_coherence: float,
    semantic_score: float,
    *,
    is_bridge: bool,
    penalty_cluster: str | None,
) -> str:
    """Honest label for explainability strings."""
    if is_bridge or penalty_cluster:
        if penalty_cluster:
            return "cross_domain_bridge"
        return "exploratory_connection"
    if intent_coherence >= 0.62 and semantic_score >= 0.45:
        return "core_intent"
    if intent_coherence >= 0.45 or semantic_score >= 0.35:
        return "moderate_intent"
    if semantic_score >= 0.22:
        return "lexically_related"
    return "citation_adjacent"


def compute_intent_coherence(
    node: dict[str, Any],
    *,
    query_text: str,
    enrichment_terms: list[str],
    intent_entry_id: str | None,
    query_domains: frozenset[str],
    semantic_score: float | None = None,
    anchor_hops: int | None = None,
) -> dict[str, Any]:
    """
    Full coherence bundle for one paper node.
    Returns scores in [0,1] plus flags for filtering and explanations.
    """
    sem = float(semantic_score if semantic_score is not None else node.get("semantic_score", 0.0) or 0.0)
    lex = lexical_similarity(query_text, node)
    vocab = vocabulary_coherence_score(
        " ".join([query_text] + list(enrichment_terms or [])), node
    )
    venue_c = venue_topic_coherence(node, enrichment_terms or [])
    dom_m = domain_consistency_multiplier(node.get("domain"), query_domains)
    era = era_coherence_score(node)
    hops = anchor_hops if anchor_hops is not None else node.get("anchor_graph_hops")
    anchor_c = anchor_consistency_score(hops if hops is not None else None)

    boost_clusters, penalty_clusters = all_clusters_for_entry(intent_entry_id)
    boost_h, pen_h, pen_name = subfield_cluster_scores(node, boost_clusters, penalty_clusters)

    # Weighted fusion — semantic + lexical dominate; penalties are multiplicative caps
    raw = (
        0.28 * sem
        + 0.22 * lex
        + 0.14 * vocab
        + 0.12 * venue_c
        + 0.10 * anchor_c
        + 0.08 * era
        + 0.06 * min(1.0, dom_m)
    )
    raw += 0.12 * boost_h
    if pen_h >= 0.34:
        raw *= max(0.18, 1.0 - 0.72 * pen_h)
    elif pen_h >= 0.2:
        raw *= max(0.42, 1.0 - 0.45 * pen_h)

    score = max(0.0, min(1.0, raw))
    expand_sc = float(node.get("relation_expand_score", 0.0) or 0.0)
    bridge = is_exploratory_bridge(
        node,
        semantic_score=sem,
        expand_score=expand_sc,
        hops=int(hops) if hops is not None else None,
        penalty_cluster_hit=pen_h,
    )
    tier = explanation_confidence_tier(score, sem, is_bridge=bridge, penalty_cluster=pen_name)

    return {
        "intent_coherence_score": round(score, 4),
        "intent_lexical_similarity": round(lex, 4),
        "intent_semantic_component": round(sem, 4),
        "intent_vocabulary_coherence": round(vocab, 4),
        "intent_venue_topic_coherence": round(venue_c, 4),
        "intent_era_coherence": round(era, 4),
        "intent_anchor_consistency": round(anchor_c, 4),
        "intent_subfield_boost": round(boost_h, 4),
        "intent_subfield_penalty": round(pen_h, 4),
        "intent_penalty_cluster": pen_name,
        "is_exploratory_bridge": bridge,
        "explanation_confidence_tier": tier,
    }


def coherence_gate_threshold(ambiguity_score: float) -> float:
    """Higher ambiguity ⇒ stricter pre-expansion filter."""
    amb = max(0.0, min(1.0, float(ambiguity_score)))
    return _COHERENCE_GATE_DEFAULT + amb * (_COHERENCE_GATE_STRICT - _COHERENCE_GATE_DEFAULT)


def passes_coherence_gate(
    coherence_score: float,
    *,
    ambiguity_score: float,
    is_seed: bool,
) -> bool:
    if is_seed:
        return True
    return float(coherence_score) >= coherence_gate_threshold(ambiguity_score)


def conservative_discovery_adjustments(ambiguity_score: float) -> dict[str, Any]:
    """
    Higher ambiguity ⇒ narrower expansion (NOT broader).
    Returns option overrides for ``run_pipeline`` / ``LITE_DISCOVERY_OPTIONS``.
    """
    amb = max(0.0, min(1.0, float(ambiguity_score)))
    if amb < 0.5:
        return {}
    factor = (amb - 0.5) / 0.5
    return {
        "two_hop_budget_scale": max(0.55, 1.0 - 0.35 * factor),
        "concept_threshold_boost": 0.05 + 0.12 * factor,
        "relation_topic_weight_cap": max(0.0, 0.15 - 0.15 * factor),
        "semantic_rank_min_score_boost": 0.04 + 0.08 * factor,
        "coherence_filter_enabled": True,
        "bridge_downrank_factor": 0.35 + 0.25 * factor,
    }


def apply_coherence_ranking_adjustment(
    relevance_raw: float,
    bundle: dict[str, Any],
    *,
    bridge_downrank_factor: float = 0.45,
) -> float:
    """Multiply fused relevance by coherence; heavily downrank exploratory bridges."""
    coh = float(bundle.get("intent_coherence_score", 0.5) or 0.5)
    mul = 0.55 + 0.45 * coh
    out = float(relevance_raw) * mul
    if bundle.get("is_exploratory_bridge"):
        out *= max(0.15, float(bridge_downrank_factor))
    pen = float(bundle.get("intent_subfield_penalty", 0.0) or 0.0)
    if pen >= 0.34:
        out *= 0.35
    elif pen >= 0.22:
        out *= 0.62
    return max(0.0, out)


__all__ = [
    "anchor_consistency_score",
    "apply_coherence_ranking_adjustment",
    "coherence_gate_threshold",
    "compute_intent_coherence",
    "conservative_discovery_adjustments",
    "era_coherence_score",
    "explanation_confidence_tier",
    "is_exploratory_bridge",
    "lexical_similarity",
    "passes_coherence_gate",
    "subfield_cluster_scores",
]
