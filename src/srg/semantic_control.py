"""
Intent-Aware Semantic Control Layer (SRG Lite v2 spec extension).

Reduces semantic drift during graph expansion via hop-based drift penalty,
local semantic connectivity, domain consistency, and degree-aware hub suppression.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

# Substrings in user query → internal domain bucket (aligned with domain_mapper / LITE allowed_domains)
QUERY_DOMAIN_HINTS: list[tuple[str, str]] = [
    ("bioinform", "biomedical"),
    ("genomics", "biomedical"),
    ("protein", "biomedical"),
    ("clinical", "biomedical"),
    ("medical", "biomedical"),
    ("neuroscience", "biomedical"),
    ("physics", "physics"),
    ("quantum", "physics"),
    ("chemistry", "physics"),
    ("law", "law"),
    ("legal", "law"),
    ("policy", "social_sciences"),
    ("economics", "social_sciences"),
    ("psychology", "social_sciences"),
    ("sociology", "social_sciences"),
    ("ethics", "ethics"),
    ("database", "cs"),
    ("sql", "cs"),
    ("distributed system", "cs"),
    ("operating system", "cs"),
    ("compiler", "cs"),
    ("nlp", "cs"),
    ("language model", "cs"),
    ("transformer", "cs"),
    ("machine learning", "cs"),
    ("deep learning", "cs"),
    ("neural", "cs"),
    ("graphics", "cs"),
    ("rendering", "cs"),
    ("tessellation", "cs"),
    ("gpu", "cs"),
    ("shader", "cs"),
    ("geometry processing", "cs"),
    ("mesh", "cs"),
    ("computer vision", "cs"),
    ("image segmentation", "cs"),
    ("robotics", "cs"),
    ("hardware accelerated", "cs"),
    ("accelerated", "cs"),
]


def infer_query_domains(query: str) -> frozenset[str]:
    """Lightweight query → domain hints (empty = no domain filter)."""
    q = (query or "").strip().lower()
    if not q:
        return frozenset()
    out: set[str] = set()
    for needle, dom in QUERY_DOMAIN_HINTS:
        if needle in q:
            out.add(dom)
    return frozenset(out)


def domain_consistency_multiplier(
    node_domain: str | None,
    query_domains: frozenset[str],
    *,
    neutral: float = 0.82,
    match: float = 1.0,
    mismatch: float = 0.58,
) -> float:
    """
    Down-weight nodes whose discipline bucket disagrees with query-inferred domains.
    When query_domains is empty, no penalty (1.0).
    """
    if not query_domains:
        return 1.0
    nd = (node_domain or "").strip().lower()
    if not nd:
        return neutral
    if nd in query_domains:
        return match
    return mismatch


def _edge_weight(conf: Any) -> float:
    if isinstance(conf, str):
        c = conf.strip().lower()
        if c == "high":
            return 1.0
        if c == "medium":
            return 0.65
        if c == "low":
            return 0.35
    return 0.5


def build_undirected_adj(edges: list[dict[str, Any]]) -> dict[str, list[tuple[str, float]]]:
    adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for e in edges:
        s, t = str(e.get("source") or ""), str(e.get("target") or "")
        if not s or not t:
            continue
        w = _edge_weight(e.get("confidence"))
        adj[s].append((t, w))
        adj[t].append((s, w))
    return adj


def local_semantic_connectivity(
    node_ids: list[str],
    sem_by_id: dict[str, float],
    edges: list[dict[str, Any]],
) -> dict[str, float]:
    """
    γ term: average semantic alignment of neighbors, weighted by edge confidence.
    ∈ [0, 1]; isolated nodes get neutral 0.45.
    """
    adj = build_undirected_adj(edges)
    out: dict[str, float] = {}
    for nid in node_ids:
        nbrs = adj.get(nid)
        if not nbrs:
            out[nid] = 0.45
            continue
        num = 0.0
        den = 0.0
        for other, ew in nbrs:
            sv = float(sem_by_id.get(other, 0.0))
            num += ew * sv
            den += ew
        out[nid] = max(0.0, min(1.0, num / den)) if den > 0 else 0.45
    return out


def hub_suppressed_graph_scores(
    graph_score_by_id: dict[str, float],
    degree_by_id: dict[str, int],
) -> dict[str, float]:
    """
    Divide structural graph score by log(1 + degree) then min-max normalize to [0,1].
    """
    raw: dict[str, float] = {}
    for nid, gs in graph_score_by_id.items():
        deg = max(1, int(degree_by_id.get(nid, 1)))
        raw[nid] = float(gs) / max(1e-6, math.log1p(deg))
    if not raw:
        return {}
    vals = list(raw.values())
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return {k: 0.5 for k in raw}
    return {k: max(0.0, min(1.0, (v - lo) / (hi - lo))) for k, v in raw.items()}


def normalized_drift_penalty(hop: int, *, cap: int = 10, unreachable: int = 98) -> float:
    """
    ∈ [0, 1] — long citation chains from anchors score higher penalty.
    Hops >= ``unreachable`` (no path from anchors in BFS) → no drift penalty.
    """
    hi = int(hop)
    if hi >= unreachable:
        return 0.0
    h = max(0, min(hi, cap))
    return float(h) / float(cap)


def aggregate_semantic_control_diagnostics(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Summary for v2.1 observability payload after nodes are scored."""
    if not nodes:
        return {}
    dr = [float(n.get("semantic_drift_penalty_norm") or 0.0) for n in nodes]
    lc = [float(n.get("local_semantic_connectivity") or 0.0) for n in nodes]
    dm = [float(n.get("domain_consistency_multiplier") or 1.0) for n in nodes]
    qdom: list[str] = []
    for n in nodes:
        qd = n.get("query_inferred_domains")
        if qd:
            qdom = list(qd) if isinstance(qd, list) else sorted(qd)
            break
    return {
        "mean_drift_penalty_norm": round(sum(dr) / len(dr), 4),
        "mean_local_semantic_connectivity": round(sum(lc) / len(lc), 4),
        "mean_domain_consistency_multiplier": round(sum(dm) / len(dm), 4),
        "query_inferred_domains": qdom,
    }


def semantic_control_adjustments(
    *,
    fused_base: float,
    local_conn: float,
    drift_pen: float,
    domain_mult: float,
    gamma: float,
    delta: float,
) -> tuple[float, float, float, float]:
    """
    Apply Score += γ·LocalConnectivity − δ·DriftPenalty, then domain multiplier.
    Returns (final_score, drift_term_subtracted, local_bonus_added, domain_mult_applied).
    """
    drift_term = delta * drift_pen
    local_bonus = gamma * local_conn
    raw = fused_base + local_bonus - drift_term
    raw = max(0.0, raw * domain_mult)
    return raw, drift_term, local_bonus, domain_mult
