"""Observability helpers — SRG Lite v2.1 §8."""

from __future__ import annotations

import statistics
from typing import Any


def _stats(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {}
    return {
        "mean": float(statistics.mean(vals)),
        "stdev": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def signal_distributions(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    """§8 — score distribution per signal."""
    keys = ("semantic_score", "graph_score", "canonicality_score", "relevance_raw")
    out: dict[str, Any] = {}
    for k in keys:
        vals = [float(n.get(k) or 0.0) for n in nodes]
        st = _stats(vals)
        if st:
            out[k] = st
    return out


def fusion_contribution_breakdown(
    nodes: list[dict[str, Any]],
    w_sem: float,
    w_graph: float,
    w_canon: float,
) -> dict[str, Any]:
    """Average absolute contribution of each fused term (interpretability)."""
    if not nodes:
        return {}
    c_sem = [w_sem * float(n.get("semantic_score") or 0.0) for n in nodes]
    c_g = [w_graph * float(n.get("graph_score") or 0.0) for n in nodes]
    c_c = [w_canon * float(n.get("canonicality_score") or 0.0) for n in nodes]
    return {
        "weighted_semantic": _stats(c_sem),
        "weighted_graph": _stats(c_g),
        "weighted_canonicality": _stats(c_c),
    }


def edge_confidence_histogram(edges: list[dict[str, Any]]) -> dict[str, int]:
    buckets = {"high": 0, "medium": 0, "low": 0, "other": 0}
    for e in edges:
        c = e.get("confidence")
        if isinstance(c, str):
            k = c.strip().lower()
            if k in buckets:
                buckets[k] += 1
            else:
                buckets["other"] += 1
        else:
            buckets["other"] += 1
    return buckets


def lite_v21_observability_payload(
    *,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    references_total: int,
    n_candidates: int,
    n_edges: int,
    intent: str,
    intent_confidence: float,
    fusion_weights: tuple[float, float, float],
) -> dict[str, Any]:
    """§8 aggregate diagnostics for API payloads."""
    w1, w2, w3 = fusion_weights
    sem_vals = [float(n.get("semantic_score") or 0.0) for n in nodes]
    spread = _stats(sem_vals).get("stdev", 0.0) if sem_vals else 0.0

    pr_vals = [float(n.get("pagerank_norm") or 0.0) for n in nodes]
    pr_skew_hint = _stats(pr_vals).get("max", 0.0) - _stats(pr_vals).get("mean", 0.0) if pr_vals else 0.0

    ref_ratio = 1.0 if n_candidates <= 0 else min(1.0, float(references_total) / max(1.0, float(n_candidates) * 8.0))

    return {
        "intent_classification": {
            "intent": intent,
            "confidence": intent_confidence,
        },
        "fusion_weights": {"semantic": w1, "graph": w2, "canonicality": w3},
        "fusion_contribution": fusion_contribution_breakdown(nodes, w1, w2, w3),
        "signal_distributions": signal_distributions(nodes),
        "semantic_dispersion": spread,
        "candidate_coverage": {
            "n_candidates": n_candidates,
            "n_edges": n_edges,
            "expansion_density": float(n_edges) / max(1.0, float(n_candidates)),
        },
        "graph_health": {
            "edge_confidence_histogram": edge_confidence_histogram(edges),
            "centrality_max_minus_mean": round(float(pr_skew_hint), 4),
            "reference_completeness_ratio": round(float(ref_ratio), 4),
        },
    }
