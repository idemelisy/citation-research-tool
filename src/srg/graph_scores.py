"""Graph-derived scores — SRG Lite v2.1 §5.2: PR + edge quality + neighborhood coherence."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

_CONF_MAP = {"high": 1.0, "medium": 0.62, "low": 0.28}


def _normalize_scores(raw: dict[str, float]) -> dict[str, float]:
    if not raw:
        return {}
    vals = list(raw.values())
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return {k: 0.5 for k in raw}
    return {k: (v - lo) / (hi - lo) for k, v in raw.items()}


def pagerank_scores(
    edges: list[dict[str, Any]],
    node_ids: list[str],
    *,
    damping: float = 0.85,
    iterations: int = 28,
) -> dict[str, float]:
    """
    Directed edges: source cites target → random walk follows source → target.
    """
    nodes = list(dict.fromkeys(node_ids))
    n = len(nodes)
    if n == 0:
        return {}
    idx = {nid: i for i, nid in enumerate(nodes)}
    out_adj: dict[int, list[int]] = defaultdict(list)
    for e in edges:
        s, t = e.get("source"), e.get("target")
        if not s or not t:
            continue
        si, ti = idx.get(str(s)), idx.get(str(t))
        if si is None or ti is None:
            continue
        out_adj[si].append(ti)

    teleport = 1.0 / n
    r = [teleport] * n
    for _ in range(iterations):
        nxt = [(1.0 - damping) * teleport] * n
        for i in range(n):
            outs = out_adj.get(i)
            if not outs:
                share = damping * r[i] / n
                for j in range(n):
                    nxt[j] += share
            else:
                share = damping * r[i] / len(outs)
                for j in outs:
                    nxt[j] += share
        r = nxt

    return {nodes[i]: float(r[i]) for i in range(n)}


def edge_quality_scores(edges: list[dict[str, Any]], node_ids: list[str]) -> dict[str, float]:
    """Mean normalized confidence + context presence on incident edges."""
    idx_set = set(node_ids)
    sums: dict[str, list[float]] = defaultdict(list)
    for e in edges:
        s, t = str(e.get("source") or ""), str(e.get("target") or "")
        if s not in idx_set or t not in idx_set:
            continue
        conf_raw = e.get("confidence")
        if isinstance(conf_raw, str):
            base = _CONF_MAP.get(conf_raw.strip().lower(), 0.45)
        else:
            base = 0.45
        ctx = e.get("context")
        ctx_bonus = 0.15 if isinstance(ctx, str) and len(ctx.strip()) > 12 else 0.0
        val = min(1.0, base + ctx_bonus)
        sums[s].append(val)
        sums[t].append(val)

    out: dict[str, float] = {}
    for nid in node_ids:
        vals = sums.get(nid)
        if vals:
            out[nid] = sum(vals) / len(vals)
        else:
            out[nid] = 0.18
    return _normalize_scores(out)


def neighborhood_coherence_scores(
    hop_dist: dict[str, int] | None,
    node_ids: list[str],
) -> dict[str, float]:
    """
    Local neighborhood coherence from anchor distance (§5.2 context_score).
    High near seeds / low hop count; weak when hop_dist unavailable.
    """
    raw: dict[str, float] = {}
    if not hop_dist:
        return {nid: 0.5 for nid in node_ids}
    for nid in node_ids:
        h = int(hop_dist.get(nid, 99))
        # Near anchors → high coherence; distant / unknown → low.
        raw[nid] = 1.0 / (1.0 + min(h, 12) * 0.22)
    return _normalize_scores(raw)


def compute_graph_score_bundle(
    edges: list[dict[str, Any]],
    node_ids: list[str],
    hop_dist: dict[str, int] | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    """
    Returns:
        pagerank_norm, edge_quality_norm, context_coherence_norm,
        S_graph combined ∈ [0,1] (§5.2).
    """
    pr_raw = pagerank_scores(edges, node_ids)
    pr_n = _normalize_scores(pr_raw)
    eq_n = edge_quality_scores(edges, node_ids)
    ctx_n = neighborhood_coherence_scores(hop_dist, node_ids)
    combined: dict[str, float] = {}
    for nid in node_ids:
        combined[nid] = max(
            0.0,
            min(
                1.0,
                0.38 * float(pr_n.get(nid, 0.5))
                + 0.32 * float(eq_n.get(nid, 0.5))
                + 0.30 * float(ctx_n.get(nid, 0.5)),
            ),
        )
    return pr_n, eq_n, ctx_n, combined
