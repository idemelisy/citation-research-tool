"""
Reusable SRG Lite reading-list pipeline — same path as UI-lite, no Streamlit.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ..api_adapter import SRGApplicationService
from ..lite_ux import build_lite_ux_payload
from .defaults import discovery_options_with_top_k

logger = logging.getLogger(__name__)


def generate_reading_list(
    query: str,
    *,
    top_k: int = 40,
    max_branch_papers: int = 8,
    include_foundational: bool = True,
    include_start_here: bool = False,
    discovery_options: dict[str, Any] | None = None,
    service: SRGApplicationService | None = None,
    text_search_backend: str = "openalex",
) -> dict[str, Any]:
    """
    Run full retrieval + lite UX bundle for one query.

    Returns a JSON-serializable dict with foundational papers, branches,
    retrieval health, ranked raw candidates, and the full pipeline payload.
    """
    _ = include_start_here  # deprecated; Start Here removed from product output
    q = (query or "").strip()
    if len(q) < 3 and not re.search(r"\b10\.\d{4,9}/", q, flags=re.IGNORECASE):
        raise ValueError("Query must be at least 3 characters (or a DOI).")

    svc = service or SRGApplicationService()
    opts = dict(discovery_options or discovery_options_with_top_k(top_k))
    opts["top_n"] = max(opts.get("top_n", top_k), max(10, min(200, int(top_k))))

    logger.info("Discovering seeds for query=%r", q[:80])
    seeds = svc.discover_from_query(q, text_search_backend=text_search_backend)
    if not seeds:
        raise RuntimeError(f"No seeds resolved for query: {q!r}")

    logger.info("Running pipeline (%d seeds, top_n=%s)", len(seeds), opts.get("top_n"))
    payload = svc.run_pipeline(seeds, discipline=None, discovery_options=opts)

    if not payload.get("lite_ux"):
        payload["lite_ux"] = build_lite_ux_payload(payload)

    lite_ux = payload.get("lite_ux") or {}
    branches = _trim_branches(lite_ux.get("branches") or [], max_branch_papers)
    foundational = list(lite_ux.get("foundational_papers") or []) if include_foundational else []
    raw_candidates = _top_raw_candidates(payload.get("papers") or [], top_k)

    stats = {
        "paper_count": len(payload.get("papers") or []),
        "branch_count": len(branches),
        "foundational_count": len(foundational),
        "edge_count": len((payload.get("graph") or {}).get("edges") or []),
        "seed_count": len(payload.get("requested_seeds") or []),
        "generated_at": payload.get("generated_at"),
    }

    return {
        "query": q,
        "foundational_papers": foundational,
        "branches": branches,
        "retrieval_health": lite_ux.get("retrieval_health") or {},
        "insights": lite_ux.get("insights") or [],
        "stats": stats,
        "raw_candidates": raw_candidates,
        "query_profile": payload.get("query_profile") or {},
        "payload": payload,
        "lite_ux": lite_ux,
    }


def _trim_branches(branches: list[Any], max_papers: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cap = max(1, int(max_papers))
    for br in branches:
        if not isinstance(br, dict):
            continue
        nb = dict(br)
        pids = list(nb.get("paper_ids") or [])[:cap]
        nb["paper_ids"] = pids
        nb["member_count"] = len(pids)
        out.append(nb)
    return out


def _top_raw_candidates(papers: list[Any], top_k: int) -> list[dict[str, Any]]:
    clean = [p for p in papers if isinstance(p, dict) and not p.get("graph_noise")]
    ranked = sorted(
        clean,
        key=lambda p: float(
            p.get("relevance_diverse_norm", p.get("relevance_norm", p.get("relevance_raw", 0.0))) or 0.0
        ),
        reverse=True,
    )
    out: list[dict[str, Any]] = []
    for p in ranked[: max(1, int(top_k))]:
        out.append(
            {
                "paper_id": str(p.get("id") or ""),
                "title": (p.get("title") or "").strip(),
                "year": p.get("year"),
                "relevance": float(
                    p.get("relevance_diverse_norm", p.get("relevance_norm", 0.0)) or 0.0
                ),
                "semantic_score": float(p.get("semantic_score", 0.0) or 0.0),
                "domain": p.get("domain"),
                "seed_origin": p.get("seed_origin"),
            }
        )
    return out
