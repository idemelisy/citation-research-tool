"""Per-paper trust strings — compact curated copy for SRG Lite."""

from __future__ import annotations

from typing import Any

from .query_intent_gating import attach_intent_category, compact_explanation


def build_why_it_matters(
    paper: dict[str, Any],
    *,
    query_text: str,
    intent_label: str,
) -> str:
    """Short editorial block — prefers compact intent-gating copy."""
    existing = (paper.get("why_it_matters") or "").strip()
    stale_markers = (
        "Core alignment with your query intent",
        "Published in the last three years",
        "Moderately recent",
        "citation neighborhood",
        "Anchored to your query seeds",
    )
    if existing and not any(m in existing for m in stale_markers):
        return existing

    qp = {
        "query_text": query_text,
        "intent_mode_v2": intent_label,
        "intent_entry_id": paper.get("intent_entry_id"),
        "intent_enrichment_terms": paper.get("intent_enrichment_terms") or [],
    }
    attach_intent_category(paper, qp)
    return compact_explanation(paper, qp)


def why_it_matters_one_line(paper: dict[str, Any], *, query_text: str, intent_label: str) -> str:
    """Single-sentence summary for export."""
    block = (paper.get("why_it_matters") or "").strip() or build_why_it_matters(
        paper, query_text=query_text, intent_label=intent_label
    )
    one = block.replace("\n", " ").strip()
    if len(one) > 240:
        one = one[:237] + "…"
    return one or compact_explanation(
        paper,
        {"query_text": query_text, "intent_mode_v2": intent_label},
    )


def ensure_why_it_matters_top5(
    nodes: list[dict[str, Any]],
    *,
    query_text: str,
    intent_label: str,
    k: int = 5,
) -> None:
    """Guarantee filled ``why_it_matters`` (+ one-line) for the top-k ranked nodes."""
    if not nodes:
        return
    ranked = sorted(
        nodes,
        key=lambda n: float(n.get("relevance_diverse_norm", n.get("relevance_norm", 0.0)) or 0.0),
        reverse=True,
    )
    for n in ranked[:k]:
        n["why_it_matters"] = build_why_it_matters(n, query_text=query_text, intent_label=intent_label)
        n["why_it_matters_one_line"] = why_it_matters_one_line(
            n, query_text=query_text, intent_label=intent_label
        )
