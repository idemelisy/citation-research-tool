"""
Canonical flagship papers per normalized query intent.

Used to hard-pin Start Here — bypasses graph-degree instability.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FlagshipSpec:
    """One canonical paper — match by arXiv id and/or title substring."""

    arxiv_id: str | None = None
    title_contains: str | None = None
    label: str = ""


# Ordered: first entries are highest priority for Start Here slot 1–2.
FLAGSHIP_BY_INTENT: dict[str, tuple[FlagshipSpec, ...]] = {
    "direct_preference_optimization": (
        FlagshipSpec(
            arxiv_id="2305.18290",
            title_contains="direct preference optimization",
            label="DPO",
        ),
        FlagshipSpec(title_contains="cal-dpo", label="Cal-DPO"),
        FlagshipSpec(title_contains="rs-dpo", label="RS-DPO"),
        FlagshipSpec(arxiv_id="2404.19733", title_contains="orpo", label="ORPO"),
        FlagshipSpec(title_contains="simpo", label="SimPO"),
        FlagshipSpec(title_contains="rrhf", label="RRHF"),
    ),
    "rlhf_alignment": (
        FlagshipSpec(arxiv_id="2203.02155", title_contains="training language models to follow instructions"),
        FlagshipSpec(arxiv_id="2305.18290", title_contains="direct preference optimization"),
    ),
    "transformer_nlp": (
        FlagshipSpec(
            arxiv_id="1706.03762",
            title_contains="attention is all you need",
            label="Transformer",
        ),
    ),
    "graph_neural_networks": (
        FlagshipSpec(arxiv_id="1609.02907", title_contains="graph convolutional"),
        FlagshipSpec(arxiv_id="1706.02216", title_contains="graphsage"),
        FlagshipSpec(arxiv_id="1710.10903", title_contains="graph attention"),
    ),
    "visual_debugging": (
        FlagshipSpec(title_contains="software visualization", label="SoftViz"),
        FlagshipSpec(title_contains="program visualization", label="ProgViz"),
        FlagshipSpec(title_contains="debugger", label="Debugger"),
    ),
}


def flagship_specs_for_intent(intent_entry_id: str | None) -> tuple[FlagshipSpec, ...]:
    if not intent_entry_id:
        return ()
    return FLAGSHIP_BY_INTENT.get(intent_entry_id, ())


def _paper_matches_spec(paper: dict[str, Any], spec: FlagshipSpec) -> bool:
    pid = str(paper.get("id") or "").lower()
    arx = str(paper.get("arxiv_id") or "").lower()
    title = (paper.get("title") or "").lower()
    if spec.arxiv_id:
        aid = spec.arxiv_id.lower().strip()
        if aid in pid or arx.endswith(aid) or arx == aid:
            return True
    if spec.title_contains and spec.title_contains.lower() in title:
        return True
    return False


def resolve_flagship_paper_ids(
    papers: list[dict[str, Any]],
    intent_entry_id: str | None,
) -> list[str]:
    """
    Return paper ids in flagship priority order (only papers present in ``papers``).
    """
    specs = flagship_specs_for_intent(intent_entry_id)
    if not specs:
        return []
    by_id = {str(p["id"]): p for p in papers if p.get("id") and not p.get("graph_noise")}
    out: list[str] = []
    seen: set[str] = set()
    for spec in specs:
        for pid, p in by_id.items():
            if pid in seen:
                continue
            if _paper_matches_spec(p, spec):
                out.append(pid)
                seen.add(pid)
                break
    return out


def pin_flagship_into_start_here(
    picked: list[str],
    papers: list[dict[str, Any]],
    query_profile: dict[str, Any],
    *,
    max_items: int = 5,
) -> list[str]:
    """
    Prepend canonical flagships (up to 2 slots) then fill remaining slots without duplicates.
    Only pins papers that pass hard Start Here intent gating.
    """
    from .query_intent_gating import attach_intent_category, start_here_allowed

    entry_id = query_profile.get("intent_entry_id")
    flag_ids = resolve_flagship_paper_ids(papers, str(entry_id) if entry_id else None)
    if not flag_ids:
        return picked[:max_items]

    by_id = {str(p["id"]): p for p in papers if p.get("id")}
    merged: list[str] = []
    seen: set[str] = set()

    def _gated_ok(pid: str) -> bool:
        p = by_id.get(pid)
        if not p:
            return False
        cat = attach_intent_category(p, query_profile)
        return start_here_allowed(cat)

    for fid in flag_ids[:2]:
        if fid not in seen and fid in by_id and _gated_ok(fid):
            merged.append(fid)
            seen.add(fid)

    for fid in flag_ids[2:]:
        if len(merged) >= max_items:
            break
        if fid not in seen and fid in by_id and _gated_ok(fid):
            merged.append(fid)
            seen.add(fid)

    for pid in picked:
        if len(merged) >= max_items:
            break
        if pid not in seen and _gated_ok(pid):
            merged.append(pid)
            seen.add(pid)

    return merged[:max_items]


def is_resolved_flagship_paper(paper: dict[str, Any], intent_entry_id: str | None) -> bool:
    specs = flagship_specs_for_intent(intent_entry_id)
    return any(_paper_matches_spec(paper, s) for s in specs)


__all__ = [
    "FLAGSHIP_BY_INTENT",
    "FlagshipSpec",
    "flagship_specs_for_intent",
    "is_resolved_flagship_paper",
    "pin_flagship_into_start_here",
    "resolve_flagship_paper_ids",
]
