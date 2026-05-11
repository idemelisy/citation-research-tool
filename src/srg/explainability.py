"""Per-paper trust strings — SRG Lite roadmap / v2.2 (`why_it_matters`)."""

from __future__ import annotations

from datetime import datetime
from typing import Any


def _recency_phrase(paper: dict[str, Any]) -> str:
    y = paper.get("year")
    if not isinstance(y, int):
        return "Publication year is unknown; recency was inferred conservatively."
    age = max(0, datetime.utcnow().year - y)
    if age <= 3:
        return "Recent publication window (strong recency signal)."
    if age <= 7:
        return "Moderately recent work relative to the current literature."
    return "Older publication; kept only when semantic and anchor signals justify inclusion."


def build_why_it_matters(
    paper: dict[str, Any],
    *,
    query_text: str,
    intent_label: str,
) -> str:
    """
    One short block the UI can show as plain text (newlines optional).
    Uses scores and graph signals already attached to the node.
    """
    existing = (paper.get("why_it_matters") or "").strip()
    if existing:
        return existing

    q = (query_text or "").strip()
    intent = (intent_label or "exploratory").strip().lower().replace(" ", "_")
    sem = float(paper.get("semantic_score", 0.0) or 0.0)
    ins = float(paper.get("intent_similarity") or sem)
    hops = paper.get("anchor_graph_hops")
    hop_s = f"{int(hops)}" if hops is not None and int(hops) < 99 else "distant"

    lines: list[str] = ["Selected because:"]
    if max(sem, ins) >= 0.45:
        lines.append(f"- Core alignment with your query intent ({intent}).")
    elif max(sem, ins) >= 0.28:
        lines.append(f"- Moderate alignment with your query ({intent}).")
    else:
        lines.append("- Limited direct match; kept for neighborhood structure on the citation map.")

    lines.append(f"- {_recency_phrase(paper)}")

    coc = int(paper.get("relation_max_cocitation") or 0)
    cou = int(paper.get("relation_max_coupling") or 0)
    dh = int(paper.get("relation_direct_hits") or 0)
    if dh > 0:
        lines.append("- Directly connected to your starting papers (citation tie to a seed in this map).")
    elif coc >= 2:
        lines.append("- Frequently cited alongside work near your seeds (co-citation signal).")
    elif cou >= 2:
        lines.append("- Shares references with the seed neighborhood (same bibliographic conversation).")
    elif hops is not None and int(hops) <= 2:
        lines.append(f"- Within {hop_s} undirected hop(s) of your query seeds on this map.")
    else:
        lines.append("- Reached via citation expansion from the seed neighborhood.")

    title = (paper.get("title") or "").strip()
    if q and title:
        qtok = [t for t in q.lower().split() if len(t) > 3][:4]
        overlap = [t for t in qtok if t in title.lower()]
        if overlap:
            lines.append(f"- Topic overlap with your query: {', '.join(overlap[:3])}.")
    if len(lines) <= 3:
        return (
            "This paper is included due to structural relevance in the citation graph "
            "and moderate semantic alignment."
        )
    return "\n".join(lines)


def why_it_matters_one_line(paper: dict[str, Any], *, query_text: str, intent_label: str) -> str:
    """PR E2 — single-sentence summary for strict reading-list export."""
    block = (paper.get("why_it_matters") or "").strip() or build_why_it_matters(
        paper, query_text=query_text, intent_label=intent_label
    )
    one = block.replace("\n", " ").strip()
    for prefix in ("Selected because:", "- "):
        if one.startswith(prefix):
            one = one[len(prefix) :].strip()
    if len(one) > 240:
        one = one[:237] + "…"
    return one or (
        "Included for query intent alignment, anchor proximity on the citation graph, and recency fit."
    )


def ensure_why_it_matters_top5(
    nodes: list[dict[str, Any]],
    *,
    query_text: str,
    intent_label: str,
    k: int = 5,
) -> None:
    """PR E3 — guarantee filled ``why_it_matters`` (+ one-line) for the top-k ranked nodes."""
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
