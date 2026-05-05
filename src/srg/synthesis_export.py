"""Markdown research summary and BibTeX export from API-shaped pipeline payloads."""

from __future__ import annotations

import re
from typing import Any


def _paper_by_id(papers: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {p["id"]: p for p in papers}


def _paper_importance_score(paper: dict[str, Any]) -> float:
    """Stable ranking score for report ordering."""
    rel = float(paper.get("relevance_diverse_norm", paper.get("relevance_norm", 0.0)) or 0.0)
    cc = float(paper.get("citation_count", 0) or 0.0)
    missing = float(paper.get("missing_link_score", 0) or 0.0)
    return rel * 2.0 + cc * 0.1 + missing * 0.25


_TOPIC_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("Core DPO Methods", ("direct preference optimization", "dpo", "preference optimization")),
    ("Multimodal / Vision", ("vision-language", "multi-image", "image", "vlm")),
    ("Reasoning / Stepwise", ("step-wise", "reasoning", "long-chain", "chain-of-thought")),
    ("Data Selection / Curation", ("data-centric", "data selection", "difficulty", "implicit reward gap")),
    ("Control / Constraints", ("constrained", "controlled")),
    ("Online / Dynamic", ("online", "dynamic", "shift")),
]


def _topic_for_paper_title(title: str | None) -> str:
    t = (title or "").strip().lower()
    if not t:
        return "Uncategorized"
    for topic, keywords in _TOPIC_RULES:
        if any(k in t for k in keywords):
            return topic
    return "General Preference Alignment"


def _group_papers_by_topic(papers: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for p in papers:
        topic = _topic_for_paper_title(p.get("title"))
        out.setdefault(topic, []).append(p)
    for topic, rows in out.items():
        out[topic] = sorted(rows, key=_paper_importance_score, reverse=True)
    return dict(sorted(out.items(), key=lambda kv: (-len(kv[1]), kv[0])))


def _bibtex_escape(s: str) -> str:
    return s.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}").replace("%", r"\%")


def _cite_key(paper_id: str) -> str:
    key = re.sub(r"[^\w]", "_", paper_id)
    key = re.sub(r"_+", "_", key).strip("_")
    return (key or "entry")[:80]


def build_bibtex(payload: dict[str, Any]) -> str:
    papers = [p for p in (payload.get("papers") or []) if not p.get("graph_noise")]
    lines: list[str] = []
    for p in papers:
        pid = p.get("id") or "unknown"
        key = _cite_key(str(pid))
        title = _bibtex_escape((p.get("title") or "Untitled").strip())
        year = p.get("year")
        year_s = str(year) if year is not None else ""
        authors = p.get("authors") or []
        if isinstance(authors, list) and authors:
            names = []
            for a in authors:
                if isinstance(a, dict):
                    names.append(_bibtex_escape(str(a.get("name", ""))))
                else:
                    names.append(_bibtex_escape(str(a)))
            author = " and ".join(n for n in names if n)
        else:
            author = "Unknown"
        doi = (p.get("doi") or "").strip()
        arxiv = (p.get("arxiv_id") or "").strip()
        venue = _bibtex_escape((p.get("venue") or "").strip())
        lines.append(f"@misc{{{key},")
        lines.append(f"  title = {{{title}}},")
        lines.append(f"  author = {{{author}}},")
        if year_s:
            lines.append(f"  year = {{{year_s}}},")
        if venue:
            lines.append(f"  howpublished = {{{venue}}},")
        if doi:
            lines.append(f"  doi = {{{_bibtex_escape(doi)}}},")
        if arxiv:
            lines.append(f"  eprint = {{{_bibtex_escape(arxiv)}}},")
            lines.append("  archivePrefix = {arXiv},")
        lines.append(f"  note = {{SRG id `{_bibtex_escape(str(pid))}`}}")
        lines.append("}\n")
    return "\n".join(lines)


def _edge_buckets(edges: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    potential: list[dict[str, Any]] = []
    supported: list[dict[str, Any]] = []
    for e in edges:
        style = e.get("style") or {}
        stroke = style.get("stroke", "")
        conf = (e.get("confidence") or "").lower()
        if stroke == "dashed" or conf == "low":
            potential.append(e)
        else:
            supported.append(e)
    return supported, potential


def build_markdown_report(payload: dict[str, Any], project_title: str = "SRG Research Report") -> str:
    papers_all = payload.get("papers") or []
    papers = [p for p in papers_all if not p.get("graph_noise")]
    by_id = _paper_by_id(papers_all)
    graph = payload.get("graph") or {}
    edges = graph.get("edges") or []
    discovery = payload.get("discovery") or {}
    trends = discovery.get("trends") or {}
    recs = discovery.get("recommendations") or []
    missing = payload.get("missing_link_candidates") or []

    lines: list[str] = []
    lines.append(f"# {project_title}\n")
    lines.append(f"- Generated: `{payload.get('generated_at', '')}`")
    qp = payload.get("query_profile") or {}
    if qp:
        lines.append(
            f"- Query profile: intent=`{qp.get('query_intent', 'method')}`, "
            f"terms={', '.join((qp.get('query_terms') or [])[:8]) or '—'}"
        )
    eq = (payload.get("evidence_quality") or "").strip().lower()
    est = payload.get("evidence_stats") or {}
    if eq:
        lines.append(
            f"- Evidence quality: **{eq.upper()}** "
            f"(references={int(est.get('references_total', 0))}, edges={int(est.get('graph_edges_total', 0))})"
        )
    if payload.get("metadata_only_mode"):
        lines.append("- Mode: **metadata-only ranking** (citation/reference evidence unavailable for this run)")
    if payload.get("discipline_applied"):
        lines.append(f"- Discipline prior: **{payload['discipline_applied']}**")
    q = payload.get("quality_report") or {}
    if q:
        lines.append(
            f"- Quality (dedup): merged_pairs={q.get('merged_pairs', 0)}, "
            f"conflicts={q.get('conflicts', 0)}"
        )
    ss = payload.get("source_summary") or {}
    if ss:
        lines.append("- Source mix: " + ", ".join(f"{k}={v}" for k, v in sorted(ss.items())))
    lines.append("")

    ranked_papers = sorted(papers, key=_paper_importance_score, reverse=True)
    topic_groups = _group_papers_by_topic(papers)
    papers_with_refs = [p for p in papers if isinstance(p.get("references"), list) and len(p.get("references")) > 0]

    lines.append("## Reading guide\n")
    if ranked_papers:
        lines.append(
            "_Importance rank uses graph relevance, connectivity, and missing-link evidence. "
            "Use this as a practical reading order, not a ground-truth citation count._\n"
        )
        for idx, p in enumerate(ranked_papers[:10], start=1):
            lines.append(
                f"- #{idx} **{p.get('title', '')}** (`{p.get('id')}`) — "
                f"importance={_paper_importance_score(p):.3f}, relevance={float(p.get('relevance_norm', 0.0) or 0.0):.3f}, "
                f"graph_links={int(p.get('citation_count', 0) or 0)}"
            )
    else:
        lines.append("_No papers in graph._")
    lines.append("")

    lines.append("## Topic separation\n")
    if topic_groups:
        for topic, rows in topic_groups.items():
            lines.append(f"### {topic}\n")
            for p in rows[:8]:
                lines.append(
                    f"- **{p.get('title', '')}** (`{p.get('id')}`) "
                    f"(importance={_paper_importance_score(p):.3f})"
                )
            if len(rows) > 8:
                lines.append(f"- _... {len(rows) - 8} more in this topic_")
            lines.append("")
    else:
        lines.append("_No topic groups found._\n")

    lines.append("## Reference coverage\n")
    lines.append(
        f"- Papers with explicit `references` field: **{len(papers_with_refs)} / {len(papers)}**"
    )
    lines.append(
        f"- Supported citation edges in graph: **{len(_edge_buckets(edges)[0])}** "
        f"(potential/low-confidence: **{len(_edge_buckets(edges)[1])}**)"
    )
    if not papers_with_refs:
        lines.append(
            "- No explicit reference lists were attached to these nodes. "
            "This commonly happens with pure arXiv search results unless OpenAlex/Crossref enrichment succeeds."
        )
    else:
        lines.append("- Per-paper reference counts (top 15 by importance):")
        for p in ranked_papers[:15]:
            refs = p.get("references") or []
            lines.append(f"  - `{p.get('id')}`: {len(refs)} references")
    lines.append("")

    lines.append("## Foundational papers\n")
    uploaded = [p for p in papers_all if p.get("seed_origin") == "uploaded_pdf"]
    if uploaded:
        lines.append("### Library seeds (uploaded PDFs)\n")
        for p in uploaded[:30]:
            lbl = p.get("source_label", "PDF")
            lines.append(f"- **{p.get('title', '')}** (`{p.get('id')}`) — _{lbl}_")
        lines.append("")
    lines.append("### Highly ranked in this graph\n")
    gold = [
        p
        for p in papers_all
        if p.get("is_foundational_hub") and not p.get("graph_noise")
    ]
    gold_sorted = sorted(
        gold,
        key=lambda p: (float(p.get("pii_concept_match", 0.0)), int(p.get("citation_count", 0))),
        reverse=True,
    )
    if gold_sorted:
        lines.append("_Bright gold (foundational) hubs — pinned at graph center in the UI._\n")
        for p in gold_sorted[:12]:
            cc = p.get("citation_count", "—")
            pm = p.get("pii_concept_match", "—")
            lines.append(
                f"- ★ **{p.get('title', '')}** — PII concept match ≈{pm}, graph degree {cc} (`{p.get('id')}`)"
            )
        lines.append("")
    for r in recs[:15]:
        pid = r.get("paper_id")
        meta = by_id.get(pid or "", {})
        title = meta.get("title") or pid
        lines.append(
            f"- **{title}** — score {r.get('score', '')}: _{r.get('reason', '')}_ "
            f"(`{pid}`)"
        )
    if not recs and not uploaded and not gold_sorted:
        lines.append("_No ranking data._\n")

    lines.append("\n## Missing link candidates (foundational PII literature)\n")
    if missing:
        lines.append(
            "_Sorted by PII concept match on OpenAlex, then global citation count, then how many of your "
            "uploaded PDFs cite this work (≥3 ⇒ strong “missing ring”). Not proof of direct citation._"
        )
        lines.append("")
        ranked = sorted(
            missing,
            key=lambda m: (
                float(m.get("pii_concept_match", 0.0)),
                int(m.get("openalex_cited_by_count", 0)),
                int(m.get("co_cited_by_uploaded_count", 0)),
            ),
            reverse=True,
        )
        for m in ranked[:25]:
            pid = m.get("paper_id")
            meta = by_id.get(pid or "", {})
            title = meta.get("title") or pid
            c = m.get("co_cited_by_uploaded_count", "")
            oa_c = m.get("openalex_cited_by_count", "")
            pii = m.get("pii_concept_match", "")
            fh = "★ " if m.get("is_foundational_hub") else ""
            lines.append(
                f"- {fh}**{title}** — PII concept≈{pii}, OpenAlex cited_by={oa_c}, "
                f"your PDFs cite this work: {c} (`{pid}`)"
            )
    else:
        lines.append("_None above threshold._\n")

    lines.append("\n## Trends (domain-aware)\n")
    tl = trends.get("timeline") or {}
    dom = trends.get("domain_distribution") or {}
    if tl:
        lines.append("### Publication years\n")
        for y in sorted(tl.keys(), key=lambda x: (x == "unknown", str(x))):
            lines.append(f"- {y}: {tl[y]}")
        lines.append("")
    if dom:
        lines.append("### Domains\n")
        for d in sorted(dom.keys(), key=lambda x: (-dom[x], str(x))):
            lines.append(f"- {d}: {dom[d]}")
        lines.append("")

    supported, potential = _edge_buckets(edges)
    lines.append("\n## Supported citation links (medium / high confidence)\n")
    if supported:
        for e in supported[:40]:
            src = by_id.get(e.get("source"), {})
            tgt = by_id.get(e.get("target"), {})
            ctx = (e.get("context") or "").replace("\n", " ")[:200]
            prov = e.get("context_source") or "—"
            lines.append(
                f"- `{e.get('source')}` → `{e.get('target')}` "
                f"({e.get('confidence', '')}) "
                f"_{src.get('title', '')[:60]}_ → _{tgt.get('title', '')[:60]}_"
            )
            if ctx:
                lines.append(f"  - Snippet: _{ctx}_ (`{prov}`)")
    else:
        lines.append("_None._\n")

    lines.append("\n## Potential connections (lower confidence)\n")
    lines.append(
        "_Dashed / low-confidence edges: abstract proxy or unresolved context; "
        "treat as exploratory._\n"
    )
    if potential:
        for e in potential[:40]:
            src = by_id.get(e.get("source"), {})
            tgt = by_id.get(e.get("target"), {})
            lines.append(
                f"- `{e.get('source')}` → `{e.get('target')}` "
                f"({e.get('confidence', '')}) "
                f"_{src.get('title', '')[:50]}_ → _{tgt.get('title', '')[:50]}_"
            )
    else:
        lines.append("_None._\n")

    lines.append("\n## Node provenance (source labels)\n")
    for p in papers[:50]:
        if p.get("seed_origin") == "uploaded_pdf" or p.get("source_label"):
            lines.append(
                f"- `{p.get('id')}`: **{p.get('title', '')[:80]}** "
                f"— {p.get('source_label', 'API')}"
            )

    return "\n".join(lines)
