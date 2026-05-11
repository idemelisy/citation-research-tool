"""Markdown research summary and BibTeX export from API-shaped pipeline payloads."""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from typing import Any

from .explainability import why_it_matters_one_line
from .intent_classifier import QueryIntent


def _paper_by_id(papers: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {p["id"]: p for p in papers}


def _reading_recency_alignment(year: int | None) -> float:
    """Same buckets as pipeline ``_recency_alignment_score`` (aligned with ranking_fusion recency)."""
    if year is None:
        return 0.55
    cy = datetime.utcnow().year
    age = max(0, cy - int(year))
    if age <= 2:
        return 1.0
    if age <= 5:
        return 0.82
    if age <= 10:
        return 0.58
    if age <= 18:
        return 0.35
    return 0.18


# OR-groups: any substring match counts one hit toward title alignment (systems / GPU literature).
_HW_TITLE_SIGNALS: tuple[tuple[str, ...], ...] = (
    ("gpu", "g.p.u."),
    ("hardware", "hw-accelerat", "hw accelerat"),
    ("tessell", "tessel"),
    ("real-time", "realtime", "real time"),
    ("parallel", "multicore"),
    ("shader", "programmable pipeline"),
    ("cuda", "opencl", "metal api"),
    ("directx", "direct3d", "d3d11", "d3d12"),
    ("vulkan",),
    ("rasteriz", "rasteris"),
)


def _hardware_title_alignment(title: str) -> float:
    t = (title or "").lower()
    hits = 0
    for group in _HW_TITLE_SIGNALS:
        if any(g in t for g in group):
            hits += 1
    return min(0.36, hits * 0.1)


def _tessellation_literature_fit_multiplier(query_text: str, paper: dict[str, Any]) -> float:
    """
    When the user query is explicitly about tessellation, demote papers that only look
    “GPU-adjacent” (parallax mapping, texture tricks, particle systems, curvature tensors)
    so the top reading slots stay defensibly on-topic.
    """
    ql = (query_text or "").lower()
    if "tessell" not in ql and "tessel" not in ql:
        return 1.0

    title = str(paper.get("title") or "")
    abs0 = str(paper.get("abstract") or "")
    blob = (title + " " + abs0[:900]).lower()

    strong_markers = (
        "tessell",
        "tessel",
        "subdivision",
        "catmull",
        "gregory",
        "displaced subdivision",
        "roam",
        "optimally adapting mesh",
        "bicubic patch",
        "gpu subdivision",
        "subdivision kernel",
        "phong tessell",
        "clipmap",
        "hardware tessell",
        "hardware-accelerated terrain",
        "adaptive hardware",
        "approximating subdivision",
        "approximating catmull-clark",
        "interpolating subdivision",
        "multiresolution analysis",
        "simplification envelope",
        "piecewise smooth subdivision",
        "sqrt(3)-subdivision",
        "sqrt(3) subdivision",
    )
    if any(s in blob for s in strong_markers):
        return 1.0

    off_topic_markers = (
        "parallax",
        "occlusion mapping",
        "tensor of curvature",
        "dynamic particle",
        "texture mapping on graphics hardware",
    )
    if any(s in blob for s in off_topic_markers):
        return 0.42

    mesh_adjacent = (
        "multiresolution",
        "level of detail",
        " lod",
        "lod-based",
        "lod rendering",
        "mesh simplification",
        "terrain rendering",
        "rendering height",
        "heightmap",
        "height map",
        "geometry clipmaps",
        "irregular network",
        "triangular irregular",
        "greedy cuts",
    )
    if any(s in blob for s in mesh_adjacent):
        return 0.88

    if "terrain" in blob and any(
        h in blob for h in ("render", "hardware", "gpu", "real-time", "realtime", "shader")
    ):
        return 0.9

    if any(
        k in blob
        for k in (
            "free-form surface",
            "fair surface",
            "variational surface",
            "b-spline surface",
            "bézier patch",
            "bezier patch",
        )
    ):
        return 0.68

    if "surface" in blob or "mesh" in blob or "geometry" in blob:
        return 0.58
    return 0.52


def paper_reading_order_score(
    paper: dict[str, Any],
    *,
    intent_mode_v2: str | None = None,
    query_text: str | None = None,
) -> float:
    """
    Reading-list and branch ordering: blends normalized relevance with graph evidence.

    For ``hardware_system`` queries, down-weights raw in-map degree (which otherwise
    rewards classic subdivision hubs) and adds publication recency plus title-level
    hardware/GPU signals so pipeline papers surface ahead of generic high-degree nodes.
    """
    rel = float(paper.get("relevance_diverse_norm", paper.get("relevance_norm", 0.0)) or 0.0)
    cc = float(paper.get("citation_count", 0) or 0.0)
    missing = float(paper.get("missing_link_score", 0) or 0.0)
    if intent_mode_v2 == QueryIntent.HARDWARE_SYSTEM:
        y = paper.get("year")
        rec = _reading_recency_alignment(int(y) if isinstance(y, int) else None)
        tbonus = _hardware_title_alignment(str(paper.get("title") or ""))
        # Keep total graph contribution modest: uncapped high-degree nodes (classic theory hubs)
        # otherwise still outrank recent GPU / tessellation pipeline papers with similar relevance_norm.
        base = rel * 4.0 + rec * 0.38 + tbonus + missing * 0.25 + min(cc, 90.0) * 0.004
        mul = _tessellation_literature_fit_multiplier(query_text or "", paper)
        return base * mul
    return rel * 2.0 + cc * 0.1 + missing * 0.25


def _paper_importance_score(paper: dict[str, Any]) -> float:
    """Backward-compatible default (no v2 intent context)."""
    return paper_reading_order_score(paper, intent_mode_v2=None, query_text=None)


_BANNED_CLUSTER_LABELS = frozenset(
    {
        "general preference alignment",
        "uncategorized",
        "related cluster",
        "related work",
    }
)


_LABEL_STOP = frozenset(
    "the and for with from that this using based paper method methods approach model study "
    "learning data new results analysis work via into over per among between".split()
)


def _label_cluster_from_titles(papers: list[dict[str, Any]], query_blob: str) -> str:
    """
    PR D2 — label = top tokens from ``query + cluster titles`` (no global topic cache).
    """
    c: Counter[str] = Counter()
    qb = (query_blob or "").lower()
    for w in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{3,}", qb):
        if w in _LABEL_STOP:
            continue
        c[w] += 3
    for p in papers:
        for w in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{3,}", (p.get("title") or "").lower()):
            if w in _LABEL_STOP:
                continue
            c[w] += 2 if w in qb else 1
    top = [w for w, _ in c.most_common(6)]
    if not top:
        return "Query-aligned cluster"
    label = " ".join(t.title() for t in top[:3])
    if label.strip().lower() in _BANNED_CLUSTER_LABELS or any(
        b in label.lower() for b in ("general preference", "uncategorized", "related cluster")
    ):
        return "Query-aligned cluster"
    return label


def group_papers_query_aware_topics(
    papers: list[dict[str, Any]],
    query_profile: dict[str, Any] | None,
) -> dict[str, list[dict[str, Any]]]:
    """
    PR D1 / D2 — cluster labels are computed per query only (no shared static topic buckets).
    """
    if not papers:
        return {}
    qp = query_profile or {}
    qtext = (qp.get("query_text") or "").strip().lower()
    qterms = [str(t).lower() for t in (qp.get("query_terms") or []) if len(str(t).strip()) > 2]
    blob_q = (qtext + " " + " ".join(qterms)).strip()
    iv2 = (qp.get("intent_mode_v2") or "").strip() or None
    qt_raw = (qp.get("query_text") or "").strip()
    ranked = sorted(
        papers,
        key=lambda p: paper_reading_order_score(p, intent_mode_v2=iv2, query_text=qt_raw),
        reverse=True,
    )
    n = len(ranked)
    n_groups = min(4, max(1, (n + 5) // 7))
    size = max(1, (n + n_groups - 1) // n_groups)
    merged: dict[str, list[dict[str, Any]]] = {}
    for g in range(n_groups):
        chunk = ranked[g * size : (g + 1) * size] if g < n_groups - 1 else ranked[g * size :]
        if not chunk:
            continue
        label = _label_cluster_from_titles(chunk, blob_q)
        key = label
        i = 1
        while key in merged:
            key = f"{label} ({i})"
            i += 1
        merged[key] = list(chunk)
    for k in merged:
        merged[k] = sorted(
            merged[k],
            key=lambda p: paper_reading_order_score(p, intent_mode_v2=iv2, query_text=qt_raw),
            reverse=True,
        )
    return dict(sorted(merged.items(), key=lambda kv: (-len(kv[1]), kv[0])))


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


def build_markdown_report(
    payload: dict[str, Any],
    project_title: str = "SRG Research Report",
    *,
    clean: bool = False,
) -> str:
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
    qp = payload.get("query_profile") or {}
    iv2_sort = (qp.get("intent_mode_v2") or "").strip() or None
    qt_sort = (qp.get("query_text") or "").strip()
    if clean:
        lines.append(f"# {project_title}\n")
        lines.append(f"Generated: {payload.get('generated_at', '')}")
        qt0 = (qp.get("query_text") or "").strip()
        if qt0:
            lines.append(f"Query: {qt0[:400]}{'…' if len(qt0) > 400 else ''}")
        lines.append("")
    else:
        lines.append(f"# {project_title}\n")
        lines.append(f"- Generated: `{payload.get('generated_at', '')}`")
    if not clean and qp:
        lines.append(
            f"- Query profile: intent=`{qp.get('query_intent', 'method')}`, "
            f"terms={', '.join((qp.get('query_terms') or [])[:8]) or '—'}"
        )
        if qp.get("intent_mode_v2"):
            lines.append(f"- Classified intent (v2): `{qp.get('intent_mode_v2')}`")
    if not clean:
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

    ranked_papers = sorted(
        papers,
        key=lambda p: paper_reading_order_score(p, intent_mode_v2=iv2_sort, query_text=qt_sort),
        reverse=True,
    )
    topic_groups = group_papers_query_aware_topics(papers, qp)
    papers_with_refs = [p for p in papers if isinstance(p.get("references"), list) and len(p.get("references")) > 0]

    lines.append("## Reading guide\n" if not clean else "## Reading list\n")
    if ranked_papers:
        if not clean:
            hw_note = (
                " Hardware-mode queries down-weight raw graph degree, add recency + GPU/pipeline title cues, "
                "and (when the query mentions tessellation) demote off-topic GPU graphics papers."
                if iv2_sort == QueryIntent.HARDWARE_SYSTEM
                else ""
            )
            lines.append(
                "_Importance rank uses graph relevance, connectivity, and missing-link evidence."
                + hw_note
                + " Use this as a practical reading order, not a ground-truth citation count._\n"
            )
        qt_for_why = (qp.get("query_text") or "").strip()
        intent_for_why = str(qp.get("intent_mode_v2") or qp.get("query_intent") or "exploratory")
        for idx, p in enumerate(ranked_papers[:10], start=1):
            if clean:
                lines.append(f"### {idx}. {p.get('title', '')}")
                w = why_it_matters_one_line(
                    p, query_text=qt_for_why, intent_label=intent_for_why
                )
                lines.append(f"Why: {w}")
                lines.append("")
            else:
                imp = paper_reading_order_score(p, intent_mode_v2=iv2_sort, query_text=qt_sort)
                lines.append(
                    f"- #{idx} **{p.get('title', '')}** (`{p.get('id')}`) — "
                    f"importance={imp:.3f}, relevance={float(p.get('relevance_norm', 0.0) or 0.0):.3f}, "
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
                if clean:
                    lines.append(f"- **{p.get('title', '')}**")
                else:
                    imp = paper_reading_order_score(p, intent_mode_v2=iv2_sort, query_text=qt_sort)
                    lines.append(
                        f"- **{p.get('title', '')}** (`{p.get('id')}`) "
                        f"(importance={imp:.3f})"
                    )
            if len(rows) > 8:
                lines.append(f"- _... {len(rows) - 8} more in this topic_")
            lines.append("")
    else:
        lines.append("_No topic groups found._\n")

    lines.append("## Reference coverage\n")
    if clean:
        lines.append(
            f"- **{len(papers_with_refs)}** of **{len(papers)}** papers include reference metadata in this export."
        )
    else:
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
            if clean:
                lines.append(f"  - **{(p.get('title') or 'Untitled')[:80]}**: {len(refs)} references")
            else:
                lines.append(f"  - `{p.get('id')}`: {len(refs)} references")
    lines.append("")

    lines.append("## Foundational papers\n")
    uploaded = [p for p in papers_all if p.get("seed_origin") == "uploaded_pdf"]
    if uploaded:
        lines.append("### Library seeds (uploaded PDFs)\n")
        for p in uploaded[:30]:
            lbl = p.get("source_label", "PDF")
            if clean:
                lines.append(f"- **{p.get('title', '')}** — _{lbl}_")
            else:
                lines.append(f"- **{p.get('title', '')}** (`{p.get('id')}`) — _{lbl}_")
        lines.append("")
    lines.append("### Highly ranked in this graph\n")
    gold = [
        p
        for p in papers_all
        if p.get("is_foundational_hub")
        and p.get("foundational_eligible", True)
        and not p.get("graph_noise")
    ]
    gold_sorted = sorted(
        gold,
        key=lambda p: (float(p.get("pii_concept_match", 0.0)), int(p.get("citation_count", 0))),
        reverse=True,
    )
    if gold_sorted:
        lines.append("_Bright gold (foundational) hubs — pinned at graph center in the UI._\n")
        for p in gold_sorted[:12]:
            if clean:
                lines.append(f"- ★ **{p.get('title', '')}** — _Foundational anchor in this map._")
            else:
                cc = p.get("citation_count", "—")
                pm = p.get("pii_concept_match", "—")
                lines.append(
                    f"- ★ **{p.get('title', '')}** — PII concept match ≈{pm}, graph degree {cc} (`{p.get('id')}`)"
                )
        lines.append("")
    recs_emitted = 0
    for r in recs:
        if recs_emitted >= 15:
            break
        pid = r.get("paper_id")
        meta = by_id.get(pid or "", {})
        title = meta.get("title") or pid
        if str(title).strip().startswith("Pending metadata"):
            continue
        if clean:
            lines.append(f"- **{title}** — _{r.get('reason', '')}_")
        else:
            lines.append(
                f"- **{title}** — score {r.get('score', '')}: _{r.get('reason', '')}_ "
                f"(`{pid}`)"
            )
        recs_emitted += 1
    if not recs_emitted and not uploaded and not gold_sorted:
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
            fh = "★ " if m.get("is_foundational_hub") else ""
            if clean:
                lines.append(f"- {fh}**{title}** — _Cited by several of your uploaded PDFs ({c})._")
            else:
                oa_c = m.get("openalex_cited_by_count", "")
                pii = m.get("pii_concept_match", "")
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
    if clean:
        lines.append(
            "\n## Citation structure\n"
            f"- **{len(supported)}** medium/high-confidence links; "
            f"**{len(potential)}** exploratory or low-confidence links (detail omitted in reading mode).\n"
        )
    else:
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
