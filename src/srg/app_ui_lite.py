"""SRG Lite — minimal research discovery UI (Streamlit)."""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from srg.api_adapter import SRGApplicationService
    from srg.app_ui import (
        _filter_graph_for_display,
        _paper_map,
        _paper_needs_metadata_refresh,
        _related_edges,
    )
    from srg.core.defaults import LITE_DISCOVERY_OPTIONS
    from srg.core.pipeline import generate_reading_list
    from srg.ingestion import fetch_arxiv_public_metadata, normalize_arxiv_list_id
    from srg.synthesis_export import build_bibtex, build_markdown_report
else:
    from .api_adapter import SRGApplicationService
    from .app_ui import (
        _filter_graph_for_display,
        _paper_map,
        _paper_needs_metadata_refresh,
        _related_edges,
    )
    from .core.defaults import LITE_DISCOVERY_OPTIONS
    from .core.pipeline import generate_reading_list
    from .ingestion import fetch_arxiv_public_metadata, normalize_arxiv_list_id
    from .synthesis_export import build_bibtex, build_markdown_report

DEV_MODE = False
LITE_MAX_GRAPH_NODES = 28
LITE_MAX_GRAPH_EDGES = 36
LITE_GRAPH_HEIGHT = 720

_CONF_RANK = {"high": 0, "medium": 1, "low": 2}

LITE_CSS = """
<style>
    .block-container { padding-top: 1.25rem; max-width: 100%; }
    .lite-tagline { color: #71717a; font-size: 0.9rem; margin: -0.25rem 0 1.25rem 0; }
    .lite-card {
        border: 1px solid #e4e4e7;
        border-radius: 10px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.75rem;
        background: #fafafa;
    }
    .lite-card-meta { color: #52525b; font-size: 0.85rem; margin: 0.35rem 0; }
    .lite-card-summary { color: #3f3f46; font-size: 0.9rem; line-height: 1.45; margin-top: 0.5rem; }
    h1 { font-weight: 600; letter-spacing: -0.02em; }
    h2, h3 { font-weight: 600; }
</style>
"""


def _fmt_domain(domain: str | None) -> str:
    d = (domain or "").strip() or "General"
    return d.replace("_", " ").title()


def _short_title(title: str, *, words: int = 3) -> str:
    t = (title or "Untitled").strip()
    parts = t.split()
    if len(parts) <= words:
        return t
    return " ".join(parts[:words]) + "..."


def _openalex_citation_count(paper: dict[str, Any]) -> int | None:
    for prov in paper.get("provenance") or []:
        if not isinstance(prov, dict) or prov.get("source") != "openalex":
            continue
        raw = prov.get("raw")
        if isinstance(raw, dict) and raw.get("cited_by_count") is not None:
            try:
                return int(raw["cited_by_count"])
            except (TypeError, ValueError):
                return None
    return None


def _authors_line(paper: dict[str, Any], *, max_names: int = 6) -> str:
    authors = paper.get("authors") or []
    if not isinstance(authors, list) or not authors:
        return ""
    names: list[str] = []
    for a in authors[:max_names]:
        if isinstance(a, dict):
            n = str(a.get("name") or "").strip()
        else:
            n = str(a).strip()
        if n:
            names.append(n)
    if not names:
        return ""
    suffix = " et al." if len(authors) > max_names else ""
    return ", ".join(names) + suffix


def _venue_line(paper: dict[str, Any]) -> str:
    v = (paper.get("venue") or "").strip()
    return v if v and v.lower() not in ("pending-ingest", "unknown") else ""


def _paper_primary_url(paper: dict[str, Any]) -> str | None:
    ax = (paper.get("arxiv_id") or "").strip()
    if ax:
        return f"https://arxiv.org/abs/{ax}"
    doi = (paper.get("doi") or "").strip()
    if doi:
        d = doi.replace("https://doi.org/", "").strip()
        return f"https://doi.org/{d}"
    return None


def _compact_summary(paper: dict[str, Any] | None, entry: dict[str, Any] | None = None) -> str:
    if paper:
        abstract = (paper.get("abstract") or "").strip()
        if abstract:
            one = abstract.split(". ")[0].strip()
            if one and len(one) > 20:
                return one + ("." if not one.endswith(".") else "")
    if entry:
        ol = (entry.get("one_line") or "").strip()
        skip = (
            "canonical anchor",
            "method extension",
            "central to the literature",
            "literature neighborhood",
            "alignment baseline",
            "cross-domain",
            "exploratory",
        )
        if ol and not any(s in ol.lower() for s in skip):
            return ol
    return ""


def _apply_ui_filters(
    papers: list[dict[str, Any]],
    *,
    domains: list[str] | None,
    year_range: tuple[int, int] | None,
    foundational_only: bool,
    foundational_ids: set[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in papers:
        pid = str(p.get("id") or "")
        if foundational_only and pid not in foundational_ids:
            continue
        dom = _fmt_domain(p.get("domain"))
        if domains and dom not in domains:
            continue
        yr = p.get("year")
        if year_range and yr is not None:
            yi = int(yr)
            if yi < year_range[0] or yi > year_range[1]:
                continue
        out.append(p)
    return out


def _node_relevance(n: dict[str, Any]) -> float:
    return float(n.get("relevance_diverse_norm", n.get("relevance_norm", 0.0)) or 0.0)


def _restrict_graph(
    graph_json: dict[str, Any],
    allowed_ids: set[str],
    *,
    max_nodes: int = LITE_MAX_GRAPH_NODES,
) -> dict[str, Any]:
    """
    Cap graph size while keeping citation connectivity.

    Previously we kept the top-N nodes by score only, which dropped most edges
    (pairs in the top-N often have no citation link to each other).
    """
    nodes = [n for n in graph_json.get("nodes", []) if n.get("id") in allowed_ids]
    node_by_id = {n["id"]: n for n in nodes}
    edges = [
        e
        for e in graph_json.get("edges", [])
        if e.get("source") in node_by_id and e.get("target") in node_by_id
    ]
    if len(nodes) <= max_nodes:
        return {"nodes": nodes, "edges": edges}

    seeds: list[str] = []
    for n in nodes:
        if (
            n.get("seed_origin") == "api_search"
            or n.get("is_foundational_hub")
            or n.get("is_missing_link_candidate")
        ):
            seeds.append(n["id"])
    if not seeds:
        seeds = sorted(node_by_id.keys(), key=lambda nid: _node_relevance(node_by_id[nid]), reverse=True)[:3]

    selected: set[str] = set()
    frontier: list[str] = []
    for sid in dict.fromkeys(seeds):
        selected.add(sid)
        frontier.append(sid)

    adj: dict[str, list[str]] = {}
    for e in edges:
        s, t = str(e["source"]), str(e["target"])
        adj.setdefault(s, []).append(t)
        adj.setdefault(t, []).append(s)

    while len(selected) < max_nodes and frontier:
        cur = frontier.pop(0)
        neighbors = sorted(
            (nb for nb in adj.get(cur, []) if nb not in selected),
            key=lambda nid: _node_relevance(node_by_id[nid]),
            reverse=True,
        )
        for nb in neighbors:
            if len(selected) >= max_nodes:
                break
            selected.add(nb)
            frontier.append(nb)

    if len(selected) < max_nodes:
        for nid in sorted(node_by_id.keys(), key=lambda x: _node_relevance(node_by_id[x]), reverse=True):
            if nid not in selected:
                selected.add(nid)
            if len(selected) >= max_nodes:
                break

    fn = [node_by_id[nid] for nid in selected if nid in node_by_id]
    ids = set(selected)
    fe = [e for e in edges if e.get("source") in ids and e.get("target") in ids]
    return {"nodes": fn, "edges": fe}


def _apply_lite_display_layout(graph_json: dict[str, Any]) -> None:
    """Static radial layout — avoids physics hairballs and label pile-ups."""
    import math

    nodes = graph_json.get("nodes", [])
    if not nodes:
        return
    ranked = sorted(nodes, key=_node_relevance, reverse=True)
    for i, n in enumerate(ranked):
        if n.get("viz_physics_fixed") and n.get("viz_x") is not None:
            continue
        ring = i // 7
        slot = i % 7
        count_in_ring = min(7, len(ranked) - ring * 7)
        r = 70.0 + ring * 55.0
        ang = (2.0 * math.pi * slot) / max(1, count_in_ring)
        n["viz_x"] = r * math.cos(ang)
        n["viz_y"] = r * math.sin(ang)
        n["viz_physics_fixed"] = True


def _prune_graph_edges(
    edges: list[dict[str, Any]],
    *,
    max_edges: int = LITE_MAX_GRAPH_EDGES,
    include_weak: bool = False,
) -> list[dict[str, Any]]:
    pool = list(edges)
    if not include_weak:
        pool = [e for e in pool if (e.get("confidence") or "").lower() != "low"]
    pool.sort(
        key=lambda e: (
            _CONF_RANK.get(str(e.get("confidence") or "").lower(), 9),
            -float(e.get("weight", 1.0) or 1.0),
        ),
    )
    return pool[:max_edges]


def _node_for_lite_graph(node: dict[str, Any], *, show_labels: bool = False) -> Node:
    title = node.get("title") or node["id"]
    label = _short_title(str(title), words=2) if show_labels else ""
    is_fh = bool(node.get("is_foundational_hub")) and bool(node.get("foundational_eligible", True))
    origin = (node.get("seed_origin") or "").strip()
    sz = int(
        node.get("viz_size")
        or (38 if is_fh else (26 if node.get("is_missing_link_candidate") else 16))
    )
    if origin == "uploaded_pdf" or node.get("source_label") == "PDF":
        grp, default_col = "uploaded", "#1D4ED8"
    elif is_fh:
        grp, default_col = "foundational", "#D4AF37"
    elif node.get("is_missing_link_candidate"):
        grp, default_col = "missing_link", "#CA8A04"
    elif origin == "api_search":
        grp, default_col = "anchor_search", "#059669"
    elif origin == "discovered":
        grp, default_col = "discovered", "#94A3B8"
    else:
        grp, default_col = "discovered", "#D4D4D8"
    col = node.get("viz_color") or default_col
    pos: dict[str, Any] = {}
    if node.get("viz_physics_fixed"):
        pos["x"] = float(node.get("viz_x", 0.0))
        pos["y"] = float(node.get("viz_y", 0.0))
        pos["fixed"] = {"x": True, "y": True}
    return Node(
        id=node["id"],
        label=label,
        size=sz,
        color=col,
        title=str(title),
        group=grp,
        **pos,
    )


def _to_agraph_payload_lite(
    graph_json: dict[str, Any],
    *,
    show_labels: bool = False,
) -> tuple[list[Node], list[Edge]]:
    nodes = [_node_for_lite_graph(n, show_labels=show_labels) for n in graph_json.get("nodes", [])]
    edges: list[Edge] = []
    for edge in graph_json.get("edges", []):
        style = edge.get("style") or {}
        conf = str(edge.get("confidence") or "").lower()
        color = "#9CA3AF" if conf == "low" else ("#93C5FD" if conf == "medium" else "#2563EB")
        edges.append(
            Edge(
                source=edge["source"],
                target=edge["target"],
                label="",
                type="DASHED" if style.get("stroke") == "dashed" else "line",
                color=color,
                width=1.2 if conf == "low" else (1.6 if conf == "medium" else 2.2),
                arrows="to",
                smooth={"enabled": True, "type": "continuous", "roundness": 0.22},
            )
        )
    return nodes, edges


def _year_distribution(papers: list[dict[str, Any]]) -> dict[int, int]:
    c: Counter[int] = Counter()
    for p in papers:
        yr = p.get("year")
        if yr is not None:
            c[int(yr)] += 1
    return dict(sorted(c.items()))


def _domain_distribution(papers: list[dict[str, Any]]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for p in papers:
        c[_fmt_domain(p.get("domain"))] += 1
    return dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))


def _select_paper_node(paper_id: str) -> None:
    st.session_state["selected_node"] = paper_id


def _render_foundational_card(entry: dict[str, Any], paper: dict[str, Any] | None) -> None:
    p = paper or {}
    pid = str(entry.get("paper_id") or p.get("id") or "")
    title = (entry.get("title") or p.get("title") or pid).strip()
    url = _paper_primary_url(p)
    authors = _authors_line(p)
    venue = _venue_line(p)
    yr = entry.get("year") if entry.get("year") is not None else p.get("year")
    cites = _openalex_citation_count(p)
    summary = _compact_summary(p, entry)

    meta_parts: list[str] = []
    if authors:
        meta_parts.append(authors)
    if venue:
        meta_parts.append(venue)
    if yr is not None:
        meta_parts.append(str(int(yr)))
    if cites is not None:
        meta_parts.append(f"{cites:,} citations")

    with st.container(border=True):
        if url:
            st.markdown(f"**[{title}]({url})**")
        else:
            st.markdown(f"**{title}**")
        if meta_parts:
            st.caption(" | ".join(meta_parts))
        if summary:
            st.markdown(summary)
        if pid:
            st.button(
                "View in panel →",
                key=f"open_detail_{pid}",
                use_container_width=True,
                on_click=_select_paper_node,
                args=(pid,),
            )


def _arxiv_id_for_paper(paper: dict[str, Any]) -> str:
    """Best-effort arXiv list id for on-demand abstract fetch."""
    raw = (paper.get("arxiv_id") or "").strip()
    if raw:
        return normalize_arxiv_list_id(raw)
    pid = str(paper.get("id") or "").strip()
    if pid.lower().startswith("arxiv:"):
        return normalize_arxiv_list_id(pid.split(":", 1)[-1])
    doi = (paper.get("doi") or "").strip()
    m = re.search(r"arxiv[./](\d{4}\.\d{4,5})", doi, flags=re.IGNORECASE)
    if m:
        return normalize_arxiv_list_id(m.group(1))
    m2 = re.search(r"(\d{4}\.\d{4,5})", doi)
    if m2 and "48550" in doi.lower():
        return normalize_arxiv_list_id(m2.group(1))
    return ""


def _persist_abstract_on_payload(payload: dict[str, Any], paper_id: str, abstract: str) -> None:
    for row in payload.get("papers") or []:
        if str(row.get("id") or "") == paper_id:
            row["abstract"] = abstract
    graph = payload.get("graph") or {}
    for row in graph.get("nodes") or []:
        if str(row.get("id") or "") == paper_id:
            row["abstract"] = abstract


def _ensure_paper_abstract(
    paper: dict[str, Any],
    *,
    paper_id: str,
    payload: dict[str, Any],
) -> tuple[str, str]:
    """Use stored abstract, session cache, or fetch from arXiv. Returns (text, source)."""
    existing = (paper.get("abstract") or "").strip()
    if existing:
        return existing, ""

    cache: dict[str, str] = st.session_state.setdefault("lite_abstract_cache", {})
    cached = (cache.get(paper_id) or "").strip()
    if cached:
        paper["abstract"] = cached
        return cached, "arxiv"

    aid = _arxiv_id_for_paper(paper)
    if not aid:
        return "", ""

    with st.spinner("Fetching abstract from arXiv…"):
        meta = fetch_arxiv_public_metadata(aid)
    summary = (meta.get("summary") or "").strip() if isinstance(meta, dict) else ""
    if not summary:
        return "", ""

    paper["abstract"] = summary
    if not paper.get("arxiv_id"):
        paper["arxiv_id"] = aid
    authors = meta.get("authors") if isinstance(meta, dict) else None
    if authors and not paper.get("authors"):
        paper["authors"] = [{"name": str(n)} for n in authors if str(n).strip()]

    _persist_abstract_on_payload(payload, paper_id, summary)
    cache[paper_id] = summary
    st.session_state.payload = payload
    return summary, "arxiv"


def _neighbor_titles(
    payload: dict[str, Any], selected: str, pmap: dict[str, dict[str, Any]], limit: int = 6
) -> list[tuple[str, str]]:
    edges = _related_edges(payload["graph"], selected)
    out: list[tuple[str, str]] = []
    for e in edges[:40]:
        other = e["target"] if e["source"] == selected else e["source"]
        op = pmap.get(other)
        if op:
            out.append((other, (op.get("title") or other)[:120]))
        if len(out) >= limit:
            break
    return out


def _render_paper_details_panel(
    paper: dict[str, Any],
    *,
    payload: dict[str, Any],
    selected: str,
    pmap: dict[str, dict[str, Any]],
    service: SRGApplicationService,
) -> None:
    """Compact paper viewer with abstract — shown beside the graph."""
    title = (paper.get("title") or "Untitled").strip()
    url = _paper_primary_url(paper)
    if url:
        st.markdown(f"### [{title}]({url})")
    else:
        st.markdown(f"### {title}")

    authors = _authors_line(paper, max_names=10)
    if authors:
        st.caption(authors)

    venue = _venue_line(paper)
    yr = paper.get("year")
    cites = _openalex_citation_count(paper)
    meta: list[str] = []
    if venue:
        meta.append(venue)
    if yr is not None:
        meta.append(str(int(yr)))
    if cites is not None:
        meta.append(f"{cites:,} citations")
    if meta:
        st.caption(" · ".join(meta))

    ax = (paper.get("arxiv_id") or "").strip()
    doi = (paper.get("doi") or "").strip()
    link_bits: list[str] = []
    if ax:
        link_bits.append(f"[arXiv](https://arxiv.org/abs/{ax})")
    if doi:
        d = doi.replace("https://doi.org/", "").strip()
        link_bits.append(f"[DOI](https://doi.org/{d})")
    if link_bits:
        st.markdown(" · ".join(link_bits))

    abs_txt, abs_source = _ensure_paper_abstract(paper, paper_id=selected, payload=payload)
    if abs_txt:
        st.markdown("**Abstract**")
        if abs_source == "arxiv":
            st.caption("Loaded from arXiv")
        st.markdown(abs_txt[:6000] + ("…" if len(abs_txt) > 6000 else ""))
    else:
        if _arxiv_id_for_paper(paper):
            st.caption("Could not load abstract from arXiv for this record.")
        else:
            st.caption("No abstract available (no arXiv id on record).")

    related = _neighbor_titles(payload, selected, pmap, limit=5)
    if related:
        st.markdown("**Related**")
        for nid, t in related:
            if st.button(
                t[:70] + ("…" if len(t) > 70 else ""),
                key=f"rel_{nid}",
                use_container_width=True,
            ):
                st.session_state.selected_node = nid
                st.rerun()

    if DEV_MODE and _paper_needs_metadata_refresh(paper):
        if st.button("Refresh metadata", key="lite_hydrate"):
            if service.hydrate_graph_node_metadata(st.session_state.payload, selected):
                st.success("Updated.")
                st.rerun()
            else:
                st.warning("Could not refresh metadata.")


def _render_dev_diagnostics(payload: dict[str, Any], pmap: dict[str, dict[str, Any]]) -> None:
    """Hidden pipeline diagnostics — only when DEV_MODE is True."""
    rq = payload.get("retrieval_quality") or {}
    lite_ux = payload.get("lite_ux") or {}
    rh = lite_ux.get("retrieval_health") or {}
    with st.expander("DEV: retrieval & pipeline diagnostics", expanded=False):
        for msg in rq.get("warnings") or []:
            st.warning(msg)
        if rh.get("signals"):
            st.json(rh)
        exd = payload.get("expand_diagnostics") or {}
        if exd:
            st.json(exd)
        for s in payload.get("requested_seeds") or []:
            st.caption(f"{s.get('provider')}:{s.get('id')}")


def run_srg_lite_ui() -> None:
    st.set_page_config(page_title="SRG Lite", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(LITE_CSS, unsafe_allow_html=True)

    if "service" not in st.session_state:
        st.session_state.service = SRGApplicationService()
    if "payload" not in st.session_state:
        st.session_state.payload = None
    if "selected_node" not in st.session_state:
        st.session_state.selected_node = None
    if "lite_explore_pending" not in st.session_state:
        st.session_state.lite_explore_pending = None

    service: SRGApplicationService = st.session_state.service

    st.markdown("# SRG Lite")
    st.markdown('<p class="lite-tagline">Discover the papers that matter — and how they connect.</p>', unsafe_allow_html=True)

    explore_pending = (st.session_state.lite_explore_pending or "").strip()
    exploring = bool(explore_pending)

    q_col, btn_col = st.columns([5, 1], vertical_alignment="center")
    with q_col:
        q = st.text_input(
            "Topic, paper title, DOI, or arXiv id",
            label_visibility="collapsed",
            placeholder="e.g. direct preference optimization · or paste a DOI / arXiv id",
            key="lite_query",
        )
    with btn_col:
        discover = st.button(
            "Running…" if exploring else "Explore",
            type="primary",
            use_container_width=True,
            key="lite_discover",
            disabled=exploring,
        )

    if discover and not exploring:
        qq = (q or "").strip()
        if len(qq) < 3 and not re.search(r"\b10\.\d{4,9}/", qq, flags=re.IGNORECASE):
            st.info("Use at least three characters, or paste a DOI (10.xxxx/…).")
        else:
            st.session_state.lite_explore_pending = qq
            st.rerun()

    if exploring:
        st.session_state.lite_explore_pending = None
        try:
            with st.spinner("Building research map…"):
                reading = generate_reading_list(
                    explore_pending,
                    discovery_options=LITE_DISCOVERY_OPTIONS,
                    service=service,
                )
            st.session_state.payload = reading["payload"]
            st.session_state.selected_node = None
            st.session_state.lite_filters_initialized = False
            st.session_state.lite_show_graph_labels = True
        except ValueError as exc:
            st.info(str(exc))
        except RuntimeError:
            st.warning(
                "No papers matched that input. Try a DOI, arXiv id, or a more specific title."
            )

    payload = st.session_state.payload
    if not payload:
        st.caption("Enter a research topic above to explore connected literature.")
        return

    if DEV_MODE:
        _render_dev_diagnostics(payload, _paper_map(payload.get("papers") or []))

    papers_all = [p for p in (payload.get("papers") or []) if not p.get("graph_noise")]
    pmap = _paper_map(payload.get("papers") or [])

    lite_ux = payload.get("lite_ux")
    if not isinstance(lite_ux, dict) or int(lite_ux.get("version") or 0) < 2:
        try:
            from .lite_ux import build_lite_ux_payload
        except ImportError:
            from srg.lite_ux import build_lite_ux_payload  # type: ignore[no-redef]

        payload["lite_ux"] = build_lite_ux_payload(payload)
        st.session_state.payload = payload
        lite_ux = payload["lite_ux"]

    foundational_entries = [
        it
        for it in (lite_ux.get("foundational_papers") or [])
        if isinstance(it, dict) and str(it.get("paper_id") or "").strip()
    ]
    foundational_ids = {str(it.get("paper_id")) for it in foundational_entries}

    years = sorted({int(p["year"]) for p in papers_all if p.get("year") is not None})
    domains_available = sorted({_fmt_domain(p.get("domain")) for p in papers_all})

    if not st.session_state.get("lite_filters_initialized") and years:
        st.session_state.lite_year_range = (years[0], years[-1])
        st.session_state.lite_filter_domains = []
        st.session_state.lite_foundational_only = False
        st.session_state.lite_filters_initialized = True

    year_range: tuple[int, int] | None = None
    yr_lo, yr_hi = 0, 9999

    left_panel, graph_panel, detail_panel = st.columns([0.95, 2.85, 1.25], gap="medium")

    with left_panel:
        st.markdown("#### Filters")
        sel_domains = st.multiselect(
            "Domain",
            options=domains_available,
            default=st.session_state.get("lite_filter_domains") or [],
            key="lite_filter_domains",
            placeholder="All domains",
        )
        if years:
            yr_lo, yr_hi = st.slider(
                "Year range",
                min_value=years[0],
                max_value=years[-1],
                value=st.session_state.get("lite_year_range") or (years[0], years[-1]),
                key="lite_year_range",
            )
            year_range = (yr_lo, yr_hi)
        else:
            st.caption("No year metadata")
        foundational_only = st.toggle(
            "Foundational only",
            value=bool(st.session_state.get("lite_foundational_only")),
            key="lite_foundational_only",
        )
        st.download_button(
            "Export Markdown",
            data=build_markdown_report(
                payload,
                project_title="SRG Lite reading list",
                clean=bool(LITE_DISCOVERY_OPTIONS.get("export_clean_reading_mode", True)),
            ),
            file_name="srg_lite_reading_list.md",
            mime="text/markdown",
            key="lite_dl_md",
            use_container_width=True,
        )
        st.download_button(
            "Export BibTeX",
            data=build_bibtex(payload),
            file_name="srg_lite_library.bib",
            mime="text/plain",
            key="lite_dl_bib",
            use_container_width=True,
        )

        papers_filtered = _apply_ui_filters(
            papers_all,
            domains=sel_domains or None,
            year_range=year_range,
            foundational_only=foundational_only,
            foundational_ids=foundational_ids,
        )
        year_dist = _year_distribution(papers_filtered)
        domain_dist = _domain_distribution(papers_filtered)

        st.markdown("**Publication timeline**")
        if year_dist:
            st.bar_chart(year_dist, height=140)
        else:
            st.caption("No years in filter.")

        st.markdown("**Domains**")
        if domain_dist:
            st.bar_chart(domain_dist, height=140)
        else:
            st.caption("No domain metadata.")

    visible_ids = {str(p["id"]) for p in papers_filtered}

    selected = st.session_state.get("selected_node")
    if selected and selected not in visible_ids and not pmap.get(selected):
        st.session_state.selected_node = None
        selected = None

    display_floor = float(payload.get("lite_display_relevance_floor") or 0.0)
    graph_base = _filter_graph_for_display(payload, display_floor, hide_weak_edges=False)
    graph_display = _restrict_graph(
        graph_base,
        visible_ids,
        max_nodes=LITE_MAX_GRAPH_NODES,
    )
    _apply_lite_display_layout(graph_display)
    n_full_edges = len((payload.get("graph") or {}).get("edges") or [])
    picked_node: str | None = None

    if "lite_show_graph_labels" not in st.session_state:
        st.session_state.lite_show_graph_labels = True

    with graph_panel:
        st.markdown("### Research graph")
        show_graph_labels = st.toggle(
            "Node labels",
            key="lite_show_graph_labels",
            help="Short title (first 2 words) on each node; hover for full title.",
        )

        pruned_edges = _prune_graph_edges(
            graph_display.get("edges", []),
            include_weak=False,
        )
        graph_viz = {
            "nodes": graph_display.get("nodes", []),
            "edges": pruned_edges,
        }
        n_disp_nodes = len(graph_viz["nodes"])
        n_disp_edges = len(pruned_edges)

        graph_config = Config(
            width="100%",
            height=LITE_GRAPH_HEIGHT,
            directed=True,
            physics=False,
            hierarchical=False,
            edges={
                "arrows": {"to": {"enabled": True, "scaleFactor": 0.45}},
                "smooth": {"enabled": True, "type": "continuous", "roundness": 0.2},
                "color": {"inherit": False},
            },
            interaction={"hover": True, "tooltipDelay": 120, "zoomView": True, "dragView": True},
            groups={
                "uploaded": {"color": "#1D4ED8"},
                "discovered": {"color": "#A1A1AA"},
                "anchor_search": {"color": "#059669"},
                "foundational": {"color": "#D4AF37"},
                "missing_link": {"color": "#CA8A04"},
            },
        )
        nodes_g, edges_g = _to_agraph_payload_lite(graph_viz, show_labels=show_graph_labels)
        picked_node = agraph(nodes=nodes_g, edges=edges_g, config=graph_config)
        if picked_node:
            st.session_state.selected_node = picked_node
        st.caption(
            f"{n_disp_nodes} papers · {n_disp_edges} links shown "
            f"({n_full_edges} in full map). Hover for title · click a dot → details on the right."
        )
        st.caption(
            "Link color: **dark blue** = strong citation · **light blue** = moderate. "
            "Green = anchors · Gold = foundational."
        )
        if n_disp_edges <= 2 and n_disp_nodes > 8:
            st.info(
                "Few citation arrows usually means sparse reference metadata for this slice, "
                "not a rendering issue. Try a DOI or arXiv seed for a denser map."
            )

    with detail_panel:
        detail_slot = st.empty()

    st.divider()
    st.markdown("### Foundational papers")
    found_visible = [
        it for it in foundational_entries if str(it.get("paper_id") or "") in visible_ids
    ][:10]
    if not found_visible:
        st.caption("No foundational papers match the current filters.")
    else:
        found_left, found_right = found_visible[:5], found_visible[5:10]
        fcol_l, fcol_r = st.columns(2, gap="medium")
        with fcol_l:
            for it in found_left:
                pid = str(it.get("paper_id") or "")
                _render_foundational_card(it, pmap.get(pid))
        with fcol_r:
            for it in found_right:
                pid = str(it.get("paper_id") or "")
                _render_foundational_card(it, pmap.get(pid))
            if not found_right:
                st.caption("")

    selected = picked_node or st.session_state.get("selected_node")
    with detail_slot.container():
        st.markdown("### Paper details")
        if not selected:
            st.caption(
                "Click a **gold**, **green**, or **grey** dot in the graph — or **View in panel →** "
                "on a foundational paper below."
            )
        else:
            paper = pmap.get(selected)
            if not paper:
                st.warning("Paper not found in the current view.")
            else:
                with st.container(height=LITE_GRAPH_HEIGHT, border=True):
                    _render_paper_details_panel(
                        paper,
                        payload=payload,
                        selected=str(selected),
                        pmap=pmap,
                        service=service,
                    )


if __name__ == "__main__":
    run_srg_lite_ui()
