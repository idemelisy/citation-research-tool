"""SRG Lite — guided research-orientation UI (Streamlit).

Presentation layer aligned with SRG Lite product doc: single discovery input,
branch-oriented sidebar, simplified graph, paper detail panel, exports only.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import streamlit as st
from streamlit_agraph import Config, agraph

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from srg.api_adapter import SRGApplicationService
    from srg.app_ui import (
        _filter_graph_for_display,
        _importance_score_for_ui,
        _paper_map,
        _paper_needs_metadata_refresh,
        _related_edges,
        _to_agraph_payload,
    )
    from srg.synthesis_export import _group_papers_by_topic, build_bibtex, build_markdown_report
else:
    from .api_adapter import SRGApplicationService
    from .app_ui import (
        _filter_graph_for_display,
        _importance_score_for_ui,
        _paper_map,
        _paper_needs_metadata_refresh,
        _related_edges,
        _to_agraph_payload,
    )
    from .synthesis_export import _group_papers_by_topic, build_bibtex, build_markdown_report

# Field-aware discovery phase: citation neighborhoods first; lexical/topic signals secondary.
LITE_DISCOVERY_OPTIONS: dict[str, Any] = {
    "top_n": 50,
    "coupling_threshold": 2,
    "enable_two_hop": True,
    "concept_threshold": 0.3,
    "relation_topic_weight": 0.0,
    "foundational_boost": 1.2,
    "allowed_domains": ["cs", "social_sciences", "law", "biomedical", "physics", "ethics"],
    "lite_field_aware": True,
    "w_coupling": 0.75,
    "w_cocitation": 1.0,
    "w_direct": 2.0,
    # Retrieval repair & citation expansion phase — deeper hop budget, stronger neighborhoods.
    "lite_retrieval_repair": True,
    "two_hop_budget_floor": 200,
    # Field coherence & canonicality — stabilize anchors, penalize distant drift, curate visible graph.
    "lite_coherence_stabilization": True,
    "lite_user_graph_relevance_floor": 0.14,
    # SRG Lite v2 / v2.1 — intent fusion + semantic control (drift / domain / hubs); see semantic_control.py.
    "lite_v2_ranking": True,
    "lite_semantic_graph_ranking": False,
    "lite_semantic_control": True,
    # Intent verification + semantic alignment (embedding cosine if sentence-transformers installed).
    "lite_intent_verification": True,
}

LITE_CSS = """
<style>
    .lite-hero { color: #52525b; font-size: 0.95rem; margin-bottom: 1.25rem; line-height: 1.5; }
    .lite-tagline { color: #71717a; font-size: 0.85rem; margin-top: -0.5rem; margin-bottom: 1rem; }
</style>
"""


def _fmt_domain(domain: str | None) -> str:
    d = (domain or "").strip() or "general"
    return d.replace("_", " ").title()


def _papers_by_branch(papers: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for p in papers:
        if p.get("graph_noise"):
            continue
        key = _fmt_domain(p.get("domain"))
        buckets.setdefault(key, []).append(p)
    for k, rows in buckets.items():
        buckets[k] = sorted(rows, key=_importance_score_for_ui, reverse=True)
    return dict(sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0])))


def _research_branches_for_lite(papers: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Prefer title-derived research directions when varied; else inferred domain buckets."""
    clean = [p for p in papers if not p.get("graph_noise")]
    by_topic = _group_papers_by_topic(clean)
    if len(by_topic) >= 2:
        return by_topic
    return _papers_by_branch(clean)


def _branch_description(branch_label: str) -> str:
    low = branch_label.lower()
    if low in ("general", "general preference alignment", "uncategorized"):
        return "Mixed papers from your query — grouped by citation structure in the map."
    return "Papers in this direction appear together in the citation neighborhood."


def _trust_explanation_lines(paper: dict[str, Any]) -> list[str]:
    """Trust / explainability — citation-backed placement and field position (coherence phase)."""
    lines: list[str] = []
    tier = (paper.get("field_coherence_tier") or "").strip().lower()
    if tier == "core":
        lines.append("Field position: core — close to anchors or strong citation ties to the query literature.")
    elif tier == "near_core":
        lines.append("Field position: near-core — connected within one–two citation steps of anchors.")
    elif tier == "peripheral":
        lines.append("Field position: peripheral — farther from anchors; verify relevance before deep reading.")
    hops = paper.get("anchor_graph_hops")
    if hops is not None and int(hops) < 99:
        lines.append(f"Citation hops from query seeds (undirected): {int(hops)}.")
    rel = float(paper.get("relation_expand_score") or 0.0)
    coc = int(paper.get("relation_max_cocitation") or 0)
    cou = int(paper.get("relation_max_coupling") or 0)
    dh = int(paper.get("relation_direct_hits") or 0)
    origin = (paper.get("seed_origin") or "").strip()
    if origin == "api_search":
        lines.append("Included as a query anchor from search (not only semantic similarity).")
    if dh > 0 and rel > 0:
        lines.append("Direct citation relationship to an anchor paper in this map.")
    elif coc >= 2:
        lines.append("Strong co-citation overlap with papers near your anchors (shared citing community).")
    elif cou >= 2:
        lines.append("Shares many references with anchor papers (bibliographic coupling — same conversation).")
    elif paper.get("is_foundational_hub"):
        lines.append("Highly cited or central in this literature slice (foundational hub).")
    elif rel > 0 and paper.get("seed_origin") == "discovered":
        lines.append("Reached through citation expansion from your anchors, ranked by research proximity.")
    return lines[:8]


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


def _recent_important(papers: list[dict[str, Any]], *, current_year: int = 2026, window: int = 5) -> list[dict[str, Any]]:
    lo = current_year - window
    cand = [p for p in papers if not p.get("graph_noise") and p.get("year") is not None and int(p["year"]) >= lo]
    cand.sort(key=_importance_score_for_ui, reverse=True)
    return cand[:12]


def _start_here_papers(payload: dict[str, Any], pmap: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    recs = (payload.get("discovery") or {}).get("recommendations") or []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for r in recs:
        pid = r.get("paper_id")
        if not pid or pid in seen:
            continue
        p = pmap.get(pid)
        if p and not p.get("graph_noise"):
            out.append(p)
            seen.add(pid)
        if len(out) >= 8:
            break
    if len(out) < 5:
        ranked = sorted(
            [p for p in (payload.get("papers") or []) if not p.get("graph_noise")],
            key=_importance_score_for_ui,
            reverse=True,
        )
        for p in ranked:
            if p["id"] not in seen:
                out.append(p)
                seen.add(p["id"])
            if len(out) >= 8:
                break
    return out[:8]


def _neighbor_titles(payload: dict[str, Any], selected: str, pmap: dict[str, dict[str, Any]], limit: int = 6) -> list[str]:
    edges = _related_edges(payload["graph"], selected)
    titles: list[str] = []
    for e in edges[:40]:
        other = e["target"] if e["source"] == selected else e["source"]
        op = pmap.get(other)
        if op:
            t = (op.get("title") or other)[:120]
            titles.append(t)
        if len(titles) >= limit:
            break
    return titles


def _why_matters_lines(paper: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if paper.get("is_foundational_hub"):
        lines.append(
            "Marked as a foundational hub — widely cited or central to connections in this map."
        )
    elif paper.get("is_missing_link_candidate"):
        lines.append(
            "Highlighted as a bridge paper — often cited alongside works in your topic area."
        )
    elif paper.get("seed_origin") == "discovered":
        lines.append("Included from discovery and reference expansion for this query.")
    reasons = paper.get("inclusion_reasons") or []
    for r in reasons[:4]:
        if r and r not in ("metadata_only_mode",):
            lines.append(str(r))
    return lines[:5] or ["This paper helps orient you within the literature shown on the map."]


def run_srg_lite_ui() -> None:
    st.set_page_config(page_title="SRG Lite", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(LITE_CSS, unsafe_allow_html=True)

    if "service" not in st.session_state:
        st.session_state.service = SRGApplicationService()
    if "payload" not in st.session_state:
        st.session_state.payload = None
    if "selected_node" not in st.session_state:
        st.session_state.selected_node = None

    service: SRGApplicationService = st.session_state.service

    st.markdown("# SRG Lite")
    st.markdown(
        '<p class="lite-tagline">Where should you start in this research area?</p>',
        unsafe_allow_html=True,
    )

    q_col, btn_col = st.columns([5, 1])
    with q_col:
        q = st.text_input(
            "Topic, paper title, DOI, or arXiv id",
            label_visibility="collapsed",
            placeholder="e.g. differential privacy in federated learning · or paste a DOI / arXiv id",
            key="lite_query",
        )
    with btn_col:
        st.markdown("<div style='height: 1.6rem'></div>", unsafe_allow_html=True)
        discover = st.button("Explore", type="primary", use_container_width=True, key="lite_discover")

    if discover:
        qq = (q or "").strip()
        if len(qq) < 3 and not re.search(r"\b10\.\d{4,9}/", qq, flags=re.IGNORECASE):
            st.info("Use at least three characters, or paste a DOI (10.xxxx/…).")
        else:
            with st.spinner("Resolving multiple anchors and expanding citation neighborhoods…"):
                seeds = service.discover_from_query(qq, text_search_backend="openalex")
            if not seeds:
                st.warning("No papers matched that input. Try different wording or a specific DOI.")
            else:
                with st.spinner("Building citation graph: references, 2-hop expansion, coupling…"):
                    st.session_state.payload = service.run_pipeline(
                        seeds,
                        discipline=None,
                        discovery_options=LITE_DISCOVERY_OPTIONS,
                    )
                st.session_state.selected_node = None

    payload = st.session_state.payload
    if not payload:
        st.markdown(
            '<p class="lite-hero">Enter a topic or paper id. Retrieval uses multiple anchors and citation expansion '
            "(not a single semantic hit). SRG Lite v2.1 adds semantic control (drift, domain, hubs) and an intent verification pass "
            "(embedding cosine when optional deps are installed, else lexical fallback) to rescore against query-specific intent.</p>",
            unsafe_allow_html=True,
        )
        return

    rq = payload.get("retrieval_quality") or {}
    for msg in rq.get("warnings") or []:
        st.warning(msg)
    v21_rq = rq.get("v2_1_diagnostics") or {}
    sc_rq = v21_rq.get("semantic_control") or {}
    if sc_rq.get("enabled"):
        doms = sc_rq.get("query_inferred_domains") or []
        dom_txt = ", ".join(doms) if doms else "none inferred"
        st.caption(
            f"Semantic control (anti-drift): δ={sc_rq.get('semantic_control_drift_lambda')}, "
            f"γ={sc_rq.get('semantic_control_local_gamma')}; inferred query domains: {dom_txt}."
        )
    iv_rq = v21_rq.get("intent_verification") or {}
    if iv_rq.get("enabled") and iv_rq.get("backend"):
        be = iv_rq.get("backend")
        md = iv_rq.get("model") or "n/a"
        st.caption(
            f"Intent verification: backend={be}, model={md}; mean query–paper intent similarity ≈ {iv_rq.get('mean_intent_similarity', '—')}."
        )

    papers = [p for p in (payload.get("papers") or []) if not p.get("graph_noise")]
    pmap = _paper_map(payload["papers"])
    branches = _research_branches_for_lite(papers)
    foundational = [p for p in papers if p.get("is_foundational_hub")]
    recent_imp = _recent_important(papers)
    start_here = _start_here_papers(payload, pmap)

    display_floor = float(payload.get("lite_display_relevance_floor") or 0.0)
    graph_display = _filter_graph_for_display(payload, display_floor, hide_weak_edges=True)
    nodes_g, edges_g = _to_agraph_payload(graph_display)
    graph_config = Config(
        width="100%",
        height=720,
        directed=True,
        physics=True,
        hierarchical=False,
        groups={
            "uploaded": {"color": "#1D4ED8"},
            "discovered": {"color": "#D4D4D8"},
            "missing_link": {"color": "#CA8A04"},
            "foundational": {"color": "#D4AF37"},
        },
    )

    left_w, mid_w, right_w = st.columns([1.05, 2.3, 1.05], gap="medium")

    with left_w:
        st.subheader("Research branches")
        if not branches:
            st.caption("No branch groups yet.")
        else:
            for branch_name, plist in list(branches.items())[:12]:
                with st.expander(f"{branch_name} ({len(plist)})", expanded=False):
                    st.caption(_branch_description(branch_name))
                    found_b = [p for p in plist if p.get("is_foundational_hub")][:4]
                    recent_b = [p for p in plist if p.get("year") and int(p["year"]) >= 2021][:4]
                    if found_b:
                        st.markdown("**Foundational**")
                        for p in found_b:
                            if st.button(p.get("title", "")[:72] + ("…" if len(p.get("title", "")) > 72 else ""), key=f"lb_{p['id']}", use_container_width=True):
                                st.session_state.selected_node = p["id"]
                                st.rerun()
                    if recent_b:
                        st.markdown("**Recent**")
                        for p in recent_b:
                            if st.button(p.get("title", "")[:72] + ("…" if len(p.get("title", "")) > 72 else ""), key=f"lr_{p['id']}", use_container_width=True):
                                st.session_state.selected_node = p["id"]
                                st.rerun()

        st.subheader("Guided reading")
        st.markdown("**Start here**")
        for p in start_here[:6]:
            label = (p.get("title") or p["id"])[:76]
            if st.button(label + ("…" if len(p.get("title", "") or "") > 76 else ""), key=f"sh_{p['id']}", use_container_width=True):
                st.session_state.selected_node = p["id"]
                st.rerun()

        st.markdown("**Foundational**")
        for p in foundational[:6]:
            label = (p.get("title") or p["id"])[:76]
            if st.button(label + ("…" if len(p.get("title", "") or "") > 76 else ""), key=f"fd_{p['id']}", use_container_width=True):
                st.session_state.selected_node = p["id"]
                st.rerun()
        if not foundational:
            st.caption("No foundational hubs in this slice yet — try a broader query.")

        st.markdown("**Recent & influential**")
        for p in recent_imp[:6]:
            yr = p.get("year")
            label = f"[{yr}] " + (p.get("title") or p["id"])[:68]
            if st.button(label + ("…" if len(str(p.get("title", ""))) > 72 else ""), key=f"ri_{p['id']}", use_container_width=True):
                st.session_state.selected_node = p["id"]
                st.rerun()

        st.markdown("**Suggested reading order**")
        for i, p in enumerate(start_here[:5], start=1):
            st.caption(f"{i}. {(p.get('title') or p['id'])[:90]}…" if len(p.get("title", "") or "") > 90 else f"{i}. {p.get('title') or p['id']}")

    with mid_w:
        st.subheader("Research map")
        st.caption(
            "Curated subgraph: weak periphery and low-confidence edges hidden for readability. "
            "Gold: foundational · Amber: bridge · Grey: neighborhood."
        )
        picked = agraph(nodes=nodes_g, edges=edges_g, config=graph_config)
        if picked:
            st.session_state.selected_node = picked

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "Export reading list (Markdown)",
                data=build_markdown_report(payload, project_title="SRG Lite reading list"),
                file_name="srg_lite_reading_list.md",
                mime="text/markdown",
                key="lite_dl_md",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                "Export BibTeX",
                data=build_bibtex(payload),
                file_name="srg_lite_library.bib",
                mime="text/plain",
                key="lite_dl_bib",
                use_container_width=True,
            )

    selected = st.session_state.selected_node
    with right_w:
        st.subheader("Paper details")
        if not selected:
            st.caption("Click a node in the map or a paper in the left column.")
            return

        paper = pmap.get(selected)
        if not paper:
            st.warning("Paper not found in the current view.")
            return

        st.markdown(f"### {paper.get('title', 'Untitled')}")
        ftier = (paper.get("field_coherence_tier") or "").strip()
        if ftier:
            st.caption(f"Map position: {ftier.replace('_', ' ')} (core vs peripheral)")
        authors = paper.get("authors") or []
        if isinstance(authors, list) and authors:
            names = []
            for a in authors[:24]:
                if isinstance(a, dict):
                    names.append(str(a.get("name", "")))
                else:
                    names.append(str(a))
            st.caption(", ".join(n for n in names if n) + (" …" if len(authors) > 24 else ""))
        yr = paper.get("year")
        cc = paper.get("citation_count")
        oa_cites = _openalex_citation_count(paper)
        meta_bits = []
        if yr is not None:
            meta_bits.append(str(yr))
        if oa_cites is not None:
            meta_bits.append(f"Citations (OpenAlex): {oa_cites:,}")
        if cc is not None:
            meta_bits.append(f"Connections in this map: {cc}")
        sem_fit = paper.get("semantic_fit_score")
        if sem_fit is not None:
            try:
                meta_bits.append(f"Semantic fit (vs query): {float(sem_fit):.2f}")
            except (TypeError, ValueError):
                pass
        gs = paper.get("graph_score")
        cs = paper.get("canonicality_score")
        if gs is not None:
            try:
                meta_bits.append(f"Graph score (hub-adjusted): {float(gs):.2f}")
            except (TypeError, ValueError):
                pass
        if cs is not None:
            try:
                meta_bits.append(f"Canonicality: {float(cs):.2f}")
            except (TypeError, ValueError):
                pass
        ctxn = paper.get("context_score_norm")
        if ctxn is not None:
            try:
                meta_bits.append(f"Neighborhood coherence: {float(ctxn):.2f}")
            except (TypeError, ValueError):
                pass
        fs = paper.get("final_score")
        if fs is not None:
            try:
                meta_bits.append(f"Final fused score: {float(fs):.4f}")
            except (TypeError, ValueError):
                pass
        ah = paper.get("anchor_graph_hops")
        if ah is not None and int(ah) < 98:
            meta_bits.append(f"Hops from query anchors: {int(ah)}")
        lsc = paper.get("local_semantic_connectivity")
        if lsc is not None:
            try:
                meta_bits.append(f"Neighbor semantic coherence: {float(lsc):.2f}")
            except (TypeError, ValueError):
                pass
        dpn = paper.get("semantic_drift_penalty_norm")
        if dpn is not None and float(dpn) > 0:
            try:
                meta_bits.append(f"Drift penalty (norm): {float(dpn):.2f}")
            except (TypeError, ValueError):
                pass
        isim = paper.get("intent_similarity")
        if isim is not None:
            try:
                meta_bits.append(f"Intent similarity: {float(isim):.2f}")
            except (TypeError, ValueError):
                pass
        ivb = paper.get("intent_verification_backend")
        if ivb:
            meta_bits.append(f"Intent backend: {ivb}")
        if meta_bits:
            st.caption(" · ".join(meta_bits))

        abs_txt = (paper.get("abstract") or "").strip()
        if abs_txt:
            st.markdown(abs_txt[:4500] + ("…" if len(abs_txt) > 4500 else ""))

        doi = (paper.get("doi") or "").strip()
        ax = (paper.get("arxiv_id") or "").strip()
        if doi:
            st.markdown(f"DOI: [{doi}](https://doi.org/{doi.lstrip('https://doi.org/')})")
        if ax:
            st.markdown(f"arXiv: [{ax}](https://arxiv.org/abs/{ax})")

        st.markdown("**Why it matters here**")
        for line in _why_matters_lines(paper):
            st.markdown(f"- {line}")

        trust_lines = _trust_explanation_lines(paper)
        if trust_lines:
            st.markdown("**How it connects**")
            for line in trust_lines:
                st.markdown(f"- {line}")
        expl = (paper.get("expand_explanation") or "").strip()
        if expl:
            with st.expander("Technical expansion detail", expanded=False):
                st.caption(expl)

        neigh = _neighbor_titles(payload, selected, pmap)
        if neigh:
            st.markdown("**Related in this map**")
            for t in neigh:
                st.caption(f"· {t}")

        if _paper_needs_metadata_refresh(paper):
            if st.button("Refresh metadata", key="lite_hydrate"):
                if service.hydrate_graph_node_metadata(st.session_state.payload, selected):
                    st.success("Updated.")
                    st.rerun()
                else:
                    st.warning("Could not refresh metadata for this item.")


if __name__ == "__main__":
    run_srg_lite_ui()
