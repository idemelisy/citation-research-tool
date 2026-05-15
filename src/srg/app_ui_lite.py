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
    from srg.lite_ux import start_here_tags_for_paper
    from srg.app_ui import (
        _filter_graph_for_display,
        _importance_score_for_ui,
        _paper_map,
        _paper_needs_metadata_refresh,
        _related_edges,
        _to_agraph_payload,
    )
    from srg.explainability import why_it_matters_one_line
    from srg.synthesis_export import build_bibtex, build_markdown_report
    from srg.core.defaults import LITE_DISCOVERY_OPTIONS
    from srg.core.pipeline import generate_reading_list
else:
    from .api_adapter import SRGApplicationService
    from .core.defaults import LITE_DISCOVERY_OPTIONS
    from .core.pipeline import generate_reading_list
    from .lite_ux import start_here_tags_for_paper
    from .app_ui import (
        _filter_graph_for_display,
        _importance_score_for_ui,
        _paper_map,
        _paper_needs_metadata_refresh,
        _related_edges,
        _to_agraph_payload,
    )
    from .explainability import why_it_matters_one_line
    from .synthesis_export import build_bibtex, build_markdown_report

LITE_CSS = """
<style>
    .lite-hero { color: #52525b; font-size: 0.95rem; margin-bottom: 1.25rem; line-height: 1.5; }
    .lite-tagline { color: #71717a; font-size: 0.85rem; margin-top: -0.5rem; margin-bottom: 1rem; }
</style>
"""


def _resolve_seed_title(seed: dict[str, Any], pmap: dict[str, dict[str, Any]]) -> str:
    """Match ``requested_seeds`` entry to a graph node to show human-readable title."""
    prov = (seed.get("provider") or "").strip().lower()
    sid = (seed.get("id") or "").strip()
    if not sid:
        return ""
    keys: list[str] = []
    if prov == "openalex":
        keys.append(f"openalex:{sid}")
    elif prov == "arxiv":
        try:
            from .ingestion import normalize_arxiv_list_id
        except ImportError:
            from srg.ingestion import normalize_arxiv_list_id  # type: ignore[no-redef]

        aid = normalize_arxiv_list_id(sid)
        keys.append(f"arxiv:{aid.lower()}")
    elif prov in ("doi", "crossref"):
        keys.append(f"crossref:{sid.lower()}")
        keys.append(f"doi:{sid.lower()}")
    for k in keys:
        p = pmap.get(k)
        if p and (p.get("title") or "").strip():
            return str(p.get("title")).strip()
    if prov == "openalex":
        for p in pmap.values():
            oid = str(p.get("openalex_id") or "").strip()
            if oid and oid.lower() == sid.lower():
                return str(p.get("title") or "").strip() or sid
    if prov == "arxiv":
        want = normalize_arxiv_list_id(aid).strip().lower()
        if want:
            for p in pmap.values():
                ttl = (p.get("title") or "").strip()
                if not ttl:
                    continue
                rid = str(p.get("arxiv_id") or "").strip()
                if rid and normalize_arxiv_list_id(rid).lower() == want:
                    return ttl
                pid = str(p.get("id") or "")
                if pid.lower().startswith("arxiv:"):
                    rid2 = normalize_arxiv_list_id(pid.split(":", 1)[-1]).lower()
                    if rid2 == want:
                        return ttl
                doi = str(p.get("doi") or "").lower()
                if want in doi and ("arxiv" in doi or "48550" in doi):
                    return ttl
    return ""


def _fmt_domain(domain: str | None) -> str:
    d = (domain or "").strip() or "general"
    return d.replace("_", " ").title()


def _papers_by_branch(
    papers: list[dict[str, Any]],
    *,
    intent_mode_v2: str | None = None,
    query_text: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for p in papers:
        if p.get("graph_noise"):
            continue
        key = _fmt_domain(p.get("domain"))
        buckets.setdefault(key, []).append(p)
    for k, rows in buckets.items():
        buckets[k] = sorted(
            rows,
            key=lambda p: _importance_score_for_ui(
                p, intent_mode_v2=intent_mode_v2, query_text=query_text
            ),
            reverse=True,
        )
    return dict(sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0])))


def _research_branches_for_lite(
    papers: list[dict[str, Any]],
    payload: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Query-aware topic branches (roadmap 2.1); fall back to domain buckets when needed."""
    try:
        from .synthesis_export import group_papers_query_aware_topics
    except ImportError:
        from srg.synthesis_export import group_papers_query_aware_topics

    clean = [p for p in papers if not p.get("graph_noise")]
    qp = (payload or {}).get("query_profile") or {}
    by_topic = group_papers_query_aware_topics(clean, qp)
    if len(by_topic) >= 2:
        return by_topic
    _qt = (qp.get("query_text") or "").strip() or None
    return _papers_by_branch(
        clean,
        intent_mode_v2=(qp.get("intent_mode_v2") or None),
        query_text=_qt,
    )


def _branch_description(branch_label: str) -> str:
    low = branch_label.lower()
    if low in ("general", "general preference alignment", "uncategorized"):
        return "Mixed papers from your query — grouped by citation structure in the map."
    return "Papers in this direction appear together in the citation neighborhood."


def _branch_label_for_paper_id(pid: str, branches_payload: list[dict[str, Any]]) -> str:
    for b in branches_payload:
        if not isinstance(b, dict):
            continue
        pids = [str(x) for x in (b.get("paper_ids") or [])]
        if str(pid) in pids:
            return str(b.get("label") or "").strip()
    return ""


def _literature_branch_rows(
    branches_payload: list[dict[str, Any]],
    fallback_branches: dict[str, list[dict[str, Any]]],
    pmap: dict[str, dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]], str, str]]:
    if branches_payload:
        rows: list[tuple[str, list[dict[str, Any]], str, str]] = []
        for b in branches_payload[:12]:
            if not isinstance(b, dict):
                continue
            label = str(b.get("label") or "Literature branch").strip()
            plist = [pmap[pid] for pid in (b.get("paper_ids") or []) if pmap.get(pid)]
            summ = str(b.get("summary") or "").strip()
            why = str(b.get("why_included") or "").strip()
            if plist:
                rows.append((label, plist, summ, why))
        return rows
    return [(k, v, _branch_description(k), "") for k, v in list(fallback_branches.items())[:12]]


def _trust_explanation_lines(paper: dict[str, Any]) -> list[str]:
    """Trust / explainability — citation-backed placement and field position (coherence phase)."""
    lines: list[str] = []
    wim = (paper.get("why_it_matters") or "").strip()
    if wim:
        for part in wim.split("\n"):
            p = part.strip()
            if p:
                lines.append(p)
    tier = (paper.get("field_coherence_tier") or "").strip().lower()
    if tier == "core":
        lines.append("Map position: core — close to your anchors or strongly tied to the query literature.")
    elif tier == "near_core":
        lines.append("Map position: near-core — one–two citation steps from anchors.")
    elif tier == "peripheral":
        lines.append("Map position: peripheral — farther out; skim before committing deep reading time.")
    hops = paper.get("anchor_graph_hops")
    if hops is not None and int(hops) < 99:
        lines.append(f"Citation hops from query seeds (undirected): {int(hops)}.")
    rel = float(paper.get("relation_expand_score") or 0.0)
    coc = int(paper.get("relation_max_cocitation") or 0)
    cou = int(paper.get("relation_max_coupling") or 0)
    dh = int(paper.get("relation_direct_hits") or 0)
    origin = (paper.get("seed_origin") or "").strip()
    if origin == "api_search":
        lines.append("Seeded from your query search (not a lone semantic hit).")
    if dh > 0 and rel > 0:
        lines.append("Direct citation link to a starting paper in this map.")
    elif coc >= 2:
        lines.append("Often cited together with work near your anchors (shared citing community).")
    elif cou >= 2:
        lines.append("Overlaps references with anchor papers (same bibliographic thread).")
    elif paper.get("is_foundational_hub"):
        lines.append("Central in this slice — widely linked or highly cited here.")
    elif rel > 0 and paper.get("seed_origin") == "discovered":
        lines.append("Reached through citation expansion from your anchors.")
    isim = paper.get("intent_similarity")
    if isim is not None:
        try:
            iv = float(isim)
            if iv >= 0.55:
                lines.append(f"Intent verification: strong query–document alignment (≈{iv:.2f}).")
            elif iv >= 0.35:
                lines.append(f"Intent verification: moderate alignment with your query (≈{iv:.2f}).")
        except (TypeError, ValueError):
            pass
    return lines[:12]


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


def _recent_important(
    papers: list[dict[str, Any]],
    *,
    payload: dict[str, Any] | None = None,
    current_year: int = 2026,
    window: int = 5,
) -> list[dict[str, Any]]:
    lo = current_year - window
    cand = [p for p in papers if not p.get("graph_noise") and p.get("year") is not None and int(p["year"]) >= lo]
    _qp = (payload or {}).get("query_profile") or {}
    _iv2 = _qp.get("intent_mode_v2")
    _qt = (_qp.get("query_text") or "").strip() or None
    cand.sort(
        key=lambda p: _importance_score_for_ui(p, intent_mode_v2=_iv2, query_text=_qt),
        reverse=True,
    )
    return cand[:12]


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
    if "lite_explore_pending" not in st.session_state:
        st.session_state.lite_explore_pending = None

    service: SRGApplicationService = st.session_state.service

    st.markdown("# SRG Lite")
    st.markdown(
        '<p class="lite-tagline">Where should you start in this research area?</p>',
        unsafe_allow_html=True,
    )

    explore_pending = (st.session_state.lite_explore_pending or "").strip()
    exploring = bool(explore_pending)

    q_col, btn_col = st.columns([5, 1], vertical_alignment="center")
    with q_col:
        q = st.text_input(
            "Topic, paper title, DOI, or arXiv id",
            label_visibility="collapsed",
            placeholder="e.g. differential privacy in federated learning · or paste a DOI / arXiv id",
            key="lite_query",
        )
    with btn_col:
        discover = st.button(
            "Running…" if exploring else "Explore",
            type="primary",
            use_container_width=True,
            key="lite_discover",
            disabled=exploring,
            help="Exploration in progress — please wait." if exploring else "Build a citation map for this query.",
        )

    if discover and not exploring:
        qq = (q or "").strip()
        if len(qq) < 3 and not re.search(r"\b10\.\d{4,9}/", qq, flags=re.IGNORECASE):
            st.info("Use at least three characters, or paste a DOI (10.xxxx/…).")
        else:
            st.session_state.lite_explore_pending = qq
            st.rerun()

    if exploring:
        qq = explore_pending
        st.session_state.lite_explore_pending = None
        try:
            with st.spinner("Resolving anchors and building citation graph…"):
                reading = generate_reading_list(
                    qq,
                    discovery_options=LITE_DISCOVERY_OPTIONS,
                    service=service,
                )
            st.session_state.payload = reading["payload"]
            st.session_state.selected_node = None
        except ValueError as exc:
            st.info(str(exc))
        except RuntimeError:
            st.warning(
                "No papers matched that input. The phrase may be too vague for OpenAlex, or outside coverage. "
                "Try a DOI (`10....`), an arXiv id (`1234.5678`), or paste an exact paper title."
            )

    payload = st.session_state.payload
    if not payload:
        st.markdown(
            '<p class="lite-hero"></p>',
            unsafe_allow_html=True,
        )
        return

    pin_adv = st.checkbox(
        "Pin advanced sections open",
        value=bool(st.session_state.get("lite_pin_adv", False)),
        key="lite_pin_adv",
    )

    rq = payload.get("retrieval_quality") or {}
    for msg in rq.get("warnings") or []:
        st.warning(msg)
    v21_rq = rq.get("v2_1_diagnostics") or {}
    sc_rq = v21_rq.get("semantic_control") or {}
    iv_rq = v21_rq.get("intent_verification") or {}
    if sc_rq.get("enabled") or (iv_rq.get("enabled") and iv_rq.get("backend")):
        with st.expander("Semantic & intent diagnostics (detail)", expanded=pin_adv):
            if sc_rq.get("enabled"):
                doms = sc_rq.get("query_inferred_domains") or []
                dom_txt = ", ".join(doms) if doms else "none inferred"
                st.caption(
                    f"Semantic control (anti-drift): δ={sc_rq.get('semantic_control_drift_lambda')}, "
                    f"γ={sc_rq.get('semantic_control_local_gamma')}; inferred query domains: {dom_txt}."
                )
            if iv_rq.get("enabled") and iv_rq.get("backend"):
                be = iv_rq.get("backend")
                md = iv_rq.get("model") or "n/a"
                st.caption(
                    f"Intent verification: backend={be}, model={md}; mean query–paper intent similarity ≈ "
                    f"{iv_rq.get('mean_intent_similarity', '—')}."
                )

    papers = [p for p in (payload.get("papers") or []) if not p.get("graph_noise")]
    pmap = _paper_map(payload["papers"])
    _lux0 = payload.get("lite_ux")
    if not isinstance(_lux0, dict) or int(_lux0.get("version") or 0) < 2:
        try:
            from .lite_ux import build_lite_ux_payload
        except ImportError:
            from srg.lite_ux import build_lite_ux_payload  # type: ignore[no-redef]

        payload["lite_ux"] = build_lite_ux_payload(payload)
        st.session_state.payload = payload
    lite_ux = payload.get("lite_ux") if isinstance(payload.get("lite_ux"), dict) else {}
    branches_payload = lite_ux.get("branches") if isinstance(lite_ux.get("branches"), list) else []
    foundational_entries = (
        lite_ux.get("foundational_papers") if isinstance(lite_ux.get("foundational_papers"), list) else []
    )
    rh = lite_ux.get("retrieval_health") if isinstance(lite_ux.get("retrieval_health"), dict) else {}
    signals = rh.get("signals") if isinstance(rh.get("signals"), list) else []
    if signals:
        st.subheader("Retrieval health")
        sig_cols = st.columns(min(len(signals), 4))
        for i, sig in enumerate(signals[:4]):
            if not isinstance(sig, dict):
                continue
            with sig_cols[i % len(sig_cols)]:
                st.markdown(f"**{sig.get('label', 'Signal')}**  \n`{sig.get('level', '—')}`")
                if sig.get("detail"):
                    st.caption(str(sig["detail"])[:140])
        es = rh.get("empty_state") if isinstance(rh.get("empty_state"), dict) else {}
        if es.get("active"):
            st.warning(es.get("message") or "This slice looks weak or ambiguous — proceed carefully.")
            for s in es.get("suggestions") or []:
                st.caption(f"· {s}")

    ar_notes = rq.get("anchor_resolution_notes") or []
    req_seeds = payload.get("requested_seeds") or []
    if ar_notes or req_seeds:
        with st.expander("Anchor seeds (retrieval diagnostics)", expanded=pin_adv):
            st.caption(
                "These are the works used to anchor citation expansion. If the list looks off-topic, "
                "results will follow that neighborhood — try a DOI, arXiv id, or a narrower phrase."
            )
            for line in ar_notes:
                st.markdown(f"- {line}")
            if req_seeds:
                st.markdown("**Resolved seed list**")
                for i, s in enumerate(req_seeds, start=1):
                    prov = (s.get("provider") or "?").strip()
                    sid = (s.get("id") or "").strip()
                    origin = (s.get("origin") or "").strip()
                    ttl = _resolve_seed_title(s, pmap)
                    if ttl:
                        st.markdown(f"{i}. `{prov}:{sid}` · _{origin}_ — **{ttl[:140]}{'…' if len(ttl) > 140 else ''}**")
                    else:
                        st.markdown(f"{i}. `{prov}:{sid}` · _{origin}_ — _(title not in current graph slice)_")
    branches_fb = _research_branches_for_lite(papers, payload)
    branch_rows = _literature_branch_rows(branches_payload, branches_fb, pmap)
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
            "discovered_edge": {"color": "#94A3B8"},
            "anchor_search": {"color": "#059669"},
            "semantic_broadened": {"color": "#9333EA"},
            "missing_link": {"color": "#CA8A04"},
            "foundational": {"color": "#D4AF37"},
        },
    )

    valid_foundational = [
        it
        for it in foundational_entries[:12]
        if isinstance(it, dict) and str(it.get("paper_id") or "").strip()
    ]
    if valid_foundational:
        st.subheader("Foundational papers")
        st.caption("Canonical method-defining works for this query (exact method matches only).")
        for it in valid_foundational:
            pid = str(it.get("paper_id") or "").strip()
            p = pmap.get(pid)
            ttl = (it.get("title") or (p.get("title") if p else "") or pid).strip()
            ol = (it.get("one_line") or "").strip()
            yr = it.get("year")
            yr_txt = f" ({yr})" if yr is not None else ""
            st.markdown(f"- **{ttl[:120]}{'…' if len(ttl) > 120 else ''}**{yr_txt}")
            if ol:
                st.caption(ol)
        st.divider()

    insights_list = lite_ux.get("insights") if isinstance(lite_ux.get("insights"), list) else []
    if insights_list:
        with st.expander("Insights (exploration quality)", expanded=False):
            _ins_first = True
            for ins in insights_list:
                if not isinstance(ins, dict):
                    continue
                if not _ins_first:
                    st.markdown("---")
                _ins_first = False
                st.markdown(f"**{ins.get('title', 'Insight')}**")
                if ins.get("body"):
                    st.markdown(str(ins["body"]))
        st.divider()

    trends = (payload.get("discovery") or {}).get("trends") or {}
    if trends and isinstance(trends, dict):
        with st.expander("Trend & gap signals (discovery pass)", expanded=pin_adv):
            for tk, tv in list(trends.items())[:24]:
                st.caption(f"**{tk}** — {tv}")

    left_w, mid_w, right_w = st.columns([1.05, 2.3, 1.05], gap="medium")

    with left_w:
        st.subheader("Literature branches")
        if not branch_rows:
            st.caption("No branch groups yet.")
        else:
            qp0 = payload.get("query_profile") or {}
            for branch_name, plist, summ, why in branch_rows:
                with st.expander(f"{branch_name} ({len(plist)})", expanded=False):
                    st.caption(summ if summ else _branch_description(branch_name))
                    if why:
                        st.markdown(f"> {why}")
                    for _br_obj in branches_payload:
                        if isinstance(_br_obj, dict) and str(_br_obj.get("label") or "") == branch_name:
                            warn = _br_obj.get("purity_warning")
                            if warn:
                                st.caption(str(warn))
                            break
                    try:
                        from .topical_ranking import semantic_purity_score
                    except ImportError:
                        from srg.topical_ranking import semantic_purity_score  # type: ignore[no-redef]

                    top_branch = sorted(
                        plist,
                        key=lambda p: semantic_purity_score(p, qp0),
                        reverse=True,
                    )[:4]
                    for p in top_branch:
                        if st.button(
                            (p.get("title") or p["id"])[:72]
                            + ("…" if len(p.get("title", "") or "") > 72 else ""),
                            key=f"lb_{p['id']}",
                            use_container_width=True,
                        ):
                            st.session_state.selected_node = p["id"]
                            st.rerun()

    with mid_w:
        st.subheader("Research map")
        st.caption(
            "Curated subgraph: weak periphery and low-confidence edges hidden for readability. "
            "Green: query anchors · Purple: semantic broadening · Grey: citation expansion · "
            "Gold: foundational · Amber: bridge · Blue: PDF."
        )
        picked = agraph(nodes=nodes_g, edges=edges_g, config=graph_config)
        if picked:
            st.session_state.selected_node = picked

        exd = payload.get("expand_diagnostics") or {}
        if isinstance(exd, dict) and exd:
            with st.expander("Graph expansion diagnostics", expanded=False):
                for ek, ev in list(exd.items())[:28]:
                    if isinstance(ev, (dict, list)):
                        st.caption(f"**{ek}**")
                        st.write(ev)
                    else:
                        st.caption(f"**{ek}** — {ev}")

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "Export reading list (Markdown)",
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
        br_lbl = _branch_label_for_paper_id(str(selected), branches_payload)
        if br_lbl:
            st.caption(f"Literature branch: {br_lbl}")
        tags_ui = start_here_tags_for_paper(paper, payload.get("query_profile"))
        if tags_ui:
            st.caption(" · ".join(tags_ui))
        _qp2 = payload.get("query_profile") or {}
        st.markdown("**Relation to your query**")
        st.caption(
            why_it_matters_one_line(
                paper,
                query_text=str(_qp2.get("query_text") or "").strip(),
                intent_label=str(_qp2.get("intent_mode_v2") or _qp2.get("query_intent") or "exploratory"),
            )
        )
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
        primary_bits: list[str] = []
        if yr is not None:
            primary_bits.append(f"Year: {yr}")
        if oa_cites is not None:
            primary_bits.append(f"Citations (OpenAlex): {oa_cites:,}")
        if cc is not None:
            primary_bits.append(f"Connections in this map: {cc}")
        if primary_bits:
            st.caption(" · ".join(primary_bits))

        technical_bits: list[str] = []
        ftier = (paper.get("field_coherence_tier") or "").strip()
        if ftier:
            technical_bits.append(f"Map position (coherence tier): {ftier.replace('_', ' ')}")
        sem_fit = paper.get("semantic_fit_score")
        if sem_fit is not None:
            try:
                technical_bits.append(f"Semantic fit (vs query): {float(sem_fit):.2f}")
            except (TypeError, ValueError):
                pass
        gs = paper.get("graph_score")
        cs = paper.get("canonicality_score")
        if gs is not None:
            try:
                technical_bits.append(f"Graph score (hub-adjusted): {float(gs):.2f}")
            except (TypeError, ValueError):
                pass
        if cs is not None:
            try:
                technical_bits.append(f"Canonicality: {float(cs):.2f}")
            except (TypeError, ValueError):
                pass
        ctxn = paper.get("context_score_norm")
        if ctxn is not None:
            try:
                technical_bits.append(f"Neighborhood coherence: {float(ctxn):.2f}")
            except (TypeError, ValueError):
                pass
        fs = paper.get("final_score")
        if fs is not None:
            try:
                technical_bits.append(f"Final fused score: {float(fs):.4f}")
            except (TypeError, ValueError):
                pass
        ah = paper.get("anchor_graph_hops")
        if ah is not None and int(ah) < 98:
            technical_bits.append(f"Hops from query anchors: {int(ah)}")
        lsc = paper.get("local_semantic_connectivity")
        if lsc is not None:
            try:
                technical_bits.append(f"Neighbor semantic coherence: {float(lsc):.2f}")
            except (TypeError, ValueError):
                pass
        dpn = paper.get("semantic_drift_penalty_norm")
        if dpn is not None and float(dpn) > 0:
            try:
                technical_bits.append(f"Drift penalty (norm): {float(dpn):.2f}")
            except (TypeError, ValueError):
                pass
        isim = paper.get("intent_similarity")
        if isim is not None:
            try:
                technical_bits.append(f"Intent similarity: {float(isim):.2f}")
            except (TypeError, ValueError):
                pass
        ivb = paper.get("intent_verification_backend")
        if ivb:
            technical_bits.append(f"Intent backend: {ivb}")
        if technical_bits:
            with st.expander("Technical scores & diagnostics", expanded=False):
                st.caption(" · ".join(technical_bits))

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
