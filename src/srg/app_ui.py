from __future__ import annotations

from typing import Any
import os
from collections import Counter
from pathlib import Path
import re
import sys

import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from srg.api_adapter import SRGApplicationService
    from srg.ingestion import (
        PDF_MAPPING_QUALITY_THRESHOLD,
        fetch_arxiv_public_metadata,
        resolve_arxiv_id_for_lookup,
    )
    from srg.persistence import (
        RESTORED_PDF_SIGNATURE,
        ProjectStore,
        collect_pdf_blobs,
        feedback_events_from_jsonable,
        feedback_events_to_jsonable,
    )
    from srg.synthesis_export import build_bibtex, build_markdown_report
else:
    from .api_adapter import SRGApplicationService
    from .ingestion import (
        PDF_MAPPING_QUALITY_THRESHOLD,
        fetch_arxiv_public_metadata,
        resolve_arxiv_id_for_lookup,
    )
    from .persistence import (
        RESTORED_PDF_SIGNATURE,
        ProjectStore,
        collect_pdf_blobs,
        feedback_events_from_jsonable,
        feedback_events_to_jsonable,
    )
    from .synthesis_export import build_bibtex, build_markdown_report

DISCIPLINE_LABELS = [
    ("Default (infer from OpenAlex / manual)", "default"),
    ("Computer Science", "cs"),
    ("Biomedical", "biomedical"),
    ("Social Sciences", "social_sciences"),
    ("Physics", "physics"),
]

ADV_PRESETS: dict[str, dict[str, Any]] = {
    "scan": {
        "label": "🔍 Scan",
        "top_n": 30,
        "coupling_threshold": 4,
        "enable_two_hop": False,
        "concept_threshold": 0.8,
        "relation_topic_weight": 0.0,
        "foundational_boost": 0.0,
        "allowed_domains": ["cs", "social_sciences", "law"],
    },
    "explore": {
        "label": "📚 Explore (Default)",
        "top_n": 50,
        "coupling_threshold": 3,
        "enable_two_hop": True,
        "concept_threshold": 0.35,
        "relation_topic_weight": 0.1,
        "foundational_boost": 1.0,
        "allowed_domains": ["cs", "social_sciences", "law"],
    },
    "dig": {
        "label": "🧬 Dig",
        "top_n": 100,
        "coupling_threshold": 2,
        "enable_two_hop": True,
        "concept_threshold": 0.2,
        "relation_topic_weight": 0.35,
        "foundational_boost": 2.5,
        "allowed_domains": ["cs", "social_sciences", "law", "ethics", "biomedical", "physics"],
    },
}

DISCIPLINE_FILTER_OPTIONS = ["cs", "law", "ethics", "social_sciences"]


def _apply_adv_preset_to_state(preset_key: str) -> None:
    p = ADV_PRESETS.get(preset_key, ADV_PRESETS["explore"])
    st.session_state["adv_preset"] = preset_key
    st.session_state["adv_top_n"] = int(p["top_n"])
    st.session_state["adv_coupling_threshold"] = int(p["coupling_threshold"])
    st.session_state["adv_enable_two_hop"] = bool(p["enable_two_hop"])
    st.session_state["adv_concept_threshold"] = float(p["concept_threshold"])
    st.session_state["adv_relation_topic_weight"] = float(p["relation_topic_weight"])
    st.session_state["adv_foundational_boost"] = float(p["foundational_boost"])
    st.session_state["adv_allowed_domains"] = list(p["allowed_domains"])


def _current_discovery_options_from_state() -> dict[str, Any]:
    return {
        "top_n": int(st.session_state.get("adv_top_n", ADV_PRESETS["explore"]["top_n"])),
        "coupling_threshold": int(
            st.session_state.get("adv_coupling_threshold", ADV_PRESETS["explore"]["coupling_threshold"])
        ),
        "enable_two_hop": bool(st.session_state.get("adv_enable_two_hop", ADV_PRESETS["explore"]["enable_two_hop"])),
        "concept_threshold": float(
            st.session_state.get("adv_concept_threshold", ADV_PRESETS["explore"]["concept_threshold"])
        ),
        "relation_topic_weight": float(
            st.session_state.get("adv_relation_topic_weight", ADV_PRESETS["explore"]["relation_topic_weight"])
        ),
        "foundational_boost": float(
            st.session_state.get("adv_foundational_boost", ADV_PRESETS["explore"]["foundational_boost"])
        ),
        "allowed_domains": list(st.session_state.get("adv_allowed_domains", ADV_PRESETS["explore"]["allowed_domains"])),
    }


def _backfill_relevance_norms(graph_json: dict[str, Any], recommendations: list[dict[str, Any]]) -> None:
    """For older payloads without relevance_norm (e.g. saved projects)."""
    nodes = graph_json.get("nodes") or []
    if not nodes or "relevance_norm" in nodes[0]:
        return
    edges = graph_json.get("edges") or []
    out_deg = Counter(e["source"] for e in edges)
    in_deg = Counter(e["target"] for e in edges)
    rec_map = {r.get("paper_id"): float(r.get("score", 0.0)) for r in recommendations}
    use_rel = any("relation_expand_score" in n for n in nodes)
    raw_scores: list[float] = []
    for node in nodes:
        pid = node["id"]
        cc = int(out_deg[pid] + in_deg[pid])
        node["citation_count"] = cc
        rec = rec_map.get(pid, 0.0)
        ml = float(node.get("missing_link_score") or 0)
        if use_rel:
            rel = float(node.get("relation_expand_score", 0.0))
            raw = cc * 0.35 + rec * 1.2 + ml * 2.0 + rel * 4.0
        else:
            raw = cc * 0.5 + rec * 1.5 + ml * 2.0
        node["relevance_raw"] = raw
        raw_scores.append(raw)
    mx = max(raw_scores) if raw_scores else 1.0
    if mx <= 0:
        mx = 1.0
    for node in nodes:
        node["relevance_norm"] = float(node.get("relevance_raw", 0.0)) / mx


def _norm_doi_sidebar(doi: str | None) -> str:
    if not doi:
        return ""
    return (
        doi.strip()
        .lower()
        .replace("https://doi.org/", "")
        .replace("http://doi.org/", "")
        .replace("doi:", "")
    )


def _paper_needs_metadata_refresh(paper: dict[str, Any]) -> bool:
    t = paper.get("title") or ""
    if "Fallback record" in t or t.startswith("Pending metadata"):
        return True
    for p in paper.get("provenance") or []:
        if isinstance(p, dict) and p.get("source") in ("fallback", "pending"):
            return True
    return False


def _pdf_metadata_badge(paper: dict[str, Any], payload: dict[str, Any]) -> str | None:
    hints = payload.get("pdf_metadata_hints") or []
    if not hints:
        return None
    doi_n = _norm_doi_sidebar(paper.get("doi"))
    oid = (paper.get("openalex_id") or "").strip().lower()
    aid = (paper.get("arxiv_id") or "").strip().lower()
    cid = (paper.get("id") or "").strip().lower()
    for h in hints:
        prov = (h.get("provider") or "").lower()
        hid = (h.get("id") or "").strip().lower()
        if not hid:
            continue
        if prov in ("doi", "crossref") and doi_n and doi_n == _norm_doi_sidebar(hid):
            return str(h.get("badge") or "")
        if prov == "openalex" and oid and hid == oid:
            return str(h.get("badge") or "")
        if prov == "arxiv" and aid and hid == aid:
            return str(h.get("badge") or "")
        if prov == "openalex" and cid.endswith(hid):
            return str(h.get("badge") or "")
    return None


def _filter_graph_for_display(
    payload: dict[str, Any],
    threshold: float,
    *,
    hide_weak_edges: bool = True,
) -> dict[str, Any]:
    """Hide nodes below relevance_norm threshold (PDF seeds always kept)."""
    nodes = [dict(n) for n in (payload.get("graph") or {}).get("nodes", [])]
    edges = [dict(e) for e in (payload.get("graph") or {}).get("edges", [])]
    gj = {"nodes": nodes, "edges": edges}
    recs = (payload.get("discovery") or {}).get("recommendations") or []
    _backfill_relevance_norms(gj, recs)
    expand_active = bool(payload.get("expand_relation_filter_active"))
    topn = int(payload.get("expand_display_top_n", 50))
    visible: set[str] = set()
    for n in nodes:
        pid = n["id"]
        if n.get("graph_noise") and n.get("seed_origin") != "uploaded_pdf" and n.get("source_label") != "PDF":
            continue
        if n.get("seed_origin") == "uploaded_pdf" or n.get("source_label") == "PDF":
            visible.add(pid)
            continue
        rel_ok = float(n.get("relevance_diverse_norm", n.get("relevance_norm", 0.0))) >= threshold - 1e-9
        if not rel_ok:
            continue
        if n.get("is_missing_link_candidate") or n.get("is_foundational_hub"):
            visible.add(pid)
            continue
        if expand_active and n.get("seed_origin") == "discovered":
            if int(n.get("expand_rank", 9999)) <= topn:
                visible.add(pid)
        else:
            visible.add(pid)
    fn = [n for n in nodes if n["id"] in visible]
    ids = {n["id"] for n in fn}
    fe = [e for e in edges if e["source"] in ids and e["target"] in ids]
    if hide_weak_edges:
        fe = [
            e
            for e in fe
            if not (
                (e.get("confidence") == "low")
                or (str((e.get("context_source") or "")).strip().lower() == "abstract_proxy")
            )
        ]
    return {"nodes": fn, "edges": fe}


def _uploaded_pdf_titles_citing(payload: dict[str, Any], target_id: str) -> list[str]:
    titles: list[str] = []
    uploaded = {
        n["id"]
        for n in (payload.get("papers") or [])
        if n.get("seed_origin") == "uploaded_pdf" or n.get("source_label") == "PDF"
    }
    pmap = _paper_map(payload.get("papers") or [])
    for e in (payload.get("graph") or {}).get("edges", []):
        if e.get("target") == target_id and e.get("source") in uploaded:
            p = pmap.get(e["source"])
            titles.append((p.get("title") if p else None) or e["source"])
    return titles


def _node_from_payload(node: dict[str, Any]) -> Node:
    label = (node["title"][:40] + "...") if len(node["title"]) > 40 else node["title"]
    sz = int(
        node.get("viz_size")
        or (
            44
            if node.get("is_foundational_hub")
            else (
                30
                if node.get("is_missing_link_candidate")
                else (34 if node.get("source_label") == "PDF" or node.get("seed_origin") == "uploaded_pdf" else 18)
            )
        )
    )
    col = node.get("viz_color") or (
        "#D4AF37"
        if node.get("is_foundational_hub")
        else (
            "#CA8A04"
            if node.get("is_missing_link_candidate")
            else (
                "#1D4ED8"
                if node.get("source_label") == "PDF" or node.get("seed_origin") == "uploaded_pdf"
                else "#D4D4D8"
            )
        )
    )
    grp = (
        "foundational"
        if node.get("is_foundational_hub")
        else (
            "missing_link"
            if node.get("is_missing_link_candidate")
            else ("uploaded" if node.get("seed_origin") == "uploaded_pdf" else "discovered")
        )
    )
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
        title=f"{node['title']} (Source: {node.get('source_label', 'API')})",
        group=grp,
        **pos,
    )


def _to_agraph_payload(graph_json: dict[str, Any]) -> tuple[list[Node], list[Edge]]:
    nodes = [_node_from_payload(node) for node in graph_json["nodes"]]
    edges = []
    for edge in graph_json["edges"]:
        style = edge["style"]
        edges.append(
            Edge(
                source=edge["source"],
                target=edge["target"],
                label=edge["confidence"],
                type="DASHED" if style["stroke"] == "dashed" else "DYNAMIC",
                color="#9CA3AF" if edge["confidence"] == "low" else ("#60A5FA" if edge["confidence"] == "medium" else "#1D4ED8"),
            )
        )
    return nodes, edges


def _paper_map(papers: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {p["id"]: p for p in papers}


def _importance_score_for_ui(paper: dict[str, Any]) -> float:
    rel = float(paper.get("relevance_diverse_norm", paper.get("relevance_norm", 0.0)) or 0.0)
    cc = float(paper.get("citation_count", 0) or 0.0)
    missing = float(paper.get("missing_link_score", 0) or 0.0)
    return rel * 2.0 + cc * 0.1 + missing * 0.25


def _is_probable_main_dpo_paper(paper: dict[str, Any]) -> bool:
    pid = str(paper.get("id") or "").lower()
    title = str(paper.get("title") or "").lower()
    return ("2305.18290" in pid) or (
        "direct preference optimization" in title and "secretly a reward model" in title
    )


def _related_edges(graph_json: dict[str, Any], node_id: str) -> list[dict[str, Any]]:
    return [e for e in graph_json["edges"] if e["source"] == node_id or e["target"] == node_id]


def _readable_edge_confidence(level: str | None) -> str:
    lv = (level or "").strip().lower()
    if lv == "high":
        return "High evidence"
    if lv == "medium":
        return "Moderate evidence"
    if lv == "low":
        return "Exploratory evidence"
    return "Unknown evidence"


def _apply_project_load(service: SRGApplicationService, store: ProjectStore, stem: str) -> None:
    session = store.load_stem(stem)
    events = feedback_events_from_jsonable(session.get("feedback_events") or [])
    service.feedback_store.replace_all(events)
    st.session_state.payload = session.get("payload")
    st.session_state.pdf_previews = list(session.get("pdf_previews") or [])
    for k in list(st.session_state.keys()):
        if k.startswith("pdf_include_") or k.startswith("pdf_provider_") or k.startswith("pdf_id_") or k.startswith("pdf_override_"):
            del st.session_state[k]
    for i_str, row in (session.get("pdf_ui") or {}).items():
        try:
            i = int(i_str)
        except ValueError:
            continue
        st.session_state[f"pdf_include_{i}"] = bool(row.get("include", False))
        st.session_state[f"pdf_provider_{i}"] = str(row.get("provider", "openalex"))
        st.session_state[f"pdf_id_{i}"] = str(row.get("id", ""))
        st.session_state[f"pdf_override_{i}"] = bool(row.get("override", False))
    st.session_state.graph_mode = session.get("graph_mode", "local")
    st.session_state.local_seed_count = int(session.get("local_seed_count", 0))
    st.session_state.selected_node = session.get("selected_node")
    st.session_state["discipline_choice_idx"] = int(session.get("discipline_choice_idx", 0))
    st.session_state._restored_session = True
    blobs = session.get("pdf_blobs") or {}
    if blobs:
        import base64

        st.session_state._pdf_files_for_save = [
            {"name": name, "content": base64.standard_b64decode(b64)}
            for name, b64 in blobs.items()
            if isinstance(b64, str)
        ]
    else:
        st.session_state._pdf_files_for_save = []
    deep = bool(session.get("deep_parse", False))
    st.session_state["deep_parse_toggle"] = deep
    names = "|".join((p.get("filename") or "") for p in st.session_state.pdf_previews)
    st.session_state.pdf_signature = f"{RESTORED_PDF_SIGNATURE}|{deep}|{names}"


def run_streamlit_ui() -> None:
    st.set_page_config(page_title="Semantic Research Graph", layout="wide")
    st.title("Semantic Research Graph")

    if "service" not in st.session_state:
        st.session_state.service = SRGApplicationService()
    if "payload" not in st.session_state:
        st.session_state.payload = None
    if "pdf_previews" not in st.session_state:
        st.session_state.pdf_previews = []
    if "pdf_signature" not in st.session_state:
        st.session_state.pdf_signature = ""
    if "selected_node" not in st.session_state:
        st.session_state.selected_node = None
    if "graph_mode" not in st.session_state:
        st.session_state.graph_mode = "local"
    if "local_seed_count" not in st.session_state:
        st.session_state.local_seed_count = 0
    if "project_store" not in st.session_state:
        pdir = os.environ.get("SRG_PROJECTS_DIR")
        st.session_state.project_store = ProjectStore(pdir) if pdir else ProjectStore()
    if "_restored_session" not in st.session_state:
        st.session_state._restored_session = False
    if "_pdf_files_for_save" not in st.session_state:
        st.session_state._pdf_files_for_save = []
    if "_arxiv_sidebar_cache" not in st.session_state:
        st.session_state._arxiv_sidebar_cache = {}
    if "ui_action_mode" not in st.session_state:
        st.session_state.ui_action_mode = ""
    if "show_delete_drawer" not in st.session_state:
        st.session_state.show_delete_drawer = False
    if "show_timeline_drawer" not in st.session_state:
        st.session_state.show_timeline_drawer = False
    if "adv_preset" not in st.session_state:
        _apply_adv_preset_to_state("explore")
    service: SRGApplicationService = st.session_state.service
    project_store: ProjectStore = st.session_state.project_store

    st.caption("Projects")
    c_save, c_upload, c_del = st.columns([1, 1, 1])
    with c_save:
        if st.button("💾", key="icon_save_project", help="Save project"):
            st.session_state.ui_action_mode = (
                "" if st.session_state.ui_action_mode == "save" else "save"
            )
            st.session_state.show_delete_drawer = False
            st.session_state.show_timeline_drawer = False
    with c_upload:
        if st.button("⤴️", key="icon_upload_project", help="Open saved project"):
            st.session_state.ui_action_mode = (
                "" if st.session_state.ui_action_mode == "load" else "load"
            )
            st.session_state.show_delete_drawer = False
            st.session_state.show_timeline_drawer = False
    with c_del:
        if st.button("🗑️", key="icon_delete_project", help="Delete project"):
            st.session_state.show_delete_drawer = not st.session_state.show_delete_drawer
            if st.session_state.show_delete_drawer:
                st.session_state.ui_action_mode = ""
                st.session_state.show_timeline_drawer = False

    if st.session_state.ui_action_mode == "save":
        with st.container(border=True):
            st.subheader("Save Project")
            st.text_input("Name", key="project_save_name_input", placeholder="my_study")
            if st.button("Save", key="btn_save_project"):
                name = (st.session_state.get("project_save_name_input") or "").strip() or "untitled"
                deep_parse_sv = bool(st.session_state.get("deep_parse_toggle", False))
                pdf_ui: dict[str, dict[str, Any]] = {}
                for i in range(len(st.session_state.pdf_previews or [])):
                    pdf_ui[str(i)] = {
                        "include": bool(st.session_state.get(f"pdf_include_{i}", False)),
                        "provider": str(st.session_state.get(f"pdf_provider_{i}", "openalex")),
                        "id": str(st.session_state.get(f"pdf_id_{i}", "")),
                        "override": bool(st.session_state.get(f"pdf_override_{i}", False)),
                    }
                raw_files = st.session_state.get("_pdf_files_for_save") or []
                blobs, blob_warnings = collect_pdf_blobs(raw_files) if raw_files else ({}, [])
                snapshot: dict[str, Any] = {
                    "payload": st.session_state.payload,
                    "pdf_previews": list(st.session_state.pdf_previews or []),
                    "pdf_ui": pdf_ui,
                    "pdf_blobs": blobs,
                    "pdf_signature": st.session_state.get("pdf_signature", ""),
                    "deep_parse": deep_parse_sv,
                    "graph_mode": st.session_state.get("graph_mode", "local"),
                    "local_seed_count": int(st.session_state.get("local_seed_count", 0)),
                    "selected_node": st.session_state.get("selected_node"),
                    "discipline_choice_idx": int(st.session_state.get("discipline_choice_idx", 0)),
                    "feedback_events": feedback_events_to_jsonable(service.feedback_store.list_all()),
                }
                try:
                    path = project_store.save(name, snapshot)
                    st.success(f"Saved to `{path}`")
                    for w in blob_warnings:
                        st.warning(w)
                except OSError as exc:
                    st.error(str(exc))

    if st.session_state.ui_action_mode == "load":
        with st.container(border=True):
            st.subheader("Open Saved Project")
            stems = project_store.list_project_stems()
            pick = st.selectbox(
                "Saved projects",
                [""] + stems,
                format_func=lambda s: project_store.project_display_name(s) if s else "— select —",
                key="project_load_pick",
            )
            if st.button("Load", key="btn_load_project") and pick:
                try:
                    _apply_project_load(service, project_store, pick)
                    st.success("Project loaded.")
                    st.rerun()
                except (OSError, ValueError, FileNotFoundError) as exc:
                    st.error(str(exc))

    if st.session_state.show_delete_drawer:
        with st.sidebar:
            st.subheader("Delete Project")
            stems_del = project_store.list_project_stems()
            del_pick = st.selectbox(
                "Choose project",
                [""] + stems_del,
                format_func=lambda s: project_store.project_display_name(s) if s else "— select —",
                key="project_delete_pick",
            )
            c_dx1, c_dx2 = st.columns(2)
            with c_dx1:
                if st.button("Cancel", key="btn_delete_cancel"):
                    st.session_state.show_delete_drawer = False
                    st.rerun()
            with c_dx2:
                if st.button("Delete", key="btn_delete_project") and del_pick:
                    project_store.delete_stem(del_pick)
                    st.session_state.show_delete_drawer = False
                    st.success("Deleted.")
                    st.rerun()

    # Product default: do not ask users for discipline/domain upfront.
    discipline = None
    discovery_opts = _current_discovery_options_from_state()

    tab_library, tab_discovery = st.tabs(["Library Synthesis Mode", "Discovery Mode"])

    with tab_library:
        st.subheader("PDF-First Ingestion")
        deep_parse = st.toggle(
            "Deep Parse bibliography",
            value=False,
            key="deep_parse_toggle",
            help="Enable this to extract more references from bibliography pages for broader discovery.",
        )
        uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
        if uploaded_files:
            st.session_state._pdf_files_for_save = [{"name": f.name, "content": f.getvalue()} for f in uploaded_files]
            st.session_state._restored_session = False
        signature = f"{deep_parse}|" + "|".join(f"{f.name}:{getattr(f, 'size', 0)}" for f in (uploaded_files or []))
        if signature != st.session_state.pdf_signature:
            restored = str(st.session_state.pdf_signature).startswith(RESTORED_PDF_SIGNATURE)
            if restored and not uploaded_files and st.session_state.get("_restored_session"):
                pass
            else:
                st.session_state.pdf_signature = signature
                if uploaded_files:
                    files = [{"name": f.name, "content": f.getvalue()} for f in uploaded_files]
                    st.session_state.pdf_previews = service.preview_pdf_seeds(files=files, deep_parse=deep_parse)
                else:
                    st.session_state.pdf_previews = []
                for k in list(st.session_state.keys()):
                    if k.startswith("pdf_include_") or k.startswith("pdf_provider_") or k.startswith("pdf_id_") or k.startswith("pdf_override_"):
                        del st.session_state[k]
                st.session_state._restored_session = False

        previews = st.session_state.pdf_previews
        if previews:
            st.write("Detected mappings (please confirm):")
            for i, item in enumerate(previews):
                include_key = f"pdf_include_{i}"
                provider_key = f"pdf_provider_{i}"
                id_key = f"pdf_id_{i}"
                override_key = f"pdf_override_{i}"
                if include_key not in st.session_state:
                    ms = float(item.get("mapping_score", 1.0))
                    gated = item.get("pdf_gate_blocked") or ms < PDF_MAPPING_QUALITY_THRESHOLD
                    st.session_state[include_key] = (not item.get("unresolved", False)) and not gated
                if provider_key not in st.session_state:
                    st.session_state[provider_key] = item.get("provider", "openalex")
                if id_key not in st.session_state:
                    st.session_state[id_key] = item.get("id", "")
                if override_key not in st.session_state:
                    st.session_state[override_key] = False
                st.markdown(f"**{item.get('filename')}**")
                st.caption(f"Detected title: {item.get('title_guess') or '-'}")
                st.caption(f"Detected authors: {item.get('authors_guess') or '-'}")
                if item.get("matched_title"):
                    st.caption(f"Matched title: {item['matched_title']}")
                ms = float(item.get("mapping_score", 0.0))
                st.caption(f"PDF mapping score: **{ms:.2f}** (gate ≥ {PDF_MAPPING_QUALITY_THRESHOLD:.2f})")
                if item.get("pdf_gate_blocked") or ms < PDF_MAPPING_QUALITY_THRESHOLD:
                    st.warning(
                        "Low-confidence PDF→metadata match. Excluded from graph unless you check "
                        "'Override quality gate' below (override is logged)."
                    )
                if item.get("warning"):
                    st.warning(item["warning"])
                if item.get("error"):
                    st.error(item["error"])
                c1, c2, c3 = st.columns([1, 1, 2])
                with c1:
                    st.checkbox("Include", key=include_key)
                    st.checkbox("Override quality gate (logged)", key=override_key)
                with c2:
                    st.selectbox("Provider", ["openalex", "arxiv", "doi", "crossref"], key=provider_key)
                with c3:
                    st.text_input("Resolved ID", key=id_key)

            reviewed = []
            for i, item in enumerate(previews):
                if not st.session_state.get(f"pdf_include_{i}", False):
                    continue
                reviewed.append(
                    {
                        **item,
                        "provider": st.session_state.get(f"pdf_provider_{i}", item.get("provider", "openalex")),
                        "id": st.session_state.get(f"pdf_id_{i}", item.get("id", "")),
                        "quality_override": bool(st.session_state.get(f"pdf_override_{i}", False)),
                    }
                )
            c_local, c_expand = st.columns(2)
            with c_local:
                if st.button("Build Local PDF Graph", use_container_width=True):
                    seeds = service.build_seeds_from_pdf_previews(reviewed, deep_parse=False)
                    st.session_state.graph_mode = "local"
                    st.session_state.local_seed_count = len(
                        [s for s in seeds if s.get("origin") == "uploaded_pdf"]
                    )
                    with st.spinner("Building a quick foundational graph from your PDF library..."):
                        st.session_state.payload = service.run_pipeline(
                            seeds, discipline=discipline, discovery_options=discovery_opts
                        )
                    st.session_state.selected_node = None
            with c_expand:
                if st.button("Expand (Missing Links)", use_container_width=True):
                    seeds = service.build_seeds_from_pdf_previews(reviewed, deep_parse=True)
                    st.session_state.graph_mode = "expanded"
                    st.session_state.local_seed_count = len(
                        [s for s in seeds if s.get("origin") == "uploaded_pdf"]
                    )
                    # In Library mode, always keep expansion gears on even if user last chose Scan preset.
                    expand_opts = {
                        **_current_discovery_options_from_state(),
                        "enable_two_hop": True,
                        "top_n": max(50, int(st.session_state.get("adv_top_n", 50))),
                    }
                    with st.spinner("Deepening the literature graph from references and hubs..."):
                        st.session_state.payload = service.run_pipeline(
                            seeds, discipline=discipline, discovery_options=expand_opts
                        )
                    st.session_state.selected_node = None
        else:
            st.info("Upload PDFs to start library synthesis.")

    with tab_discovery:
        st.subheader("Search-First Discovery")
        q_col, b_col = st.columns([6, 1])
        with q_col:
            q = st.text_input(
                "Keyword / DOI / Title / arXiv",
                help="Use a specific phrase (3+ chars) for better relevance, e.g. 'privacy preserving NLP'.",
            )
        # Product default: balanced exploration behavior with arXiv as plain-text backend.
        if st.session_state.get("adv_preset") != "explore":
            _apply_adv_preset_to_state("explore")
        text_backend = "arxiv"
        with b_col:
            do_discover = st.button("Discover", key="btn_discover_main")
        if do_discover:
            qq = (q or "").strip()
            if len(qq) < 3 and not re.search(r"\b10\.\d{4,9}/", qq, flags=re.IGNORECASE):
                st.info("Try a more specific query (at least 3 characters), e.g. 'PII in NLP pipelines'.")
                return
            with st.spinner("Searching literature and building candidate seeds..."):
                seeds = service.discover_from_query(q, text_search_backend=text_backend)
            if not seeds:
                st.warning("No candidate found.")
            else:
                st.session_state.graph_mode = "expanded"
                st.session_state.local_seed_count = 0
                with st.spinner("Building graph and linking foundational papers..."):
                    st.session_state.payload = service.run_pipeline(
                        seeds,
                        discipline=discipline,
                        discovery_options=_current_discovery_options_from_state(),
                    )
                st.session_state.selected_node = None

    payload = st.session_state.payload
    if not payload:
        return

    graph_id = str(payload.get("generated_at") or "")
    if st.session_state.get("_arxiv_cache_graph_id") != graph_id:
        st.session_state._arxiv_sidebar_cache = {}
        st.session_state._arxiv_cache_graph_id = graph_id

    ex1, ex2 = st.columns(2)
    with ex1:
        st.download_button(
            label="Download research report (Markdown)",
            data=build_markdown_report(payload, project_title="SRG Research Report"),
            file_name="srg_research_report.md",
            mime="text/markdown",
            key="dl_md_report",
        )
    with ex2:
        st.download_button(
            label="Download BibTeX library",
            data=build_bibtex(payload),
            file_name="srg_library.bib",
            mime="text/plain",
            key="dl_bibtex",
        )

    source_summary = payload.get("source_summary", {})
    if source_summary:
        st.caption("Source summary: " + ", ".join(f"{k}={v}" for k, v in source_summary.items()))
    eq = (payload.get("evidence_quality") or "").strip().lower()
    est = payload.get("evidence_stats") or {}
    if eq:
        if eq == "high":
            st.success(
                f"Evidence quality: HIGH (references={int(est.get('references_total', 0))}, edges={int(est.get('graph_edges_total', 0))})"
            )
        elif eq == "medium":
            st.info(
                f"Evidence quality: MEDIUM (references={int(est.get('references_total', 0))}, edges={int(est.get('graph_edges_total', 0))})"
            )
        else:
            st.warning(
                f"Evidence quality: LOW / metadata-only (references={int(est.get('references_total', 0))}, edges={int(est.get('graph_edges_total', 0))})"
            )
    total_nodes = len(payload.get("papers", []))
    mode = st.session_state.graph_mode
    if mode == "expanded":
        added = int(payload.get("expanded_added_count", 0))
        st.info(f"🌐 Mode: Global Discovery (Expanded) | Added papers: {added}")
        if added == 0:
            st.caption(
                "No external papers were added. This usually means no resolvable reference IDs "
                "were extracted from uploaded PDFs yet."
            )
    else:
        st.info("📍 Mode: Local Library Only")
    kpi_new = int((payload.get("expand_diagnostics") or {}).get("new_nodes_added", 0) or 0)
    kpi_res = int((payload.get("expand_diagnostics") or {}).get("references_resolved", 0) or 0)
    st.info(
        f"🔍 Scanning Status: {kpi_new} nodes added | Connectivity boost active (Resolved Refs: {kpi_res})"
    )
    if payload.get("auto_escalation_message"):
        st.info(str(payload.get("auto_escalation_message")))
    diagnostics = payload.get("diagnostics", [])
    ing = payload.get("ingestion_stats") or {}
    pq = payload.get("pdf_quality_gate") or {}
    expand_diag = payload.get("expand_diagnostics", {})
    if diagnostics or ing or pq or expand_diag or payload.get("discipline_applied"):
        with st.expander("🔍 System Stats & Logs", expanded=False):
            if payload.get("discipline_applied"):
                st.caption(f"Discipline prior applied to undecided papers: **{payload['discipline_applied']}**")
            if diagnostics:
                st.markdown("**Pipeline diagnostics**")
                for item in diagnostics:
                    st.write(f"- {item}")
            if ing:
                st.markdown("**Ingestion / rate-limit stats**")
                st.json(ing)
            if pq:
                st.markdown("**PDF quality gate**")
                st.json(pq)
            if expand_diag:
                st.markdown("**Expand diagnostics**")
                st.write(f"- references_extracted: {expand_diag.get('references_extracted', 0)}")
                st.write(f"- references_resolved: {expand_diag.get('references_resolved', 0)}")
                st.write(f"- references_skipped: {expand_diag.get('references_skipped', 0)}")
                sk = expand_diag.get("openalex_skipped_ids") or []
                if sk:
                    preview = ", ".join(str(x) for x in sk[:25])
                    st.write(f"- openalex_skipped_ids ({len(sk)}): {preview}{' …' if len(sk) > 25 else ''}")
                st.write(f"- ordered_pool_size: {expand_diag.get('ordered_pool_size', 0)}")
                st.write(f"- expand_candidate_pool_max: {expand_diag.get('expand_candidate_pool_max', '—')}")
                st.write(f"- expand_display_top_n: {expand_diag.get('expand_display_top_n', '—')}")
                st.write(f"- new_nodes_added: {expand_diag.get('new_nodes_added', 0)}")
                notes = expand_diag.get("pool_fill_notes") or []
                if notes:
                    st.markdown("**Pool / reference diagnostics**")
                    for line in notes:
                        st.caption(line)
    missing_links = payload.get("missing_link_candidates", [])
    if missing_links:
        top = sorted(
            missing_links,
            key=lambda m: (
                float(m.get("pii_concept_match", 0.0)),
                int(m.get("openalex_cited_by_count", 0)),
                int(m.get("co_cited_by_uploaded_count", 0)),
            ),
            reverse=True,
        )
        st.success(
            "Foundational / missing-link candidates (PII-ranked): "
            + ", ".join(
                f"{item['paper_id']} (PII≈{item.get('pii_concept_match', '—')}, "
                f"cited_by={item.get('openalex_cited_by_count', '—')}, "
                f"uploads→{item['co_cited_by_uploaded_count']})"
                for item in top[:5]
            )
        )

    rel_thr = 0.0
    hide_weak = True
    graph_full = payload["graph"]
    graph_display = _filter_graph_for_display(payload, float(rel_thr), hide_weak_edges=bool(hide_weak))
    n_show = len(graph_display["nodes"])
    n_tot = len(graph_full.get("nodes") or [])
    st.caption(f"Showing **{n_show}** / {n_tot} nodes (filtered view). Full graph is preserved for details.")
    if payload.get("expand_relation_filter_active"):
        st.caption(
            f"Expand mode: up to **{int(payload.get('expand_display_top_n', 50))}** highest relation-score "
            "discovered papers are eligible for the graph view (see Expand diagnostics)."
        )

    nodes, edges = _to_agraph_payload(graph_display)
    config = Config(
        width="100%",
        height=760,
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
    col_main, col_side = st.columns([4.2, 1.2])
    with col_main:
        selected = agraph(nodes=nodes, edges=edges, config=config)
        if selected:
            st.session_state.selected_node = selected
        st.markdown(
            "**Legend:** **Blue, larger** — Local (PDF upload) · **Light grey, smaller** — API / discovered · "
            "**Bright gold, largest** — Foundational hub (co-cited by ≥3 uploads, top-cited 2-hop, or strong PII concept match) · "
            "**Amber** — Missing-link candidate (co-cited by several local PDFs). "
            "**Edges:** solid blue tones = stronger citation context; dashed = weaker / potential."
        )
        ranked_now = sorted(payload.get("papers") or [], key=_importance_score_for_ui, reverse=True)
        if ranked_now:
            st.markdown("### Most important papers in current graph")
            for i, p in enumerate(ranked_now[:10], start=1):
                marker = " ★ Main DPO" if _is_probable_main_dpo_paper(p) else ""
                reasons = ", ".join((p.get("inclusion_reasons") or [])[:3])
                st.write(
                    f"{i}. {p.get('title', '')} (`{p.get('id')}`) "
                    f"— importance={_importance_score_for_ui(p):.2f}, links={int(p.get('citation_count', 0) or 0)}, "
                    f"why={reasons}{marker}"
                )
            if not any(_is_probable_main_dpo_paper(p) for p in ranked_now):
                st.info(
                    "Canonical DPO paper is not in this graph slice yet. Try Discover again (arXiv backend), "
                    "then use Explore/Dig settings so expansion can connect it."
                )
    with col_side:
        st.subheader("Insights")
        st.caption("Basic view keeps extra charts hidden.")
        trends = (payload.get("discovery") or {}).get("trends") or {}
        timeline = trends.get("timeline") or {}
        if st.button("🕒 Timeline", key="btn_timeline_drawer", help="Open publication timeline panel"):
            st.session_state.show_timeline_drawer = not st.session_state.show_timeline_drawer
            st.session_state.show_delete_drawer = False
        if not timeline:
            st.caption("No trend data yet.")
    if st.session_state.show_timeline_drawer:
        with st.sidebar:
            st.subheader("Publication Timeline")
            tl = {str(k): int(v) for k, v in sorted((timeline or {}).items(), key=lambda x: str(x[0]))}
            if tl:
                st.bar_chart(tl)
            else:
                st.caption("No timeline data.")
            if st.button("Close", key="btn_timeline_close"):
                st.session_state.show_timeline_drawer = False
                st.rerun()
    found_hubs = [n for n in (payload.get("papers") or []) if n.get("is_foundational_hub")]
    if not found_hubs:
        st.warning(
            "No foundational (gold) nodes yet. This usually means the current graph lacks shared-reference hubs "
            "or the corpus is still too sparse."
        )

    selected = st.session_state.selected_node
    if not selected:
        return
    papers = _paper_map(payload["papers"])
    paper = papers.get(selected)
    if not paper:
        return
    st.sidebar.header("Selected Paper")
    with st.sidebar.expander("Why this paper appears in your map", expanded=True):
        if paper.get("is_foundational_hub"):
            st.markdown(
                "**Foundational hub** — this paper is either cited by multiple uploaded PDFs, "
                "or it is among the highly cited references of papers cited by your library; "
                f"OpenAlex concept match ≈{paper.get('pii_concept_match', '—')}."
            )
        elif paper.get("is_missing_link_candidate"):
            citers = _uploaded_pdf_titles_citing(payload, selected)
            if citers:
                st.markdown("**Missing-link candidate** — your uploaded PDFs cite this work:")
                for t in citers:
                    st.markdown(f"- {t}")
            else:
                st.markdown(
                    "**Missing-link candidate** — flagged by co-citation from uploads; "
                    "no direct uploaded→this edge in the current graph slice."
                )
        elif paper.get("seed_origin") == "uploaded_pdf" or paper.get("source_label") == "PDF":
            st.markdown("**Local library** — mapped from one of your uploaded PDFs.")
        else:
            st.markdown(
                "**Discovered** — reached via references, expansion, or API seeds (not a direct PDF upload row)."
            )
        if paper.get("seed_origin") == "discovered" and (paper.get("expand_explanation") or "").strip():
            st.markdown(f"**Expansion rationale:** {paper['expand_explanation']}")
            rk = paper.get("expand_rank")
            rs = paper.get("relation_expand_score")
            if rk is not None and rs is not None:
                st.caption(f"Relation rank: #{int(rk)} · ACCE-style score: {float(rs):.3f}")
        cc = paper.get("citation_count")
        rn = paper.get("relevance_norm")
        if cc is not None and rn is not None:
            st.caption(
                f"Connectivity in this graph: {cc} links · Relevance score: {rn:.2f} "
                "(higher means more central to your current literature map)."
            )
        elif cc is not None:
            st.caption(f"Connectivity in this graph: {cc} links")
    with st.sidebar.expander("Paper details", expanded=True):
        st.write(f"**ID:** {paper['id']}")
        st.write(f"**Title:** {paper['title']}")
        st.write(f"**Source:** {paper.get('source_label', 'API')}")
    if _paper_needs_metadata_refresh(paper):
        if st.button("Fetch metadata (OpenAlex)", key=f"hydrate_meta_{selected}"):
            try:
                if service.hydrate_graph_node_metadata(st.session_state.payload, selected):
                    st.success("Metadata refreshed from OpenAlex.")
                    st.rerun()
                else:
                    st.warning("Could not refresh (need OpenAlex W… id or network error).")
            except Exception as exc:  # pragma: no cover
                st.error(str(exc))
    badge = _pdf_metadata_badge(paper, payload)
    with st.sidebar.expander("Metadata quality", expanded=False):
        if badge and (paper.get("seed_origin") == "uploaded_pdf" or paper.get("source_label") == "PDF"):
            st.caption(f"Metadata source hint: **{badge}**")
        st.write(f"**DOI:** {paper.get('doi') or '-'}")
        if paper.get("year"):
            st.caption(f"Year (from graph): {paper['year']}")
    abs_graph = paper.get("abstract")
    if abs_graph:
        with st.sidebar.expander("Abstract summary (graph sources)", expanded=False):
            st.write(abs_graph[:4000] + ("…" if len(abs_graph or "") > 4000 else ""))

    arxiv_lookup = resolve_arxiv_id_for_lookup(paper)
    if arxiv_lookup:
        cache: dict[str, Any] = st.session_state._arxiv_sidebar_cache
        if selected not in cache:
            cache[selected] = fetch_arxiv_public_metadata(arxiv_lookup)
        ax = cache.get(selected)
        if ax:
            with st.sidebar.expander("arXiv record (export.arxiv.org)", expanded=False):
                st.caption(f"arXiv:{ax.get('id', arxiv_lookup)}")
                if ax.get("published"):
                    st.write(f"**Submitted (arXiv):** {ax['published'][:10]}")
                if ax.get("updated") and ax.get("updated") != ax.get("published"):
                    st.write(f"**Last updated:** {ax['updated'][:10]}")
                auth = ax.get("authors") or []
                if auth:
                    st.write("**Authors:** " + ", ".join(auth[:40]) + (" …" if len(auth) > 40 else ""))
                summ = ax.get("summary") or ""
                if summ:
                    st.write("**Abstract (arXiv):**")
                    st.write(summ[:6000] + ("…" if len(summ) > 6000 else ""))
                if ax.get("doi"):
                    st.caption(f"arXiv-record DOI: {ax['doi']}")
        else:
            st.sidebar.caption(f"Could not load arXiv:{arxiv_lookup} (network or id).")
    else:
        st.sidebar.caption(
            "No arXiv id on this node (no `arxiv_id`, arXiv DOI, or `arxiv:` canonical id). "
            "OpenAlex/Crossref-only papers without an arXiv link cannot be queried on export.arxiv.org."
        )

    rel_edges = _related_edges(payload["graph"], selected)
    with st.sidebar.expander("Citation context and evidence", expanded=True):
        if not rel_edges:
            st.caption("No citation links are currently attached to this paper.")
        for rel in rel_edges[:20]:
            prov = rel.get("context_source") or "—"
            ctx = rel.get("context") or "No snippet available."
            conf = _readable_edge_confidence(rel.get("confidence"))
            st.markdown(
                f"**{rel['source']} → {rel['target']}**  \n"
                f"- Evidence level: {conf}  \n"
                f"- Context snippet: {ctx}  \n"
                f"- Source of evidence: `{prov}`"
            )

    with st.sidebar.expander("Manual feedback (merge / wrong citation)", expanded=False):
        st.caption("Use only when two records are the same work or an edge is wrong.")
        neighbor_choices: list[tuple[str, str]] = []
        for e in rel_edges:
            other = e["target"] if e["source"] == selected else e["source"]
            label = f"{other[:16]}…" if len(other) > 20 else other
            neighbor_choices.append((f"Edge → {label}", f"{e['source']}|{e['target']}"))
        st.text_input("Merge with paper ID (canonical id)", key="merge_hint_target")
        if st.button("Mark as Same (merge hint)", key="btn_merge"):
            tgt = (st.session_state.get("merge_hint_target") or "").strip()
            if tgt:
                try:
                    service.submit_merge_feedback(selected, tgt, accepted=True)
                    st.success("Merge hint saved.")
                except Exception as exc:  # pragma: no cover
                    st.error(str(exc))
            else:
                st.warning("Enter a target paper id.")

        if neighbor_choices:
            edge_ix = st.selectbox(
                "Report wrong citation (edge)",
                list(range(len(neighbor_choices))),
                format_func=lambda i: neighbor_choices[i][0],
                key="bad_edge_ix",
            )
            if st.button("Report Wrong", key="btn_bad"):
                pair = neighbor_choices[int(edge_ix)][1].split("|", maxsplit=1)
                if len(pair) == 2:
                    try:
                        service.submit_bad_citation_feedback(pair[0], pair[1])
                        st.success("Feedback saved.")
                    except Exception as exc:  # pragma: no cover
                        st.error(str(exc))
        else:
            st.caption("No edges to report for this node.")


if __name__ == "__main__":
    run_streamlit_ui()

