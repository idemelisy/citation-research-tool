"""
Topical vs graph importance — SRG Lite precision ranking layer.

Separates citation-graph centrality from query-specific topical centrality.
Used heavily for Start Here; balanced elsewhere.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any

from .intent_coherence import lexical_similarity
from .query_intent_normalization import (
    SubfieldCluster,
    all_clusters_for_entry,
    registry_entry_by_id,
)

_STOPWORDS = frozenset(
    "the a an of for to in on and or with by as at from into over per via using "
    "method methods approach model models learning data based new using study "
    "analysis work results between among".split()
)

# Generic umbrella patterns — penalize only when query specificity is low.
_GENERIC_SURVEY_MARKERS: tuple[str, ...] = (
    "comprehensive review",
    "comprehensive survey",
    "literature review",
    "systematic review",
    "survey of",
    "overview of",
    "state of the art",
    "recent advances in",
    "applications of",
    "applications and",
    "benchmark",
    "dataset",
    "foundation model",
    "large language model",
    "generative ai",
    "llm applications",
    "vulnerabilities",
)

_GENERIC_LLM_WITHOUT_ALIGNMENT: tuple[str, ...] = (
    "large language model",
    "foundation model",
    "generative ai",
    "chatgpt",
    "gpt-4",
    "llm-based",
)

# Soft domain-lock: off-cluster markers when intent registry defines dominant topic.
_DOMAIN_DRIFT_CLUSTERS: dict[str, tuple[SubfieldCluster, ...]] = {
    "direct_preference_optimization": (
        SubfieldCluster("cybersecurity", ("cybersecurity", "cyber security", "malware", "intrusion detection")),
        SubfieldCluster("diffusion_image", ("stable diffusion", "text-to-image", "image generation", "denoising diffusion")),
        SubfieldCluster("biology", ("protein folding", "genomics", "clinical trial", "drug discovery")),
    ),
    "transformer_nlp": (
        SubfieldCluster("power_systems", ("power transformer", "fault diagnosis", "substation", "electrical grid")),
    ),
    "visual_debugging": (
        SubfieldCluster("program_slicing_classic", ("program slicing", "static analysis survey", "formal verification survey")),
        SubfieldCluster("medical_imaging", ("histopathology", "mri segmentation", "clinical imaging")),
    ),
    "graph_neural_networks": (
        SubfieldCluster("social_network_only", ("social network analysis survey", "community detection survey")),
    ),
}

# Flagship phrase boosts keyed by intent registry id.
_FLAGSHIP_MARKERS: dict[str, tuple[str, ...]] = {
    "direct_preference_optimization": (
        "direct preference optimization",
        "dpo",
        "orpo",
        "rrhf",
        "reinforcement learning from human feedback",
        "rlhf",
        "preference optimization",
        "reward model",
        "alignment",
        "without reward model",
    ),
    "transformer_nlp": (
        "attention is all you need",
        "self-attention",
        "transformer",
        "encoder-decoder",
        "multi-head attention",
    ),
    "visual_debugging": (
        "software visualization",
        "program visualization",
        "debugger",
        "debugging",
        "execution trace",
        "devtools",
    ),
    "graph_neural_networks": (
        "graph convolutional",
        "graph attention",
        "message passing",
        "graphsage",
        "graph neural network",
    ),
}


def _blob(node: dict[str, Any]) -> str:
    return f"{node.get('title', '')} {str(node.get('abstract') or '')[:3000]}".lower()


def _query_tokens(query: str, enrichment: list[str] | None = None) -> set[str]:
    parts = [query or ""] + list(enrichment or [])
    out: set[str] = set()
    for p in parts:
        for t in re.split(r"[^\w\-]+", p.lower()):
            if len(t) > 2 and t not in _STOPWORDS:
                out.add(t)
    return out


def query_specificity_score(query: str, enrichment: list[str] | None = None) -> float:
    """How specific the user query is (0=broad, 1=phrase-heavy)."""
    qtok = _query_tokens(query, enrichment)
    if not qtok:
        return 0.3
    ql = (query or "").lower()
    phrase_bonus = 0.0
    if len(ql.split()) >= 3:
        phrase_bonus = 0.25
    if any(x in ql for x in ("optimization", "alignment", "neural", "debugging", "convolutional")):
        phrase_bonus += 0.15
    return min(1.0, len(qtok) / 8.0 + phrase_bonus)


def exact_phrase_overlap(query: str, node: dict[str, Any]) -> float:
    ql = (query or "").strip().lower()
    title = (node.get("title") or "").lower()
    if not ql or not title:
        return 0.0
    if ql in title:
        return 1.0
    # multi-word query fragments
    words = ql.split()
    if len(words) >= 2:
        for n in range(len(words), 1, -1):
            for i in range(len(words) - n + 1):
                frag = " ".join(words[i : i + n])
                if len(frag) > 6 and frag in title:
                    return min(1.0, 0.55 + 0.12 * n)
    return 0.0


def compute_query_centrality(
    query: str,
    paper: dict[str, Any],
    *,
    enrichment_terms: list[str] | None = None,
    intent_entry_id: str | None = None,
    cluster_context: dict[str, Any] | None = None,
) -> float:
    """
    How prototypical is this paper for the user's exact intent? ∈ [0, 1].
    """
    blob = _blob(paper)
    qtok = _query_tokens(query, enrichment_terms)
    if not qtok:
        return 0.45

    title = (paper.get("title") or "").lower()
    title_hits = sum(1 for t in qtok if t in title)
    title_ratio = title_hits / max(len(qtok), 1)
    blob_hits = sum(1 for t in qtok if t in blob)
    blob_ratio = min(1.0, blob_hits / max(len(qtok), 1))
    phrase = exact_phrase_overlap(query, paper)
    lex = lexical_similarity(query, paper)

    boost_c, pen_c, _ = (0.0, 0.0, None)
    if intent_entry_id:
        boosts, penalties = all_clusters_for_entry(intent_entry_id)
        from .intent_coherence import subfield_cluster_scores

        boost_c, pen_c, _ = subfield_cluster_scores(paper, boosts, penalties)

    flagship = flagship_match_strength(paper, intent_entry_id, query)
    sem = float(paper.get("semantic_score", paper.get("semantic_fit_score", 0.0)) or 0.0)

    raw = (
        0.28 * phrase
        + 0.22 * title_ratio
        + 0.16 * blob_ratio
        + 0.14 * lex
        + 0.10 * sem
        + 0.10 * boost_c
    )
    raw += 0.18 * flagship
    if pen_c >= 0.34:
        raw *= max(0.2, 1.0 - 0.75 * pen_c)
    elif pen_c >= 0.2:
        raw *= max(0.5, 1.0 - 0.4 * pen_c)

    return max(0.0, min(1.0, raw))


def generic_paper_penalty(
    paper: dict[str, Any],
    query: str,
    enrichment_terms: list[str] | None = None,
    intent_entry_id: str | None = None,
) -> tuple[float, bool]:
    """
    Returns (multiplier ∈ (0,1], is_generic_survey).
    Only applies strong penalty when paper lacks query-specific overlap.
    """
    title = (paper.get("title") or "").lower()
    blob = _blob(paper)
    specificity = query_specificity_score(query, enrichment_terms)
    qtok = _query_tokens(query, enrichment_terms)
    title_hits = sum(1 for t in qtok if t in title) if qtok else 0
    strong_specific = (
        exact_phrase_overlap(query, paper) >= 0.55
        or title_hits >= max(2, len(qtok) // 2)
        or flagship_match_strength(paper, intent_entry_id, query) >= 0.7
    )
    if strong_specific:
        return 1.0, False

    is_generic = any(m in title or m in blob[:1200] for m in _GENERIC_SURVEY_MARKERS)
    has_llm_umbrella = any(m in title for m in _GENERIC_LLM_WITHOUT_ALIGNMENT)
    alignment_terms = ("alignment", "preference", "rlhf", "dpo", "reward model", "human feedback")
    has_alignment = any(t in blob for t in alignment_terms)

    if not is_generic and not has_llm_umbrella:
        return 1.0, False

    mult = 1.0
    if is_generic:
        mult *= 0.42 if specificity >= 0.5 else 0.55
    if has_llm_umbrella and not has_alignment:
        mult *= 0.48
    if "survey" in title or "review" in title:
        mult *= 0.72
    return max(0.15, mult), is_generic


def flagship_match_strength(
    paper: dict[str, Any],
    intent_entry_id: str | None,
    query: str,
) -> float:
    if not intent_entry_id:
        return 0.0
    markers = _FLAGSHIP_MARKERS.get(intent_entry_id, ())
    if not markers:
        entry = registry_entry_by_id(intent_entry_id)
        if entry:
            markers = tuple(t.lower() for t in entry.enrichment_terms)
    blob = _blob(paper)
    title = (paper.get("title") or "").lower()
    hits = sum(1 for m in markers if m in title or m in blob[:1500])
    if not hits:
        return 0.0
    # Canonical arXiv seed boost
    pid = str(paper.get("id") or "").lower()
    if intent_entry_id == "direct_preference_optimization" and "2305.18290" in pid:
        return 1.0
    if intent_entry_id == "transformer_nlp" and "1706.03762" in pid:
        return 1.0
    return min(1.0, hits / max(2, min(5, len(markers) // 3 + 1)) + (0.25 if hits >= 2 else 0.0))


def graph_importance(node: dict[str, Any]) -> float:
    """Citation structure + metadata richness — NOT query-specific. ∈ [0, 1]."""
    cc = float(node.get("citation_count", 0) or 0)
    rel = float(node.get("relation_expand_score", 0.0) or 0.0)
    gsc = float(node.get("graph_score", node.get("pagerank_norm", 0.0)) or 0.0)
    deg_norm = min(1.0, math.log1p(cc) / math.log1p(120))
    return max(0.0, min(1.0, 0.35 * deg_norm + 0.35 * rel + 0.30 * gsc))


def topical_importance(
    node: dict[str, Any],
    query_profile: dict[str, Any],
) -> float:
    """Query-centric score for ranking / Start Here. ∈ [0, 1]."""
    q = (query_profile.get("query_text") or "").strip()
    enrich = list(query_profile.get("intent_enrichment_terms") or [])
    entry_id = query_profile.get("intent_entry_id")
    cent = compute_query_centrality(
        q, node, enrichment_terms=enrich, intent_entry_id=entry_id
    )
    coh = float(node.get("intent_coherence_score", cent) or cent)
    gen_m, _ = generic_paper_penalty(node, q, enrich, entry_id)
    dom_m = domain_lock_multiplier(node, entry_id, q, enrich)
    flag = flagship_match_strength(node, entry_id, q)
    raw = 0.40 * cent + 0.25 * coh + 0.20 * flag + 0.15 * min(cent, coh)
    raw *= gen_m * dom_m
    if node.get("is_exploratory_bridge"):
        raw *= 0.35
    return max(0.0, min(1.0, raw))


def domain_lock_multiplier(
    node: dict[str, Any],
    intent_entry_id: str | None,
    query: str,
    enrichment_terms: list[str] | None,
) -> float:
    """Soft penalty for drift clusters unless strong query terminology or anchor."""
    if not intent_entry_id:
        return 1.0
    clusters = _DOMAIN_DRIFT_CLUSTERS.get(intent_entry_id, ())
    if not clusters:
        return 1.0
    blob = _blob(node)
    from .intent_coherence import subfield_cluster_scores

    _, pen, _ = subfield_cluster_scores(node, (), clusters)
    if pen < 0.25:
        return 1.0
    is_anchor = node.get("seed_origin") in ("uploaded_pdf", "api_search")
    hops = node.get("anchor_graph_hops")
    near_anchor = hops is not None and int(hops) <= 1
    if (is_anchor or near_anchor) and exact_phrase_overlap(query, node) >= 0.4:
        return 1.0
    qtok = _query_tokens(query, enrichment_terms)
    if sum(1 for t in qtok if t in blob) >= max(2, len(qtok) // 2):
        return max(0.72, 1.0 - 0.35 * pen)
    return max(0.25, 1.0 - 0.85 * pen)


def foundational_topical_score(node: dict[str, Any], query_profile: dict[str, Any]) -> float:
    """
    Foundational = concept-defining, not recent broad surveys.
    """
    topical = topical_importance(node, query_profile)
    y = node.get("year")
    age_pen = 1.0
    if isinstance(y, int):
        cy = datetime.utcnow().year
        age = cy - y
        if age <= 2:
            title = (node.get("title") or "").lower()
            if any(m in title for m in ("survey", "review", "comprehensive", "overview")):
                age_pen = 0.55
        elif age <= 5:
            title = (node.get("title") or "").lower()
            if "comprehensive" in title and "survey" in title:
                age_pen = 0.7
    gen_m, is_gen = generic_paper_penalty(
        node,
        query_profile.get("query_text") or "",
        query_profile.get("intent_enrichment_terms"),
        query_profile.get("intent_entry_id"),
    )
    flag = flagship_match_strength(
        node, query_profile.get("intent_entry_id"), query_profile.get("query_text") or ""
    )
    hist = min(1.0, math.log1p(float(node.get("citation_count", 0) or 0)) / math.log1p(200))
    return topical * age_pen * gen_m * (1.0 + 0.35 * flag) * (0.65 + 0.35 * hist)


def semantic_purity_score(paper: dict[str, Any], query_profile: dict[str, Any]) -> float:
    """
    Start Here gate — how semantically pure is this paper for the query identity?
    """
    from .flagship_registry import is_resolved_flagship_paper

    entry_id = query_profile.get("intent_entry_id")
    if is_resolved_flagship_paper(paper, str(entry_id) if entry_id else None):
        return 1.0
    q = (query_profile.get("query_text") or "").strip()
    enrich = list(query_profile.get("intent_enrichment_terms") or [])
    cent = compute_query_centrality(q, paper, enrichment_terms=enrich, intent_entry_id=entry_id)
    topical = float(paper.get("topical_importance", 0.0) or 0.0) or topical_importance(paper, query_profile)
    phrase = exact_phrase_overlap(q, paper)
    coh = float(paper.get("intent_coherence_score", 0.5) or 0.5)
    gen_m, _ = generic_paper_penalty(paper, q, enrich, entry_id)
    raw = 0.35 * cent + 0.30 * topical + 0.20 * phrase + 0.15 * coh
    raw *= gen_m
    if paper.get("is_exploratory_bridge"):
        raw *= 0.25
    pen = float(paper.get("intent_subfield_penalty", 0.0) or 0.0)
    if pen >= 0.34:
        raw *= 0.3
    return max(0.0, min(1.0, raw))


def strict_start_here_score(paper: dict[str, Any], query_profile: dict[str, Any]) -> float:
    """
    High-trust Start Here ranker — topical dominance, minimal graph hub bias.
    """
    if not _start_here_eligible_strict(paper, query_profile):
        return -1.0
    from .flagship_registry import is_resolved_flagship_paper
    from .query_intent_gating import attach_intent_category, query_centrality_score, start_here_allowed

    attach_intent_category(paper, query_profile)
    cat = paper.get("intent_match_category", "")
    if not start_here_allowed(cat):
        return -1.0

    eid = query_profile.get("intent_entry_id")
    q_cent = query_centrality_score(paper, query_profile)
    if is_resolved_flagship_paper(paper, str(eid) if eid else None) and start_here_allowed(cat):
        return 1e6 + q_cent * 100.0 + topical_importance(paper, query_profile) * 10.0
    topical = topical_importance(paper, query_profile)
    q = (query_profile.get("query_text") or "").strip()
    entry_id = query_profile.get("intent_entry_id")
    phrase = exact_phrase_overlap(q, paper)
    flag = flagship_match_strength(paper, entry_id, q)
    gen_m, is_gen = generic_paper_penalty(
        paper, q, query_profile.get("intent_enrichment_terms"), entry_id
    )
    graph = graph_importance(paper)
    graph_component = min(0.15, graph * 0.15)
    score = q_cent * 8.0 + topical * 3.0 + phrase * 2.0 + flag * 1.5 + graph_component
    if cat == "exact_method_match":
        score += 4.0
    elif cat == "strong_method_variant":
        score += 1.5
    if paper.get("seed_origin") in ("uploaded_pdf", "api_search") and start_here_allowed(cat):
        score += 0.8
    if is_gen:
        score *= 0.2
    if paper.get("is_exploratory_bridge"):
        return -1.0
    pen = float(paper.get("intent_subfield_penalty", 0.0) or 0.0)
    if pen >= 0.34 and flag < 0.55:
        return -1.0
    return score


def _start_here_eligible_strict(paper: dict[str, Any], query_profile: dict[str, Any] | None = None) -> bool:
    from .flagship_registry import is_resolved_flagship_paper
    from .query_intent_gating import attach_intent_category, start_here_allowed

    if paper.get("graph_noise") or paper.get("coherence_suppressed"):
        return False
    if paper.get("is_exploratory_bridge"):
        return False
    if query_profile:
        from .query_intent_gating import attach_paper_role, start_here_role_allowed

        cat = attach_intent_category(paper, query_profile)
        role = attach_paper_role(paper, query_profile)
        if not start_here_allowed(cat) and not start_here_role_allowed(role):
            return False
        eid = query_profile.get("intent_entry_id")
        if is_resolved_flagship_paper(paper, str(eid) if eid else None):
            return True
        purity = semantic_purity_score(paper, query_profile)
        if purity < 0.45 and paper.get("seed_origin") == "discovered":
            return False
        if purity < 0.35:
            return False
    coh = float(paper.get("intent_coherence_score", 0.5) or 0.5)
    topical = float(paper.get("topical_importance", 0.0) or 0.0)
    if topical > 0 and topical < 0.32 and paper.get("seed_origin") == "discovered":
        return False
    if coh < 0.34 and paper.get("seed_origin") == "discovered":
        return False
    return True


def branch_purity_score(
    members: list[dict[str, Any]],
    query_profile: dict[str, Any],
) -> float:
    """Intra-branch semantic/domain consistency ∈ [0, 1]."""
    if len(members) < 2:
        return 1.0
    q = (query_profile.get("query_text") or "").strip()
    enrich = list(query_profile.get("intent_enrichment_terms") or [])
    entry_id = query_profile.get("intent_entry_id")
    topical_vals = [topical_importance(p, query_profile) for p in members]
    mean_top = sum(topical_vals) / len(topical_vals)
    domains = {str(p.get("domain") or "").strip().lower() for p in members if p.get("domain")}
    domain_pen = 0.15 if len(domains) > 2 else 0.0
    gen_frac = 0.0
    for p in members:
        _, is_gen = generic_paper_penalty(p, q, enrich, entry_id)
        if is_gen:
            gen_frac += 1.0
    gen_frac /= len(members)
    return max(0.0, min(1.0, mean_top * (1.0 - domain_pen) * (1.0 - 0.5 * gen_frac)))


def apply_topical_ranking_layer(
    nodes: list[dict[str, Any]],
    query_profile: dict[str, Any],
    *,
    graph_blend: float = 0.28,
) -> None:
    """
    Attach topical/graph scores and rebalance ``relevance_raw`` (graph vs topical).
    """
    q = (query_profile.get("query_text") or "").strip()
    enrich = list(query_profile.get("intent_enrichment_terms") or [])
    entry_id = query_profile.get("intent_entry_id")
    for node in nodes:
        if node.get("graph_noise"):
            continue
        g_imp = graph_importance(node)
        t_imp = topical_importance(node, query_profile)
        cent = compute_query_centrality(q, node, enrichment_terms=enrich, intent_entry_id=entry_id)
        gen_m, is_gen = generic_paper_penalty(node, q, enrich, entry_id)
        flag = flagship_match_strength(node, entry_id, q)
        node["graph_importance"] = round(g_imp, 4)
        node["topical_importance"] = round(t_imp, 4)
        node["query_centrality"] = round(cent, 4)
        node["generic_paper_penalty_mult"] = round(gen_m, 4)
        node["is_generic_survey"] = bool(is_gen)
        node["flagship_match_strength"] = round(flag, 4)
        raw = float(node.get("relevance_raw", 0.0) or 0.0)
        # Multiplicative rebalance: topical dominates; graph hubs without topical fit are damped.
        topical_mult = 0.42 + 0.58 * t_imp
        hub_drift = max(0.0, g_imp - t_imp - 0.15)
        graph_pen = 1.0 - (1.0 - graph_blend) * hub_drift
        blended = raw * topical_mult * graph_pen * gen_m
        if node.get("is_exploratory_bridge"):
            blended *= 0.38
        if is_gen and flag < 0.55:
            blended *= 0.42
        if flag >= 0.75:
            blended *= 1.0 + 0.35 * flag
        node["relevance_raw"] = blended
        node["final_score"] = round(blended, 6)
        if node.get("is_foundational_hub"):
            f_top = foundational_topical_score(node, query_profile)
            node["foundational_topical_score"] = round(f_top, 4)


def explanation_topical_tier(paper: dict[str, Any]) -> str:
    """Editorial explanation tier: foundational | related_method | historical_lineage | recent_frontier | exploratory."""
    from datetime import datetime

    flag = float(paper.get("flagship_match_strength", 0.0) or 0.0)
    cent = float(paper.get("query_centrality", 0.0) or 0.0)
    topical = float(paper.get("topical_importance", 0.0) or 0.0)
    cy = datetime.utcnow().year
    y = paper.get("year")
    try:
        year = int(y) if y is not None else None
    except (TypeError, ValueError):
        year = None

    if paper.get("is_exploratory_bridge") or topical < 0.30:
        return "exploratory"
    if flag >= 0.72:
        return "foundational"
    if paper.get("is_foundational_hub") and paper.get("foundational_eligible", True):
        if year is not None and year <= cy - 5:
            return "historical_lineage"
        return "foundational"
    if year is not None and year >= cy - 3 and (topical >= 0.45 or cent >= 0.50):
        return "recent_frontier"
    if topical >= 0.52 or cent >= 0.55:
        return "related_method"
    if paper.get("is_generic_survey"):
        return "exploratory"
    tier = (paper.get("explanation_confidence_tier") or "").strip().lower()
    if tier in ("cross_domain_bridge", "citation_adjacent", "lexically_related"):
        return "exploratory"
    if topical >= 0.38:
        return "related_method"
    if year is not None and year <= cy - 6:
        return "historical_lineage"
    return "exploratory"


__all__ = [
    "apply_topical_ranking_layer",
    "branch_purity_score",
    "compute_query_centrality",
    "domain_lock_multiplier",
    "exact_phrase_overlap",
    "flagship_match_strength",
    "foundational_topical_score",
    "generic_paper_penalty",
    "graph_importance",
    "semantic_purity_score",
    "strict_start_here_score",
    "topical_importance",
    "explanation_topical_tier",
]
