"""
Hard query-intent gating — precision-first Start Here / Foundational selection.

Classifies each paper before ranking surfaces; blocks adjacent RLHF and generic surveys
from Start Here when the query targets a specific method (e.g. DPO).
"""

from __future__ import annotations

import difflib
import re
from typing import Any

EXACT_METHOD_MATCH = "exact_method_match"
STRONG_METHOD_VARIANT = "strong_method_variant"
ADJACENT_ALIGNMENT = "adjacent_alignment"
GENERIC_RLHF = "generic_rlhf"
UNRELATED_NOISE = "unrelated_noise"

START_HERE_CATEGORIES = frozenset({EXACT_METHOD_MATCH, STRONG_METHOD_VARIANT, ADJACENT_ALIGNMENT})
FOUNDATIONAL_CATEGORIES = frozenset({EXACT_METHOD_MATCH, STRONG_METHOD_VARIANT})

# Semantic roles for hierarchy + information-dense labels.
ROLE_CANONICAL_ANCHOR = "canonical_anchor"
ROLE_CORE_VARIANT = "core_variant"
ROLE_METHOD_EXTENSION = "method_extension"
ROLE_ALIGNMENT_BASELINE = "alignment_baseline"
ROLE_CROSS_DOMAIN = "cross_domain_adaptation"
ROLE_EXPLORATORY = "exploratory_application"

FOUNDATIONAL_ROLES = frozenset({ROLE_CANONICAL_ANCHOR, ROLE_CORE_VARIANT, ROLE_METHOD_EXTENSION})
START_HERE_ROLES = frozenset({ROLE_ALIGNMENT_BASELINE, ROLE_CORE_VARIANT, ROLE_METHOD_EXTENSION})

_ROLE_SORT_KEY = {
    ROLE_CANONICAL_ANCHOR: 0,
    ROLE_METHOD_EXTENSION: 1,
    ROLE_CORE_VARIANT: 2,
    ROLE_ALIGNMENT_BASELINE: 3,
    ROLE_CROSS_DOMAIN: 9,
    ROLE_EXPLORATORY: 9,
}

_DPO_CROSS_DOMAIN_MARKERS = (
    "protein",
    "diffusion model",
    "text-to-audio",
    "text-to-image",
    "audio generation",
    "image generation",
    "stable diffusion",
    "experimental fitness",
    "tango 2",
    "histopathology",
    "drug discovery",
)

_DPO_LLM_CORE_MARKERS = (
    "language model",
    "large language model",
    " llm",
    "reward model",
    "secretly a reward",
    "human feedback",
)

# --- Global off-topic markers (unless query explicitly targets domain) ---
_OFF_TOPIC: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("protein_biomed", ("protein folding", "genomics", "histopathology", "clinical trial", "drug discovery")),
    ("cybersecurity", ("cybersecurity", "cyber security", "malware", "intrusion detection", "vulnerability")),
    ("recommender", ("recommender system", "collaborative filtering", "sequential recommendation", "click-through")),
    ("diffusion_image", ("stable diffusion", "text-to-image", "denoising diffusion", "image generation")),
)

_GENERIC_SURVEY = (
    "comprehensive review",
    "comprehensive survey",
    "literature review",
    "systematic review",
    "survey of",
    "overview of",
    "state of the art",
)


def _blob(paper: dict[str, Any]) -> str:
    return f"{paper.get('title', '')} {str(paper.get('abstract') or '')[:3000]}".lower()


def _title(paper: dict[str, Any]) -> str:
    return (paper.get("title") or "").lower()


def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(m in text for m in markers)


def _norm_title_blob(title: str) -> str:
    return re.sub(r"[^\w\s\-]", " ", (title or "").lower()).strip()


def _titles_near_duplicate(a: str, b: str) -> bool:
    na, nb = _norm_title_blob(a), _norm_title_blob(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if difflib.SequenceMatcher(None, na, nb).ratio() >= 0.82:
        return True
    ta = set(re.findall(r"[a-z][a-z0-9\-]{3,}", na))
    tb = set(re.findall(r"[a-z][a-z0-9\-]{3,}", nb))
    if not ta or not tb:
        return False
    j = len(ta & tb) / max(1, len(ta | tb))
    return j >= 0.62


def _query_blob(qp: dict[str, Any]) -> str:
    parts = [qp.get("query_text") or "", qp.get("intent_entry_id") or ""]
    parts.extend(str(t) for t in (qp.get("intent_enrichment_terms") or []))
    return " ".join(parts).lower()


def _is_off_topic_noise(paper: dict[str, Any], qp: dict[str, Any], *, allow_method_signal: bool) -> bool:
    if allow_method_signal:
        return False
    blob = _blob(paper)
    qb = _query_blob(qp)
    for theme_id, markers in _OFF_TOPIC:
        if theme_id == "cybersecurity" and "cyber" in qb:
            continue
        if theme_id == "recommender" and "recommend" in qb:
            continue
        if _has_any(blob, markers):
            return True
    return False


def _is_dpo_llm_core(title: str, blob: str) -> bool:
    return _has_any(f"{title} {blob[:800]}", _DPO_LLM_CORE_MARKERS)


def _is_dpo_cross_domain_application(title: str, blob: str) -> bool:
    mentions_dpo = "direct preference optimization" in title or bool(re.search(r"\bdpo\b", title))
    if not mentions_dpo:
        return False
    if _is_dpo_llm_core(title, blob):
        return False
    return _has_any(f"{title} {blob[:1200]}", _DPO_CROSS_DOMAIN_MARKERS)


def _classify_dpo(paper: dict[str, Any], qp: dict[str, Any]) -> str:
    title = _title(paper)
    blob = _blob(paper)
    pid = str(paper.get("id") or "").lower()

    has_dpo_title = (
        "direct preference optimization" in title
        or bool(re.search(r"\bdpo\b", title))
        or bool(re.search(r"\bcal-?dpo\b", title))
        or bool(re.search(r"\brs-?dpo\b", title))
    )
    has_dpo_blob = "direct preference optimization" in blob or bool(re.search(r"\bdpo\b", blob[:1200]))

    if _is_dpo_cross_domain_application(title, blob):
        return UNRELATED_NOISE

    if _is_off_topic_noise(paper, qp, allow_method_signal=has_dpo_title or has_dpo_blob):
        return UNRELATED_NOISE

    if "2305.18290" in pid or (
        "direct preference optimization" in title and _is_dpo_llm_core(title, blob)
    ):
        return EXACT_METHOD_MATCH
    if re.search(r"\bcal-?dpo\b", title) or re.search(r"\brs-?dpo\b", title):
        return STRONG_METHOD_VARIANT
    if re.search(r"\bdpo\b", title) and _is_dpo_llm_core(title, blob):
        return STRONG_METHOD_VARIANT

    variant_title = (
        "orpo",
        "simpo",
        "ipo",
        "kto",
        "cipo",
        "rrhf",
        "iterative reasoning preference",
    )
    if any(v in title for v in variant_title):
        return STRONG_METHOD_VARIANT

    adjacent = (
        "rlhf",
        "reinforcement learning from human feedback",
        "follow instructions",
        "instruction following",
        "instructgpt",
        "instruction tuning",
        "constitutional ai",
    )
    if _has_any(title, adjacent) or (_has_any(blob[:900], adjacent) and not has_dpo_title):
        if _has_any(title, _GENERIC_SURVEY) or _has_any(blob[:400], _GENERIC_SURVEY):
            return GENERIC_RLHF
        return ADJACENT_ALIGNMENT

    if _has_any(title, _GENERIC_SURVEY) and _has_any(
        blob, ("alignment", "preference", "rlhf", "language model", "llm")
    ):
        return GENERIC_RLHF

    if has_dpo_blob and not has_dpo_title and _is_dpo_llm_core(title, blob):
        return STRONG_METHOD_VARIANT

    return UNRELATED_NOISE


def _classify_transformer(paper: dict[str, Any], qp: dict[str, Any]) -> str:
    title = _title(paper)
    blob = _blob(paper)
    if _has_any(blob, ("power transformer", "substation", "electrical grid", "fault diagnosis")):
        return UNRELATED_NOISE
    if "attention is all you need" in title or "1706.03762" in str(paper.get("id") or "").lower():
        return EXACT_METHOD_MATCH
    if _has_any(title, ("transformer", "self-attention", "multi-head attention")) and not _has_any(
        title, ("power", "electrical", "substation")
    ):
        return EXACT_METHOD_MATCH
    if _has_any(title, ("bert", "gpt-", "encoder-decoder")):
        return STRONG_METHOD_VARIANT
    if _has_any(title, _GENERIC_SURVEY):
        return GENERIC_RLHF
    if "transformer" in blob:
        return ADJACENT_ALIGNMENT
    return UNRELATED_NOISE


def _classify_visual_debugging(paper: dict[str, Any], qp: dict[str, Any]) -> str:
    title = _title(paper)
    blob = _blob(paper)
    if _has_any(blob, ("histopathology", "mri segmentation", "clinical imaging")):
        return UNRELATED_NOISE
    if _has_any(title, ("software visualization", "program visualization", "debugger", "debugging")):
        return EXACT_METHOD_MATCH
    if _has_any(title, ("execution trace", "dynamic analysis", "developer tool")):
        return STRONG_METHOD_VARIANT
    if _has_any(title, ("program slicing", "static analysis survey")):
        return ADJACENT_ALIGNMENT
    if _has_any(title, _GENERIC_SURVEY):
        return GENERIC_RLHF
    return UNRELATED_NOISE


def _classify_gnn(paper: dict[str, Any], qp: dict[str, Any]) -> str:
    title = _title(paper)
    blob = _blob(paper)
    if _has_any(blob, ("social network analysis survey", "community detection survey")):
        return UNRELATED_NOISE
    if _has_any(
        title,
        ("graph convolutional", "graph attention", "graph neural network", "message passing", "graphsage"),
    ):
        return EXACT_METHOD_MATCH
    if "graph" in title and _has_any(title, ("network", "learning", "embedding")):
        return STRONG_METHOD_VARIANT
    if _has_any(title, _GENERIC_SURVEY):
        return GENERIC_RLHF
    return UNRELATED_NOISE


def _classify_fallback(paper: dict[str, Any], qp: dict[str, Any]) -> str:
    from .topical_ranking import _query_tokens, compute_query_centrality, exact_phrase_overlap, generic_paper_penalty

    q = (qp.get("query_text") or "").strip()
    entry = qp.get("intent_entry_id")
    title = _title(paper)
    qtok = _query_tokens(q, qp.get("intent_enrichment_terms"))
    title_hits = sum(1 for t in qtok if t in title) if qtok else 0
    cent = compute_query_centrality(
        q, paper, enrichment_terms=qp.get("intent_enrichment_terms"), intent_entry_id=entry
    )
    phrase = exact_phrase_overlap(q, paper)
    gen_m, is_gen = generic_paper_penalty(
        paper, q, qp.get("intent_enrichment_terms"), entry
    )
    if _is_off_topic_noise(paper, qp, allow_method_signal=phrase >= 0.5 or title_hits >= 2):
        return UNRELATED_NOISE
    if is_gen or gen_m < 0.55:
        return GENERIC_RLHF
    if phrase >= 0.55 or (title_hits >= max(2, len(qtok) // 2) and title_hits >= 2):
        return EXACT_METHOD_MATCH
    if cent >= 0.48 or title_hits >= max(1, len(qtok) // 2):
        return STRONG_METHOD_VARIANT
    if cent >= 0.32:
        return ADJACENT_ALIGNMENT
    return UNRELATED_NOISE


def classify_paper_intent_category(paper: dict[str, Any], query_profile: dict[str, Any]) -> str:
    """Assign one intent-match category using title + abstract rules."""
    entry = str(query_profile.get("intent_entry_id") or "")
    if entry == "direct_preference_optimization":
        return _classify_dpo(paper, query_profile)
    if entry == "transformer_nlp":
        return _classify_transformer(paper, query_profile)
    if entry == "visual_debugging":
        return _classify_visual_debugging(paper, query_profile)
    if entry == "graph_neural_networks":
        return _classify_gnn(paper, query_profile)
    if entry == "rlhf_alignment":
        blob = _blob(paper)
        if "direct preference" in blob or re.search(r"\bdpo\b", _title(paper)):
            return STRONG_METHOD_VARIANT
        if _has_any(_title(paper), ("rlhf", "human feedback", "reward model", "follow instructions")):
            return EXACT_METHOD_MATCH
        if _has_any(_title(paper), _GENERIC_SURVEY):
            return GENERIC_RLHF
        return ADJACENT_ALIGNMENT
    return _classify_fallback(paper, query_profile)


def attach_intent_category(paper: dict[str, Any], query_profile: dict[str, Any]) -> str:
    cat = classify_paper_intent_category(paper, query_profile)
    paper["intent_match_category"] = cat
    return cat


def _role_for_dpo(paper: dict[str, Any], category: str) -> str:
    title = _title(paper)
    blob = _blob(paper)
    pid = str(paper.get("id") or "").lower()

    if _is_dpo_cross_domain_application(title, blob):
        return ROLE_CROSS_DOMAIN
    if "2305.18290" in pid or (
        "direct preference optimization" in title
        and "secretly a reward" in title
        and _is_dpo_llm_core(title, blob)
    ):
        return ROLE_CANONICAL_ANCHOR
    if re.search(r"\bcal-?dpo\b", title):
        return ROLE_METHOD_EXTENSION
    if re.search(r"\brs-?dpo\b", title):
        return ROLE_METHOD_EXTENSION
    if "orpo" in title or "simpo" in title or "ipo" in title or "rrhf" in title:
        return ROLE_METHOD_EXTENSION
    if "iterative reasoning preference" in title:
        return ROLE_CORE_VARIANT
    if category == ADJACENT_ALIGNMENT:
        return ROLE_ALIGNMENT_BASELINE
    if category == GENERIC_RLHF:
        return ROLE_EXPLORATORY
    if category == STRONG_METHOD_VARIANT and _is_dpo_llm_core(title, blob):
        if "disentangling" in title or "hybrid" in title:
            return ROLE_CORE_VARIANT
        return ROLE_METHOD_EXTENSION
    if category == EXACT_METHOD_MATCH and _is_dpo_llm_core(title, blob):
        return ROLE_METHOD_EXTENSION
    return ROLE_EXPLORATORY


def _role_for_generic_intent(
    paper: dict[str, Any],
    category: str,
    entry: str,
) -> str:
    title = _title(paper)
    if category in (UNRELATED_NOISE, GENERIC_RLHF):
        return ROLE_EXPLORATORY if category == GENERIC_RLHF else ROLE_CROSS_DOMAIN
    if category == ADJACENT_ALIGNMENT:
        return ROLE_ALIGNMENT_BASELINE
    if entry == "transformer_nlp" and "attention is all you need" in title:
        return ROLE_CANONICAL_ANCHOR
    if category == EXACT_METHOD_MATCH:
        return ROLE_CANONICAL_ANCHOR
    if category == STRONG_METHOD_VARIANT:
        return ROLE_METHOD_EXTENSION
    return ROLE_EXPLORATORY


def classify_paper_role(paper: dict[str, Any], query_profile: dict[str, Any]) -> str:
    """Fine-grained role for hierarchy, labels, and section assignment."""
    entry = str(query_profile.get("intent_entry_id") or "")
    cat = paper.get("intent_match_category") or classify_paper_intent_category(paper, query_profile)
    if entry == "direct_preference_optimization":
        return _role_for_dpo(paper, cat)
    return _role_for_generic_intent(paper, cat, entry)


def attach_paper_role(paper: dict[str, Any], query_profile: dict[str, Any]) -> str:
    role = classify_paper_role(paper, query_profile)
    paper["paper_role"] = role
    return role


def start_here_allowed(category: str) -> bool:
    return category in START_HERE_CATEGORIES


def start_here_role_allowed(role: str) -> bool:
    return role in START_HERE_ROLES


def foundational_allowed(category: str) -> bool:
    return category in FOUNDATIONAL_CATEGORIES


def foundational_role_allowed(role: str) -> bool:
    return role in FOUNDATIONAL_ROLES


def branch_member_allowed(category: str, *, strict: bool = True) -> bool:
    """Strict mode drops adjacent RLHF-style papers from branch previews."""
    if category in (UNRELATED_NOISE, GENERIC_RLHF):
        return False
    if strict and category == ADJACENT_ALIGNMENT:
        return False
    return True


def query_centrality_score(paper: dict[str, Any], query_profile: dict[str, Any]) -> float:
    """
    How central is this paper to the exact queried method?
    Large bonuses for title phrase, acronym, and intro/canonical papers.
    """
    from .topical_ranking import compute_query_centrality, exact_phrase_overlap, graph_importance

    qp = query_profile or {}
    q = (qp.get("query_text") or "").strip()
    entry = qp.get("intent_entry_id")
    base = compute_query_centrality(
        q, paper, enrichment_terms=qp.get("intent_enrichment_terms"), intent_entry_id=entry
    )
    title = _title(paper)
    phrase = exact_phrase_overlap(q, paper)
    cat = paper.get("intent_match_category") or classify_paper_intent_category(paper, qp)

    bonus = 0.0
    if phrase >= 0.55:
        bonus += 0.35
    ql = q.lower()
    if len(ql) >= 3 and ql in title:
        bonus += 0.25
    # acronym: DPO, GNN, etc.
    words = re.findall(r"[a-z]{2,}", ql)
    for acr in words:
        if len(acr) <= 5 and re.search(rf"\b{re.escape(acr)}\b", title):
            bonus += 0.18
            break

    if cat == EXACT_METHOD_MATCH:
        bonus += 0.42
    elif cat == STRONG_METHOD_VARIANT:
        bonus += 0.22
    elif cat == ADJACENT_ALIGNMENT:
        bonus -= 0.35
    elif cat in (GENERIC_RLHF, UNRELATED_NOISE):
        bonus -= 0.55

    pid = str(paper.get("id") or "").lower()
    if entry == "direct_preference_optimization" and "2305.18290" in pid:
        bonus += 0.5
    if entry == "transformer_nlp" and "1706.03762" in pid:
        bonus += 0.5

    graph = graph_importance(paper)
    bonus += min(0.12, graph * 0.12)

    return max(0.0, min(1.0, base + bonus))


def _role_label_prefix(role: str) -> str:
    return {
        ROLE_CANONICAL_ANCHOR: "Canonical anchor paper",
        ROLE_CORE_VARIANT: "Core variant of DPO",
        ROLE_METHOD_EXTENSION: "Method extension",
        ROLE_ALIGNMENT_BASELINE: "Alignment baseline / predecessor",
        ROLE_CROSS_DOMAIN: "Cross-domain adaptation",
        ROLE_EXPLORATORY: "Exploratory application",
    }.get(role, "Related literature")


def rich_paper_description(
    paper: dict[str, Any],
    query_profile: dict[str, Any],
    *,
    role: str | None = None,
) -> str:
    """Role-aware, information-dense one-liner — unique per paper where possible."""
    qp = query_profile or {}
    r = role or attach_paper_role(paper, qp)
    title = _title(paper)
    prefix = _role_label_prefix(r)

    if r == ROLE_CANONICAL_ANCHOR and "direct preference" in title:
        return (
            f"{prefix} defining Direct Preference Optimization; "
            "introduces the reward-model equivalence for LLM alignment"
        )
    if "cal-dpo" in title:
        return f"{prefix}; calibrated objective improves reliability of DPO-style alignment"
    if "rs-dpo" in title:
        return f"{prefix}; hybrid rejection-sampling plus DPO for stronger LLM alignment"
    if "orpo" in title:
        return f"{prefix}; monolithic preference optimization without a separate reference model"
    if "simpo" in title:
        return f"{prefix}; simplified preference optimization competitive with DPO/ORPO"
    if "rrhf" in title:
        return f"{prefix}; rank-based human feedback alignment without full RLHF pipeline"
    if "iterative reasoning preference" in title:
        return f"{prefix}; extends preference optimization to reasoning-centric LLM training"
    if "disentangling length" in title:
        return f"{prefix}; separates length bias from quality in DPO training"
    if r == ROLE_ALIGNMENT_BASELINE and "follow instructions" in title:
        return f"{prefix}; classic RLHF pipeline that DPO methods build upon"
    if r == ROLE_ALIGNMENT_BASELINE:
        return f"{prefix}; contextualizes modern preference-tuning against human-feedback training"
    if r == ROLE_CROSS_DOMAIN:
        entry = str(qp.get("intent_entry_id") or "")
        short = (paper.get("title") or "This work")[:60]
        if entry == "direct_preference_optimization":
            return f"{prefix}; applies DPO machinery outside core LLM alignment ({short}…)"
        return f"{prefix}; off-topic domain relative to your query ({short}…)"
    if r == ROLE_EXPLORATORY:
        return f"{prefix}; weak or indirect tie to the queried method"
    if r == ROLE_CORE_VARIANT:
        return f"{prefix}; directly modifies or tightens the DPO training objective"
    if r == ROLE_METHOD_EXTENSION:
        return f"{prefix}; alternative preference-optimization objective in the DPO family"
    if "attention is all you need" in title:
        return f"{prefix}; defines the Transformer architecture for sequence modeling"
    return f"{prefix}; central to the literature neighborhood for your query"


def compact_explanation(
    paper: dict[str, Any],
    query_profile: dict[str, Any],
    *,
    category: str | None = None,
) -> str:
    """Alias for rich_paper_description (back-compat)."""
    return rich_paper_description(paper, query_profile)


def _entry_dict(p: dict[str, Any], qp: dict[str, Any]) -> dict[str, Any]:
    pid = str(p.get("id") or "")
    yr = p.get("year")
    try:
        year_out = int(yr) if yr is not None else None
    except (TypeError, ValueError):
        year_out = None
    role = attach_paper_role(p, qp)
    return {
        "paper_id": pid,
        "title": (p.get("title") or pid).strip(),
        "year": year_out,
        "one_line": rich_paper_description(p, qp, role=role),
        "paper_role": role,
        "intent_match_category": p.get("intent_match_category"),
    }


def build_foundational_paper_entries(
    papers: list[dict[str, Any]],
    query_profile: dict[str, Any],
    *,
    max_items: int = 10,
) -> list[dict[str, Any]]:
    """Static reference backbone — anchor, then extensions/variants (LLM-core only)."""
    from .topical_ranking import foundational_topical_score

    qp = query_profile or {}
    pool: list[dict[str, Any]] = []
    for p in papers:
        if p.get("graph_noise"):
            continue
        attach_intent_category(p, qp)
        role = attach_paper_role(p, qp)
        if not foundational_role_allowed(role):
            continue
        pool.append(p)

    pool.sort(
        key=lambda p: (
            _ROLE_SORT_KEY.get(p.get("paper_role", ""), 5),
            -query_centrality_score(p, qp),
            -foundational_topical_score(p, qp),
            -float(p.get("citation_count", 0) or 0),
        ),
    )

    out: list[dict[str, Any]] = []
    seen_titles: list[str] = []
    seen_lines: set[str] = set()
    for p in pool:
        if len(out) >= max_items:
            break
        ttl = (p.get("title") or "").strip()
        ttl_l = ttl.lower()
        if ttl_l in {t.lower() for t in seen_titles}:
            continue
        if any(_titles_near_duplicate(ttl, prev) for prev in seen_titles):
            continue
        entry = _entry_dict(p, qp)
        line = entry["one_line"]
        if line in seen_lines and entry["paper_role"] != ROLE_CANONICAL_ANCHOR:
            continue
        seen_titles.append(ttl)
        seen_lines.add(line)
        out.append(entry)

    anchors = [e for e in out if e.get("paper_role") == ROLE_CANONICAL_ANCHOR]
    rest = [e for e in out if e.get("paper_role") != ROLE_CANONICAL_ANCHOR]
    return anchors + rest


# --- DPO Start Here: hard filter (precision >> recall) ---------------------------------

_DPO_HUB_TITLE_MARKERS = (
    "direct preference optimization",
    "cal-dpo",
    "calibrated direct preference",
    "orpo:",
    "orpo ",
)

_STANDARD_RLHF_DPO_LINEAGE = (
    "training language models to follow instructions",
    "learning to summarize from human feedback",
    "deep reinforcement learning from human preferences",
    "fine-tuning language models from human preferences",
    "proximal policy optimization",
    "instructgpt",
    "constitutional ai: feedback",
)

_START_HERE_HARD_BLOCK = (
    "consciousness",
    "recursive intelligence",
    "constraint-preserving",
    "philosophical",
    "empirical validation and architectural",
    "cat's theory",
    "cats theory",
    "alignment baseline / predecessor; contextualizes",
    "conceptual alignment",
    "ai safety debate",
    "multi-agent reinforcement",
    "robotics control",
    "game playing",
    "atari",
    "mujoco",
    "generic reinforcement learning",
)

_PPO_RLHF_LINEAGE = (
    "proximal policy optimization",
    "ppo",
    "reinforcement learning from human feedback",
    "rlhf",
    "reward model",
    "instructgpt",
    "follow instructions with human feedback",
)


def _resolve_dpo_anchor_ids(papers: list[dict[str, Any]], qp: dict[str, Any]) -> set[str]:
    from .flagship_registry import resolve_flagship_paper_ids

    ids = set(resolve_flagship_paper_ids(papers, "direct_preference_optimization"))
    for p in papers:
        pid = str(p.get("id") or "")
        if pid and "2305.18290" in pid.lower():
            ids.add(pid)
    return ids


def _resolve_dpo_hub_ids(papers: list[dict[str, Any]]) -> set[str]:
    hubs: set[str] = set()
    for p in papers:
        pid = str(p.get("id") or "")
        if not pid:
            continue
        t = _title(p)
        if "2305.18290" in pid.lower():
            hubs.add(pid)
            continue
        if any(m in t for m in _DPO_HUB_TITLE_MARKERS):
            hubs.add(pid)
        elif re.search(r"\bcal-?dpo\b", t) or re.search(r"\borpo\b", t):
            hubs.add(pid)
    return hubs


def _reference_id_set(papers: list[dict[str, Any]], source_ids: set[str]) -> set[str]:
    by_id = {str(p["id"]): p for p in papers if p.get("id")}
    refs: set[str] = set()
    for sid in source_ids:
        p = by_id.get(sid)
        if not p:
            continue
        raw = p.get("references")
        if not isinstance(raw, list):
            continue
        for r in raw:
            rs = str(r).strip().lower()
            if rs:
                refs.add(rs)
    return refs


def _paper_matches_reference(paper: dict[str, Any], ref_ids: set[str]) -> bool:
    if not ref_ids:
        return False
    pid = str(paper.get("id") or "").strip().lower()
    arx = str(paper.get("arxiv_id") or "").strip().lower()
    doi = str(paper.get("doi") or "").strip().lower()
    for ref in ref_ids:
        if pid and (pid == ref or pid in ref or ref in pid):
            return True
        if arx and (arx in ref or ref.endswith(arx) or arx.endswith(ref.replace("arxiv:", ""))):
            return True
        if doi and doi in ref:
            return True
    return False


def _graph_linked_to_hubs(
    paper_id: str,
    hub_ids: set[str],
    edges: list[dict[str, Any]],
) -> bool:
    if not hub_ids or not edges:
        return False
    for e in edges:
        if not isinstance(e, dict):
            continue
        s, t = str(e.get("source") or ""), str(e.get("target") or "")
        if s == paper_id and t in hub_ids:
            return True
        if t == paper_id and s in hub_ids:
            return True
    return False


def _anchor_neighborhood_evidence(
    paper: dict[str, Any],
    papers: list[dict[str, Any]],
    *,
    anchor_ids: set[str],
    top_k: int = 12,
    min_semantic: float = 0.52,
    max_hops: int = 2,
) -> bool:
    pid = str(paper.get("id") or "")
    if pid in anchor_ids:
        return True
    hops = paper.get("anchor_graph_hops")
    sem = float(paper.get("semantic_score", paper.get("semantic_fit_score", 0.0)) or 0.0)
    if hops is not None and int(hops) <= max_hops and sem >= min_semantic:
        return True
    clean = [p for p in papers if not p.get("graph_noise") and p.get("id")]
    ranked = sorted(
        clean,
        key=lambda p: float(p.get("semantic_score", p.get("semantic_fit_score", 0.0)) or 0.0),
        reverse=True,
    )
    top_ids = {str(p["id"]) for p in ranked[:top_k]}
    return pid in top_ids and sem >= min_semantic


def _is_dpo_start_here_blocked(paper: dict[str, Any]) -> bool:
    blob = _blob(paper)
    title = _title(paper)
    if _has_any(blob, _START_HERE_HARD_BLOCK):
        return True
    if _has_any(title, ("consciousness", "philosophical", "recursive intelligence", "cat's theory")):
        return True
    if "human feedback" in blob and not _has_any(
        blob, _PPO_RLHF_LINEAGE + ("preference optimization", "dpo", "rlhf", "reward model", "instruction")
    ):
        return True
    if "reinforcement learning" in blob and not _has_any(
        blob, ("preference", "alignment", "language model", "llm", "rlhf", "reward model", "dpo")
    ):
        return True
    return False


def _is_standard_rlhf_dpo_lineage(paper: dict[str, Any]) -> bool:
    title = _title(paper)
    blob = _blob(paper)[:1200]
    return _has_any(title, _STANDARD_RLHF_DPO_LINEAGE) or _has_any(
        title, ("follow instructions with human feedback", "instructgpt")
    )


def _is_dpo_family_role(role: str) -> bool:
    return role in (ROLE_CANONICAL_ANCHOR, ROLE_METHOD_EXTENSION, ROLE_CORE_VARIANT)


def dpo_start_here_strict_eligible(
    paper: dict[str, Any],
    papers: list[dict[str, Any]],
    query_profile: dict[str, Any],
    *,
    graph_edges: list[dict[str, Any]] | None = None,
) -> bool:
    """
  Return True only when a paper is clearly in the DPO family or standard RLHF→DPO
    lineage AND has citation / neighborhood / lineage evidence. If uncertain, False.
    """
    if str(query_profile.get("intent_entry_id") or "") != "direct_preference_optimization":
        return True

    if paper.get("graph_noise") or _is_dpo_start_here_blocked(paper):
        return False

    attach_intent_category(paper, query_profile)
    role = attach_paper_role(paper, query_profile)

    if role in (ROLE_CROSS_DOMAIN, ROLE_EXPLORATORY):
        return False
    if role == ROLE_ALIGNMENT_BASELINE and not _is_standard_rlhf_dpo_lineage(paper):
        return False

    is_family = _is_dpo_family_role(role)
    is_lineage = role == ROLE_ALIGNMENT_BASELINE and _is_standard_rlhf_dpo_lineage(paper)
    if not is_family and not is_lineage:
        return False

    anchor_ids = _resolve_dpo_anchor_ids(papers, query_profile)
    hub_ids = _resolve_dpo_hub_ids(papers)
    hub_ids |= anchor_ids
    ref_union = _reference_id_set(papers, hub_ids)
    pid = str(paper.get("id") or "")
    edges = graph_edges or []

    if _paper_matches_reference(paper, ref_union):
        return True
    if pid in hub_ids:
        return True
    if _graph_linked_to_hubs(pid, hub_ids, edges):
        return True
    if _anchor_neighborhood_evidence(paper, papers, anchor_ids=anchor_ids | hub_ids):
        if is_family:
            return True
        if is_lineage and _has_any(_blob(paper), _PPO_RLHF_LINEAGE):
            return True

    if is_lineage:
        return False

    if is_family and paper.get("seed_origin") in ("api_search", "uploaded_pdf"):
        sem = float(paper.get("semantic_score", 0.0) or 0.0)
        if sem >= 0.62 and _is_dpo_llm_core(_title(paper), _blob(paper)):
            return True

    return False


def _pick_dpo_start_here_strict(
    papers: list[dict[str, Any]],
    query_profile: dict[str, Any],
    exclude_ids: set[str],
    *,
    max_items: int = 5,
    graph_edges: list[dict[str, Any]] | None = None,
) -> list[str]:
    from .topical_ranking import strict_start_here_score

    qp = query_profile or {}
    extensions: list[tuple[int, float, str]] = []
    baselines: list[tuple[float, str]] = []

    for p in papers:
        pid = str(p.get("id") or "")
        if not pid or pid in exclude_ids:
            continue
        if not dpo_start_here_strict_eligible(p, papers, qp, graph_edges=graph_edges):
            continue
        role = attach_paper_role(p, qp)
        score = strict_start_here_score(p, qp)
        if score < 0:
            score = query_centrality_score(p, qp)
        rk = _ROLE_SORT_KEY.get(role, 5)
        if role == ROLE_ALIGNMENT_BASELINE:
            baselines.append((score, pid))
        elif _is_dpo_family_role(role):
            extensions.append((rk, score, pid))

    extensions.sort(key=lambda t: (t[0], -t[1]))
    baselines.sort(reverse=True)

    picked: list[str] = []
    for _, _, pid in extensions:
        if len(picked) >= max_items:
            break
        if pid not in picked:
            picked.append(pid)
    if len(picked) < max_items and baselines:
        pid = baselines[0][1]
        if pid not in picked:
            picked.append(pid)
    return picked[:max_items]


def pick_start_here_ids(
    papers: list[dict[str, Any]],
    query_profile: dict[str, Any],
    exclude_ids: set[str],
    *,
    max_items: int = 5,
    graph_edges: list[dict[str, Any]] | None = None,
) -> list[str]:
    """
    Curated onboarding path — disjoint from foundational IDs.
    DPO queries use a hard evidence filter; other intents use role-gated ranking.
    """
    qp = query_profile or {}
    if str(qp.get("intent_entry_id") or "") == "direct_preference_optimization":
        return _pick_dpo_start_here_strict(
            papers,
            qp,
            exclude_ids,
            max_items=max_items,
            graph_edges=graph_edges,
        )

    from .topical_ranking import strict_start_here_score

    baselines: list[tuple[float, str]] = []
    learners: list[tuple[float, str]] = []

    for p in papers:
        if p.get("graph_noise"):
            continue
        pid = str(p.get("id") or "")
        if not pid or pid in exclude_ids:
            continue
        attach_intent_category(p, qp)
        role = attach_paper_role(p, qp)
        if not start_here_role_allowed(role):
            continue
        if role == ROLE_CANONICAL_ANCHOR:
            continue
        score = strict_start_here_score(p, qp)
        if score < 0:
            if role == ROLE_ALIGNMENT_BASELINE:
                score = query_centrality_score(p, qp) * 2.0
            else:
                continue
        if role == ROLE_ALIGNMENT_BASELINE:
            baselines.append((score, pid))
        else:
            learners.append((score, pid))

    picked: list[str] = []
    baselines.sort(reverse=True)
    learners.sort(reverse=True)
    if baselines:
        picked.append(baselines[0][1])
    for _, pid in learners:
        if len(picked) >= max_items:
            break
        if pid not in picked:
            picked.append(pid)
    return picked[:max_items]


__all__ = [
    "ADJACENT_ALIGNMENT",
    "EXACT_METHOD_MATCH",
    "FOUNDATIONAL_CATEGORIES",
    "FOUNDATIONAL_ROLES",
    "GENERIC_RLHF",
    "ROLE_ALIGNMENT_BASELINE",
    "ROLE_CANONICAL_ANCHOR",
    "ROLE_CORE_VARIANT",
    "ROLE_CROSS_DOMAIN",
    "ROLE_EXPLORATORY",
    "ROLE_METHOD_EXTENSION",
    "START_HERE_CATEGORIES",
    "START_HERE_ROLES",
    "STRONG_METHOD_VARIANT",
    "UNRELATED_NOISE",
    "attach_intent_category",
    "attach_paper_role",
    "branch_member_allowed",
    "build_foundational_paper_entries",
    "classify_paper_intent_category",
    "classify_paper_role",
    "compact_explanation",
    "foundational_allowed",
    "foundational_role_allowed",
    "dpo_start_here_strict_eligible",
    "pick_start_here_ids",
    "query_centrality_score",
    "rich_paper_description",
    "start_here_allowed",
    "start_here_role_allowed",
]
