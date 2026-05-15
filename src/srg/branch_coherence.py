"""
Literature branch coherence — purity scoring, title normalization, and suppression rules.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .flagship_registry import is_resolved_flagship_paper
from .query_intent_gating import attach_intent_category, branch_member_allowed
from .topical_ranking import generic_paper_penalty, semantic_purity_score, topical_importance

# Off-topic themes when query does NOT explicitly target them.
_CONTAMINATION_THEMES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("protein_biomed", ("protein folding", "genomics", "antibody", "histopathology", "clinical trial", "drug discovery")),
    ("cybersecurity", ("cybersecurity", "cyber security", "malware", "intrusion detection", "vulnerability", "bug report")),
    ("recommender", ("recommender system", "collaborative filtering", "sequential recommendation", "click-through")),
    ("software_vuln", ("software vulnerability", "commit message", "security issue", "cve ")),
    ("generic_ir", ("information retrieval benchmark", "learning to rank dataset", "click model")),
)

# Intent → preferred branch labels (heuristic).
_BRANCH_TYPE_LABELS: dict[str, dict[str, str]] = {
    "direct_preference_optimization": {
        "dpo": "DPO Alignment Variants",
        "rlhf": "RLHF Foundations",
        "preference": "Preference Optimization Extensions",
        "default": "LLM Alignment",
    },
    "rlhf_alignment": {
        "default": "RLHF & Human Feedback",
    },
    "transformer_nlp": {
        "default": "Transformer & Attention",
    },
    "graph_neural_networks": {
        "default": "Graph Neural Networks",
    },
    "visual_debugging": {
        "default": "Software Visualization & Debugging",
    },
}

_PURITY_LOW = 0.38
_PURITY_MEDIUM = 0.55


@dataclass(frozen=True)
class BranchPurityReport:
    score: float
    level: str  # high | medium | low
    match_fraction: float
    contamination_fraction: float
    dominant_contamination: str | None
    branch_type: str
    suggested_label: str
    warning: str | None


def _blob(p: dict[str, Any]) -> str:
    return f"{p.get('title', '')} {str(p.get('abstract') or '')[:2000]}".lower()


def _contamination_hit(blob: str, markers: tuple[str, ...]) -> bool:
    return any(m in blob for m in markers)


def _query_targets_theme(qp: dict[str, Any], theme_id: str) -> bool:
    """Skip contamination penalty when the query explicitly targets that domain."""
    blob = f"{qp.get('query_text') or ''} {qp.get('intent_entry_id') or ''}".lower()
    enrich = " ".join(str(t) for t in (qp.get("intent_enrichment_terms") or [])).lower()
    blob = f"{blob} {enrich}"
    theme_queries: dict[str, tuple[str, ...]] = {
        "cybersecurity": ("cyber", "security", "malware", "vulnerability"),
        "protein_biomed": ("protein", "genomic", "clinical", "drug", "biomed"),
        "recommender": ("recommend", "collaborative filtering"),
        "software_vuln": ("vulnerability", "cve", "commit", "software security"),
        "generic_ir": ("information retrieval", "learning to rank", "click-through"),
    }
    keys = theme_queries.get(theme_id, ())
    return any(k in blob for k in keys)


def _paper_matches_core_intent(p: dict[str, Any], qp: dict[str, Any]) -> bool:
    cat = attach_intent_category(p, qp)
    strict = bool(qp.get("intent_entry_id"))
    if not branch_member_allowed(cat, strict=strict):
        return False
    if is_resolved_flagship_paper(p, str(qp.get("intent_entry_id") or "") or None):
        return True
    if semantic_purity_score(p, qp) >= 0.42:
        return True
    q = (qp.get("query_text") or "").strip().lower()
    if q:
        title = (p.get("title") or "").lower()
        qtok = [t for t in q.split() if len(t) > 3]
        if qtok and sum(1 for t in qtok if t in title) >= max(1, len(qtok) // 2):
            return True
    return False


def classify_branch_type(members: list[dict[str, Any]], qp: dict[str, Any]) -> str:
    """Lightweight branch family from member titles + query intent."""
    entry = str(qp.get("intent_entry_id") or "")
    blob = " ".join(_blob(p) for p in members[:12])
    if entry == "direct_preference_optimization":
        if "direct preference" in blob or " dpo" in blob or blob.startswith("dpo"):
            return "dpo"
        if "rlhf" in blob or "human feedback" in blob or "reward model" in blob:
            return "rlhf"
        if "preference" in blob and "optimization" in blob:
            return "preference"
        return "default"
    if entry == "rlhf_alignment":
        return "rlhf"
    return "default"


def normalize_branch_label(
    members: list[dict[str, Any]],
    qp: dict[str, Any],
    *,
    branch_type: str | None = None,
    fallback: str = "",
) -> str:
    """Human-readable branch title — intent-aware, not raw token soup."""
    entry = str(qp.get("intent_entry_id") or "")
    btype = branch_type or classify_branch_type(members, qp)
    if entry in _BRANCH_TYPE_LABELS:
        fam = _BRANCH_TYPE_LABELS[entry]
        if btype in fam:
            return fam[btype]
        if "default" in fam:
            return fam["default"]

    # Token voting from high-purity members only
    qp_local = dict(qp)
    pure = [p for p in members if semantic_purity_score(p, qp_local) >= 0.4][:8]
    if not pure:
        pure = members[:6]
    c: Counter[str] = Counter()
    block = set(
        re.findall(
            r"[a-z][a-z0-9\-]{2,}",
            (qp.get("query_text") or "").lower(),
        )
    )
    stop = {
        "the", "and", "for", "with", "from", "using", "based", "paper", "method",
        "methods", "approach", "model", "models", "learning", "data", "via", "new",
    }
    for p in pure:
        title = (p.get("title") or "").lower()
        for w in re.findall(r"[a-z][a-z0-9\-]{3,}", title):
            if w in stop or w in block:
                continue
            c[w] += 2
    top = [w for w, _ in c.most_common(4)]
    if len(top) >= 2:
        label = " ".join(w.title() for w in top[:4])
        if len(label) <= 48:
            return label
    # Clean fallback — never return long malformed strings
    fb = (fallback or "").strip()
    if fb and len(fb) <= 56 and len(fb.split()) <= 7:
        return fb
    qt = (qp.get("query_text") or "").strip()
    if len(qt) <= 48:
        return qt.title() if qt else "Related literature"
    return (qt[:45] + "…").title() if qt else "Related literature"


def analyze_branch_purity(
    members: list[dict[str, Any]],
    query_profile: dict[str, Any],
) -> BranchPurityReport:
    """
    Score branch for query alignment; recommend suppress / warn / show.
    """
    if not members:
        return BranchPurityReport(
            score=0.0,
            level="low",
            match_fraction=0.0,
            contamination_fraction=1.0,
            dominant_contamination="empty",
            branch_type="default",
            suggested_label="Related literature",
            warning=None,
        )

    qp = query_profile or {}
    n = len(members)
    match_n = sum(1 for p in members if _paper_matches_core_intent(p, qp))
    match_frac = match_n / n

    contam_counts: Counter[str] = Counter()
    for p in members:
        b = _blob(p)
        for theme_id, markers in _CONTAMINATION_THEMES:
            if _query_targets_theme(qp, theme_id):
                continue
            if _contamination_hit(b, markers):
                contam_counts[theme_id] += 1

    contam_frac = 0.0
    dom_contam: str | None = None
    if contam_counts:
        dom_contam, dom_n = contam_counts.most_common(1)[0]
        contam_frac = dom_n / n

    topical_mean = sum(topical_importance(p, qp) for p in members) / n
    purity_sem = sum(semantic_purity_score(p, qp) for p in members) / n

    score = 0.45 * match_frac + 0.35 * topical_mean + 0.20 * purity_sem
    score -= 0.55 * contam_frac
    if match_frac < 0.25:
        score -= 0.2
    score = max(0.0, min(1.0, score))

    if score >= _PURITY_MEDIUM:
        level = "high"
    elif score >= _PURITY_LOW:
        level = "medium"
    else:
        level = "low"

    btype = classify_branch_type(members, qp)
    label = normalize_branch_label(members, qp, branch_type=btype)
    warning: str | None = None
    if level == "medium":
        warning = "Mixed topical focus — exploratory branch."
    elif level == "low":
        warning = None

    return BranchPurityReport(
        score=round(score, 4),
        level=level,
        match_fraction=round(match_frac, 4),
        contamination_fraction=round(contam_frac, 4),
        dominant_contamination=dom_contam,
        branch_type=btype,
        suggested_label=label,
        warning=warning,
    )


def filter_and_curate_branches(
    branches: list[dict[str, Any]],
    papers: list[dict[str, Any]],
    query_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """Drop low-purity branches; relabel and annotate survivors."""
    by_id = {str(p["id"]): p for p in papers if p.get("id")}
    scored: list[tuple[BranchPurityReport, dict[str, Any], list[dict[str, Any]]]] = []
    for br in branches:
        if not isinstance(br, dict):
            continue
        pids = [str(x) for x in (br.get("paper_ids") or []) if str(x).strip()]
        members = [by_id[pid] for pid in pids if pid in by_id]
        if not members:
            continue
        report = analyze_branch_purity(members, query_profile)
        scored.append((report, br, members))

    out: list[dict[str, Any]] = []
    survivors = [(r, b, m) for r, b, m in scored if r.level != "low"]
    if not survivors and scored:
        survivors = [max(scored, key=lambda t: t[0].score)]

    for report, br, members in survivors:
        nb = dict(br)
        nb["label"] = report.suggested_label
        nb["branch_purity"] = report.score
        nb["branch_purity_level"] = report.level
        nb["branch_type"] = report.branch_type
        if report.warning or report.level == "low":
            warn = report.warning or "Mixed topical focus — exploratory branch."
            nb["purity_warning"] = warn
            summ = (nb.get("summary") or "").strip()
            if summ and warn not in summ:
                nb["summary"] = f"{summ} {warn}"
        strict_br = bool(query_profile.get("intent_entry_id"))
        gated_members = [
            p
            for p in members
            if branch_member_allowed(attach_intent_category(p, query_profile), strict=strict_br)
        ]
        if not gated_members:
            if strict_br:
                continue
            gated_members = members
        members_sorted = sorted(
            gated_members,
            key=lambda p: (
                semantic_purity_score(p, query_profile),
                float(p.get("flagship_match_strength", 0) or 0),
                topical_importance(p, query_profile),
            ),
            reverse=True,
        )
        nb["paper_ids"] = [str(p["id"]) for p in members_sorted if p.get("id")]
        nb["member_count"] = len(nb["paper_ids"])
        nb["central_paper_id"] = nb["paper_ids"][0] if nb["paper_ids"] else nb.get("central_paper_id")
        out.append(nb)
    out.sort(key=lambda b: (-float(b.get("branch_purity", 0)), -int(b.get("member_count", 0))))
    return out


__all__ = [
    "BranchPurityReport",
    "analyze_branch_purity",
    "classify_branch_type",
    "filter_and_curate_branches",
    "normalize_branch_label",
]
