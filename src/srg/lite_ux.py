"""SRG Lite presentation layer — single source of truth for curated UX fields on pipeline payloads.

Keep ``payload["lite_ux"]`` disciplined: only documented top-level keys
(``start_here``, ``branches``, ``reading_paths``, ``retrieval_health``, ``insights``).
Do not use ad-hoc keys like ``lite_ux["random_feature"]`` — extend via named sections above.
"""

from __future__ import annotations

import difflib
import re
from collections import defaultdict
from typing import Any

from .explainability import why_it_matters_one_line
from .lite_ux_extended import (
    build_insights,
    build_reading_paths,
    build_retrieval_health,
    enrich_branches_why_included,
)
from .synthesis_export import paper_reading_order_score

_LITE_UX_VERSION = 2

# Display order for Start Here role tags (short labels; no long prose).
_START_HERE_ROLE_TAG_ORDER: tuple[str, ...] = (
    "Foundational",
    "Survey",
    "Recent",
    "Bridge",
    "Canonical",
    "Cross-domain",
)

# Query-conditioned canonicality is on ~[0,1]; high values → core survey / field anchors.
_CANONICAL_TAG_MIN_SCORE = 0.62

_BRANCH_LABEL_STOP = frozenset(
    (
        "the and for with from that this using based paper method methods approach model study "
        "learning data new results analysis work via into over per among between its use "
        "towards toward large deep neural networks training proposed novel efficient "
        "improving improved performance evaluation experimental experiments survey review "
        "overview introduction"
    ).split()
)


def _fmt_domain(domain: str | None) -> str:
    d = (domain or "").strip() or "general"
    return d.replace("_", " ").title()


def _papers_by_domain_branch(
    papers: list[dict[str, Any]],
    *,
    intent_mode_v2: str | None,
    query_text: str | None,
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
            key=lambda p: paper_reading_order_score(p, intent_mode_v2=intent_mode_v2, query_text=query_text),
            reverse=True,
        )
    return dict(sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0])))


def _partition_query_aware_chunks(
    papers: list[dict[str, Any]],
    query_profile: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    """Same chunk sizes as ``group_papers_query_aware_topics`` (ranked slice, fixed group count)."""
    qp = query_profile or {}
    iv2 = (qp.get("intent_mode_v2") or "").strip() or None
    qt_raw = (qp.get("query_text") or "").strip()
    ranked = sorted(
        papers,
        key=lambda p: paper_reading_order_score(p, intent_mode_v2=iv2, query_text=qt_raw),
        reverse=True,
    )
    n = len(ranked)
    if not n:
        return []
    n_groups = min(4, max(1, (n + 5) // 7))
    size = max(1, (n + n_groups - 1) // n_groups)
    chunks: list[list[dict[str, Any]]] = []
    for g in range(n_groups):
        chunk = ranked[g * size : (g + 1) * size] if g < n_groups - 1 else ranked[g * size :]
        if chunk:
            chunks.append(list(chunk))
    return chunks


def _query_token_blocklist(qp: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    qt = (qp.get("query_text") or "").strip().lower()
    for w in re.findall(r"[a-z][a-z0-9\-]{2,}", qt):
        out.add(w)
    for t in qp.get("query_terms") or []:
        for w in re.findall(r"[a-z][a-z0-9\-]{2,}", str(t).lower()):
            out.add(w)
    return out


def _literature_branch_label(central: dict[str, Any], qp: dict[str, Any]) -> str:
    """Representative phrase from the central paper title; strip query terms and generic tokens."""
    block = _query_token_blocklist(qp)
    title = (central.get("title") or "").strip()
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", title)
    kept: list[str] = []
    for w in words:
        wl = w.lower()
        if wl in _BRANCH_LABEL_STOP or wl in block:
            continue
        kept.append(w)
    if len(kept) >= 3:
        phrase = " ".join(t.title() for t in kept[:8])
        if len(phrase) > 72:
            phrase = phrase[:72].rsplit(" ", 1)[0] + "…"
        return phrase
    short = title[:70] if title else ""
    if len(short) == 70 and " " in short:
        short = short.rsplit(" ", 1)[0] + "…"
    return short.strip() or "Literature branch"


def _branch_summary(label: str) -> str:
    lab = (label or "").strip()
    if not lab:
        return "Papers grouped by shared citation neighborhood in this map."
    tail = lab[0].lower() + lab[1:] if len(lab) > 1 else lab.lower()
    return (
        f"This branch focuses on {tail} — "
        "clustered from citation structure and reading-order signals for your query."
    )


def _norm_title_blob(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def _titles_near_duplicate(a: str, b: str) -> bool:
    na, nb = _norm_title_blob(a), _norm_title_blob(b)
    if not na or not nb:
        return False
    if difflib.SequenceMatcher(None, na, nb).ratio() >= 0.82:
        return True
    ta = set(re.findall(r"[a-z][a-z0-9\-]{3,}", na))
    tb = set(re.findall(r"[a-z][a-z0-9\-]{3,}", nb))
    if not ta or not tb:
        return False
    j = len(ta & tb) / max(1, len(ta | tb))
    return j >= 0.62


def _central_paper(cluster: list[dict[str, Any]], *, iv2: str | None, qt: str) -> dict[str, Any]:
    return max(
        cluster,
        key=lambda p: (
            paper_reading_order_score(p, intent_mode_v2=iv2, query_text=qt),
            float(p.get("citation_count", 0) or 0),
        ),
    )


def _build_branch_objects(
    papers: list[dict[str, Any]],
    qp: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """
    Returns (branches, paper_id -> branch_id) for downstream start-here diversification.
    """
    clean = [p for p in papers if not p.get("graph_noise")]
    iv2 = (qp.get("intent_mode_v2") or "").strip() or None
    qt_raw = (qp.get("query_text") or "").strip()

    chunks = _partition_query_aware_chunks(clean, qp)
    use_domain = len(chunks) < 2
    if use_domain:
        domain_map = _papers_by_domain_branch(clean, intent_mode_v2=iv2, query_text=qt_raw or None)
        groups: list[tuple[str, list[dict[str, Any]]]] = list(domain_map.items())
    else:
        groups = [("", ch) for ch in chunks]

    branches: list[dict[str, Any]] = []
    pid_to_branch: dict[str, str] = {}
    used_labels: set[str] = set()

    for idx, (_domain_name, plist) in enumerate(groups):
        if not plist:
            continue
        central = _central_paper(plist, iv2=iv2, qt=qt_raw)
        if use_domain:
            label = _domain_name
            summary = (
                f"Papers in the {_domain_name} domain bucket — shown when the map has few separable "
                "citation communities for this query."
            )
        else:
            label = _literature_branch_label(central, qp)
            summary = _branch_summary(label)
        base = label
        n = 1
        while label.lower() in used_labels:
            label = f"{base} ({n})"
            n += 1
        used_labels.add(label.lower())
        bid = f"lite-branch-{idx}"
        sorted_members = sorted(
            plist,
            key=lambda p: paper_reading_order_score(p, intent_mode_v2=iv2, query_text=qt_raw),
            reverse=True,
        )
        for p in sorted_members:
            pid = str(p.get("id") or "")
            if pid:
                pid_to_branch[pid] = bid
        branches.append(
            {
                "id": bid,
                "label": label,
                "summary": summary,
                "central_paper_id": central.get("id"),
                "paper_ids": [str(p.get("id")) for p in sorted_members if p.get("id")],
                "member_count": len(sorted_members),
            }
        )

    branches.sort(key=lambda b: (-int(b.get("member_count") or 0), str(b.get("label") or "")))
    return branches, pid_to_branch


def _query_inferred_domains_lower(qp: dict[str, Any] | None) -> frozenset[str]:
    if not qp:
        return frozenset()
    raw = qp.get("query_inferred_domains")
    if isinstance(raw, (list, tuple, set)):
        return frozenset(str(d).strip().lower() for d in raw if str(d).strip())
    return frozenset()


def _paper_domain_lower(p: dict[str, Any]) -> str:
    return str(p.get("domain") or "").strip().lower()


def _start_here_tags(p: dict[str, Any], qp: dict[str, Any] | None = None) -> list[str]:
    """Role tags for Start Here / paper cards — explain *why* this paper is surfaced, without long text."""
    chosen: set[str] = set()
    if p.get("is_foundational_hub") and p.get("foundational_eligible", True):
        chosen.add("Foundational")
    ttl = (p.get("title") or "").lower()
    if any(k in ttl for k in ("survey", " literature review", " review of", "taxonomy", "overview of")):
        chosen.add("Survey")
    y = p.get("year")
    try:
        if y is not None and int(y) >= 2021:
            chosen.add("Recent")
    except (TypeError, ValueError):
        pass
    if p.get("is_missing_link_candidate"):
        chosen.add("Bridge")
    try:
        cs = float(p.get("canonicality_score") or 0.0)
    except (TypeError, ValueError):
        cs = 0.0
    if cs >= _CANONICAL_TAG_MIN_SCORE:
        chosen.add("Canonical")
    qdom = _query_inferred_domains_lower(qp)
    pd = _paper_domain_lower(p)
    if qdom and pd and pd not in qdom:
        chosen.add("Cross-domain")
    return [t for t in _START_HERE_ROLE_TAG_ORDER if t in chosen]


def _candidate_paper_order(
    papers: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
    qp: dict[str, Any],
) -> list[str]:
    """Stable priority: discovery recommendations first, then reading-order fill."""
    iv2 = (qp.get("intent_mode_v2") or "").strip() or None
    qt_raw = (qp.get("query_text") or "").strip()
    by_id = {str(p["id"]): p for p in papers if p.get("id")}
    ordered: list[str] = []
    seen: set[str] = set()
    for r in recommendations:
        pid = str(r.get("paper_id") or "").strip()
        if not pid or pid in seen:
            continue
        ordered.append(pid)
        seen.add(pid)
    ranked = sorted(
        [p for p in papers if not p.get("graph_noise") and p.get("id")],
        key=lambda p: paper_reading_order_score(p, intent_mode_v2=iv2, query_text=qt_raw),
        reverse=True,
    )
    for p in ranked:
        pid = str(p["id"])
        if pid not in seen:
            ordered.append(pid)
            seen.add(pid)
    return [pid for pid in ordered if pid in by_id]


def _pick_start_here_ids(
    papers: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
    qp: dict[str, Any],
    pid_to_branch: dict[str, str],
    *,
    max_items: int = 5,
) -> list[str]:
    by_id = {str(p["id"]): p for p in papers if p.get("id")}
    cand_order = _candidate_paper_order(papers, recommendations, qp)
    by_branch: dict[str, list[str]] = defaultdict(list)
    for pid in cand_order:
        p = by_id.get(pid)
        if not p:
            continue
        ttl = str(p.get("title") or "")
        if ttl.startswith("Pending metadata"):
            continue
        bid = pid_to_branch.get(pid, "_unassigned")
        by_branch[bid].append(pid)

    picked: list[str] = []
    titles: list[str] = []

    def titles_conflict(title: str) -> bool:
        return any(_titles_near_duplicate(title, t) for t in titles)

    branch_keys = sorted(by_branch.keys(), key=lambda k: -len(by_branch[k]))
    while len(picked) < max_items and branch_keys:
        progressed = False
        for bk in branch_keys:
            lst = by_branch[bk]
            while lst:
                pid = lst.pop(0)
                p = by_id.get(pid)
                if not p:
                    continue
                t = str(p.get("title") or "")
                if titles_conflict(t):
                    continue
                picked.append(pid)
                titles.append(t)
                progressed = True
                break
            if len(picked) >= max_items:
                break
        if not progressed:
            break

    if len(picked) < max_items:
        for pid in cand_order:
            if len(picked) >= max_items:
                break
            if pid in picked:
                continue
            p = by_id.get(pid)
            if not p:
                continue
            t = str(p.get("title") or "")
            if t.startswith("Pending metadata"):
                continue
            if titles_conflict(t):
                continue
            picked.append(pid)
            titles.append(t)

    return picked[:max_items]


def _start_here_entries(
    papers: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
    qp: dict[str, Any],
    pid_to_branch: dict[str, str],
    branch_list: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {str(p["id"]): p for p in papers if p.get("id")}
    branch_label_by_id = {str(b["id"]): str(b.get("label") or "") for b in branch_list}
    bid_by_pid = pid_to_branch
    qt_why = (qp.get("query_text") or "").strip()
    intent_for_why = str(qp.get("intent_mode_v2") or qp.get("query_intent") or "exploratory")

    pids = _pick_start_here_ids(papers, recommendations, qp, pid_to_branch, max_items=5)
    out: list[dict[str, Any]] = []
    for pid in pids:
        p = by_id.get(pid)
        if not p:
            continue
        bid = bid_by_pid.get(pid, "")
        branch_label = branch_label_by_id.get(bid, "")
        one_line = why_it_matters_one_line(p, query_text=qt_why, intent_label=intent_for_why)
        yr = p.get("year")
        year_out: int | None
        try:
            year_out = int(yr) if yr is not None else None
        except (TypeError, ValueError):
            year_out = None
        out.append(
            {
                "paper_id": pid,
                "title": (p.get("title") or pid).strip(),
                "year": year_out,
                "one_line": one_line,
                "tags": _start_here_tags(p, qp),
                "branch_id": bid or None,
                "branch_label": branch_label or None,
            }
        )
    return out


def build_lite_ux_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Build the SRG Lite UX bundle from a completed pipeline payload (graph + discovery).

    Consumers: Streamlit lite UI, markdown export, future API. Do not re-derive these lists ad hoc.

    Reserved top-level keys (extend only along these lines): ``start_here``, ``branches``,
    ``reading_paths``, ``retrieval_health``, ``insights``. Omit unknown keys until defined.
    """
    papers = list(payload.get("papers") or [])
    qp = payload.get("query_profile") or {}
    recs = (payload.get("discovery") or {}).get("recommendations") or []
    if not isinstance(recs, list):
        recs = []

    branches, pid_to_branch = _build_branch_objects(papers, qp)
    enrich_branches_why_included(branches, papers, qp)
    start_here = _start_here_entries(papers, recs, qp, pid_to_branch, branches)
    start_ids = [str(e["paper_id"]) for e in start_here if isinstance(e, dict) and e.get("paper_id")]
    reading_paths = build_reading_paths(payload, start_ids, pid_to_branch)
    retrieval_health = build_retrieval_health(payload, branches=branches)
    insights = build_insights(payload, branches, pid_to_branch)

    return {
        "version": _LITE_UX_VERSION,
        "start_here": start_here,
        "branches": branches,
        "reading_paths": reading_paths,
        "retrieval_health": retrieval_health,
        "insights": insights,
    }


def start_here_tags_for_paper(p: dict[str, Any], query_profile: dict[str, Any] | None = None) -> list[str]:
    """Public wrapper — same tag rules as ``start_here`` entries in ``build_lite_ux_payload``."""
    return _start_here_tags(p, query_profile)


__all__ = ["build_lite_ux_payload", "start_here_tags_for_paper"]
