"""Heavy heuristics for ``lite_ux`` — reading paths, retrieval health, branch copy, insights.

Kept separate from ``lite_ux.py`` so the main bundle stays readable. Only imported from ``build_lite_ux_payload``.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any

from .synthesis_export import paper_reading_order_score
from .topical_ranking import branch_purity_score


def _tokenize_meaningful(text: str) -> set[str]:
    return set(re.findall(r"[a-z][a-z0-9\-]{3,}", (text or "").lower()))


def _title_tokens_for_branch_overlap(titles: list[str]) -> set[str]:
    c: Counter[str] = Counter()
    for t in titles:
        for w in _tokenize_meaningful(t):
            c[w] += 1
    return {w for w, n in c.items() if n >= 2}


def _shared_reference_count(members: list[dict[str, Any]]) -> int:
    refs_sets: list[set[str]] = []
    for p in members:
        r = p.get("references")
        if isinstance(r, list) and r:
            refs_sets.append({str(x).lower() for x in r if x})
    if len(refs_sets) < 2:
        return 0
    inter = set.intersection(*refs_sets) if refs_sets else set()
    return len(inter)


def enrich_branches_why_included(
    branches: list[dict[str, Any]],
    papers: list[dict[str, Any]],
    qp: dict[str, Any],
) -> None:
    """Mutates each branch dict with ``why_included`` (template + lexical/citation hints)."""
    by_id = {str(p["id"]): p for p in papers if p.get("id")}
    query_text = (qp.get("query_text") or "").strip()
    qtok = _tokenize_meaningful(query_text)
    for b in branches:
        if not isinstance(b, dict):
            continue
        pids = [str(x) for x in (b.get("paper_ids") or [])]
        members = [by_id[pid] for pid in pids if pid in by_id]
        if not members:
            b["why_included"] = b.get("summary") or ""
            continue
        titles = [str(p.get("title") or "") for p in members]
        branch_vocab = _title_tokens_for_branch_overlap(titles)
        overlap = sorted((branch_vocab & qtok), key=len, reverse=True)[:5]
        shared = _shared_reference_count(members)
        parts: list[str] = []
        if overlap:
            parts.append(f"papers here align with your query on terms like «{', '.join(overlap[:4])}»")
        elif branch_vocab:
            top = sorted(branch_vocab, key=lambda w: (-len(w), w))[:4]
            parts.append(f"this cluster coheres on themes such as «{', '.join(top)}»")
        else:
            parts.append("papers in this branch sit in the same citation neighborhood as your anchors")
        if shared >= 3:
            parts.append(f"they share {shared} reference identifiers in common")
        elif shared >= 1:
            parts.append("they share overlapping reference lists")
        else:
            parts.append("they are linked through citation edges and ranking signals in this map")
        purity = branch_purity_score(members, qp)
        if purity < 0.45:
            parts.append("this branch has mixed topical focus — interpret with caution")
        elif purity >= 0.72:
            parts.append("papers here share strong topical coherence for your query")
        b["branch_purity"] = round(purity, 3)
        b["why_included"] = "This branch appears because " + " and ".join(parts) + "."


def _level_from_score(x: float, *, hi: float, lo: float) -> str:
    if x >= hi:
        return "Strong"
    if x >= lo:
        return "Medium"
    return "Low"


def build_retrieval_health(
    payload: dict[str, Any],
    *,
    branches: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    papers = [p for p in (payload.get("papers") or []) if not p.get("graph_noise")]
    graph = payload.get("graph") or {}
    edges = graph.get("edges") or []
    qp = payload.get("query_profile") or {}
    rq = payload.get("retrieval_quality") or {}
    est = payload.get("evidence_stats") or {}

    n = max(1, len(papers))
    e_ct = len(edges)
    density = e_ct / n

    qt = (qp.get("query_text") or "").strip()
    n_words = len(qt.split()) if qt else 0
    n_terms = len(qp.get("query_terms") or [])
    amb_raw = min(1.0, (1.0 / max(1, n_words)) * 2.2 + (1.0 / max(1, n_terms)) * 1.1)
    if len(qt) > 80:
        amb_raw *= 0.88
    if amb_raw >= 0.55:
        amb = "High"
    elif amb_raw >= 0.35:
        amb = "Medium"
    else:
        amb = "Low"

    cite_lvl = _level_from_score(density, hi=2.5, lo=0.9)

    with_year = sum(1 for p in papers if p.get("year") is not None)
    good_title = sum(
        1
        for p in papers
        if (p.get("title") or "").strip() and not str(p.get("title")).startswith("Pending metadata")
    )
    meta_ratio = (0.5 * (with_year / n) + 0.5 * (good_title / n)) if n else 0.0
    if meta_ratio >= 0.82:
        meta = "Strong"
    elif meta_ratio >= 0.55:
        meta = "Moderate"
    else:
        meta = "Weak"

    lite_branches = branches if branches is not None else (payload.get("lite_ux") or {}).get("branches")
    if isinstance(lite_branches, list) and len(lite_branches) >= 2:
        sizes = [max(1, int(b.get("member_count") or len(b.get("paper_ids") or []))) for b in lite_branches]
        tot = sum(sizes)
        probs = [s / tot for s in sizes]
        ent = -sum(p * math.log(p + 1e-9) for p in probs)
        max_ent = math.log(len(sizes))
        norm = ent / max_ent if max_ent > 0 else 0.5
        if norm >= 0.88:
            bcons = "High"
        elif norm >= 0.55:
            bcons = "Medium"
        else:
            bcons = "Low"
    else:
        bcons = "Medium"

    sparse = n >= 2 and e_ct < 3
    weak_meta = meta == "Weak"
    high_amb = amb == "High"

    suggestions: list[str] = []
    if high_amb or sparse or weak_meta:
        suggestions.append("Try a specific DOI or arXiv id to anchor the map.")
        suggestions.append("Add field-specific terms (e.g. model name, venue, year) to narrow the query.")
        suggestions.append("Paste an exact paper title from Google Scholar or OpenAlex.")

    if high_amb or weak_meta:
        empty_msg = (
            "This query appears ambiguous or weakly represented in the slice shown - "
            "use the suggestions below before trusting distant papers."
        )
    elif sparse:
        empty_msg = (
            "The citation map looks sparse; retrieval may be incomplete or the topic is narrow in metadata."
        )
    else:
        empty_msg = ""

    signals = [
        {"id": "query_ambiguity", "label": "Query ambiguity", "level": amb, "detail": f"~{n_words} words, {n_terms} extracted terms."},
        {"id": "citation_density", "label": "Citation density", "level": cite_lvl, "detail": f"{e_ct} edges across {n} papers (~{density:.2f} edges/paper)."},
        {"id": "metadata_quality", "label": "Metadata quality", "level": meta, "detail": f"{with_year}/{n} with year, titles resolved."},
        {"id": "branch_consistency", "label": "Branch consistency", "level": bcons, "detail": "Entropy of branch sizes (balanced ⇒ clearer branches)."},
    ]

    return {
        "signals": signals,
        "raw": {
            "edge_count": e_ct,
            "node_count": n,
            "density": round(density, 4),
            "ambiguity_score": round(amb_raw, 4),
            "metadata_ratio": round(meta_ratio, 4),
        },
        "empty_state": {
            "active": bool(high_amb or sparse or weak_meta),
            "message": empty_msg,
            "suggestions": suggestions,
        },
        "references_total": int(est.get("references_total", 0) or 0),
        "warnings_echo": list(rq.get("warnings") or [])[:6],
    }


def _citation_adjacency(edges: list[dict[str, Any]]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """source cites target → refs_out[source] includes target; cited_by[target] includes source."""
    refs_out: dict[str, list[str]] = defaultdict(list)
    cited_by: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        if not isinstance(e, dict):
            continue
        s = str(e.get("source") or "")
        t = str(e.get("target") or "")
        if not s or not t:
            continue
        refs_out[s].append(t)
        cited_by[t].append(s)
    return dict(refs_out), dict(cited_by)


def _year_of(pid: str, by_id: dict[str, dict[str, Any]]) -> int | None:
    p = by_id.get(pid)
    if not p:
        return None
    y = p.get("year")
    try:
        return int(y) if y is not None else None
    except (TypeError, ValueError):
        return None


def _build_path_forward_time(
    seed: str,
    by_id: dict[str, dict[str, Any]],
    cited_by: dict[str, list[str]],
    pid_to_branch: dict[str, str],
    *,
    max_len: int = 5,
    iv2: str | None,
    qt: str,
) -> list[str]:
    """Greedy: follow papers that cite ``seed``, prefer newer year + same branch + score."""
    path = [seed]
    cur = seed
    seen = {seed}
    for _ in range(max_len - 1):
        cands = [p for p in cited_by.get(cur, []) if p in by_id and p not in seen]
        if not cands:
            break
        cur_branch = pid_to_branch.get(cur, "")

        def key(pid: str) -> tuple:
            p = by_id[pid]
            y = _year_of(pid, by_id) or 0
            br_pen = 0 if pid_to_branch.get(pid, "") == cur_branch else 1
            sc = paper_reading_order_score(p, intent_mode_v2=iv2, query_text=qt)
            return (-y, br_pen, -sc)

        nxt = min(cands, key=key)
        path.append(nxt)
        seen.add(nxt)
        cur = nxt
    return path


def _pick_oldest_high_signal(by_id: dict[str, dict[str, Any]], ranked_ids: list[str]) -> str | None:
    best_key: tuple[int, int, float] | None = None
    best_pid: str | None = None
    for pid in ranked_ids:
        p = by_id.get(pid)
        if not p:
            continue
        y = _year_of(pid, by_id)
        if y is None:
            continue
        hub = 1 if p.get("is_foundational_hub") else 0
        rel = float(p.get("relation_expand_score") or 0.0)
        key = (y, -hub, -rel)
        if best_key is None or key < best_key:
            best_key = key
            best_pid = pid
    return best_pid or (ranked_ids[0] if ranked_ids else None)


def build_reading_paths(
    payload: dict[str, Any],
    start_here_ids: list[str],
    pid_to_branch: dict[str, str],
) -> list[dict[str, Any]]:
    papers = [p for p in (payload.get("papers") or []) if not p.get("graph_noise")]
    by_id = {str(p["id"]): p for p in papers if p.get("id")}
    graph = payload.get("graph") or {}
    edges = graph.get("edges") or []
    qp = payload.get("query_profile") or {}
    iv2 = (qp.get("intent_mode_v2") or "").strip() or None
    qt = (qp.get("query_text") or "").strip()
    _refs_out, cited_by = _citation_adjacency(edges)

    ranked = sorted(
        papers,
        key=lambda p: paper_reading_order_score(p, intent_mode_v2=iv2, query_text=qt),
        reverse=True,
    )
    ranked_ids = [str(p["id"]) for p in ranked if p.get("id")]

    paths: list[dict[str, Any]] = []
    if not ranked_ids:
        return paths

    seed_old = _pick_oldest_high_signal(by_id, start_here_ids or ranked_ids[:12])
    if seed_old:
        chain = _build_path_forward_time(
            seed_old, by_id, cited_by, pid_to_branch, max_len=5, iv2=iv2, qt=qt
        )
        ys = [y for y in (_year_of(pid, by_id) for pid in chain) if y is not None]
        label = "Foundational → recent line" if ys and max(ys) - min(ys) >= 5 else "Core reading line"
        paths.append(
            {
                "id": "path-core",
                "label": label,
                "rationale": "Follows citation links from an older or central anchor toward work that cites it (time-forward).",
                "paper_ids": chain,
            }
        )

    sh = [pid for pid in (start_here_ids or ranked_ids[:5]) if pid in by_id][:5]
    sh_sorted = sorted(sh, key=lambda pid: (_year_of(pid, by_id) or 0), reverse=True)
    if len(sh_sorted) >= 2:
        paths.append(
            {
                "id": "path-recent",
                "label": "Recent frontier stack",
                "rationale": "Prioritizes newer papers among your curated entry points (same map slice).",
                "paper_ids": sh_sorted[:5],
            }
        )

    by_br: dict[str, list[str]] = defaultdict(list)
    for pid in start_here_ids or ranked_ids[:8]:
        b = pid_to_branch.get(pid, "_")
        by_br[b].append(pid)
    cross: list[str] = []
    for _b, lst in sorted(by_br.items(), key=lambda kv: -len(kv[1])):
        if lst:
            cross.append(lst[0])
        if len(cross) >= 4:
            break
    if len(cross) >= 2:
        paths.append(
            {
                "id": "path-cross-branch",
                "label": "Cross-branch tour",
                "rationale": "One strong paper per literature branch to compare sub-threads without staying in one cluster.",
                "paper_ids": cross[:4],
            }
        )

    return paths[:3]


def build_insights(
    payload: dict[str, Any],
    branches: list[dict[str, Any]],
    pid_to_branch: dict[str, str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    papers = [p for p in (payload.get("papers") or []) if not p.get("graph_noise")]
    by_id = {str(p["id"]): p for p in papers if p.get("id")}
    graph = payload.get("graph") or {}
    edges = graph.get("edges") or []
    edge_pairs = {(str(e.get("source")), str(e.get("target"))) for e in edges if isinstance(e, dict)}

    if len(branches) >= 2:
        def branch_tokens(b: dict[str, Any]) -> set[str]:
            toks: set[str] = set()
            for pid in (b.get("paper_ids") or [])[:12]:
                p = by_id.get(str(pid))
                if p:
                    toks |= _tokenize_meaningful(str(p.get("title") or ""))
            return toks

        best_gap: tuple[float, int, int] | None = None
        for i, bi in enumerate(branches):
            for j, bj in enumerate(branches):
                if i >= j:
                    continue
                ti, tj = branch_tokens(bi), branch_tokens(bj)
                if not ti or not tj:
                    continue
                inter = len(ti & tj)
                union = len(ti | tj)
                jacc = inter / union if union else 1.0
                if jacc > 0.35:
                    continue
                pids_i = [str(x) for x in (bi.get("paper_ids") or [])]
                pids_j = [str(x) for x in (bj.get("paper_ids") or [])]
                bridge = None
                for a in pids_i:
                    for b in pids_j:
                        if (a, b) in edge_pairs or (b, a) in edge_pairs:
                            bridge = (a, b)
                            break
                    if bridge:
                        break
                if not bridge:
                    for pid in pids_i + pids_j:
                        p = by_id.get(pid)
                        if p and p.get("is_missing_link_candidate"):
                            bridge = (pid, pid)
                            break
                if bridge and (best_gap is None or jacc < best_gap[0]):
                    best_gap = (jacc, i, j)
        if best_gap is not None:
            _, i, j = best_gap
            bi, bj = branches[i], branches[j]
            lab_i = str(bi.get("label") or "Branch A")
            lab_j = str(bj.get("label") or "Branch B")
            out.append(
                {
                    "type": "unexpected_connections",
                    "title": "Unexpected connections",
                    "body": (
                        f"«{lab_i}» and «{lab_j}» use different vocabulary on the surface, yet papers in this map "
                        "link them through shared citations or bridge works — worth a short detour if both matter to you."
                    ),
                    "branch_ids": [bi.get("id"), bj.get("id")],
                }
            )

    return out
