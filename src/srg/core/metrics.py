"""Reading-list quality metrics for automated evaluation."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from .datasets import canonical_titles_for_query

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower().strip())


def _tokens(s: str) -> set[str]:
    return set(_TOKEN_RE.findall(_norm(s)))


def title_similarity(a: str, b: str) -> float:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    if na in nb or nb in na:
        return 0.95
    return SequenceMatcher(None, na, nb).ratio()


def _collect_titles(result: dict[str, Any]) -> list[str]:
    titles: list[str] = []
    for block in ("foundational_papers",):
        for row in result.get(block) or []:
            if isinstance(row, dict):
                t = (row.get("title") or "").strip()
                if t:
                    titles.append(t)
    for br in result.get("branches") or []:
        if not isinstance(br, dict):
            continue
        for t in br.get("member_titles") or []:
            if t:
                titles.append(str(t))
    for c in result.get("raw_candidates") or []:
        if isinstance(c, dict):
            t = (c.get("title") or "").strip()
            if t:
                titles.append(t)
    return titles


def _query_tokens(query: str) -> set[str]:
    return _tokens(query)


def intent_precision(result: dict[str, Any]) -> float:
    """Share of surfaced titles whose tokens overlap the query intent."""
    query = str(result.get("query") or "")
    qt = _query_tokens(query)
    if not qt:
        return 0.0
    titles = _collect_titles(result)[:40]
    if not titles:
        return 0.0
    hits = 0
    for t in titles:
        tt = _tokens(t)
        overlap = len(qt & tt) / max(1, len(qt))
        if overlap >= 0.25 or any(tok in _norm(t) for tok in qt if len(tok) > 3):
            hits += 1
    return hits / len(titles)


def canonical_hit_rate(result: dict[str, Any]) -> float:
    """Whether any gold canonical title appears in foundational or top raw candidates."""
    gold = canonical_titles_for_query(str(result.get("query") or ""))
    if not gold:
        return 1.0  # no gold defined — neutral
    pool: list[str] = []
    for row in result.get("foundational_papers") or []:
        if isinstance(row, dict):
            pool.append(str(row.get("title") or ""))
    for c in (result.get("raw_candidates") or [])[:15]:
        if isinstance(c, dict):
            pool.append(str(c.get("title") or ""))
    for g in gold:
        for t in pool:
            if title_similarity(g, t) >= 0.72:
                return 1.0
    return 0.0


def foundational_quality(result: dict[str, Any]) -> float:
    """Foundational set: presence, non-empty explanations, no duplicate titles."""
    found = result.get("foundational_papers") or []
    if not found:
        return 0.5
    scores: list[float] = []
    seen: set[str] = set()
    for row in found:
        if not isinstance(row, dict):
            continue
        title = _norm(str(row.get("title") or ""))
        expl = str(row.get("why_included") or row.get("description") or "").strip()
        dup_penalty = 0.0 if title not in seen else 0.4
        seen.add(title)
        expl_score = 1.0 if len(expl) >= 24 else 0.3
        role = str(row.get("role") or "")
        role_score = 1.0 if role and role != "exploratory" else 0.6
        scores.append(max(0.0, min(1.0, 0.5 * expl_score + 0.5 * role_score - dup_penalty)))
    return sum(scores) / len(scores) if scores else 0.0


def noise_ratio(result: dict[str, Any]) -> float:
    """Fraction of top raw candidates that look off-topic (lower is better)."""
    query = str(result.get("query") or "")
    qt = _query_tokens(query)
    cands = result.get("raw_candidates") or []
    if not cands or not qt:
        return 0.0
    noisy = 0
    for c in cands[:20]:
        if not isinstance(c, dict):
            continue
        tt = _tokens(str(c.get("title") or ""))
        overlap = len(qt & tt) / max(1, len(qt))
        sem = float(c.get("semantic_score", 0.0) or 0.0)
        if overlap < 0.12 and sem < 0.35:
            noisy += 1
    return noisy / min(20, len(cands))


def branch_coherence(result: dict[str, Any]) -> float:
    """Average within-branch title token Jaccard."""
    branches = result.get("branches") or []
    if not branches:
        return 0.0
    scores: list[float] = []
    for br in branches:
        if not isinstance(br, dict):
            continue
        titles = [str(t) for t in (br.get("member_titles") or []) if t]
        if len(titles) < 2:
            scores.append(0.5)
            continue
        toks = [_tokens(t) for t in titles]
        pairs = 0
        acc = 0.0
        for i in range(len(toks)):
            for j in range(i + 1, len(toks)):
                u = toks[i] | toks[j]
                inter = toks[i] & toks[j]
                acc += len(inter) / max(1, len(u))
                pairs += 1
        scores.append(acc / pairs if pairs else 0.5)
    return sum(scores) / len(scores) if scores else 0.0


def diversity_score(result: dict[str, Any]) -> float:
    """Branch count + year spread in raw candidates (higher = more diverse)."""
    branches = result.get("branches") or []
    years = [
        int(c.get("year"))
        for c in (result.get("raw_candidates") or [])
        if isinstance(c, dict) and c.get("year")
    ]
    year_span = (max(years) - min(years)) if len(years) >= 2 else 0
    branch_part = min(1.0, len(branches) / 4.0)
    year_part = min(1.0, year_span / 12.0)
    return 0.6 * branch_part + 0.4 * year_part


def readability_score(result: dict[str, Any]) -> float:
    """Explanation length and absence of boilerplate phrases."""
    bad = ("graph expansion", "citation neighborhood", "may be incomplete")
    texts: list[str] = []
    for row in result.get("foundational_papers") or []:
        if isinstance(row, dict):
            texts.append(str(row.get("why_included") or row.get("description") or ""))
    for br in result.get("branches") or []:
        if isinstance(br, dict):
            texts.append(str(br.get("why_included") or ""))
    if not texts:
        return 0.5
    ok = 0
    for t in texts:
        low = t.lower()
        if any(b in low for b in bad):
            continue
        if 20 <= len(t) <= 280:
            ok += 1
    return ok / len(texts)


def hallucination_heuristic(result: dict[str, Any]) -> float:
    """
    Penalize empty IDs, missing titles, or contradictory stats (0 = clean, 1 = bad).
    """
    issues = 0
    checks = 0
    for row in result.get("foundational_papers") or []:
        if not isinstance(row, dict):
            continue
        checks += 1
        if not str(row.get("paper_id") or row.get("id") or "").strip():
            issues += 1
        if not str(row.get("title") or "").strip():
            issues += 1
    stats = result.get("stats") or {}
    if int(stats.get("paper_count") or 0) == 0 and (result.get("raw_candidates") or []):
        issues += 2
        checks += 2
    return issues / max(1, checks) if checks else 0.0


def compute_all_metrics(result: dict[str, Any]) -> dict[str, float]:
    noise = noise_ratio(result)
    return {
        "intent_precision": round(intent_precision(result), 4),
        "canonical_hit_rate": round(canonical_hit_rate(result), 4),
        "foundational_quality": round(foundational_quality(result), 4),
        "noise_ratio": round(noise, 4),
        "branch_coherence": round(branch_coherence(result), 4),
        "diversity": round(diversity_score(result), 4),
        "readability": round(readability_score(result), 4),
        "hallucination_risk": round(hallucination_heuristic(result), 4),
        "composite": round(
            0.22 * intent_precision(result)
            + 0.18 * canonical_hit_rate(result)
            + 0.18 * foundational_quality(result)
            + 0.12 * (1.0 - noise)
            + 0.12 * branch_coherence(result)
            + 0.08 * diversity_score(result)
            + 0.06 * readability_score(result)
            + 0.04 * (1.0 - hallucination_heuristic(result)),
            4,
        ),
    }
