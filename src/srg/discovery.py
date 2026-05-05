from __future__ import annotations

import difflib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .graph import GraphSnapshot
from .schema import PaperRecord


@dataclass(frozen=True, slots=True)
class RelationParams:
    """ACCE-style graph relation weights — pure citation mode (no text/topic semantics in scoring)."""

    w_direct: float = 2.0
    w_coupling: float = 0.6
    w_cocitation: float = 0.8
    w_topic: float = 0.0


def _hub_norm(deg_a: int, deg_b: int) -> float:
    """sqrt(deg_a * deg_b) with floors — hub-dampening denominator (ACCE robust relatedness)."""
    return max(math.sqrt(max(deg_a, 1) * max(deg_b, 1)), 1e-9)


def robust_pair_term(raw: float, deg_a: int, deg_b: int) -> float:
    if raw <= 0:
        return 0.0
    return float(raw) / _hub_norm(deg_a, deg_b)


def _build_cited_by(graph: GraphSnapshot) -> dict[str, set[str]]:
    cited_by: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges:
        cited_by[edge.target_paper_id].add(edge.source_paper_id)
    return dict(cited_by)


def _graph_degrees(graph: GraphSnapshot) -> dict[str, int]:
    out_d = Counter(e.source_paper_id for e in graph.edges)
    in_d = Counter(e.target_paper_id for e in graph.edges)
    return {pid: int(out_d[pid] + in_d[pid]) for pid in graph.nodes}


def _topic_probe_score(paper: PaperRecord) -> float:
    for prov in paper.provenance:
        if prov.source == "topic_probe":
            raw = prov.raw or {}
            return float(raw.get("score") or 0.0)
    return 0.0


def _best_title_similarity(graph: GraphSnapshot, anchor_ids: set[str], candidate_id: str) -> float:
    cand = graph.nodes.get(candidate_id)
    if not cand or not cand.title:
        return 0.0
    c_low = cand.title.lower().strip()
    best = 0.0
    for uid in anchor_ids:
        up = graph.nodes.get(uid)
        if not up or not up.title:
            continue
        best = max(
            best,
            difflib.SequenceMatcher(a=c_low, b=up.title.lower().strip()).ratio(),
        )
    return best


def score_expand_relations(
    graph: GraphSnapshot,
    anchor_paper_ids: set[str],
    params: RelationParams | None = None,
) -> dict[str, tuple[float, str, dict[str, Any]]]:
    """
    Multi-hop expansion score vs. anchor papers (PDF uploads and/or search-first seeds): **only**
    direct citations, bibliographic coupling, and co-citation (hub-normalized). No title similarity or
    topic/PII concept boosts — graph edges remain citation-derived; this score only ranks discovered pool nodes.
    """
    p = params or RelationParams()
    cited_by = _build_cited_by(graph)
    degrees = _graph_degrees(graph)
    edge_pairs: set[tuple[str, str]] = set()
    for e in graph.edges:
        edge_pairs.add((e.source_paper_id, e.target_paper_id))

    def directed(a: str, b: str) -> bool:
        return (a, b) in edge_pairs

    out: dict[str, tuple[float, str, dict[str, Any]]] = {}

    for cid in graph.nodes:
        if cid in anchor_paper_ids:
            continue

        total = 0.0
        direct_hits = 0
        max_coupling_shared = 0
        max_cocitation_shared = 0

        for uid in anchor_paper_ids:
            deg_u = degrees.get(uid, 1)
            deg_c = degrees.get(cid, 1)
            ref_u = graph.adjacency.get(uid, set())
            ref_c = graph.adjacency.get(cid, set())
            cit_u = len(cited_by.get(uid, set()))
            cit_c = len(cited_by.get(cid, set()))

            if directed(uid, cid) or directed(cid, uid):
                direct_hits += 1
                total += p.w_direct * robust_pair_term(1.0, deg_u, deg_c)

            shared_refs = len(ref_u & ref_c)
            max_coupling_shared = max(max_coupling_shared, shared_refs)
            if shared_refs > 0:
                total += p.w_coupling * robust_pair_term(
                    float(shared_refs),
                    max(len(ref_u), 1),
                    max(len(ref_c), 1),
                )

            coc = len(cited_by.get(uid, set()) & cited_by.get(cid, set()))
            max_cocitation_shared = max(max_cocitation_shared, coc)
            if coc > 0:
                total += p.w_cocitation * robust_pair_term(
                    float(coc),
                    max(cit_u, 1),
                    max(cit_c, 1),
                )

        sem = _best_title_similarity(graph, anchor_paper_ids, cid) if anchor_paper_ids else 0.0
        topic_s = _topic_probe_score(graph.nodes[cid])
        if p.w_topic > 0.0:
            if sem >= 0.78:
                total += 0.25 * (sem - 0.75)
            if topic_s >= 0.8:
                total += p.w_topic * topic_s

        diag: dict[str, Any] = {
            "relation_direct_hits": direct_hits,
            "relation_max_coupling": max_coupling_shared,
            "relation_max_cocitation": max_cocitation_shared,
            "relation_title_sim": round(sem, 4),
            "relation_topic_pii": round(topic_s, 4) if topic_s > 0 else 0.0,
        }

        parts: list[str] = []
        if max_cocitation_shared >= 2:
            parts.append(f"{max_cocitation_shared} ortak atıfçı nedeniyle eklendi (co-citation)")
        elif max_coupling_shared >= 2:
            parts.append(f"{max_coupling_shared} ortak referans nedeniyle eklendi (bibliographic coupling)")
        elif direct_hits > 0:
            parts.append("Çapa makale(ler) ile doğrudan atıf ilişkisi")
        if p.w_topic > 0.0 and sem >= 0.82:
            parts.append("Yüksek semantik benzerlik (başlık)")
        if p.w_topic > 0.0 and topic_s >= 0.8:
            parts.append(f"OpenAlex PII konusu eşleşmesi (concepts, skor≈{topic_s:.2f})")
        explanation = " · ".join(parts) if parts else "Genişletme (referans havuzu) ile grafa eklendi"

        if total <= 1e-9 and direct_hits == 0 and max_coupling_shared < 2 and max_cocitation_shared < 2:
            continue
        out[cid] = (round(total, 6), explanation, diag)

    return out


@dataclass(slots=True)
class DiscoveryRecommendation:
    paper_id: str
    score: float
    reason: str


class DomainProfile:
    RECENCY_WINDOWS = {
        "cs": 1.0,
        "biomedical": 2.0,
        "social_sciences": 5.0,
        "physics": 2.0,
        "default": 3.0,
    }

    @classmethod
    def recency_decay(cls, domain: str | None, year: int | None) -> float:
        if year is None:
            return 0.7
        now = datetime.utcnow().year
        age = max(0, now - year)
        window = cls.RECENCY_WINDOWS.get((domain or "default").lower(), cls.RECENCY_WINDOWS["default"])
        return 1.0 / (1.0 + (age / window))


class SuggestionEngine:
    def recommend_missing_links(
        self,
        graph: GraphSnapshot,
        top_k: int = 10,
        foundational_boost: float = 0.0,
    ) -> list[DiscoveryRecommendation]:
        inbound = Counter(edge.target_paper_id for edge in graph.edges)
        outbound = Counter(edge.source_paper_id for edge in graph.edges)
        recs: list[DiscoveryRecommendation] = []
        for paper_id, paper in graph.nodes.items():
            deg = inbound[paper_id] + outbound[paper_id]
            recency = DomainProfile.recency_decay(paper.domain, paper.year)
            metadata_signal = 0.0
            if paper.abstract:
                metadata_signal += 0.25
            if paper.doi or paper.arxiv_id or paper.openalex_id:
                metadata_signal += 0.15
            # Sublinear in degree (hub resistance): high-degree nodes gain less per extra edge.
            centrality_term = math.sqrt(float(max(deg, 0)))
            score = (centrality_term * recency) + (0.2 * recency) + metadata_signal
            if foundational_boost > 0.0 and any(p.source == "foundational_hub" for p in (paper.provenance or [])):
                score += foundational_boost
            recs.append(
                DiscoveryRecommendation(
                    paper_id=paper_id,
                    score=round(score, 3),
                    reason="sqrt(degree) × recency + metadata completeness (hub-resistant)",
                )
            )
        recs.sort(key=lambda r: r.score, reverse=True)
        return recs[:top_k]

    def trend_and_gap(self, graph: GraphSnapshot) -> dict[str, dict[str, int]]:
        timeline: dict[str, int] = defaultdict(int)
        domain_counts: dict[str, int] = defaultdict(int)
        for paper in graph.nodes.values():
            timeline[str(paper.year or "unknown")] += 1
            domain_counts[paper.domain or "unknown"] += 1
        return {"timeline": dict(timeline), "domain_distribution": dict(domain_counts)}
