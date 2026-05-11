from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict
from datetime import datetime
import difflib
import math
import re
import unicodedata
from typing import Any

import requests

from .canonicality import (
    anchor_proximity_strength_normalized,
    compute_canonicality,
    compute_query_conditioned_canonicality,
    infer_survey_context_flag,
)
from .citation_context import enrich_graph_with_citation_snippets
from .domain_mapper import domain_from_openalex_payload, domain_from_venue_name
from .discovery import RelationParams, SuggestionEngine, score_expand_relations
from .graph_scores import compute_graph_score_bundle
from .explainability import build_why_it_matters, ensure_why_it_matters_top5
from .intent_classifier import QueryIntent, classify_query
from .evaluation import QualityReport, build_quality_report
from .feedback import FeedbackStore
from .graph import CitationGraphBuilder, GraphSnapshot, SemanticMetrics
from .intent_verification import apply_intent_verification_layer
from .ingestion import (
    ArXivClient,
    CacheStore,
    CrossrefClient,
    IngestionOrchestrator,
    OpenAlexClient,
    PDF_MAPPING_QUALITY_THRESHOLD,
    PDFSeedExtractor,
    SemanticScholarClient,
    _normalize_doi,
    arxiv_search_query_from_user_text,
    normalize_arxiv_list_id,
    openalex_fetch_payload_valid,
    openalex_response_first_work,
    year_from_arxiv_list_id,
)
from .schema import Author, FeedbackEvent, PaperRecord, Provenance

# Expand: ingest a wide reference pool, rank with ACCE-style relations, show only top-N in the UI graph slice.
EXPAND_CANDIDATE_POOL_MAX = 200
EXPAND_DISPLAY_TOP_N = 50

# OpenAlex /works search hits in unrelated applied fields (bearings, vibration, etc.) when the user
# asked a general ML query that also has canonical arXiv anchors — skip those hits as seeds.
_OPENALEX_INDUSTRIAL_APPLIED_RE = re.compile(
    r"\b(fault diagnosis|rolling bearing|bearing fault|vibration diagnos|condition monitor(?:ing)?|"
    r"gearbox|motor current|defect detection|rotating machinery|prognostic health)\b",
    re.I,
)


def _titles_match_score(a: str, b: str) -> float:
    """Conservative [0,1] similarity for title alignment (cross-provider OpenAlex fallback)."""
    if not (a or "").strip() or not (b or "").strip():
        return 0.0
    return float(difflib.SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio())


# Alignment lexicon for semantic ranking / IR-noise suppression (multi-stage semantic–graph phase).
_SEMANTIC_ALIGNMENT_LEXICON: tuple[str, ...] = (
    "preference optimization",
    "reward model",
    "rlhf",
    "human feedback",
    "instruction tuning",
    "policy optimization",
    "alignment",
    "preference learning",
    "dpo",
)

from .ranking_diagnostics import lite_v21_observability_payload
from .roadmap_metrics import FAILURE_TAXONOMY
from .semantic_control import (
    aggregate_semantic_control_diagnostics,
    domain_consistency_multiplier,
    hub_suppressed_graph_scores,
    infer_query_domains,
    local_semantic_connectivity,
    normalized_drift_penalty,
    semantic_control_adjustments,
    vocabulary_coherence_score,
)
from .ranking_fusion import (
    RECENCY_TRIPLE_INTENTS,
    apply_hardware_top10_sem_rec_graph_lock,
    compute_final_score,
    fusion_weights_for_intent,
    normalize_query_intent,
)
from .validation import DeduplicationEngine
from .visualization import edge_style


def _paper_to_json(p: PaperRecord) -> dict[str, Any]:
    return {
        "id": p.canonical_id,
        "title": p.title,
        "abstract": p.abstract,
        "year": p.year,
        "venue": p.venue,
        "domain": p.domain,
        "doi": p.doi,
        "arxiv_id": p.arxiv_id,
        "semantic_scholar_id": p.semantic_scholar_id,
        "openalex_id": p.openalex_id,
        "authors": [asdict(a) for a in p.authors],
        "references": list(p.references),
        "provenance": [
            {
                "source": prov.source,
                "source_id": prov.source_id,
                "fetched_at": prov.fetched_at.isoformat(),
                "field_confidence": {k: v.value for k, v in prov.field_confidence.items()},
            }
            for prov in p.provenance
        ],
    }


def _graph_to_json(graph: GraphSnapshot) -> dict[str, Any]:
    nodes = [_paper_to_json(p) for p in graph.nodes.values()]
    edges: list[dict[str, Any]] = []
    for edge in graph.edges:
        style = edge_style(edge)
        edges.append(
            {
                "source": edge.source_paper_id,
                "target": edge.target_paper_id,
                "context": edge.context,
                "context_source": edge.context_source,
                "confidence": edge.confidence.value,
                "style": style,
            }
        )
    return {"nodes": nodes, "edges": edges}


def _report_to_json(report: QualityReport) -> dict[str, Any]:
    return asdict(report)


class SRGApplicationService:
    """UI-facing facade that mimics API-shaped payloads."""

    @staticmethod
    def _anchor_hop_distances(
        node_ids: list[str],
        edges: list[dict[str, Any]],
        anchor_ids: set[str],
    ) -> dict[str, int]:
        """Undirected BFS from seed anchors — citation distance for coherence decay (Phase 2)."""
        adj: dict[str, set[str]] = defaultdict(set)
        for e in edges:
            s, t = e.get("source"), e.get("target")
            if not s or not t:
                continue
            adj[str(s)].add(str(t))
            adj[str(t)].add(str(s))
        dist: dict[str, int] = {}
        dq: deque[str] = deque()
        for a in anchor_ids:
            if a in adj or a in node_ids:
                if a not in dist:
                    dist[a] = 0
                    dq.append(a)
        while dq:
            u = dq.popleft()
            for v in adj.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    dq.append(v)
        return {nid: int(dist.get(nid, 99)) for nid in node_ids}

    @staticmethod
    def _targets_alignment_topic(query_profile: dict[str, Any]) -> bool:
        blob = ((query_profile.get("query_text") or "") + " " + " ".join(query_profile.get("query_terms") or [])).lower()
        keys = (
            "dpo",
            "rlhf",
            "preference",
            "alignment",
            "orpo",
            "reward",
            "human feedback",
            "instruction",
            "constitutional",
            "direct preference",
        )
        return any(k in blob for k in keys)

    @staticmethod
    def _alignment_lexicon_density_semantic(node: dict[str, Any]) -> float:
        blob = f"{node.get('title', '')} {node.get('abstract', '')}".lower()
        hits = sum(1 for t in _SEMANTIC_ALIGNMENT_LEXICON if t in blob)
        return hits / max(len(_SEMANTIC_ALIGNMENT_LEXICON), 1)

    def _semantic_fit_score(self, node: dict[str, Any], query_profile: dict[str, Any]) -> float:
        """
        Lightweight semantic relevance [0,1]: token overlap + title similarity (BM25/cross-encoder surrogate).
        Used when lite_semantic_graph_ranking — semantic intent dominates over raw graph hubs.
        """
        qt = (query_profile.get("query_text") or "").strip().lower()
        title = (node.get("title") or "").lower()
        abstract = (node.get("abstract") or "").lower()
        terms = [str(t).lower() for t in (query_profile.get("query_terms") or []) if len(str(t).strip()) > 2]
        blob = f"{title} {abstract[:2200]}"
        if not qt:
            return 0.45
        overlap = sum(1 for t in terms if t in blob)
        tok = min(1.0, overlap / max(len(terms), 1) * 1.75)
        seq = difflib.SequenceMatcher(a=qt[:160], b=title[:220]).ratio()
        score = 0.4 * tok + 0.48 * seq
        if any(x in qt for x in ("dpo", "direct preference", "preference optimization")):
            if "direct preference optimization" in title:
                score += 0.16
            if "2305.18290" in str(node.get("id", "")).lower():
                score += 0.12
        if self._targets_alignment_topic(query_profile):
            noise = (
                "learning to rank",
                "clickthrough",
                "search engine",
                "incomplete block design",
                "information retrieval",
                "optimizing search engines",
            )
            if any(p in title for p in noise) and self._alignment_lexicon_density_semantic(node) < 0.07:
                score *= 0.32
        return max(0.0, min(1.0, score))

    def __init__(self, cache_db: str = "srg_cache.db", feedback_db: str = "srg_feedback.db") -> None:
        self.feedback_store = FeedbackStore(db_path=feedback_db)
        self.orchestrator = IngestionOrchestrator(
            clients=[OpenAlexClient(), CrossrefClient(), ArXivClient(), SemanticScholarClient()],
            cache=CacheStore(db_path=cache_db),
        )
        self.deduper = DeduplicationEngine()
        self.graph_builder = CitationGraphBuilder()
        self.semantic_metrics = SemanticMetrics()
        self.discovery = SuggestionEngine()
        self.pdf_extractor = PDFSeedExtractor()
        self.provider_aliases = {
            "doi": "crossref",
            "crossref": "crossref",
            "openalex": "openalex",
            "arxiv": "arxiv",
        }
        self._cached_pii_concept_tail: str | None = None
        self._last_seed_selection_notes: list[str] = []

    def fetch_metadata_for_seed(self, provider: str, seed_id: str) -> dict[str, Any] | None:
        """Return API title + authors for a DOI / arXiv / OpenAlex id (PDF verification path; never uses PDF text)."""
        from .ingestion import fetch_public_metadata_for_seed

        return fetch_public_metadata_for_seed(provider, seed_id)

    @staticmethod
    def _is_pii_concept_query(q: str) -> bool:
        low = (q or "").lower()
        return (
            "pii" in low
            or "personally identifiable" in low
            or "personal identifiable information" in low
            or "personally identifiable information" in low
        )

    def _openalex_max_concept_score_for_id(self, concepts: list[Any] | None, concept_short_id: str) -> float:
        best = 0.0
        for item in concepts or []:
            if not isinstance(item, dict):
                continue
            iid = (item.get("id") or "").rsplit("/", maxsplit=1)[-1]
            if iid == concept_short_id:
                best = max(best, float(item.get("score") or 0.0))
        return best

    def _openalex_pii_concept_seed_hits(self, diagnostics: list[str]) -> list[dict[str, Any]]:
        """OpenAlex concepts: papers strongly tagged with a Personal Identifiable Information–like concept."""
        oa = OpenAlexClient()
        try:
            self.orchestrator.budget.acquire("openalex")
            concepts = oa._get("/concepts", params={"search": "personal identifiable information", "per-page": 12})
        except requests.RequestException as exc:
            diagnostics.append(f"PII concept search (OpenAlex /concepts) failed: {exc}")
            return []
        cid: str | None = None
        for c in concepts.get("results", []) or []:
            if not isinstance(c, dict):
                continue
            dn = (c.get("display_name") or "").lower()
            if "personal" in dn and "identif" in dn:
                cid = (c.get("id") or "").rsplit("/", maxsplit=1)[-1]
                self._cached_pii_concept_tail = cid
                break
        if not cid:
            diagnostics.append(
                "PII concept: OpenAlex returned no concept whose display_name matches "
                "'personal … identif…' (try a different query wording)."
            )
            return []
        try:
            self.orchestrator.budget.acquire("openalex")
            works = oa._get("/works", params={"filter": f"concepts.id:{cid}", "per-page": 25})
        except requests.RequestException as exc:
            diagnostics.append(f"PII concept works fetch failed (OpenAlex): {exc}")
            return []
        out: list[dict[str, Any]] = []
        for w in works.get("results", []) or []:
            if not isinstance(w, dict):
                continue
            oid = (w.get("id") or "").rsplit("/", maxsplit=1)[-1]
            if not oid:
                continue
            score = self._openalex_max_concept_score_for_id(w.get("concepts"), cid)
            if score >= 0.8:
                out.append(
                    {
                        "provider": "openalex",
                        "id": oid,
                        "origin": "topic_discovered",
                        "concept_match_score": score,
                    }
                )
        if not out:
            diagnostics.append(
                f"PII concept id={cid}: no works with concept score ≥0.8 in the first results page "
                "(metadata sparse or filter mismatch)."
            )
        return out

    @staticmethod
    def _expanded_query_variants(q: str) -> list[str]:
        base = (q or "").strip()
        if not base:
            return []
        low = base.lower()
        variants = [base]
        if "pii" in low or "identifiable" in low or "privacy" in low:
            variants.append(f'{base} OR "Personally Identifiable Information"')
            variants.append(f'{base} OR "data privacy"')
        elif any(
            k in low
            for k in (
                "dpo",
                "direct preference",
                "preference optimization",
                "rlhf",
                "human feedback",
                "preference learning",
                "alignment",
                "reward model",
                "orpo",
                "constitutional ai",
            )
        ):
            variants.extend(
                [
                    "direct preference optimization language model",
                    "reinforcement learning from human feedback language model",
                    "preference optimization large language model",
                ]
            )
        elif len(base.split()) <= 2:
            variants.append(f"{base} OR survey")
        return list(dict.fromkeys(v for v in variants if v))

    @staticmethod
    def _normalize_user_query_for_canonical_match(q: str) -> str:
        t = (q or "").strip()
        t = unicodedata.normalize("NFKC", t)
        t = re.sub(r"[\u200b-\u200d\ufeff]", "", t)
        return t.lower()

    @staticmethod
    def _canonical_anchor_seeds_for_query(q: str) -> list[dict[str, Any]]:
        """
        Phase 1 — anchor papers for known acronyms / canonical lines of work (arXiv ids).
        Keeps retrieval anchored before citation-neighborhood expansion.
        """
        low = SRGApplicationService._normalize_user_query_for_canonical_match(q)
        words = set(re.findall(r"[a-z0-9]+", low))
        out: list[dict[str, Any]] = []

        def add(aid: str) -> None:
            out.append({"provider": "arxiv", "id": aid, "origin": "api_search"})

        # Preference / alignment neighborhood (retrieval repair — multiple anchors, not one semantic hit).
        if any(
            k in low
            for k in (
                "dpo",
                "direct preference optimization",
                "preference optimization",
                "rlhf",
                "reinforcement learning from human feedback",
                "preference learning",
                "human feedback",
                "reward model",
                "constitutional ai",
                "orpo",
                "simpo",
            )
        ) or ("alignment" in low and any(w in words for w in ("llm", "language", "model", "lm"))):
            for aid in (
                "2305.18290",
                "2404.19733",
                "2203.02155",
                "2212.08073",
                "2403.07691",
                "2009.01325",
            ):
                add(aid)
        if "attention is all you need" in low or (
            "transformer" in low and ("attention" in low or "nlp" in low or "language model" in low)
        ):
            add("1706.03762")
        if "lora" in words or "low-rank adaptation" in low:
            add("2106.09685")
        if "bert" in words and len(low.split()) <= 8:
            add("1810.04805")
        # Modern GNN line (Kipf & Welling GCN, GraphSAGE, GAT, GIN) — OpenAlex title search often lands on 2000s precursors only.
        if (
            "graph neural network" in low
            or re.search(r"\bgnn\b", low)
            or "graph convolutional" in low
            or "graphsage" in low.replace("-", "").replace(" ", "")
            or "graph attention network" in low
            or "graph attention" in low
            or "graph isomorph" in low
            or (
                "graph" in words
                and "neural" in words
                and ("network" in words or "networks" in words)
            )
        ):
            for aid in ("1609.02907", "1706.02216", "1710.10903", "1810.00826"):
                add(aid)

        dedup: list[dict[str, Any]] = []
        seen: set[str] = set()
        for s in out:
            k = f"{s['provider']}:{s['id']}"
            if k not in seen:
                seen.add(k)
                dedup.append(s)
        return dedup

    @staticmethod
    def _openalex_title_query_fit_ratio(query: str, title: str) -> float:
        """Loose string similarity between user query and OpenAlex work title (for anchor gating)."""
        qn = " ".join((query or "").lower().split())
        tn = " ".join((title or "").lower().split())[:520]
        if not qn or not tn:
            return 0.0
        return float(difflib.SequenceMatcher(None, qn, tn).ratio())

    @staticmethod
    def _reject_openalex_seed_given_canonicals(
        query: str, title: str, *, has_canonical_anchors: bool
    ) -> tuple[bool, str]:
        """
        When canonical arXiv anchors exist for this query, reject obvious off-domain OpenAlex hits
        so they are not promoted as equal ``api_search`` anchors before canonical literature.
        """
        if not has_canonical_anchors or not (title or "").strip():
            return False, ""
        t = (title or "").strip()
        if _OPENALEX_INDUSTRIAL_APPLIED_RE.search(t):
            return True, "applied/industrial topic mismatch vs canonical ML anchors"
        r = SRGApplicationService._openalex_title_query_fit_ratio(query, t)
        if r < 0.26:
            return True, f"weak title–query match (ratio={r:.2f})"
        return False, ""

    def _openalex_seed_from_work(self, work: dict[str, Any]) -> dict[str, Any] | None:
        oid = (work.get("id") or "").rsplit("/", maxsplit=1)[-1]
        if not oid:
            return None
        return {"provider": "openalex", "id": oid, "origin": "api_search"}

    @staticmethod
    def _tokenize_query_terms(text: str) -> list[str]:
        """
        Compact, reusable query terms for topical relevance scoring.
        """
        stop = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "into",
            "that",
            "this",
            "using",
            "use",
            "based",
            "towards",
            "toward",
            "via",
            "over",
            "under",
            "about",
            "paper",
            "study",
            "method",
            "methods",
            "approach",
            "approaches",
            "model",
            "models",
        }
        parts = re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]{2,}", (text or "").lower())
        out: list[str] = []
        seen: set[str] = set()
        for p in parts:
            if p in stop:
                continue
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
        return out[:12]

    def _query_intent_from_text(self, query: str) -> str:
        q = (query or "").lower()
        if any(k in q for k in ("survey", "review", "overview", "state of the art")):
            return "survey"
        if any(k in q for k in ("latest", "recent", "new", "2024", "2025", "2026")):
            return "recent"
        if any(k in q for k in ("application", "case study", "industry", "clinical", "policy")):
            return "application"
        return "method"

    @staticmethod
    def _derive_query_profile(seeds: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Build query profile from seed metadata if available.
        """
        q = ""
        terms: list[str] = []
        intent = "method"
        for s in seeds:
            sq = (s.get("query_text") or "").strip()
            if sq:
                q = sq
                break
        for s in seeds:
            st = s.get("query_terms") or []
            if isinstance(st, list) and st:
                terms = [str(x).strip().lower() for x in st if str(x).strip()]
                break
        for s in seeds:
            si = (s.get("query_intent") or "").strip().lower()
            if si:
                intent = si
                break
        return {"query_text": q, "query_terms": terms[:12], "query_intent": intent}

    @staticmethod
    def _openalex_cited_by_count(merged_by_id: dict[str, PaperRecord], paper_id: str) -> int:
        prx = merged_by_id.get(paper_id)
        if not prx:
            return 0
        for pv in prx.provenance or []:
            if pv.source == "openalex" and isinstance(pv.raw, dict):
                return int(pv.raw.get("cited_by_count") or 0)
        return 0

    @staticmethod
    def _paper_query_match_score(paper: dict[str, Any], query_terms: list[str]) -> float:
        if not query_terms:
            return 0.0
        title = (paper.get("title") or "").lower()
        abstract = (paper.get("abstract") or "").lower()
        venue = (paper.get("venue") or "").lower()
        domain = (paper.get("domain") or "").lower()
        score = 0.0
        for term in query_terms:
            if term in title:
                score += 1.0
            elif term in abstract:
                score += 0.5
            elif term in venue or term in domain:
                score += 0.3
        return min(1.0, score / max(1.0, float(len(query_terms)) * 0.8))

    @staticmethod
    def _paper_topic_key_for_diversity(paper: dict[str, Any]) -> str:
        title = (paper.get("title") or "").lower()
        if not title:
            return "unknown"
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{3,}", title)
        stop = {
            "using",
            "based",
            "method",
            "methods",
            "model",
            "models",
            "study",
            "approach",
            "approaches",
            "analysis",
            "learning",
            "paper",
        }
        kept = [t for t in tokens if t not in stop]
        if not kept:
            return "general"
        # Stable pseudo-topic signature from strongest title terms.
        return "|".join(sorted(dict.fromkeys(kept[:3])))

    def _apply_diversity_rerank(self, nodes: list[dict[str, Any]], *, top_n: int = 60) -> None:
        """
        Penalize near-duplicate topical clusters so top visible papers are more usable.
        """
        if not nodes:
            return
        topic_seen: dict[str, int] = {}
        ordered = sorted(nodes, key=lambda n: float(n.get("relevance_raw", 0.0)), reverse=True)
        for n in ordered:
            topic = self._paper_topic_key_for_diversity(n)
            n["topic_bucket"] = topic
            seen = topic_seen.get(topic, 0)
            # First item in a topic keeps full score; repeats get softer penalties.
            penalty = min(0.45, seen * 0.12)
            n["diversity_penalty"] = penalty
            n["relevance_raw_diverse"] = float(n.get("relevance_raw", 0.0)) * (1.0 - penalty)
            topic_seen[topic] = seen + 1
        final = sorted(nodes, key=lambda n: float(n.get("relevance_raw_diverse", 0.0)), reverse=True)
        for i, n in enumerate(final, start=1):
            n["diverse_rank"] = i
            n["is_top_diverse"] = i <= max(10, top_n)

    @staticmethod
    def _inclusion_reasons_for_node(node: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        if float(node.get("query_match_score", 0.0)) >= 0.45:
            reasons.append("strong_query_match")
        if int(node.get("citation_count", 0) or 0) >= 8:
            reasons.append("well_connected_in_graph")
        if float(node.get("relation_expand_score", 0.0) or 0.0) >= 0.35:
            reasons.append("strong_relation_signal")
        if node.get("is_missing_link_candidate"):
            reasons.append("missing_link_candidate")
        if node.get("is_foundational_hub"):
            reasons.append("foundational_hub")
        if not reasons:
            reasons.append("seed_or_discovered_candidate")
        return reasons

    @staticmethod
    def _apply_semantic_hard_floor_v2(nodes: list[dict[str, Any]], opts: dict[str, Any]) -> None:
        """Roadmap 1.3 — demote very low semantic / intent alignment before normalization."""
        if not bool(opts.get("lite_semantic_rank_hard_floor", True)):
            return
        floor = float(opts.get("semantic_rank_min_score", 0.25) or 0.25)
        mult = float(opts.get("below_semantic_floor_multiplier", 0.4) or 0.4)
        floor = max(0.0, min(0.45, floor))
        mult = max(0.25, min(1.0, mult))
        for n in nodes:
            sem = float(n.get("semantic_score", 0.0) or 0.0)
            ins = float(n.get("intent_similarity") or sem)
            m = max(sem, ins)
            if m < floor:
                n["below_semantic_floor"] = True
                n["relevance_raw"] = float(n.get("relevance_raw", 0.0)) * mult
            else:
                n["below_semantic_floor"] = False

    @staticmethod
    def _apply_foundational_eligibility_flags(nodes: list[dict[str, Any]], opts: dict[str, Any]) -> None:
        """Roadmap 2.2 — gold hubs must meet semantic + anchor proximity."""
        if not bool(opts.get("lite_foundational_eligibility_rules", True)):
            for n in nodes:
                n["foundational_eligible"] = bool(n.get("is_foundational_hub"))
            return
        f_sem = float(opts.get("foundational_min_semantic_score", 0.35) or 0.35)
        f_int = float(opts.get("foundational_min_intent_similarity", 0.30) or 0.30)
        mx_h = int(opts.get("foundational_max_anchor_hops", 3) or 3)
        f_sem = max(0.0, min(0.6, f_sem))
        f_int = max(0.0, min(0.6, f_int))
        mx_h = max(1, min(20, mx_h))
        for n in nodes:
            if not n.get("is_foundational_hub"):
                n["foundational_eligible"] = False
                continue
            sem = float(n.get("semantic_score", 0.0) or 0.0)
            ins = float(n.get("intent_similarity") or sem)
            hops = int(n.get("anchor_graph_hops", 99))
            if sem < 0.25:
                n["foundational_eligible"] = False
                n["foundational_excluded_low_semantic"] = True
                continue
            n["foundational_eligible"] = bool(sem >= f_sem and ins >= f_int and hops <= mx_h)

    @staticmethod
    def _recency_alignment_score(node: dict[str, Any]) -> float:
        """[0,1] — higher for recent publication years (roadmap RECENT_PROGRESS fusion)."""
        y = node.get("year")
        if not isinstance(y, int):
            return 0.55
        cy = datetime.utcnow().year
        age = max(0, cy - y)
        if age <= 2:
            return 1.0
        if age <= 5:
            return 0.82
        if age <= 10:
            return 0.58
        if age <= 18:
            return 0.35
        return 0.18

    @staticmethod
    def _apply_semantic_graph_top10_rule_a(nodes: list[dict[str, Any]], opts: dict[str, Any]) -> None:
        """
        Roadmap Rule A — if semantic is below the hard-floor threshold, graph-heavy scores
        cannot keep the paper in the top-10 raw ranking slots (heavy demotion).
        """
        if not bool(opts.get("lite_semantic_rank_hard_floor", True)):
            return
        thr = float(opts.get("semantic_rank_min_score", 0.25) or 0.25)
        thr = max(0.0, min(0.45, thr))
        cap_mult = float(opts.get("semantic_top10_below_floor_multiplier", 0.35) or 0.35)
        cap_mult = max(0.05, min(1.0, cap_mult))
        ordered = sorted(nodes, key=lambda n: float(n.get("relevance_raw", 0.0)), reverse=True)
        for n in ordered[:10]:
            sem = float(n.get("semantic_score", 0.0) or 0.0)
            ins = float(n.get("intent_similarity") or sem)
            if max(sem, ins) < thr:
                n["relevance_raw"] = float(n.get("relevance_raw", 0.0)) * cap_mult
                n["semantic_top10_graph_capped"] = True

    def _broaden_seed_pool(
        self,
        seeds: list[dict[str, Any]],
        diagnostics: list[str],
        *,
        max_extra: int = 120,
    ) -> list[dict[str, Any]]:
        """
        Broaden mode: if root seeds are sparse, add both outgoing references and incoming citers.
        """
        if len(seeds) >= 96:
            return seeds
        oa = OpenAlexClient()
        out = list(seeds)
        seen = {f"{(s.get('provider') or '').lower()}:{(s.get('id') or '').lower()}" for s in seeds}
        added = 0
        for s in seeds[:25]:
            if added >= max_extra:
                break
            provider = (s.get("provider") or "").lower().strip()
            sid = (s.get("id") or "").strip()
            if not sid:
                continue
            work: dict[str, Any] | None = None
            try:
                self.orchestrator.budget.acquire("openalex")
                if provider == "openalex":
                    work = oa.fetch_paper(sid)
                elif provider in {"doi", "crossref"}:
                    work = openalex_response_first_work(oa.fetch_by_doi(sid))
                elif provider == "arxiv":
                    work = oa.fetch_work_for_arxiv_id(sid)
            except requests.RequestException:
                continue
            if not isinstance(work, dict):
                continue
            # outgoing refs
            for t in OpenAlexClient.referenced_work_tail_ids(work)[:35]:
                if added >= max_extra:
                    break
                if not re.match(r"^W\d+$", t):
                    continue
                k = f"openalex:{t.lower()}"
                if k in seen:
                    continue
                seen.add(k)
                out.append({"provider": "openalex", "id": t, "origin": "discovered_2hop"})
                added += 1
            # incoming citers from cited_by_api_url
            cb = (work.get("cited_by_api_url") or "").strip()
            if cb and added < max_extra:
                try:
                    self.orchestrator.budget.acquire("openalex")
                    rows = oa._get(cb.replace(oa.base_url, ""), params={"per-page": 25}).get("results") or []
                    for row in rows:
                        if added >= max_extra:
                            break
                        if not isinstance(row, dict):
                            continue
                        srow = self._openalex_seed_from_work(row)
                        if not srow:
                            continue
                        k = f"openalex:{srow['id'].lower()}"
                        if k in seen:
                            continue
                        seen.add(k)
                        srow["origin"] = "broaden_citers"
                        out.append(srow)
                        added += 1
                except requests.RequestException:
                    pass
        if added:
            diagnostics.append(
                f"Yetersiz doğrudan sonuç; literatür derinleştiriliyor... broaden mode +{added} aday ekledi."
            )
        return out

    def _second_hop_reference_seeds(
        self,
        seeds: list[dict[str, Any]],
        diagnostics: list[str],
        skipped_ids: list[str] | None = None,
        *,
        max_seeds: int = 160,
        per_parent_cap: int = 30,
    ) -> list[dict[str, Any]]:
        """2-hop: OpenAlex referenced_works from search-first or PDF-resolved openalex/doi/arxiv seeds."""
        oa = OpenAlexClient()
        seen: set[str] = {f"{s.get('provider','').lower()}:{s.get('id','').strip().lower()}" for s in seeds if s.get("id")}
        out: list[dict[str, Any]] = []
        for s in seeds:
            if len(out) >= max_seeds:
                break
            origin = (s.get("origin") or "").strip()
            if origin not in ("api_search", "uploaded_pdf"):
                continue
            provider = (s.get("provider") or "").strip().lower()
            sid = (s.get("id") or "").strip()
            if not sid:
                continue
            work: dict[str, Any] | None = None
            try:
                if provider == "openalex":
                    self.orchestrator.budget.acquire("openalex")
                    work = oa.fetch_paper(sid)
                elif provider in ("doi", "crossref"):
                    self.orchestrator.budget.acquire("openalex")
                    resp = oa.fetch_by_doi(sid)
                    work = openalex_response_first_work(resp)
                elif provider == "arxiv":
                    self.orchestrator.budget.acquire("openalex")
                    work = oa.fetch_work_for_arxiv_id(sid)
            except requests.HTTPError as exc:
                resp = exc.response
                if resp is not None and resp.status_code == 404:
                    diagnostics.append(
                        f"OpenAlex 2-hop: merged or invalid id ({provider}:{sid[:28]}) — skipped (not API pressure)."
                    )
                    if skipped_ids is not None and provider == "openalex":
                        stail = (sid or "").strip()
                        if re.match(r"^W\d+$", stail) and stail not in skipped_ids:
                            skipped_ids.append(stail)
                    continue
                diagnostics.append(f"2-hop referenced_works fetch failed ({provider}:{sid[:28]}): {exc}")
                continue
            except requests.RequestException as exc:
                diagnostics.append(f"2-hop referenced_works fetch failed ({provider}:{sid[:28]}): {exc}")
                continue
            if not isinstance(work, dict):
                continue
            tails = OpenAlexClient.referenced_work_tail_ids(work)
            for tail in tails[:per_parent_cap]:
                if len(out) >= max_seeds:
                    break
                if re.match(r"^W\d+$", tail):
                    key = f"openalex:{tail.lower()}"
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append({"provider": "openalex", "id": tail, "origin": "discovered_2hop"})
        return out

    def _merge_openalex_refs_into_arxiv_records(self, records: list[PaperRecord], diagnostics: list[str]) -> int:
        """Cross-provider: arXiv-ingested rows get referenced_works from OpenAlex (enables coupling/co-citation)."""
        oa = OpenAlexClient()
        added = 0
        for rec in records:
            from_arxiv = rec.canonical_id.lower().startswith("arxiv:") or any(
                p.source == "arxiv" for p in (rec.provenance or [])
            )
            arxiv_linked = bool(rec.arxiv_id) or (
                bool(rec.doi) and "arxiv" in (rec.doi or "").lower() and "10.48550" in (rec.doi or "").lower()
            )
            if not (from_arxiv or arxiv_linked):
                continue
            aid = (rec.arxiv_id or "").strip() or normalize_arxiv_list_id(rec.canonical_id.split(":", 1)[-1])
            if not aid:
                continue
            try:
                self.orchestrator.budget.acquire("openalex")
                work = oa.fetch_work_for_arxiv_id(aid)
            except requests.RequestException as exc:
                diagnostics.append(f"Cross-provider OpenAlex fetch for arXiv:{aid} failed: {exc}")
                continue
            if not work:
                diagnostics.append(
                    f"Cross-provider: OpenAlex has no work for arXiv:{aid} (missing DOI bridge or not indexed)."
                )
                continue
            tails = [t for t in OpenAlexClient.referenced_work_tail_ids(work) if re.match(r"^W\d+$", t)]
            if not tails:
                continue
            before = len(rec.references)
            rec.references = sorted(set(rec.references + tails))
            added += len(rec.references) - before
            oid = (work.get("id") or "").rsplit("/", maxsplit=1)[-1]
            if oid and not rec.openalex_id:
                rec.openalex_id = oid
            rec.provenance.append(
                Provenance(
                    source="openalex",
                    source_id=rec.openalex_id or oid or aid,
                    field_confidence={},
                    raw={"enriched_from": "referenced_works_for_arxiv"},
                )
            )
        return added

    def _enrich_arxiv_records_openalex_fallback(self, records: list[PaperRecord], diagnostics: list[str]) -> int:
        """
        Best-effort OpenAlex enrichment for arXiv rows:
        1) ids.arxiv / arXiv DOI bridge (existing helper),
        2) title search fallback when bridge is missing.
        """
        oa = OpenAlexClient()
        enriched = 0
        for rec in records:
            is_arxiv = rec.canonical_id.lower().startswith("arxiv:") or bool(rec.arxiv_id)
            if not is_arxiv:
                continue
            # If already has solid reference graph support, skip.
            if rec.openalex_id and len(rec.references or []) >= 8:
                continue
            aid = (rec.arxiv_id or "").strip() or normalize_arxiv_list_id(rec.canonical_id.split(":", 1)[-1])
            work: dict[str, Any] | None = None
            if aid:
                try:
                    self.orchestrator.budget.acquire("openalex")
                    work = oa.fetch_work_for_arxiv_id(aid)
                except requests.RequestException:
                    work = None
            if not isinstance(work, dict):
                title_q = (rec.title or "").strip()
                if len(title_q) >= 12:
                    try:
                        self.orchestrator.budget.acquire("openalex")
                        rows = (oa.search_by_title(title_q).get("results") or [])
                        if rows and isinstance(rows[0], dict):
                            cand = rows[0]
                            t0 = (cand.get("title") or "").strip().lower()
                            t1 = (rec.title or "").strip().lower()
                            # Conservative title alignment before trusting fallback.
                            if t0 and t1 and (t0 in t1 or t1 in t0 or _titles_match_score(t0, t1) >= 0.82):
                                work = cand
                    except requests.RequestException:
                        work = None
            if not isinstance(work, dict):
                continue
            tails = [t for t in OpenAlexClient.referenced_work_tail_ids(work) if re.match(r"^W\d+$", t)]
            if tails:
                before = len(rec.references)
                rec.references = sorted(set(rec.references + tails))
                if len(rec.references) > before:
                    enriched += 1
            oid = (work.get("id") or "").rsplit("/", maxsplit=1)[-1]
            if oid and not rec.openalex_id:
                rec.openalex_id = oid
            if oid:
                try:
                    self.orchestrator.cache.put("openalex", oid, work)
                except Exception:
                    pass
            rec.provenance.append(
                Provenance(
                    source="openalex",
                    source_id=rec.openalex_id or aid or "unknown",
                    field_confidence={},
                    raw={"enriched_from": "arxiv_openalex_fallback"},
                )
            )
        if enriched:
            diagnostics.append(f"OpenAlex enrichment fallback improved {enriched} arXiv records with references.")
        return enriched

    def discover_from_query(
        self,
        query: str,
        *,
        text_search_backend: str = "openalex",
        arxiv_search_max_results: int = 25,
    ) -> list[dict[str, Any]]:
        q = query.strip()
        diagnostics: list[str] = []
        self._last_seed_selection_notes = []
        if not q:
            return []
        ql = q.lower()
        if "directed preference optimization" in ql:
            q = ql.replace("directed preference optimization", "direct preference optimization", 1)
        q_terms = self._tokenize_query_terms(q)
        q_intent = self._query_intent_from_text(q)

        def enrich_seed(seed: dict[str, Any]) -> dict[str, Any]:
            return {
                **seed,
                "query_text": q,
                "query_terms": list(q_terms),
                "query_intent": q_intent,
            }

        if re.search(r"\b10\.\d{4,9}/", q, flags=re.IGNORECASE):
            doi = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", q, flags=re.IGNORECASE)
            return [enrich_seed({"provider": "doi", "id": (doi.group(0) if doi else q), "origin": "api_search"})]
        arx = re.search(r"\b(?:arXiv:)?(\d{4}\.\d{4,5})(?:v\d+)?\b", q, flags=re.IGNORECASE)
        if arx:
            return [enrich_seed({"provider": "arxiv", "id": arx.group(1), "origin": "api_search"})]
        backend = (text_search_backend or "openalex").strip().lower()
        if backend == "arxiv":
            try:
                seeds: list[dict[str, Any]] = []
                seeds.extend(self._canonical_anchor_seeds_for_query(q))
                for qq in self._expanded_query_variants(q):
                    sq = arxiv_search_query_from_user_text(qq)
                    hits = ArXivClient().search(sq, max_results=max(1, min(arxiv_search_max_results, 50)))
                    seeds.extend({"provider": "arxiv", "id": h["id"], "origin": "api_search"} for h in hits if h.get("id"))
                if self._is_pii_concept_query(q):
                    seeds.extend(self._openalex_pii_concept_seed_hits(diagnostics))
                dedup: list[dict[str, Any]] = []
                seen: set[str] = set()
                for s in seeds:
                    k = f"{(s.get('provider') or '').lower()}:{(s.get('id') or '').lower()}"
                    if not s.get("id") or k in seen:
                        continue
                    seen.add(k)
                    dedup.append(s)
                return [enrich_seed(s) for s in self._broaden_seed_pool(dedup, diagnostics)]
            except requests.RequestException:
                return []
        try:
            canonical_seeds = self._canonical_anchor_seeds_for_query(q)
            has_canonical = bool(canonical_seeds)
            oa_seeds: list[dict[str, Any]] = []
            oa = OpenAlexClient()
            oa_cap = 20
            for qq in self._expanded_query_variants(q):
                if len(oa_seeds) >= oa_cap:
                    break
                result = oa.search_by_title(qq, per_page=5)
                rows = result.get("results", []) or []
                accepted_any = False
                for top in rows:
                    if len(oa_seeds) >= oa_cap:
                        break
                    oid = (top.get("id") or "").rsplit("/", maxsplit=1)[-1]
                    if not oid:
                        continue
                    title_disp = (top.get("display_name") or top.get("title") or "").strip()
                    bad, reason = self._reject_openalex_seed_given_canonicals(
                        q, title_disp, has_canonical_anchors=has_canonical
                    )
                    if bad:
                        self._last_seed_selection_notes.append(
                            f"OpenAlex seed skipped ({reason}): {title_disp[:92]}{'…' if len(title_disp) > 92 else ''}"
                        )
                        continue
                    oa_seeds.append({"provider": "openalex", "id": oid, "origin": "api_search"})
                    accepted_any = True
                if not accepted_any and rows:
                    top = rows[0]
                    oid = (top.get("id") or "").rsplit("/", maxsplit=1)[-1]
                    if oid:
                        oa_seeds.append({"provider": "openalex", "id": oid, "origin": "api_search"})
                        if has_canonical:
                            t0 = (top.get("display_name") or top.get("title") or "").strip()
                            self._last_seed_selection_notes.append(
                                "OpenAlex: all ranked hits failed canonical-aware filters; kept search rank-1 as fallback "
                                f"({t0[:80]}{'…' if len(t0) > 80 else ''})."
                            )
                        else:
                            self._last_seed_selection_notes.append(
                                "OpenAlex: using search rank-1 (no canonical anchor list for this query)."
                            )
            seeds = list(canonical_seeds) + oa_seeds
            if self._is_pii_concept_query(q):
                seeds.extend(self._openalex_pii_concept_seed_hits(diagnostics))
            if not seeds:
                return []
            seen: set[str] = set()
            deduped: list[dict[str, Any]] = []
            for s in seeds:
                kid = f"{(s.get('provider') or '').strip().lower()}:{(s.get('id') or '').strip().lower()}"
                if not s.get("id") or kid in seen:
                    continue
                seen.add(kid)
                deduped.append(s)
            if self._last_seed_selection_notes:
                self._last_seed_selection_notes.insert(
                    0,
                    f"Anchor resolution: {len(canonical_seeds)} canonical arXiv seed(s) first, "
                    f"then {len(oa_seeds)} OpenAlex work(s) (up to 5 hits per query variant for disambiguation).",
                )
            gnn_pack_ids = frozenset({"1609.02907", "1706.02216", "1710.10903", "1810.00826"})
            canon_arxiv_ids = [s["id"] for s in canonical_seeds if (s.get("provider") or "").lower() == "arxiv"]
            gnn_active = any(x in gnn_pack_ids for x in canon_arxiv_ids)
            ca_txt = ", ".join(canon_arxiv_ids) if canon_arxiv_ids else "(none)"
            self._last_seed_selection_notes.append(
                f"GNN anchor rule: {'matched (Kipf/GraphSAGE/GAT/GIN pack prepended)' if gnn_active else 'did not match'}; "
                f"canonical arXiv ids for this query: {ca_txt}."
            )
            return [enrich_seed(s) for s in self._broaden_seed_pool(deduped, diagnostics)]
        except requests.RequestException:
            return []

    @staticmethod
    def _resolution_badge(code: str | None) -> str:
        key = (code or "pdf_text").strip().lower()
        if key == "doi":
            return "Found via DOI"
        if key == "filename":
            return "Found via Filename"
        return "Found via PDF Text"

    def _expand_pool_fill_notes(
        self,
        effective_seeds: list[dict[str, Any]],
        hop_seeds: list[dict[str, Any]],
        diagnostics: list[str],
        expand_active: bool,
    ) -> list[str]:
        stats = getattr(self, "_last_expand_stats", {}) or {}
        notes: list[str] = list(stats.get("pool_empty_reasons") or [])
        ordered = int(stats.get("ordered_pool_size", 0) or 0)
        if ordered == 0 and any(s.get("origin") == "uploaded_pdf" for s in effective_seeds):
            notes.append(
                "PDF reference pool empty: `_extract_reference_dois` found no DOI tokens in the parsed "
                "first pages (bibliography format, scanned PDF, or missing DOIs)."
            )
        if expand_active and not hop_seeds:
            notes.append(
                "2-hop OpenAlex `referenced_works` expansion returned no extra ids — check seed providers "
                "(need openalex / doi / arxiv), API errors in diagnostics, or OpenAlex coverage for those works."
            )
        elif hop_seeds:
            notes.append(f"2-hop referenced_works seeds added: {len(hop_seeds)}.")
        for d in diagnostics:
            low = d.lower()
            if "merged or invalid id" in low or "not api pressure" in low:
                continue
            if "429" in d or "503" in d or "rate" in low or "timeout" in low:
                notes.append(f"Possible API pressure / rate limit: {d}")
        return notes

    def run_pipeline(
        self,
        seeds: list[dict[str, Any]],
        discipline: str | None = None,
        discovery_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not any(s.get("origin") == "uploaded_pdf" for s in seeds):
            # Avoid leaking PDF expand counters into Search-First-only runs.
            self._last_expand_stats = {}
        effective_seeds = [
            {
                "provider": s.get("provider", "").strip(),
                "id": s.get("id", "").strip(),
                "origin": s.get("origin", "discovered"),
                "metadata_resolution": s.get("metadata_resolution"),
                "concept_match_score": s.get("concept_match_score"),
                "query_text": s.get("query_text"),
                "query_terms": s.get("query_terms"),
                "query_intent": s.get("query_intent"),
            }
            for s in seeds
        ]
        query_profile = self._derive_query_profile(effective_seeds)
        opts = dict(discovery_options or {})
        top_n = int(opts.get("top_n", EXPAND_DISPLAY_TOP_N) or EXPAND_DISPLAY_TOP_N)
        top_n = max(10, min(200, top_n))
        coupling_threshold = int(opts.get("coupling_threshold", 3) or 3)
        coupling_threshold = max(2, min(8, coupling_threshold))
        enable_two_hop = bool(opts.get("enable_two_hop", True))
        concept_threshold = float(opts.get("concept_threshold", 0.25) or 0.25)
        concept_threshold = max(0.0, min(1.0, concept_threshold))
        relation_topic_weight = float(opts.get("relation_topic_weight", 0.0) or 0.0)
        relation_topic_weight = max(0.0, min(1.0, relation_topic_weight))
        foundational_boost = float(opts.get("foundational_boost", 0.0) or 0.0)
        foundational_boost = max(0.0, min(5.0, foundational_boost))
        allowed_domains = {
            str(x).strip().lower() for x in (opts.get("allowed_domains") or []) if str(x).strip()
        }
        two_hop_budget = int(opts.get("two_hop_budget", 120) or 120)
        two_hop_budget = max(30, min(300, two_hop_budget))
        lite_field_aware = bool(opts.get("lite_field_aware", False))
        lite_retrieval_repair = bool(opts.get("lite_retrieval_repair", False))
        lite_coherence_stabilization = bool(opts.get("lite_coherence_stabilization", False))
        lite_v2_ranking = bool(opts.get("lite_v2_ranking", False))
        lite_semantic_graph_ranking = bool(opts.get("lite_semantic_graph_ranking", False))
        if lite_v2_ranking:
            # SRG Lite v2 replaces the static semantic/graph blend with intent + canonicality fusion.
            lite_semantic_graph_ranking = False
        sem_w = float(opts.get("semantic_graph_sem_weight", 0.7) or 0.7)
        graph_w = float(opts.get("semantic_graph_graph_weight", 0.3) or 0.3)
        sw_sum = sem_w + graph_w
        if sw_sum > 0:
            sem_w, graph_w = sem_w / sw_sum, graph_w / sw_sum
        if lite_retrieval_repair:
            floor = int(opts.get("two_hop_budget_floor", 200) or 200)
            two_hop_budget = max(two_hop_budget, max(30, min(300, floor)))
        diagnostics: list[str] = []
        diagnostics.extend(list(getattr(self, "_last_seed_selection_notes", []) or []))
        skipped_ids: list[str] = []
        if len(effective_seeds) < 12:
            concept_threshold = max(0.15, concept_threshold - 0.15)
            if not lite_field_aware:
                relation_topic_weight = max(0.2, relation_topic_weight)
            coupling_threshold = min(coupling_threshold, 2)
            diagnostics.append(
                "Soğuk başlangıç algılandı: concept eşiği düşürüldü ve topic sinyali geçici olarak artırıldı."
                if not lite_field_aware
                else "Cold start: keeping citation-first ranking (field-aware mode)."
            )
        auto_escalated = False
        if len(effective_seeds) < top_n and not enable_two_hop:
            enable_two_hop = True
            two_hop_budget = max(two_hop_budget, min(220, top_n * 2))
            auto_escalated = True
        elif len(effective_seeds) < top_n and enable_two_hop and two_hop_budget < 200:
            two_hop_budget = min(260, max(two_hop_budget + 60, top_n * 2))
            auto_escalated = True
        if auto_escalated:
            diagnostics.append(
                "Literature is being deepened automatically: discovery depth escalated due to sparse results."
            )
        hop_seeds = (
            self._second_hop_reference_seeds(
                effective_seeds, diagnostics, skipped_ids, max_seeds=two_hop_budget
            )
            if enable_two_hop
            else []
        )
        if enable_two_hop and len(hop_seeds) < max(5, min(20, top_n // 5)):
            # Library-mode safety net: if reference-based 2-hop is sparse, include incoming-citation broadening too.
            broadened = self._broaden_seed_pool(effective_seeds + hop_seeds, diagnostics, max_extra=two_hop_budget)
            hop_seen = {f"{(s.get('provider') or '').lower()}:{(s.get('id') or '').lower()}" for s in (effective_seeds + hop_seeds)}
            extra_hop: list[dict[str, Any]] = []
            for s in broadened:
                k = f"{(s.get('provider') or '').lower()}:{(s.get('id') or '').lower()}"
                if k in hop_seen:
                    continue
                hop_seen.add(k)
                extra_hop.append(s)
            if extra_hop:
                hop_seeds.extend(extra_hop[:two_hop_budget])
                diagnostics.append(
                    f"Auto-broadening fallback added {len(extra_hop[:two_hop_budget])} extra candidates for connectivity."
                )
        effective_seeds.extend(hop_seeds)
        self._batch_prefetch_openalex_seed_works(effective_seeds, diagnostics)
        w_seed_tails = self._collect_openalex_w_tails_from_seeds(effective_seeds)
        self._ensure_openalex_tails_in_store(w_seed_tails, diagnostics, skipped_ids)
        origin_map: dict[str, str] = {}
        for s in effective_seeds:
            key = f"{s.get('provider', '').strip().lower()}:{s.get('id', '').strip().lower()}"
            if not s.get("id"):
                continue
            raw_origin = s.get("origin", "discovered")
            o_store = (
                "discovered"
                if raw_origin in ("topic_discovered", "discovered_2hop")
                else raw_origin
            )
            if key not in origin_map or o_store == "uploaded_pdf":
                origin_map[key] = o_store
        for seed in effective_seeds:
            provider = seed["provider"].strip().lower()
            resolved_provider = self.provider_aliases.get(provider, provider)
            try:
                self.orchestrator.enqueue(resolved_provider, seed["id"])
            except ValueError as exc:
                diagnostics.append(str(exc))
        raw_records: list[PaperRecord] = []
        try:
            raw_records = self.orchestrator.run()
        except requests.RequestException as exc:
            diagnostics.append(f"Ingestion batch error -> {exc}")

        for w in (self.orchestrator.last_ingestion_stats or {}).get("warnings") or []:
            diagnostics.append(str(w))

        for x in getattr(self.orchestrator, "last_openalex_404_seeds", []) or []:
            self._append_skipped_openalex(skipped_ids, str(x))

        if not raw_records:
            diagnostics.append("No records fetched from providers; using pending placeholders (not Fallback).")
            raw_records = self._build_pending_seed_placeholders(effective_seeds)
        else:
            for i, seed in enumerate(effective_seeds):
                if i >= len(raw_records):
                    break
                rec = raw_records[i]
                cms = seed.get("concept_match_score")
                if cms is not None and float(cms) >= 0.8:
                    rec.provenance.append(
                        Provenance(
                            source="topic_probe",
                            source_id="pii",
                            raw={"score": float(cms)},
                        )
                    )

        feedback = self.feedback_store.list_all()
        merged_records, audit = self.deduper.deduplicate(raw_records, feedback=feedback)
        oa_ref_merged = self._merge_openalex_refs_into_arxiv_records(merged_records, diagnostics)
        oa_ref_merged += self._enrich_arxiv_records_openalex_fallback(merged_records, diagnostics)
        merged_records, graph_noise_ids = self._hydrate_metadata(merged_records, diagnostics, skipped_ids)
        ref_tails_only = self._collect_openalex_w_reference_tails_not_in_nodes(merged_records)
        self._ensure_openalex_tails_in_store(ref_tails_only, diagnostics, skipped_ids)
        self._inject_foundational_literature(merged_records, origin_map, diagnostics, skipped_ids)
        self._ensure_openalex_tails_in_store(
            self._all_openalex_w_tails_for_batch_warm(merged_records), diagnostics, skipped_ids
        )
        stats = getattr(self, "_last_expand_stats", {}) or {}
        pdf_lines = int(stats.get("pdf_references_raw_lines", 0) or 0)
        total_refs_in_graph = sum(len(p.references) for p in merged_records)
        stats["arxiv_openalex_referenced_works_merged"] = int(oa_ref_merged)
        stats["references_extracted"] = max(pdf_lines + int(oa_ref_merged), total_refs_in_graph)
        tails_union = set(w_seed_tails) | set(ref_tails_only)
        extra_resolved = sum(
            1
            for t in tails_union
            if openalex_fetch_payload_valid(self.orchestrator.cache.get("openalex", t) or {})
        )
        oa404_recov = int((self.orchestrator.last_ingestion_stats or {}).get("openalex_404_recoveries") or 0)
        resolved_refs = int(stats.get("ordered_pool_size", 0) or 0) + extra_resolved + oa404_recov
        stats["references_resolved"] = resolved_refs
        if resolved_refs < 5:
            # Priority hydration: with few resolved refs, pull their local neighborhoods first.
            self._promote_resolved_reference_hubs(merged_records, tails_union, diagnostics, skipped_ids, max_ref_seeds=2)
            self._force_include_incoming_citers(merged_records, diagnostics, max_extra=36)
            coupling_threshold = 1
            diagnostics.append(
                "Low resolved references detected (<5): incoming-citation recovery enabled and coupling threshold set to 1."
            )
        self._last_expand_stats = stats
        self._apply_record_year_domain_hard_fix(merged_records)
        dom = self._canonical_discipline(discipline)
        if dom:
            for paper in merged_records:
                if not paper.domain:
                    paper.domain = dom
        if allowed_domains:
            before_allow = len(merged_records)
            merged_records = [
                p for p in merged_records if not p.domain or (p.domain or "").strip().lower() in allowed_domains
            ]
            dropped_allow = before_allow - len(merged_records)
            if dropped_allow > 0:
                diagnostics.append(
                    f"Discipline filters: dropped {dropped_allow} papers outside allowed domains "
                    f"{sorted(allowed_domains)}."
                )
        graph = self.graph_builder.build(merged_records)
        nodes_map = {p.canonical_id: p for p in merged_records}
        graph = enrich_graph_with_citation_snippets(graph, nodes_map, budget=self.orchestrator.budget)
        quality = build_quality_report(audit)
        papers_payload = [self._paper_to_json_with_origin(p, origin_map) for p in merged_records]
        graph_json = _graph_to_json(graph)
        graph_json["nodes"] = papers_payload
        self._inject_strong_coupling_edges(
            graph_json["nodes"],
            graph_json["edges"],
            min_shared_refs=coupling_threshold,
            potential_mode=(coupling_threshold <= 1),
        )
        for node in graph_json["nodes"]:
            if node.get("id") in graph_noise_ids:
                node["graph_noise"] = True
        missing_links = self._compute_missing_links(graph_json["nodes"], graph_json["edges"], threshold=3)
        for node in graph_json["nodes"]:
            mlink = missing_links.get(node["id"])
            node["missing_link_score"] = mlink["co_cited_by_uploaded_count"] if mlink else 0
            node["is_missing_link_candidate"] = bool(mlink and mlink["co_cited_by_uploaded_count"] >= 3)

        pii_tail = self._ensure_pii_concept_tail(diagnostics)
        merged_by_id = {p.canonical_id: p for p in merged_records}
        for node in graph_json["nodes"]:
            pr = merged_by_id.get(node["id"])
            pii_m = 0.0
            if pr:
                for prov in pr.provenance or []:
                    if prov.source == "openalex" and isinstance(prov.raw, dict):
                        pii_m = self._openalex_max_concept_score_for_id(prov.raw.get("concepts"), pii_tail or "")
                        break
            node["pii_concept_match"] = round(pii_m, 4)
            node["is_foundational_hub"] = bool(
                pr and any(p.source == "foundational_hub" for p in (pr.provenance or []))
            ) or (
                bool(node.get("is_missing_link_candidate"))
                and (pii_m >= concept_threshold or float(node.get("missing_link_score", 0)) >= 4)
            )

        anchor_ids = {
            n["id"]
            for n in graph_json["nodes"]
            if n.get("seed_origin") in ("uploaded_pdf", "api_search")
        }
        expand_relation_filter_active = any(
            s.get("origin") in ("discovered", "topic_discovered", "discovered_2hop") for s in effective_seeds
        )
        rel_params = RelationParams(
            w_direct=float(opts.get("w_direct", 2.0) or 2.0),
            w_coupling=float(opts.get("w_coupling", 0.6) or 0.6),
            w_cocitation=float(opts.get("w_cocitation", 0.8) or 0.8),
            w_topic=relation_topic_weight,
        )
        expand_scores = score_expand_relations(
            graph,
            anchor_ids,
            rel_params,
        )
        for node in graph_json["nodes"]:
            if node.get("seed_origin") != "discovered":
                node["relation_expand_score"] = 0.0
                node["expand_explanation"] = ""
                node["expand_rank"] = 0
                continue
            tpl = expand_scores.get(node["id"])
            if tpl:
                sc, expl, diag = tpl
                node["relation_expand_score"] = sc
                node["expand_explanation"] = expl
                for k, v in diag.items():
                    node[k] = v
            else:
                node["relation_expand_score"] = 0.0
                node["expand_explanation"] = "Genişletme (referans havuzu) ile grafa eklendi"
                node["expand_rank"] = 0

        discovered_ranked = sorted(
            (n for n in graph_json["nodes"] if n.get("seed_origin") == "discovered"),
            key=lambda x: float(x.get("relation_expand_score", 0.0)),
            reverse=True,
        )
        for i, n in enumerate(discovered_ranked, start=1):
            n["expand_rank"] = i

        hop_map: dict[str, int] = {}
        if lite_coherence_stabilization or lite_v2_ranking:
            hop_map = SRGApplicationService._anchor_hop_distances(
                [n["id"] for n in graph_json["nodes"]],
                graph_json.get("edges") or [],
                anchor_ids,
            )

        out_deg = Counter(e["source"] for e in graph_json["edges"])
        in_deg = Counter(e["target"] for e in graph_json["edges"])
        rec_list = self.discovery.recommend_missing_links(graph, foundational_boost=foundational_boost)
        rec_map = {r.paper_id: float(r.score) for r in rec_list}
        raw_scores: list[float] = []
        references_total = sum(len(p.references or []) for p in merged_records)
        evidence_edges = len(graph_json.get("edges") or [])
        metadata_only_mode = references_total == 0 and evidence_edges == 0
        node_ids_list = [n["id"] for n in graph_json["nodes"]]
        v2_intent = ""
        v2_confidence = 0.0
        v2_intent_distribution: dict[str, Any] = {}
        v2_hardware_recency_boost = False
        v2_fusion = (0.6, 0.3, 0.1)
        pr_n: dict[str, float] = {}
        eq_n: dict[str, float] = {}
        ctx_n: dict[str, float] = {}
        graph_combined: dict[str, float] = {}
        qtext_for_v2 = ""
        query_domains_v2: frozenset[str] = frozenset()
        sem_by_id_pre: dict[str, float] = {}
        local_conn_map: dict[str, float] = {}
        graph_for_fusion: dict[str, float] = {}
        lite_semantic_control = bool(opts.get("lite_semantic_control", True))
        lite_intent_verification = bool(opts.get("lite_intent_verification", True))
        lite_query_conditioned_canonicality = bool(opts.get("lite_query_conditioned_canonicality", True))
        intent_verification_meta: dict[str, Any] = {}
        sc_gamma = float(opts.get("semantic_control_local_gamma", 0.09) or 0.09)
        sc_delta = float(opts.get("semantic_control_drift_lambda", 0.12) or 0.12)
        sc_gamma = max(0.0, min(0.35, sc_gamma))
        sc_delta = max(0.0, min(0.45, sc_delta))
        if lite_v2_ranking:
            qtext_for_v2 = (query_profile.get("query_text") or "").strip()
            if not qtext_for_v2:
                qtext_for_v2 = " ".join(str(t) for t in (query_profile.get("query_terms") or []))
            icr = classify_query(qtext_for_v2, {})
            v2_intent = str(icr.get("intent") or "")
            v2_confidence = float(icr.get("confidence") or 0.0)
            v2_intent_distribution = dict(
                icr.get("intent_distribution") or icr.get("distribution") or {}
            )
            v2_hardware_recency_boost = bool(icr.get("hardware_recency_fusion_boost"))
            v2_fusion = fusion_weights_for_intent(v2_intent)
            hop_for_bundle = hop_map if hop_map else None
            pr_n, eq_n, ctx_n, graph_combined = compute_graph_score_bundle(
                graph_json.get("edges") or [], node_ids_list, hop_for_bundle
            )
            query_domains_v2 = infer_query_domains(qtext_for_v2)
            for _n in graph_json["nodes"]:
                sem_by_id_pre[_n["id"]] = float(self._semantic_fit_score(_n, query_profile))
            local_conn_map = local_semantic_connectivity(
                node_ids_list,
                sem_by_id_pre,
                graph_json.get("edges") or [],
            )
            deg_map = {nid: int(out_deg[nid] + in_deg[nid]) for nid in node_ids_list}
            if lite_semantic_control:
                graph_for_fusion = hub_suppressed_graph_scores(graph_combined, deg_map)
            else:
                graph_for_fusion = dict(graph_combined)

        for node in graph_json["nodes"]:
            pid = node["id"]
            citation_count = int(out_deg[pid] + in_deg[pid])
            node["citation_count"] = citation_count

            if lite_v2_ranking:
                q_match = self._paper_query_match_score(node, query_profile.get("query_terms") or [])
                node["query_match_score"] = q_match
                sem = float(sem_by_id_pre.get(pid, self._semantic_fit_score(node, query_profile)))
                g_struct = float(graph_combined.get(pid, 0.45))
                gscore = float(graph_for_fusion.get(pid, g_struct))
                node["pagerank_norm"] = round(float(pr_n.get(pid, 0.0)), 4)
                node["edge_quality_norm"] = round(float(eq_n.get(pid, 0.0)), 4)
                node["context_score_norm"] = round(float(ctx_n.get(pid, 0.5)), 4)
                node["graph_score_structural_raw"] = round(g_struct, 4)
                inbound = SRGApplicationService._openalex_cited_by_count(merged_by_id, pid)
                survey_ctx = infer_survey_context_flag(node)
                hops = int(hop_map.get(pid, 99))
                node["anchor_graph_hops"] = hops
                vocab_c = vocabulary_coherence_score(qtext_for_v2, node)
                node["vocabulary_coherence"] = round(float(vocab_c), 4)
                if lite_query_conditioned_canonicality:
                    canon = compute_query_conditioned_canonicality(
                        node,
                        qtext_for_v2,
                        v2_intent,
                        inbound_citations=inbound,
                        is_in_survey_context=survey_ctx,
                        anchor_hops=hops,
                        semantic_alignment=sem,
                        vocabulary_coherence=vocab_c,
                    )
                else:
                    canon = compute_canonicality(
                        node,
                        qtext_for_v2,
                        v2_intent,
                        inbound_citations=inbound,
                        is_in_survey_context=survey_ctx,
                    )
                _ni = normalize_query_intent(v2_intent)
                recency_v = (
                    SRGApplicationService._recency_alignment_score(node)
                    if _ni in RECENCY_TRIPLE_INTENTS
                    else None
                )
                fused = compute_final_score(
                    sem,
                    gscore,
                    canon,
                    v2_intent,
                    recency=recency_v,
                    intent_distribution=v2_intent_distribution,
                    hardware_recency_boost=v2_hardware_recency_boost,
                )
                drift_n = normalized_drift_penalty(hops)
                loc_c = float(local_conn_map.get(pid, 0.45))
                dom_m = domain_consistency_multiplier(node.get("domain"), query_domains_v2)
                if lite_semantic_control:
                    raw, drift_sub, loc_add, _dm = semantic_control_adjustments(
                        fused_base=fused,
                        local_conn=loc_c,
                        drift_pen=drift_n,
                        domain_mult=dom_m,
                        gamma=sc_gamma,
                        delta=sc_delta,
                    )
                    node["semantic_drift_penalty_norm"] = round(drift_n, 4)
                    node["semantic_drift_penalty_applied"] = round(float(drift_sub), 6)
                    node["local_semantic_connectivity"] = round(loc_c, 4)
                    node["local_semantic_connectivity_bonus"] = round(float(loc_add), 6)
                    node["domain_consistency_multiplier"] = round(dom_m, 4)
                    node["query_inferred_domains"] = sorted(query_domains_v2)
                else:
                    raw = fused * dom_m
                    node["semantic_drift_penalty_norm"] = 0.0
                    node["local_semantic_connectivity"] = round(loc_c, 4)
                    node["domain_consistency_multiplier"] = round(dom_m, 4)
                    node["query_inferred_domains"] = sorted(query_domains_v2)
                node["fused_score_pre_semantic_control"] = round(float(fused), 6)
                node["semantic_score"] = round(float(sem), 4)
                node["graph_score"] = round(float(gscore), 4)
                node["canonicality_score"] = round(float(canon), 4)
                node["semantic_fit_score"] = round(float(sem), 4)
                if metadata_only_mode:
                    raw *= 0.72
                if _ni == QueryIntent.HARDWARE_SYSTEM:
                    raw += 0.06 * anchor_proximity_strength_normalized(hops)
                node["relevance_raw"] = raw
                node["final_score"] = round(float(raw), 6)
                raw_scores.append(raw)
                continue

            rec = rec_map.get(pid, 0.0)
            ml = float(node.get("missing_link_score") or 0)
            rel = float(node.get("relation_expand_score", 0.0))
            q_match = self._paper_query_match_score(node, query_profile.get("query_terms") or [])
            node["query_match_score"] = q_match
            intent = (query_profile.get("query_intent") or "method").lower()
            # Intent-aware weighting; lite_field_aware prioritizes citation neighborhoods over lexical match.
            if lite_field_aware:
                if intent == "survey":
                    raw = citation_count * 0.42 + rec * 1.0 + ml * 1.85 + rel * 3.6 + q_match * 1.35
                elif intent == "recent":
                    year_bonus = 0.0
                    y = node.get("year")
                    if isinstance(y, int):
                        year_bonus = max(0.0, min(1.0, (y - 2018) / 8.0))
                    raw = citation_count * 0.28 + rec * 1.0 + ml * 1.65 + rel * 4.3 + q_match * 1.45 + year_bonus * 1.2
                elif intent == "application":
                    raw = citation_count * 0.28 + rec * 1.0 + ml * 2.0 + rel * 4.1 + q_match * 1.65
                else:
                    raw = citation_count * 0.28 + rec * 1.05 + ml * 2.0 + rel * 4.8 + q_match * 1.6
            elif intent == "survey":
                raw = citation_count * 0.55 + rec * 1.1 + ml * 1.6 + rel * 2.0 + q_match * 2.5
            elif intent == "recent":
                year_bonus = 0.0
                y = node.get("year")
                if isinstance(y, int):
                    year_bonus = max(0.0, min(1.0, (y - 2018) / 8.0))
                raw = citation_count * 0.25 + rec * 1.0 + ml * 1.5 + rel * 2.5 + q_match * 3.2 + year_bonus * 1.5
            elif intent == "application":
                raw = citation_count * 0.25 + rec * 1.0 + ml * 1.8 + rel * 2.8 + q_match * 3.0
            else:
                raw = citation_count * 0.25 + rec * 1.1 + ml * 1.9 + rel * 3.2 + q_match * 3.4
            # Canonical DPO anchor boost when query clearly targets DPO-like intent.
            qt = " ".join(query_profile.get("query_terms") or []).lower()
            if ("dpo" in qt or "direct" in qt and "preference" in qt and "optimization" in qt):
                nid = str(node.get("id") or "").lower()
                ttl = str(node.get("title") or "").lower()
                if "2305.18290" in nid or (
                    "direct preference optimization" in ttl and "secretly a reward model" in ttl
                ):
                    raw += 2.5
            # Field coherence & canonicality (Phase: stabilize ranking, penalize distant / lexical-only nodes).
            if lite_coherence_stabilization:
                pid = node["id"]
                hops = int(hop_map.get(pid, 99))
                node["anchor_graph_hops"] = hops
                decay_table = {0: 1.0, 1: 1.0, 2: 0.9, 3: 0.78, 4: 0.62}
                decay = decay_table.get(min(hops, 4), 0.48 if hops < 99 else 0.4)
                raw *= decay
                qm = float(node.get("query_match_score", 0.0))
                rel_sc = float(node.get("relation_expand_score", 0.0))
                if (
                    qm > 0.42
                    and rel_sc < 0.12
                    and node.get("seed_origin") == "discovered"
                    and not node.get("is_foundational_hub")
                ):
                    raw *= 0.8
                prx = merged_by_id.get(pid)
                cited_by_n = 0
                if prx:
                    for pv in prx.provenance or []:
                        if pv.source == "openalex" and isinstance(pv.raw, dict):
                            cited_by_n = int(pv.raw.get("cited_by_count") or 0)
                            break
                raw += math.log1p(max(0, cited_by_n)) * 0.11
                if node.get("is_foundational_hub"):
                    raw += 0.5
                if node.get("seed_origin") == "api_search":
                    raw += 0.32
            # Multi-stage semantic–graph ranking: semantic fit dominates; graph score enriches (Stage 4).
            if lite_semantic_graph_ranking:
                sem_fit = self._semantic_fit_score(node, query_profile)
                node["semantic_fit_score"] = round(float(sem_fit), 4)
                graph_component = float(raw)
                sem_scale = float(opts.get("semantic_fit_scale", 14.0) or 14.0)
                sem_component = sem_fit * sem_scale
                raw = sem_w * sem_component + graph_w * graph_component
                if sem_fit >= 0.78:
                    raw = max(raw, sem_component * 0.94 + 1.0)
                if sem_fit >= 0.72 and float(node.get("query_match_score", 0.0)) >= 0.48:
                    raw += 0.55
            # Hard quality gate: if no references/edges, keep ranking but mark metadata-only and soften scores.
            if metadata_only_mode:
                raw *= 0.72
            node["relevance_raw"] = raw
            raw_scores.append(raw)
        if lite_v2_ranking and lite_intent_verification and (qtext_for_v2 or "").strip():
            intent_verification_meta = apply_intent_verification_layer(
                graph_json["nodes"],
                qtext_for_v2,
                opts,
                semantic_fallback=self._semantic_fit_score,
                query_profile=query_profile,
            )
            raw_scores = [float(n["relevance_raw"]) for n in graph_json["nodes"]]
        if lite_v2_ranking:
            SRGApplicationService._apply_semantic_hard_floor_v2(graph_json["nodes"], opts)
            SRGApplicationService._apply_semantic_graph_top10_rule_a(graph_json["nodes"], opts)
            apply_hardware_top10_sem_rec_graph_lock(
                graph_json["nodes"],
                v2_intent,
                hardware_recency_boost=v2_hardware_recency_boost,
            )
            SRGApplicationService._apply_foundational_eligibility_flags(graph_json["nodes"], opts)
            raw_scores = [float(n["relevance_raw"]) for n in graph_json["nodes"]]
        max_raw = max(raw_scores) if raw_scores else 1.0
        if max_raw <= 0:
            max_raw = 1.0
        for node in graph_json["nodes"]:
            node["relevance_norm"] = float(node["relevance_raw"]) / max_raw
        self._apply_diversity_rerank(graph_json["nodes"], top_n=top_n)
        max_div = max(float(n.get("relevance_raw_diverse", 0.0)) for n in graph_json["nodes"]) if graph_json["nodes"] else 1.0
        if max_div <= 0:
            max_div = 1.0
        for node in graph_json["nodes"]:
            node["relevance_diverse_norm"] = float(node.get("relevance_raw_diverse", 0.0)) / max_div
            node["inclusion_reasons"] = self._inclusion_reasons_for_node(node)
            if metadata_only_mode:
                node["inclusion_reasons"] = list(dict.fromkeys(node["inclusion_reasons"] + ["metadata_only_mode"]))

        # If no foundational hubs were detected by strict criteria, promote top query-relevant nodes
        # so users can still see meaningful gold anchors in sparse metadata runs.
        if not any(bool(n.get("is_foundational_hub")) for n in graph_json["nodes"]):
            if lite_coherence_stabilization:
                promoted = sorted(
                    graph_json["nodes"],
                    key=lambda n: (
                        float(n.get("relation_expand_score", 0.0)),
                        -float(n.get("anchor_graph_hops", 99)),
                        float(n.get("relevance_diverse_norm", n.get("relevance_norm", 0.0))),
                    ),
                    reverse=True,
                )
                promoted_count = 0
                for n in promoted:
                    if int(n.get("anchor_graph_hops", 99)) > 6:
                        continue
                    if float(n.get("relation_expand_score", 0.0)) < 0.04 and float(n.get("query_match_score", 0.0)) < 0.38:
                        continue
                    n["is_foundational_hub"] = True
                    n["inclusion_reasons"] = list(dict.fromkeys((n.get("inclusion_reasons") or []) + ["promoted_anchor"]))
                    promoted_count += 1
                    if promoted_count >= 2:
                        break
            else:
                promoted = sorted(
                    graph_json["nodes"],
                    key=lambda n: (
                        float(n.get("query_match_score", 0.0)),
                        float(n.get("relevance_diverse_norm", n.get("relevance_norm", 0.0))),
                    ),
                    reverse=True,
                )
                promoted_count = 0
                for n in promoted:
                    if float(n.get("query_match_score", 0.0)) < 0.5:
                        continue
                    n["is_foundational_hub"] = True
                    n["inclusion_reasons"] = list(dict.fromkeys((n.get("inclusion_reasons") or []) + ["promoted_anchor"]))
                    promoted_count += 1
                    if promoted_count >= 2:
                        break

        if lite_v2_ranking:
            SRGApplicationService._apply_foundational_eligibility_flags(graph_json["nodes"], opts)

        if lite_coherence_stabilization:
            for node in graph_json["nodes"]:
                hops = int(node.get("anchor_graph_hops", 99))
                rel_sc = float(node.get("relation_expand_score", 0.0))
                if node.get("is_foundational_hub") or hops <= 1 or rel_sc >= 0.22:
                    node["field_coherence_tier"] = "core"
                elif hops <= 2 or rel_sc >= 0.1:
                    node["field_coherence_tier"] = "near_core"
                else:
                    node["field_coherence_tier"] = "peripheral"

        if lite_v2_ranking:
            qt_wim = (qtext_for_v2 or "").strip() or " ".join(
                str(t) for t in (query_profile.get("query_terms") or [])
            )
            for node in graph_json["nodes"]:
                node["why_it_matters"] = build_why_it_matters(
                    node, query_text=qt_wim, intent_label=v2_intent
                )
            ensure_why_it_matters_top5(
                graph_json["nodes"], query_text=qt_wim, intent_label=v2_intent
            )

        if metadata_only_mode:
            evidence_quality = "low"
        elif references_total >= 50 and evidence_edges >= 30:
            evidence_quality = "high"
        else:
            evidence_quality = "medium"

        for node in graph_json["nodes"]:
            if node.get("is_foundational_hub") and node.get("foundational_eligible", True):
                node["viz_color"] = "#D4AF37"
                node["viz_size"] = 44
            elif node.get("is_missing_link_candidate"):
                node["viz_color"] = "#CA8A04"
                node["viz_size"] = 30
            elif node.get("source_label") == "PDF" or node.get("seed_origin") == "uploaded_pdf":
                node["viz_color"] = "#1D4ED8"
                node["viz_size"] = 34
            else:
                node["viz_color"] = "#D4D4D8"
                node["viz_size"] = 16

        self._assign_foundational_graph_layout(graph_json["nodes"])

        n_gnodes = len(graph_json.get("nodes") or [])
        n_gedges = len(graph_json.get("edges") or [])
        n_api_seeds = sum(1 for s in effective_seeds if (s.get("origin") or "") == "api_search")
        rq_warnings: list[str] = []
        anchor_notes = list(getattr(self, "_last_seed_selection_notes", []) or [])
        if lite_retrieval_repair or lite_field_aware:
            if n_gnodes >= 2 and n_gedges < 2:
                rq_warnings.append(
                    "This map has very few citation links, so it may look disconnected. "
                    "Try a specific DOI or arXiv id, or run again if the APIs were rate-limited."
                )
            if n_gnodes >= 4 and int(references_total) < 20:
                rq_warnings.append(
                    "Reference metadata looks sparse — citation expansion may be incomplete until providers return full records."
                )
            if n_api_seeds < 3 and n_gnodes >= 2:
                rq_warnings.append(
                    "Few query anchors resolved — OpenAlex may have returned limited matches for this phrasing."
                )
        if lite_coherence_stabilization:
            lite_display_relevance_floor = float(opts.get("lite_user_graph_relevance_floor", 0.14) or 0.14)
        else:
            lite_display_relevance_floor = float(opts.get("lite_user_graph_relevance_floor", 0.0) or 0.0)
        v21_diag: dict[str, Any] = {}
        if lite_v2_ranking:
            v21_diag = lite_v21_observability_payload(
                nodes=graph_json["nodes"],
                edges=graph_json.get("edges") or [],
                references_total=int(references_total),
                n_candidates=n_gnodes,
                n_edges=n_gedges,
                intent=v2_intent,
                intent_confidence=v2_confidence,
                fusion_weights=v2_fusion,
            )
            v21_diag["semantic_control"] = {
                "enabled": lite_semantic_control,
                "semantic_control_local_gamma": sc_gamma,
                "semantic_control_drift_lambda": sc_delta,
                **aggregate_semantic_control_diagnostics(graph_json["nodes"]),
            }
            v21_diag["intent_verification"] = {
                "enabled": lite_intent_verification and bool((qtext_for_v2 or "").strip()),
                **intent_verification_meta,
            }
            v21_diag["roadmap"] = {
                "failure_taxonomy": list(FAILURE_TAXONOMY),
                "lite_query_conditioned_canonicality": lite_query_conditioned_canonicality,
                "lite_semantic_rank_hard_floor": bool(opts.get("lite_semantic_rank_hard_floor", True)),
                "lite_foundational_eligibility_rules": bool(opts.get("lite_foundational_eligibility_rules", True)),
            }
            v21_diag["roadmap_flags"] = {
                "query_conditioned_canonicality": bool(lite_query_conditioned_canonicality),
                "semantic_hard_floor": bool(opts.get("lite_semantic_rank_hard_floor", True)),
                "foundational_filtering": bool(opts.get("lite_foundational_eligibility_rules", True)),
                "clean_export_mode": bool(opts.get("export_clean_reading_mode", True)),
            }
        retrieval_quality: dict[str, Any] = {
            "warnings": rq_warnings,
            "anchor_resolution_notes": anchor_notes,
            "node_count": n_gnodes,
            "edge_count": n_gedges,
            "api_search_seed_count": n_api_seeds,
            "effective_seed_total": len(effective_seeds),
            "references_total": int(references_total),
            "lite_retrieval_repair": lite_retrieval_repair,
            "lite_coherence_stabilization": lite_coherence_stabilization,
            "lite_semantic_graph_ranking": lite_semantic_graph_ranking,
            "lite_v2_ranking": lite_v2_ranking,
            "v2_intent": v2_intent if lite_v2_ranking else "",
            "v2_intent_confidence": v2_confidence if lite_v2_ranking else 0.0,
            "v2_1_diagnostics": v21_diag if lite_v2_ranking else {},
            "lite_semantic_control": lite_semantic_control if lite_v2_ranking else False,
            "lite_intent_verification": lite_intent_verification if lite_v2_ranking else False,
        }

        return {
            "papers": graph_json["nodes"],
            "graph": graph_json,
            "metrics": {
                "co_citation": {f"{a}|{b}": s for (a, b), s in self.semantic_metrics.co_citation(graph).items()},
                "bibliographic_coupling": {
                    f"{a}|{b}": s for (a, b), s in self.semantic_metrics.bibliographic_coupling(graph).items()
                },
            },
            "quality_report": _report_to_json(quality),
            "discovery": {
                "recommendations": [asdict(r) for r in rec_list],
                "trends": self.discovery.trend_and_gap(graph),
            },
            "feedback_count": len(feedback),
            "generated_at": datetime.utcnow().isoformat(),
            "diagnostics": diagnostics,
            "requested_seeds": effective_seeds,
            "used_fallback": any(
                any(prov.source == "fallback" for prov in paper.provenance)
                for paper in merged_records
            ),
            "used_pending_metadata": any(
                any(prov.source == "pending" for prov in paper.provenance) for paper in merged_records
            ),
            "source_summary": self._source_summary(merged_records),
            "expanded_added_count": sum(
                1 for n in graph_json["nodes"] if n.get("seed_origin") == "discovered"
            ),
            "expand_diagnostics": {
                **getattr(self, "_last_expand_stats", {}),
                "new_nodes_added": sum(
                    1 for n in graph_json["nodes"] if n.get("seed_origin") == "discovered"
                ),
                "expand_display_top_n": top_n,
                "expand_candidate_pool_max": EXPAND_CANDIDATE_POOL_MAX,
                "expand_relation_filter_active": expand_relation_filter_active,
                "coupling_threshold": coupling_threshold,
                "enable_two_hop": enable_two_hop,
                "concept_threshold": concept_threshold,
                "relation_topic_weight": relation_topic_weight,
                "lite_field_aware": lite_field_aware,
                "lite_retrieval_repair": lite_retrieval_repair,
                "lite_coherence_stabilization": lite_coherence_stabilization,
                "lite_semantic_graph_ranking": lite_semantic_graph_ranking,
                "semantic_graph_sem_weight": sem_w,
                "semantic_graph_graph_weight": graph_w,
                "lite_v2_ranking": lite_v2_ranking,
                "v2_intent": v2_intent,
                "v2_intent_confidence": v2_confidence,
                "v2_fusion_weights": {
                    "semantic": v2_fusion[0],
                    "graph": v2_fusion[1],
                    "canonicality": v2_fusion[2],
                },
                "v2_1_diagnostics": v21_diag,
                "lite_semantic_control": lite_semantic_control,
                "semantic_control_local_gamma": sc_gamma,
                "semantic_control_drift_lambda": sc_delta,
                "lite_intent_verification": lite_intent_verification,
                "foundational_boost": foundational_boost,
                "openalex_skipped_ids": list(dict.fromkeys(skipped_ids)),
                "pool_fill_notes": self._expand_pool_fill_notes(
                    effective_seeds,
                    hop_seeds,
                    diagnostics,
                    expand_relation_filter_active,
                ),
            },
            "expand_display_top_n": top_n,
            "expand_relation_filter_active": expand_relation_filter_active,
            "auto_escalated": auto_escalated,
            "auto_escalation_message": (
                "Literature is being deepened automatically."
                if auto_escalated
                else ""
            ),
            "pdf_metadata_hints": [
                {
                    "provider": s["provider"].lower(),
                    "id": s["id"].strip(),
                    "badge": self._resolution_badge(s.get("metadata_resolution")),
                }
                for s in effective_seeds
                if s.get("origin") == "uploaded_pdf"
            ],
            "missing_link_candidates": sorted(
                [
                    self._enrich_missing_link_candidate(pid, meta, merged_by_id, pii_tail)
                    for pid, meta in missing_links.items()
                    if meta["co_cited_by_uploaded_count"] >= 3
                ],
                key=lambda x: (
                    float(x.get("pii_concept_match", 0.0)),
                    int(x.get("openalex_cited_by_count", 0)),
                    int(x.get("co_cited_by_uploaded_count", 0)),
                ),
                reverse=True,
            ),
            "ingestion_stats": dict(getattr(self.orchestrator, "last_ingestion_stats", {}) or {}),
            "discipline_applied": dom,
            "pdf_quality_gate": dict(getattr(self, "_last_pdf_gate_stats", {}) or {}),
            "query_profile": (
                {
                    **query_profile,
                    "intent_mode_v2": v2_intent,
                    "intent_confidence_v2": v2_confidence,
                    "intent_distribution_v2": v2_intent_distribution,
                    "distribution": v2_intent_distribution,
                    "hardware_recency_fusion_boost": v2_hardware_recency_boost,
                    "query_inferred_domains": sorted(query_domains_v2),
                }
                if lite_v2_ranking
                else query_profile
            ),
            "metadata_only_mode": metadata_only_mode,
            "evidence_quality": evidence_quality,
            "evidence_stats": {
                "references_total": int(references_total),
                "graph_edges_total": int(evidence_edges),
            },
            "lite_display_relevance_floor": lite_display_relevance_floor,
            "retrieval_quality": retrieval_quality,
        }

    def rank_nodes_v2(
        self,
        nodes: list[dict[str, Any]],
        query: str,
        *,
        merged_by_id: dict[str, PaperRecord],
        edges: list[dict[str, Any]],
        query_terms: list[str] | None = None,
        anchor_ids: set[str] | None = None,
        semantic_control: bool = True,
    ) -> list[dict[str, Any]]:
        """
        SRG Lite v2.1 + semantic-control ordering — same signals as ``lite_v2_ranking`` when anchors are passed;
        without ``anchor_ids``, hop drift is skipped (graph coherence from PR/edges only).
        """
        qp: dict[str, Any] = {
            "query_text": (query or "").strip(),
            "query_terms": list(query_terms or []),
            "query_intent": "method",
        }
        qstrip = (query or "").strip()
        ids = [str(n["id"]) for n in nodes]
        hop_map: dict[str, int] = {}
        if anchor_ids:
            hop_map = SRGApplicationService._anchor_hop_distances(ids, edges, anchor_ids)
        pr_n, eq_n, ctx_n, graph_combined = compute_graph_score_bundle(edges, ids, hop_map if hop_map else None)
        out_d = Counter(str(e["source"]) for e in edges)
        in_d = Counter(str(e["target"]) for e in edges)
        deg_map = {nid: int(out_d[nid] + in_d[nid]) for nid in ids}
        graph_for = (
            hub_suppressed_graph_scores(graph_combined, deg_map) if semantic_control else dict(graph_combined)
        )
        sem_by_id = {n["id"]: float(self._semantic_fit_score(n, qp)) for n in nodes}
        local_conn_map = local_semantic_connectivity(ids, sem_by_id, edges)
        qdom = infer_query_domains(qstrip)
        icr = classify_query(qstrip, {})
        intent = str(icr.get("intent") or "")
        idist = dict(icr.get("intent_distribution") or icr.get("distribution") or {})
        hw_boost = bool(icr.get("hardware_recency_fusion_boost"))
        gamma, delta = 0.09, 0.12
        ranked: list[tuple[float, dict[str, Any]]] = []
        for node in nodes:
            pid = node["id"]
            sem = sem_by_id.get(pid, 0.0)
            gs = float(graph_for.get(pid, 0.45))
            inbound = SRGApplicationService._openalex_cited_by_count(merged_by_id, pid)
            canon = compute_canonicality(
                node,
                qstrip,
                intent,
                inbound_citations=inbound,
                is_in_survey_context=infer_survey_context_flag(node),
            )
            _ni_r = normalize_query_intent(intent)
            recency_v = (
                SRGApplicationService._recency_alignment_score(node)
                if _ni_r in RECENCY_TRIPLE_INTENTS
                else None
            )
            fused = compute_final_score(
                sem,
                gs,
                canon,
                intent,
                recency=recency_v,
                intent_distribution=idist,
                hardware_recency_boost=hw_boost,
            )
            dom_m = domain_consistency_multiplier(node.get("domain"), qdom)
            if semantic_control and hop_map:
                drift_n = normalized_drift_penalty(int(hop_map.get(pid, 99)))
                loc_c = float(local_conn_map.get(pid, 0.45))
                score, _, _, _ = semantic_control_adjustments(
                    fused_base=fused,
                    local_conn=loc_c,
                    drift_pen=drift_n,
                    domain_mult=dom_m,
                    gamma=gamma,
                    delta=delta,
                )
            else:
                score = max(0.0, fused * dom_m)
            node["pagerank_norm"] = round(float(pr_n.get(pid, 0.0)), 4)
            node["edge_quality_norm"] = round(float(eq_n.get(pid, 0.0)), 4)
            node["context_score_norm"] = round(float(ctx_n.get(pid, 0.5)), 4)
            node["graph_score"] = round(float(gs), 4)
            node["semantic_score"] = round(float(sem), 4)
            node["canonicality_score"] = round(float(canon), 4)
            node["final_score"] = round(float(score), 6)
            ranked.append((score, node))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in ranked]

    def submit_merge_feedback(self, left_id: str, right_id: str, accepted: bool = True) -> None:
        self.feedback_store.add(
            FeedbackEvent(event_type="merge_hint", left_id=left_id, right_id=right_id, accepted=accepted)
        )

    def submit_bad_citation_feedback(self, source_id: str, target_id: str) -> None:
        self.feedback_store.add(
            FeedbackEvent(event_type="bad_citation", left_id=source_id, right_id=target_id, accepted=False)
        )

    def submit_pdf_quality_override(self, label: str, mapping_score: float, provider: str, seed_id: str) -> None:
        self.feedback_store.add(
            FeedbackEvent(
                event_type="pdf_quality_override",
                left_id=label,
                right_id=f"{provider}:{seed_id}|score={mapping_score:.4f}",
                accepted=True,
            )
        )

    @staticmethod
    def _canonical_discipline(discipline: str | None) -> str | None:
        if not discipline or discipline.strip().lower() in {"", "default"}:
            return None
        return discipline.strip().lower()

    def _build_pending_seed_placeholders(self, seeds: list[dict[str, Any]]) -> list[PaperRecord]:
        """When ingestion returns no rows — pending placeholders (never empty 'Fallback' cache rows)."""
        records: list[PaperRecord] = []
        for idx, seed in enumerate(seeds):
            provider = str(seed.get("provider", "unknown"))
            seed_id = str(seed.get("id", f"seed-{idx}"))
            canonical_id = f"{provider}:{seed_id}"
            refs: list[str] = []
            if idx > 0:
                refs.append(
                    f"{seeds[idx - 1].get('provider', 'unknown')}:{seeds[idx - 1].get('id', f'seed-{idx - 1}')}"
                )
            records.append(
                PaperRecord(
                    canonical_id=canonical_id,
                    title=f"Pending metadata: {seed_id}",
                    abstract="No provider rows returned for this run; use Fetch metadata or retry after backoff.",
                    year=datetime.utcnow().year,
                    venue="pending-ingest",
                    domain="cs",
                    doi=seed_id if provider in {"doi", "crossref"} else None,
                    arxiv_id=seed_id if provider == "arxiv" else None,
                    openalex_id=seed_id if provider == "openalex" else None,
                    authors=[],
                    references=refs,
                    provenance=[Provenance(source="pending", source_id=seed_id, raw={"batch": "no_rows"})],
                )
            )
        return records

    def _batch_prefetch_openalex_seed_works(self, seeds: list[dict[str, Any]], diagnostics: list[str]) -> None:
        """Populate SQLite cache with full work JSON via ids.openalex batch filter (50 per request)."""
        uniq: list[str] = []
        seen: set[str] = set()
        for s in seeds:
            if (s.get("provider") or "").strip().lower() != "openalex":
                continue
            oid = (s.get("id") or "").strip()
            if not oid or not re.match(r"^W\d+$", oid) or oid in seen:
                continue
            seen.add(oid)
            uniq.append(oid)
        if not uniq:
            return
        oa = OpenAlexClient()
        try:
            merged: dict[str, dict[str, Any]] = {}
            for i in range(0, len(uniq), 50):
                chunk = uniq[i : i + 50]
                self.orchestrator.budget.acquire("openalex")
                part = oa.fetch_works_batch_by_openalex_tails(chunk, chunk_size=50)
                merged.update(part)
            for tail, work in merged.items():
                self.orchestrator.cache.put("openalex", tail, work)
            diagnostics.append(
                f"OpenAlex batch prefetch: cached {len(merged)} / {len(uniq)} requested work payloads."
            )
        except requests.RequestException as exc:
            diagnostics.append(f"OpenAlex batch prefetch failed (non-fatal): {exc}")

    @staticmethod
    def _append_skipped_openalex(skipped_ids: list[str], tail: str) -> None:
        t = (tail or "").strip()
        if t and t not in skipped_ids:
            skipped_ids.append(t)

    def _collect_openalex_w_tails_from_seeds(self, seeds: list[dict[str, Any]]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for s in seeds:
            if (s.get("provider") or "").strip().lower() != "openalex":
                continue
            sid = (s.get("id") or "").strip()
            if not re.match(r"^W\d+$", sid) or sid in seen:
                continue
            seen.add(sid)
            out.append(sid)
        return out

    def _collect_openalex_w_reference_tails_not_in_nodes(self, records: list[PaperRecord]) -> list[str]:
        have: set[str] = set()
        for p in records:
            oid = (p.openalex_id or "").strip().upper()
            if oid and re.match(r"^W\d+$", oid, re.IGNORECASE):
                have.add(oid.upper())
            low = (p.canonical_id or "").lower()
            if low.startswith("openalex:"):
                tail = low.split(":", 1)[-1].strip().upper()
                if re.match(r"^W\d+$", tail, re.IGNORECASE):
                    have.add(tail)
        out: list[str] = []
        seen: set[str] = set()
        for p in records:
            for r in p.references or []:
                rr = (r or "").strip()
                if not re.match(r"^W\d+$", rr) or rr.upper() in seen:
                    continue
                if rr.upper() in have:
                    continue
                seen.add(rr.upper())
                out.append(rr)
        return out

    def _ensure_openalex_tails_in_store(
        self,
        tails: list[str],
        diagnostics: list[str],
        skipped_ids: list[str],
    ) -> None:
        """Batch-fetch missing OpenAlex work JSON into CacheStore (merged-id friendly filter)."""
        oa = OpenAlexClient()
        uniq: list[str] = []
        seen: set[str] = set()
        for t in tails:
            u = (t or "").strip()
            if not re.match(r"^W\d+$", u) or u in seen:
                continue
            seen.add(u)
            uniq.append(u)
        if not uniq:
            return
        missing: list[str] = []
        for u in uniq:
            hit = self.orchestrator.cache.get("openalex", u)
            if hit is not None and openalex_fetch_payload_valid(hit):
                continue
            missing.append(u)
        if not missing:
            return
        try:
            merged: dict[str, dict[str, Any]] = {}
            for i in range(0, len(missing), 50):
                chunk = missing[i : i + 50]
                self.orchestrator.budget.acquire("openalex")
                part = oa.fetch_works_batch_by_openalex_tails(chunk, chunk_size=50)
                merged.update(part)
            for tail, work in merged.items():
                if openalex_fetch_payload_valid(work):
                    self.orchestrator.cache.put("openalex", tail, work)
            for t in missing:
                got = merged.get(t)
                if not openalex_fetch_payload_valid(got):
                    self._append_skipped_openalex(skipped_ids, t)
        except requests.RequestException as exc:
            diagnostics.append(f"OpenAlex batch cache warm failed: {exc}")

    def _ensure_pii_concept_tail(self, diagnostics: list[str]) -> str | None:
        """Resolve OpenAlex PII-like concept id once (cached) for reporting / concept-match scores."""
        if self._cached_pii_concept_tail:
            return self._cached_pii_concept_tail
        oa = OpenAlexClient()
        try:
            self.orchestrator.budget.acquire("openalex")
            concepts = oa._get("/concepts", params={"search": "personal identifiable information", "per-page": 12})
        except requests.RequestException as exc:
            diagnostics.append(f"PII concept tail lookup failed: {exc}")
            return None
        for c in concepts.get("results", []) or []:
            if not isinstance(c, dict):
                continue
            dn = (c.get("display_name") or "").lower()
            if "personal" in dn and "identif" in dn:
                cid = (c.get("id") or "").rsplit("/", maxsplit=1)[-1]
                if cid:
                    self._cached_pii_concept_tail = cid
                    return cid
        return None

    @staticmethod
    def _normalize_citation_ref_key(ref: str) -> str:
        r = (ref or "").strip()
        if not r:
            return ""
        if re.match(r"^W\d+$", r, flags=re.IGNORECASE):
            return r.upper()
        nd = _normalize_doi(r)
        return (nd or r).strip().lower()

    def _record_is_uploaded_pdf(self, paper: PaperRecord, origin_map: dict[str, str]) -> bool:
        keys = [
            f"doi:{(paper.doi or '').lower()}",
            f"openalex:{(paper.openalex_id or '').lower()}",
            f"arxiv:{(paper.arxiv_id or '').lower()}",
            (paper.canonical_id or "").lower(),
        ]
        for k in keys:
            if k in origin_map and origin_map[k] == "uploaded_pdf":
                return True
        return False

    def _existing_identifier_keys(self, records: list[PaperRecord]) -> set[str]:
        s: set[str] = set()
        for p in records:
            s.add(p.canonical_id.lower())
            if p.openalex_id:
                oid = p.openalex_id.strip()
                s.add(oid.upper())
                s.add(f"openalex:{oid.lower()}")
            if p.doi:
                nd = _normalize_doi(p.doi)
                if nd:
                    s.add(f"doi:{nd}")
            if p.arxiv_id:
                s.add(f"arxiv:{normalize_arxiv_list_id(p.arxiv_id).lower()}")
        return s

    @staticmethod
    def _graph_has_ref_key(keys: set[str], ref_key: str) -> bool:
        if not ref_key:
            return True
        if re.match(r"^W\d+$", ref_key, flags=re.IGNORECASE):
            u = ref_key.upper()
            return f"openalex:{u.lower()}" in keys or u in keys
        return f"doi:{ref_key}" in keys or ref_key in keys

    def _all_openalex_w_tails_for_batch_warm(self, records: list[PaperRecord]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for p in records:
            for t in [p.openalex_id, *(p.references or [])]:
                u = (str(t) or "").strip()
                if not re.match(r"^W\d+$", u, flags=re.IGNORECASE):
                    continue
                uu = u.upper()
                if uu not in seen:
                    seen.add(uu)
                    out.append(uu)
        return out

    def _fetch_foundational_work_record(
        self, ref_key: str, oa: OpenAlexClient, diagnostics: list[str], skipped_ids: list[str]
    ) -> PaperRecord | None:
        try:
            self.orchestrator.budget.acquire("openalex")
            if re.match(r"^W\d+$", ref_key, flags=re.IGNORECASE):
                wk = ref_key.upper()
                batch = oa.fetch_works_batch_by_openalex_tails([wk], chunk_size=50)
                w = batch.get(wk)
                if not openalex_fetch_payload_valid(w):
                    return None
                self.orchestrator.cache.put("openalex", wk, w)
                return oa.normalize(w)
            nd = ref_key if ref_key.startswith("10.") else ref_key
            resp = oa.fetch_by_doi(nd)
            work = openalex_response_first_work(resp)
            if not openalex_fetch_payload_valid(work):
                return None
            oid = (work.get("id") or "").rsplit("/", maxsplit=1)[-1]
            if oid:
                self.orchestrator.cache.put("openalex", oid, work)
            return oa.normalize(work)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.status_code == 404:
                diagnostics.append(f"Foundational: merged or invalid id for ref {ref_key[:48]} — skipped.")
                if re.match(r"^W\d+$", ref_key, flags=re.IGNORECASE):
                    self._append_skipped_openalex(skipped_ids, ref_key.upper())
            else:
                diagnostics.append(f"Foundational fetch failed for {ref_key[:48]}: {exc}")
            return None
        except requests.RequestException as exc:
            diagnostics.append(f"Foundational fetch failed for {ref_key[:48]}: {exc}")
            return None

    def _inject_foundational_literature(
        self,
        records: list[PaperRecord],
        origin_map: dict[str, str],
        diagnostics: list[str],
        skipped_ids: list[str],
    ) -> None:
        """
        (1) If ≥3 uploaded PDF rows cite the same unresolved DOI/W, ingest that hub as a golden node.
        (2) 2-hop backwards: for papers your PDFs cite (already in graph), add top cited_by_count among their
        OpenAlex referenced_works not yet in the graph.
        """
        if not any(self._record_is_uploaded_pdf(p, origin_map) for p in records):
            return
        keys = self._existing_identifier_keys(records)
        nodes = {p.canonical_id: p for p in records}
        cited_by_sources: dict[str, set[str]] = defaultdict(set)
        for p in records:
            if not self._record_is_uploaded_pdf(p, origin_map):
                continue
            seen_r: set[str] = set()
            for r in p.references or []:
                rk = self._normalize_citation_ref_key(str(r))
                if not rk or rk in seen_r:
                    continue
                seen_r.add(rk)
                cited_by_sources[rk].add(p.canonical_id)

        oa = OpenAlexClient()
        max_nodes = 24
        top_per_parent = 5
        max_intermediates = 12
        appended = 0

        for ref_key, sources in sorted(cited_by_sources.items(), key=lambda x: (-len(x[1]), x[0])):
            if len(sources) < 3:
                continue
            if self._graph_has_ref_key(keys, ref_key):
                continue
            if appended >= max_nodes:
                break
            rec = self._fetch_foundational_work_record(ref_key, oa, diagnostics, skipped_ids)
            if rec is None:
                continue
            rec.provenance.append(
                Provenance(
                    source="foundational_hub",
                    source_id="co_citation_3plus",
                    raw={"cited_by_uploads": sorted(sources), "ref_key": ref_key},
                )
            )
            records.append(rec)
            keys.add(rec.canonical_id.lower())
            if rec.openalex_id:
                oid = rec.openalex_id.strip()
                keys.add(oid.upper())
                keys.add(f"openalex:{oid.lower()}")
            if rec.doi:
                nd = _normalize_doi(rec.doi)
                if nd:
                    keys.add(f"doi:{nd}")
            nodes[rec.canonical_id] = rec
            appended += 1

        if appended:
            diagnostics.append(
                f"Foundational mining: added {appended} hub paper(s) cited by ≥3 distinct uploaded PDF rows."
            )

        nodes = {p.canonical_id: p for p in records}
        intermediates: list[PaperRecord] = []
        for p in records:
            if not self._record_is_uploaded_pdf(p, origin_map):
                continue
            for r in p.references or []:
                tid = self.graph_builder.resolve_reference_to_canonical(str(r), nodes)
                if tid:
                    tp = nodes.get(tid)
                    if tp and tp not in intermediates:
                        intermediates.append(tp)
                if len(intermediates) >= max_intermediates:
                    break
            if len(intermediates) >= max_intermediates:
                break

        hop_added = 0
        for inter in intermediates:
            if appended + hop_added >= max_nodes:
                break
            oid = (inter.openalex_id or "").strip()
            if not oid or not re.match(r"^W\d+$", oid):
                continue
            try:
                self.orchestrator.budget.acquire("openalex")
                work = self.orchestrator.cache.get("openalex", oid)
                if work is None or not openalex_fetch_payload_valid(work):
                    work = oa.fetch_paper(oid)
                    if openalex_fetch_payload_valid(work):
                        self.orchestrator.cache.put("openalex", oid, work)
                if not isinstance(work, dict):
                    continue
                tails = [t for t in OpenAlexClient.referenced_work_tail_ids(work) if re.match(r"^W\d+$", t)]
                if not tails:
                    continue
                self.orchestrator.budget.acquire("openalex")
                batch = oa.fetch_works_batch_by_openalex_tails(tails[:50], chunk_size=50)
                ranked = sorted(
                    ((t, batch.get(t)) for t in tails if openalex_fetch_payload_valid(batch.get(t))),
                    key=lambda x: int((x[1] or {}).get("cited_by_count") or 0),
                    reverse=True,
                )
                for tail, w in ranked[:top_per_parent]:
                    if appended + hop_added >= max_nodes:
                        break
                    if self._graph_has_ref_key(keys, tail.upper()):
                        continue
                    try:
                        child = oa.normalize(w)
                    except Exception:
                        continue
                    child.provenance.append(
                        Provenance(
                            source="foundational_hub",
                            source_id="top_cited_ref_of_upload_cited",
                            raw={"via_intermediate": inter.canonical_id, "openalex_tail": tail},
                        )
                    )
                    records.append(child)
                    keys.add(child.canonical_id.lower())
                    if child.openalex_id:
                        o2 = child.openalex_id.strip()
                        keys.add(o2.upper())
                        keys.add(f"openalex:{o2.lower()}")
                    if child.doi:
                        nd2 = _normalize_doi(child.doi)
                        if nd2:
                            keys.add(f"doi:{nd2}")
                    nodes[child.canonical_id] = child
                    hop_added += 1
            except requests.RequestException as exc:
                diagnostics.append(f"Foundational 2-hop OpenAlex fetch failed for {oid}: {exc}")

        if hop_added:
            diagnostics.append(
                f"Foundational mining: added {hop_added} top-cited reference(s) from papers your PDFs cite."
            )

    @staticmethod
    def _is_placeholder_record(rec: PaperRecord) -> bool:
        if any(p.source in ("fallback", "pending") for p in (rec.provenance or [])):
            return True
        t = (rec.title or "")
        return "Fallback record" in t or t.startswith("Pending metadata")

    def _hydrate_metadata(
        self,
        records: list[PaperRecord],
        diagnostics: list[str],
        skipped_ids: list[str],
    ) -> tuple[list[PaperRecord], set[str]]:
        """Replace Fallback / pending OpenAlex stubs using batch filter (up to 3 rounds); mark unrecoverable W… as graph_noise."""
        oa = OpenAlexClient()
        for round_ix in range(3):
            idxs = [
                i
                for i, r in enumerate(records)
                if self._is_placeholder_record(r) and (r.openalex_id or "").strip()
            ]
            if not idxs:
                break
            tails = sorted(
                {
                    (records[i].openalex_id or "").strip()
                    for i in idxs
                    if re.match(r"^W\d+$", (records[i].openalex_id or "").strip())
                }
            )
            if not tails:
                break
            try:
                self.orchestrator.budget.acquire("openalex")
                batch = oa.fetch_works_batch_by_openalex_tails(tails, chunk_size=50)
            except requests.RequestException as exc:
                if round_ix == 2:
                    diagnostics.append(f"Metadata hydration batch failed (after 3 rounds): {exc}")
                batch = {}
            for i in idxs:
                rec = records[i]
                wid = (rec.openalex_id or "").strip()
                if not wid or not re.match(r"^W\d+$", wid):
                    continue
                work = batch.get(wid)
                if not work:
                    continue
                try:
                    fresh = oa.normalize(work)
                    records[i] = fresh
                except Exception as exc:  # pragma: no cover
                    diagnostics.append(f"Hydrate normalize failed for {wid}: {exc}")
        noise_ids: set[str] = set()
        for r in records:
            if not self._is_placeholder_record(r):
                continue
            oid = (r.openalex_id or "").strip()
            if re.match(r"^W\d+$", oid):
                noise_ids.add(r.canonical_id)
                self._append_skipped_openalex(skipped_ids, oid)
        still = sum(1 for r in records if self._is_placeholder_record(r))
        if still:
            diagnostics.append(
                f"Hydration: {still} placeholder rows remain after batch retries; "
                f"unresolved OpenAlex W… stubs flagged graph_noise ({len(noise_ids)} tails)."
            )
        return records, noise_ids

    def hydrate_graph_node_metadata(self, payload: dict[str, Any], node_id: str) -> bool:
        """Sidebar manual refresh: re-fetch OpenAlex work and merge into payload papers + graph nodes."""
        oa = OpenAlexClient()
        papers = payload.get("papers") or []
        node = next((n for n in papers if n.get("id") == node_id), None)
        if not node:
            return False
        oid = (node.get("openalex_id") or "").strip()
        if not oid and node_id.lower().startswith("openalex:"):
            oid = node_id.split(":", maxsplit=1)[-1].strip()
        if not oid or not re.match(r"^W\d+$", oid):
            return False
        try:
            self.orchestrator.budget.acquire("openalex")
            batch = oa.fetch_works_batch_by_openalex_tails([oid], chunk_size=50)
            work = batch.get(oid)
            if not isinstance(work, dict) or not work.get("id"):
                return False
            rec = oa.normalize(work)
        except requests.RequestException:
            return False
        fresh = _paper_to_json(rec)
        preserve = (
            "seed_origin",
            "source_label",
            "viz_color",
            "viz_size",
            "relation_expand_score",
            "expand_explanation",
            "expand_rank",
            "relevance_norm",
            "relevance_raw",
            "citation_count",
            "missing_link_score",
            "is_missing_link_candidate",
            "is_foundational_hub",
            "pii_concept_match",
            "viz_x",
            "viz_y",
            "viz_physics_fixed",
        )
        merged = {**fresh, **{k: node[k] for k in preserve if k in node}}
        node.update(merged)
        for gn in (payload.get("graph") or {}).get("nodes") or []:
            if gn.get("id") == node_id:
                merged_g = {**fresh, **{k: gn[k] for k in preserve if k in gn}}
                gn.update(merged_g)
                break
        try:
            self.orchestrator.cache.put("openalex", oid, work)
        except Exception:
            pass
        return True

    def _apply_record_year_domain_hard_fix(self, records: list[PaperRecord]) -> None:
        """Force year from arXiv / 4-digit patterns; fill unknown domain from venue then OpenAlex raw."""
        for p in records:
            y = p.year
            if isinstance(y, str) and y.strip().lower() == "unknown":
                p.year = None
                y = None
            if y is None or (isinstance(y, int) and y <= 0):
                if p.arxiv_id:
                    inferred = year_from_arxiv_list_id(p.arxiv_id)
                    if inferred is not None:
                        p.year = inferred
                if p.year is None:
                    blob = " ".join(
                        str(x)
                        for x in [p.doi or "", p.arxiv_id or "", p.canonical_id or "", p.venue or "", p.title or ""]
                        if x
                    )
                    for m in re.finditer(r"\b(19[89]\d|20[0-3]\d)\b", blob):
                        cand = int(m.group(1))
                        if 1980 <= cand <= 2035:
                            p.year = cand
                            break
            d = (str(p.domain).strip().lower() if p.domain is not None else "")
            if d in ("", "unknown"):
                p.domain = None
            if p.domain is None and p.venue:
                vd = domain_from_venue_name(p.venue)
                if vd:
                    p.domain = vd
            if p.domain is None:
                for prov in p.provenance or []:
                    if prov.source == "openalex" and isinstance(prov.raw, dict):
                        inferred_d = domain_from_openalex_payload(prov.raw)
                        if inferred_d:
                            p.domain = inferred_d
                            break

    def _force_include_incoming_citers(
        self,
        records: list[PaperRecord],
        diagnostics: list[str],
        *,
        max_extra: int = 36,
    ) -> None:
        """
        Citation recovery: include incoming citers for current node set, even when outgoing refs are sparse.
        """
        oa = OpenAlexClient()
        have = self._existing_identifier_keys(records)
        extras: list[PaperRecord] = []
        for base in records[:24]:
            if len(extras) >= max_extra:
                break
            oid = (base.openalex_id or "").strip()
            if not re.match(r"^W\d+$", oid):
                continue
            work = self.orchestrator.cache.get("openalex", oid)
            if not openalex_fetch_payload_valid(work):
                try:
                    self.orchestrator.budget.acquire("openalex")
                    work = oa.fetch_paper(oid)
                    if openalex_fetch_payload_valid(work):
                        self.orchestrator.cache.put("openalex", oid, work)
                except requests.RequestException:
                    continue
            cb = (work or {}).get("cited_by_api_url") if isinstance(work, dict) else None
            if not isinstance(cb, str) or not cb.strip():
                continue
            try:
                self.orchestrator.budget.acquire("openalex")
                rows = oa._get(cb.replace(oa.base_url, ""), params={"per-page": 30}).get("results") or []
            except requests.RequestException:
                continue
            for row in rows:
                if len(extras) >= max_extra:
                    break
                if not isinstance(row, dict):
                    continue
                try:
                    rec = oa.normalize(row)
                except Exception:
                    continue
                key = rec.canonical_id.lower()
                if key in have:
                    continue
                have.add(key)
                rec.provenance.append(
                    Provenance(
                        source="incoming_citation_recovery",
                        source_id=base.canonical_id,
                        raw={"recovered_from": oid},
                    )
                )
                extras.append(rec)
        if extras:
            records.extend(extras)
            diagnostics.append(
                f"Citation recovery: included {len(extras)} incoming-citation papers to increase connectivity."
            )

    def _promote_resolved_reference_hubs(
        self,
        records: list[PaperRecord],
        resolved_ref_tails: set[str],
        diagnostics: list[str],
        skipped_ids: list[str],
        *,
        max_ref_seeds: int = 2,
    ) -> None:
        """
        Metadata hydration priority when resolved refs are scarce:
        pick a few resolved ids, fetch their referenced_works in batch, and promote top cited as foundational hubs.
        """
        oa = OpenAlexClient()
        candidates = [t for t in sorted(resolved_ref_tails) if re.match(r"^W\d+$", str(t))]
        if not candidates:
            return
        have = self._existing_identifier_keys(records)
        added = 0
        for tail in candidates[:max_ref_seeds]:
            try:
                work = self.orchestrator.cache.get("openalex", tail)
                if not openalex_fetch_payload_valid(work):
                    self.orchestrator.budget.acquire("openalex")
                    part = oa.fetch_works_batch_by_openalex_tails([tail], chunk_size=50)
                    work = part.get(tail)
                if not openalex_fetch_payload_valid(work):
                    self._append_skipped_openalex(skipped_ids, tail)
                    continue
                refs = [r for r in OpenAlexClient.referenced_work_tail_ids(work) if re.match(r"^W\d+$", r)]
                if not refs:
                    continue
                self.orchestrator.budget.acquire("openalex")
                batch = oa.fetch_works_batch_by_openalex_tails(refs[:50], chunk_size=50)
                ranked = sorted(
                    [w for w in batch.values() if isinstance(w, dict)],
                    key=lambda w: int(w.get("cited_by_count") or 0),
                    reverse=True,
                )
                for w in ranked[:4]:
                    rec = oa.normalize(w)
                    if rec.canonical_id.lower() in have:
                        continue
                    rec.provenance.append(
                        Provenance(
                            source="foundational_hub",
                            source_id="resolved_ref_priority",
                            raw={"seed_tail": tail},
                        )
                    )
                    records.append(rec)
                    have.add(rec.canonical_id.lower())
                    added += 1
            except requests.RequestException:
                continue
        if added:
            diagnostics.append(
                f"Low-resolved-ref priority hydration: promoted {added} foundational candidates from resolved ids."
            )

    @staticmethod
    def _assign_foundational_graph_layout(nodes: list[dict[str, Any]]) -> None:
        """
        Pin bright-gold foundational hubs near the graph origin (vis.js x/y).

        PR G2 — layout ordering is intentionally **not** the same comparator as
        general relevance ranking (uses semantic + intent + hops, not degree).
        """
        import math

        hubs = [n for n in nodes if n.get("is_foundational_hub") and n.get("foundational_eligible", True)]
        if not hubs:
            return
        # PR C2 — layout order uses semantic + intent + proximity, not graph degree / citation_count.
        hubs.sort(
            key=lambda n: (
                float(n.get("semantic_score", 0.0) or 0.0)
                + float(n.get("intent_similarity") or n.get("semantic_score", 0.0) or 0.0),
                -int(n.get("anchor_graph_hops", 99)),
            ),
            reverse=True,
        )
        hubs[0]["viz_x"] = 0.0
        hubs[0]["viz_y"] = 0.0
        hubs[0]["viz_physics_fixed"] = True
        if len(hubs) == 1:
            return
        r = 48.0
        ring = hubs[1:]
        for i, n in enumerate(ring):
            ang = (2.0 * math.pi * i) / max(len(ring), 1)
            n["viz_x"] = r * math.cos(ang)
            n["viz_y"] = r * math.sin(ang)
            n["viz_physics_fixed"] = True

    def _source_summary(self, records: list[PaperRecord]) -> dict[str, int]:
        summary: dict[str, int] = {}
        for record in records:
            if not record.provenance:
                summary["unknown"] = summary.get("unknown", 0) + 1
                continue
            primary = record.provenance[0].source
            summary[primary] = summary.get(primary, 0) + 1
        return summary

    def preview_pdf_seeds(self, files: list[dict[str, Any]], deep_parse: bool = False) -> list[dict[str, Any]]:
        previews: list[dict[str, Any]] = []
        for file in files:
            try:
                preview = self.pdf_extractor.extract(
                    filename=file["name"],
                    content=file["content"],
                    deep_parse=deep_parse,
                )
            except RuntimeError as exc:
                preview = {
                    "filename": file["name"],
                    "provider": "openalex",
                    "id": "",
                    "confidence": "low",
                    "origin": "uploaded_pdf",
                    "unresolved": True,
                    "error": str(exc),
                    "mapping_score": 0.0,
                    "pdf_gate_blocked": True,
                    "metadata_resolution": "unknown",
                }
            previews.append(preview)
        return previews

    def build_seeds_from_pdf_previews(self, previews: list[dict[str, Any]], deep_parse: bool = False) -> list[dict[str, Any]]:
        seeds: list[dict[str, Any]] = []
        unresolved_references = 0
        low_confidence_count = 0
        overridden_count = 0
        blocked_skipped = 0
        for item in previews:
            pid = (item.get("id") or "").strip()
            if not pid:
                continue
            score = float(item.get("mapping_score", 1.0))
            if score < PDF_MAPPING_QUALITY_THRESHOLD:
                low_confidence_count += 1
            if score < PDF_MAPPING_QUALITY_THRESHOLD and not item.get("quality_override"):
                blocked_skipped += 1
                continue
            if score < PDF_MAPPING_QUALITY_THRESHOLD and item.get("quality_override"):
                overridden_count += 1
                self.submit_pdf_quality_override(
                    item.get("filename") or pid,
                    score,
                    item.get("provider") or "unknown",
                    pid,
                )
            seeds.append(
                {
                    "provider": item["provider"],
                    "id": pid,
                    "origin": "uploaded_pdf",
                    "metadata_resolution": item.get("metadata_resolution", "pdf_text"),
                }
            )
            if deep_parse:
                for ref in item.get("references", []):
                    if not ref:
                        unresolved_references += 1
        pdf_seed_objs = [s for s in seeds if s.get("origin") == "uploaded_pdf"]
        pool_notes: list[str] = []
        if not deep_parse:
            pool_notes.append(
                "Expand pool: deep_parse=False — PDF reference DOIs are not extracted; ordered_pool stays empty."
            )
        elif not previews:
            pool_notes.append("Expand pool: no PDF previews passed to build_seeds_from_pdf_previews.")
        elif not any(len(item.get("references", []) or []) for item in previews):
            pool_notes.append(
                "Expand pool: deep_parse=True but every preview has references=[] "
                "(no DOI-like tokens in parsed pages — bibliography layout or missing DOIs)."
            )
        hop_room = max(0, EXPAND_CANDIDATE_POOL_MAX)
        hop_pdf = self._second_hop_reference_seeds(
            pdf_seed_objs, pool_notes, None, max_seeds=min(120, hop_room)
        )
        for h in hop_pdf:
            seeds.append(h)
        ordered_pool: list[str] = []
        seen_ordered: set[str] = set()
        if deep_parse:
            for item in previews:
                for ref in item.get("references", []) or []:
                    r = (ref or "").strip()
                    if not r or r in seen_ordered:
                        continue
                    seen_ordered.add(r)
                    ordered_pool.append(r)
                    if len(ordered_pool) >= EXPAND_CANDIDATE_POOL_MAX:
                        break
                if len(ordered_pool) >= EXPAND_CANDIDATE_POOL_MAX:
                    break
        for doi in ordered_pool:
            seeds.append({"provider": "doi", "id": doi, "origin": "discovered"})
        if deep_parse and not ordered_pool and not pool_notes:
            pool_notes.append(
                "ordered_pool_size=0 after scanning previews: duplicate DOIs only, or cap reached before first unique id."
            )
        self._last_expand_ordered_pool = len(ordered_pool)
        pdf_raw_lines = sum(len(item.get("references", []) or []) for item in previews) if deep_parse else 0
        self._last_expand_stats = {
            "pdf_references_raw_lines": pdf_raw_lines,
            "references_extracted": pdf_raw_lines,
            "references_resolved": len(ordered_pool) if deep_parse else 0,
            "references_skipped": unresolved_references if deep_parse else 0,
            "ordered_pool_size": len(ordered_pool),
            "pool_empty_reasons": pool_notes,
        }
        self._last_pdf_gate_stats = {
            "threshold": PDF_MAPPING_QUALITY_THRESHOLD,
            "low_confidence_count": low_confidence_count,
            "overridden_count": overridden_count,
            "blocked_excluded_from_seeds": blocked_skipped,
        }
        return seeds

    def _paper_to_json_with_origin(self, paper: PaperRecord, origin_map: dict[str, str]) -> dict[str, Any]:
        payload = _paper_to_json(paper)
        keys = [
            f"doi:{(paper.doi or '').lower()}",
            f"openalex:{(paper.openalex_id or '').lower()}",
            f"arxiv:{(paper.arxiv_id or '').lower()}",
            paper.canonical_id.lower(),
        ]
        origin = "discovered"
        for key in keys:
            if key in origin_map:
                origin = origin_map[key]
                break
        payload["seed_origin"] = origin
        payload["source_label"] = "PDF" if origin == "uploaded_pdf" else "API"
        return payload

    _last_expand_stats: dict[str, Any] = {}
    _last_expand_ordered_pool: int = 0
    _last_pdf_gate_stats: dict[str, Any] = {}

    def _enrich_missing_link_candidate(
        self,
        pid: str,
        meta: dict[str, int],
        merged_by_id: dict[str, PaperRecord],
        pii_tail: str | None,
    ) -> dict[str, Any]:
        paper = merged_by_id.get(pid)
        cited_by_n = 0
        pii_m = 0.0
        is_fh = False
        if paper:
            for prov in paper.provenance or []:
                if prov.source == "openalex" and isinstance(prov.raw, dict):
                    cited_by_n = int(prov.raw.get("cited_by_count") or 0)
                    if pii_tail:
                        pii_m = self._openalex_max_concept_score_for_id(prov.raw.get("concepts"), pii_tail)
                    break
            is_fh = any(pr.source == "foundational_hub" for pr in (paper.provenance or []))
        is_fh = is_fh or (
            pii_m >= 0.25 and int(meta.get("co_cited_by_uploaded_count", 0)) >= 3
        )
        return {
            "paper_id": pid,
            **meta,
            "openalex_cited_by_count": cited_by_n,
            "pii_concept_match": round(pii_m, 4),
            "is_foundational_hub": is_fh,
        }

    def _compute_missing_links(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        threshold: int = 3,
    ) -> dict[str, dict[str, int]]:
        node_map = {n["id"]: n for n in nodes}
        uploaded_ids = {n["id"] for n in nodes if n.get("seed_origin") == "uploaded_pdf"}
        counts: dict[str, int] = {}
        counted_pairs: set[tuple[str, str]] = set()
        for edge in edges:
            src = edge["source"]
            tgt = edge["target"]
            if src not in uploaded_ids or tgt in uploaded_ids:
                continue
            pair = (src, tgt)
            if pair in counted_pairs:
                continue
            counted_pairs.add(pair)
            counts[tgt] = counts.get(tgt, 0) + 1
        result: dict[str, dict[str, int]] = {}
        for pid, c in counts.items():
            if pid not in node_map:
                continue
            result[pid] = {"co_cited_by_uploaded_count": c, "threshold": threshold}
        return result

    def _inject_strong_coupling_edges(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        *,
        min_shared_refs: int = 3,
        potential_mode: bool = False,
    ) -> None:
        """
        Hybrid connectivity: if two papers share >= min_shared_refs references but have no direct citation edge,
        add a synthetic strong-evidence bibliographic coupling edge.
        """
        if len(nodes) < 2:
            return
        existing_direct: set[frozenset[str]] = set()
        for e in edges:
            s = str(e.get("source") or "").strip()
            t = str(e.get("target") or "").strip()
            if s and t and s != t:
                existing_direct.add(frozenset((s, t)))
        ref_sets: dict[str, set[str]] = {}
        for n in nodes:
            pid = str(n.get("id") or "").strip()
            if not pid:
                continue
            refs = {
                self._normalize_citation_ref_key(str(r))
                for r in (n.get("references") or [])
                if self._normalize_citation_ref_key(str(r))
            }
            if refs:
                ref_sets[pid] = refs
        pids = sorted(ref_sets.keys())
        added = 0
        for i, left in enumerate(pids):
            for right in pids[i + 1 :]:
                if frozenset((left, right)) in existing_direct:
                    continue
                shared = len(ref_sets[left].intersection(ref_sets[right]))
                if shared < min_shared_refs:
                    continue
                edges.append(
                    {
                        "source": left,
                        "target": right,
                        "context": (
                            f"Potential Connection: shared references={shared}"
                            if potential_mode
                            else f"Strong Evidence: bibliographic coupling (shared references={shared})"
                        ),
                        "context_source": "bibliographic_coupling",
                        "confidence": "low" if potential_mode else "high",
                        "style": {
                            "stroke": "dashed" if potential_mode else "solid",
                            "weight": "thin" if potential_mode else "normal",
                            "opacity": "0.55" if potential_mode else "1.0",
                            "tooltip": (
                                f"potential connection: shared references={shared}"
                                if potential_mode
                                else f"strong evidence: shared references={shared}"
                            ),
                        },
                    }
                )
                added += 1
        if added:
            for n in nodes:
                n["hybrid_coupling_edges_added"] = int(n.get("hybrid_coupling_edges_added", 0)) + added

