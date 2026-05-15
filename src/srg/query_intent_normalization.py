"""
Centralized query intent normalization for SRG Lite.

Maps ambiguous research phrases to canonical intents via a maintainable registry
(pattern-driven expansion, not scattered if-statements).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", (text or "").strip())
    t = re.sub(r"[\u200b-\u200d\ufeff]", "", t)
    return t.lower()


@dataclass(frozen=True)
class SubfieldCluster:
    """Markers that identify a literature subfield for boost or penalty."""

    name: str
    markers: tuple[str, ...]


@dataclass(frozen=True)
class IntentRegistryEntry:
    """One ambiguous-query profile — matched when any pattern hits."""

    id: str
    patterns: tuple[str, ...]
    pattern_regex: tuple[str, ...] = ()
    exclude_patterns: tuple[str, ...] = ()
    enrichment_terms: tuple[str, ...] = ()
    canonical_domains: frozenset[str] = frozenset({"cs"})
    boost_clusters: tuple[SubfieldCluster, ...] = ()
    penalty_clusters: tuple[SubfieldCluster, ...] = ()
    search_variants: tuple[str, ...] = ()
    intent_label: str = "technical_method"
    ambiguity_base: float = 0.55
    canonical_arxiv_ids: tuple[str, ...] = ()


def _cluster(name: str, *markers: str) -> SubfieldCluster:
    return SubfieldCluster(name=name, markers=tuple(m.lower() for m in markers))


# --- Registry (ordered: first match wins) -----------------------------------

_ALIGNMENT_BOOST = (
    _cluster("llm_alignment", "rlhf", "human feedback", "language model", "llm", "alignment", "reward model"),
    _cluster("preference_tuning", "preference optimization", "direct preference", "dpo", "orpo", "simpo"),
)

_ALIGNMENT_PENALTY = (
    _cluster(
        "recommender_systems",
        "recommendation system",
        "sequential recommendation",
        "collaborative filtering",
        "click-through",
        "clickthrough",
        "learning to rank",
        "session-based recommendation",
    ),
    _cluster(
        "generic_preference",
        "preference learning survey",
        "multi-objective preference",
        "stated preference",
        "consumer preference",
    ),
    _cluster("ir_summarization", "text summarization", "abstractive summarization", "document summarization"),
)

_NLP_TRANSFORMER_BOOST = (
    _cluster("attention_nlp", "attention mechanism", "self-attention", "transformer", "bert", "language model"),
    _cluster("nlp", "natural language", "machine translation", "nlp"),
)

_POWER_TRANSFORMER_PENALTY = (
    _cluster("power_systems", "power transformer", "fault diagnosis", "substation", "electrical grid", "insulation"),
)

_VIS_DEBUG_BOOST = (
    _cluster("software_viz", "program visualization", "software visualization", "debugging", "debugger"),
    _cluster("dev_tools", "developer tool", "ide", "execution trace", "call graph", "dynamic analysis"),
)

_VIS_DEBUG_PENALTY = (
    _cluster("medical_only", "histopathology", "mri segmentation", "clinical imaging"),
    _cluster("generic_visual", "visual analytics dashboard", "infographic"),
)

INTENT_REGISTRY: tuple[IntentRegistryEntry, ...] = (
    IntentRegistryEntry(
        id="direct_preference_optimization",
        patterns=(
            "direct preference optimization",
            "directed preference optimization",
            "dpo ",
            " dpo",
        ),
        pattern_regex=(r"\bdpo\b",),
        enrichment_terms=(
            "LLM alignment",
            "RLHF",
            "language model",
            "preference tuning",
            "reward model",
            "DPO",
        ),
        boost_clusters=_ALIGNMENT_BOOST,
        penalty_clusters=_ALIGNMENT_PENALTY,
        search_variants=(
            "direct preference optimization language model alignment",
            "DPO RLHF large language model",
            "preference optimization LLM without reward model",
        ),
        intent_label="technical_method",
        ambiguity_base=0.72,
        canonical_arxiv_ids=("2305.18290", "2203.02155", "2404.19733"),
    ),
    IntentRegistryEntry(
        id="transformer_nlp",
        patterns=("transformer",),
        exclude_patterns=(
            "power transformer",
            "electrical transformer",
            "fault diagnosis",
            "substation",
            "grid",
            "insulation",
            "hv ",
        ),
        enrichment_terms=("attention", "NLP", "language model", "self-attention", "encoder-decoder"),
        boost_clusters=_NLP_TRANSFORMER_BOOST,
        penalty_clusters=_POWER_TRANSFORMER_PENALTY,
        search_variants=(
            "transformer attention mechanism natural language processing",
            "transformer neural network language model",
        ),
        intent_label="technical_method",
        ambiguity_base=0.78,
        canonical_arxiv_ids=("1706.03762",),
    ),
    IntentRegistryEntry(
        id="visual_debugging",
        patterns=(
            "visual debugging",
            "software visualization debugging",
            "program visualization",
        ),
        enrichment_terms=(
            "software debugging",
            "program analysis",
            "execution trace",
            "developer tools",
            "dynamic analysis",
        ),
        boost_clusters=_VIS_DEBUG_BOOST,
        penalty_clusters=_VIS_DEBUG_PENALTY,
        search_variants=(
            "software visualization debugging tools",
            "program execution trace visualization",
        ),
        intent_label="technical_method",
        ambiguity_base=0.68,
    ),
    IntentRegistryEntry(
        id="graph_neural_networks",
        patterns=(
            "graph neural network",
            "graph neural networks",
            "graph convolutional",
            "graphsage",
            "graph attention network",
        ),
        pattern_regex=(r"\bgnn\b",),
        enrichment_terms=("GCN", "message passing", "graph learning", "graph embedding"),
        boost_clusters=(
            _cluster("gnn", "graph convolutional", "message passing", "graphsage", "graph attention"),
        ),
        penalty_clusters=(
            _cluster("unrelated_graph", "social network survey", "knowledge graph construction survey"),
        ),
        search_variants=("graph neural networks message passing survey",),
        intent_label="technical_method",
        ambiguity_base=0.62,
        canonical_arxiv_ids=("1609.02907", "1706.02216", "1710.10903"),
    ),
    IntentRegistryEntry(
        id="rlhf_alignment",
        patterns=(
            "rlhf",
            "reinforcement learning from human feedback",
            "reward modeling",
            "constitutional ai",
        ),
        enrichment_terms=("alignment", "language model", "preference", "reward model"),
        boost_clusters=_ALIGNMENT_BOOST,
        penalty_clusters=_ALIGNMENT_PENALTY,
        search_variants=(
            "reinforcement learning human feedback language model",
            "reward model alignment LLM",
        ),
        intent_label="technical_method",
        ambiguity_base=0.65,
        canonical_arxiv_ids=("2203.02155", "2305.18290"),
    ),
)


def _compiled_regexes(entry: IntentRegistryEntry) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p, re.IGNORECASE) for p in entry.pattern_regex)


_REGEX_CACHE: dict[str, tuple[re.Pattern[str], ...]] = {
    e.id: _compiled_regexes(e) for e in INTENT_REGISTRY
}


def _entry_matches(query_norm: str, entry: IntentRegistryEntry) -> bool:
    for ex in entry.exclude_patterns:
        if ex.lower() in query_norm:
            return False
    for p in entry.patterns:
        if p.lower() in query_norm:
            return True
    for rx in _REGEX_CACHE.get(entry.id, ()):
        if rx.search(query_norm):
            return True
    return False


def match_intent_entry(query: str) -> IntentRegistryEntry | None:
    qn = _norm(query)
    if not qn:
        return None
    for entry in INTENT_REGISTRY:
        if _entry_matches(qn, entry):
            return entry
    return None


def compute_ambiguity_score(query: str, entry: IntentRegistryEntry | None) -> float:
    """Higher ⇒ more conservative retrieval (0–1)."""
    qn = _norm(query)
    words = qn.split()
    short_query = len(words) <= 2
    if entry is None:
        if short_query:
            return 0.45
        return 0.25
    score = float(entry.ambiguity_base)
    if short_query:
        score = min(1.0, score + 0.12)
    if len(words) == 1:
        score = min(1.0, score + 0.08)
    return max(0.0, min(1.0, score))


@dataclass
class NormalizedQueryIntent:
    original_query: str
    normalized_query: str
    enrichment_terms: list[str]
    intent_entry_id: str | None
    intent_label: str
    ambiguity_score: float
    canonical_domains: frozenset[str]
    search_variants: list[str]
    canonical_arxiv_ids: list[str]
    boost_cluster_names: list[str]
    penalty_cluster_names: list[str]

    def to_profile_dict(self) -> dict[str, Any]:
        return {
            "query_text_normalized": self.normalized_query,
            "intent_enrichment_terms": list(self.enrichment_terms),
            "intent_entry_id": self.intent_entry_id,
            "intent_label_normalized": self.intent_label,
            "query_ambiguity_score": round(self.ambiguity_score, 4),
            "intent_canonical_domains": sorted(self.canonical_domains),
            "intent_search_variants": list(self.search_variants),
            "intent_canonical_arxiv_ids": list(self.canonical_arxiv_ids),
        }


def normalize_query_intent(query: str) -> NormalizedQueryIntent:
    """
    Detect ambiguous phrases, map to canonical research intent, enrich retrieval context.
    """
    original = (query or "").strip()
    qn = _norm(original)
    if "directed preference optimization" in qn:
        qn = qn.replace("directed preference optimization", "direct preference optimization", 1)
        original = re.sub(
            r"directed preference optimization",
            "direct preference optimization",
            original,
            count=1,
            flags=re.IGNORECASE,
        )

    entry = match_intent_entry(original)
    enrichment: list[str] = []
    search_variants: list[str] = []
    canonical_ids: list[str] = []
    domains = frozenset({"cs"})
    intent_label = "technical_method"
    entry_id: str | None = None

    if entry:
        entry_id = entry.id
        enrichment = list(entry.enrichment_terms)
        search_variants = [original] + list(entry.search_variants)
        canonical_ids = list(entry.canonical_arxiv_ids)
        domains = entry.canonical_domains
        intent_label = entry.intent_label

    ambiguity = compute_ambiguity_score(original, entry)

    merged_terms = list(
        dict.fromkeys(
            [
                t.strip()
                for t in re.split(r"[^\w\-]+", original)
                if len(t.strip()) > 2
            ]
            + [t.lower() for t in enrichment if str(t).strip()]
        )
    )

    return NormalizedQueryIntent(
        original_query=original,
        normalized_query=original,
        enrichment_terms=enrichment,
        intent_entry_id=entry_id,
        intent_label=intent_label,
        ambiguity_score=ambiguity,
        canonical_domains=domains,
        search_variants=list(dict.fromkeys(v for v in search_variants if v.strip())),
        canonical_arxiv_ids=canonical_ids,
        boost_cluster_names=[c.name for c in (entry.boost_clusters if entry else ())],
        penalty_cluster_names=[c.name for c in (entry.penalty_clusters if entry else ())],
    )


def registry_entry_by_id(entry_id: str) -> IntentRegistryEntry | None:
    for e in INTENT_REGISTRY:
        if e.id == entry_id:
            return e
    return None


def expanded_query_variants_from_intent(nqi: NormalizedQueryIntent) -> list[str]:
    """Search variants for OpenAlex / arXiv discovery."""
    base = nqi.normalized_query or nqi.original_query
    if not base:
        return []
    variants = [base]
    variants.extend(nqi.search_variants)
    entry = registry_entry_by_id(nqi.intent_entry_id) if nqi.intent_entry_id else None
    if entry:
        for sv in entry.search_variants:
            if sv not in variants:
                variants.append(sv)
    if len(base.split()) <= 2 and not entry:
        variants.append(f"{base} survey")
    return list(dict.fromkeys(v for v in variants if v.strip()))


def all_clusters_for_entry(entry_id: str | None) -> tuple[tuple[SubfieldCluster, ...], tuple[SubfieldCluster, ...]]:
    entry = registry_entry_by_id(entry_id) if entry_id else None
    if not entry:
        return (), ()
    return entry.boost_clusters, entry.penalty_clusters


__all__ = [
    "INTENT_REGISTRY",
    "IntentRegistryEntry",
    "NormalizedQueryIntent",
    "SubfieldCluster",
    "all_clusters_for_entry",
    "expanded_query_variants_from_intent",
    "match_intent_entry",
    "normalize_query_intent",
    "registry_entry_by_id",
]
