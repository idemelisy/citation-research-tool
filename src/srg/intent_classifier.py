"""Intent & mode classifier — SRG Lite v2.1 (paper-style taxonomy)."""

from __future__ import annotations

import re
from typing import Any


class QueryIntent:
    """Structured intents from SRG Lite v2.1 §3 + roadmap Phase 1.2."""

    DEFINITION = "definition"
    COMPARISON = "comparison"
    METHOD_LOOKUP = "method_lookup"
    SURVEY = "survey"
    EXPLORATORY = "exploratory"
    # Roadmap 1.2 — finer technical / systems / temporal intents
    TECHNICAL_METHOD = "technical_method"
    HARDWARE_SYSTEM = "hardware_system"
    HISTORICAL_FOUNDATION = "historical_foundation"
    RECENT_PROGRESS = "recent_progress"

    # SRG Lite v2.0 string compatibility (normalize via ranking_fusion.normalize_query_intent)
    CANONICAL_LOOKUP = "canonical_lookup"
    SURVEY_LEGACY = "survey_mode"
    EXPLORATION_LEGACY = "exploratory_mode"


_DEF_WORDS = frozenset(
    (
        "what is",
        "what are",
        "define",
        "definition",
        "meaning of",
        "explain",
        "introduction to",
        "tell me about",
    )
)

_SURVEY_CUES = frozenset(
    (
        "survey",
        "literature review",
        "systematic review",
        "review of",
        "overview of",
        "state of the art",
    )
)

_COMPARE_STRICT = frozenset(
    (
        " vs ",
        " versus ",
        "compare ",
        "comparison",
        "difference between",
        "differences between",
        "contrasted",
        "benchmark",
    )
)


def detect_definition_words(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    return any(w in q for w in _DEF_WORDS)


def detect_survey_cues(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    return any(w in q for w in _SURVEY_CUES)


def detect_comparison_strict(query: str) -> bool:
    """Head-to-head or benchmarking phrasing (§3 COMPARISON)."""
    q = (query or "").strip().lower()
    if not q:
        return False
    if any(w in q for w in _COMPARE_STRICT):
        return True
    return bool(re.search(r"\b(and|or)\b.*\b(and|or)\b", q))


def detect_entity_like_query(query: str) -> bool:
    """Short, noun-phrase style queries (likely pointing at one paper/method)."""
    q = (query or "").strip()
    if not q:
        return False
    words = q.split()
    if len(words) > 14:
        return False
    if len(words) <= 1 and len(q) <= 40:
        return True
    lower = q.lower()
    if lower.endswith("?"):
        return len(words) <= 10
    stop_ratio = sum(1 for w in words if w.lower() in _STOP) / max(len(words), 1)
    return len(words) <= 10 and stop_ratio < 0.45


_STOP = frozenset(
    (
        "the",
        "a",
        "an",
        "for",
        "to",
        "of",
        "in",
        "on",
        "and",
        "or",
        "with",
        "using",
        "via",
    )
)


_RECENT_CUES = frozenset(
    (
        "latest",
        "recent",
        "new ",
        "2024",
        "2025",
        "2026",
        "state of the art",
        "sota",
        "current",
        "this year",
        "modern gpu",
        "neural rendering",
    )
)

_HISTORICAL_CUES = frozenset(
    (
        "history of",
        "historical",
        "seminal",
        "foundational",
        "original formulation",
        "classic paper",
        "pioneering",
        "first introduced",
    )
)

_HARDWARE_CUES = frozenset(
    (
        "gpu",
        "hardware",
        "accelerat",
        "tessellation",
        "shader",
        "cuda",
        "rendering pipeline",
        "graphics hardware",
        "graphics",
        "real-time",
        "real time",
        "pipeline",
        "fpga",
        "asic",
        "simd",
        "parallel ",
        "distributed system",
        "runtime",
        "throughput",
        "latency",
    )
)

# PR B2 — v2.2 discrete hardware features (≥2 hits ⇒ hardware_system + recency fusion boost).
_HW_V22_FEATURES = (
    "gpu",
    "shader",
    "tessellation",
    "cuda",
    "pipeline",
    "acceleration",
    "accelerated",
    "parallel",
    "graphics",
)

_TECHNICAL_CUES = frozenset(
    (
        "implementation",
        "engineering",
        "algorithm",
        "method",
        "methods",
        "optimization",
        "framework",
        "model",
        "benchmark",
        "complexity",
        "architecture",
        "system design",
        "deployment",
    )
)


def detect_recent_progress_cues(query: str) -> bool:
    q = (query or "").strip().lower()
    return any(c in q for c in _RECENT_CUES)


def detect_historical_foundation_cues(query: str) -> bool:
    q = (query or "").strip().lower()
    return any(c in q for c in _HISTORICAL_CUES)


def hardware_v22_feature_hit_count(query: str) -> int:
    """Count PR B2 keyword hits (substring or word-boundary for *parallel*)."""
    q = (query or "").strip().lower()
    if not q:
        return 0
    n = 0
    for t in _HW_V22_FEATURES:
        if t == "parallel":
            if re.search(r"\bparallel\b", q):
                n += 1
        elif t in q:
            n += 1
    return n


def detect_hardware_system_cues(query: str) -> bool:
    q = (query or "").strip().lower()
    return any(c in q for c in _HARDWARE_CUES) or hardware_v22_feature_hit_count(q) >= 2


def detect_technical_method_cues(query: str) -> bool:
    q = (query or "").strip().lower()
    if detect_hardware_system_cues(q):
        return False
    return any(c in q for c in _TECHNICAL_CUES)


def intent_distribution_for_query(query: str) -> dict[str, float]:
    """
    Soft mixture over roadmap intents (Phase 1.2) from lexical overlap — sums to 1.0.
    Keys use the same strings as ``QueryIntent`` constants.
    """
    q = (query or "").strip().lower()
    raw: dict[str, float] = {
        QueryIntent.HARDWARE_SYSTEM: 0.08,
        QueryIntent.TECHNICAL_METHOD: 0.08,
        QueryIntent.HISTORICAL_FOUNDATION: 0.06,
        QueryIntent.RECENT_PROGRESS: 0.08,
        QueryIntent.SURVEY: 0.05,
        QueryIntent.COMPARISON: 0.05,
        QueryIntent.DEFINITION: 0.05,
        QueryIntent.METHOD_LOOKUP: 0.05,
        QueryIntent.EXPLORATORY: 0.12,
    }
    if not q:
        s = sum(raw.values()) or 1.0
        return {k: v / s for k, v in raw.items()}

    raw[QueryIntent.HARDWARE_SYSTEM] += sum(0.22 for c in _HARDWARE_CUES if c in q)
    raw[QueryIntent.TECHNICAL_METHOD] += sum(0.18 for c in _TECHNICAL_CUES if c in q and c not in _HARDWARE_CUES)
    raw[QueryIntent.HISTORICAL_FOUNDATION] += sum(0.2 for c in _HISTORICAL_CUES if c in q)
    raw[QueryIntent.RECENT_PROGRESS] += sum(0.16 for c in _RECENT_CUES if c in q)
    raw[QueryIntent.SURVEY] += sum(0.18 for c in _SURVEY_CUES if c in q)
    raw[QueryIntent.COMPARISON] += sum(0.2 for c in _COMPARE_STRICT if c in q)
    raw[QueryIntent.DEFINITION] += sum(0.15 for c in _DEF_WORDS if c in q)
    if detect_method_name_like(query):
        raw[QueryIntent.METHOD_LOOKUP] += 0.25
    s = sum(raw.values()) or 1.0
    return {k: round(v / s, 4) for k, v in raw.items()}


def detect_method_name_like(query: str) -> bool:
    """
    Reads like a standalone method/paper title (e.g. 'Direct Preference Optimization').
    """
    q = (query or "").strip()
    if not q:
        return False
    words = q.split()
    if len(words) < 2 or len(words) > 12:
        return False
    if detect_comparison_strict(q):
        return False
    if detect_definition_words(q):
        return False
    if detect_survey_cues(q):
        return False
    caps = sum(1 for w in words if w[:1].isupper())
    if caps >= max(2, len(words) // 2):
        return True
    if len(words) <= 6 and all(len(w) <= 40 for w in words):
        return True
    return False


def _spike_intent_distribution(
    dist: dict[str, float],
    intent: str,
    *,
    spike: float = 0.9,
) -> dict[str, float]:
    """Sharpen distribution toward one intent so fusion does not over-blend (PR B3)."""
    keys = set(dist) | {intent}
    if len(keys) <= 1:
        return {intent: 1.0}
    rest = (1.0 - spike) / (len(keys) - 1)
    out = {k: rest for k in keys}
    out[intent] = spike
    s = sum(out.values()) or 1.0
    return {k: round(float(v) / s, 4) for k, v in out.items()}


def _classify_pack(
    intent: str,
    confidence: float,
    dist: dict[str, float],
    *,
    hardware_recency_fusion_boost: bool = False,
) -> dict[str, Any]:
    """PR B1 — stable keys including ``distribution`` alias for downstream fusion."""
    return {
        "intent": intent,
        "confidence": confidence,
        "intent_distribution": dist,
        "distribution": dist,
        "hardware_recency_fusion_boost": hardware_recency_fusion_boost,
    }


def classify_query(query: str, context: dict[str, Any]) -> dict[str, Any]:
    """
    Hybrid heuristic classifier (§3). Returns ``intent``, ``confidence``,
    ``intent_distribution`` / ``distribution``, optional ``hardware_recency_fusion_boost`` (PR B2).
    """
    _ = context
    q = (query or "").strip()
    dist = intent_distribution_for_query(q)

    if detect_comparison_strict(q):
        return _classify_pack(QueryIntent.COMPARISON, 0.72, dist)

    if hardware_v22_feature_hit_count(q) >= 2:
        dist_hw = _spike_intent_distribution(dist, QueryIntent.HARDWARE_SYSTEM, spike=0.92)
        return _classify_pack(
            QueryIntent.HARDWARE_SYSTEM, 0.9, dist_hw, hardware_recency_fusion_boost=True
        )

    if detect_survey_cues(q):
        return _classify_pack(QueryIntent.SURVEY, 0.7, dist)

    if detect_recent_progress_cues(q) and detect_entity_like_query(q):
        return _classify_pack(QueryIntent.RECENT_PROGRESS, 0.68, dist)

    if detect_historical_foundation_cues(q) and detect_entity_like_query(q):
        return _classify_pack(QueryIntent.HISTORICAL_FOUNDATION, 0.7, dist)

    if detect_hardware_system_cues(q):
        boost = hardware_v22_feature_hit_count(q) >= 2
        return _classify_pack(QueryIntent.HARDWARE_SYSTEM, 0.74, dist, hardware_recency_fusion_boost=boost)

    if detect_technical_method_cues(q) and detect_entity_like_query(q):
        return _classify_pack(QueryIntent.TECHNICAL_METHOD, 0.66, dist)

    if detect_definition_words(q) and detect_entity_like_query(q):
        return _classify_pack(QueryIntent.DEFINITION, 0.8, dist)

    if detect_method_name_like(q) and detect_entity_like_query(q):
        return _classify_pack(QueryIntent.METHOD_LOOKUP, 0.72, dist)

    return _classify_pack(QueryIntent.EXPLORATORY, 0.6, dist)
