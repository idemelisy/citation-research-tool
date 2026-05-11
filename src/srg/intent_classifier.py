"""Intent & mode classifier — SRG Lite v2.1 (paper-style taxonomy)."""

from __future__ import annotations

import re
from typing import Any


class QueryIntent:
    """Structured intents from SRG Lite v2.1 §3."""

    DEFINITION = "definition"
    COMPARISON = "comparison"
    METHOD_LOOKUP = "method_lookup"
    SURVEY = "survey"
    EXPLORATORY = "exploratory"

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


def classify_query(query: str, context: dict[str, Any]) -> dict[str, Any]:
    """
    Hybrid heuristic classifier (§3). Returns:
        {"intent": str, "confidence": float}
    """
    _ = context
    q = (query or "").strip()

    if detect_comparison_strict(q):
        return {"intent": QueryIntent.COMPARISON, "confidence": 0.72}

    if detect_survey_cues(q):
        return {"intent": QueryIntent.SURVEY, "confidence": 0.7}

    if detect_definition_words(q) and detect_entity_like_query(q):
        return {"intent": QueryIntent.DEFINITION, "confidence": 0.8}

    if detect_method_name_like(q) and detect_entity_like_query(q):
        return {"intent": QueryIntent.METHOD_LOOKUP, "confidence": 0.72}

    return {"intent": QueryIntent.EXPLORATORY, "confidence": 0.6}
