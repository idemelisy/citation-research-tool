"""Intent & mode classifier for SRG Lite v2 (query understanding layer)."""

from __future__ import annotations

import re
from typing import Any


class QueryIntent:
    CANONICAL_LOOKUP = "canonical_lookup"
    SURVEY = "survey_mode"
    EXPLORATION = "exploratory_mode"


_DEF_WORDS = frozenset(
    (
        "what is",
        "what are",
        "define",
        "definition",
        "meaning of",
        "explain",
        "overview of",
        "introduction to",
        "tell me about",
    )
)

_COMPARE_WORDS = frozenset(
    (
        " vs ",
        " versus ",
        "compare",
        "comparison",
        "difference between",
        "differences between",
        "contrasted",
        "benchmark",
        "state of the art",
        "survey",
        "literature review",
        "systematic review",
    )
)


def detect_definition_words(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    return any(w in q for w in _DEF_WORDS)


def detect_comparison(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    if any(w in q for w in _COMPARE_WORDS):
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
    Heuristic: reads like a standalone method/paper title (e.g. 'Direct Preference Optimization')
    rather than a prose question.
    """
    q = (query or "").strip()
    if not q:
        return False
    words = q.split()
    if len(words) < 2 or len(words) > 12:
        return False
    if detect_comparison(q):
        return False
    if detect_definition_words(q):
        return False
    caps = sum(1 for w in words if w[:1].isupper())
    # Prefer capitalized technical titles or acronym-heavy short phrases.
    if caps >= max(2, len(words) // 2):
        return True
    if len(words) <= 6 and all(len(w) <= 40 for w in words):
        return True
    return False


def classify_query(query: str, context: dict[str, Any]) -> dict[str, Any]:
    """
    Returns:
        {"intent": str, "confidence": float}
    """
    _ = context  # reserved for session / discipline hints
    features = {
        "has_definition_terms": detect_definition_words(query),
        "has_compare_terms": detect_comparison(query),
        "is_single_entity_query": detect_entity_like_query(query),
        "looks_like_method_name": detect_method_name_like(query),
    }

    if features["has_definition_terms"] and features["is_single_entity_query"]:
        return {"intent": QueryIntent.CANONICAL_LOOKUP, "confidence": 0.8}

    if features["has_compare_terms"]:
        return {"intent": QueryIntent.SURVEY, "confidence": 0.7}

    if features["looks_like_method_name"] and features["is_single_entity_query"]:
        return {"intent": QueryIntent.CANONICAL_LOOKUP, "confidence": 0.72}

    return {"intent": QueryIntent.EXPLORATION, "confidence": 0.6}
