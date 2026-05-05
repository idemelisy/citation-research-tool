from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ConfidenceTier(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(slots=True)
class Provenance:
    source: str
    source_id: str
    fetched_at: datetime = field(default_factory=datetime.utcnow)
    field_confidence: dict[str, ConfidenceTier] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Author:
    name: str
    orcid: str | None = None


@dataclass(slots=True)
class PaperRecord:
    canonical_id: str
    title: str
    abstract: str | None
    year: int | None
    venue: str | None
    domain: str | None
    doi: str | None = None
    arxiv_id: str | None = None
    semantic_scholar_id: str | None = None
    openalex_id: str | None = None
    authors: list[Author] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    provenance: list[Provenance] = field(default_factory=list)


@dataclass(slots=True)
class CitationEdge:
    source_paper_id: str
    target_paper_id: str
    context: str | None
    context_source: str | None
    confidence: ConfidenceTier


@dataclass(slots=True)
class FeedbackEvent:
    event_type: str
    left_id: str
    right_id: str
    accepted: bool
    created_at: datetime = field(default_factory=datetime.utcnow)

