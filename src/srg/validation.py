from __future__ import annotations

import difflib
import re
from collections import defaultdict
from dataclasses import dataclass, field

from .schema import FeedbackEvent, PaperRecord, Provenance

AUTHOR_CROSSCHECK_TITLE_THRESHOLD = 0.70
AUTHOR_MATCH_BOOST_STRONG = 0.82
AUTHOR_MATCH_BOOST_WEAK = 0.74
AUTHOR_MATCH_MIN_STRONG = 2
AUTHOR_MATCH_MIN_WEAK = 1


def _tokenize_author_line(pdf_authors_line: str | None) -> set[str]:
    if not pdf_authors_line:
        return set()
    raw = pdf_authors_line.lower()
    raw = re.sub(r"[^\w\s,;]", " ", raw)
    parts = re.split(r"[,;]|\band\b|\bal\b", raw, flags=re.IGNORECASE)
    tokens: set[str] = set()
    for p in parts:
        for w in p.split():
            w = w.strip()
            if len(w) >= 2:
                tokens.add(w)
                tokens.add(w.rstrip("."))
    return tokens


def _tokens_from_api_author_names(api_names: list[str]) -> set[str]:
    tokens: set[str] = set()
    for name in api_names:
        if not name:
            continue
        low = name.lower()
        for w in re.split(r"[\s,]+", low):
            w = w.strip().strip(".")
            if len(w) >= 2:
                tokens.add(w)
    return tokens


def author_intersection_count(pdf_authors_line: str | None, api_author_names: list[str]) -> int:
    """Count overlapping tokens between a PDF author line and API author display names."""
    if not pdf_authors_line or not api_author_names:
        return 0
    a = _tokenize_author_line(pdf_authors_line)
    b = _tokens_from_api_author_names(api_author_names)
    return len(a.intersection(b))


def author_crosscheck_adjust_score(
    mapping_score: float,
    pdf_authors_line: str | None,
    api_author_names: list[str],
    title_threshold: float = AUTHOR_CROSSCHECK_TITLE_THRESHOLD,
) -> tuple[float, str | None]:
    """
    If title-based mapping is weak, boost score when PDF vs API authors overlap.
    Strong: >=2 token matches -> floor 0.82; weak: 1 match -> floor 0.74.
    """
    if mapping_score >= title_threshold or not api_author_names:
        return mapping_score, None
    c = author_intersection_count(pdf_authors_line, api_author_names)
    if c >= AUTHOR_MATCH_MIN_STRONG:
        return max(mapping_score, AUTHOR_MATCH_BOOST_STRONG), f"author_crosscheck:{c}"
    if c >= AUTHOR_MATCH_MIN_WEAK:
        return max(mapping_score, AUTHOR_MATCH_BOOST_WEAK), f"author_crosscheck:{c}"
    return mapping_score, None


def finalize_pdf_mapping_score(
    mapping_score: float,
    pdf_authors_line: str | None,
    api_author_names: list[str],
    title_threshold: float = AUTHOR_CROSSCHECK_TITLE_THRESHOLD,
) -> tuple[float, str | None]:
    """
    Strict author gate: if both sides have author strings and token overlap is zero,
    force a low score (wrong paper) regardless of title similarity.
    Otherwise apply author_crosscheck_adjust_score for weak-title boosts.
    """
    if (
        pdf_authors_line
        and pdf_authors_line.strip()
        and api_author_names
        and author_intersection_count(pdf_authors_line, api_author_names) == 0
    ):
        return 0.1, "Wrong Paper Detected"
    return author_crosscheck_adjust_score(mapping_score, pdf_authors_line, api_author_names, title_threshold)


@dataclass(slots=True)
class ValidationAudit:
    merged_pairs: list[tuple[str, str]] = field(default_factory=list)
    rejected_pairs: list[tuple[str, str]] = field(default_factory=list)
    conflict_log: list[str] = field(default_factory=list)


def _title_similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(a=left.lower(), b=right.lower()).ratio()


def _norm_identifier(value: str | None) -> str | None:
    if not value:
        return None
    v = value.strip().lower()
    v = v.replace("https://doi.org/", "").replace("http://doi.org/", "").replace("doi:", "")
    return v or None


class DeduplicationEngine:
    def __init__(self, title_threshold: float = 0.92) -> None:
        self.title_threshold = title_threshold

    def deduplicate(self, records: list[PaperRecord], feedback: list[FeedbackEvent] | None = None) -> tuple[list[PaperRecord], ValidationAudit]:
        audit = ValidationAudit()
        blocked_pairs = {(f.left_id, f.right_id) for f in (feedback or []) if not f.accepted}
        merge_pairs: set[tuple[str, str]] = set()
        for f in feedback or []:
            if f.event_type == "merge_hint" and f.accepted:
                merge_pairs.add((f.left_id, f.right_id))
                merge_pairs.add((f.right_id, f.left_id))
        working = records[:]
        merged_records: list[PaperRecord] = []

        while working:
            base = working.pop(0)
            survivors: list[PaperRecord] = []
            for candidate in working:
                if (base.canonical_id, candidate.canonical_id) in blocked_pairs:
                    audit.rejected_pairs.append((base.canonical_id, candidate.canonical_id))
                    survivors.append(candidate)
                    continue
                if (base.canonical_id, candidate.canonical_id) in merge_pairs:
                    base = ConflictResolver.resolve(base, candidate, audit)
                    audit.merged_pairs.append((base.canonical_id, candidate.canonical_id))
                    continue
                if self._is_same_work(base, candidate):
                    base = ConflictResolver.resolve(base, candidate, audit)
                    audit.merged_pairs.append((base.canonical_id, candidate.canonical_id))
                else:
                    survivors.append(candidate)
            merged_records.append(base)
            working = survivors

        return merged_records, audit

    def _is_same_work(self, left: PaperRecord, right: PaperRecord) -> bool:
        left_ids = {_norm_identifier(i) for i in (left.doi, left.arxiv_id, left.semantic_scholar_id, left.openalex_id) if i}
        right_ids = {_norm_identifier(i) for i in (right.doi, right.arxiv_id, right.semantic_scholar_id, right.openalex_id) if i}
        left_ids.discard(None)
        right_ids.discard(None)
        if left_ids.intersection(right_ids):
            return True
        return _title_similarity(left.title, right.title) >= self.title_threshold


class ConflictResolver:
    # Trust hierarchy tuned for identifier integrity.
    SOURCE_PRIORITY = {
        "crossref": 100,
        "openalex": 80,
        "semantic_scholar": 70,
        "arxiv": 60,
    }

    @classmethod
    def resolve(cls, left: PaperRecord, right: PaperRecord, audit: ValidationAudit) -> PaperRecord:
        preferred = cls._pick_preferred(left, right)
        other = right if preferred is left else left

        preferred.abstract = preferred.abstract or other.abstract
        preferred.venue = preferred.venue or other.venue
        preferred.domain = preferred.domain or other.domain
        preferred.references = sorted(set(preferred.references + other.references))
        preferred.authors = preferred.authors or other.authors
        preferred.provenance = preferred.provenance + other.provenance

        for field_name in ("doi", "arxiv_id", "semantic_scholar_id", "openalex_id"):
            pval = getattr(preferred, field_name)
            oval = getattr(other, field_name)
            if pval and oval and pval != oval:
                audit.conflict_log.append(f"conflict({field_name}): {pval} != {oval}")
            if not pval and oval:
                setattr(preferred, field_name, oval)

        return preferred

    @classmethod
    def _pick_preferred(cls, left: PaperRecord, right: PaperRecord) -> PaperRecord:
        left_score = cls._max_source_priority(left.provenance)
        right_score = cls._max_source_priority(right.provenance)
        return left if left_score >= right_score else right

    @classmethod
    def _max_source_priority(cls, prov: list[Provenance]) -> int:
        if not prov:
            return 0
        return max(cls.SOURCE_PRIORITY.get(p.source, 0) for p in prov)

