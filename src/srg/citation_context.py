"""Provider-backed citation context (snippets) with graceful fallback."""

from __future__ import annotations

import threading
import time
from typing import Any

import requests

from .graph import GraphSnapshot
from .schema import CitationEdge, ConfidenceTier, PaperRecord


class ProviderBudget:
    """Per-provider minimum interval between HTTP calls (rate-limit shield)."""

    def __init__(self, intervals: dict[str, float] | None = None) -> None:
        self._lock = threading.Lock()
        self._last: dict[str, float] = {}
        self.intervals = intervals or {
            "crossref": 1.0,
            "openalex": 0.25,
            "arxiv": 0.5,
            "semantic_scholar": 0.35,
            "doi_org": 1.0,
        }

    def acquire(self, provider: str) -> None:
        interval = self.intervals.get(provider, 0.35)
        with self._lock:
            last = self._last.get(provider, 0.0)
            now = time.time()
            wait = max(0.0, interval - (now - last))
            if wait > 0:
                time.sleep(wait)
            self._last[provider] = time.time()


def fetch_s2_reference_context(
    source_s2_id: str,
    target_s2_id: str,
    budget: ProviderBudget | None = None,
    timeout: int = 25,
) -> tuple[str | None, str | None]:
    """
    Fetch citation contexts from Semantic Scholar: source paper's references
    that point to target paperId.
    Returns (snippet_text, source_label) or (None, None).
    """
    if budget:
        budget.acquire("semantic_scholar")
    url = f"https://api.semanticscholar.org/graph/v1/paper/{source_s2_id}"
    params = {"fields": "references.paperId,references.contexts"}
    try:
        r = requests.get(
            url,
            params=params,
            timeout=timeout,
            headers={"User-Agent": "SemanticResearchGraph/0.1"},
        )
        r.raise_for_status()
        data = r.json()
    except requests.RequestException:
        return None, None

    for ref in data.get("references") or []:
        if not isinstance(ref, dict):
            continue
        if ref.get("paperId") != target_s2_id:
            continue
        contexts = ref.get("contexts") or []
        if isinstance(contexts, list) and contexts:
            text = contexts[0] if isinstance(contexts[0], str) else str(contexts[0])
            text = text.strip()
            if text:
                return text, "semantic_scholar_snippet"
    return None, None


def enrich_graph_with_citation_snippets(
    snapshot: GraphSnapshot,
    papers: dict[str, PaperRecord],
    budget: ProviderBudget | None = None,
) -> GraphSnapshot:
    """Replace LOW/MEDIUM abstract-proxy edges with S2 snippets when both ends have S2 IDs."""
    if budget is None:
        budget = ProviderBudget()
    new_edges: list[CitationEdge] = []
    for edge in snapshot.edges:
        src = papers.get(edge.source_paper_id)
        tgt = papers.get(edge.target_paper_id)
        context = edge.context
        context_source = edge.context_source
        confidence = edge.confidence

        if src and tgt and src.semantic_scholar_id and tgt.semantic_scholar_id:
            snippet, src_label = fetch_s2_reference_context(
                src.semantic_scholar_id,
                tgt.semantic_scholar_id,
                budget=budget,
            )
            if snippet:
                context = snippet
                context_source = src_label
                confidence = ConfidenceTier.HIGH

        new_edges.append(
            CitationEdge(
                source_paper_id=edge.source_paper_id,
                target_paper_id=edge.target_paper_id,
                context=context,
                context_source=context_source,
                confidence=confidence,
            )
        )

    return GraphSnapshot(nodes=snapshot.nodes, edges=new_edges, adjacency=snapshot.adjacency)
