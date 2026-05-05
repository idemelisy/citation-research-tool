from __future__ import annotations

from srg.discovery import SuggestionEngine
from srg.graph import CitationGraphBuilder
from srg.schema import Author, ConfidenceTier, FeedbackEvent, PaperRecord, Provenance
from srg.validation import DeduplicationEngine
from srg.visualization import edge_style


def _paper(pid: str, title: str, refs: list[str], doi: str | None = None) -> PaperRecord:
    return PaperRecord(
        canonical_id=pid,
        title=title,
        abstract="transformer models cite attention",
        year=2024,
        venue="testconf",
        domain="cs",
        doi=doi,
        authors=[Author(name="Ada")],
        references=refs,
        provenance=[Provenance(source="crossref", source_id=doi or pid)],
    )


def test_dedup_merges_by_identifier() -> None:
    left = _paper("left", "Attention Is All You Need", refs=[], doi="10.5555/abc")
    right = _paper("right", "Attention is all you need", refs=[], doi="10.5555/abc")
    merged, audit = DeduplicationEngine().deduplicate([left, right])
    assert len(merged) == 1
    assert len(audit.merged_pairs) == 1


def test_dedup_merge_hint_forces_merge() -> None:
    left = _paper("left", "Totally Different Title A", refs=[], doi="10.5555/aaa")
    right = _paper("right", "Unrelated Title B", refs=[], doi="10.5555/bbb")
    feedback = [FeedbackEvent(event_type="merge_hint", left_id="left", right_id="right", accepted=True)]
    merged, audit = DeduplicationEngine(title_threshold=0.99).deduplicate([left, right], feedback=feedback)
    assert len(merged) == 1
    assert len(audit.merged_pairs) >= 1


def test_graph_marks_low_confidence_when_no_context() -> None:
    a = _paper("a", "Paper A", refs=["b"])
    a.abstract = None
    b = _paper("b", "Paper B", refs=[])
    graph = CitationGraphBuilder().build([a, b])
    assert len(graph.edges) == 1
    assert graph.edges[0].confidence == ConfidenceTier.LOW
    assert edge_style(graph.edges[0])["stroke"] == "dashed"


def test_domain_aware_recommendations() -> None:
    a = _paper("a", "Paper A", refs=["b"])
    b = _paper("b", "Paper B", refs=[])
    graph = CitationGraphBuilder().build([a, b])
    recs = SuggestionEngine().recommend_missing_links(graph)
    assert len(recs) > 0
    assert recs[0].score >= recs[-1].score

