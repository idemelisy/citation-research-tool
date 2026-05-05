from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .schema import CitationEdge, ConfidenceTier, PaperRecord


@dataclass(slots=True)
class GraphSnapshot:
    nodes: dict[str, PaperRecord]
    edges: list[CitationEdge]
    adjacency: dict[str, set[str]]


class CitationGraphBuilder:
    def resolve_reference_to_canonical(self, ref: str, nodes: dict[str, PaperRecord]) -> str | None:
        """Public wrapper: map a reference token (DOI tail, W…, etc.) to an existing node's canonical_id."""
        return self._resolve_reference(ref, nodes)

    def build(self, papers: list[PaperRecord]) -> GraphSnapshot:
        nodes = {p.canonical_id: p for p in papers}
        edges: list[CitationEdge] = []
        adjacency: dict[str, set[str]] = defaultdict(set)

        for paper in papers:
            for ref in paper.references:
                target = self._resolve_reference(ref, nodes)
                if not target:
                    continue
                context, source_tier, confidence = self._compute_context(paper, nodes[target])
                edges.append(
                    CitationEdge(
                        source_paper_id=paper.canonical_id,
                        target_paper_id=target,
                        context=context,
                        context_source=source_tier,
                        confidence=confidence,
                    )
                )
                adjacency[paper.canonical_id].add(target)

        return GraphSnapshot(nodes=nodes, edges=edges, adjacency=dict(adjacency))

    def _resolve_reference(self, ref: str, nodes: dict[str, PaperRecord]) -> str | None:
        ref_norm = ref.strip().lower().replace("https://doi.org/", "").replace("http://doi.org/", "").replace("doi:", "")
        if ref in nodes:
            return ref
        for pid, record in nodes.items():
            identifiers = {
                (record.doi or "").lower().replace("https://doi.org/", "").replace("http://doi.org/", "").replace("doi:", ""),
                (record.openalex_id or "").lower(),
                (record.arxiv_id or "").lower(),
                (record.semantic_scholar_id or "").lower(),
                record.canonical_id.lower(),
            }
            if ref_norm in identifiers:
                return pid
        return None

    def _compute_context(self, source: PaperRecord, target: PaperRecord) -> tuple[str | None, str | None, ConfidenceTier]:
        # Fallback hierarchy:
        # 1) explicit snippet from provider metadata (not available in this baseline)
        # 2) infer from full text metadata (not available in this baseline)
        # 3) abstract-based proxy if source abstract includes target title terms
        if source.abstract and target.title:
            needle = target.title.split(" ")[0].lower()
            if needle and needle in source.abstract.lower():
                return (f"Abstract mentions '{needle}'", "abstract_proxy", ConfidenceTier.MEDIUM)
        return (None, "unresolved_context", ConfidenceTier.LOW)


class SemanticMetrics:
    def co_citation(self, graph: GraphSnapshot) -> dict[tuple[str, str], int]:
        cited_by: dict[str, set[str]] = defaultdict(set)
        for edge in graph.edges:
            cited_by[edge.target_paper_id].add(edge.source_paper_id)
        keys = sorted(graph.nodes.keys())
        scores: dict[tuple[str, str], int] = {}
        for i, left in enumerate(keys):
            for right in keys[i + 1 :]:
                scores[(left, right)] = len(cited_by[left].intersection(cited_by[right]))
        return scores

    def bibliographic_coupling(self, graph: GraphSnapshot) -> dict[tuple[str, str], int]:
        keys = sorted(graph.nodes.keys())
        scores: dict[tuple[str, str], int] = {}
        for i, left in enumerate(keys):
            for right in keys[i + 1 :]:
                left_refs = graph.adjacency.get(left, set())
                right_refs = graph.adjacency.get(right, set())
                scores[(left, right)] = len(left_refs.intersection(right_refs))
        return scores

