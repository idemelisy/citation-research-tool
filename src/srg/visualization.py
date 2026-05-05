from __future__ import annotations

from .schema import ConfidenceTier, CitationEdge


def edge_style(edge: CitationEdge) -> dict[str, str]:
    if edge.confidence == ConfidenceTier.HIGH:
        return {"stroke": "solid", "weight": "normal", "opacity": "1.0", "tooltip": "high-confidence context"}
    if edge.confidence == ConfidenceTier.MEDIUM:
        return {"stroke": "solid", "weight": "normal", "opacity": "0.6", "tooltip": "medium-confidence context"}
    return {
        "stroke": "dashed",
        "weight": "thin",
        "opacity": "0.5",
        "tooltip": "citation exists, context unavailable",
    }

