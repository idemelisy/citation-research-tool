from __future__ import annotations

from .graph import GraphSnapshot
from .visualization import edge_style


def render_edge_legend() -> str:
    return (
        "Edge legend:\n"
        "  - solid/opacity=1.0: high-confidence context\n"
        "  - solid/opacity=0.6: medium-confidence context\n"
        "  - dashed/thin: citation exists, context unavailable\n"
    )


def render_graph_text(graph: GraphSnapshot) -> str:
    lines = [f"nodes={len(graph.nodes)} edges={len(graph.edges)}", render_edge_legend()]
    for edge in graph.edges[:20]:
        style = edge_style(edge)
        lines.append(
            f"{edge.source_paper_id} -> {edge.target_paper_id} "
            f"[{edge.confidence.value}, {style['stroke']}, {style['tooltip']}]"
        )
    return "\n".join(lines)

