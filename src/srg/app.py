from __future__ import annotations

import argparse
import os
import subprocess
import sys

from .discovery import SuggestionEngine
from .evaluation import build_quality_report
from .feedback import FeedbackStore
from .graph import CitationGraphBuilder, SemanticMetrics
from .ingestion import ArXivClient, CacheStore, CrossrefClient, IngestionOrchestrator, OpenAlexClient, SemanticScholarClient
from .schema import FeedbackEvent
from .ui import render_graph_text
from .validation import DeduplicationEngine
from .visualization import edge_style


def run_demo() -> None:
    # Seed examples are placeholders to demonstrate the pipeline wiring.
    orchestrator = IngestionOrchestrator(
        clients=[OpenAlexClient(), CrossrefClient(), ArXivClient(), SemanticScholarClient()],
        cache=CacheStore(),
    )
    for provider, seed in [
        ("openalex", "W2741809807"),
        ("crossref", "10.1038/nphys1170"),
        ("arxiv", "1706.03762"),
    ]:
        orchestrator.enqueue(provider, seed)

    records = orchestrator.run()

    feedback_store = FeedbackStore()
    feedback = (
        [FeedbackEvent(event_type="merge_hint", left_id=records[0].canonical_id, right_id=records[1].canonical_id, accepted=True)]
        if len(records) > 1
        else []
    )
    if feedback:
        feedback_store.bulk_add(feedback)
    active_feedback = feedback_store.list_all()

    deduper = DeduplicationEngine()
    merged, audit = deduper.deduplicate(records, feedback=active_feedback)
    quality = build_quality_report(audit)

    graph = CitationGraphBuilder().build(merged)
    metrics = SemanticMetrics()
    co_citation = metrics.co_citation(graph)
    coupling = metrics.bibliographic_coupling(graph)

    discovery = SuggestionEngine()
    recommendations = discovery.recommend_missing_links(graph)
    trends = discovery.trend_and_gap(graph)

    print("=== SRG Demo ===")
    print(f"ingested={len(records)} merged={len(merged)}")
    print(
        f"merged_pairs={quality.merged_pairs} conflicts={quality.conflicts} "
        f"precision={quality.precision_proxy} recall={quality.recall_proxy}"
    )
    print(f"edges={len(graph.edges)} co_citation_pairs={len(co_citation)} coupling_pairs={len(coupling)}")
    if recommendations:
        print(f"top_recommendation={recommendations[0].paper_id} score={recommendations[0].score}")
    print(f"trend_keys={list(trends['timeline'].keys())[:5]}")
    if graph.edges:
        print(f"first_edge_style={edge_style(graph.edges[0])}")
    print(render_graph_text(graph))
    print(f"feedback_events={len(active_feedback)}")


def run_streamlit(projects_dir: str | None = None) -> int:
    env = os.environ.copy()
    if projects_dir:
        env["SRG_PROJECTS_DIR"] = projects_dir
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", "src/srg/app_ui.py"],
        env=env,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="SRG application entrypoint")
    parser.add_argument(
        "--mode",
        choices=["demo", "ui"],
        default="demo",
        help="demo: CLI pipeline output, ui: launch Streamlit interface",
    )
    parser.add_argument(
        "--projects-dir",
        default=None,
        metavar="DIR",
        help="Directory for saved SRG project JSON files; passed to the UI as SRG_PROJECTS_DIR (default: ./srg_projects under cwd).",
    )
    args = parser.parse_args()

    if args.mode == "ui":
        raise SystemExit(run_streamlit(args.projects_dir))
    run_demo()


if __name__ == "__main__":
    main()

