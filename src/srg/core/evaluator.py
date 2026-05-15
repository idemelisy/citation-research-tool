"""
CLI: ``python -m srg.core.evaluator`` — batch benchmark over reading-list pipeline.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from .cache import cache_key, load_cached, save_cached
from .datasets import benchmark_query_list
from .metrics import compute_all_metrics
from .pipeline import generate_reading_list
from .report import (
    write_failure_cases_md,
    write_html_dashboard,
    write_leaderboard_md,
    write_metrics_csv,
    write_results_json,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("evaluation_outputs")


def _enrich_result_for_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Attach branch member titles for coherence metrics."""
    payload = result.get("payload") or {}
    papers = {str(p.get("id")): p for p in (payload.get("papers") or []) if isinstance(p, dict)}
    branches = []
    for br in result.get("branches") or []:
        if not isinstance(br, dict):
            continue
        nb = dict(br)
        titles = []
        for pid in nb.get("paper_ids") or []:
            p = papers.get(str(pid)) or {}
            t = (p.get("title") or "").strip()
            if t:
                titles.append(t)
        nb["member_titles"] = titles
        branches.append(nb)
    out = dict(result)
    out["branches"] = branches
    return out


def evaluate_query(
    query: str,
    *,
    top_k: int = 40,
    use_cache: bool = False,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    key = cache_key(query, top_k)
    cdir = cache_dir or (DEFAULT_OUTPUT / "cache")
    if use_cache:
        cached = load_cached(cdir, key)
        if cached:
            cached["from_cache"] = True
            return cached

    try:
        result = generate_reading_list(query, top_k=top_k)
        result["from_cache"] = False
    except Exception as exc:
        logger.exception("Query failed: %s", query)
        return {"query": query, "error": str(exc), "metrics": {"composite": 0.0}}

    enriched = _enrich_result_for_metrics(result)
    result["metrics"] = compute_all_metrics(enriched)
    if use_cache:
        save_cached(cdir, key, result)
    return result


def run_evaluation(
    queries: list[str],
    *,
    top_k: int = 40,
    output_dir: Path = DEFAULT_OUTPUT,
    use_cache: bool = True,
    save_html: bool = True,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    rows: list[dict[str, Any]] = []

    for i, q in enumerate(queries, 1):
        logger.info("[%d/%d] %s", i, len(queries), q)
        row = evaluate_query(q, top_k=top_k, use_cache=use_cache, cache_dir=cache_dir)
        rows.append(row)

    write_results_json(output_dir / "results.json", rows)
    write_metrics_csv(output_dir / "metrics.csv", rows)
    write_leaderboard_md(output_dir / "leaderboard.md", rows)
    write_failure_cases_md(output_dir / "failure_cases.md", rows)
    if save_html:
        write_html_dashboard(output_dir / "dashboard.html", rows)

    logger.info("Wrote %d results to %s", len(rows), output_dir.resolve())
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SRG Lite reading-list evaluator")
    parser.add_argument("--query", type=str, help="Evaluate a single query only")
    parser.add_argument("--top-k", type=int, default=40, help="Graph expansion / candidate cap")
    parser.add_argument("--limit", type=int, default=None, help="Max benchmark queries to run")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-cache", action="store_true", help="Disable disk cache")
    parser.add_argument("--save-html", action="store_true", default=True)
    parser.add_argument("--no-html", action="store_true", help="Skip HTML dashboard")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.query:
        row = evaluate_query(
            args.query.strip(),
            top_k=args.top_k,
            use_cache=not args.no_cache,
            cache_dir=args.output_dir / "cache",
        )
        m = row.get("metrics") or {}
        print(f"Query: {row.get('query')}")
        if row.get("error"):
            print(f"Error: {row['error']}")
            return 1
        print(f"Composite: {m.get('composite')}")
        for k, v in sorted(m.items()):
            if k != "composite":
                print(f"  {k}: {v}")
        found = row.get("foundational_papers") or []
        if found:
            print("\nFoundational:")
            for f in found[:8]:
                if isinstance(f, dict):
                    print(f"  - {f.get('title', '?')}")
        return 0

    queries = benchmark_query_list(limit=args.limit)
    if not queries:
        print("No benchmark queries.", file=sys.stderr)
        return 1

    run_evaluation(
        queries,
        top_k=args.top_k,
        output_dir=args.output_dir,
        use_cache=not args.no_cache,
        save_html=not args.no_html,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
