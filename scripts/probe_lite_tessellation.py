"""One-off: run SRG Lite pipeline for tessellation query and print ranking diagnostics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

# repo root = parent of scripts/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from srg.api_adapter import SRGApplicationService  # noqa: E402
from srg.app_ui_lite import LITE_DISCOVERY_OPTIONS  # noqa: E402
from srg.intent_classifier import QueryIntent  # noqa: E402
from srg.synthesis_export import group_papers_query_aware_topics, paper_reading_order_score  # noqa: E402

QUERY = "hardware accelerated tessellation"


def row(p: dict) -> dict:
    title = p.get("title") or ""
    return {
        "title": title[:72],
        "sem": p.get("semantic_score"),
        "graph": p.get("graph_score"),
        "canon": p.get("canonicality_score"),
        "fused_pre": p.get("fused_score_pre_semantic_control"),
        "intent_sim": p.get("intent_similarity"),
        "rel_raw": p.get("relevance_raw"),
        "rel_pre_iv": p.get("relevance_raw_pre_intent_verification"),
        "rel_div_n": p.get("relevance_diverse_norm"),
        "rel_n": p.get("relevance_norm"),
        "cc": p.get("citation_count"),
        "hops": p.get("anchor_graph_hops"),
        "year": p.get("year"),
        "imp": round(
            paper_reading_order_score(
                p,
                intent_mode_v2=QueryIntent.HARDWARE_SYSTEM,
                query_text=QUERY,
            ),
            4,
        ),
        "pending": title.startswith("Pending metadata"),
    }


def main() -> None:
    svc = SRGApplicationService()
    q = "hardware accelerated tessellation"
    print("=== discover ===", flush=True)
    seeds = svc.discover_from_query(q, text_search_backend="openalex")
    print("seeds:", len(seeds), flush=True)
    if not seeds:
        raise SystemExit("no seeds")
    print("=== pipeline ===", flush=True)
    payload = svc.run_pipeline(seeds, discipline=None, discovery_options=LITE_DISCOVERY_OPTIONS)
    papers = [p for p in (payload.get("papers") or []) if not p.get("graph_noise")]
    qp = payload.get("query_profile") or {}
    print("intent_mode_v2:", qp.get("intent_mode_v2"))
    print("papers:", len(papers), flush=True)

    ranked = sorted(
        papers,
        key=lambda p: paper_reading_order_score(
            p,
            intent_mode_v2=QueryIntent.HARDWARE_SYSTEM,
            query_text=QUERY,
        ),
        reverse=True,
    )
    print("\n=== top 12 reading-list importance ===")
    for i, p in enumerate(ranked[:12], 1):
        r = row(p)
        print(i, r["imp"], "y=", r["year"], "cc=", r["cc"], "hops=", r["hops"], r["title"][:64])

    targets = ("signal processing approach to fair surface", "realtime gpu subdivision")
    print("\n=== target rows ===")
    for p in papers:
        t = (p.get("title") or "").lower()
        if any(s in t for s in targets):
            print(json.dumps(row(p), indent=2))

    print("\n=== foundational hubs (first 12) ===")
    found = [p for p in papers if p.get("is_foundational_hub")]
    found.sort(
        key=lambda p: paper_reading_order_score(
            p,
            intent_mode_v2=QueryIntent.HARDWARE_SYSTEM,
            query_text=QUERY,
        ),
        reverse=True,
    )
    for p in found[:12]:
        tag = "P " if (p.get("title") or "").startswith("Pending") else "  "
        print(
            tag,
            round(
                paper_reading_order_score(
                    p,
                    intent_mode_v2=QueryIntent.HARDWARE_SYSTEM,
                    query_text=QUERY,
                ),
                3,
            ),
            (p.get("title") or "")[:68],
            "eligible=",
            p.get("foundational_eligible"),
        )

    branches = group_papers_query_aware_topics(papers, qp)
    print("\n=== branch labels ===")
    for k in branches:
        print("-", repr(k), "n=", len(branches[k]))

    rq = payload.get("retrieval_quality") or {}
    print("\n=== rq warnings ===", rq.get("warnings"))
    iv = (rq.get("v2_1_diagnostics") or {}).get("intent_verification") or {}
    print("intent_verification:", iv)


if __name__ == "__main__":
    main()
