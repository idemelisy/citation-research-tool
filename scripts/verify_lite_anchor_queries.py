#!/usr/bin/env python3
"""
Verify SRG Lite anchor retrieval against known benchmark queries (live APIs).

Uses the same entry points as the Streamlit Lite UI:
  SRGApplicationService.discover_from_query(..., text_search_backend="openalex")
  SRGApplicationService.run_pipeline(..., discovery_options=LITE_DISCOVERY_OPTIONS)

Run from repo root:
  python scripts/verify_lite_anchor_queries.py
  python scripts/verify_lite_anchor_queries.py --discover-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from srg.api_adapter import SRGApplicationService  # noqa: E402
from srg.app_ui import _paper_map  # noqa: E402
from srg.app_ui_lite import LITE_DISCOVERY_OPTIONS, _resolve_seed_title  # noqa: E402

# OpenAlex work id that used to dominate "transformer attention mechanism" (industrial fault diagnosis).
_TRANSFORMER_FAULT_OPENALEX = "W4200049650"
_GNN_PACK = frozenset({"1609.02907", "1706.02216", "1710.10903", "1810.00826"})


def _arxiv_ids(seeds: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for s in seeds:
        if (s.get("provider") or "").lower() == "arxiv" and (s.get("id") or "").strip():
            out.add(str(s["id"]).strip())
    return out


def _openalex_api_search_ids(seeds: list[dict[str, Any]]) -> list[str]:
    return [
        str(s["id"]).strip()
        for s in seeds
        if (s.get("provider") or "").lower() == "openalex"
        and (s.get("origin") or "").strip() == "api_search"
        and (s.get("id") or "").strip()
    ]


def _paper_has_arxiv(p: dict[str, Any], aid: str) -> bool:
    aid = aid.strip().lower()
    raw = str(p.get("arxiv_id") or "").strip().lower()
    if raw.endswith(aid) or raw == aid:
        return True
    pid = str(p.get("id") or "").lower()
    return aid in pid and "arxiv" in pid


def check_gnn_discover(seeds: list[dict[str, Any]], notes: list[str]) -> tuple[bool, str]:
    aids = _arxiv_ids(seeds)
    missing = sorted(_GNN_PACK - aids)
    if missing:
        return False, f"GNN canonical arXiv pack incomplete; missing: {missing}. Have: {sorted(aids)[:20]}"
    note_hit = any("GNN anchor rule" in n and "matched" in n for n in notes)
    if not note_hit:
        return False, "Expected anchor note containing 'GNN anchor rule' and 'matched' (update api_adapter if message changed)."
    return True, f"GNN pack present; anchor notes OK ({len(notes)} note(s))."


def check_gnn_pipeline(payload: dict[str, Any]) -> tuple[bool, str]:
    papers = payload.get("papers") or []
    if any(_paper_has_arxiv(p, "1609.02907") for p in papers):
        return True, "Kipf GCN (1609.02907) present in graph payload."
    titles = " ".join((p.get("title") or "").lower() for p in papers)
    if "graph convolutional" in titles or "graphsage" in titles or "graph attention network" in titles:
        return True, "Modern GNN line detected in titles (heuristic)."
    return False, "No Kipf id and no obvious modern GNN title in graph — neighborhood may still be precursor-only."


def check_transformer_discover(seeds: list[dict[str, Any]]) -> tuple[bool, str]:
    aids = _arxiv_ids(seeds)
    if "1706.03762" not in aids:
        return False, "Canonical Vaswani arXiv 1706.03762 missing from discover seeds."
    oa_api = _openalex_api_search_ids(seeds)
    if _TRANSFORMER_FAULT_OPENALEX in oa_api:
        return (
            False,
            f"Industrial fault-diagnosis OpenAlex seed {_TRANSFORMER_FAULT_OPENALEX} still in api_search list: {oa_api[:8]}",
        )
    return True, f"Vaswani in seeds; fault OpenAlex id not among api_search OpenAlex ids (n={len(oa_api)})."


def check_transformer_pipeline(payload: dict[str, Any]) -> tuple[bool, str]:
    papers = payload.get("papers") or []
    pmap = _paper_map(papers)
    title = _resolve_seed_title(
        {"provider": "arxiv", "id": "1706.03762", "origin": "api_search"},
        pmap,
    )
    if not title.strip():
        return False, "_resolve_seed_title returned empty for arxiv:1706.03762 (phantom seed vs graph)."
    if "attention" not in title.lower():
        return False, f"Unexpected title for 1706.03762: {title[:120]}"
    if any(_paper_has_arxiv(p, "1706.03762") for p in papers):
        return True, f"Vaswani resolved and in graph: {title[:80]}"
    return True, f"Vaswani title resolved off merged node: {title[:80]}"


def check_rlhf_discover(seeds: list[dict[str, Any]]) -> tuple[bool, str]:
    aids = _arxiv_ids(seeds)
    # At least one well-known alignment paper from canonical list
    if "2305.18290" in aids or "2203.02155" in aids:
        return True, f"RLHF/DPO canonical arXiv anchors present (sample of {len(aids)} arXiv ids)."
    return False, f"Expected RLHF-related arXiv anchors; got arXiv ids: {sorted(aids)[:15]}"


def run_query(
    svc: SRGApplicationService,
    label: str,
    query: str,
    *,
    discover_only: bool,
) -> bool:
    print(f"\n{'=' * 60}\n{label}: {query!r}\n{'=' * 60}", flush=True)
    ok_all = True
    seeds = svc.discover_from_query(query, text_search_backend="openalex")
    notes = list(getattr(svc, "_last_seed_selection_notes", []) or [])
    print(f"discover: {len(seeds)} seeds", flush=True)
    for n in notes[:12]:
        print(f"  note: {n}", flush=True)
    if len(notes) > 12:
        print(f"  note: ... ({len(notes) - 12} more)", flush=True)

    if label == "gnn":
        ok, msg = check_gnn_discover(seeds, notes)
        print(f"  [discover] GNN: {'PASS' if ok else 'FAIL'} - {msg}", flush=True)
        ok_all = ok_all and ok
    elif label == "transformer":
        ok, msg = check_transformer_discover(seeds)
        print(f"  [discover] transformer: {'PASS' if ok else 'FAIL'} - {msg}", flush=True)
        ok_all = ok_all and ok
    elif label == "rlhf":
        ok, msg = check_rlhf_discover(seeds)
        print(f"  [discover] RLHF: {'PASS' if ok else 'FAIL'} - {msg}", flush=True)
        ok_all = ok_all and ok

    if discover_only:
        return ok_all

    print("  run_pipeline ...", flush=True)
    payload = svc.run_pipeline(seeds, discipline=None, discovery_options=LITE_DISCOVERY_OPTIONS)
    n_papers = len([p for p in (payload.get("papers") or []) if not p.get("graph_noise")])
    rq = payload.get("retrieval_quality") or {}
    print(f"  pipeline: {n_papers} papers (non-noise), refs={rq.get('references_total')}", flush=True)

    if label == "gnn":
        ok, msg = check_gnn_pipeline(payload)
        print(f"  [pipeline] GNN: {'PASS' if ok else 'FAIL'} - {msg}", flush=True)
        ok_all = ok_all and ok
    elif label == "transformer":
        ok, msg = check_transformer_pipeline(payload)
        print(f"  [pipeline] transformer: {'PASS' if ok else 'FAIL'} - {msg}", flush=True)
        ok_all = ok_all and ok

    return ok_all


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Lite anchor queries via live APIs.")
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only run discover_from_query (faster; skips full citation pipeline).",
    )
    parser.add_argument(
        "--only",
        choices=("gnn", "transformer", "rlhf"),
        default=None,
        help="Run a single benchmark instead of all three.",
    )
    args = parser.parse_args()

    svc = SRGApplicationService()
    results: list[tuple[str, bool]] = []

    jobs: list[tuple[str, str, str]] = [
        ("gnn", "gnn", "graph neural networks"),
        ("transformer", "transformer", "transformer attention mechanism"),
        ("rlhf", "rlhf", "RLHF reward modeling"),
    ]
    if args.only:
        jobs = [j for j in jobs if j[0] == args.only]

    for key, label, query in jobs:
        results.append((key, run_query(svc, label, query, discover_only=args.discover_only)))

    print(f"\n{'=' * 60}\nSummary\n{'=' * 60}", flush=True)
    failed: list[str] = []
    for name, ok in results:
        line = f"  {name}: {'PASS' if ok else 'FAIL'}"
        print(line, flush=True)
        if not ok:
            failed.append(name)

    if failed:
        print(f"\nFailed: {', '.join(failed)}", flush=True)
        return 1
    print("\nAll checks passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
