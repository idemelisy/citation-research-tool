"""Unit tests for srg.core metrics, cache, and report helpers."""

from __future__ import annotations

import json
from pathlib import Path

from srg.core.cache import cache_key, load_cached, save_cached
from srg.core.datasets import benchmark_query_list, canonical_titles_for_query
from srg.core.metrics import compute_all_metrics, title_similarity
from srg.core.report import write_leaderboard_md, write_metrics_csv, write_results_json


def test_benchmark_query_count():
    qs = benchmark_query_list()
    assert len(qs) >= 40


def test_canonical_titles_dpo():
    titles = canonical_titles_for_query("direct preference optimization")
    assert any("Direct Preference Optimization" in t for t in titles)


def test_title_similarity_substring():
    a = "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
    b = "direct preference optimization your language model"
    assert title_similarity(a, b) >= 0.7


def test_metrics_on_synthetic_result():
    result = {
        "query": "direct preference optimization",
        "foundational_papers": [
            {
                "paper_id": "p1",
                "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
                "role": "canonical_anchor",
                "why_included": "Canonical anchor paper defining Direct Preference Optimization for LLM alignment.",
            }
        ],
        "branches": [
            {
                "id": "b1",
                "paper_ids": ["p1", "p2"],
                "member_titles": [
                    "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
                    "ORPO: Monolithic Preference Optimization",
                ],
                "why_included": "Papers cluster around preference optimization objectives.",
            }
        ],
        "raw_candidates": [
            {
                "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
                "semantic_score": 0.8,
                "year": 2023,
            },
            {"title": "ORPO: Monolithic Preference Optimization", "semantic_score": 0.7, "year": 2024},
        ],
        "stats": {"paper_count": 2},
    }
    m = compute_all_metrics(result)
    assert m["canonical_hit_rate"] == 1.0
    assert m["composite"] > 0.5
    assert 0.0 <= m["noise_ratio"] <= 1.0


def test_cache_roundtrip(tmp_path: Path):
    key = cache_key("dpo", 40)
    data = {"query": "dpo", "metrics": {"composite": 0.9}}
    save_cached(tmp_path, key, data)
    loaded = load_cached(tmp_path, key)
    assert loaded is not None
    assert loaded["query"] == "dpo"


def test_report_writers(tmp_path: Path):
    rows = [
        {"query": "dpo", "metrics": {"composite": 0.8, "intent_precision": 0.7, "noise_ratio": 0.1}},
        {"query": "rag", "metrics": {"composite": 0.5, "intent_precision": 0.4, "noise_ratio": 0.3}},
    ]
    write_results_json(tmp_path / "results.json", rows)
    write_metrics_csv(tmp_path / "metrics.csv", rows)
    write_leaderboard_md(tmp_path / "leaderboard.md", rows)
    assert json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))["results"]
    assert "dpo" in (tmp_path / "leaderboard.md").read_text(encoding="utf-8")
