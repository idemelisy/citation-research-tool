"""PR F3 — lightweight schema checks for ``eval_set_v1.json``."""

from __future__ import annotations

import json
from pathlib import Path

from srg.intent_classifier import classify_query


def _eval_set_path() -> Path:
    return Path(__file__).resolve().parents[1] / "eval_set_v1.json"


def test_eval_set_v1_schema_and_coverage() -> None:
    raw = json.loads(_eval_set_path().read_text(encoding="utf-8"))
    queries = raw["queries"]
    assert len(queries) >= 15
    domains = {q["domain"] for q in queries}
    for d in ("graphics", "ml", "systems", "theory", "biomedical"):
        assert d in domains, f"missing domain {d}"
    for q in queries:
        assert q.get("id")
        assert q.get("text")
        assert q.get("gold_intent")
    out = classify_query(queries[0]["text"], {})
    assert "distribution" in out and "intent_distribution" in out
