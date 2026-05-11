"""SRG Lite v2.2 — fusion, metrics, and classifier smoke tests."""

from __future__ import annotations

from srg.intent_classifier import QueryIntent, classify_query, hardware_v22_feature_hit_count
from srg.ranking_fusion import (
    compute_final_score,
    distribution_uncertain,
    primary_sem_rec_graph_key,
)
from srg.roadmap_metrics import (
    hardware_recency_bias_score,
    intent_accuracy,
    precision_at_5,
    recall_at_10,
)


def test_hardware_keyword_hits_and_classification() -> None:
    q = "GPU tessellation pipeline"
    assert hardware_v22_feature_hit_count(q) >= 2
    r = classify_query(q, {})
    assert r["intent"] == QueryIntent.HARDWARE_SYSTEM
    assert r.get("hardware_recency_fusion_boost") is True
    assert "distribution" in r and isinstance(r["distribution"], dict)


def test_recency_triple_fusion_demotes_old_recency() -> None:
    high_rec = compute_final_score(0.5, 0.9, 0.8, QueryIntent.HARDWARE_SYSTEM, recency=0.9)
    low_rec = compute_final_score(0.5, 0.9, 0.8, QueryIntent.HARDWARE_SYSTEM, recency=0.1)
    assert high_rec > low_rec


def test_low_recency_multiplier() -> None:
    base = compute_final_score(0.6, 0.5, 0.5, QueryIntent.TECHNICAL_METHOD, recency=0.3)
    low = compute_final_score(0.6, 0.5, 0.5, QueryIntent.TECHNICAL_METHOD, recency=0.1)
    assert low < base * 0.99


def test_distribution_uncertain_flag() -> None:
    assert distribution_uncertain({"a": 0.2, "b": 0.2, "c": 0.2}) is True
    assert distribution_uncertain({"a": 0.95, "b": 0.05}) is False


def test_primary_sem_rec_key() -> None:
    k = primary_sem_rec_graph_key(0.5, 0.9, 0.2, hardware_recency_boost=False)
    assert 0.0 <= k <= 1.0


def test_roadmap_metrics_helpers() -> None:
    assert precision_at_5(["a", "b", "c", "d", "e", "f"], {"a", "x"}) == 0.2
    assert recall_at_10(["x", "a", "b"], {"a", "b"}) == 2.0 / 2.0
    assert intent_accuracy("hardware_system", "hardware_system") == 1.0
    nodes = [{"year": 2024}, {"year": 2010}]
    s = hardware_recency_bias_score(nodes, query_intent=QueryIntent.HARDWARE_SYSTEM, top_k=2)
    assert s > 0.5


def test_classify_query_always_has_distribution() -> None:
    r = classify_query("anything unknown xyzabc", {})
    assert isinstance(r.get("distribution"), dict)
    assert abs(sum(r["distribution"].values()) - 1.0) < 0.02
