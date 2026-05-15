"""Default discovery / ranking options for SRG Lite (shared by UI and evaluator)."""

from __future__ import annotations

from typing import Any

# Field-aware discovery: citation neighborhoods first; lexical/topic signals secondary.
LITE_DISCOVERY_OPTIONS: dict[str, Any] = {
    "top_n": 50,
    "coupling_threshold": 2,
    "enable_two_hop": True,
    "concept_threshold": 0.3,
    "relation_topic_weight": 0.0,
    "foundational_boost": 1.2,
    "allowed_domains": ["cs", "social_sciences", "law", "biomedical", "physics", "ethics"],
    "lite_field_aware": True,
    "w_coupling": 0.75,
    "w_cocitation": 1.0,
    "w_direct": 2.0,
    "lite_retrieval_repair": True,
    "two_hop_budget_floor": 200,
    "lite_coherence_stabilization": True,
    "lite_user_graph_relevance_floor": 0.14,
    "lite_v2_ranking": True,
    "lite_semantic_graph_ranking": False,
    "lite_semantic_control": True,
    "lite_intent_verification": True,
    "export_clean_reading_mode": True,
    "lite_query_conditioned_canonicality": True,
    "lite_semantic_rank_hard_floor": True,
    "lite_foundational_eligibility_rules": True,
    "semantic_rank_min_score": 0.25,
    "below_semantic_floor_multiplier": 0.4,
    "semantic_top10_below_floor_multiplier": 0.35,
    "foundational_min_semantic_score": 0.35,
    "foundational_min_intent_similarity": 0.30,
    "foundational_max_anchor_hops": 3,
}


def discovery_options_with_top_k(top_k: int) -> dict[str, Any]:
    """Copy defaults with graph expansion cap tied to evaluator ``top_k``."""
    opts = dict(LITE_DISCOVERY_OPTIONS)
    opts["top_n"] = max(10, min(200, int(top_k)))
    return opts
