"""Reading-order score: hardware + tessellation query guards."""

from srg.intent_classifier import QueryIntent
from srg.synthesis_export import paper_reading_order_score

_Q = "hardware accelerated tessellation"


def test_hardware_mode_prefers_gpu_title_over_high_degree_theory_hub() -> None:
    """Synthetic snapshot from tessellation query: high-degree classic paper vs GPU kernel."""
    hub = {
        "title": "A signal processing approach to fair surface design",
        "relevance_diverse_norm": 0.6135,
        "citation_count": 62,
        "year": 1995,
        "missing_link_score": 0.0,
    }
    gpu = {
        "title": "A realtime GPU subdivision kernel",
        "relevance_diverse_norm": 0.6221,
        "citation_count": 27,
        "year": 2005,
        "missing_link_score": 0.0,
    }
    s_hub = paper_reading_order_score(
        hub, intent_mode_v2=QueryIntent.HARDWARE_SYSTEM, query_text=_Q
    )
    s_gpu = paper_reading_order_score(
        gpu, intent_mode_v2=QueryIntent.HARDWARE_SYSTEM, query_text=_Q
    )
    assert s_gpu > s_hub, (s_gpu, s_hub)

    legacy_hub = paper_reading_order_score(hub, intent_mode_v2=None, query_text=_Q)
    legacy_gpu = paper_reading_order_score(gpu, intent_mode_v2=None, query_text=_Q)
    assert legacy_hub > legacy_gpu, (legacy_hub, legacy_gpu)


def test_tessellation_query_demotes_parallax_over_subdivision_hardware_paper() -> None:
    """Same synthetic relevance — parallax tricks should lose to Gregory / hardware tessellation."""
    parallax = {
        "title": "Dynamic parallax occlusion mapping with approximate soft shadows",
        "relevance_diverse_norm": 0.72,
        "citation_count": 8,
        "year": 2006,
        "missing_link_score": 0.0,
    }
    gregory = {
        "title": "Approximating subdivision surfaces with Gregory patches for hardware tessellation",
        "relevance_diverse_norm": 0.66,
        "citation_count": 20,
        "year": 2009,
        "missing_link_score": 0.0,
    }
    s_bad = paper_reading_order_score(
        parallax, intent_mode_v2=QueryIntent.HARDWARE_SYSTEM, query_text=_Q
    )
    s_good = paper_reading_order_score(
        gregory, intent_mode_v2=QueryIntent.HARDWARE_SYSTEM, query_text=_Q
    )
    assert s_good > s_bad, (s_good, s_bad)


def test_tessellation_multiplier_inactive_without_tessell_in_query() -> None:
    """Broad hardware query should not apply tessellation-only demotions."""
    parallax = {
        "title": "Dynamic parallax occlusion mapping with approximate soft shadows",
        "relevance_diverse_norm": 0.7,
        "citation_count": 8,
        "year": 2006,
        "missing_link_score": 0.0,
    }
    a = paper_reading_order_score(
        parallax,
        intent_mode_v2=QueryIntent.HARDWARE_SYSTEM,
        query_text="real-time gpu rendering",
    )
    b = paper_reading_order_score(
        parallax,
        intent_mode_v2=QueryIntent.HARDWARE_SYSTEM,
        query_text=_Q,
    )
    assert a > b
