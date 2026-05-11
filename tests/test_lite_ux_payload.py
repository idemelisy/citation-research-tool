"""Tests for pipeline ``lite_ux`` bundle (start_here, branches)."""

from __future__ import annotations

from srg.lite_ux import build_lite_ux_payload, start_here_tags_for_paper


def _paper(pid: str, title: str, **kw: object) -> dict:
    row = {
        "id": pid,
        "title": title,
        "year": 2022,
        "graph_noise": False,
        "citation_count": 3,
        "relevance_norm": 0.9,
        "relevance_diverse_norm": 0.88,
        "seed_origin": "discovered",
    }
    row.update(kw)
    return row


def test_build_lite_ux_payload_structure():
    papers = [
        _paper("a", "Direct Preference Optimization for Language Model Alignment", relevance_norm=0.95),
        _paper("b", "Reward Modeling Alternatives for Preference Learning", relevance_norm=0.85),
        _paper("c", "Protein Structure Prediction with Energy-Based Models", relevance_norm=0.75),
        _paper("d", "Survey of Preference Optimization Methods", relevance_norm=0.7, is_foundational_hub=True),
    ]
    qp = {
        "query_text": "preference optimization DPO",
        "query_terms": ["preference", "optimization", "dpo"],
        "intent_mode_v2": "method",
    }
    recs = [{"paper_id": "a", "score": 1.0, "reason": "test"}, {"paper_id": "b", "score": 0.9, "reason": "test"}]
    payload = {"papers": papers, "query_profile": qp, "discovery": {"recommendations": recs}}
    ux = build_lite_ux_payload(payload)
    assert int(ux.get("version") or 0) >= 2
    assert isinstance(ux.get("start_here"), list)
    assert isinstance(ux.get("branches"), list)
    rh = ux.get("retrieval_health")
    assert isinstance(rh, dict) and "signals" in rh
    assert isinstance(ux.get("reading_paths"), list)
    assert isinstance(ux.get("insights"), list)
    for e in ux["start_here"]:
        assert "paper_id" in e and "title" in e and "one_line" in e and "tags" in e
    for b in ux["branches"]:
        assert "label" in b and "paper_ids" in b and "summary" in b and "central_paper_id" in b


def test_start_here_tags_foundational():
    p = {"is_foundational_hub": True, "foundational_eligible": True, "year": 2020, "title": "A survey of RLHF"}
    tags = start_here_tags_for_paper(p)
    assert "Foundational" in tags
    assert "Survey" in tags
    # Stable product order
    assert tags == ["Foundational", "Survey"]


def test_start_here_tags_canonical_and_cross_domain():
    p = {
        "title": "Some method paper",
        "year": 2019,
        "canonicality_score": 0.75,
        "domain": "biomedical",
    }
    qp = {"query_inferred_domains": ["cs", "social_sciences"]}
    tags = start_here_tags_for_paper(p, qp)
    assert "Canonical" in tags
    assert "Cross-domain" in tags
    assert tags.index("Canonical") < tags.index("Cross-domain")


def test_branch_label_strips_query_terms():
    from srg.lite_ux import _literature_branch_label

    central = {"title": "Scaling Laws for Reward Model Overfitting in DPO Training"}
    qp = {"query_text": "DPO training", "query_terms": ["dpo"]}
    lab = _literature_branch_label(central, qp)
    assert "dpo" not in lab.lower()


def test_reading_paths_and_branch_why_included():
    papers = [
        _paper("a", "Old Paper Alpha", year=2018, relevance_norm=0.92, relation_expand_score=0.2),
        _paper("b", "New Paper Beta", year=2024, relevance_norm=0.88, relation_expand_score=0.2),
    ]
    graph = {"edges": [{"source": "b", "target": "a", "style": {}, "confidence": "high"}]}
    qp = {"query_text": "alpha beta methods", "query_terms": ["alpha", "beta"], "intent_mode_v2": "method"}
    recs = [{"paper_id": "a", "score": 1.0, "reason": "r"}]
    payload = {
        "papers": papers,
        "graph": graph,
        "query_profile": qp,
        "discovery": {"recommendations": recs},
        "evidence_stats": {"references_total": 40, "graph_edges_total": 1},
        "retrieval_quality": {"warnings": [], "effective_seed_total": 2},
    }
    ux = build_lite_ux_payload(payload)
    assert any((b.get("why_included") or "").strip() for b in ux["branches"])
    assert isinstance(ux.get("reading_paths"), list) and len(ux["reading_paths"]) >= 1


def test_near_duplicate_filters_start_here():
    papers = [
        _paper("p1", "Learning from Human Preferences via Reward Models", relevance_norm=0.99),
        _paper(
            "p2",
            "Learning from Human Preferences via Reward Models: Extended Analysis",
            relevance_norm=0.98,
        ),
        _paper("p3", "Completely Different Topic in Robotics Control", relevance_norm=0.5),
    ]
    qp = {"query_text": "RLHF", "query_terms": ["rlhf"], "intent_mode_v2": "method"}
    recs = [{"paper_id": "p1", "score": 1.0, "reason": "r"}, {"paper_id": "p2", "score": 0.9, "reason": "r"}]
    payload = {"papers": papers, "query_profile": qp, "discovery": {"recommendations": recs}}
    ux = build_lite_ux_payload(payload)
    ids = [e["paper_id"] for e in ux["start_here"]]
    assert "p1" in ids
    assert "p2" not in ids or ids.count("p2") == 0
