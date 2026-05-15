"""Ambiguity regression tests — DPO, transformer, visual debugging precision."""

from __future__ import annotations

from srg.api_adapter import SRGApplicationService
from srg.explainability import build_why_it_matters
from srg.intent_coherence import compute_intent_coherence
from srg.lite_ux import build_lite_ux_payload
from srg.query_intent_normalization import normalize_query_intent
from srg.flagship_registry import pin_flagship_into_start_here, resolve_flagship_paper_ids
from srg.topical_ranking import (
    apply_topical_ranking_layer,
    compute_query_centrality,
    generic_paper_penalty,
    strict_start_here_score,
)


def _paper(pid: str, title: str, **kw: object) -> dict:
    row = {
        "id": pid,
        "title": title,
        "year": 2023,
        "abstract": "",
        "graph_noise": False,
        "seed_origin": "discovered",
        "citation_count": 5,
        "relevance_norm": 0.8,
        "relevance_diverse_norm": 0.78,
        "semantic_score": 0.5,
        "canonicality_score": 0.5,
        "anchor_graph_hops": 2,
        "relation_expand_score": 0.15,
    }
    row.update(kw)
    return row


def _coherence(title: str, abstract: str, query: str, entry_id: str, enrichment: list[str]) -> float:
    node = _paper("x", title, abstract=abstract)
    bundle = compute_intent_coherence(
        node,
        query_text=query,
        enrichment_terms=enrichment,
        intent_entry_id=entry_id,
        query_domains=frozenset({"cs"}),
        semantic_score=0.55,
        anchor_hops=2,
    )
    return float(bundle["intent_coherence_score"])


def test_normalize_dpo_intent():
    nqi = normalize_query_intent("direct preference optimization")
    assert nqi.intent_entry_id == "direct_preference_optimization"
    assert "RLHF" in nqi.enrichment_terms or "rlhf" in [t.lower() for t in nqi.enrichment_terms]
    assert "2305.18290" in nqi.canonical_arxiv_ids
    assert nqi.ambiguity_score >= 0.65


def test_normalize_transformer_excludes_power_context():
    nqi = normalize_query_intent("transformer")
    assert nqi.intent_entry_id == "transformer_nlp"
    assert "attention" in [t.lower() for t in nqi.enrichment_terms]
    nqi_power = normalize_query_intent("power transformer fault diagnosis")
    assert nqi_power.intent_entry_id != "transformer_nlp"


def test_normalize_visual_debugging():
    nqi = normalize_query_intent("visual debugging")
    assert nqi.intent_entry_id == "visual_debugging"
    assert any("debug" in t.lower() for t in nqi.enrichment_terms)


def test_dpo_cluster_dominance_and_recommender_suppression():
    query = "direct preference optimization"
    nqi = normalize_query_intent(query)
    enrich = list(nqi.enrichment_terms)
    dpo_score = _coherence(
        "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        "RLHF alignment language model preference tuning DPO",
        query,
        nqi.intent_entry_id or "",
        enrich,
    )
    reco_score = _coherence(
        "Session-based Recommendation with Sequential Preference Learning",
        "collaborative filtering click-through recommender system",
        query,
        nqi.intent_entry_id or "",
        enrich,
    )
    assert dpo_score > reco_score + 0.12
    reco_bundle = compute_intent_coherence(
        _paper("r", "Session-based Recommendation", abstract="collaborative filtering"),
        query_text=query,
        enrichment_terms=enrich,
        intent_entry_id=nqi.intent_entry_id,
        query_domains=frozenset({"cs"}),
        semantic_score=0.4,
    )
    assert reco_bundle.get("intent_penalty_cluster") == "recommender_systems"


def test_transformer_nlp_over_power_systems():
    query = "transformer"
    nqi = normalize_query_intent(query)
    enrich = list(nqi.enrichment_terms)
    nlp_score = _coherence(
        "Attention Is All You Need",
        "transformer self-attention natural language processing",
        query,
        nqi.intent_entry_id or "",
        enrich,
    )
    power_score = _coherence(
        "Transformer Fault Diagnosis via Vibration Analysis in Substations",
        "power transformer electrical grid insulation fault",
        query,
        nqi.intent_entry_id or "",
        enrich,
    )
    assert nlp_score > power_score + 0.1


def test_visual_debugging_cluster_and_flagship_in_start_here():
    query = "visual debugging"
    nqi = normalize_query_intent(query)
    enrich = list(nqi.enrichment_terms)
    viz_score = _coherence(
        "Software Visualization for Debugging Multi-threaded Programs",
        "debugger execution trace program analysis developer tools",
        query,
        nqi.intent_entry_id or "",
        enrich,
    )
    medical_score = _coherence(
        "Histopathology Image Segmentation for Clinical Diagnosis",
        "medical imaging MRI clinical",
        query,
        nqi.intent_entry_id or "",
        enrich,
    )
    assert viz_score > medical_score + 0.08

    flagship = _paper(
        "arxiv:debug1",
        "Software Visualization for Debugging Multi-threaded Programs",
        seed_origin="api_search",
        relevance_diverse_norm=0.95,
        canonicality_score=0.72,
        intent_coherence_score=viz_score,
        semantic_score=0.62,
        is_foundational_hub=True,
        foundational_eligible=True,
    )
    bridge = _paper(
        "bridge1",
        "Histopathology Image Segmentation for Clinical Diagnosis",
        is_exploratory_bridge=True,
        intent_subfield_penalty=0.5,
        intent_coherence_score=medical_score,
        relevance_diverse_norm=0.7,
    )
    noise = _paper(
        "reco1",
        "Sequential Recommendation with Preference Learning",
        intent_subfield_penalty=0.45,
        intent_coherence_score=0.25,
        coherence_suppressed=True,
    )
    qp = {
        "query_text": query,
        "query_terms": ["visual", "debugging"],
        "intent_mode_v2": "technical_method",
        **nqi.to_profile_dict(),
    }
    payload = {
        "papers": [flagship, bridge, noise],
        "query_profile": qp,
        "discovery": {"recommendations": [{"paper_id": "arxiv:debug1", "score": 1.0, "reason": "r"}]},
    }
    ux = build_lite_ux_payload(payload)
    assert "start_here" not in ux
    found_ids = [e["paper_id"] for e in ux.get("foundational_papers") or []]
    assert "arxiv:debug1" in found_ids
    assert "bridge1" not in found_ids
    assert "reco1" not in found_ids


def test_exploratory_bridge_explanation_not_overconfident():
    p = _paper(
        "b",
        "Generic Citation Hub Paper",
        is_exploratory_bridge=True,
        explanation_confidence_tier="exploratory_connection",
        semantic_score=0.2,
        intent_coherence_score=0.28,
    )
    text = build_why_it_matters(p, query_text="transformer", intent_label="technical_method")
    assert "Core alignment with your query intent" not in text
    assert (
        "Exploratory" in text
        or "exploratory" in text
        or "Cross-domain" in text
        or "off-topic" in text.lower()
    )


def test_coherence_layer_adjusts_ranking():
    nodes = [
        _paper(
            "good",
            "Direct Preference Optimization for LLM Alignment",
            seed_origin="api_search",
            relevance_raw=2.0,
            semantic_score=0.7,
        ),
        _paper(
            "bad",
            "Sequential Recommendation Systems",
            relevance_raw=2.0,
            semantic_score=0.35,
        ),
    ]
    qp = {
        "query_text": "direct preference optimization",
        "intent_entry_id": "direct_preference_optimization",
        "intent_enrichment_terms": ["RLHF", "language model", "DPO"],
        "query_ambiguity_score": 0.72,
    }
    opts = {"bridge_downrank_factor": 0.4, "coherence_filter_enabled": True}
    SRGApplicationService._apply_intent_coherence_layer(
        nodes,
        query_profile=qp,
        query_text=qp["query_text"],
        query_domains=frozenset({"cs"}),
        opts=opts,
    )
    by_id = {n["id"]: n for n in nodes}
    assert float(by_id["good"]["relevance_raw"]) > float(by_id["bad"]["relevance_raw"])


def _attach_topical_scores(papers: list[dict], qp: dict) -> None:
    apply_topical_ranking_layer(papers, qp)


def test_flagship_registry_resolves_dpo_arxiv():
    papers = [
        _paper("arxiv:2305.18290", "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"),
    ]
    ids = resolve_flagship_paper_ids(papers, "direct_preference_optimization")
    assert ids == ["arxiv:2305.18290"]


def test_pin_flagship_forces_dpo_first():
    query = "direct preference optimization"
    nqi = normalize_query_intent(query)
    qp = {"query_text": query, **nqi.to_profile_dict()}
    dpo = _paper("arxiv:2305.18290", "Direct Preference Optimization: Your Language Model is Secretly a Reward Model")
    other = _paper("x", "Some Other Paper")
    picked = pin_flagship_into_start_here(["x"], [dpo, other], qp, max_items=5)
    assert picked[0] == "arxiv:2305.18290"


def test_dpo_cybersecurity_survey_below_flagship_start_here():
    """Regression: broad LLM cybersecurity survey must not beat DPO flagship."""
    query = "direct preference optimization"
    nqi = normalize_query_intent(query)
    qp = {
        "query_text": query,
        "query_terms": ["direct", "preference", "optimization"],
        "intent_mode_v2": "technical_method",
        **nqi.to_profile_dict(),
    }
    dpo = _paper(
        "arxiv:2305.18290",
        "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        abstract="RLHF alignment language model preference optimization DPO",
        seed_origin="api_search",
        relevance_diverse_norm=0.88,
        relevance_raw=2.2,
        semantic_score=0.78,
        citation_count=120,
        graph_score=0.55,
    )
    cyber_survey = _paper(
        "cyber1",
        "Generative AI in cybersecurity: A comprehensive review of LLM applications and vulnerabilities",
        abstract="large language models generative ai cybersecurity survey applications vulnerabilities",
        relevance_diverse_norm=0.96,
        relevance_raw=3.5,
        semantic_score=0.52,
        citation_count=200,
        graph_score=0.92,
        relation_expand_score=0.35,
    )
    reco = _paper(
        "reco1",
        "Session-based Recommendation with Sequential Preference Learning",
        abstract="collaborative filtering recommender system",
        relevance_diverse_norm=0.7,
        relevance_raw=1.8,
        semantic_score=0.4,
    )
    for p in (dpo, cyber_survey, reco):
        bundle = compute_intent_coherence(
            p,
            query_text=query,
            enrichment_terms=qp["intent_enrichment_terms"],
            intent_entry_id=qp["intent_entry_id"],
            query_domains=frozenset({"cs"}),
            semantic_score=float(p.get("semantic_score", 0.5)),
        )
        p.update(bundle)
    _attach_topical_scores([dpo, cyber_survey, reco], qp)

    assert strict_start_here_score(dpo, qp) > strict_start_here_score(cyber_survey, qp)
    assert strict_start_here_score(dpo, qp) > strict_start_here_score(reco, qp)
    cyber_mult, cyber_is_gen = generic_paper_penalty(
        cyber_survey, query, qp["intent_enrichment_terms"], qp["intent_entry_id"]
    )
    assert cyber_mult < 0.75
    assert cyber_is_gen is True

    payload = {
        "papers": [dpo, cyber_survey, reco],
        "query_profile": qp,
        "discovery": {
            "recommendations": [
                {"paper_id": "cyber1", "score": 3.5, "reason": "hub"},
                {"paper_id": "arxiv:2305.18290", "score": 2.0, "reason": "seed"},
            ]
        },
    }
    ux = build_lite_ux_payload(payload)
    assert "start_here" not in ux
    found_ids = [e["paper_id"] for e in ux.get("foundational_papers") or []]
    assert found_ids[0] == "arxiv:2305.18290"
    assert "cyber1" not in found_ids


def test_dpo_query_centrality_beats_generic_hub():
    query = "direct preference optimization"
    nqi = normalize_query_intent(query)
    dpo_cent = compute_query_centrality(
        query,
        _paper("d", "Direct Preference Optimization for Language Model Alignment", abstract="DPO RLHF"),
        enrichment_terms=nqi.enrichment_terms,
        intent_entry_id=nqi.intent_entry_id,
    )
    generic_cent = compute_query_centrality(
        query,
        _paper(
            "g",
            "A Comprehensive Review of Large Language Models in Modern AI",
            abstract="survey overview benchmark dataset foundation model",
        ),
        enrichment_terms=nqi.enrichment_terms,
        intent_entry_id=nqi.intent_entry_id,
    )
    assert dpo_cent > generic_cent + 0.15


def test_transformer_strict_start_here_prefers_vaswani():
    query = "transformer"
    nqi = normalize_query_intent(query)
    qp = {"query_text": query, **nqi.to_profile_dict()}
    vaswani = _paper(
        "arxiv:1706.03762",
        "Attention Is All You Need",
        abstract="transformer self-attention NLP sequence",
        seed_origin="api_search",
        semantic_score=0.7,
    )
    power = _paper(
        "pwr",
        "Transformer Fault Diagnosis in Electrical Substations",
        abstract="power transformer fault electrical grid",
        semantic_score=0.35,
        citation_count=150,
        graph_score=0.85,
    )
    _attach_topical_scores([vaswani, power], qp)
    assert strict_start_here_score(vaswani, qp) > strict_start_here_score(power, qp)


def test_gnn_message_passing_centrality():
    query = "graph neural networks"
    nqi = normalize_query_intent(query)
    gcn = compute_query_centrality(
        query,
        _paper("g", "Semi-Supervised Classification with Graph Convolutional Networks", abstract="GCN message passing"),
        enrichment_terms=nqi.enrichment_terms,
        intent_entry_id=nqi.intent_entry_id,
    )
    social = compute_query_centrality(
        query,
        _paper("s", "Social Network Analysis: A Comprehensive Survey", abstract="social network survey"),
        enrichment_terms=nqi.enrichment_terms,
        intent_entry_id=nqi.intent_entry_id,
    )
    assert gcn > social + 0.1


def test_flagship_explanation_label():
    p = _paper(
        "dpo",
        "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        flagship_match_strength=0.95,
        query_centrality=0.82,
        topical_importance=0.88,
    )
    qp = {"query_text": "direct preference optimization", "intent_entry_id": "direct_preference_optimization"}
    text = build_why_it_matters(p, query_text="direct preference optimization", intent_label="technical_method")
    assert "Canonical anchor" in text or "Direct Preference Optimization" in text
    assert "Core alignment with your query intent" not in text
    assert "Published in the last three years" not in text


def test_visual_debugging_suppresses_program_slicing_survey():
    query = "visual debugging"
    nqi = normalize_query_intent(query)
    qp = {"query_text": query, **nqi.to_profile_dict()}
    viz = _paper(
        "v1",
        "Software Visualization for Debugging Concurrent Programs",
        abstract="debugger execution trace developer tools IDE",
        seed_origin="api_search",
        semantic_score=0.65,
    )
    slicing = _paper(
        "sl1",
        "A Survey of Program Slicing Techniques",
        abstract="static analysis program slicing formal methods",
        semantic_score=0.4,
        graph_score=0.8,
        citation_count=180,
    )
    _attach_topical_scores([viz, slicing], qp)
    assert strict_start_here_score(viz, qp) > strict_start_here_score(slicing, qp)


def test_generic_survey_explanation():
    p = _paper(
        "s",
        "Generative AI: A Comprehensive Review of LLM Applications",
        is_generic_survey=True,
        topical_importance=0.25,
        query_centrality=0.18,
    )
    text = build_why_it_matters(p, query_text="direct preference optimization", intent_label="technical_method")
    assert "Broad survey" in text or "Peripheral" in text or "Exploratory" in text


def test_branch_suppression_drops_contaminated_cluster():
    from srg.branch_coherence import analyze_branch_purity, filter_and_curate_branches

    query = "direct preference optimization"
    nqi = normalize_query_intent(query)
    qp = {"query_text": query, **nqi.to_profile_dict()}
    dpo_members = [
        _paper(
            "d1",
            "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
            abstract="preference optimization RLHF alignment",
            topical_importance=0.85,
            semantic_score=0.8,
        ),
        _paper(
            "d2",
            "Iterative Preference Learning from Human Feedback",
            abstract="DPO RLHF human feedback language model",
            topical_importance=0.78,
            semantic_score=0.75,
        ),
    ]
    cyber_members = [
        _paper(
            "c1",
            "Cybersecurity Threat Detection with Deep Learning",
            abstract="malware intrusion detection vulnerability cyber security",
            topical_importance=0.12,
            semantic_score=0.15,
        ),
        _paper(
            "c2",
            "A Survey of Software Vulnerability Analysis",
            abstract="CVE commit message security issue vulnerability",
            topical_importance=0.10,
            semantic_score=0.12,
        ),
        _paper(
            "c3",
            "Intrusion Detection Systems: A Comprehensive Review",
            abstract="network intrusion malware cybersecurity",
            topical_importance=0.08,
            semantic_score=0.10,
        ),
    ]
    _attach_topical_scores(dpo_members + cyber_members, qp)
    good_report = analyze_branch_purity(dpo_members, qp)
    bad_report = analyze_branch_purity(cyber_members, qp)
    assert good_report.level in ("high", "medium")
    assert bad_report.level == "low"

    branches = [
        {"id": "b-good", "label": "DPO", "paper_ids": [p["id"] for p in dpo_members], "member_count": 2},
        {"id": "b-bad", "label": "Mixed", "paper_ids": [p["id"] for p in cyber_members], "member_count": 3},
    ]
    curated = filter_and_curate_branches(branches, dpo_members + cyber_members, qp)
    assert len(curated) == 1
    assert curated[0]["id"] == "b-good"
    assert "DPO" in curated[0]["label"] or "Alignment" in curated[0]["label"] or "Preference" in curated[0]["label"]


def test_rlhf_excluded_from_dpo_start_here():
    from srg.query_intent_gating import classify_paper_intent_category, start_here_allowed

    query = "direct preference optimization"
    nqi = normalize_query_intent(query)
    qp = {"query_text": query, **nqi.to_profile_dict()}
    rlhf = _paper(
        "rlhf1",
        "Training language models to follow instructions with human feedback",
        abstract="RLHF reinforcement learning human feedback reward model alignment",
        relevance_diverse_norm=0.95,
        relevance_raw=3.0,
        semantic_score=0.72,
        citation_count=5000,
        graph_score=0.9,
    )
    dpo = _paper(
        "arxiv:2305.18290",
        "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        abstract="DPO preference optimization alignment",
        relevance_diverse_norm=0.85,
        semantic_score=0.78,
    )
    _attach_topical_scores([rlhf, dpo], qp)
    from srg.query_intent_gating import attach_paper_role, start_here_role_allowed

    assert classify_paper_intent_category(rlhf, qp) == "adjacent_alignment"
    assert start_here_role_allowed(attach_paper_role(rlhf, qp))
    assert strict_start_here_score(rlhf, qp) >= 0 or attach_paper_role(rlhf, qp) == "alignment_baseline"
    assert attach_paper_role(dpo, qp) == "canonical_anchor"

    payload = {
        "papers": [rlhf, dpo],
        "query_profile": qp,
        "discovery": {"recommendations": [{"paper_id": "rlhf1", "score": 5.0, "reason": "hub"}]},
    }
    ux = build_lite_ux_payload(payload)
    assert "start_here" not in ux
    found_ids = [e["paper_id"] for e in ux.get("foundational_papers") or []]
    assert "arxiv:2305.18290" in found_ids
    assert "rlhf1" not in found_ids


def test_cross_domain_dpo_excluded_from_foundational():
    from srg.query_intent_gating import (
        ROLE_CROSS_DOMAIN,
        attach_paper_role,
        build_foundational_paper_entries,
    )

    query = "direct preference optimization"
    nqi = normalize_query_intent(query)
    qp = {"query_text": query, **nqi.to_profile_dict()}
    diffusion = _paper(
        "diff1",
        "Diffusion Model Alignment Using Direct Preference Optimization",
        abstract="denoising diffusion text-to-image generative model",
    )
    protein = _paper(
        "prot1",
        "Aligning protein generative models with experimental fitness via Direct Preference Optimization",
        abstract="protein folding experimental fitness",
    )
    dpo = _paper(
        "arxiv:2305.18290",
        "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        abstract="language model reward model alignment",
    )
    assert attach_paper_role(diffusion, qp) == ROLE_CROSS_DOMAIN
    assert attach_paper_role(protein, qp) == ROLE_CROSS_DOMAIN
    found = build_foundational_paper_entries([diffusion, protein, dpo], qp)
    found_ids = [e["paper_id"] for e in found]
    assert "arxiv:2305.18290" in found_ids
    assert "diff1" not in found_ids
    assert "prot1" not in found_ids
    assert len({e["one_line"] for e in found}) == len(found)


def test_dpo_start_here_blocks_philosophical_paper():
    from srg.query_intent_gating import dpo_start_here_strict_eligible

    query = "direct preference optimization"
    nqi = normalize_query_intent(query)
    qp = {"query_text": query, **nqi.to_profile_dict()}
    cats = _paper(
        "cats",
        "CAT'S THEORY: Empirical Validation and Architectural Applications Cross-Architecture AI Consciousness",
        abstract="consciousness recursive intelligence constraint-preserving",
        semantic_score=0.7,
        anchor_graph_hops=1,
    )
    dpo = _paper(
        "arxiv:2305.18290",
        "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        abstract="language model alignment DPO",
    )
    assert not dpo_start_here_strict_eligible(cats, [cats, dpo], qp)
    payload = {
        "papers": [cats, dpo],
        "query_profile": qp,
        "discovery": {"recommendations": [{"paper_id": "cats", "score": 9.0, "reason": "x"}]},
    }
    ux = build_lite_ux_payload(payload)
    assert "start_here" not in ux


def test_dpo_start_here_rlhf_requires_lineage_and_evidence():
    from srg.query_intent_gating import dpo_start_here_strict_eligible

    query = "direct preference optimization"
    nqi = normalize_query_intent(query)
    qp = {"query_text": query, **nqi.to_profile_dict()}
    dpo = _paper(
        "arxiv:2305.18290",
        "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        abstract="DPO alignment",
        references=["openalex:RLHF1"],
    )
    rlhf_std = _paper(
        "openalex:RLHF1",
        "Training language models to follow instructions with human feedback",
        abstract="RLHF PPO reward model",
        semantic_score=0.68,
        anchor_graph_hops=2,
    )
    rlhf_generic = _paper(
        "hf1",
        "A Generic Human Feedback System for Interactive Agents",
        abstract="human feedback interactive agents",
        semantic_score=0.65,
        anchor_graph_hops=1,
    )
    assert dpo_start_here_strict_eligible(rlhf_std, [dpo, rlhf_std, rlhf_generic], qp)
    assert not dpo_start_here_strict_eligible(rlhf_generic, [dpo, rlhf_generic], qp)


def test_foundational_papers_have_role_aware_labels():
    query = "direct preference optimization"
    nqi = normalize_query_intent(query)
    qp = {"query_text": query, **nqi.to_profile_dict()}
    dpo = _paper(
        "arxiv:2305.18290",
        "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        abstract="DPO language model",
    )
    cal = _paper("cal", "Cal-DPO: Calibrated Direct Preference Optimization for Language Model Alignment")
    orpo = _paper("orpo", "ORPO: Monolithic Preference Optimization without Reference Model")
    rlhf = _paper(
        "rlhf",
        "Training language models to follow instructions with human feedback",
        abstract="RLHF human feedback",
    )
    _attach_topical_scores([dpo, cal, orpo, rlhf], qp)
    ux = build_lite_ux_payload(
        {"papers": [dpo, cal, orpo, rlhf], "query_profile": qp, "discovery": {"recommendations": []}}
    )
    assert "start_here" not in ux
    fids = {e["paper_id"] for e in ux["foundational_papers"]}
    assert "arxiv:2305.18290" in fids
    for e in ux["foundational_papers"]:
        assert "Original DPO paper" not in e.get("one_line", "")
        assert "Canonical anchor" in e.get("one_line", "") or "Method extension" in e.get("one_line", "")
