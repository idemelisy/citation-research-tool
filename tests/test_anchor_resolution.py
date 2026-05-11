"""Anchor resolution: canonical seeds, OpenAlex title gating helpers."""

from srg.api_adapter import SRGApplicationService, _titles_match_score
from srg.app_ui_lite import _resolve_seed_title


def test_canonical_gnn_seeds_present() -> None:
    seeds = SRGApplicationService._canonical_anchor_seeds_for_query("graph neural networks survey")
    aids = {s["id"] for s in seeds if s.get("provider") == "arxiv"}
    assert "1609.02907" in aids
    assert "1706.02216" in aids
    assert "1710.10903" in aids
    assert "1810.00826" in aids


def test_canonical_gnn_seeds_with_zero_width_chars() -> None:
    q = "graph\u200b neural networks"
    seeds = SRGApplicationService._canonical_anchor_seeds_for_query(q)
    aids = {s["id"] for s in seeds if s.get("provider") == "arxiv"}
    assert "1609.02907" in aids


def test_titles_match_score_basic() -> None:
    assert _titles_match_score("attention is all you need", "attention is all you need") >= 0.99
    assert _titles_match_score("a", "b") < 0.5


def test_resolve_arxiv_seed_title_via_doi_on_openalex_node() -> None:
    """Graph node may be OpenAlex-primary while seed stays arxiv:id — match via arXiv DOI."""
    pmap = {
        "openalex:W123": {
            "id": "openalex:W123",
            "title": "Attention Is All You Need",
            "arxiv_id": "",
            "doi": "10.48550/arXiv.1706.03762",
            "openalex_id": "W123",
        }
    }
    t = _resolve_seed_title({"provider": "arxiv", "id": "1706.03762", "origin": "api_search"}, pmap)
    assert "attention" in t.lower()


def test_reject_industrial_title_when_canonical_context() -> None:
    q = "transformer attention mechanism"
    title = "A novel Transformer based on self-attention for fault diagnosis of rolling bearings"
    bad, reason = SRGApplicationService._reject_openalex_seed_given_canonicals(
        q, title, has_canonical_anchors=True
    )
    assert bad is True
    assert "industrial" in reason or "applied" in reason


def test_title_query_fit_ratio_orders_related_above_unrelated() -> None:
    q = "graph neural networks"
    r_related = SRGApplicationService._openalex_title_query_fit_ratio(q, "The Graph Neural Network Model")
    r_unrelated = SRGApplicationService._openalex_title_query_fit_ratio(
        q, "Lattice QCD thermodynamics and holographic transport coefficients"
    )
    assert r_related > r_unrelated


def test_no_reject_without_canonical_flag() -> None:
    bad, _ = SRGApplicationService._reject_openalex_seed_given_canonicals(
        "transformer attention mechanism",
        "xyzabc unrelated completely different field",
        has_canonical_anchors=False,
    )
    assert bad is False
