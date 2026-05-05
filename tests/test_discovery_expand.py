from __future__ import annotations

from srg.discovery import RelationParams, robust_pair_term, score_expand_relations
from srg.graph import CitationGraphBuilder
from srg.schema import Author, PaperRecord, Provenance


def _p(cid: str, title: str, refs: list[str]) -> PaperRecord:
    return PaperRecord(
        canonical_id=cid,
        title=title,
        abstract=None,
        year=2020,
        venue="t",
        domain="cs",
        authors=[Author(name="A")],
        references=refs,
        provenance=[Provenance(source="crossref", source_id=cid)],
    )


def test_robust_pair_term_hub_dampens() -> None:
    assert robust_pair_term(10.0, 100, 100) < robust_pair_term(10.0, 5, 5)


def test_score_expand_relations_coupling_and_explanation() -> None:
    upload = _p("upload", "My PDF Paper", refs=["hub", "r2"])
    hub = _p("hub", "Hub Survey", refs=[f"t{i}" for i in range(30)])
    r2 = _p("r2", "Ref Two", refs=[])
    cand = _p("cand", "Candidate Match", refs=["hub", "r2"])
    graph = CitationGraphBuilder().build([upload, hub, r2, cand])
    out = score_expand_relations(graph, {"upload"}, RelationParams())
    assert "cand" in out
    score, expl, diag = out["cand"]
    assert score > 0
    assert "referans" in expl.lower() or "atıf" in expl.lower() or "genişletme" in expl.lower()
    assert diag.get("relation_max_coupling", 0) >= 1


def test_relation_params_weights() -> None:
    p = RelationParams()
    assert p.w_direct == 2.0 and p.w_coupling == 0.6 and p.w_cocitation == 0.8 and p.w_topic == 0.0
