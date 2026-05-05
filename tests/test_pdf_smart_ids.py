from srg.ingestion import openalex_search_query_from_pdf_title, smart_id_hunt
from srg.validation import author_crosscheck_adjust_score, author_intersection_count, finalize_pdf_mapping_score


def test_smart_id_hunt_finds_doi_and_arxiv() -> None:
    text = "Preprint see arxiv:1706.03762 and doi:10.1038/nphys1170\nmore"
    d, a = smart_id_hunt(text)
    assert d == "10.1038/nphys1170"
    assert a == "1706.03762"


def test_smart_id_hunt_prefers_doi_in_tail_over_header() -> None:
    filler = "x" * 1500
    text = "Header noise doi:10.1111/wrong.journal.12345 " + filler + " footnote doi:10.1038/nphys1170"
    d, _ = smart_id_hunt(text)
    assert d == "10.1038/nphys1170"


def test_author_intersection() -> None:
    pdf = "Alice Smith, Bob Jones"
    api = ["Alice Smith", "Carol Lee"]
    assert author_intersection_count(pdf, api) >= 1


def test_author_crosscheck_boosts_low_title_score() -> None:
    score, note = author_crosscheck_adjust_score(
        0.5,
        "Alice Smith, Bob Jones",
        ["Alice Smith", "Bob Jones", "Other"],
    )
    assert score >= 0.82
    assert note is not None


def test_finalize_pdf_mapping_score_wrong_paper_when_no_author_overlap() -> None:
    score, note = finalize_pdf_mapping_score(
        0.95,
        "Gabriel Loiseau, Marie Dupont",
        ["John Smith", "Jane Doe"],
    )
    assert score == 0.1
    assert note == "Wrong Paper Detected"


def test_openalex_search_query_strips_proceedings_boilerplate() -> None:
    raw = "Proceedings of the 42nd International Conference on Foo — My Real Paper Title, Pages 1–12"
    q = openalex_search_query_from_pdf_title(raw)
    assert q is not None
    assert "proceedings" not in q.lower()
    assert "My Real Paper Title" in q
