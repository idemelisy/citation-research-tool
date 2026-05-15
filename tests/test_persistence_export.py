from __future__ import annotations

from datetime import datetime, timezone

from srg.persistence import (
    ProjectStore,
    feedback_events_from_jsonable,
    feedback_events_to_jsonable,
)
from srg.schema import FeedbackEvent
from srg.synthesis_export import build_bibtex, build_markdown_report


def test_feedback_json_roundtrip() -> None:
    original = [
        FeedbackEvent(
            event_type="merge_hint",
            left_id="a:1",
            right_id="b:2",
            accepted=True,
            created_at=datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        )
    ]
    rows = feedback_events_to_jsonable(original)
    restored = feedback_events_from_jsonable(rows)
    assert len(restored) == 1
    assert restored[0].event_type == "merge_hint"
    assert restored[0].left_id == "a:1"
    assert restored[0].right_id == "b:2"
    assert restored[0].accepted is True


def test_project_store_save_load_roundtrip(tmp_path) -> None:
    store = ProjectStore(tmp_path)
    session = {
        "payload": {"papers": [], "graph": {"nodes": [], "edges": []}, "discovery": {"recommendations": [], "trends": {}}},
        "pdf_previews": [],
        "pdf_ui": {},
        "pdf_blobs": {},
        "feedback_events": [],
        "deep_parse": False,
        "graph_mode": "local",
        "local_seed_count": 0,
        "selected_node": None,
        "discipline_choice_idx": 0,
    }
    path = store.save("Test Project", session)
    assert path.is_file()
    stem = path.stem
    loaded = store.load_stem(stem)
    assert loaded["graph_mode"] == "local"


def test_markdown_includes_potential_connections_for_dashed_edges() -> None:
    payload = {
        "generated_at": "2024-01-01T00:00:00",
        "papers": [
            {"id": "a", "title": "Paper A", "seed_origin": "uploaded_pdf", "source_label": "PDF"},
            {"id": "b", "title": "Paper B", "source_label": "API"},
        ],
        "graph": {
            "nodes": [],
            "edges": [
                {
                    "source": "a",
                    "target": "b",
                    "confidence": "low",
                    "context": None,
                    "context_source": "unresolved_context",
                    "style": {"stroke": "dashed"},
                },
                {
                    "source": "a",
                    "target": "b",
                    "confidence": "high",
                    "context": "Seen in abstract",
                    "context_source": "abstract_proxy",
                    "style": {"stroke": "solid"},
                },
            ],
        },
        "discovery": {
            "recommendations": [{"paper_id": "b", "score": 1.0, "reason": "test"}],
            "trends": {"timeline": {"2024": 2}, "domain_distribution": {"cs": 2}},
        },
        "missing_link_candidates": [],
        "quality_report": {},
        "source_summary": {},
    }
    md = build_markdown_report(payload)
    assert "Potential connections" in md
    assert "Foundational papers" in md or "Literature branches" in md or "Missing link" in md
    assert "### Start here" not in md


def test_bibtex_contains_entries() -> None:
    payload = {
        "papers": [
            {
                "id": "openalex:W1",
                "title": "Hello {World}",
                "year": 2020,
                "authors": [{"name": "Doe, Jane"}],
                "doi": "10.1000/182",
                "arxiv_id": "",
                "venue": "Journal",
            }
        ]
    }
    bib = build_bibtex(payload)
    assert "@misc{" in bib
    assert "Hello" in bib
