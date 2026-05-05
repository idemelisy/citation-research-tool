from __future__ import annotations

from unittest.mock import MagicMock, patch

from srg.ingestion import ArXivClient, arxiv_search_query_from_user_text


def test_arxiv_search_query_wraps_plain_text() -> None:
    assert arxiv_search_query_from_user_text("BERT") == "all:BERT"
    assert arxiv_search_query_from_user_text('  one  two\n') == "all:one two"
    assert arxiv_search_query_from_user_text("ti:large language model") == "ti:large language model"
    assert arxiv_search_query_from_user_text("AU:smith") == "AU:smith"


def test_parse_arxiv_feed_entries() -> None:
    xml = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1234.5678v1</id>
    <title> Alpha </title>
    <summary>Abstract one</summary>
    <published>2020-01-01T00:00:00Z</published>
    <author><name>Ada</name></author>
    <arxiv:doi>10.48550/arXiv.1234.5678</arxiv:doi>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/1234.9999v2</id>
    <title>Beta</title>
  </entry>
</feed>"""
    rows = ArXivClient._parse_atom_feed_entries(xml)
    assert len(rows) == 2
    assert rows[0]["id"] == "1234.5678"
    assert rows[0]["title"] == "Alpha"
    assert rows[0]["summary"] == "Abstract one"
    assert rows[0]["authors"] == ["Ada"]
    assert rows[1]["id"] == "1234.9999"
    assert rows[1]["title"] == "Beta"


@patch("srg.ingestion.requests.get")
def test_arxiv_client_search_dedupes_versions(mock_get: MagicMock) -> None:
    xml = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1234.5678v1</id>
    <title>A</title>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/1234.5678v2</id>
    <title>A revised</title>
  </entry>
</feed>"""
    resp = MagicMock()
    resp.text = xml
    resp.raise_for_status = MagicMock()
    mock_get.return_value = resp
    hits = ArXivClient().search("all:test", max_results=10)
    assert len(hits) == 1
    assert hits[0]["id"] == "1234.5678"
    mock_get.assert_called_once()
    params = mock_get.call_args.kwargs.get("params") or mock_get.call_args[1]["params"]
    assert params["search_query"] == "all:test"
    assert params["max_results"] == 10
