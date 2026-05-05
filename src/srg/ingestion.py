from __future__ import annotations

import difflib
import io
import json
import queue
import re
import sqlite3
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Iterable, Protocol

import requests
try:
    from pypdf import PdfReader
except ModuleNotFoundError:  # pragma: no cover - environment-dependent dependency
    PdfReader = None  # type: ignore[assignment]

from .domain_mapper import domain_from_openalex_payload
from .citation_context import ProviderBudget
from .schema import Author, ConfidenceTier, PaperRecord, Provenance
from .validation import finalize_pdf_mapping_score


def _normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    doi = value.strip()
    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    doi = doi.replace("doi:", "")
    doi = doi.strip().lower()
    return doi or None


def openalex_response_first_work(resp: Any) -> dict[str, Any] | None:
    """Normalize OpenAlex JSON: {results: [...]} (possibly empty) or a bare work object with id."""
    if not isinstance(resp, dict):
        return None
    if "results" in resp:
        rows = resp.get("results") or []
        if not rows or not isinstance(rows[0], dict):
            return None
        return rows[0]
    if resp.get("id"):
        return resp
    return None


def _register_openalex_work_index(work: dict[str, Any], by_tail: dict[str, dict[str, Any]]) -> None:
    """Index a work object by every W-tail we can infer (primary id + ids.* URLs)."""
    oid = (work.get("id") or "").rsplit("/", maxsplit=1)[-1]
    if oid and re.match(r"^W\d+$", oid, re.IGNORECASE):
        by_tail[oid.upper()] = work
    ids_obj = work.get("ids") or {}
    for val in ids_obj.values():
        if not isinstance(val, str):
            continue
        tail = val.rsplit("/", maxsplit=1)[-1]
        if tail and re.match(r"^W\d+$", tail, re.IGNORECASE):
            by_tail[tail.upper()] = work


def _batch_map_openalex_results_to_requested_tails(
    chunk: list[str],
    results: list[Any],
) -> dict[str, dict[str, Any]]:
    """Map requested tails (e.g. merged legacy ids) to full work dicts from a /works batch response."""
    reg: dict[str, dict[str, Any]] = {}
    for w in results:
        if isinstance(w, dict):
            _register_openalex_work_index(w, reg)
    out: dict[str, dict[str, Any]] = {}
    for t in chunk:
        tt = (t or "").strip()
        if not tt:
            continue
        hit = reg.get(tt.upper())
        if hit:
            out[tt] = hit
    return out


def openalex_fetch_payload_valid(payload: Any) -> bool:
    """True if payload is a usable OpenAlex work response (never cache/store invalid bodies as success)."""
    if not isinstance(payload, dict):
        return False
    if "results" in payload:
        return openalex_response_first_work(payload) is not None
    return bool(payload.get("id"))


PDF_MAPPING_QUALITY_THRESHOLD = 0.80


def _titles_match_score(left: str | None, right: str | None) -> float:
    if not left or not right:
        return 0.0
    a = re.sub(r"\s+", " ", left.lower().strip())
    b = re.sub(r"\s+", " ", right.lower().strip())
    return difflib.SequenceMatcher(a=a, b=b).ratio()


def _first_non_empty(values: list[str] | None) -> str | None:
    if not values:
        return None
    for item in values:
        val = (item or "").strip()
        if val:
            return val
    return None


def _strip_xml_tags(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"<[^>]+>", " ", value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None


def _reconstruct_openalex_abstract(index: dict[str, list[int]] | None) -> str | None:
    if not index:
        return None
    pos_to_word: dict[int, str] = {}
    for word, positions in index.items():
        for pos in positions:
            pos_to_word[pos] = word
    if not pos_to_word:
        return None
    ordered = [pos_to_word[i] for i in sorted(pos_to_word.keys())]
    return " ".join(ordered).strip() or None


def _extract_acl_anthology_id_from_doi(doi: str | None) -> str | None:
    normalized = _normalize_doi(doi)
    if not normalized:
        return None
    prefix = "10.18653/v1/"
    if not normalized.startswith(prefix):
        return None
    anthology_id = normalized[len(prefix) :].strip()
    return anthology_id.upper() or None


def _extract_bibtex_field(bibtex: str, field: str) -> str | None:
    pattern = rf"{field}\s*=\s*\{{(.*?)\}}"
    match = re.search(pattern, bibtex, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return re.sub(r"\s+", " ", match.group(1)).strip() or None


def _extract_doi_from_text(text: str) -> str | None:
    match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    return _normalize_doi(match.group(0))


def _extract_arxiv_id_from_text(text: str) -> str | None:
    match = re.search(r"\b(?:arXiv:)?(\d{4}\.\d{4,5})(?:v\d+)?\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1)


def normalize_arxiv_list_id(raw_id: str) -> str:
    """Strip version suffix from an arXiv id for stable seeds (e.g. 2301.00001v2 -> 2301.00001)."""
    aid = (raw_id or "").strip()
    if not aid:
        return aid
    return re.sub(r"v\d+$", "", aid, flags=re.IGNORECASE)


def year_from_arxiv_list_id(aid: str | None) -> int | None:
    """
    Infer calendar year from arXiv id when OpenAlex has no publication_year (YY from 2401.xxxx -> 2024).
    """
    if not aid:
        return None
    s = normalize_arxiv_list_id(aid.strip())
    if re.match(r"^\d{4}\.\d{4,5}", s):
        yy = int(s[:2])
        return 2000 + yy if yy < 91 else 1900 + yy
    if "/" in s:
        rest = s.split("/", 1)[1]
        if len(rest) >= 2 and rest[:2].isdigit():
            yy = int(rest[:2])
            return 2000 + yy if yy < 91 else 1900 + yy
    return None


def _arxiv_id_from_openalex_payload(payload: dict[str, Any]) -> str | None:
    ids_obj = payload.get("ids") or {}
    av = ids_obj.get("arxiv")
    if isinstance(av, str) and av.strip():
        tail = av.rsplit("/", maxsplit=1)[-1]
        if tail:
            return normalize_arxiv_list_id(tail)
    doi = (payload.get("doi") or "").strip().lower()
    if "arxiv" in doi and "10.48550/" in doi:
        m = re.search(r"arxiv\.(\d{4}\.\d{4,5})", doi, flags=re.IGNORECASE)
        if m:
            return normalize_arxiv_list_id(m.group(1))
    return None


_ARXIV_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def arxiv_search_query_from_user_text(text: str) -> str:
    """
    Build an export.arxiv.org `search_query` string from free text.

    If the user already passes arXiv query syntax (ti:, au:, all:, cat:, etc.), it is used as-is.
    Otherwise the text is wrapped as `all:...` (title, abstract, comments, journal ref, etc.).
    """
    t = (text or "").strip()
    if not t:
        return t
    if re.match(r"^(all|ti|au|abs|co|id|cat)\s*:\s*", t, flags=re.IGNORECASE):
        return t
    safe = re.sub(r"\s+", " ", t.replace('"', " ").replace("\n", " ")).strip()
    return f"all:{safe}" if safe else safe


def _first_doi_in_string(s: str) -> str | None:
    for m in re.finditer(r"\b10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+\b", s, flags=re.IGNORECASE):
        nd = _normalize_doi(m.group(0))
        if nd:
            return nd
    return None


def _first_arxiv_in_string(s: str) -> str | None:
    for m in re.finditer(
        r"(?:arxiv\s*:\s*)?(\d{4}\.\d{4,5})(?:v\d+)?",
        s,
        flags=re.IGNORECASE,
    ):
        return re.sub(r"v\d+$", "", m.group(1), flags=re.IGNORECASE)
    return None


def smart_id_hunt(text_first_pages: str) -> tuple[str | None, str | None]:
    """
    Regex scan of the first pages for DOI and arXiv.
    Footnote tail (last 1000 chars) is scanned first so end-of-page DOIs/arXiv ids win over header noise.
    """
    if not text_first_pages:
        return None, None
    tail = text_first_pages[-1000:] if len(text_first_pages) > 1000 else text_first_pages
    doi = _first_doi_in_string(tail) or _first_doi_in_string(text_first_pages)
    arxiv = _first_arxiv_in_string(tail) or _first_arxiv_in_string(text_first_pages)
    return doi, arxiv


_ACADEMIC_NOISE_SUBSTRINGS = (
    "proceedings of",
    "conference on",
    "workshop on",
    "symposium on",
    "international conference",
    "annual conference",
    "pages ",
    "page ",
    " pp.",
    " pp ",
    "vol.",
    "volume ",
    "issue ",
    "isbn",
    "doi:",
    "arxiv",
    "abstract",
    "keywords",
    "introduction",
    "references",
    "bibliography",
    "ieee ",
    " acm ",
    "copyright",
    "received ",
    "accepted ",
    "figure ",
    "table ",
    "committee",
    "editorial board",
    "program committee",
)


def _is_academic_noise_line(line: str) -> bool:
    low = line.lower().strip()
    if not low:
        return True
    if any(s in low for s in _ACADEMIC_NOISE_SUBSTRINGS):
        return True
    if re.search(r"\bpages?\s+\d", low):
        return True
    if re.match(r"^\s*doi\s*:", low):
        return True
    return False


_RE_PAGES_TRAIL = re.compile(r"\s*[-–,]\s*(pages?|pp\.)\s*\d.*$", re.IGNORECASE)
_RE_VOL_TRAIL = re.compile(r"\s*[,;]\s*(vol\.|volume|issue)\s*.+$", re.IGNORECASE)


def openalex_search_query_from_pdf_title(raw: str | None) -> str | None:
    """
    Strip proceedings/conference boilerplate and trailing 'Pages …' for OpenAlex title search.
    """
    if not raw:
        return None
    q = re.sub(r"\s+", " ", raw.strip())
    q = re.sub(
        r"^\s*(the\s+)?(\d{1,2}(st|nd|rd|th)\s+)?(annual\s+)?(international\s+)?(conference|workshop|symposium)\s+on\s+",
        "",
        q,
        flags=re.IGNORECASE,
    )
    q = re.sub(r"^\s*in\s+proceedings\s+of\s+(the\s+)?", "", q, flags=re.IGNORECASE)
    q = re.sub(r"^\s*proceedings\s+of\s+(the\s+)?", "", q, flags=re.IGNORECASE)
    q = re.sub(r"^\s*proc\.\s+", "", q, flags=re.IGNORECASE)
    q = _RE_PAGES_TRAIL.sub("", q).strip()
    q = _RE_VOL_TRAIL.sub("", q).strip()
    if len(q) < 6:
        return None
    return q


def openalex_authorship_display_names(work: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for item in work.get("authorships", []) or []:
        if not isinstance(item, dict):
            continue
        auth = item.get("author") or {}
        dn = (auth.get("display_name") or "").strip()
        if dn:
            names.append(dn)
    return names


def resolve_arxiv_id_for_lookup(paper: dict[str, Any]) -> str | None:
    """
    Best-effort arXiv id (YYMM.NNNNN) for papers ingested via OpenAlex/Crossref/DOI
    when a linked arXiv DOI or explicit arxiv_id / canonical id is present.
    """
    raw = (paper.get("arxiv_id") or "").strip()
    if raw:
        return normalize_arxiv_list_id(raw)
    pid = (paper.get("id") or "").strip()
    if pid.lower().startswith("arxiv:"):
        return normalize_arxiv_list_id(pid.split(":", 1)[1])
    doi = _normalize_doi(paper.get("doi"))
    if doi and doi.startswith("10.48550/") and "arxiv." in doi:
        tail = doi.split("arxiv.", 1)[-1]
        m = re.match(r"^(\d{4}\.\d{4,5})", tail)
        if m:
            return m.group(1)
    return None


def _guess_title_from_text(text: str) -> str | None:
    """
    Pick a title-like line from the document head, skipping academic boilerplate.
    Uses longest plausible non-noise line in the first block as a proxy for 'main title' (large/central text).
    """
    lines = [re.sub(r"\s+", " ", l.strip()) for l in text.splitlines() if l.strip()]
    candidates: list[str] = []
    for line in lines[:30]:
        if len(line) < 12 or len(line) > 220:
            continue
        if _is_academic_noise_line(line):
            continue
        low = line.lower()
        if "arxiv" in low or "doi" in low or low.startswith("abstract"):
            continue
        if re.match(r"^\d{4}\.\d{4,5}", line):
            continue
        candidates.append(line)
    if not candidates:
        return None
    return max(candidates, key=lambda s: min(len(s), 180))


def _guess_authors_from_text(text: str) -> str | None:
    lines = [re.sub(r"\s+", " ", l.strip()) for l in text.splitlines() if l.strip()]
    for line in lines[:35]:
        low = line.lower()
        if len(line) < 8 or len(line) > 220:
            continue
        if _is_academic_noise_line(line):
            continue
        if any(token in low for token in ("abstract", "introduction", "arxiv", "doi", "references")):
            continue
        if line.count(",") >= 1 or " and " in low:
            if re.search(r"\b(19|20)\d{2}\b", line):
                continue
            return line
    return None


class ProviderClient(Protocol):
    source_name: str

    def fetch_paper(self, seed: str) -> dict[str, Any]:
        ...

    def normalize(self, payload: dict[str, Any]) -> PaperRecord:
        ...


@dataclass(slots=True)
class RetryPolicy:
    max_attempts: int = 3
    backoff_seconds: float = 1.0


class CacheStore:
    def __init__(self, db_path: str = "srg_cache.db") -> None:
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_payloads (
                source TEXT NOT NULL,
                source_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                fetched_unix INTEGER NOT NULL,
                PRIMARY KEY (source, source_id)
            )
            """
        )
        self.conn.commit()

    def put(self, source: str, source_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO raw_payloads VALUES (?, ?, ?, strftime('%s','now'))",
                (source, source_id, json.dumps(payload)),
            )
            self.conn.commit()

    def get(self, source: str, source_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT payload FROM raw_payloads WHERE source = ? AND source_id = ?",
                (source, source_id),
            ).fetchone()
        if not row:
            return None
        return json.loads(row[0])


class BaseHTTPClient:
    source_name = "base"
    base_url = ""

    def __init__(self, timeout_seconds: int = 20) -> None:
        self.timeout_seconds = timeout_seconds

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}{path}", params=params, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()


class OpenAlexClient(BaseHTTPClient):
    source_name = "openalex"
    base_url = "https://api.openalex.org"
    _openalex_batch_chunk = 50

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET with exponential backoff on HTTP 429 (OpenAlex polite pool)."""
        url = f"{self.base_url}{path}"
        max_attempts = 8
        base_wait = 2.0
        last: requests.Response | None = None
        for attempt in range(max_attempts):
            last = requests.get(url, params=params, timeout=self.timeout_seconds)
            if last.status_code == 429:
                wait_s = min(120.0, base_wait * (2**attempt))
                time.sleep(wait_s)
                continue
            last.raise_for_status()
            return last.json()
        if last is not None:
            last.raise_for_status()
        raise RuntimeError("OpenAlex _get: exhausted retries without response")

    def fetch_paper(self, seed: str) -> dict[str, Any]:
        return self._get(f"/works/{seed}")

    def fetch_works_batch_by_openalex_tails(
        self,
        tails: list[str],
        *,
        chunk_size: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Batch-fetch works via OpenAlex OR filter (≤50 ids per request).
        Prefers ``filter=openalex:W1|W2`` so merged ids resolve to canonical works; falls back to
        ``ids.openalex:https://openalex.org/W1|…`` if the API rejects the primary form.
        Returns requested tail id (e.g. W123) -> full work dict.
        """
        size = chunk_size or self._openalex_batch_chunk
        out: dict[str, dict[str, Any]] = {}
        clean = []
        seen: set[str] = set()
        for t in tails:
            u = (t or "").strip()
            if not u or u in seen:
                continue
            if not re.match(r"^W\d+$", u):
                continue
            seen.add(u)
            clean.append(u)
        for i in range(0, len(clean), size):
            chunk = clean[i : i + size]
            per_page = min(200, max(len(chunk), 1))
            pipe_short = "|".join(chunk)
            resp: dict[str, Any]
            try:
                resp = self._get(
                    "/works",
                    params={"filter": f"openalex:{pipe_short}", "per-page": per_page},
                )
            except requests.HTTPError as exc:
                r = exc.response
                if r is not None and r.status_code == 400:
                    pipe = "|".join(f"https://openalex.org/{t}" for t in chunk)
                    resp = self._get(
                        "/works",
                        params={"filter": f"ids.openalex:{pipe}", "per-page": per_page},
                    )
                else:
                    raise
            mapped = _batch_map_openalex_results_to_requested_tails(chunk, resp.get("results") or [])
            out.update(mapped)
        return out

    def fetch_by_doi(self, doi: str) -> dict[str, Any]:
        normalized = _normalize_doi(doi) or doi
        try:
            # Prefer direct DOI resolution first.
            return self._get(f"/works/https://doi.org/{normalized}")
        except requests.RequestException:
            return self._get("/works", params={"filter": f"doi:{normalized}", "per-page": 1})

    def fetch_work_for_arxiv_id(self, arxiv_id: str) -> dict[str, Any] | None:
        """
        Resolve an arXiv listing id to a full OpenAlex work payload (includes referenced_works).
        Tries arXiv DOI alias first, then ids.arxiv filter.
        """
        aid = normalize_arxiv_list_id((arxiv_id or "").strip())
        if not aid:
            return None
        doi = f"10.48550/arXiv.{aid}"
        try:
            resp = self.fetch_by_doi(doi)
            work = openalex_response_first_work(resp)
            if isinstance(work, dict) and work.get("id"):
                return work
        except requests.RequestException:
            pass
        try:
            resp = self._get("/works", params={"filter": f"ids.arxiv:{aid}", "per-page": 1})
            rows = resp.get("results") or []
            return rows[0] if rows and isinstance(rows[0], dict) else None
        except requests.RequestException:
            return None

    @staticmethod
    def referenced_work_tail_ids(work: dict[str, Any]) -> list[str]:
        """OpenAlex referenced_works URL tails (W… / DOI keys) for graph.reference resolution."""
        out: list[str] = []
        for r in work.get("referenced_works") or []:
            if not isinstance(r, str):
                continue
            tail = r.rsplit("/", maxsplit=1)[-1]
            if tail:
                out.append(tail)
        return out

    def search_by_title(self, title: str) -> dict[str, Any]:
        return self._get("/works", params={"search": title, "per-page": 1})

    def normalize(self, payload: dict[str, Any]) -> PaperRecord:
        pid = payload.get("id", "")
        source_id = pid.rsplit("/", maxsplit=1)[-1] if pid else "unknown"
        authors = [
            Author(name=item.get("author", {}).get("display_name", "unknown"))
            for item in payload.get("authorships", [])
        ]
        refs = [r.rsplit("/", maxsplit=1)[-1] for r in payload.get("referenced_works", []) if isinstance(r, str)]
        pub_y = payload.get("publication_year")
        year: int | None
        if isinstance(pub_y, int):
            year = pub_y
        elif isinstance(pub_y, str):
            low = pub_y.strip().lower()
            if low.isdigit():
                year = int(pub_y.strip())
            elif low in ("unknown", "none", ""):
                year = None
            else:
                year = None
        else:
            year = None
        if year is None:
            inferred = year_from_arxiv_list_id(_arxiv_id_from_openalex_payload(payload))
            if inferred is not None:
                year = inferred
        return PaperRecord(
            canonical_id=f"openalex:{source_id}",
            title=payload.get("title", "untitled"),
            abstract=None,
            year=year,
            venue=(payload.get("host_venue") or {}).get("display_name"),
            domain=domain_from_openalex_payload(payload),
            doi=_normalize_doi(payload.get("doi")),
            openalex_id=source_id,
            authors=authors,
            references=refs,
            provenance=[
                Provenance(
                    source=self.source_name,
                    source_id=source_id,
                    field_confidence={"doi": ConfidenceTier.MEDIUM, "title": ConfidenceTier.HIGH},
                    raw=payload,
                )
            ],
        )


class CrossrefClient(BaseHTTPClient):
    source_name = "crossref"
    base_url = "https://api.crossref.org"

    def fetch_paper(self, seed: str) -> dict[str, Any]:
        return self._get(f"/works/{seed}")

    def normalize(self, payload: dict[str, Any]) -> PaperRecord:
        message = payload.get("message", {})
        doi = _normalize_doi(message.get("DOI"))
        title = (
            _first_non_empty(message.get("title"))
            or _first_non_empty(message.get("subtitle"))
            or _first_non_empty(message.get("short-title"))
            or (f"DOI:{doi}" if doi else "untitled")
        )
        abstract = _strip_xml_tags(message.get("abstract"))
        venue = (
            _first_non_empty(message.get("container-title"))
            or _first_non_empty(message.get("short-container-title"))
        )
        authors = [
            Author(name=f"{a.get('given', '').strip()} {a.get('family', '').strip()}".strip(), orcid=a.get("ORCID"))
            for a in message.get("author", [])
        ]
        references = []
        for ref in message.get("reference", []):
            ref_doi = ref.get("DOI")
            if ref_doi:
                references.append(ref_doi.lower())
        return PaperRecord(
            canonical_id=f"doi:{doi or 'unknown'}",
            title=title,
            abstract=abstract,
            year=((message.get("published-print") or {}).get("date-parts") or [[None]])[0][0],
            venue=venue,
            domain=None,
            doi=doi,
            authors=authors,
            references=references,
            provenance=[
                Provenance(
                    source=self.source_name,
                    source_id=doi or "unknown",
                    field_confidence={"doi": ConfidenceTier.HIGH, "title": ConfidenceTier.HIGH},
                    raw=payload,
                )
            ],
        )


class ArXivClient(BaseHTTPClient):
    source_name = "arxiv"
    base_url = "https://export.arxiv.org/api"

    @staticmethod
    def _arxiv_request_headers() -> dict[str, str]:
        return {"User-Agent": "SemanticResearchGraph/0.1 (mailto:local@localhost)"}

    def fetch_paper(self, seed: str) -> dict[str, Any]:
        response = requests.get(
            f"{self.base_url}/query",
            params={"id_list": seed, "max_results": 1},
            timeout=self.timeout_seconds,
            headers=self._arxiv_request_headers(),
        )
        response.raise_for_status()
        return self._parse_atom_entry(response.text, seed)

    def search(self, search_query: str, *, max_results: int = 8) -> list[dict[str, Any]]:
        """
        Full-text / field search via the Atom API (same query language as the arxiv PyPI client).
        Returns one payload dict per entry (same shape as fetch_paper / normalize).
        """
        q = (search_query or "").strip()
        if not q:
            return []
        response = requests.get(
            f"{self.base_url}/query",
            params={
                "search_query": q,
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
            timeout=self.timeout_seconds,
            headers=self._arxiv_request_headers(),
        )
        response.raise_for_status()
        entries = self._parse_atom_feed_entries(response.text)
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for row in entries:
            aid = normalize_arxiv_list_id(str(row.get("id") or ""))
            if not aid or aid in seen:
                continue
            seen.add(aid)
            row["id"] = aid
            unique.append(row)
        return unique

    @classmethod
    def _arxiv_entry_to_dict(cls, entry: ET.Element | None, fallback_id: str) -> dict[str, Any]:
        ns = _ARXIV_ATOM_NS
        if entry is None:
            return {
                "id": fallback_id,
                "title": fallback_id,
                "summary": None,
                "authors": [],
                "doi": None,
                "published": None,
                "updated": None,
            }

        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip() or None
        aid = (entry.findtext("atom:id", default="", namespaces=ns) or fallback_id).rsplit("/", maxsplit=1)[-1]
        aid = normalize_arxiv_list_id(aid) or aid
        doi = entry.findtext("arxiv:doi", default=None, namespaces=ns)
        authors = [
            (author.findtext("atom:name", default="unknown", namespaces=ns) or "unknown").strip()
            for author in entry.findall("atom:author", ns)
        ]
        published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip() or None
        updated = (entry.findtext("atom:updated", default="", namespaces=ns) or "").strip() or None
        return {
            "id": aid,
            "title": title or fallback_id,
            "summary": summary,
            "authors": authors,
            "doi": doi,
            "published": published,
            "updated": updated,
        }

    @classmethod
    def _parse_atom_feed_entries(cls, xml_text: str) -> list[dict[str, Any]]:
        ns = _ARXIV_ATOM_NS
        root = ET.fromstring(xml_text)
        out: list[dict[str, Any]] = []
        for idx, entry in enumerate(root.findall("atom:entry", ns)):
            fb = f"entry-{idx}"
            out.append(cls._arxiv_entry_to_dict(entry, fb))
        return out

    def _parse_atom_entry(self, xml_text: str, fallback_id: str) -> dict[str, Any]:
        ns = _ARXIV_ATOM_NS
        root = ET.fromstring(xml_text)
        entry = root.find("atom:entry", ns)
        return self._arxiv_entry_to_dict(entry, fallback_id)

    def normalize(self, payload: dict[str, Any]) -> PaperRecord:
        aid = payload.get("id", "unknown")
        derived_doi = _normalize_doi(payload.get("doi")) or _normalize_doi(f"10.48550/arXiv.{aid}")
        return PaperRecord(
            canonical_id=f"arxiv:{aid}",
            title=payload.get("title", "untitled"),
            abstract=payload.get("summary"),
            year=None,
            venue="arXiv",
            domain=None,
            doi=derived_doi,
            arxiv_id=aid,
            authors=[Author(name=a) for a in payload.get("authors", [])],
            provenance=[
                Provenance(
                    source=self.source_name,
                    source_id=aid,
                    field_confidence={"arxiv_id": ConfidenceTier.HIGH, "title": ConfidenceTier.MEDIUM},
                    raw=payload,
                )
            ],
        )


def fetch_arxiv_public_metadata(arxiv_id: str, timeout: int = 25) -> dict[str, Any] | None:
    """
    Query export.arxiv.org for abstract, authors, and submission/update dates.
    Returns dict with id, title, summary, authors (list[str]), published, updated, doi or None.
    """
    aid = (arxiv_id or "").strip()
    if not aid:
        return None
    aid = normalize_arxiv_list_id(aid)
    if not re.match(r"^\d{4}\.\d{4,5}$", aid):
        return None
    try:
        client = ArXivClient()
        response = requests.get(
            f"{client.base_url}/query",
            params={"id_list": aid, "max_results": 1},
            timeout=timeout,
            headers=ArXivClient._arxiv_request_headers(),
        )
        response.raise_for_status()
        return client._parse_atom_entry(response.text, aid)
    except (requests.RequestException, ET.ParseError):
        return None


def fetch_public_metadata_for_seed(provider: str, seed: str) -> dict[str, Any] | None:
    """
    Resolve title + author list for a seed (PDF verification and SRGApplicationService).
    Uses provider APIs only — never PDF-derived titles.
    """
    p = provider.strip().lower()
    s = seed.strip()
    if not s:
        return None
    try:
        if p in ("doi", "crossref"):
            doi = _normalize_doi(s) or s
            oa = OpenAlexClient()
            resp = oa.fetch_by_doi(doi)
            work = openalex_response_first_work(resp)
            if isinstance(work, dict) and (work.get("title") or "").strip():
                return {
                    "title": (work.get("title") or "").strip(),
                    "authors": openalex_authorship_display_names(work),
                    "source": "openalex_doi",
                    "openalex_id": (work.get("id") or "").rsplit("/", maxsplit=1)[-1] or None,
                }
            cr = CrossrefClient()
            payload = cr.fetch_paper(doi)
            msg = payload.get("message", {})
            title = _first_non_empty(msg.get("title")) or ""
            if title:
                authors: list[str] = []
                for a in msg.get("author", []) or []:
                    gn = (a.get("given") or "").strip()
                    fn = (a.get("family") or "").strip()
                    nm = f"{gn} {fn}".strip()
                    if nm:
                        authors.append(nm)
                return {"title": title.strip(), "authors": authors, "source": "crossref"}
        if p == "arxiv":
            meta = fetch_arxiv_public_metadata(s)
            if meta and (meta.get("title") or "").strip():
                return {
                    "title": (meta.get("title") or "").strip(),
                    "authors": list(meta.get("authors") or []),
                    "source": "arxiv",
                }
        if p == "openalex":
            oa = OpenAlexClient()
            work: dict[str, Any] | None = None
            try:
                work = oa.fetch_paper(s)
            except requests.HTTPError as exc:
                if exc.response is not None and exc.status_code == 404 and re.match(r"^W\d+$", s, flags=re.IGNORECASE):
                    part = oa.fetch_works_batch_by_openalex_tails([s.upper()], chunk_size=1)
                    work = part.get(s.upper()) or part.get(s)
                else:
                    raise
            except requests.RequestException:
                return None
            if isinstance(work, dict) and (work.get("title") or "").strip():
                return {
                    "title": (work.get("title") or "").strip(),
                    "authors": openalex_authorship_display_names(work),
                    "source": "openalex",
                    "openalex_id": s,
                }
    except requests.RequestException:
        return None
    return None


class SemanticScholarClient(BaseHTTPClient):
    source_name = "semantic_scholar"
    base_url = "https://api.semanticscholar.org/graph/v1"

    def fetch_paper(self, seed: str) -> dict[str, Any]:
        fields = "title,abstract,year,venue,authors,references,citations,externalIds"
        return self._get(f"/paper/{seed}", params={"fields": fields})

    def fetch_by_doi(self, doi: str) -> dict[str, Any]:
        normalized = _normalize_doi(doi) or doi
        fields = "title,abstract,year,venue,authors,paperId,externalIds"
        return self._get(f"/paper/DOI:{normalized}", params={"fields": fields})

    def normalize(self, payload: dict[str, Any]) -> PaperRecord:
        sid = payload.get("paperId", "unknown")
        doi = _normalize_doi((payload.get("externalIds") or {}).get("DOI"))
        authors = [Author(name=a.get("name", "unknown")) for a in payload.get("authors", [])]
        refs = [r.get("paperId") for r in payload.get("references", []) if r.get("paperId")]
        return PaperRecord(
            canonical_id=f"s2:{sid}",
            title=payload.get("title", "untitled"),
            abstract=payload.get("abstract"),
            year=payload.get("year"),
            venue=payload.get("venue"),
            domain=None,
            doi=doi,
            semantic_scholar_id=sid,
            authors=authors,
            references=refs,
            provenance=[
                Provenance(
                    source=self.source_name,
                    source_id=sid,
                    field_confidence={"semantic_scholar_id": ConfidenceTier.HIGH, "title": ConfidenceTier.HIGH},
                    raw=payload,
                )
            ],
        )


class IngestionOrchestrator:
    def __init__(
        self,
        clients: Iterable[ProviderClient],
        cache: CacheStore | None = None,
        retry_policy: RetryPolicy | None = None,
        per_provider_delay_seconds: float = 0.5,
        max_concurrent_requests: int = 4,
    ) -> None:
        self.clients = list(clients)
        self.cache = cache or CacheStore()
        self.retry_policy = retry_policy or RetryPolicy()
        self.per_provider_delay_seconds = per_provider_delay_seconds
        self.max_concurrent_requests = max_concurrent_requests
        self.seed_queue: queue.Queue[tuple[ProviderClient, str]] = queue.Queue()
        self.budget = ProviderBudget()
        self.last_ingestion_stats: dict[str, Any] = {}
        self.last_openalex_404_seeds: list[str] = []
        self._openalex_404_recoveries: int = 0
        self._ingestion_warnings: list[str] = []

    def _pending_placeholder_paper(self, client: ProviderClient, seed: str) -> PaperRecord:
        """Placeholder when OpenAlex (or other) fetch cannot return cacheable metadata — not a 'Fallback' stub."""
        seed = (seed or "").strip()
        pn = client.source_name
        if pn == "openalex":
            return PaperRecord(
                canonical_id=f"openalex:{seed}",
                title=f"Pending metadata: {seed}",
                abstract=(
                    "Metadata fetch deferred (rate limit, empty API response, or transient error). "
                    "Use **Fetch metadata** in the sidebar or rebuild the graph."
                ),
                year=None,
                venue=None,
                domain=None,
                openalex_id=seed,
                authors=[],
                references=[],
                provenance=[Provenance(source="pending", source_id=seed, raw={"provider": "openalex"})],
            )
        return PaperRecord(
            canonical_id=f"{pn}:{seed}",
            title=f"Pending metadata: {pn}:{seed}",
            abstract="Provider fetch failed; retry later.",
            year=None,
            venue=None,
            domain=None,
            doi=seed if pn in {"doi", "crossref"} else None,
            arxiv_id=seed if pn == "arxiv" else None,
            openalex_id=seed if pn == "openalex" else None,
            authors=[],
            references=[],
            provenance=[Provenance(source="pending", source_id=seed, raw={"provider": pn})],
        )

    def enqueue(self, provider_name: str, seed: str) -> None:
        normalized_provider = provider_name.strip().lower()
        client = next((c for c in self.clients if c.source_name == normalized_provider), None)
        if client is None:
            available = ", ".join(sorted(c.source_name for c in self.clients))
            raise ValueError(
                f"Unknown provider '{provider_name}'. Available providers: {available}"
            )
        self.seed_queue.put((client, seed))

    def _recover_openalex_record_after_404(
        self,
        seed: str,
        *,
        openalex_client: OpenAlexClient | None,
        crossref_client: CrossrefClient | None,
        semantic_client: SemanticScholarClient | None,
        diagnostics_sink: list[str] | None,
    ) -> PaperRecord | None:
        """
        When GET /works/{W…} returns 404: try merge batch, OpenAlex DOI, Crossref, Semantic Scholar (DOI only).
        """
        sid = (seed or "").strip()
        if not sid:
            return None
        if openalex_client and re.match(r"^W\d+$", sid, flags=re.IGNORECASE):
            try:
                self.budget.acquire("openalex")
                part = openalex_client.fetch_works_batch_by_openalex_tails([sid.upper()], chunk_size=1)
                w = part.get(sid.upper()) or part.get(sid)
                if openalex_fetch_payload_valid(w):
                    self.cache.put("openalex", sid, w)
                    if diagnostics_sink is not None:
                        diagnostics_sink.append(f"OpenAlex 404 recovery: merged batch succeeded for {sid}.")
                    return openalex_client.normalize(w)
            except requests.RequestException:
                pass
        doi_try = _normalize_doi(sid) if re.search(r"\b10\.\d{4,9}/", sid, flags=re.IGNORECASE) else None
        if openalex_client and doi_try:
            try:
                self.budget.acquire("openalex")
                resp = openalex_client.fetch_by_doi(doi_try)
                work = openalex_response_first_work(resp)
                if openalex_fetch_payload_valid(work):
                    oid = (work.get("id") or "").rsplit("/", maxsplit=1)[-1]
                    if oid:
                        self.cache.put("openalex", oid, work)
                    self.cache.put("openalex", sid, work)
                    if diagnostics_sink is not None:
                        diagnostics_sink.append(f"OpenAlex 404 recovery: DOI {doi_try} via OpenAlex.")
                    return openalex_client.normalize(work)
            except requests.RequestException:
                pass
        if crossref_client and doi_try:
            try:
                self.budget.acquire("crossref")
                cr_pl = crossref_client.fetch_paper(doi_try)
                rec = crossref_client.normalize(cr_pl)
                rec.provenance.append(
                    Provenance(
                        source="openalex_recovery",
                        source_id=sid,
                        raw={"via": "crossref", "doi": doi_try},
                    )
                )
                self.cache.put("crossref", doi_try, cr_pl)
                if diagnostics_sink is not None:
                    diagnostics_sink.append(f"OpenAlex 404 recovery: Crossref for DOI {doi_try}.")
                return rec
            except requests.RequestException:
                pass
        if semantic_client and doi_try:
            try:
                self.budget.acquire("semantic_scholar")
                s2p = semantic_client.fetch_by_doi(doi_try)
                rec = semantic_client.normalize(s2p)
                rec.provenance.append(
                    Provenance(
                        source="openalex_recovery",
                        source_id=sid,
                        raw={"via": "semantic_scholar", "doi": doi_try},
                    )
                )
                pid = str(s2p.get("paperId") or doi_try)
                self.cache.put("semantic_scholar", pid, s2p)
                if diagnostics_sink is not None:
                    diagnostics_sink.append(f"OpenAlex 404 recovery: Semantic Scholar for DOI {doi_try}.")
                return rec
            except requests.RequestException:
                pass
        return None

    def run(self) -> list[PaperRecord]:
        openalex_client = next((c for c in self.clients if isinstance(c, OpenAlexClient)), None)
        semantic_client = next((c for c in self.clients if isinstance(c, SemanticScholarClient)), None)
        crossref_client = next((c for c in self.clients if isinstance(c, CrossrefClient)), None)
        jobs: list[tuple[ProviderClient, str]] = []
        while not self.seed_queue.empty():
            jobs.append(self.seed_queue.get())
        if not jobs:
            self.last_ingestion_stats = {"jobs": 0, "requests_by_provider": {}, "openalex_404_recoveries": 0}
            return []

        self.last_openalex_404_seeds = []
        self._openalex_404_recoveries = 0
        self._ingestion_warnings = []
        requests_by_provider: dict[str, int] = {}
        stats_lock = threading.Lock()

        def process_one(job: tuple[ProviderClient, str]) -> PaperRecord:
            client, seed = job
            try:
                payload = self.cache.get(client.source_name, seed)
                if client.source_name == "openalex" and payload is not None and not openalex_fetch_payload_valid(
                    payload
                ):
                    payload = None
                if payload is None or self._is_stale_cached_payload(client, seed, payload):
                    self.budget.acquire(client.source_name)
                    payload = self._fetch_with_retry(client, seed)
                    if client.source_name == "openalex" and not openalex_fetch_payload_valid(payload):
                        self._ingestion_warnings.append(
                            f"OpenAlex empty/invalid payload for {seed}; left uncached, using pending placeholder."
                        )
                        return self._pending_placeholder_paper(client, seed)
                    self.cache.put(client.source_name, seed, payload)
                with stats_lock:
                    requests_by_provider[client.source_name] = requests_by_provider.get(client.source_name, 0) + 1
                record = client.normalize(payload)
                if (
                    client.source_name == "crossref"
                    and record.doi
                    and (not record.title or record.title.lower().startswith("doi:"))
                ):
                    record = self._enrich_crossref_record(record, openalex_client, semantic_client)
                return record
            except requests.HTTPError as exc:
                if client.source_name == "openalex":
                    resp = exc.response
                    if resp is not None and resp.status_code == 404:
                        sid = str(seed).strip()
                        recovered = self._recover_openalex_record_after_404(
                            sid,
                            openalex_client=openalex_client,
                            crossref_client=crossref_client,
                            semantic_client=semantic_client,
                            diagnostics_sink=self._ingestion_warnings,
                        )
                        if recovered is not None:
                            self._openalex_404_recoveries += 1
                            with stats_lock:
                                requests_by_provider["openalex"] = requests_by_provider.get("openalex", 0) + 1
                            return recovered
                        self.last_openalex_404_seeds.append(sid)
                        self._ingestion_warnings.append(
                            f"OpenAlex merged or invalid id (HTTP 404) for {sid}; skipped (not API pressure)."
                        )
                        return self._pending_placeholder_paper(client, seed)
                if client.source_name == "openalex":
                    self._ingestion_warnings.append(f"OpenAlex fetch failed for {seed} -> {exc}")
                    return self._pending_placeholder_paper(client, seed)
                raise
            except requests.RequestException as exc:
                if client.source_name == "openalex":
                    self._ingestion_warnings.append(f"OpenAlex fetch failed for {seed} -> {exc}")
                    return self._pending_placeholder_paper(client, seed)
                raise

        ordered: list[PaperRecord | None] = [None] * len(jobs)
        with ThreadPoolExecutor(max_workers=self.max_concurrent_requests) as executor:
            future_to_index = {
                executor.submit(process_one, job): idx for idx, job in enumerate(jobs)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                job = jobs[idx]
                client, seed = job
                try:
                    ordered[idx] = future.result()
                except requests.RequestException as exc:
                    if client.source_name == "openalex":
                        self._ingestion_warnings.append(f"OpenAlex worker error for {seed} -> {exc}")
                        ordered[idx] = self._pending_placeholder_paper(client, seed)
                    else:
                        raise

        self.last_ingestion_stats = {
            "jobs": len(jobs),
            "requests_by_provider": requests_by_provider,
            "max_workers": self.max_concurrent_requests,
            "warnings": list(self._ingestion_warnings),
            "openalex_404_recoveries": int(self._openalex_404_recoveries),
        }
        return [r for r in ordered if r is not None]

    def _fetch_with_retry(self, client: ProviderClient, seed: str) -> dict[str, Any]:
        for attempt in range(1, self.retry_policy.max_attempts + 1):
            try:
                return client.fetch_paper(seed)
            except requests.RequestException:
                if attempt >= self.retry_policy.max_attempts:
                    raise
                time.sleep(self.retry_policy.backoff_seconds * attempt)
        raise RuntimeError("unreachable retry state")

    def _is_stale_cached_payload(self, client: ProviderClient, seed: str, payload: dict[str, Any]) -> bool:
        # Backward compatibility: older arXiv placeholder cache stored only id/title with no metadata.
        if client.source_name != "arxiv":
            return False
        title = (payload.get("title") or "").strip()
        summary = payload.get("summary")
        authors = payload.get("authors") or []
        has_only_placeholder = title == seed and not summary and len(authors) == 0
        return bool(has_only_placeholder)

    def _enrich_crossref_record(
        self,
        record: PaperRecord,
        openalex_client: OpenAlexClient | None,
        semantic_client: SemanticScholarClient | None,
    ) -> PaperRecord:
        if openalex_client is not None:
            try:
                self.budget.acquire("openalex")
                response = openalex_client.fetch_by_doi(record.doi or "")
                oa = openalex_response_first_work(response)
                if isinstance(oa, dict):
                    enriched_title = (oa.get("title") or "").strip()
                    if enriched_title:
                        record.title = enriched_title
                    if not record.abstract:
                        record.abstract = _reconstruct_openalex_abstract(oa.get("abstract_inverted_index"))
                    if not record.openalex_id:
                        oid = (oa.get("id") or "").rsplit("/", maxsplit=1)[-1]
                        record.openalex_id = oid or record.openalex_id
                    if record.title and not record.title.lower().startswith("doi:"):
                        record.provenance.append(
                            Provenance(
                                source="openalex",
                                source_id=record.openalex_id or (record.doi or "unknown"),
                                field_confidence={"title": ConfidenceTier.MEDIUM},
                                raw={"enriched_from": "openalex_by_doi"},
                            )
                        )
                        return record
            except requests.RequestException:
                pass

        if semantic_client is not None:
            try:
                self.budget.acquire("semantic_scholar")
                payload = semantic_client.fetch_by_doi(record.doi or "")
                enriched_title = (payload.get("title") or "").strip()
                if enriched_title:
                    record.title = enriched_title
                if not record.abstract:
                    record.abstract = payload.get("abstract")
                if not record.semantic_scholar_id:
                    record.semantic_scholar_id = payload.get("paperId")
                if record.title and not record.title.lower().startswith("doi:"):
                    record.provenance.append(
                        Provenance(
                            source="semantic_scholar",
                            source_id=record.semantic_scholar_id or (record.doi or "unknown"),
                            field_confidence={"title": ConfidenceTier.MEDIUM},
                            raw={"enriched_from": "semantic_scholar_by_doi"},
                        )
                    )
            except requests.RequestException:
                pass

        # Final fallback: DOI content negotiation (CSL JSON)
        if not record.title or record.title.lower().startswith("doi:"):
            self.budget.acquire("doi_org")
            csl = self._fetch_doi_csl(record.doi)
            if csl:
                title = (csl.get("title") or "").strip()
                if title:
                    record.title = title
                if not record.abstract:
                    record.abstract = csl.get("abstract")
                issued = (csl.get("issued") or {}).get("date-parts") or []
                if issued and issued[0]:
                    record.year = record.year or issued[0][0]
                if not record.venue:
                    container = csl.get("container-title")
                    if isinstance(container, list):
                        record.venue = _first_non_empty(container)
                    elif isinstance(container, str):
                        record.venue = container.strip() or None
                if record.title and not record.title.lower().startswith("doi:"):
                    record.provenance.append(
                        Provenance(
                            source="doi_org",
                            source_id=record.doi or "unknown",
                            field_confidence={"title": ConfidenceTier.MEDIUM},
                            raw={"enriched_from": "doi_content_negotiation"},
                        )
                    )

        # ACL Anthology fallback for 10.18653/v1/* DOIs.
        if not record.title or record.title.lower().startswith("doi:"):
            acl = self._fetch_acl_bib_by_doi(record.doi)
            if acl:
                if acl.get("title"):
                    record.title = acl["title"]
                if not record.venue and acl.get("booktitle"):
                    record.venue = acl["booktitle"]
                if not record.year and acl.get("year"):
                    try:
                        record.year = int(acl["year"])
                    except ValueError:
                        pass
                if record.title and not record.title.lower().startswith("doi:"):
                    record.provenance.append(
                        Provenance(
                            source="acl_anthology",
                            source_id=acl.get("id") or (record.doi or "unknown"),
                            field_confidence={"title": ConfidenceTier.HIGH},
                            raw={"enriched_from": "acl_bibtex"},
                        )
                    )
        return record

    def _fetch_doi_csl(self, doi: str | None) -> dict[str, Any] | None:
        normalized = _normalize_doi(doi)
        if not normalized:
            return None
        try:
            response = requests.get(
                f"https://doi.org/{normalized}",
                headers={
                    "Accept": "application/vnd.citationstyles.csl+json",
                    "User-Agent": "SemanticResearchGraph/0.1 (mailto:local@localhost)",
                },
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else None
        except requests.RequestException:
            return None

    def _fetch_acl_bib_by_doi(self, doi: str | None) -> dict[str, str] | None:
        anthology_id = _extract_acl_anthology_id_from_doi(doi)
        if not anthology_id:
            return None
        try:
            response = requests.get(
                f"https://aclanthology.org/{anthology_id}.bib",
                headers={"User-Agent": "SemanticResearchGraph/0.1"},
                timeout=20,
            )
            response.raise_for_status()
            bib = response.text
            return {
                "id": anthology_id,
                "title": _extract_bibtex_field(bib, "title") or "",
                "booktitle": _extract_bibtex_field(bib, "booktitle") or "",
                "year": _extract_bibtex_field(bib, "year") or "",
            }
        except requests.RequestException:
            return None


class PDFSeedExtractor:
    def __init__(self, openalex_client: OpenAlexClient | None = None) -> None:
        self.openalex_client = openalex_client or OpenAlexClient()

    def _work_from_openalex_doi(self, doi: str) -> dict[str, Any] | None:
        try:
            response = self.openalex_client.fetch_by_doi(doi)
            oa = openalex_response_first_work(response)
            return oa if isinstance(oa, dict) else None
        except requests.RequestException:
            return None

    def _title_from_openalex_doi(self, doi: str) -> str | None:
        oa = self._work_from_openalex_doi(doi)
        if not oa:
            return None
        t = (oa.get("title") or "").strip()
        return t or None

    def _authors_from_openalex_doi(self, doi: str) -> list[str]:
        oa = self._work_from_openalex_doi(doi)
        if not oa:
            return []
        return openalex_authorship_display_names(oa)

    def _title_from_arxiv(self, arxiv_id: str) -> str | None:
        try:
            response = requests.get(
                f"http://export.arxiv.org/api/query?id_list={arxiv_id}",
                timeout=20,
                headers={"User-Agent": "SemanticResearchGraph/0.1"},
            )
            response.raise_for_status()
            root = ET.fromstring(response.content)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entry = root.find("atom:entry", ns)
            if entry is None:
                return None
            title_el = entry.find("atom:title", ns)
            if title_el is None or not title_el.text:
                return None
            return re.sub(r"\s+", " ", title_el.text.strip())
        except (requests.RequestException, ET.ParseError):
            return None

    def _read_pdf_text_pages(self, content: bytes, max_pages: int) -> str:
        if PdfReader is None:
            raise RuntimeError("PDF extraction requires 'pypdf'. Install dependency and retry.")
        reader = PdfReader(io.BytesIO(content))
        parts: list[str] = []
        for page in reader.pages[:max_pages]:
            parts.append(page.extract_text() or "")
        return "\n".join(parts)

    @staticmethod
    def _stem_for_fallback_search(filename: str) -> str | None:
        stem = Path(filename).stem
        stem = re.sub(r"[_\-\.]+", " ", stem).strip()
        stem = re.sub(r"\s+", " ", stem)
        if len(stem) < 6:
            return None
        low = stem.lower()
        noise = ("untitled", "document", "paper", "draft", "main", "manuscript", "ms", "final", "new")
        if low in noise or all(c.isdigit() or c in "._-" for c in stem):
            return None
        return stem

    def _openalex_authors_for_work_id(self, work_id: str) -> list[str]:
        try:
            work = self.openalex_client.fetch_paper(work_id)
            if isinstance(work, dict):
                return openalex_authorship_display_names(work)
        except requests.RequestException:
            pass
        return []

    def _openalex_match_payload(
        self, match: dict[str, str], authors_guess: str | None, title_for_score: str | None
    ) -> dict[str, Any]:
        """Build preview dict from OpenAlex id match; title_for_score may be PDF guess or filename stem."""
        t_src = title_for_score or ""
        mapping_score = _titles_match_score(t_src, match.get("title"))
        api_authors = self._openalex_authors_for_work_id(match["id"])
        mapping_score, cross_note = finalize_pdf_mapping_score(mapping_score, authors_guess, api_authors)
        out: dict[str, Any] = {
            "filename": "",
            "provider": "openalex",
            "id": match["id"],
            "title_guess": title_for_score,
            "authors_guess": authors_guess,
            "matched_title": match["title"],
            "confidence": "medium",
            "origin": "uploaded_pdf",
            "references": [],
            "mapping_score": round(float(mapping_score), 4),
            "pdf_gate_blocked": float(mapping_score) < PDF_MAPPING_QUALITY_THRESHOLD,
        }
        if cross_note:
            out["mapping_note"] = cross_note
        return out

    def extract(
        self,
        filename: str,
        content: bytes,
        deep_parse: bool = False,
    ) -> dict[str, Any]:
        text = self._read_pdf_text(content)
        page12_text = self._read_pdf_text_pages(content, 2)
        header_text = self._header_region(text)
        hunt_doi, hunt_arxiv = smart_id_hunt(page12_text)
        doi = hunt_doi or _extract_doi_from_text(header_text)
        arxiv_id = hunt_arxiv or _extract_arxiv_id_from_text(text)
        title_guess = _guess_title_from_text(text)
        authors_guess = _guess_authors_from_text(text)
        references: list[str] = []
        if deep_parse:
            references = self._extract_reference_dois(text)

        def base(**kwargs: Any) -> dict[str, Any]:
            d = {
                "filename": filename,
                "title_guess": title_guess,
                "authors_guess": authors_guess,
                "origin": "uploaded_pdf",
                "references": references,
            }
            d.update(kwargs)
            d.setdefault("metadata_resolution", "pdf_text")
            return d

        # If both are present, prefer arXiv ID unless DOI is the arXiv DOI alias.
        if arxiv_id and doi and doi != _normalize_doi(f"10.48550/arXiv.{arxiv_id}"):
            vm = fetch_public_metadata_for_seed("arxiv", arxiv_id)
            if vm:
                return base(
                    provider="arxiv",
                    id=arxiv_id,
                    confidence="high",
                    matched_title=vm["title"],
                    mapping_score=0.98,
                    pdf_gate_blocked=False,
                    verification=vm.get("source", "arxiv"),
                    warning="Both DOI and arXiv id detected; arXiv prioritized. Metadata from API (not PDF title).",
                    metadata_resolution="pdf_text",
                )
            meta = fetch_arxiv_public_metadata(arxiv_id)
            api_title = ((meta.get("title") or "").strip() if meta else None) or self._title_from_arxiv(arxiv_id)
            mapping_score = _titles_match_score(title_guess, api_title) if api_title else 0.82
            api_authors = list((meta or {}).get("authors") or [])
            mapping_score, cross_note = finalize_pdf_mapping_score(mapping_score, authors_guess, api_authors)
            out = base(
                provider="arxiv",
                id=arxiv_id,
                confidence="medium",
                warning="Both DOI and arXiv id detected; arXiv prioritized for safety.",
                mapping_score=round(mapping_score, 4),
                pdf_gate_blocked=mapping_score < PDF_MAPPING_QUALITY_THRESHOLD,
                metadata_resolution="pdf_text",
            )
            if cross_note:
                out["mapping_note"] = cross_note
            return out

        if doi:
            vm = fetch_public_metadata_for_seed("doi", doi)
            if vm:
                return base(
                    provider="doi",
                    id=doi,
                    confidence="high",
                    matched_title=vm["title"],
                    mapping_score=0.98,
                    pdf_gate_blocked=False,
                    verification=vm.get("source", "api"),
                    metadata_resolution="doi",
                )
            api_title = self._title_from_openalex_doi(doi)
            mapping_score = _titles_match_score(title_guess, api_title) if api_title else 0.92
            api_authors = self._authors_from_openalex_doi(doi)
            mapping_score, cross_note = finalize_pdf_mapping_score(mapping_score, authors_guess, api_authors)
            out = base(
                provider="doi",
                id=doi,
                confidence="high",
                mapping_score=round(mapping_score, 4),
                pdf_gate_blocked=mapping_score < PDF_MAPPING_QUALITY_THRESHOLD,
                metadata_resolution="doi",
            )
            if cross_note:
                out["mapping_note"] = cross_note
            return out
        if arxiv_id:
            vm = fetch_public_metadata_for_seed("arxiv", arxiv_id)
            if vm:
                return base(
                    provider="arxiv",
                    id=arxiv_id,
                    confidence="high",
                    matched_title=vm["title"],
                    mapping_score=0.98,
                    pdf_gate_blocked=False,
                    verification=vm.get("source", "arxiv"),
                    metadata_resolution="pdf_text",
                )
            meta = fetch_arxiv_public_metadata(arxiv_id)
            api_title = ((meta.get("title") or "").strip() if meta else None) or self._title_from_arxiv(arxiv_id)
            mapping_score = _titles_match_score(title_guess, api_title) if api_title else 0.78
            api_authors = list((meta or {}).get("authors") or [])
            mapping_score, cross_note = finalize_pdf_mapping_score(mapping_score, authors_guess, api_authors)
            out = base(
                provider="arxiv",
                id=arxiv_id,
                confidence="medium",
                mapping_score=round(mapping_score, 4),
                pdf_gate_blocked=mapping_score < PDF_MAPPING_QUALITY_THRESHOLD,
                metadata_resolution="pdf_text",
            )
            if cross_note:
                out["mapping_note"] = cross_note
            return out
        if title_guess:
            match = self._resolve_openalex_by_pdf_title(title_guess)
            if match:
                pl = self._openalex_match_payload(match, authors_guess, title_guess)
                pl["filename"] = filename
                pl["references"] = references
                pl["metadata_resolution"] = "pdf_text"
                return pl
        stem_q = self._stem_for_fallback_search(filename)
        if stem_q:
            stem_search = openalex_search_query_from_pdf_title(stem_q) or stem_q
            match = self._resolve_openalex_by_title(stem_search)
            if match:
                pl = self._openalex_match_payload(match, authors_guess, stem_q)
                pl["filename"] = filename
                pl["references"] = references
                pl["title_guess"] = title_guess
                pl["warning"] = "Resolved via filename stem search (no DOI/arXiv on first pages)."
                pl["metadata_resolution"] = "filename"
                return pl
        return base(
            provider="openalex",
            id="",
            confidence="low",
            unresolved=True,
            mapping_score=0.0,
            pdf_gate_blocked=True,
            metadata_resolution="unknown",
        )

    def _read_pdf_text(self, content: bytes) -> str:
        if PdfReader is None:
            raise RuntimeError("PDF extraction requires 'pypdf'. Install dependency and retry.")
        reader = PdfReader(io.BytesIO(content))
        parts: list[str] = []
        for page in reader.pages[:3]:
            txt = page.extract_text() or ""
            parts.append(txt)
        return "\n".join(parts)

    def _header_region(self, text: str) -> str:
        # Avoid accidental DOI pickup from bibliography by limiting to document head.
        lowered = text.lower()
        cut_points = []
        for token in ("references", "bibliography"):
            idx = lowered.find(token)
            if idx != -1:
                cut_points.append(idx)
        cutoff = min(cut_points) if cut_points else min(len(text), 2500)
        return text[:cutoff]

    def _extract_reference_dois(self, text: str) -> list[str]:
        matches = re.findall(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", text, flags=re.IGNORECASE)
        normalized = []
        seen: set[str] = set()
        for m in matches:
            doi = _normalize_doi(m)
            if doi and doi not in seen:
                seen.add(doi)
                normalized.append(doi)
        return normalized[:30]

    def _resolve_openalex_by_title(self, title: str) -> dict[str, str] | None:
        try:
            result = self.openalex_client.search_by_title(title)
            rows = result.get("results", [])
            if not rows:
                return None
            top = rows[0]
            oid = (top.get("id") or "").rsplit("/", maxsplit=1)[-1]
            t = (top.get("title") or "").strip()
            if not oid:
                return None
            return {"id": oid, "title": t or title}
        except requests.RequestException:
            return None

    def _resolve_openalex_by_pdf_title(self, raw_title: str | None) -> dict[str, str] | None:
        if not (raw_title or "").strip():
            return None
        q = openalex_search_query_from_pdf_title(raw_title) or raw_title.strip()
        return self._resolve_openalex_by_title(q)


def process_pdf_to_metadata(
    filename: str,
    content: bytes,
    deep_parse: bool = False,
) -> dict[str, Any]:
    """Convert raw PDF bytes into canonical seed-oriented metadata."""
    extractor = PDFSeedExtractor()
    return extractor.extract(filename=filename, content=content, deep_parse=deep_parse)

