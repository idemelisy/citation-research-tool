"""Map OpenAlex concepts / fields / venues to internal discipline buckets."""

from __future__ import annotations

from typing import Any

# OpenAlex field ids (top-level) -> internal domain
OPENALEX_FIELD_TO_DOMAIN: dict[str, str] = {
    "https://openalex.org/fields/17": "cs",
    "https://openalex.org/fields/27": "cs",
    "https://openalex.org/fields/33": "biomedical",
    "https://openalex.org/fields/19": "biomedical",
    "https://openalex.org/fields/38": "social_sciences",
    "https://openalex.org/fields/22": "physics",
}

# Substring hints in concept display_name
# Conference / venue name fragments → internal domain (when OpenAlex primary_topic is missing).
VENUE_NAME_TO_DOMAIN: list[tuple[str, str]] = [
    ("emnlp", "cs"),
    ("acl", "cs"),
    ("naacl", "cs"),
    ("eacl", "cs"),
    ("coling", "cs"),
    ("iclr", "cs"),
    ("neurips", "cs"),
    ("nips", "cs"),
    ("icml", "cs"),
    ("cvpr", "cs"),
    ("eccv", "cs"),
    ("iccv", "cs"),
    ("aaai", "cs"),
    ("ijcai", "cs"),
    ("sigir", "cs"),
    ("www conference", "cs"),
    ("the web conference", "cs"),
    ("kdd", "cs"),
    ("wsdm", "cs"),
    ("findings of", "cs"),
    ("machine learning", "cs"),
    ("arxiv", "cs"),
    ("bioinformatics", "biomedical"),
    ("nature medicine", "biomedical"),
    ("jama", "biomedical"),
    ("lancet", "biomedical"),
    ("physical review", "physics"),
]


def domain_from_venue_name(venue: str | None) -> str | None:
    """Infer domain from host venue / proceedings title (ACL-style NLP → cs)."""
    v = (venue or "").strip().lower()
    if not v:
        return None
    for hint, dom in VENUE_NAME_TO_DOMAIN:
        if hint in v:
            return dom
    return None


CONCEPT_HINTS: list[tuple[str, str]] = [
    ("computer", "cs"),
    ("machine learning", "cs"),
    ("artificial intelligence", "cs"),
    ("computational", "cs"),
    ("medicine", "biomedical"),
    ("biology", "biomedical"),
    ("psychology", "social_sciences"),
    ("sociology", "social_sciences"),
    ("physics", "physics"),
    ("chemistry", "physics"),
]


def domain_from_openalex_primary_topic(payload: dict[str, Any]) -> str | None:
    """Map OpenAlex `primary_topic` (field / subfield ids) to internal domain."""
    pt = payload.get("primary_topic")
    if not isinstance(pt, dict):
        return None
    for key in ("field", "domain", "subfield"):
        obj = pt.get(key)
        if isinstance(obj, dict):
            fid = obj.get("id")
            if isinstance(fid, str) and fid in OPENALEX_FIELD_TO_DOMAIN:
                return OPENALEX_FIELD_TO_DOMAIN[fid]
    return None


def domain_from_openalex_payload(payload: dict[str, Any]) -> str | None:
    """Infer internal domain from OpenAlex work JSON."""
    dom = domain_from_openalex_primary_topic(payload)
    if dom:
        return dom
    primary = payload.get("primary_location") or {}
    # Some responses use host_organization / source
    topics = payload.get("topics") or []
    concepts = payload.get("concepts") or []

    # Field from primary topic
    for t in topics[:3]:
        if isinstance(t, dict):
            fid = t.get("field", {}).get("id") if isinstance(t.get("field"), dict) else None
            if fid and fid in OPENALEX_FIELD_TO_DOMAIN:
                return OPENALEX_FIELD_TO_DOMAIN[fid]

    # Concepts (sorted by score in API)
    for c in concepts[:5]:
        if not isinstance(c, dict):
            continue
        name = (c.get("display_name") or "").lower()
        for hint, dom in CONCEPT_HINTS:
            if hint in name:
                return dom

    # Venue / host name (conference proceedings)
    venue_bits: list[str] = []
    src = primary.get("source")
    if isinstance(src, dict):
        venue_bits.append(str(src.get("display_name") or ""))
    ho = primary.get("host_organization") or primary.get("host_organization_name")
    if isinstance(ho, dict):
        venue_bits.append(str(ho.get("display_name") or ""))
    elif isinstance(ho, str):
        venue_bits.append(ho)
    for bit in venue_bits:
        vd = domain_from_venue_name(bit)
        if vd:
            return vd

    return None
