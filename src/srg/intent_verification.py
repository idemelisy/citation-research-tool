"""
Intent Verification and Semantic Alignment Layer (SRG Lite v2 extension).

Re-scores candidates after fusion + semantic control using:
  final = α·intent_similarity + β·graph_score + γ·local_coherence − δ·semantic_drift

When ``sentence-transformers`` is installed, intent_similarity uses cosine similarity
of normalized embeddings for query vs title+abstract. Otherwise falls back to the
existing lexical semantic score (same scale [0,1]).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# model_name -> SentenceTransformer instance, or None if load failed
_st_models: dict[str, Any] = {}
_st_failed: set[str] = set()


def _get_sentence_transformer(model_name: str) -> Any:
    if model_name in _st_failed:
        return None
    if model_name in _st_models:
        return _st_models[model_name]
    try:
        from sentence_transformers import SentenceTransformer

        m = SentenceTransformer(model_name)
        _st_models[model_name] = m
        return m
    except Exception:
        _st_failed.add(model_name)
        return None


def _paper_text_for_embed(node: dict[str, Any]) -> str:
    title = (node.get("title") or "").strip()
    abstract = (str(node.get("abstract") or "").strip())[:2800]
    if not title and not abstract:
        return "untitled"
    return f"{title}. {abstract}".strip()


def _embedding_intent_scores(
    query_text: str,
    nodes: list[dict[str, Any]],
    model_name: str,
    *,
    batch_size: int = 32,
) -> dict[str, float] | None:
    if not query_text.strip() or not nodes:
        return {}
    model = _get_sentence_transformer(model_name)
    if model is None:
        return None
    try:
        import numpy as np
    except ImportError:
        return None

    q_emb = model.encode([query_text], normalize_embeddings=True, show_progress_bar=False)
    qv = np.asarray(q_emb[0], dtype=np.float64)
    qn = np.linalg.norm(qv) or 1.0
    qv = qv / qn

    texts = [_paper_text_for_embed(n) for n in nodes]
    out: dict[str, float] = {}
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        chunk_nodes = nodes[i : i + batch_size]
        emb = model.encode(chunk, normalize_embeddings=True, show_progress_bar=False)
        mat = np.asarray(emb, dtype=np.float64)
        dots = mat @ qv
        for j, n in enumerate(chunk_nodes):
            cos = float(dots[j])
            out[str(n["id"])] = max(0.0, min(1.0, (cos + 1.0) * 0.5))
    return out


def compute_intent_similarities(
    query_text: str,
    nodes: list[dict[str, Any]],
    opts: dict[str, Any],
    semantic_fallback: Callable[[dict[str, Any], dict[str, Any]], float],
    query_profile: dict[str, Any],
) -> tuple[dict[str, float], str, str | None]:
    """
    Returns (scores_by_paper_id, backend_label, model_name_or_none).
    backend_label: ``embedding`` | ``semantic_fallback``
    """
    model_name = str(opts.get("intent_verification_model") or "all-MiniLM-L6-v2").strip()
    if bool(opts.get("intent_verification_use_embeddings", True)):
        emb_scores = _embedding_intent_scores(query_text, nodes, model_name)
        if emb_scores is not None:
            return emb_scores, "embedding", model_name
    by_id: dict[str, float] = {}
    for n in nodes:
        by_id[str(n["id"])] = float(semantic_fallback(n, query_profile))
    return by_id, "semantic_fallback", None


def apply_intent_verification_layer(
    nodes: list[dict[str, Any]],
    query_text: str,
    opts: dict[str, Any],
    *,
    semantic_fallback: Callable[[dict[str, Any], dict[str, Any]], float],
    query_profile: dict[str, Any],
) -> dict[str, Any]:
    """
    Overwrites ``relevance_raw`` and ``final_score`` on each node (preserves
    ``relevance_raw_pre_intent_verification`` snapshot).
    """
    alpha = float(opts.get("intent_verification_alpha", 0.45) or 0.45)
    beta = float(opts.get("intent_verification_beta", 0.32) or 0.32)
    gamma = float(opts.get("intent_verification_gamma", 0.15) or 0.15)
    delta = float(opts.get("intent_verification_delta", 0.12) or 0.12)
    alpha = max(0.0, min(1.0, alpha))
    beta = max(0.0, min(1.0, beta))
    gamma = max(0.0, min(1.0, gamma))
    delta = max(0.0, min(0.5, delta))

    intent_by_id, backend, model_used = compute_intent_similarities(
        query_text, nodes, opts, semantic_fallback, query_profile
    )

    sims: list[float] = []
    for node in nodes:
        pid = str(node["id"])
        node["relevance_raw_pre_intent_verification"] = float(node.get("relevance_raw", 0.0))
        intent_sim = float(intent_by_id.get(pid, node.get("semantic_score", 0.0) or 0.0))
        node["intent_similarity"] = round(intent_sim, 4)
        node["intent_verification_backend"] = backend
        if model_used:
            node["intent_verification_model"] = model_used

        g = float(node.get("graph_score", 0.0))
        loc = float(node.get("local_semantic_connectivity", 0.45))
        drift = float(node.get("semantic_drift_penalty_norm", 0.0))
        raw = alpha * intent_sim + beta * g + gamma * loc - delta * drift
        raw = max(0.0, raw)
        node["relevance_raw"] = raw
        node["final_score"] = round(float(raw), 6)
        sims.append(intent_sim)

    mean_sim = sum(sims) / len(sims) if sims else 0.0
    return {
        "backend": backend,
        "model": model_used,
        "weights": {"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta},
        "mean_intent_similarity": round(mean_sim, 4),
    }
