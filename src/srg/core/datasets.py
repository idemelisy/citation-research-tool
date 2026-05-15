"""Benchmark query sets and canonical-paper gold expectations."""

from __future__ import annotations

from typing import Any

# Fuzzy title match targets (substring / token overlap in metrics).
CANONICAL_GOLD: dict[str, list[str]] = {
    "direct preference optimization": [
        "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
    ],
    "dpo": [
        "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
    ],
    "dpo alignment": [
        "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
    ],
    "rlhf": [
        "Training language models to follow instructions with human feedback",
    ],
    "reinforcement learning from human feedback": [
        "Training language models to follow instructions with human feedback",
    ],
    "transformers": [
        "Attention Is All You Need",
    ],
    "transformer": [
        "Attention Is All You Need",
    ],
    "bert": [
        "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    ],
    "rag": [
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    ],
    "retrieval augmented generation": [
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    ],
    "graph neural networks": [
        "Semi-Supervised Classification with Graph Convolutional Networks",
    ],
    "gcn": [
        "Semi-Supervised Classification with Graph Convolutional Networks",
    ],
    "diffusion": [
        "Denoising Diffusion Probabilistic Models",
    ],
    "diffusion models": [
        "Denoising Diffusion Probabilistic Models",
    ],
    "constitutional ai": [
        "Constitutional AI: Harmlessness from AI Feedback",
    ],
    "mixture of experts": [
        "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer",
    ],
    "moe": [
        "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer",
    ],
    "visual debugging": [
        "Software Visualization",
    ],
    "chain of thought": [
        "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
    ],
    "chain of thought reasoning": [
        "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
    ],
}

BENCHMARK_QUERIES: list[str] = [
    # alignment / preference
    "direct preference optimization",
    "dpo alignment",
    "dpo",
    "preference optimization",
    "rlhf",
    "reinforcement learning from human feedback",
    "reward modeling",
    "constitutional ai",
    "rlaif",
    "orpo preference optimization",
    "simpo alignment",
    # transformers / NLP
    "transformers",
    "transformer interpretability",
    "attention is all you need",
    "bert",
    "gpt scaling laws",
    "large language models",
    "llm reasoning",
    "chain of thought reasoning",
    "instruction tuning",
    # retrieval / RAG
    "retrieval augmented generation",
    "rag",
    "dense passage retrieval",
    "hybrid search retrieval",
    "semantic search",
    # agents / tools
    "llm agents",
    "tool use llms",
    "autonomous agents",
    "multi-agent systems",
    "reAct prompting",
    # multimodal
    "multimodal reasoning",
    "vision language models",
    "clip multimodal",
    "diffusion transformers",
    "stable diffusion",
    # graph / ML
    "graph neural networks",
    "message passing neural networks",
    "graph attention network",
    "graph learning",
    "knowledge graphs embedding",
    # domains
    "protein language models",
    "protein structure prediction",
    "biological foundation models",
    "single cell genomics transformer",
    # diffusion / generative
    "diffusion models",
    "denoising diffusion",
    "score based generative models",
    "flow matching generative",
    # MoE / efficiency
    "mixture of experts",
    "moe llm",
    "sparse mixture of experts",
    # code
    "code llms",
    "program synthesis with llms",
    "code generation transformer",
    # summarization / IR
    "summarization",
    "abstractive summarization neural",
    "text summarization survey",
    "information retrieval neural",
    # ambiguous / short
    "alignment",
    "agents",
    "reasoning",
    "retrieval",
    "optimization",
    "gpt-4 technical report",
    # long natural language
    "how do large language models learn from human preferences without reinforcement learning",
    "papers about direct preference optimization for aligning language models",
    "graph neural networks for citation networks and literature discovery",
    "multimodal models that combine vision and language for reasoning tasks",
    # method + acronym
    "PPO proximal policy optimization language models",
    "DPO vs RLHF comparison",
    "GNN survey",
    "RAG for knowledge intensive tasks",
    # specific titles (seed resolution stress)
    "Attention Is All You Need",
    "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
    "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
]


def canonical_titles_for_query(query: str) -> list[str]:
    """Return expected canonical title strings for a query (best-effort)."""
    q = (query or "").strip().lower()
    if q in CANONICAL_GOLD:
        return list(CANONICAL_GOLD[q])
    for key, titles in CANONICAL_GOLD.items():
        if key in q or q in key:
            return list(titles)
    return []


def benchmark_query_list(*, limit: int | None = None) -> list[str]:
    qs = list(BENCHMARK_QUERIES)
    if limit is not None:
        return qs[: max(0, int(limit))]
    return qs
