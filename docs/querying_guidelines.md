# SRG Lite — querying guidelines

SRG Lite builds a **citation neighborhood** from **anchors** (your query, resolved to papers via OpenAlex and optional canonical seeds). Results follow whatever literature cluster those anchors sit in. These guidelines help you steer that cluster on purpose.

---

## 1. How retrieval thinks (short mental model)

1. **Discover** — Your text is turned into **seed works** (OpenAlex title search, plus known canonical packs for a few topics like GNN / Transformer / RLHF-style alignment).
2. **Expand** — The graph grows along **references and citations** from those seeds.
3. **Rank** — Papers are scored with semantics, graph structure, intent, etc.

If step 1 lands in the **wrong** subfield, step 2 will usually produce a **coherent but wrong** map. Fixing the **anchors** fixes the map faster than hoping ranking will rescue a bad neighborhood.

---

## 2. Prefer strong anchors

| Approach | When to use it |
|----------|----------------|
| **DOI** (`10.xxxx/...`) | You have one definitive paper; you want the graph centered on it. |
| **arXiv id** (`1706.03762` or `arxiv:1706.03762`) | Same as DOI; very stable for CS / ML. |
| **Exact or distinctive title** | Disambiguates better than three generic words. |
| **Narrow technical phrase** | Better than a slogan when you care about a specific line of work. |

Weak anchors: **very short**, **multi‑meaning**, or **three unrelated words** (e.g. “browser visual debugging”) — OpenAlex may match **program debugging / slicing** instead of **browser devtools**, because that citation ball is huge.

---

## 3. Query phrasing

### Do

- **Name the artifact or subfield:** e.g. “Chrome DevTools”, “React concurrent rendering”, “post‑quantum lattice signatures”.
- **Add one disambiguator:** “graph neural networks survey”, “transformer attention NLP” (still not perfect, but better than one ambiguous word).
- **Paste a paper title** you already trust and let the system branch from it.

### Avoid (unless you intend the classical CS meaning)

- **Overloaded pairs:** “visual debugging” alone → many non‑UI meanings in the literature.
- **Product jargon without product name:** “browser visual debugging” → prefer **“browser developer tools”**, **“in‑browser JavaScript debugging”**, or a **specific tool/paper**.

### If results look old or off‑topic

1. Open **Anchor seeds (retrieval diagnostics)** in Lite and read the list — if seeds look wrong, the graph will follow.
2. **Tighten the query** or **seed with a DOI/arXiv** from a paper in the right cluster.
3. Run the repo check script when debugging the product:  
   `python scripts/verify_lite_anchor_queries.py` (known benchmark queries).

---

## 4. Examples (illustrative)

| Intent | Risky query | Stronger alternative |
|--------|-------------|----------------------|
| Web UI debugging in the browser | browser visual debugging | Chrome DevTools; in‑browser JavaScript debugging; web application debugging |
| Modern GNN methods | graph nets (ambiguous) | graph neural networks (canonical pack applies); or cite Kipf / survey DOI |
| NLP Transformers | transformer | transformer attention language model; or arXiv `1706.03762` |
| Alignment / preferences | RLHF | RLHF reward modeling; DPO; or a specific paper title |

---

## 5. What Lite is not (scope)

- **Not a single‑snippet answer engine** — it orients you in a **literature map**, not one paragraph of truth.
- **Not guaranteed coverage** for every niche — sparse or industrial topics may need **explicit seeds** (DOI/arXiv).
- **API and coverage limits** — rate limits, missing references, or OpenAlex gaps can thin the graph; retrying or narrowing sometimes helps.

---

## 6. Quick checklist before you export

- [ ] Query is **specific enough** that a colleague could guess your subfield from it alone.
- [ ] **Anchor seeds** (diagnostics) look like papers you would actually start from.
- [ ] Top reading list **years and titles** match what you mean (e.g. modern ML vs 1980s PL).
- [ ] If not, **change the query** or **add a DOI/arXiv** and run again.

---

*Last updated: aligns with SRG Lite citation‑first discovery and `scripts/verify_lite_anchor_queries.py` benchmark behavior.*
