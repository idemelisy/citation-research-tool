# SRG Lite — Product brief for decision-makers

**Audience:** Engineering managers, product leads, research leads evaluating whether to invest in, pilot, or compare this tool.  
**Purpose:** Describe what SRG Lite does, how users should operate it, where it excels, where it fails, and what criteria to use when accepting or criticizing it.  
**Companion docs:** [Querying guidelines](querying_guidelines.md) (user-facing phrasing and anchors); repo script `scripts/verify_lite_anchor_queries.py` (automated regression on a small benchmark set).

This document is written to **enable criticism**: limitations are stated explicitly, not minimized.

---

## 1. Executive summary

**SRG Lite** is a **literature-orientation** tool. Given a topic string, a paper title, a DOI, or an arXiv identifier, it:

1. **Resolves** one or more **anchor papers** (seeds) via external scholarly APIs (primarily **OpenAlex**, with **arXiv** and related ingestion paths).
2. **Expands** a **citation graph** (references, citations, controlled multi-hop expansion, coupling) around those anchors.
3. **Ranks and explains** nodes for reading-list style output, optional graph visualization, and **Markdown export**.

It is **not** a general question-answering chatbot, **not** a guaranteed systematic review generator, and **not** a replacement for domain expertise. It is best understood as **“show me the citation neighborhood and a defensible reading order around how I anchored this query.”**

**Strategic fit:** Strong when users need **structure** (how papers cite each other, what cluster they landed in) and **orientation** (where to start reading). Weak when the user needs **one correct answer**, **complete coverage** of a field, or **high precision** from **ambiguous three-word queries** without disambiguation.

**Operational reality:** Output quality is **anchor-sensitive** and **API-sensitive**. The same product can produce excellent maps for well-tied queries (e.g. modern “graph neural networks” when canonical seeds and graph expansion align) and misleading maps for ambiguous phrasing (e.g. “browser visual debugging” drifting into classical program slicing).

---

## 2. Product definition

### 2.1 What SRG Lite is

| Dimension | Description |
|-----------|-------------|
| **Core object** | A **directed citation graph** whose nodes are works (papers, preprints) and whose edges are evidence-weighted citation or derived relations (e.g. coupling), enriched with metadata where APIs provide it. |
| **Primary input** | Free text (topic or title-like phrase), **DOI**, or **arXiv id**. |
| **Primary output** | Ranked **reading list**, **topic buckets**, **reference-coverage** stats, optional **graph view**, **Markdown** (and related) export; **retrieval diagnostics** (anchor seeds) for transparency. |
| **Differentiator vs linear search** | Surfaces **multi-paper context**: neighborhoods, branches, co-citation–style structure, not only a ranked list from a single keyword search. |

### 2.2 What SRG Lite is not

- **Not** authoritative truth: it aggregates bibliographic and model-derived signals; errors in sources or ranking propagate.
- **Not** completeness: expansion budgets, API coverage, and deduplication choices truncate the universe of papers.
- **Not** domain-tuned for every niche: a small set of **canonical anchor packs** exists for high-traffic ML topics (e.g. GNN line, Transformer, RLHF-style alignment). Most other topics rely on **OpenAlex search quality + user phrasing**.
- **Not** a substitute for reading: “why” strings and scores are **heuristic explanations**, not peer review.

---

## 3. Problem statement and value proposition

### 3.1 User pain (aligned with broader SRG vision)

Researchers suffer **information overload with weak global structure**: keyword tools return long lists; important methodological links live in **citation topology**, not in titles alone. Multi-source scholarly data is **inconsistent** (duplicates, missing references, ID fragmentation across OpenAlex / arXiv / DOI).

### 3.2 Value SRG Lite offers

- **Orientation:** “If I start from these anchors, what cluster of literature do I inherit?”
- **Transparency:** Users can inspect **which papers anchored** the expansion (diagnostics), reducing black-box frustration when results look wrong.
- **Exportable artifact:** Markdown reading lists suitable for notes, onboarding reading, or appendices to internal memos (with the caveat that quality depends on anchors).

### 3.3 Value SRG Lite does not claim

- Automatic detection of **research gaps** at publication quality.
- Legal, medical, or compliance-grade **evidence synthesis**.
- Neutral coverage of **all** viewpoints in a controversy (graph follows citations, which encode academic fashion and power laws).

---

## 4. Intended users and primary use cases

### 4.1 Personas

| Persona | Typical goal | Fit |
|---------|--------------|-----|
| **PhD student / new joiner** | “I was told to read up on X; where do I start?” | **Good** if X is phrased with strong anchors or a seed paper; **risky** if X is vague. |
| **Staff / applied researcher** | “Map the papers around this method for a design doc.” | **Good** for methods with dense graphs (ML, CS theory with good OpenAlex coverage). |
| **Tech lead / PM** | “What is the literature around this buzzword?” | **Mixed** — buzzwords without disambiguation often land in the wrong cluster. |
| **Meta-research / tooling evaluator** | “Does this tool behave sanely on benchmarks?” | **Good** — scripted checks exist (see §10). |

### 4.2 Primary use cases (recommended)

1. **Seed-first exploration:** User pastes a **trusted DOI or arXiv id**, then reads outward into citations.
2. **Method survey with tight naming:** User enters a **specific method name** that matches scholarly titles (e.g. “direct preference optimization” — with awareness of drift; see §8.2).
3. **Known-good topic phrases:** Examples that align with canonical packs or dense OpenAlex clusters (e.g. “graph neural networks” when the pipeline includes modern canonical anchors).
4. **Diagnostics-driven iteration:** User inspects anchor list; if wrong, **rephrases or re-seeds** before trusting reading order.

### 4.3 Secondary / discouraged use cases (without extra safeguards)

- **Single ambiguous phrase** with multiple meanings in CS (“visual debugging”, “browser” + “debugging”).
- **Legal / clinical** decision support without human verification.
- **Competitive intelligence** where recall of non-academic sources matters (tool is academic-API-centric).

---

## 5. How the system works (manager-level architecture)

This section is accurate enough to support architectural questions; it omits file-level detail.

### 5.1 Pipeline stages

1. **Discovery (`discover_from_query`)**  
   - Normalizes input; detects DOI / arXiv literal patterns.  
   - For text: **OpenAlex work search** (title-style search) with multiple hits per variant where configured.  
   - **Canonical anchor seeds:** For a **fixed set** of query patterns (e.g. GNN-related wording, Transformer + NLP-style wording, RLHF / preference optimization lexicon), the system **prepends** known **arXiv** identifiers to anchor modern literature before expansion. This is a **curated repair**, not general AI understanding of every topic.  
   - **Filtering:** When canonical anchors exist, some OpenAlex hits can be **rejected** (e.g. industrial/applied titles that regex-match known failure modes; weak title–query string similarity). Fallback rules may still retain a rank-1 hit if all rows fail filters (documented in diagnostics).  
   - Optional **broadening** adds more candidate ids when the seed pool is sparse (thresholds configurable in code).

2. **Ingestion (`run_pipeline`)**  
   - Fetches metadata and reference lists from providers within **budget** limits.  
   - **Dedupes** and merges records across providers; hydrates missing fields where possible.  
   - Builds the **citation graph** and downstream scores.

3. **Ranking and UX layers**  
   - Combines **semantic** (embeddings when available, else lexical fallbacks), **graph** (proximity, degree, coupling), **intent** classification, and **Lite v2** options (semantic control, intent verification when dependencies exist).  
   - Produces **reading list ordering**, **topic separation** buckets, **foundational** highlights where rules fire, and **Markdown** export.

### 5.2 External dependencies (risk surface)

| Dependency | Role | Manager implication |
|------------|------|---------------------|
| **OpenAlex** | Primary discovery and many reference spines | Rate limits, partial metadata, search ranking quirks directly affect anchors. |
| **arXiv** | Preprints, canonical ids | Strong for CS/ML; not universal. |
| **Crossref / others** | DOI bridges, enrichment | Coverage gaps → thinner graphs. |
| **Embedding model** (optional) | Semantic scoring / intent verification | If installed: heavier runtime and model load; if not: weaker semantic discrimination. |

### 5.3 Honest limitation on “semantic”

Even with embeddings, **semantic similarity is not semantic correctness**. The model can score unrelated papers highly if they share embedding-space neighborhood with query wording. **Citation structure** can dominate once the graph is built—so **wrong anchors** remain the dominant failure mode.

---

## 6. How users should use the tool (operating manual)

This section is prescriptive: following it improves outcomes measurably.

### 6.1 Golden rules

1. **Trust anchors first.** If “Anchor seeds (retrieval diagnostics)” looks off-topic, treat the reading list as **unsafe** until the query changes.  
2. **Prefer identifiers over slogans.** DOI and arXiv ids collapse ambiguity.  
3. **Disambiguate overloaded language.** “Transformer” alone can mean power electronics, ViT, or NLP; “debugging” can mean program slicing or browser devtools.  
4. **Iterate once.** If the first map is wrong, **change the anchor**, not only the sort order in your head.

### 6.2 Recommended query patterns (examples)

| User intent | Weak input | Stronger input |
|-------------|------------|----------------|
| LLM alignment, DPO | `preference optimization` (recsys collision) | `direct preference optimization` + verify flagship paper appears; or paste **arXiv `2305.18290`** |
| Modern GNNs | `graph nets` | `graph neural networks` (canonical pack); or seed **Kipf** / a survey DOI |
| NLP Transformer | `transformer` | `transformer attention language model` or **arXiv `1706.03762`** |
| Browser devtools | `browser visual debugging` | `Chrome DevTools`, `browser developer tools`, `in-browser JavaScript debugging` |

Fuller tables live in [querying_guidelines.md](querying_guidelines.md).

### 6.3 How to read the outputs

| Output region | What it means | How to misuse it |
|---------------|---------------|------------------|
| **Reading list (top N)** | Heuristic ordering for “where to click first” in *this* graph | Treating it as a canonical curriculum for a PhD oral exam |
| **Topic separation** | Clustered labels for exploration; can include **noise** when tokens collide | Assuming every bucket title implies thematic purity |
| **Reference coverage counts** | How much reference metadata exists in-node | Confusing “many references” with “user read them” |
| **Foundational papers** | Hub-like nodes under graph/stat rules | Confusing with “most important paper ethically or historically” |
| **Citation structure line** | Evidence vs exploratory edge mix | Ignoring that exploratory edges still influence exploration UX |

### 6.4 Exports (Markdown)

Exports summarize the **current graph slice** after filters. They are suitable for:

- Personal study notes  
- Internal “starter reading” appendices  
- Demos **with** the disclaimer that anchors were sane

They are **not** suitable as sole evidence in peer-reviewed systematic reviews without methodology and independent search.

### 6.5 Running the application (engineering handoff)

From project materials, full UI mode has been documented as:

`python -m srg.app --mode ui`

(Use the project’s documented environment, e.g. Conda env with dependencies installed.)  
**SRG Lite** is the Streamlit-oriented “lite” experience in the codebase (`app_ui_lite.py`); exact launch command should follow whatever entrypoint the team standardizes in README or internal runbooks.

---

## 7. Strengths (what to praise in a pilot review)

1. **Citation-first mental model** matches how many researchers actually explore after they have one good paper.  
2. **Diagnostic seeds** improve **debuggability** compared to pure neural retrieval: users see *why* the map exists.  
3. **Targeted repairs** for known high-value ML queries (canonical arXiv packs, industrial-title rejection for certain Transformer queries) show engineering responsiveness to **documented failure modes**.  
4. **Automated benchmark script** (`scripts/verify_lite_anchor_queries.py`) supports **regression discipline** on a small set of queries without manual UI clicking.  
5. **Rich graph statistics** (edge counts, coverage) help managers distinguish “thin API day” from “conceptually empty topic.”

---

## 8. Limitations and known failure modes (for criticism)

This section lists issues managers **should** ask about in a pilot retrospective.

### 8.1 Anchor ambiguity (structural)

**Symptom:** Coherent graph that answers the *wrong* research question.  
**Example observed:** “browser visual debugging” mapped heavily into **program dependence graphs and slicing** (1970s–1990s CS), not browser devtools—because “debugging” + “program” literature dominates citation space.  
**Mitigation:** User rephrasing; DOI/arXiv seeding; future product work could add domain hints or query-class detectors (not guaranteed today).

### 8.2 Lexical collision inside CS/ML

**Symptom:** “Preference optimization” appears in **recommender systems** and **LLM alignment**. DPO-related queries can surface **recommendation** papers in the top list even when the user meant **Rafailov-style DPO**.  
**Mitigation:** Explicit arXiv seed for the flagship paper; query adds “language model” / “LLM”; product work could add stronger **intent gating** for this collision class.

### 8.3 Canonical coverage is partial by design

Canonical arXiv prepends exist for **enumerated** patterns (e.g. GNN pack, Vaswani for certain Transformer phrasings, RLHF-style lexicon). **Any topic outside that enumeration** depends on OpenAlex. Managers should not assume “canonical packs exist for all of CS.”

### 8.4 API and rate-limit fragility

Graph size and quality swing with **429/503**, partial reference lists, or **OpenAlex search rank changes**. The product should be evaluated under **normal** and **degraded** network conditions.

### 8.5 Embedding and model load

When semantic embeddings are enabled, startup and runs are **heavier**; environments without GPU or with cold caches feel slower. This affects **adoption** for casual users.

### 8.6 Export ≠ full reproducibility

Markdown exports typically do not include every internal id and diagnostic unless configured; reproducing a session exactly may require **stored seeds + version hash** of code and models (if the team wants audit-grade reproducibility, that is additional work).

### 8.7 Ranking cannot fully fix wrong-cluster anchors

If discovery lands in Cluster A, ranking within Cluster A will still look “sensible.” **Only anchor repair** reliably moves the user to Cluster B.

---

## 9. Evaluation criteria for managers (accept / iterate / reject)

Use this as a **scorecard** in a pilot.

| Criterion | Pass signal | Fail signal |
|-----------|-------------|-------------|
| **Anchor transparency** | Users routinely open diagnostics and understand seeds | Users distrust tool because seeds are opaque |
| **Strong-seed success rate** | DOI/arXiv paths produce on-topic lists ≥ agreed threshold | Frequent misses even with DOI |
| **Ambiguous-query behavior** | Team documents “minimum viable query specificity” and users comply | Users blame model for predictable ambiguity failures |
| **Regression hygiene** | `verify_lite_anchor_queries.py` passes in CI or pre-release | Drift in anchors between releases unnoticed |
| **Time-to-first-useful-map** | Acceptable p95 latency in target env | Too slow for intended workflow |
| **Export usefulness** | Markdown used in real notes / onboarding | Exports ignored; users screenshot graph only |

**Decision framing:**

- **Iterate** if value is clear for **seed-first** workflows but ambiguous queries fail often.  
- **Reject** (for a given use case) if the organization needs **guaranteed recall** or **non-academic corpora** as primary sources.  
- **Adopt narrowly** as an **internal orientation tool** with training + querying guidelines, rather than as a customer-facing “answer engine.”

---

## 10. Quality engineering and regression

### 10.1 Scripted benchmark

`scripts/verify_lite_anchor_queries.py` exercises:

- `discover_from_query(..., text_search_backend="openalex")`  
- `run_pipeline(..., discovery_options=LITE_DISCOVERY_OPTIONS)`  

on a small set of **benchmark queries** (e.g. graph neural networks, transformer attention mechanism, RLHF reward modeling) and asserts properties such as:

- Presence of **expected canonical arXiv anchors** where designed  
- Absence of **known bad OpenAlex seeds** for certain Transformer queries  
- **Graph presence** of flagship identifiers (e.g. Kipf GCN id) after pipeline  
- **Seed title resolution** for arXiv seeds that merge into OpenAlex-primary nodes  

**Flags:** `--discover-only` for faster checks; `--only gnn|transformer|rlhf` for single-case debugging.

### 10.2 What automated checks do *not* cover

- Subjective reading-list quality for **all** of science  
- UI/UX satisfaction  
- Longitudinal drift of OpenAlex ranking  
- Novel failure modes for queries never added to the script

Managers should budget **human spot checks** on representative internal queries quarterly if the tool is production-adjacent.

---

## 11. Positioning vs alternatives (not competitive marketing—framing)

| Alternative | SRG Lite difference |
|-------------|---------------------|
| **Google Scholar / single search** | SRG emphasizes **multi-seed graph structure** and exports, not one ranked list. |
| **ChatGPT / RAG over PDFs** | Chat tools optimize fluent answers; SRG Lite optimizes **bibliographic neighborhood** with explicit anchors. Chat can hallucinate citations; SRG is only as good as APIs but is **grounded in fetched records** for ingested ids. |
| **Dedicated systematic review tools** | SRG Lite does not enforce PRISMA-style protocols or dual screening. |
| **Citation managers (Zotero, etc.)** | Managers excel at **personal libraries**; SRG Lite focuses on **exploratory graph from a query** (can complement, not replace). |

---

## 12. Product and engineering recommendations (optional roadmap hooks)

These are **not commitments**; they are sensible follow-ups if leadership wants higher precision on ambiguous classes.

1. **Query interpreter UI:** Suggest disambiguations (“Did you mean browser devtools vs program slicing?”) based on simple rules or a classifier.  
2. **Expand canonical packs** cautiously with versioning and tests (each pack is a product liability if wrong).  
3. **Session reproducibility bundle:** export seeds + code version + model ids for internal audits.  
4. **Pilot playbook:** 30-minute onboarding using [querying_guidelines.md](querying_guidelines.md) + two hands-on tasks (strong seed, weak seed).  
5. **Telemetry (if policy allows):** query length, share of DOI/arXiv vs free text, time-to-export—**privacy reviewed**.

---

## 13. Glossary

| Term | Meaning |
|------|---------|
| **Anchor / seed** | A work used to start graph expansion. |
| **OpenAlex work id** | Identifier like `W…` for a work in OpenAlex. |
| **Canonical pack** | Curated list of arXiv ids prepended for specific query patterns. |
| **api_search seed** | Seed attributed to search/discovery (vs uploaded PDF, vs discovered hops). |
| **Two-hop / broaden** | Mechanisms to deepen the candidate pool when seeds are sparse. |
| **Reading list** | Ordered list derived from graph + scoring; not a canonical syllabus. |

---

## 14. Summary verdict for leadership

**SRG Lite is a credible orientation and exploration layer** over scholarly APIs when users **anchor deliberately** and treat diagnostics as part of the workflow. Its main product risk is **user expectation mismatch**: people who want “always correct topical answers from three words” will be disappointed. Its main engineering risk is **external API behavior + partial canonical coverage**.

**Pilot SRG Lite** if your organization values **citation-structured maps** and can invest in **light user training** and **a small regression harness**. **Do not position it** as unsupervised systematic review or as a general enterprise search appliance without additional IR design.

---

*Document version: aligned with repository behavior and internal benchmarks as of the authoring pass; code evolves—validate critical claims against `scripts/verify_lite_anchor_queries.py` and the Lite UI on the target branch.*
