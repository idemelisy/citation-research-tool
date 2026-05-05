# Semantic Research Graph (SRG)

Semantic Research Graph ingests records from scholarly sources, normalizes and validates them,
builds citation-based semantic relationships, and returns discovery insights with confidence-aware edges.

## Features
- Canonical schema with field-level provenance and confidence tiers
- Source ingestion clients for OpenAlex, ArXiv, Crossref (+ Semantic Scholar scaffold)
- Deduplication and conflict resolution with trust hierarchy
- Graph construction, co-citation, bibliographic coupling, and citation context fallback
- Domain-aware discovery (key paper ranking, trend/gap snapshots)
- Manual feedback loop for merge/reject corrections
- Confidence-aware visualization primitives (solid/light/dashed edges)

## Quick Start
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
python -m srg.app
python -m srg.app --mode ui
```

## Project Structure
- `src/srg/schema.py`: canonical entities and contracts
- `src/srg/ingestion.py`: API clients and orchestrator
- `src/srg/validation.py`: dedup/conflict engine
- `src/srg/graph.py`: graph and semantic relation metrics
- `src/srg/discovery.py`: recommendation and trend logic
- `src/srg/feedback.py`: active-learning feedback store
- `src/srg/visualization.py`: confidence-aware edge styling
- `src/srg/app.py`: end-to-end demo pipeline
