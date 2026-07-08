# AGENTS.md

## Mission

This repository implements `activegraph-memory`, a semantic and epistemic memory pack for ActiveGraph.

It is not a replacement for `packs.memory_gateway`. The existing gateway remains the low-level memory lifecycle and backend seam. This repo adds higher-level memory semantics: claims, episodes, temporal validity, contradiction/supersession, retrieval planning, coverage reports, confidence vectors, and evidence-backed answers.

## ActiveGraph Invariants

- Event-first: meaningful work should be represented as graph-visible objects and events.
- Candidate-first memory: do not write durable memory directly when the correct path is `memory_candidate -> evaluation -> memory_item`.
- Core stays small: never redefine `source`, `observation`, `memory_candidate`, `evaluation`, or Core provenance relations.
- Compose through graph state, not hidden direct calls between packs.
- Keep behavior outputs auditable: behavior names, source ids, evidence ids, and rationale fields matter.
- Tests must run deterministically with no API key and no live network.
- Secrets must never enter graph state, fixtures, tests, prompts, or docs examples.
- Prefer small, high-confidence changes over broad speculative architecture.

## Relationship To activegraph-packs

Assume `activegraph-packs` contains the reference pack conventions. Inspect it when available, especially:

- `packs/core`
- `packs/memory_gateway`
- `packs/_template`
- `docs/concepts.md`
- `docs/long-term-memory.md`
- `CONTRIBUTING.md`

Do not refactor `activegraph-packs` from this repo.

## Package Conventions

- Distribution name: `activegraph-memory`
- Python package: `activegraph_memory`
- Pack entry point: `activegraph_memory = "activegraph_memory:pack"`
- Pack name: `activegraph_memory`
- Python: 3.11+
- Pydantic: v2
- Tests: pytest

## Required Commands

After changes, run:

```bash
pip install -e ".[dev]"
pytest -q
```

Use `python3 -m pip` if this environment does not provide `python` or `pip` shims.
