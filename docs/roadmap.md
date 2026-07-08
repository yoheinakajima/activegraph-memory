# Roadmap

## Phase 1 - Pack Foundation

- Package metadata and entry point
- Object and relation types
- Settings
- Docs and tests
- Offline deterministic helpers

Status: implemented in 0.1.

## Phase 2 - Deterministic Planning

- `memory_query -> retrieval_plan`
- Query classes for lookup, latest/current/final, negative existence, aggregate, temporal, preference, multi-hop, and decision reconstruction
- Risk flags and required guarantees

Status: implemented as pure helpers and `memory_query_planner`.

## Phase 3 - Gateway Integration

- Convert retrieval plans into graph-visible `memory_retrieval_request` objects
- Wrap `memory_retrieval` results as `evidence_bundle` objects
- Keep retrieval auditable through graph state

Status: helper functions exist; runtime behavior still needs full integration tests with `memory_gateway`.

## Phase 4 - Coverage And Confidence

- Coverage report behavior
- Confidence vector calculation
- Answer status selection

Status: helper functions exist; graph behavior pending.

## Phase 5 - Temporal, Conflict, Supersession

- Temporal reference extraction and resolution
- Conflict and supersession detection
- Claim status updates
- Evidence-aware answer synthesis

Status: only a small deterministic duration helper exists.

## Phase 6 - Benchmarks And Evals

- LongMemEval-style fixtures
- Query class evaluation
- Coverage calibration tests
- Regression suite for latest/current/final and negative questions

Status: planned.
