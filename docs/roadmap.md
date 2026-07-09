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

Status: implemented as pure helper/runtime code. `retrieve_memory` returns
`EvidenceBundle`, `CoverageReport`, confidence, epistemic status, and selected
evidence ids. A graph-visible behavior wrapper for retrieval is still pending.

## Phase 5 - Compiled Graph Query

- Compile entity-like refs, coarse category refs, and event rows from claims
- Attach quantities, temporal refs, predicates, event dates, claim ids, and source-turn ids
- Run deterministic count, sum, and chronological reducers before context packing
- Render computed graph-query rows into retrieval context

Status: implemented as the first deterministic pass. The taxonomy and predicate
rules are intentionally lightweight; richer extraction, typed relations, and
domain-specific schemas remain future work.

## Phase 6 - Temporal, Conflict, Supersession

- Temporal reference extraction and resolution
- Conflict and supersession detection
- Claim status updates
- Evidence-aware answer synthesis

Status: partially implemented. The compiler records claim temporal refs,
simple quantities, rough authority, lifecycle status, and lightweight
supersession metadata. The retriever renders normalized simple relative dates
and durations into claim headers. Full conflict detection and graph mutation
behaviors are still pending.

## Phase 7 - Benchmarks And Evals

- LongMemEval-style fixtures
- Query class evaluation
- Coverage calibration tests
- Regression suite for latest/current/final and negative questions

Status: initial LongMemEval integration complete via
`yoheinakajima/activegraph-longmemeval`.

Smoke result on LongMemEval-S, 50 frozen IDs:

```text
activegraph-memory-pack
run: agmem-fullarch2-smoke-20260709T014835Z__activegraph-memory-pack__s__smoke
overall accuracy:       0.94
task-averaged accuracy: 0.9634
abstention accuracy:    1.0
```

Full-500 validation should wait for persistent embedding caching because the
compiled-memory adapter currently replays claim/turn embedding work slowly.
