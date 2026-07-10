# Roadmap

## Implemented Through 0.3

- deterministic source-turn and claim compilation
- typed entities, event mentions, canonical events, state histories,
  preferences, quantities, temporal references, and list items
- actual/planned/hypothetical/recommendation modality separation
- multi-operator query IR and proof requirements
- fielded lexical and embedding retrieval
- reciprocal-rank fusion and entity-to-neighbor graph propagation
- executable snapshot, count, sum, maximum, temporal-order, date-delta,
  ordinal, preference, and lookup paths
- category/action-bounded aggregates with item cardinality and source coverage
- profile-driven optional reasoning at four independent stages
- fail-open reasoning telemetry and validated evidence-ID packaging
- 2,500, 4,000, 6,000, and 10,000-token built-in profiles
- graph-visible compiled objects, proofs, and measured stages
- SQLite and ActiveGraph embedding persistence
- deterministic profile benchmark API and CLI
- LongMemEval adapter with resumable retrieval artifacts in the companion repo
- provider-neutral typed extraction plus deterministic raw-turn fallback
- typed compiler overrides for entities, predicates, modality, event time,
  quantities, state identity, preference scope, and confidence
- replayable source-turn, conflict, and retrieval-assessment graph objects
- replayable ingestion-stage objects with source scope and provider usage
- graph-to-index restart reconstruction and append with stable projection refresh
- deterministic sufficiency assessment and confidence-driven targeted rounds
- calibrated candidate rendering and explicit source-context budget ratios
- cumulative reasoning call/token/cost/latency budgets
- bounded negative-existence certificates
- deterministic option matrices, reasoning ablations, and ingestion benchmarks

## Near-Term Work

### Extraction Quality

- calibrate extraction coverage against raw-source scans
- connect accepted `memory_gateway` items without adapter-specific conversion

### Belief Maintenance

- replace topic-key supersession with typed subject/property identity
- add configurable conflict resolution policies and authority-weighted resolution
- calibrate source authority, extraction confidence, and belief confidence
- support bitemporal valid-time and transaction-time queries

### Query Execution

- add relational multi-hop joins over entities and episodes
- add units and currency conversion through an explicit conversion provider
- add query-aware list identity for multiple similar lists from one source
- expose a compiled-query explanation format suitable for UI inspection

### Durability

- add incremental graph embedding invalidation for changed source text
- add retrieval-run identities for concurrent attempts of the same query
- document retention and deletion semantics for graph-stored vectors

### Evaluation

- calibrate proof-complete against actual answer correctness
- publish live profile speed, token, and cost tables on a fixed corpus
- maintain LongMemEval smoke and full-500 regressions
- add non-benchmark application traces for current state, finance, scheduling,
  preferences, and project history

## Out Of Scope

- replacing Core source, observation, candidate, or evaluation objects
- bypassing `memory_gateway` for durable memory writes
- treating vector similarity as proof
- storing provider credentials in graph state
- claiming benchmark improvement from the synthetic profile fixture
