# Roadmap

## Implemented In 0.2

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

## Near-Term Work

### Extraction Contracts

- define a provider-neutral typed extraction response for events, entities,
  state properties, quantities, and time basis
- calibrate extraction coverage against raw-source scans
- add incremental compilation when one source event is appended
- connect accepted `memory_gateway` items without adapter-specific conversion

### Belief Maintenance

- replace topic-key supersession with typed subject/property identity
- add explicit contradiction objects and resolution policies
- separate source authority, extraction confidence, and belief confidence
- support bitemporal valid-time and transaction-time queries

### Query Execution

- add negative-existence executor with a bounded search certificate
- add relational multi-hop joins over entities and episodes
- add units and currency conversion through an explicit conversion provider
- add query-aware list identity for multiple similar lists from one source
- expose a compiled-query explanation format suitable for UI inspection

### Durability

- reconstruct `MemoryIndex` directly from materialized graph objects
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
