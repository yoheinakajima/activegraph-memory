# Architecture

## Boundaries

ActiveGraph Core remains the event-sourced substrate:

```text
source -> observation -> memory_candidate -> evaluation
```

`memory_gateway` remains the durable memory lifecycle and backend seam:

```text
accepted memory_candidate -> memory_item -> backend retrieval
```

`activegraph-memory` compiles accepted, source-grounded information into a
typed query projection. It does not write around candidate evaluation and does
not redefine Core object or provenance types.

## Data Planes

The system has three explicit data planes.

1. The source plane contains the immutable event log, replayable source-turn
   projections, observations, and accepted claims.
2. The compiled plane contains derived entities, event mentions, canonical
   events, state versions, preferences, quantities, temporal references, list
   items, and unresolved conflicts.
3. The query plane contains query analysis, retrieval plans, measured stages,
   evidence, coverage, deterministic sufficiency assessments, and proof objects.

The compiled and query planes are replayable projections. They are not more
authoritative than their sources.

## Compile Pipeline

```text
source turns + accepted claims
  -> typed extractor or lossless deterministic fallback
  -> provenance anchoring
  -> entity and category extraction
  -> quantity and temporal parsing
  -> modality and polarity classification
  -> event mention compilation
  -> canonical event deduplication
  -> state history and supersession compilation
  -> contradiction compilation
  -> preference/profile facets
  -> list position extraction
```

Canonical event identity uses compatible actions, category overlap, entity
overlap, quantities, event time, observation time, and semantic token overlap.
Two explicitly dated events at different times remain distinct. A later
restatement with only observation time may merge into an earlier explicitly
dated event when the object and quantity evidence agree.

## Query Pipeline

```text
memory_query
  -> deterministic multi-operator query IR
  -> optional classification reasoning
  -> deterministic retrieval strategy
  -> optional strategy reasoning
  -> lexical + fielded embedding candidates
  -> reciprocal-rank fusion
  -> entity-to-neighbor graph propagation
  -> typed operator executor
  -> proof requirement evaluation
  -> deterministic confidence/conflict/sufficiency assessment
  -> targeted retrieval against explicit missing requirements
  -> optional reasoned sufficiency analysis
  -> deterministic source packaging
  -> optional evidence-ID prioritization
  -> evidence context + telemetry
```

Classification, strategy, analysis, and packaging reasoning are independent.
Each can be `off`, `fallback`, or `always`. Reasoning is a control-plane input,
not a new evidence source. Its output is schema-validated, measured, and stored
in stage telemetry. Invalid or failed outputs fail open by default.

Each profile also has cumulative reasoning stop thresholds for calls,
input/output tokens, provider-reported cost, and latency. They are checked before
each optional call; because final provider usage is not knowable in advance, one
call may cross a threshold and later calls are then skipped. Deterministic
assessment and execution continue without a reasoner.

Packaging reasoning cannot generate context text. It may only select IDs from
the existing evidence set. Context rendering remains deterministic.

## Operator Proofs

Each operator has different correctness requirements.

| Operator | Required checks |
| --- | --- |
| lookup | source provenance, entity compatibility |
| count/sum/max | bounded candidate set, canonical deduplication, source coverage |
| current/latest/previous | state history, time cutoff, supersession |
| order/date delta | all operands, compatible event semantics, resolved event times |
| ordinal | list identity, preserved position, source role |
| recommendation | preference scope, target-domain compatibility, constraints |
| negative existence | bounded candidate set, source coverage, absence certificate |

A vector score cannot satisfy these requirements by itself. Embeddings generate
and rank candidates; typed execution and graph state determine whether a proof
is complete.

## Graph Use

When materialized, ActiveGraph contains:

- `memory_source_turn` replayable projections grounded in Core sources
- `memory_claim` grounded in source-turn and Core source objects
- `memory_entity` linked with `memory_about`
- `memory_event` grounded in claims and sources
- `memory_state` chains linked by `memory_version_of` and `memory_supersedes`
- `quantity_claim` and `temporal_ref` linked to claims
- `memory_conflict` linked to incompatible claims
- `memory_query_analysis`, `memory_retrieval_assessment`,
  `memory_retrieval_stage`, and `memory_proof`
- `memory_ingestion_stage` with restart-safe extraction usage and source scope
- optional `memory_embedding` objects

Stable logical keys make compilation and trace replay idempotent. ActiveGraph's
event store provides crash durability. `GraphMemoryRepository.load()` rebuilds
the Python `MemoryIndex` from graph-visible source turns and claims after a
restart. `append()` refreshes stable compiled objects and state histories. The
recovery boundary is the committed graph write; an interrupted provider call is
retried, while committed ingestion stages and facts replay idempotently.

## Embeddings

Fielded vectors are generated for claims, turns, entities, events, states, and
preferences. Entity and event vectors include structured type information, not
only display text. Entity similarity propagates to graph neighbors before
operator execution.

Storage is configurable:

- process memory for short-lived tasks
- `SQLiteEmbeddingStore` for standalone persistence
- `GraphEmbeddingStore` for graph-visible vectors
- a caller-supplied provider cache

Embedding keys include model, field, logical subject ID, and text hash. Query
vectors are not persisted by these stores; compiled corpus vectors are.

## Cost And Telemetry

Every stage records implementation, duration, input/output tokens, cost,
candidate counts, cache state, and stage-specific metadata. Benchmarks report
cold/warm latency, ingestion graph size, rounds, sufficiency, proof completion,
candidate rendering, reasoning calls, and per-stage invocation counts.

Costs are provider-reported for reasoning and caller-estimated for embeddings.
The runtime does not guess a price when none is supplied.
