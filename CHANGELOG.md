# Changelog

## 0.2.0

- Added profile-driven `MemoryRuntime` orchestration with `fast`, `balanced`,
  `quality`, and `max_quality` cost/latency profiles.
- Added optional reasoning seams for query classification, retrieval strategy,
  retrieval analysis, and context packaging, with per-stage off/fallback/always
  policies and provider-reported usage telemetry.
- Added multi-operator query IR with explicit operands, time windows, answer
  types, and proof obligations.
- Added a typed compiled projection for canonical entities and events, state
  histories, scoped preference evidence, and position-preserving list items.
- Added fielded embedding retrieval through ActiveGraph's embedding-provider
  seam, reciprocal-rank fusion, targeted retrieval rounds, and session-diverse
  source packing.
- Added proof-oriented aggregate, temporal, state, preference, ordinal, and
  lookup executors plus compact compiled evidence packets.
- Added graph materialization for compiled memory, proof objects, and measured
  retrieval stages, along with a `GraphMemoryRepository` facade.
- Added idempotent graph materialization for quantities and temporal refs plus
  claim/source and proof/evidence provenance relations.
- Added persistent `SQLiteEmbeddingStore` and graph-visible
  `GraphEmbeddingStore` options for compiled corpus vectors.
- Added entity-embedding propagation through compiled claim, turn, event, and
  state edges.
- Added category/action-bounded aggregate scans, source-coverage proofs, item
  cardinality, multi-quantity measure matching, and repeated-event merging.
- Constrained optional packaging reasoning to known evidence ids and added
  schema validation plus fail-open audit telemetry to every reasoning stage.
- Added reusable cold/warm latency, token, cost, proof-rate, and quality
  benchmarking helpers plus a JSON-in CLI and reproducible fixture.
- Hardened quantity extraction against product model numbers, ordinals, dates,
  and years, and added common relative-day/weekday temporal resolution.

## 0.1.0

- Initial ActiveGraph semantic memory pack.
- Added Pydantic schemas and ActiveGraph object/relation type declarations.
- Added deterministic query classification and retrieval planning.
- Added compiled entity/category/event projection for graph-query reducers.
- Added deterministic count, sum, and chronological reducers during retrieval.
- Increased the standalone retrieval default budget to 10000 rough tokens.
- Added temporal order comparison over named operands, including explicit
  insufficient-evidence packets when a comparison operand is missing.
- Added month-name date normalization for dates like "February 25th" anchored
  to the source/session year.
- Added compact preference/advice observation packets for recommendation-style
  queries, grounded in user claims and source ids.
- Added concept expansion for device/accessory and business-milestone queries
  so retrieval can bridge wording gaps without benchmark-specific rules.
- Added a compact near-date source packet for relative-date lookups when graph
  context is weak or unavailable.
- Added exact assistant-source recall routing for questions about what the
  assistant previously said, listed, recommended, or provided.
- Render low-confidence graph reducers as evidence rows without an authoritative
  computed answer candidate.
- Hardened graph-query matching for negated events, phrase punctuation,
  comma/word quantities, two-date windows, repeated event counts, and tight
  token budgets.
- Hardened temporal sequence planning so "earliest to latest" order questions
  are not treated as latest/current queries, and sequence "order" is not
  mistaken for purchase/order events.
- Added coverage and confidence helpers.
- Added graph-visible `memory_query_planner` behavior.
- Added gateway adapter helpers for creating `memory_retrieval_request` data.
- Added docs, fixtures, examples, and offline tests.
