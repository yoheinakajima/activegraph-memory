# Changelog

## Unreleased — ADR 0026 steps 5-7: memory onto the shared annotation layer

- The pack form consumes the shared annotation layer: new behaviors
  `memory_ingest_shared_annotation` (mints memory claims from
  `semantic_annotation` records, consuming canonical `entity` ids, zero
  provider calls, idempotent by shared annotation id) and
  `memory_deprecate_entity` (maps legacy `memory_entity` objects to canonical
  ids — never dropped). Gated by `consume_shared_extraction` (default off).
- `memory_entity` deprecated with `status` / `canonical_entity_id` fields;
  existing data is mapped or marked superseded.
- Extraction-run coverage in proof completeness: `audit_source_coverage` gains
  an `extraction_run_source_ids` plane (read from `memory_ingestion_stage`
  records) folded into `confidence`; count/sum/temporal/absence can no longer
  certify an answer over sources no extraction run covered. Backward compatible
  — `None` (claims supplied directly) reproduces prior behavior exactly.
- The standalone extractor is kept only as an explicit
  `CompatibilityMemoryExtractor`; inert (zero provider calls) when shared
  extraction is configured. `claims_from_shared_annotations` bridges shared
  annotations into the standalone compile path.
- The pack now requires `memory_gateway`, `activity_normalizer`,
  `semantic_extraction`, and `entity` (previously optional).

## 0.4.0

- Added query-bounded source coverage audits that separately measure relevant
  source extraction, typed compilation, compiled selection, and reader-visible
  recovery. Raw recovery evidence never certifies a computed candidate.
- Added raw-source recovery for incomplete aggregate and preference projections,
  with bounded excerpts and direct provenance in the final evidence set.
- Added explicit evidence slots for aggregate events, temporal operands,
  positive preferences, negative constraints, and recovered sources.
- Added polarity- and scope-aware preference selection plus preference-specific
  extraction, compilation, and selection coverage.
- Added configurable per-operator confidence thresholds and held-out calibration
  helpers that fit the broadest candidate gate meeting a target precision.
- Added benchmark fields for mean coverage confidence, recovery-source rate,
  and evidence-slot count.
- Added a non-benchmark application fixture covering finance, project state,
  scheduling, travel constraints, and completed-versus-planned agent work.
- Hardened plan exclusion in complete-set coverage, inflected preference
  detection, source-grounded temporal proof fields, and aggregate core terms.

## 0.3.0

- Renamed compiled answer labels from `Verified candidate` / `Tentative
  candidate` to `Proof-complete candidate` / `Incomplete candidate`.
- Clarified that proof completion certifies the presence of operator-required
  evidence fields, not semantic answer correctness; readers must still verify
  candidates against cited sources.
- Added provider-neutral typed extraction with deterministic and ActiveGraph LLM
  backends. Structured entities, predicates, modality, polarity, event time,
  quantities, state identity, preference scope, and confidence now drive the
  compiler instead of being discarded and re-inferred from text.
- Added replayable `memory_source_turn`, `memory_conflict`,
  `memory_ingestion_stage`, and `memory_retrieval_assessment` graph objects.
- Added graph-to-index reconstruction, repository restart/load, append with
  deterministic stable projection refresh, and graph-visible conflict links.
- Added deterministic sufficiency assessment and confidence-driven targeted
  retrieval rounds with explicit stop reasons and query traces.
- Added calibrated candidate rendering, per-profile source budget ratios, and
  bounded negative-existence certificates.
- Added cumulative reasoning stop thresholds for calls, tokens,
  provider-reported cost, and latency, checked before each optional stage.
- Added deterministic option matrices, reasoning-stage ablations, and ingestion
  benchmarks with latency, tokens, cost, graph size, rounds, sufficiency, proof,
  and candidate-render metrics.
- Wired pack settings into the repository facade so extraction, temporal
  normalization, conflict detection, profile overrides, and provider models
  change runtime behavior rather than remaining declarative only.
- Added bounded long-history extraction batches with aggregate and per-batch
  usage telemetry, plus restart-safe graph-visible ingestion runs.

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
