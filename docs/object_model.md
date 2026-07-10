# Object Model

## Source And Belief

### `memory_source_turn`

A replayable projection of one immutable source turn with logical turn/session
identity, role, observation date, raw content, rendered text, and optional link
to the authoritative Core source object.

### `memory_claim`

A source-grounded fact, preference, instruction, decision, procedure, or
summary. It carries subject, scope, status, authority, separate extraction and
belief confidence, validity, observation time, source-turn IDs, and observation
IDs.

### `memory_episode`

A coherent grouping of claims, sources, entities, and events.

### `temporal_ref`

A temporal phrase with normalized start/end, anchor, resolution method, and
confidence.

### `quantity_claim`

A measured property with value, unit, exactness, source text, and confidence.

## Compiled Projection

### `memory_entity`

A canonical name, kind, aliases, and source provenance. Entities are retrieval
and graph-traversal nodes, not replacements for source claims.

### `memory_event`

A canonical event containing action predicate, entities, categories, modality,
polarity, event time, source claims, source turns, confidence, quantities, and
deduplication metadata.

### `memory_state`

One version of a subject-property state. State chains use
`memory_version_of` and `memory_supersedes`.

### `memory_preference`

Positive or negative preference/profile evidence with scope terms, explicitness,
observation time, confidence, and provenance.

### `memory_list_item`

A list ID, preserved ordinal position, item text, source role, and source ID.

### `memory_embedding`

An optional persistent vector with model, field/subject identity, text hash,
dimensions, and metadata. It is created only when `GraphEmbeddingStore` is used.

### `memory_conflict`

An explicit unresolved or resolved incompatibility with claim IDs, optional
state identity, reason, confidence, status, and source-turn provenance.

## Query And Proof

### `memory_ingestion_stage`

A replayable extraction run with source-turn scope, extractor/model identity,
fact count, token usage, provider cost, latency, cache state, and batch metadata.

### `memory_query`

A graph-visible memory question with subject, time anchor, requested guarantees,
top-k, and metadata.

### `memory_query_analysis`

The deterministic query IR: operators, operands, entity terms, proof
requirements, confidence, and resolved query metadata.

### `retrieval_plan`

Strategies, source targets, freshness/coverage requirements, risk flags, and
steps for a query.

### `memory_retrieval_stage`

One measured pipeline stage with implementation, latency, token usage, cost,
candidate counts, cache status, and decisions.

### `memory_retrieval_assessment`

One deterministic stop/continue decision with per-dimension confidence, missing
proof requirements, conflicts, reasons, and targeted next queries.

### `evidence_bundle`

Claims, memory items, sources, conflicts, and compiled event IDs selected for a
query.

### `coverage_report`

Searched and unsearched scopes, boundedness, missing data, adequate and
inadequate answer classes, and coverage confidence.

### `memory_proof`

Operator, candidate answer, required/satisfied/missing checks, selected evidence
IDs, completion status, and confidence.

### `memory_answer`

Answer text, epistemic status, confidence vector, evidence reference, caveats,
and missing data.

### `memory_benchmark`

Aggregated latency, context, rounds, sufficiency, candidate-rendering, reasoner
calls, token usage, cost, proof rate, quality, and benchmark metadata.

## Governance

### `memory_policy`

Named policy settings for planning, retrieval, or answering.

### `memory_evaluation`

Human or automated judgment of a memory query, answer, or retrieval process.

## Relations

| Relation | Meaning |
| --- | --- |
| `memory_supports` | Evidence or one claim supports a claim or answer. |
| `memory_contradicts` | Evidence or one claim conflicts with a claim or answer. |
| `memory_supersedes` | A newer memory replaces an older version. |
| `memory_has_temporal_ref` | A claim/query/plan/answer has a temporal reference. |
| `memory_has_quantity` | A claim or answer has a structured quantity. |
| `memory_retrieved_for` | A plan or evidence object serves a query. |
| `memory_used_as_evidence` | A source or memory was used in evidence/answering. |
| `memory_has_coverage` | A query artifact has a coverage report. |
| `memory_governed_by_policy` | A memory object is governed by a policy. |
| `memory_about` | A claim/event/state/preference concerns an entity. |
| `memory_grounded_in` | A compiled object or proof traces to claims/sources. |
| `memory_version_of` | A state/event/episode is another version of one identity. |
| `memory_stage_for` | Analysis, stage, proof, or benchmark belongs to a query. |
| `memory_proves` | A proof supports a query, evidence bundle, or answer. |

Core provenance relations such as `derived_from`, `grounds`, and `proposes` are
not redefined.

## Runtime Dataclasses

Before graph materialization, the same semantics are represented by immutable
compiled records:

- `MemoryEntityRecord`
- `EventMentionRecord`
- `CanonicalEventRecord`
- `StateVersionRecord`
- `PreferenceEvidenceRecord`
- `ListItemRecord`
- `MemoryConflictRecord`

`CompiledMemoryProjection` indexes these records by logical entity, event, and
conflict IDs. All rows retain source claim and source turn IDs.

`CompiledEvidence.evidence_slots` gives aggregate events, temporal operands,
positive preferences, negative constraints, and raw recovery sources stable
roles in the reader packet. `SourceCoverageAudit` separately measures relevant
source extraction, compilation, computed selection, and reader-visible
recovery. These are runtime contracts rather than new authoritative graph
objects; retrieval assessments and proofs persist their serialized values.

## Confidence

Answer confidence is a vector, not one average:

```json
{
  "relevance": 0.91,
  "entity_match": 0.85,
  "authority": 0.72,
  "freshness": 0.61,
  "coverage": 0.48,
  "consistency": 0.80,
  "extraction": 0.93,
  "reasoning": 0.70
}
```

Required dimensions are combined conservatively. A high relevance score cannot
hide missing coverage or unresolved contradiction.
