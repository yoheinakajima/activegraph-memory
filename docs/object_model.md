# Object Model

## Objects

### `memory_claim`

A source-grounded belief with scope, status, authority, confidence, and temporal validity.

Use it for facts, preferences, instructions, decisions, procedures, summaries, and uncertain claims.

### `memory_episode`

A coherent bundle of related sources, claims, entities, and events. Episodes make longer histories easier to retrieve without flattening everything into one memory item.

### `memory_query`

A graph-visible memory question. The `query_type` may be supplied or left as `unknown` for deterministic inference.

### `retrieval_plan`

A strategy object generated from a `memory_query`. It records strategies, target sources, coverage/freshness requirements, risk flags, and steps.

### `coverage_report`

Records searched scopes, unsearched scopes, missing data, boundedness, and what the answer can or cannot safely claim.

### `evidence_bundle`

Groups claims, memory items, sources, and conflicts gathered for a query.

### `memory_answer`

An answer plus epistemic status, confidence vector, evidence bundle reference, caveats, and missing data.

### `temporal_ref`

A temporal phrase and its resolved interval when deterministic resolution is possible.

### `quantity_claim`

A numeric claim with owner, property, value, unit, exactness, and source text.

### `memory_policy`

Policy data for governing memory planning, retrieval, and answering.

### `memory_evaluation`

Human or automated judgment about a memory query, answer, or retrieval process.

## Relations

This pack uses memory-specific relation names:

- `memory_supports`
- `memory_contradicts`
- `memory_supersedes`
- `memory_has_temporal_ref`
- `memory_has_quantity`
- `memory_retrieved_for`
- `memory_used_as_evidence`
- `memory_has_coverage`
- `memory_governed_by_policy`

Core provenance relations such as `derived_from`, `grounds`, and `proposes` are intentionally not redefined here.

## Confidence Vector

Memory answers should prefer a vector over one blended score:

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

The helper in `activegraph_memory.scoring` computes an overall value by taking the minimum of required dimensions, not the average.
