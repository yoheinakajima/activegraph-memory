# Integration Notes

## memory_gateway

Use `memory_gateway` for candidate evaluation, durable memory items, and backend
retrieval. `activegraph-memory` accepts the resulting source-grounded claims and
adds semantics above that boundary.

```python
from activegraph_memory.gateway import build_memory_retrieval_request
from activegraph_memory.object_types import MemoryQuery
from activegraph_memory.planner import plan_query

query = MemoryQuery(query="Did we ever agree to exclusivity?")
plan = plan_query(query, query_id="q1")
request_data = build_memory_retrieval_request(query, plan)
```

Write `request_data` as a `memory_retrieval_request` when `memory_gateway` is
loaded. Do not call a backend invisibly from a behavior.

## ActiveGraph Runtime

Bind optional providers through the ActiveGraph runtime:

```python
from activegraph_memory import GraphMemoryRepository

repository = GraphMemoryRepository.from_activegraph(
    activegraph_runtime,
    profile="quality",
    extraction_model="your-extraction-model",
    reasoning_model="your-reasoning-model",
    embedding_model="your-embedding-model",
)
```

Provider credentials remain in provider configuration. They are not copied into
memory objects, prompts, or telemetry.

The repository accepts raw turns, external accepted claims, or both. When
claims are omitted it uses its configured typed extractor, or the lossless
deterministic fallback when no extraction model is configured. Pass a mapping
from source-turn logical IDs to Core source object IDs so projections retain
their authoritative grounding.

```python
repository.compile(
    turns=turns,
    source_object_ids={"session-1#0": source_object.id},
)
```

For proof and stage relations, `query_id` should be the graph object ID of a
`memory_query`. External query keys are accepted, but graph relations to a
missing query object are intentionally skipped.

## Persistence And Resume

ActiveGraph event-store persistence records every graph mutation. Replaying a
run rebuilds the materialized projection. `materialize_memory_index` and
`materialize_retrieval_trace` use stable keys, so replaying completed work does
not create duplicate objects or relations.

The in-process `MemoryIndex` is not itself a database. After a process crash,
construct a new repository over the replayed graph and call `load()`. New source
turns can then be added with `append()`. Stable-key upserts patch changed state
histories without duplicating logical objects.

```python
restarted = GraphMemoryRepository(replayed_graph, runtime=memory_runtime)
restarted.load()
restarted.append(turns=new_turns, claims=new_claims)
```

## Standalone Agents

Standalone use does not require a running behavior loop:

- `compile_memory_index` builds the projection
- `extract_claim_inputs` provides deterministic or typed provider ingestion
- `MemoryRuntime.retrieve` executes the pipeline
- `GraphMemoryRepository.load` restores graph-persisted memory
- `SQLiteEmbeddingStore` persists vectors
- `benchmark_profiles` measures profile behavior

The package still depends on ActiveGraph because schemas, providers, graph
materialization, and pack declarations use ActiveGraph APIs.

## Custom Providers

Embedding providers implement ActiveGraph's provider signature:

```python
provider.embed(texts=["..."], model="model-name")
```

Reasoning providers implement:

```python
response = reasoner.reason(request)
```

`ReasoningResponse` carries structured data, model, token usage, cost, latency,
cache state, and metadata. Custom outputs are validated against the stage schema
before they can affect retrieval.

Extraction providers implement `MemoryExtractor.extract(turns)` and return a
`MemoryExtractionResult`. `CallableMemoryExtractor` adapts existing functions.
`ActiveGraphLLMMemoryExtractor` uses the same provider contract as ActiveGraph
and bounds long-history requests with configurable turn and character batch
limits. Usage is aggregated while each fact retains immutable source-turn IDs.

## Custom Profiles

Use a copied built-in profile or construct `MemoryRuntimeProfile` directly.
Important switches include:

- `use_embeddings`
- `embed_entities`
- `embed_events`
- `use_compiled_projection`
- `use_rank_fusion`
- `use_diversity_selection`
- `include_raw_sources`
- `compact_context`
- `adaptive_retrieval`
- `min_sufficiency_confidence`
- `source_budget_ratio`
- `max_packet_rows`
- `candidate_answer_mode`
- `reasoning_budget`
- `reasoning_fail_open`

Every switch is enforced by `MemoryRuntime`; none is documentation-only.

## Reader Contract

`MemoryRuntime.retrieve()` may prepend a `[compiled-memory: ...]` proof packet.
Consumers should preserve its labels when passing context to an answer model:

- `Proof-complete candidate` means the executor found every required evidence
  field. It does not certify semantic answer correctness.
- `Incomplete candidate` is a ranking aid with known missing proof
  requirements.
- Every candidate must be checked against cited rows and raw sources. Callers
  may adopt stronger trust policies only after calibrating each operator on
  representative data.
- `temporal_distance_days` is an accepted distance under the query's explicit
  tolerance. For approximate relative-time language, semantic fit inside that
  tolerance outranks calendar-day equality by itself.
- Under `candidate_answer_mode="calibrated"`, a candidate is omitted unless
  proof, deterministic sufficiency, execution confidence, and conflict checks
  all pass the profile threshold. Evidence rows remain available either way.

This contract is provider-neutral. It does not ask the answer model to invent
facts, and every candidate remains traceable to claim and source-turn ids.
