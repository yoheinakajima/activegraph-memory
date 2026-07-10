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
from activegraph_memory import MemoryRuntime

memory_runtime = MemoryRuntime.from_activegraph(
    activegraph_runtime,
    "quality",
    reasoning_model="your-reasoning-model",
    embedding_model="your-embedding-model",
)
```

Provider credentials remain in provider configuration. They are not copied into
memory objects, prompts, or telemetry.

Use `GraphMemoryRepository` when compiled memory and retrieval traces should be
materialized into the same graph. Pass a mapping from source-turn logical IDs to
Core source object IDs so claims, entities, events, states, and proofs receive
`memory_grounded_in` relations.

```python
repository.compile(
    turns=turns,
    claims=claims,
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
recompile it from source turns and claims, or reconstruct an application-level
index from graph objects. Corpus vectors can survive separately through
`SQLiteEmbeddingStore`, `GraphEmbeddingStore`, or the provider's cache.

## Standalone Agents

Standalone use does not require a running behavior loop:

- `compile_memory_index` builds the projection
- `MemoryRuntime.retrieve` executes the pipeline
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
- `reasoning_fail_open`

Every switch is enforced by `MemoryRuntime`; none is documentation-only.
