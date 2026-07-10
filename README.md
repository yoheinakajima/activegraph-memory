# activegraph-memory

`activegraph-memory` is the semantic, temporal, and epistemic memory layer for
[ActiveGraph](https://pypi.org/project/activegraph/). It compiles source-grounded
claims into queryable entities, canonical events, state histories, preferences,
quantities, temporal references, and position-preserving lists.

It does not replace `memory_gateway`. The gateway remains the low-level lifecycle
and backend seam:

```text
memory_candidate -> evaluation -> memory_item -> backend retrieval
```

This package adds the higher-level path:

```text
source log + accepted claims
  -> typed compiled projection
  -> multi-operator query IR
  -> hybrid candidate generation
  -> graph signal propagation
  -> deterministic operator execution
  -> proof and coverage checks
  -> provenance-preserving context
```

## Install

```bash
pip install activegraph-memory
```

For development:

```bash
python3 -m pip install -e ".[dev]"
pytest -q
```

The pack entry point is:

```toml
[project.entry-points."activegraph.packs"]
activegraph_memory = "activegraph_memory:pack"
```

## Runtime

Callers provide authoritative source turns and claims from an extractor,
connector, or accepted `memory_gateway` item. Compilation is deterministic and
does not require an API key.

```python
from activegraph_memory import (
    ExtractedClaimInput,
    MemoryRuntime,
    SourceTurn,
    compile_memory_index,
)

turn = SourceTurn(
    turn_id="session-1#0",
    session_id="session-1",
    session_date="2026-07-09",
    session_idx=0,
    turn_idx=0,
    role="user",
    content="I bought bike lights for $40 yesterday.",
    text="[Session session-1 (2026-07-09)] user: I bought bike lights for $40 yesterday.",
)
claim = ExtractedClaimInput(
    text="The user bought bike lights for $40 yesterday.",
    session_id="session-1",
    session_date="2026-07-09",
    session_idx=0,
    role="user",
    mentioned_turn_idxs=(0,),
)

index = compile_memory_index(turns=[turn], claims=[claim])
result = MemoryRuntime("balanced").retrieve(
    index,
    "How much did I spend on bike accessories?",
    question_date="2026-07-10",
)

print(result.context_text)
print(result.metadata["compiled_evidence"])
print(result.metadata["pipeline_telemetry"])
```

The compiled evidence packet is placed before raw provenance. A computed answer
is labeled verified only when its operator-specific proof obligations are met.
Otherwise the packet is tentative and lists missing requirements.

## Compiled Memory

`compile_memory_index` produces:

- canonical entities and aliases
- event mentions with modality, polarity, event time, and observation time
- canonical events with repeated-mention deduplication
- versioned state histories with supersession
- scoped preference and professional-profile evidence
- structured quantities with measure names and units
- normalized temporal references
- position-preserving list items

The compiler distinguishes actual, planned, hypothetical, and recommended
events. It also distinguishes event time from observation time. Source turns
remain authoritative; every compiled row retains claim and source IDs.

The query IR supports multiple operators and explicit proof requirements for
lookup; current, latest, and previous state; count, sum, and maximum; temporal
order and date delta; ordinal list lookup; negative existence; and personalized
recommendation.

Counts use item cardinality, not just row count. Aggregates scan typed categories
and compatible actions, deduplicate canonical events, and verify source coverage.
Temporal comparisons require one compatible dated event per operand. State
queries inspect the version history and supersession state.

## Retrieval Signals

Retrieval combines deterministic lexical scores with caller-supplied or
provider-backed embeddings across claims, source turns, entities, events,
states, and preferences.

Reciprocal-rank fusion prevents one score scale from dominating. Entity scores
are propagated through compiled entity-to-claim, entity-to-turn,
entity-to-event, and entity-to-state edges. Embeddings therefore affect graph
neighbors instead of remaining an isolated similarity list.

## Profiles

All optional reasoning stages use `off`, `fallback`, or `always`. `fallback`
uses the deterministic result first and calls the reasoner only when confidence
or proof completeness requires it.

| Profile | Context budget | Rounds | Entity/event embeddings | Classification | Strategy | Retrieval analysis | Packaging |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| `fast` | 2,500 | 1 | off | off | off | off | off |
| `balanced` | 4,000 | 2 | on | fallback | off | fallback | off |
| `quality` | 6,000 | 2 | on | fallback | fallback | always | fallback |
| `max_quality` | 10,000 | 3 | on | always | always | always | always |

Reasoning is optional even in a reasoning-enabled profile. Without a reasoner,
the runtime remains deterministic. Reasoner failures fail open by default and
are recorded in stage telemetry. Set `reasoning_fail_open=False` in a custom
`MemoryRuntimeProfile` when a failed control-plane call should fail the query.

Packaging reasoning may only prioritize or exclude IDs from the already selected
evidence set. It cannot write or rewrite memory text. The final context is always
rendered deterministically from known claims and sources.

Profiles are Pydantic models and can be copied safely:

```python
from activegraph_memory import MemoryRuntime, runtime_profile

profile = runtime_profile("balanced").model_copy(
    update={"token_budget": 10_000, "include_raw_sources": True}
)
runtime = MemoryRuntime(profile)
```

Pack settings can override individual reasoning stages through
`profile_from_settings(settings)`.

## ActiveGraph Persistence

`GraphMemoryRepository` materializes claims, entities, events, states,
preferences, list items, quantities, temporal references, proofs, and measured
retrieval stages as graph objects. Materialization uses stable keys, so replaying
the same compile or retrieval trace is idempotent.

```python
from activegraph import Graph, Runtime
from activegraph_memory import GraphMemoryRepository, MemoryRuntime

graph = Graph(run_id="memory")
activegraph_runtime = Runtime(graph, persist_to="sqlite:///memory-events.db")
repository = GraphMemoryRepository(graph, runtime=MemoryRuntime("balanced"))
repository.compile(turns=turns, claims=claims)
result = repository.retrieve("What changed?", query_id="memory-query-object-id")
```

Durability is provided by the configured ActiveGraph store. The source log can
be replayed after a crash to rebuild the same compiled objects. Retrieval proof
and telemetry objects make completed stages auditable.

Corpus embeddings can be process-local, SQLite-backed, or graph-backed:

```python
from activegraph_memory import GraphEmbeddingStore, MemoryRuntime

runtime = MemoryRuntime(
    "balanced",
    embedding_provider=provider,
    embedding_model="your-model",
    embedding_store=GraphEmbeddingStore(graph),
)
```

Use `SQLiteEmbeddingStore(".cache/memory-vectors.sqlite3")` when vectors should
remain outside graph state. Cache keys include model, field, subject ID, and a
hash of the embedded text, so changed text cannot silently reuse a stale vector.

## ActiveGraph Providers

`MemoryRuntime.from_activegraph(...)` binds to ActiveGraph's configured LLM and
embedding provider seams. The provider remains responsible for API credentials.
Secrets are never written into graph objects, telemetry, fixtures, or prompts.

## Benchmarking

The benchmark API records end-to-end mean, p50, p95, cold, and warm latency;
context tokens; input and output tokens; estimated provider cost; proof-complete
rate; optional application quality; and per-stage latency, tokens, cost, cache
state, and candidate counts.

Run the committed offline fixture:

```bash
python3.11 -m activegraph_memory.benchmark_cli \
  --input examples/benchmark_fixture.json \
  --profiles fast,balanced,quality,max_quality \
  --repetitions 100 \
  --hash-embeddings \
  --score-expected \
  --format markdown
```

See [docs/benchmark-results.md](docs/benchmark-results.md) for the latest
reproducible control-plane result and its limitations. For live embeddings or
reasoning, use `benchmark_profiles(..., runtime_factory=...)` so the report
includes provider-reported tokens and caller-supplied pricing.

## Pack Surface

The pack registers 21 object types, 14 relation types, two deterministic
behaviors, and three tools. The main additions are:

- `memory_entity`, `memory_event`, `memory_state`, `memory_preference`
- `memory_query_analysis`, `memory_proof`, `memory_retrieval_stage`
- `memory_embedding`, `memory_benchmark`
- `memory_about`, `memory_grounded_in`, `memory_version_of`
- `memory_stage_for`, `memory_proves`

The registered tools are `plan_memory_query`, `analyze_memory_query`, and
`list_memory_profiles`.

## Gateway Boundary

`activegraph-memory` composes with Core and `memory_gateway`:

```text
source -> observation -> memory_candidate -> evaluation -> memory_item
                                                   |
memory_query -> query_analysis -> retrieval_plan -> memory_retrieval_request
       |
       -> retrieval stages -> evidence bundle -> proof -> memory answer
```

Core objects and provenance relations are not redefined. Candidate evaluation
still precedes durable memory. Backend-specific retrieval stays behind the
gateway protocol.

## Validation

```bash
python3 -m pip install -e ".[dev]"
pytest -q
```

Tests are deterministic and run without API keys or network access. LongMemEval
integration and large run artifacts live in
[`yoheinakajima/activegraph-longmemeval`](https://github.com/yoheinakajima/activegraph-longmemeval).
