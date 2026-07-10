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
source log + accepted memory items
  -> typed extraction or lossless deterministic fallback
  -> replayable ActiveGraph source/claim projection
  -> entities + events + state histories + preferences + conflicts
  -> multi-operator query IR
  -> hybrid candidate generation
  -> graph signal propagation
  -> deterministic operator execution
  -> extraction/compilation/selection coverage audit
  -> confidence/proof assessment
  -> targeted retrieval while coverage is insufficient
  -> calibrated candidate + provenance-preserving context
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

The compiled evidence packet is placed before raw provenance. Aggregate,
temporal, and recommendation packets expose named evidence slots so readers
preserve required people, dates, quantities, positive preferences, and negative
constraints. A computed answer
is labeled proof-complete only when its operator-specific evidence fields are
present. Proof completion is structural, not a claim that the answer is
semantically correct; readers must check every candidate against its cited rows
and raw sources. Incomplete packets list their missing requirements.

## Ingestion

Raw source turns can be ingested without an API key. The deterministic fallback
keeps one lossless fact per source turn, while a typed extractor can atomize
facts and supply entities, predicates, modality, polarity, event time,
quantities, state identity, preference scope, and confidence.

```python
from activegraph_memory import DeterministicMemoryExtractor, extract_claim_inputs

claims, extraction = extract_claim_inputs(
    [turn],
    extractor=DeterministicMemoryExtractor(),
)
index = compile_memory_index(turns=[turn], claims=claims)
```

`ActiveGraphLLMMemoryExtractor` uses an ActiveGraph `LLMProvider` and a strict
structured output schema. Unknown source IDs are rejected. Typed fields drive
the compiler directly; they are not converted back to prose and guessed again.
Extraction confidence and belief confidence are stored separately. Long
histories are processed in stable, bounded batches (40 turns or 60,000 source
characters by default), and the result reports aggregate tokens, cost, latency,
cache state, and per-batch metadata.

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
- unresolved claim/state conflicts

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

Negative-existence queries run a bounded scan and report either positive
counterevidence or "not found in the compiled accepted-memory scope." They do
not convert retrieval failure into a claim about the outside world.

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

| Profile | Context | Rounds | Sufficiency | Source ratio | Entity/event vectors | Reasoning stages | Max calls / cost |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| `fast` | 2,500 | 1 | 0.60 | 0.90 | off | all off | 0 / $0 |
| `balanced` | 4,000 | 2 | 0.65 | 0.85 | on | classification/analysis fallback | 1 / $0.02 |
| `quality` | 6,000 | 2 | 0.72 | 0.82 | on | mixed fallback/always | 2 / $0.08 |
| `max_quality` | 10,000 | 3 | 0.78 | 0.82 | on | all always | 4 / $0.25 |

Reasoning is optional even in a reasoning-enabled profile. Without a reasoner,
the runtime remains deterministic. Before each optional call, the runtime checks
cumulative call, token, provider-reported cost, and latency thresholds. A call's
unknown final cost can cross a threshold; subsequent stages are then skipped and
audited. Provider backends also cap output tokens per call. Reasoner failures
fail open by default and are recorded in telemetry.

Packaging reasoning may only prioritize or exclude IDs from the already selected
evidence set. It cannot write or rewrite memory text. The final context is always
rendered deterministically from known claims and sources.

Profiles are Pydantic models and can be copied safely:

```python
from activegraph_memory import MemoryRuntime, runtime_profile

profile = runtime_profile("balanced").model_copy(
    update={
        "token_budget": 10_000,
        "candidate_answer_mode": "calibrated",
        "include_raw_sources": True,
    }
)
runtime = MemoryRuntime(profile)
```

Pack settings can override individual reasoning stages through
`profile_from_settings(settings)`. Passing `settings=` to
`GraphMemoryRepository.from_activegraph(...)` applies those profile overrides,
provider model choices, and extraction/temporal/conflict switches directly.

Per-operator confidence floors are available through
`operator_min_confidence`. Applications can fit them on held-out traces instead
of copying the built-in risk priors:

```python
from activegraph_memory import (
    apply_operator_calibration,
    calibrate_operator_thresholds,
    runtime_profile,
)

calibration = calibrate_operator_thresholds(
    held_out_records,  # operator, confidence, correct, proof_complete
    target_precision=0.9,
)
profile = apply_operator_calibration(runtime_profile("quality"), calibration)
```

Coverage audits separately report query-relevant source candidates represented
by extraction, typed compilation, compiled selection, and reader-visible
recovery. Recovery can put a missing raw source in context, but cannot make a
computed count, sum, or recommendation proof complete.

## ActiveGraph Persistence

`GraphMemoryRepository` materializes replayable source turns, claims, entities,
events, states, preferences, conflicts, list items, quantities, temporal
references, assessments, proofs, and measured retrieval stages. Stable keys
make compile, append, load, and retrieval trace writes idempotent.

```python
from activegraph import Graph, Runtime
from activegraph_memory import GraphMemoryRepository

graph = Graph(run_id="memory")
activegraph_runtime = Runtime(graph, persist_to="sqlite:///memory-events.db")
repository = GraphMemoryRepository.from_activegraph(
    activegraph_runtime,
    profile="balanced",
    extraction_model=None,  # deterministic fallback; set a model for typed LLM extraction
    reasoning_model=None,
    embedding_model="your-embedding-model",
)
repository.compile(turns=turns)
result = repository.retrieve("What changed?", query_id="memory-query-object-id")

# A new process can restore and continue from the graph event store.
restarted = GraphMemoryRepository(graph)
restarted.load()
restarted.append(turns=new_turns)
```

Durability is provided by the configured ActiveGraph store. The repository can
reconstruct its `MemoryIndex` from graph-visible source turns and claims after a
crash. Appends rebuild stable projections and patch changed state histories
without duplicating logical objects. The recovery boundary is a committed graph
write: a crash during an uncommitted provider extraction retries that batch,
while a committed `memory_ingestion_stage` and its facts load idempotently.
Provider recording or caching is recommended when retry charges matter.

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

`GraphMemoryRepository.from_activegraph(...)` binds typed extraction, staged
reasoning, and fielded embeddings to ActiveGraph's configured providers.
`MemoryRuntime.from_activegraph(...)` remains available for read-only use. The
provider owns API credentials; secrets are never written into graph state.

## Benchmarking

The benchmark APIs measure ingestion and retrieval separately. They report
latency percentiles, context and provider tokens, cost, graph size, retrieval
rounds, deterministic sufficiency, proof completion, candidate rendering,
reasoner calls, optional quality, and per-stage invocation/latency/cost data.

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

A fixed-budget deterministic option matrix is available from the CLI:

```bash
python3.11 -m activegraph_memory.benchmark_cli \
  --input examples/benchmark_fixture.json \
  --base-profile quality \
  --option-matrix \
  --repetitions 100 \
  --hash-embeddings \
  --score-expected
```

For live reasoning, use `benchmark_reasoning_ablations(...)` with a
`runtime_factory` that attaches the same provider/model to each ablation. It
isolates classification, strategy, analysis, packaging, all-stage, and no-stage
policies while recording provider-reported usage. `benchmark_ingestion(...)`
does the same for extraction, compilation, and optional graph materialization.

See [docs/benchmark-results.md](docs/benchmark-results.md) for the latest
reproducible control-plane result and its limitations. For live embeddings or
reasoning, use `benchmark_profiles(..., runtime_factory=...)` so the report
includes provider-reported tokens and caller-supplied pricing.

The benchmark-independent v4 fixture uses the same CLI:

```bash
python3.11 -m activegraph_memory.benchmark_cli \
  --input examples/v4_application_traces.json \
  --profiles fast,balanced,quality,max_quality \
  --repetitions 100 \
  --hash-embeddings \
  --score-expected
```

## Pack Surface

The pack registers 25 object types, 14 relation types, two deterministic
behaviors, and three tools. The main additions are:

- `memory_entity`, `memory_event`, `memory_state`, `memory_preference`
- `memory_query_analysis`, `memory_proof`, `memory_ingestion_stage`, `memory_retrieval_stage`
- `memory_embedding`, `memory_benchmark`
- `memory_source_turn`, `memory_conflict`, `memory_retrieval_assessment`
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
