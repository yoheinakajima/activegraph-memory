# activegraph-memory

`activegraph-memory` is a semantic and epistemic memory pack for [ActiveGraph](https://pypi.org/project/activegraph/).

It sits above `activegraph-packs/packs/memory_gateway`. The gateway owns the low-level lifecycle:

```text
memory_candidate -> evaluation -> memory_item -> retrieval/backend
```

This package owns the higher-level memory layer:

- claim and episode objects
- temporal validity and quantity claims
- supersession and contradiction relations
- deterministic query planning
- coverage reports
- confidence vectors
- evidence-backed memory answers

The first release is intentionally deterministic. It defines the object model, relation model, planner utilities, compiler/retriever runtime, coverage/confidence helpers, and a graph-visible query-planning behavior. It does not own connector-specific extraction and does not replace mem0, Zep, pgvector, SQLite, or the existing memory backend seam.

## Install

```bash
pip install -e ".[dev]"
```

## Pack Entry Point

After installation, ActiveGraph can load the pack by entry point:

```python
from activegraph_memory import ActiveGraphMemorySettings, pack

settings = ActiveGraphMemorySettings()
```

The package registers:

```toml
[project.entry-points."activegraph.packs"]
activegraph_memory = "activegraph_memory:pack"
```

## What This Pack Adds

Objects:

- `memory_claim`
- `memory_episode`
- `memory_query`
- `retrieval_plan`
- `coverage_report`
- `evidence_bundle`
- `memory_answer`
- `temporal_ref`
- `quantity_claim`
- `memory_policy`
- `memory_evaluation`

Relations:

- `memory_supports`
- `memory_contradicts`
- `memory_supersedes`
- `memory_has_temporal_ref`
- `memory_has_quantity`
- `memory_retrieved_for`
- `memory_used_as_evidence`
- `memory_has_coverage`
- `memory_governed_by_policy`

Core provenance relations such as `derived_from` stay in Core and are not redefined here.

## Deterministic Planning

The planner turns `memory_query` data into a `retrieval_plan`:

```python
from activegraph_memory.object_types import MemoryQuery
from activegraph_memory.planner import plan_query

query = MemoryQuery(query="What is the latest launch plan?")
plan = plan_query(query)

assert plan.requires_freshness is True
assert "supersession_scan" in plan.strategies
```

The pack also registers a `memory_query_planner` behavior. When loaded in an ActiveGraph runtime, creating a `memory_query` can create a graph-visible `retrieval_plan`.

## Compile From The Log

The standalone compiler/retriever path accepts source turns plus extracted
claim inputs from any upstream extractor or connector:

```python
from activegraph_memory import ExtractedClaimInput, SourceTurn
from activegraph_memory import compile_memory_index, retrieve_memory

turn = SourceTurn(
    turn_id="session-1#0",
    session_id="session-1",
    session_date="2023-05-27",
    session_idx=0,
    turn_idx=0,
    role="user",
    content="I have been taking Spanish classes for the past three months.",
    text="[Session session-1 (2023-05-27)] user: I have been taking Spanish classes for the past three months.",
)
claim = ExtractedClaimInput(
    text="The user has been taking Spanish classes for the past three months.",
    session_id="session-1",
    session_date="2023-05-27",
    session_idx=0,
    role="user",
    mentioned_turn_idxs=(0,),
)
index = compile_memory_index(turns=[turn], claims=[claim])
result = retrieve_memory(
    index,
    "Which happened first, Spanish classes or the festival?",
    question_date="2023/05/27 (Sat)",
)
```

The retriever returns context text plus structured artifacts:

- `RetrievalPlan`
- `EvidenceBundle`
- `CoverageReport`
- confidence vector
- selected claim and source-turn ids

Claim headers are rendered above source turns, so semantic memory acts as an
index into the event log rather than a replacement for the log.

## Behavior Map

| Behavior | Trigger | Output | Notes |
| --- | --- | --- | --- |
| `memory_query_planner` | `memory_query.created` | `retrieval_plan` | Deterministic, offline, no API key. |

Future behavior layers should add connector-specific extraction, full graph-visible evidence retrieval, conflict/supersession writes, and answer synthesis as graph-visible steps.

## Gateway Boundary

This pack should compose with `core` and `memory_gateway`:

```text
source -> observation -> memory_candidate -> evaluation -> memory_item
                                              |
memory_query -> retrieval_plan -> memory_retrieval_request -> memory_retrieval
                         |
                         -> evidence_bundle -> coverage_report -> memory_answer
```

Retrieval remains auditable because plans and evidence are represented as graph objects instead of hidden function calls.

## Validation

```bash
pytest -q
```

Tests run offline with no API key and no live network.

## Research Posture

The design follows the idea that vector search is only one access path. Strong memory needs continuity, provenance, belief maintenance, and epistemic reporting:

- semantic recall finds candidate evidence
- ActiveGraph preserves chronology and relationships
- structured metadata handles freshness, authority, and coverage
- answer contracts report uncertainty instead of hiding it

This repo is a foundation for that direction, not a benchmark claim.
