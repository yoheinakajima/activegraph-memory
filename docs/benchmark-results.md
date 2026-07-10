# Profile Benchmark Results

## Offline Control Plane

Command:

```bash
python3.11 -m activegraph_memory.benchmark_cli \
  --input examples/benchmark_fixture.json \
  --profiles fast,balanced,quality,max_quality \
  --repetitions 100 \
  --hash-embeddings \
  --score-expected \
  --format markdown
```

Environment:

- date: 2026-07-10
- machine: Apple Silicon, arm64
- operating system: macOS 26.5.1
- Python: 3.11.15
- cases: 5
- executions per profile: 500
- embeddings: ActiveGraph deterministic 64-dimensional hash provider
- reasoning backend: none

| Profile | Cases | Mean ms | P95 ms | Context tokens | Rounds | Sufficient | Proof rate | Reason calls | Cost USD | Quality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fast | 5 | 1.604 | 2.424 | 289.80 | 1.000 | 0.8000 | 1.0000 | 0 | 0.00000000 | 1.0000 |
| balanced | 5 | 1.955 | 3.894 | 289.80 | 1.200 | 1.0000 | 1.0000 | 0 | 0.00000000 | 1.0000 |
| quality | 5 | 1.996 | 3.947 | 289.80 | 1.200 | 1.0000 | 1.0000 | 0 | 0.00000000 | 1.0000 |
| max_quality | 5 | 1.959 | 3.899 | 289.80 | 1.200 | 1.0000 | 1.0000 | 0 | 0.00000000 | 1.0000 |

## Deterministic Option Matrix

Command:

```bash
python3.11 -m activegraph_memory.benchmark_cli \
  --input examples/benchmark_fixture.json \
  --base-profile quality \
  --option-matrix \
  --repetitions 100 \
  --hash-embeddings \
  --score-expected \
  --format markdown
```

| Option | Mean ms | P95 ms | Context tokens | Rounds | Sufficient | Proof rate | Quality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 1.994 | 3.944 | 289.80 | 1.200 | 1.0000 | 1.0000 | 1.0000 |
| embeddings off | 1.628 | 3.421 | 287.40 | 1.400 | 0.6000 | 0.8000 | 1.0000 |
| adaptive retrieval off | 1.690 | 2.466 | 288.40 | 1.000 | 0.8000 | 1.0000 | 1.0000 |
| compiled projection off | 1.374 | 1.568 | 161.00 | 2.000 | 0.0000 | 0.0000 | 0.0000 |
| raw sources off | 1.950 | 3.864 | 128.20 | 1.200 | 1.0000 | 1.0000 | 1.0000 |
| candidate answer off | 1.902 | 3.789 | 283.20 | 1.200 | 1.0000 | 1.0000 | 1.0000 |

This small fixture is intentionally easy, so exact-answer quality is not a
claim about general performance. The useful control signals are that removing
the compiled projection collapses proof/sufficiency and fixture quality, while
removing embeddings or adaptive rounds makes more cases fail the sufficiency
gate even when the tiny fixture's scalar answer remains correct.

## Ingestion And Materialization

The same six-turn fixture was run 100 times through
`DeterministicMemoryExtractor`, compilation, and materialization into a fresh
ActiveGraph per repetition.

| Turns | Facts | Mean ms | P50 ms | P95 ms | Cold ms | Warm mean ms | Graph objects | Graph relations | Cost USD |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 6 | 3.731 | 3.620 | 3.925 | 6.663 | 3.701 | 38.0 | 62.0 | 0.00000000 |

`Quality` is exact normalized equality against four compiled-answer fixture
cases. The fifth recommendation case has no scalar expected answer and is not
included in the quality mean.

These benchmarks measure deterministic extraction/compilation/materialization,
fielded retrieval, graph signal propagation, adaptive rounds, operator
execution, sufficiency/proof checking, and context assembly. They do not
estimate hosted model latency/cost, reader answer quality, or LongMemEval
accuracy. Use `benchmark_reasoning_ablations` and `benchmark_ingestion` with a
live provider-bound factory to capture provider-reported usage.

The profiles are close here because the fixture is small and no reasoning
backend is attached. The important control result is that all profiles execute
the same semantics and expose their measured overhead. Larger corpora and live
reasoning are expected to separate the profiles more clearly.

## v4 Application Traces

The v4 fixture is independent of LongMemEval and covers infrastructure spend,
project phase transitions, launch-date deltas, positive/negative hotel
constraints, and completed-versus-planned agent tasks. It ran 100 repetitions
per profile with the deterministic 64-dimensional hash provider.

| Profile | Mean ms | P95 ms | Context tokens | Rounds | Sufficient | Proof | Scalar quality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fast | 2.273 | 3.115 | 628.40 | 1.000 | 1.0000 | 1.0000 | 1.0000 |
| balanced | 2.328 | 3.135 | 628.40 | 1.000 | 1.0000 | 1.0000 | 1.0000 |
| quality | 2.652 | 4.598 | 626.80 | 1.200 | 0.8000 | 1.0000 | 1.0000 |
| max_quality | 2.988 | 6.225 | 626.80 | 1.400 | 0.8000 | 1.0000 | 1.0000 |

Across profiles, mean measured coverage confidence was 1.0000, raw recovery was
not needed, and packets contained 1.6 evidence slots on average. The stricter
profiles deliberately continue retrieval for the recommendation case because
its operator confidence floor is higher; exact scalar outputs remain unchanged.
This fixture is a regression/generalization control, not a claim about broad
application accuracy.
