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

- date: 2026-07-09
- machine: Apple Silicon, arm64
- operating system: macOS 26.5.1
- Python: 3.11.15
- cases: 5
- executions per profile: 500
- embeddings: ActiveGraph deterministic 64-dimensional hash provider
- reasoning backend: none

| Profile | Cases | Mean ms | P95 ms | Cold ms | Warm mean ms | Context tokens | Proof rate | Cost USD | Quality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fast | 5 | 1.372 | 1.738 | 2.657 | 1.370 | 350.80 | 1.0000 | 0.00000000 | 1.0000 |
| balanced | 5 | 1.592 | 2.076 | 1.406 | 1.593 | 350.80 | 1.0000 | 0.00000000 | 1.0000 |
| quality | 5 | 1.611 | 2.098 | 1.339 | 1.611 | 350.80 | 1.0000 | 0.00000000 | 1.0000 |
| max_quality | 5 | 1.581 | 2.048 | 1.318 | 1.581 | 350.80 | 1.0000 | 0.00000000 | 1.0000 |

`Quality` is exact normalized equality against four compiled-answer fixture
cases. The fifth recommendation case has no scalar expected answer and is not
included in the quality mean.

This benchmark measures deterministic compilation, fielded retrieval, graph
signal propagation, operator execution, proof checking, and context assembly.
It does not estimate network latency, hosted embedding cost, LLM reasoning cost,
reader answer quality, or LongMemEval accuracy. Those must be measured with the
same benchmark API and a live `runtime_factory`.

The profiles are close here because the fixture is small and no reasoning
backend is attached. The important control result is that all profiles execute
the same semantics and expose their measured overhead. Larger corpora and live
reasoning are expected to separate the profiles more clearly.
