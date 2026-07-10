"""JSON-in benchmark CLI for deterministic profile comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmarking import (
    MemoryBenchmarkCase,
    benchmark_profiles,
    benchmark_runtime_options,
    render_benchmark_markdown,
)
from .compiler import ExtractedClaimInput, SourceTurn, compile_memory_index
from .runtime import MemoryRuntime


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSON file containing turns, claims, and cases")
    parser.add_argument(
        "--profiles",
        default="fast,balanced,quality,max_quality",
        help="Comma-separated profile names",
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--option-matrix",
        action="store_true",
        help="Ablate deterministic features at a constant base profile instead of comparing profiles",
    )
    parser.add_argument("--base-profile", default="quality")
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument(
        "--hash-embeddings",
        action="store_true",
        help="Use ActiveGraph's deterministic local hash provider for offline fielded-retrieval timing",
    )
    parser.add_argument("--hash-embedding-dimensions", type=int, default=64)
    parser.add_argument(
        "--score-expected",
        action="store_true",
        help="Score exact normalized compiled candidates for cases with an expected value",
    )
    parser.add_argument("--output", help="Optional output file; stdout when omitted")
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.input).read_text())
    turns = [SourceTurn(**raw) for raw in payload.get("turns", [])]
    claims = [
        ExtractedClaimInput(
            **{
                **raw,
                "mentioned_turn_idxs": tuple(raw.get("mentioned_turn_idxs", [])),
            }
        )
        for raw in payload.get("claims", [])
    ]
    cases = [MemoryBenchmarkCase(**raw) for raw in payload.get("cases", [])]
    index = compile_memory_index(
        turns=turns,
        claims=claims,
        metadata=dict(payload.get("metadata") or {}),
    )
    runtime_factory = None
    if args.hash_embeddings:
        from activegraph.llm import HashEmbeddingProvider

        provider = HashEmbeddingProvider(dimensions=max(8, args.hash_embedding_dimensions))
        runtime_factory = lambda profile: MemoryRuntime(
            profile,
            embedding_provider=provider,
            embedding_model=f"hash-{max(8, args.hash_embedding_dimensions)}",
        )
    benchmark_kwargs = {
        "repetitions": max(1, args.repetitions),
        "runtime_factory": runtime_factory,
        "evaluator": _exact_candidate_evaluator if args.score_expected else None,
    }
    if args.option_matrix:
        results = list(
            benchmark_runtime_options(
                index,
                cases,
                base_profile=args.base_profile,
                **benchmark_kwargs,
            ).values()
        )
    else:
        results = benchmark_profiles(
            index,
            cases,
            profiles=tuple(value.strip() for value in args.profiles.split(",") if value.strip()),
            **benchmark_kwargs,
        )
    if args.format == "json":
        rendered = json.dumps([result.as_dict() for result in results], indent=2, sort_keys=True)
    else:
        rendered = render_benchmark_markdown(results)
    if args.output:
        Path(args.output).write_text(rendered + "\n")
    else:
        print(rendered)
    return 0


def _exact_candidate_evaluator(case: MemoryBenchmarkCase, result) -> float | None:
    if case.expected is None:
        return None
    candidate = (result.metadata.get("compiled_evidence") or {}).get("candidate_answer")
    if candidate is None:
        return 0.0
    normalize = lambda value: " ".join(str(value).lower().strip().rstrip(".").split())
    return float(normalize(candidate) == normalize(case.expected))


if __name__ == "__main__":
    raise SystemExit(main())
