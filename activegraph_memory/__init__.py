"""Semantic and epistemic memory pack for ActiveGraph."""

from __future__ import annotations

from pathlib import Path

from activegraph.packs import Pack, load_prompts_from_dir

from .behaviors import BEHAVIORS
from .constants import PACK_NAME, PACK_VERSION
from .compiler import (
    CategoryRef,
    EntityRef,
    ExtractedClaimInput,
    MemoryEventRecord,
    MemoryIndex,
    SourceTurn,
    compile_memory_index,
)
from .calibration import (
    OperatorCalibrationResult,
    apply_operator_calibration,
    calibrate_operator_thresholds,
)
from .coverage_audit import SourceCoverageAudit, audit_source_coverage
from .ingestion_coverage import extraction_run_source_ids, extraction_runs_cover
from .shared_extraction import (
    CompatibilityMemoryExtractor,
    SharedExtractionActive,
    claims_from_shared_annotations,
    shared_extraction_result,
)
from .benchmarking import (
    MemoryBenchmarkCase,
    MemoryBenchmarkResult,
    MemoryIngestionBenchmarkResult,
    benchmark_ingestion,
    benchmark_profiles,
    benchmark_reasoning_ablations,
    benchmark_runtime_options,
    benchmark_runtime,
    reasoning_ablation_profiles,
    runtime_option_profiles,
    render_benchmark_markdown,
)
from .graph_runtime import (
    GraphMemoryRepository,
    graph_materialization,
    load_memory_index,
    materialize_memory_index,
    materialize_retrieval_trace,
)
from .extraction import (
    ActiveGraphLLMMemoryExtractor,
    CallableMemoryExtractor,
    DeterministicMemoryExtractor,
    ExtractedEntityInput,
    ExtractedMemoryFact,
    MemoryExtractionOutput,
    MemoryExtractionResult,
    MemoryExtractor,
    extract_claim_inputs,
)
from .embedding_store import EmbeddingVectorStore, GraphEmbeddingStore, SQLiteEmbeddingStore
from .graph_query import GraphQueryResult, run_graph_query
from .object_types import OBJECT_TYPES, RELATION_TYPES
from .profiles import (
    MemoryRuntimeProfile,
    ReasoningBudget,
    StageReasoningPolicy,
    profile_from_settings,
    runtime_profile,
    runtime_profiles,
)
from .query_ir import QueryAnalysis, analyze_query
from .ranking import EmbeddingSignalProvider, RetrievalSignals
from .reasoning import (
    ActiveGraphLLMReasoningBackend,
    ReasoningBackend,
    ReasoningRequest,
    ReasoningResponse,
)
from .retrieval import MemoryRetrievalResult, retrieve_memory
from .runtime import MemoryRuntime
from .settings import ActiveGraphMemorySettings
from .tools import TOOLS

_PROMPTS_DIR = Path(__file__).parent / "prompts"

pack = Pack(
    name=PACK_NAME,
    version=PACK_VERSION,
    description=(
        "Semantic and epistemic memory layer for ActiveGraph. Adds claims, "
        "episodes, temporal references, retrieval plans, coverage reports, "
        "confidence vectors, and evidence-backed answers while composing "
        "with core and memory_gateway."
    ),
    object_types=OBJECT_TYPES,
    relation_types=RELATION_TYPES,
    behaviors=BEHAVIORS,
    tools=TOOLS,
    policies=(),
    prompts=load_prompts_from_dir(_PROMPTS_DIR) if _PROMPTS_DIR.exists() else (),
    settings_schema=ActiveGraphMemorySettings,
)

__all__ = [
    "pack",
    "ActiveGraphMemorySettings",
    "ActiveGraphLLMReasoningBackend",
    "ActiveGraphLLMMemoryExtractor",
    "CategoryRef",
    "EmbeddingSignalProvider",
    "EmbeddingVectorStore",
    "EntityRef",
    "ExtractedClaimInput",
    "ExtractedEntityInput",
    "ExtractedMemoryFact",
    "MemoryExtractionOutput",
    "MemoryExtractionResult",
    "MemoryExtractor",
    "CallableMemoryExtractor",
    "DeterministicMemoryExtractor",
    "GraphQueryResult",
    "GraphMemoryRepository",
    "GraphEmbeddingStore",
    "MemoryEventRecord",
    "MemoryIndex",
    "MemoryBenchmarkCase",
    "MemoryBenchmarkResult",
    "MemoryIngestionBenchmarkResult",
    "MemoryRuntime",
    "MemoryRuntimeProfile",
    "OperatorCalibrationResult",
    "ReasoningBudget",
    "MemoryRetrievalResult",
    "SourceTurn",
    "SQLiteEmbeddingStore",
    "QueryAnalysis",
    "ReasoningBackend",
    "ReasoningRequest",
    "ReasoningResponse",
    "RetrievalSignals",
    "StageReasoningPolicy",
    "SourceCoverageAudit",
    "CompatibilityMemoryExtractor",
    "SharedExtractionActive",
    "claims_from_shared_annotations",
    "shared_extraction_result",
    "extraction_run_source_ids",
    "extraction_runs_cover",
    "analyze_query",
    "apply_operator_calibration",
    "audit_source_coverage",
    "benchmark_profiles",
    "benchmark_ingestion",
    "benchmark_reasoning_ablations",
    "benchmark_runtime_options",
    "benchmark_runtime",
    "render_benchmark_markdown",
    "reasoning_ablation_profiles",
    "runtime_option_profiles",
    "compile_memory_index",
    "calibrate_operator_thresholds",
    "extract_claim_inputs",
    "retrieve_memory",
    "profile_from_settings",
    "runtime_profile",
    "runtime_profiles",
    "run_graph_query",
    "materialize_memory_index",
    "materialize_retrieval_trace",
    "graph_materialization",
    "load_memory_index",
]
