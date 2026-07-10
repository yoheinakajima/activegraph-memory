"""Cost and quality profiles for the memory retrieval pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ReasoningMode = Literal["off", "fallback", "always"]
RuntimeProfileName = Literal["fast", "balanced", "quality", "max_quality"]
CandidateAnswerMode = Literal["never", "proof_complete", "calibrated"]


class StageReasoningPolicy(BaseModel):
    """When an optional reasoning backend may participate in one stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_classification: ReasoningMode = "off"
    retrieval_strategy: ReasoningMode = "off"
    retrieval_analysis: ReasoningMode = "off"
    context_packaging: ReasoningMode = "off"


class ReasoningBudget(BaseModel):
    """Cumulative stop thresholds checked before each optional reasoning call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_calls: int = Field(default=0, ge=0, le=20)
    max_input_tokens: int = Field(default=0, ge=0)
    max_output_tokens: int = Field(default=0, ge=0)
    max_cost_usd: float = Field(default=0.0, ge=0.0)
    max_latency_ms: float = Field(default=0.0, ge=0.0)


class MemoryRuntimeProfile(BaseModel):
    """A reproducible bundle of retrieval quality, latency, and cost knobs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: RuntimeProfileName
    token_budget: int = Field(ge=256, le=100_000)
    max_retrieval_rounds: int = Field(ge=1, le=5)
    max_claims_per_session: int = Field(ge=1, le=100)
    max_direct_turns_per_session: int = Field(ge=1, le=100)
    use_embeddings: bool = True
    embed_entities: bool = False
    embed_events: bool = False
    use_compiled_projection: bool = True
    use_graph_reducers: bool = True
    use_rank_fusion: bool = True
    use_diversity_selection: bool = True
    include_raw_sources: bool = True
    compact_context: bool = True
    adaptive_retrieval: bool = True
    min_sufficiency_confidence: float = Field(default=0.65, ge=0.0, le=1.0)
    source_budget_ratio: float = Field(default=0.85, ge=0.1, le=1.0)
    max_packet_rows: int = Field(default=16, ge=1, le=100)
    candidate_answer_mode: CandidateAnswerMode = "calibrated"
    reasoning_fail_open: bool = True
    reasoning: StageReasoningPolicy = Field(default_factory=StageReasoningPolicy)
    reasoning_budget: ReasoningBudget = Field(default_factory=ReasoningBudget)


_PROFILES: dict[RuntimeProfileName, MemoryRuntimeProfile] = {
    "fast": MemoryRuntimeProfile(
        name="fast",
        token_budget=2_500,
        max_retrieval_rounds=1,
        max_claims_per_session=4,
        max_direct_turns_per_session=4,
        embed_entities=False,
        embed_events=False,
        use_graph_reducers=False,
        adaptive_retrieval=False,
        min_sufficiency_confidence=0.6,
        source_budget_ratio=0.9,
        max_packet_rows=8,
        candidate_answer_mode="proof_complete",
        reasoning=StageReasoningPolicy(),
        reasoning_budget=ReasoningBudget(),
    ),
    "balanced": MemoryRuntimeProfile(
        name="balanced",
        token_budget=4_000,
        max_retrieval_rounds=2,
        max_claims_per_session=5,
        max_direct_turns_per_session=5,
        embed_entities=True,
        embed_events=True,
        use_graph_reducers=False,
        min_sufficiency_confidence=0.65,
        source_budget_ratio=0.85,
        max_packet_rows=12,
        reasoning=StageReasoningPolicy(
            query_classification="fallback",
            retrieval_analysis="fallback",
        ),
        reasoning_budget=ReasoningBudget(
            max_calls=1,
            max_input_tokens=12_000,
            max_output_tokens=1_500,
            max_cost_usd=0.02,
            max_latency_ms=30_000,
        ),
    ),
    "quality": MemoryRuntimeProfile(
        name="quality",
        token_budget=6_000,
        max_retrieval_rounds=2,
        max_claims_per_session=6,
        max_direct_turns_per_session=6,
        embed_entities=True,
        embed_events=True,
        use_graph_reducers=False,
        min_sufficiency_confidence=0.72,
        source_budget_ratio=0.82,
        max_packet_rows=16,
        reasoning=StageReasoningPolicy(
            query_classification="fallback",
            retrieval_strategy="fallback",
            retrieval_analysis="always",
            context_packaging="fallback",
        ),
        reasoning_budget=ReasoningBudget(
            max_calls=2,
            max_input_tokens=30_000,
            max_output_tokens=3_000,
            max_cost_usd=0.08,
            max_latency_ms=60_000,
        ),
    ),
    "max_quality": MemoryRuntimeProfile(
        name="max_quality",
        token_budget=10_000,
        max_retrieval_rounds=3,
        max_claims_per_session=8,
        max_direct_turns_per_session=8,
        embed_entities=True,
        embed_events=True,
        use_graph_reducers=False,
        min_sufficiency_confidence=0.78,
        source_budget_ratio=0.82,
        max_packet_rows=16,
        reasoning=StageReasoningPolicy(
            query_classification="always",
            retrieval_strategy="always",
            retrieval_analysis="always",
            context_packaging="always",
        ),
        reasoning_budget=ReasoningBudget(
            max_calls=4,
            max_input_tokens=80_000,
            max_output_tokens=6_000,
            max_cost_usd=0.25,
            max_latency_ms=180_000,
        ),
    ),
}


def runtime_profile(name: RuntimeProfileName | str = "balanced") -> MemoryRuntimeProfile:
    """Return a copy of a named runtime profile."""

    try:
        profile = _PROFILES[name]  # type: ignore[index]
    except KeyError as exc:
        choices = ", ".join(_PROFILES)
        raise ValueError(f"Unknown memory runtime profile {name!r}; choose one of: {choices}") from exc
    return profile.model_copy(deep=True)


def runtime_profiles() -> tuple[MemoryRuntimeProfile, ...]:
    """Return all built-in profiles in increasing quality/cost order."""

    return tuple(profile.model_copy(deep=True) for profile in _PROFILES.values())


def profile_from_settings(settings) -> MemoryRuntimeProfile:
    """Resolve a built-in profile plus per-stage settings overrides."""

    profile = runtime_profile(settings.runtime_profile)
    reasoning_updates = {}
    for field in (
        "query_classification",
        "retrieval_strategy",
        "retrieval_analysis",
        "context_packaging",
    ):
        value = getattr(settings, f"{field}_reasoning", None)
        if value is not None:
            reasoning_updates[field] = value
    profile_updates = {}
    for field in (
        "adaptive_retrieval",
        "min_sufficiency_confidence",
        "source_budget_ratio",
        "max_packet_rows",
        "candidate_answer_mode",
    ):
        value = getattr(settings, field, None)
        if value is not None:
            profile_updates[field] = value
    if reasoning_updates:
        profile_updates["reasoning"] = profile.reasoning.model_copy(update=reasoning_updates)
    budget_updates = {}
    for settings_field, budget_field in (
        ("max_reasoning_calls", "max_calls"),
        ("max_reasoning_input_tokens", "max_input_tokens"),
        ("max_reasoning_output_tokens", "max_output_tokens"),
        ("max_reasoning_cost_usd", "max_cost_usd"),
        ("max_reasoning_latency_ms", "max_latency_ms"),
    ):
        value = getattr(settings, settings_field, None)
        if value is not None:
            budget_updates[budget_field] = value
    if budget_updates:
        profile_updates["reasoning_budget"] = profile.reasoning_budget.model_copy(update=budget_updates)
    return profile.model_copy(update=profile_updates) if profile_updates else profile
