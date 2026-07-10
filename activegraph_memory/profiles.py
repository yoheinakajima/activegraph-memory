"""Cost and quality profiles for the memory retrieval pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ReasoningMode = Literal["off", "fallback", "always"]
RuntimeProfileName = Literal["fast", "balanced", "quality", "max_quality"]


class StageReasoningPolicy(BaseModel):
    """When an optional reasoning backend may participate in one stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_classification: ReasoningMode = "off"
    retrieval_strategy: ReasoningMode = "off"
    retrieval_analysis: ReasoningMode = "off"
    context_packaging: ReasoningMode = "off"


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
    reasoning_fail_open: bool = True
    reasoning: StageReasoningPolicy = Field(default_factory=StageReasoningPolicy)


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
        reasoning=StageReasoningPolicy(),
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
        reasoning=StageReasoningPolicy(
            query_classification="fallback",
            retrieval_analysis="fallback",
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
        reasoning=StageReasoningPolicy(
            query_classification="fallback",
            retrieval_strategy="fallback",
            retrieval_analysis="always",
            context_packaging="fallback",
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
        reasoning=StageReasoningPolicy(
            query_classification="always",
            retrieval_strategy="always",
            retrieval_analysis="always",
            context_packaging="always",
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
    updates = {}
    for field in (
        "query_classification",
        "retrieval_strategy",
        "retrieval_analysis",
        "context_packaging",
    ):
        value = getattr(settings, f"{field}_reasoning", None)
        if value is not None:
            updates[field] = value
    if not updates:
        return profile
    return profile.model_copy(
        update={"reasoning": profile.reasoning.model_copy(update=updates)}
    )
