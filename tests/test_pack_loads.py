from pathlib import Path

from activegraph_memory import ActiveGraphMemorySettings, pack
from activegraph_memory.object_types import OBJECT_TYPE_NAMES, RELATION_TYPE_NAMES
from activegraph.packs.manifest import load_manifest, verify_content_hash, verify_surface


def test_pack_loads_with_expected_metadata():
    assert pack.name == "activegraph_memory"
    assert pack.version == "0.2.0"
    assert pack.settings_schema is ActiveGraphMemorySettings


def test_manifest_matches_runtime_surface_and_package_content():
    package_root = Path(__file__).parents[1] / "activegraph_memory"
    manifest = load_manifest(package_root)

    verify_surface(manifest, pack)
    verify_content_hash(manifest, package_root)


def test_pack_declares_memory_objects_and_relations():
    assert len(pack.object_types) == 21
    assert len(pack.relation_types) == 14

    assert "memory_claim" in OBJECT_TYPE_NAMES
    assert "retrieval_plan" in OBJECT_TYPE_NAMES
    assert "memory_answer" in OBJECT_TYPE_NAMES
    assert "memory_entity" in OBJECT_TYPE_NAMES
    assert "memory_event" in OBJECT_TYPE_NAMES
    assert "memory_state" in OBJECT_TYPE_NAMES
    assert "memory_query_analysis" in OBJECT_TYPE_NAMES
    assert "memory_retrieval_stage" in OBJECT_TYPE_NAMES
    assert "memory_proof" in OBJECT_TYPE_NAMES
    assert "memory_embedding" in OBJECT_TYPE_NAMES

    assert "memory_supports" in RELATION_TYPE_NAMES
    assert "memory_supersedes" in RELATION_TYPE_NAMES
    assert "memory_has_coverage" in RELATION_TYPE_NAMES
    assert "memory_about" in RELATION_TYPE_NAMES
    assert "memory_stage_for" in RELATION_TYPE_NAMES
    assert "memory_proves" in RELATION_TYPE_NAMES
    assert {tool.name for tool in pack.tools} == {
        "plan_memory_query",
        "analyze_memory_query",
        "list_memory_profiles",
    }


def test_default_settings_match_phase_one_contract():
    settings = ActiveGraphMemorySettings()
    assert settings.enable_claim_extraction is False
    assert settings.enable_temporal_resolution is False
    assert settings.enable_conflict_detection is False
    assert settings.enable_gateway_integration is True
    assert settings.default_top_k == 10
    assert settings.min_confidence == 0.5
    assert settings.strict_coverage_for_latest is True
    assert settings.runtime_profile == "balanced"
    assert settings.enable_query_analysis_behavior is True
