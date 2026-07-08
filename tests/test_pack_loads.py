from activegraph_memory import ActiveGraphMemorySettings, pack
from activegraph_memory.object_types import OBJECT_TYPE_NAMES, RELATION_TYPE_NAMES


def test_pack_loads_with_expected_metadata():
    assert pack.name == "activegraph_memory"
    assert pack.version == "0.1.0"
    assert pack.settings_schema is ActiveGraphMemorySettings


def test_pack_declares_memory_objects_and_relations():
    assert len(pack.object_types) == 11
    assert len(pack.relation_types) == 9

    assert "memory_claim" in OBJECT_TYPE_NAMES
    assert "retrieval_plan" in OBJECT_TYPE_NAMES
    assert "memory_answer" in OBJECT_TYPE_NAMES

    assert "memory_supports" in RELATION_TYPE_NAMES
    assert "memory_supersedes" in RELATION_TYPE_NAMES
    assert "memory_has_coverage" in RELATION_TYPE_NAMES


def test_default_settings_match_phase_one_contract():
    settings = ActiveGraphMemorySettings()
    assert settings.enable_claim_extraction is False
    assert settings.enable_temporal_resolution is False
    assert settings.enable_conflict_detection is False
    assert settings.enable_gateway_integration is True
    assert settings.default_top_k == 10
    assert settings.min_confidence == 0.5
    assert settings.strict_coverage_for_latest is True
