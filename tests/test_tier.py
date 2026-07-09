"""
Tests for tier configuration management.

Covers: DEFAULT_TIERS structure, set_tiers, reset_tiers, get_tiers/get_tier,
and the TIERS alias staying in sync with CONFIG.
"""

import pytest
from smart_router.tier import (
    DEFAULT_TIERS,
    DEFAULT_TIER,
    MIN_CONFIDENCE,
    CONFIG,
    TIERS,
    set_tiers,
    get_tiers,
    get_tier,
    reset_tiers,
)


class TestDefaultTiers:
    """Validate the structure and content of DEFAULT_TIERS."""

    def test_default_tiers_has_three_tiers(self):
        assert set(DEFAULT_TIERS.keys()) == {"local", "flash", "pro"}

    def test_each_tier_has_required_fields(self):
        for name, config in DEFAULT_TIERS.items():
            assert "models" in config, f"{name} missing 'models'"
            assert "utterances" in config, f"{name} missing 'utterances'"
            assert "description" in config, f"{name} missing 'description'"

    def test_models_have_provider_and_model(self):
        for name, config in DEFAULT_TIERS.items():
            assert "provider" in config["models"], f"{name} models missing 'provider'"
            assert "model" in config["models"], f"{name} models missing 'model'"

    def test_utterances_are_non_empty(self):
        for name, config in DEFAULT_TIERS.items():
            assert len(config["utterances"]) > 0, f"{name} has no utterances"

    def test_local_tier_has_base_url(self):
        assert "base_url" in DEFAULT_TIERS["local"]["models"]

    def test_default_tier_is_flash(self):
        assert DEFAULT_TIER == "flash"

    def test_min_confidence_is_reasonable(self):
        assert 0 < MIN_CONFIDENCE < 1


class TestConfigAlias:
    """Verify TIERS and CONFIG are the same object and stay in sync."""

    def test_tiers_is_config(self):
        assert TIERS is CONFIG

    def test_tiers_reflects_set_tiers(self):
        set_tiers({"local": None})
        try:
            assert "local" not in TIERS
            assert "local" not in CONFIG
        finally:
            reset_tiers()

    def test_tiers_reflects_reset(self):
        set_tiers({"local": None, "flash": None})
        reset_tiers()
        assert set(TIERS.keys()) == {"local", "flash", "pro"}
        assert TIERS is CONFIG


class TestSetTiers:
    """Test the set_tiers function for overriding tier config."""

    def setup_method(self):
        reset_tiers()

    def teardown_method(self):
        reset_tiers()

    def test_disable_tier(self):
        set_tiers({"pro": None})
        tiers = get_tiers()
        assert "pro" not in tiers
        assert "local" in tiers
        assert "flash" in tiers

    def test_override_tier_utterances(self):
        custom_utterances = ["custom query one", "custom query two"]
        set_tiers({"local": {"models": {"provider": "custom", "model": "tiny"}, "utterances": custom_utterances}})
        assert get_tier("local")["utterances"] == custom_utterances

    def test_add_new_tier(self):
        new_tier = {
            "models": {"provider": "openai", "model": "gpt-4o-mini"},
            "utterances": ["ultra simple question"],
        }
        set_tiers({"ultra": new_tier})
        tiers = get_tiers()
        assert "ultra" in tiers
        assert tiers["ultra"]["models"]["provider"] == "openai"

    def test_empty_set_tiers_does_nothing(self):
        before = get_tiers()
        set_tiers({})
        after = get_tiers()
        assert before == after

    def test_set_tiers_with_none_does_nothing(self):
        before = get_tiers()
        set_tiers(None)
        after = get_tiers()
        assert before == after


class TestGetTier:
    """Test get_tier lookup."""

    def setup_method(self):
        reset_tiers()

    def test_get_existing_tier(self):
        tier = get_tier("local")
        assert tier is not None
        assert tier["models"]["provider"] == "custom"

    def test_get_missing_tier_returns_none(self):
        assert get_tier("nonexistent") is None


class TestResetTiers:
    """Test reset to defaults."""

    def test_reset_restores_all_tiers(self):
        set_tiers({"local": None, "flash": None})
        reset_tiers()
        tiers = get_tiers()
        assert set(tiers.keys()) == {"local", "flash", "pro"}

    def test_reset_restores_original_config(self):
        set_tiers({"local": {"models": {"provider": "x", "model": "y"}, "utterances": ["z"]}})
        reset_tiers()
        assert get_tier("local")["models"]["provider"] == "custom"
        assert get_tier("local")["models"]["model"] == "llama3.2:3b"
