"""
Tests for the core ModelRouter.

All tests mock the encoder and SemanticRouter to avoid network calls and
model downloads. This gives us fast, deterministic tests for routing logic.
"""

from unittest.mock import MagicMock, patch, PropertyMock
import pytest

from smart_router.router import ModelRouter
from smart_router.tier import TIERS, DEFAULT_TIER, MIN_CONFIDENCE, reset_tiers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_route_choice(name: str, score: float = 0.9):
    """Create a mock RouteChoice with the given name and similarity score."""
    choice = MagicMock()
    choice.name = name
    choice.similarity_score = score
    return choice


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestModelRouterInit:
    """Test ModelRouter construction and lazy initialization."""

    def test_default_encoder_model(self):
        router = ModelRouter()
        assert router.encoder_model == "nomic-embed-text"
        assert router._initialized is False

    def test_custom_encoder_model(self):
        router = ModelRouter(encoder_model="custom/model-name")
        assert router.encoder_model == "custom/model-name"

    def test_not_initialized_on_creation(self):
        router = ModelRouter()
        assert router._router is None
        assert router._encoder is None


class TestAvailableTiers:
    """Test the available_tiers property."""

    def test_returns_active_tier_names(self):
        router = ModelRouter()
        tiers = router.available_tiers
        assert isinstance(tiers, list)
        assert "local" in tiers
        assert "flash" in tiers
        assert "pro" in tiers


class TestClassify:
    """Test the classify method with mocked encoder/router."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Reset tiers before and after each test."""
        reset_tiers()
        yield
        reset_tiers()

    def _make_router(self, mock_router):
        """Create a ModelRouter with a pre-configured mock."""
        router = ModelRouter()
        # Skip actual init — inject mocks directly
        router._encoder = MagicMock()
        router._router = mock_router
        router._initialized = True
        return router

    def test_classify_local(self):
        """Query that semantically matches local tier."""
        mock = MagicMock()
        mock.return_value = _make_mock_route_choice("local", 0.92)
        router = self._make_router(mock)

        result = router.classify("What is the capital of France?")
        assert result == "local"

    def test_classify_flash(self):
        """Query that matches flash tier."""
        mock = MagicMock()
        mock.return_value = _make_mock_route_choice("flash", 0.88)
        router = self._make_router(mock)

        result = router.classify("Explain how DNS works")
        assert result == "flash"

    def test_classify_pro(self):
        """Query that matches pro tier."""
        mock = MagicMock()
        mock.return_value = _make_mock_route_choice("pro", 0.95)
        router = self._make_router(mock)

        result = router.classify("Design a microservice architecture for an e-commerce platform")
        assert result == "pro"

    def test_classify_returns_string(self):
        """Classify always returns a string, never None."""
        mock = MagicMock()
        mock.return_value = _make_mock_route_choice("flash", 0.85)
        router = self._make_router(mock)

        result = router.classify("any query")
        assert isinstance(result, str)
        assert result in TIERS

    def test_classify_low_confidence_falls_back(self):
        """When score is below MIN_CONFIDENCE, return DEFAULT_TIER."""
        mock = MagicMock()
        mock.return_value = _make_mock_route_choice("pro", 0.3)  # below 0.6
        router = self._make_router(mock)

        result = router.classify("ambiguous query")
        assert result == DEFAULT_TIER  # "flash"

    def test_classify_exactly_at_threshold(self):
        """Score == threshold should NOT fall back (>= check)."""
        mock = MagicMock()
        mock.return_value = _make_mock_route_choice("local", MIN_CONFIDENCE)  # 0.6
        router = self._make_router(mock)

        result = router.classify("borderline query")
        assert result == "local"

    def test_classify_none_result_falls_back(self):
        """When router returns None, should fall back to default."""
        mock = MagicMock()
        mock.return_value = None
        router = self._make_router(mock)

        result = router.classify("no route matched")
        assert result == DEFAULT_TIER

    def test_classify_unknown_tier_falls_back(self):
        """When router returns an unknown tier name, fall back."""
        mock = MagicMock()
        mock.return_value = _make_mock_route_choice("unknown_tier", 0.9)
        router = self._make_router(mock)

        result = router.classify("something weird")
        assert result == DEFAULT_TIER

    def test_classify_router_exception_falls_back(self):
        """When the router raises an exception, fall back to default."""
        mock = MagicMock()
        mock.side_effect = RuntimeError("something broke")
        router = self._make_router(mock)

        result = router.classify("error-causing query")
        assert result == DEFAULT_TIER

    def test_classify_no_similarity_score(self):
        """When RouteChoice has no similarity_score attribute, still classify by name."""
        mock = MagicMock()
        choice = MagicMock()
        choice.name = "local"
        # No similarity_score attribute at all — use getattr default
        del choice.similarity_score
        mock.return_value = choice
        router = self._make_router(mock)

        result = router.classify("simple question")
        assert result == "local"


class TestGetModel:
    """Test the get_model method."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        reset_tiers()
        yield
        reset_tiers()

    def _make_router(self, mock_router):
        router = ModelRouter()
        router._encoder = MagicMock()
        router._router = mock_router
        router._initialized = True
        return router

    def test_get_model_local(self):
        mock = MagicMock()
        mock.return_value = _make_mock_route_choice("local", 0.9)
        router = self._make_router(mock)

        model = router.get_model("simple query")
        assert model["provider"] == "custom"
        assert model["model"] == "llama3.2:3b"
        assert "base_url" in model

    def test_get_model_flash(self):
        mock = MagicMock()
        mock.return_value = _make_mock_route_choice("flash", 0.9)
        router = self._make_router(mock)

        model = router.get_model("moderate query")
        assert model["provider"] == "openrouter"
        assert model["model"] == "google/gemini-flash-1.5"

    def test_get_model_pro(self):
        mock = MagicMock()
        mock.return_value = _make_mock_route_choice("pro", 0.9)
        router = self._make_router(mock)

        model = router.get_model("complex query")
        assert model["provider"] == "anthropic"
        assert model["model"] == "claude-sonnet-4"

    def test_get_model_returns_dict(self):
        mock = MagicMock()
        mock.return_value = _make_mock_route_choice("flash", 0.9)
        router = self._make_router(mock)

        result = router.get_model("any query")
        assert isinstance(result, dict)
        assert "provider" in result
        assert "model" in result

    def test_get_model_fallback_on_low_confidence(self):
        """Low confidence should fall back to DEFAULT_TIER's model."""
        mock = MagicMock()
        mock.return_value = _make_mock_route_choice("pro", 0.2)
        router = self._make_router(mock)

        model = router.get_model("ambiguous")
        # Should be flash (DEFAULT_TIER), not pro
        assert model["provider"] == "openrouter"
        assert model["model"] == "google/gemini-flash-1.5"


class TestRouteInfo:
    """Test the route_info debug method."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        reset_tiers()
        yield
        reset_tiers()

    def _make_router(self, mock_router):
        router = ModelRouter()
        router._encoder = MagicMock()
        router._router = mock_router
        router._initialized = True
        return router

    def test_route_info_returns_all_fields(self):
        mock = MagicMock()
        mock.return_value = _make_mock_route_choice("pro", 0.95)
        router = self._make_router(mock)

        info = router.route_info("complex architecture query")
        assert info["query"] == "complex architecture query"
        assert info["tier"] == "pro"
        assert info["model"]["provider"] == "anthropic"
        assert info["model"]["model"] == "claude-sonnet-4"
        assert isinstance(info["description"], str)
        assert len(info["description"]) > 0

    def test_route_info_fallback_tier_has_description(self):
        mock = MagicMock()
        mock.return_value = None  # Force fallback
        router = self._make_router(mock)

        info = router.route_info("whatever")
        assert info["tier"] == DEFAULT_TIER
        assert len(info["description"]) > 0


class TestLazyInit:
    """Test that _initialize is called automatically and only once."""

    def test_classify_triggers_init(self):
        router = ModelRouter()
        assert router._initialized is False

        def _fake_init():
            router._encoder = MagicMock()
            router._router = MagicMock()
            router._router.return_value = _make_mock_route_choice("flash", 0.9)
            router._initialized = True

        with patch.object(router, "_initialize", side_effect=_fake_init) as spy:
            router.classify("test")
            assert spy.call_count == 1
            assert router._initialized is True

    def test_second_classify_does_not_reinit(self):
        router = ModelRouter()
        router._encoder = MagicMock()
        router._router = MagicMock()
        router._router.return_value = _make_mock_route_choice("flash", 0.9)
        router._initialized = True

        with patch.object(router, "_initialize", wraps=router._initialize) as spy:
            router.classify("first")
            router.classify("second")
            # Since already initialized, _initialize should return early
            assert spy.call_count == 2  # called but returns immediately


# ---------------------------------------------------------------------------
# Ollama integration tests (Phase 2)
# ---------------------------------------------------------------------------

class TestOllamaIntegration:
    """Test ModelRouter + OllamaManager lifecycle integration."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        from smart_router.tier import reset_tiers
        reset_tiers()
        yield
        reset_tiers()

    def _make_router(self, mock_router, ollama_mgr=None):
        """Create a ModelRouter with mocked encoder and optional ollama manager."""
        router = ModelRouter(ollama_manager=ollama_mgr)
        router._encoder = MagicMock()
        router._router = mock_router
        router._initialized = True
        return router

    def test_ollama_manager_attached(self):
        """ModelRouter stores the ollama manager reference."""
        ollama = MagicMock()
        router = ModelRouter(ollama_manager=ollama)
        assert router._ollama is ollama

    def test_ollama_manager_none_by_default(self):
        router = ModelRouter()
        assert router._ollama is None

    def test_classify_local_marks_used(self):
        """Routing to local tier calls ollama.mark_used()."""
        ollama = MagicMock()
        mock = MagicMock()
        mock.return_value = _make_mock_route_choice("local", 0.9)
        router = self._make_router(mock, ollama)

        result = router.classify("simple question")
        assert result == "local"
        ollama.mark_used.assert_called_once()

    def test_classify_non_local_checks_idle(self):
        """Routing to non-local tier calls check_idle_and_kill()."""
        ollama = MagicMock()
        mock = MagicMock()
        mock.return_value = _make_mock_route_choice("pro", 0.9)
        router = self._make_router(mock, ollama)

        result = router.classify("complex architecture")
        assert result == "pro"
        ollama.check_idle_and_kill.assert_called_once()

    def test_classify_local_no_ollama_manager(self):
        """Without ollama manager, classify local works fine."""
        mock = MagicMock()
        mock.return_value = _make_mock_route_choice("local", 0.9)
        router = self._make_router(mock)  # no ollama

        result = router.classify("simple question")
        assert result == "local"  # doesn't crash

    def test_classify_flash_checks_idle(self):
        """Routing to flash (non-local) triggers idle check."""
        ollama = MagicMock()
        mock = MagicMock()
        mock.return_value = _make_mock_route_choice("flash", 0.85)
        router = self._make_router(mock, ollama)

        result = router.classify("general question")
        assert result == "flash"
        ollama.check_idle_and_kill.assert_called_once()
        ollama.mark_used.assert_not_called()

    def test_classify_low_confidence_checks_idle(self):
        """Even with low confidence (fallback to flash), check idle."""
        ollama = MagicMock()
        mock = MagicMock()
        mock.return_value = _make_mock_route_choice("pro", 0.2)  # below threshold
        router = self._make_router(mock, ollama)

        result = router.classify("vague question")
        assert result == "flash"  # default
        ollama.check_idle_and_kill.assert_called_once()


class TestResolve:
    """Test the resolve() method that handles full lifecycle."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        from smart_router.tier import reset_tiers
        reset_tiers()
        yield
        reset_tiers()

    def _make_router(self, mock_router, ollama_mgr=None):
        router = ModelRouter(ollama_manager=ollama_mgr)
        router._encoder = MagicMock()
        router._router = mock_router
        router._initialized = True
        return router

    def test_resolve_local_ensures_running(self):
        """resolve() to local tier calls ensure_running()."""
        ollama = MagicMock()
        ollama.ensure_running.return_value = True
        mock = MagicMock()
        mock.return_value = _make_mock_route_choice("local", 0.9)
        router = self._make_router(mock, ollama)

        result = router.resolve("simple question")
        assert result["tier"] == "local"
        assert result["model"]["provider"] == "custom"
        assert result["ollama_ready"] is True
        ollama.ensure_running.assert_called_once()

    def test_resolve_local_ollama_fails(self):
        """When ensure_running fails, ollama_ready is False."""
        ollama = MagicMock()
        ollama.ensure_running.return_value = False
        mock = MagicMock()
        mock.return_value = _make_mock_route_choice("local", 0.9)
        router = self._make_router(mock, ollama)

        result = router.resolve("simple question")
        assert result["tier"] == "local"
        assert result["ollama_ready"] is False

    def test_resolve_pro_no_ollama_interaction(self):
        """resolve() to pro tier doesn't call ensure_running."""
        ollama = MagicMock()
        mock = MagicMock()
        mock.return_value = _make_mock_route_choice("pro", 0.95)
        router = self._make_router(mock, ollama)

        result = router.resolve("complex architecture")
        assert result["tier"] == "pro"
        assert result["model"]["provider"] == "anthropic"
        assert result["ollama_ready"] is None  # no local tier → no check
        ollama.ensure_running.assert_not_called()

    def test_resolve_without_ollama_manager(self):
        """resolve() works without an ollama manager."""
        mock = MagicMock()
        mock.return_value = _make_mock_route_choice("flash", 0.9)
        router = self._make_router(mock)  # no ollama

        result = router.resolve("general question")
        assert result["tier"] == "flash"
        assert result["ollama_ready"] is None
        assert "provider" in result["model"]

    def test_resolve_flash_returns_correct_model(self):
        """resolve() to flash returns deepseek flash model."""
        mock = MagicMock()
        mock.return_value = _make_mock_route_choice("flash", 0.9)
        router = self._make_router(mock)

        result = router.resolve("explain DNS")
        assert result["tier"] == "flash"
        assert result["model"] == {"provider": "openrouter", "model": "google/gemini-flash-1.5"}

    def test_resolve_returns_dict_with_required_keys(self):
        """resolve() always returns a well-formed dict."""
        mock = MagicMock()
        mock.return_value = _make_mock_route_choice("pro", 0.9)
        router = self._make_router(mock)

        result = router.resolve("any query")
        assert isinstance(result, dict)
        assert "tier" in result
        assert "model" in result
        assert "ollama_ready" in result
        assert "needs_switch" in result
        assert "provider" in result["model"]
        assert "model" in result["model"]


class TestResolveWithCurrentTier:
    """Test the resolve() method with current_tier parameter for needs_switch."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        from smart_router.tier import reset_tiers
        reset_tiers()
        yield
        reset_tiers()

    def _make_router(self, mock_router, ollama_mgr=None):
        router = ModelRouter(ollama_manager=ollama_mgr)
        router._encoder = MagicMock()
        router._router = mock_router
        router._initialized = True
        return router

    # --- needs_switch = True cases (recommended > current) ---

    def test_needs_switch_local_to_flash(self):
        """local → flash: needs_switch should be True (upgrade)."""
        mock = MagicMock()
        mock.return_value = _make_mock_route_choice("flash", 0.9)
        router = self._make_router(mock)

        result = router.resolve("explain DNS", current_tier="local")
        assert result["tier"] == "flash"
        assert result["needs_switch"] is True

    def test_needs_switch_local_to_pro(self):
        """local → pro: needs_switch should be True (big upgrade)."""
        mock = MagicMock()
        mock.return_value = _make_mock_route_choice("pro", 0.9)
        router = self._make_router(mock)

        result = router.resolve("design microservices", current_tier="local")
        assert result["tier"] == "pro"
        assert result["needs_switch"] is True

    def test_needs_switch_flash_to_pro(self):
        """flash → pro: needs_switch should be True (upgrade)."""
        mock = MagicMock()
        mock.return_value = _make_mock_route_choice("pro", 0.95)
        router = self._make_router(mock)

        result = router.resolve("architect a distributed database", current_tier="flash")
        assert result["tier"] == "pro"
        assert result["needs_switch"] is True

    # --- needs_switch = False cases (same tier or downgrade) ---

    def test_needs_switch_same_tier(self):
        """flash → flash: needs_switch should be False."""
        mock = MagicMock()
        mock.return_value = _make_mock_route_choice("flash", 0.9)
        router = self._make_router(mock)

        result = router.resolve("explain DNS", current_tier="flash")
        assert result["tier"] == "flash"
        assert result["needs_switch"] is False

    def test_needs_switch_downgrade(self):
        """pro → local: needs_switch should be True (downgrade saves cost)."""
        mock = MagicMock()
        mock.return_value = _make_mock_route_choice("local", 0.9)
        router = self._make_router(mock)

        result = router.resolve("what is 2+2", current_tier="pro")
        assert result["tier"] == "local"
        assert result["needs_switch"] is True

    def test_needs_switch_pro_to_flash(self):
        """pro → flash: needs_switch should be True (downgrade saves cost)."""
        mock = MagicMock()
        mock.return_value = _make_mock_route_choice("flash", 0.9)
        router = self._make_router(mock)

        result = router.resolve("explain DNS", current_tier="pro")
        assert result["tier"] == "flash"
        assert result["needs_switch"] is True

    def test_needs_switch_pro_same(self):
        """pro → pro: needs_switch should be False."""
        mock = MagicMock()
        mock.return_value = _make_mock_route_choice("pro", 0.95)
        router = self._make_router(mock)

        result = router.resolve("complex architecture", current_tier="pro")
        assert result["tier"] == "pro"
        assert result["needs_switch"] is False

    # --- current_tier not provided ---

    def test_no_current_tier_defaults_to_false(self):
        """When current_tier is omitted, needs_switch is always False."""
        mock = MagicMock()
        mock.return_value = _make_mock_route_choice("pro", 0.95)
        router = self._make_router(mock)

        result = router.resolve("complex architecture")
        assert result["needs_switch"] is False

    # --- Fallback to flash on low confidence still respects needs_switch ---

    def test_low_confidence_needs_switch(self):
        """Low confidence → flash. If current is local, needs_switch is True."""
        mock = MagicMock()
        mock.return_value = _make_mock_route_choice("pro", 0.2)  # below threshold
        router = self._make_router(mock)

        result = router.resolve("vague query", current_tier="local")
        assert result["tier"] == "flash"  # default
        assert result["needs_switch"] is True

    def test_low_confidence_no_needs_switch(self):
        """Low confidence → flash. Current is pro → downgrade, needs_switch is True."""
        mock = MagicMock()
        mock.return_value = _make_mock_route_choice("pro", 0.2)
        router = self._make_router(mock)

        result = router.resolve("vague query", current_tier="pro")
        assert result["tier"] == "flash"
        assert result["needs_switch"] is True

    # --- Unknown current_tier ---

    def test_unknown_current_tier(self):
        """Unknown current_tier defaults to flash for comparison."""
        mock = MagicMock()
        mock.return_value = _make_mock_route_choice("pro", 0.95)
        router = self._make_router(mock)

        result = router.resolve("complex query", current_tier="nonexistent")
        # nonexistent → flash (0), pro = 2 → needs_switch is True
        assert result["tier"] == "pro"
        assert result["needs_switch"] is True


class TestGetRouterSingleton:
    """Test the module-level get_router() singleton."""

    def setup_method(self):
        from smart_router.router import reset_router
        reset_router()

    def teardown_method(self):
        from smart_router.router import reset_router
        reset_router()

    def test_get_router_returns_model_router(self):
        from smart_router.router import get_router
        router = get_router()
        assert isinstance(router, ModelRouter)

    def test_get_router_same_instance(self):
        """Second call returns the same object (singleton)."""
        from smart_router.router import get_router
        r1 = get_router()
        r2 = get_router()
        assert r1 is r2

    def test_get_router_respects_encoder_model(self):
        """First call's encoder_model is used; subsequent calls ignore it."""
        from smart_router.router import get_router, reset_router
        reset_router()
        r1 = get_router(encoder_model="custom-embed")
        r2 = get_router(encoder_model="ignored")
        assert r1 is r2
        assert r1.encoder_model == "custom-embed"

    def test_reset_router_creates_new_instance(self):
        """After reset, get_router returns a fresh instance."""
        from smart_router.router import get_router, reset_router
        r1 = get_router()
        reset_router()
        r2 = get_router()
        assert r1 is not r2

    def test_get_router_passes_ollama_manager(self):
        """Ollama manager is stored on the singleton."""
        from smart_router.router import get_router, reset_router
        reset_router()
        ollama = MagicMock()
        r1 = get_router(ollama_manager=ollama)
        assert r1._ollama is ollama
        # Second call with different ollama is ignored (singleton already cached)
        r2 = get_router(ollama_manager=MagicMock())
        assert r2._ollama is ollama  # still the first one
