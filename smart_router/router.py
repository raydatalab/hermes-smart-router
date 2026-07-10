"""
Core routing engine — wraps semantic-router for query classification.

Uses OllamaEncoder with nomic-embed-text for 100% local, µs-level classification.
Zero network dependency — runs on local Ollama, no SSL issues, no model downloads.
"""

import logging
from typing import Optional, TYPE_CHECKING

from semantic_router import Route, SemanticRouter
from semantic_router.encoders import OllamaEncoder

from .tier import TIERS, DEFAULT_TIER, MIN_CONFIDENCE, SHORT_QUERY_THRESHOLD, ensure_config_loaded

if TYPE_CHECKING:
    from .ollama import OllamaManager

logger = logging.getLogger(__name__)

# Tier ordering for needs_switch comparison: higher index = more capable.
_TIER_ORDER = {"local": 0, "flash": 1, "pro": 2}

# Module-level singleton — agent pays init cost once per session.
_router_instance: Optional["ModelRouter"] = None


def get_router(
    encoder_model: str = "nomic-embed-text",
    ollama_manager: "Optional[OllamaManager]" = None,
) -> "ModelRouter":
    """Return the module-level ModelRouter singleton.

    The first call initializes the router (OllamaEncoder + config loading,
    ~2-3s). Subsequent calls return the cached instance in microseconds.
    """
    global _router_instance
    if _router_instance is None:
        _router_instance = ModelRouter(
            encoder_model=encoder_model,
            ollama_manager=ollama_manager,
        )
    return _router_instance


def reset_router() -> None:
    """Reset the singleton (for tests)."""
    global _router_instance
    _router_instance = None


class ModelRouter:
    """Routes queries to the optimal model tier using semantic similarity."""

    def __init__(
        self,
        encoder_model: str = "nomic-embed-text",
        ollama_manager: "Optional[OllamaManager]" = None,
    ):
        # Apply local config overrides and auto-detection once per process.
        # Lazy — not at tier.py import time — so tests see raw DEFAULT_TIERS.
        ensure_config_loaded()

        self.encoder_model = encoder_model
        self._encoder: Optional[OllamaEncoder] = None
        self._router: Optional[SemanticRouter] = None
        self._initialized = False
        self._ollama = ollama_manager

    def _initialize(self):
        """Lazy-init encoder and routes. Uses local Ollama — zero network needed."""
        if self._initialized:
            return

        logger.info(f"Initializing Smart Router with Ollama encoder: {self.encoder_model}")
        try:
            self._encoder = OllamaEncoder(name=self.encoder_model)
        except Exception as e:
            msg = (
                f"Failed to initialize Ollama encoder with model '{self.encoder_model}'. "
                f"Is it pulled? Run: ollama pull {self.encoder_model}\n"
                f"Original error: {e}"
            )
            logger.error(msg)
            raise RuntimeError(msg) from e

        routes = []
        for tier_name, tier_config in TIERS.items():
            route = Route(
                name=tier_name,
                utterances=tier_config["utterances"],
                description=tier_config.get("description", ""),
            )
            routes.append(route)
            logger.debug(f"  Route '{tier_name}': {len(tier_config['utterances'])} utterances")

        self._router = SemanticRouter(encoder=self._encoder, routes=routes, auto_sync="local")
        self._initialized = True
        logger.info(f"Smart Router ready: {len(routes)} tiers, encoder={self.encoder_model} (Ollama)")

    def classify(self, query: str) -> str:
        """
        Classify a query into a tier.

        If an OllamaManager is attached, this method automatically marks
        the local tier as used when routing there.

        Args:
            query: The user's input text.

        Returns:
            One of: "local", "flash", "pro". Defaults to DEFAULT_TIER on low
            confidence or error.
        """
        # Fast-path: queries under SHORT_QUERY_THRESHOLD chars skip embedding.
        # Greetings, yes/no, short lookups don't need semantic routing.
        stripped = query.strip()
        if len(stripped) < SHORT_QUERY_THRESHOLD:
            logger.debug(
                f"Short query ({len(stripped)} chars) — fast-path to default '{DEFAULT_TIER}'"
            )
            tier = DEFAULT_TIER
            if self._ollama is not None:
                if tier == "local":
                    self._ollama.mark_used()
                else:
                    self._ollama.check_idle_and_kill()
            return tier

        self._initialize()

        try:
            result = self._router(query)

            if result is None:
                logger.debug(f"No route matched for query, defaulting to '{DEFAULT_TIER}'")
                tier = DEFAULT_TIER
            else:
                tier = result.name
                score = getattr(result, "similarity_score", None)

                # Confidence check: fall back when score is below threshold
                if score is not None and score < MIN_CONFIDENCE:
                    logger.debug(
                        f"Low confidence ({score:.3f} < {MIN_CONFIDENCE}) for tier "
                        f"'{tier}', defaulting to '{DEFAULT_TIER}'"
                    )
                    tier = DEFAULT_TIER

                elif tier not in TIERS:
                    logger.warning(f"Unknown tier '{tier}' returned, defaulting to '{DEFAULT_TIER}'")
                    tier = DEFAULT_TIER

                else:
                    score_str = f"{score:.3f}" if score is not None else "N/A"
                    logger.debug(f"Routed '{query[:60]}...' → {tier} (score={score_str})")

            # --- Ollama lifecycle integration ---
            if self._ollama is not None:
                if tier == "local":
                    self._ollama.mark_used()
                else:
                    # Query is going to a non-local tier — check if we
                    # should shut down idle Ollama
                    self._ollama.check_idle_and_kill()

            return tier

        except Exception as e:
            logger.error(f"Classification failed: {e}", exc_info=True)
            return DEFAULT_TIER

    def resolve(self, query: str, current_tier: Optional[str] = None) -> dict:
        """
        Full routing resolution: classify + lifecycle management + model config.

        When routing to the local tier this ensures Ollama is running (if a
        manager is attached). When routing away it starts the idle kill timer.

        Args:
            query: The user's input text.
            current_tier: The tier the agent is currently using (local/flash/pro).
                When provided, the result includes a ``needs_switch`` boolean
                that is True when the recommended tier differs from the
                current one (upgrade or downgrade). Omit to skip comparison.

        Returns:
            {
                "tier": str,
                "model": {"provider": ..., "model": ...},
                "ollama_ready": bool | None,   # None if no ollama manager
                "needs_switch": bool,           # True when recommended != current
                "reason": str,                  # Human-readable explanation
            }
        """
        tier = self.classify(query)

        ollama_ready: Optional[bool] = None
        if self._ollama is not None and tier == "local":
            ollama_ready = self._ollama.ensure_running()

        # Build a human-readable reason from the tier description
        tier_config = TIERS.get(tier, TIERS.get(DEFAULT_TIER, {}))
        tier_desc = tier_config.get("description", "")
        reason = f"Query classified as '{tier}' — {tier_desc}"

        result: dict = {
            "tier": tier,
            "model": self._get_model_for_tier(tier),
            "ollama_ready": ollama_ready,
            "reason": reason,
        }

        if current_tier is not None:
            recommended = _TIER_ORDER.get(tier, _TIER_ORDER[DEFAULT_TIER])
            current = _TIER_ORDER.get(current_tier, _TIER_ORDER[DEFAULT_TIER])
            needs_switch = recommended != current
            result["needs_switch"] = needs_switch
            if needs_switch and current_tier in _TIER_ORDER:
                direction = "upgrade" if recommended > current else "downgrade"
                result["reason"] = reason + f" ({direction} from '{current_tier}')"
        else:
            result["needs_switch"] = False

        return result

    def get_model(self, query: str) -> dict:
        """
        Classify query and return the model configuration dict.

        Returns:
            {provider: str, model: str, base_url?: str}
        """
        tier = self.classify(query)
        return self._get_model_for_tier(tier)

    def _get_model_for_tier(self, tier: str) -> dict:
        """Get model config from tier name with safe fallback."""
        config = TIERS.get(tier, TIERS[DEFAULT_TIER])
        return dict(config["models"])

    @property
    def available_tiers(self) -> list[str]:
        """Return list of available tier names."""
        return list(TIERS.keys())

    def route_info(self, query: str) -> dict:
        """
        Return full routing info for a query (dry-run / debug).

        Returns:
            {query, tier, model_config, tier_description}
        """
        tier = self.classify(query)  # handles lazy init + fast-path internally
        config = TIERS.get(tier, TIERS[DEFAULT_TIER])
        return {
            "query": query,
            "tier": tier,
            "model": config["models"],
            "description": config.get("description", ""),
        }
