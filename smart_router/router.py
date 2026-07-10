"""
Core routing engine — wraps semantic-router for query classification.

Uses OllamaEncoder with nomic-embed-text for 100% local, µs-level classification.
Zero network dependency — runs on local Ollama, no SSL issues, no model downloads.
"""

import logging
import re
from typing import Optional, TYPE_CHECKING

from semantic_router import Route, SemanticRouter
from semantic_router.encoders import OllamaEncoder

from .tier import TIERS, DEFAULT_TIER, MIN_CONFIDENCE, SHORT_QUERY_THRESHOLD, ensure_config_loaded

if TYPE_CHECKING:
    from .ollama import OllamaManager

logger = logging.getLogger(__name__)

# Tier ordering for needs_switch comparison: higher index = more capable.
_TIER_ORDER = {"local": 0, "flash": 1, "pro": 2}

# Fast-path patterns: cheap regex checks before expensive embedding.
# Each tuple is (compiled_regex, recommended_tier).
# Greetings and one-word queries don't need semantic routing.
_FAST_PATH_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Greetings / chitchat — any tier works, flash is safe default
    (re.compile(r"^(hi|hello|hey|thanks|ok|okay|yes|no|bye|yo|good\s*(morning|afternoon|evening|night))[.!，。！？?]*$", re.IGNORECASE), "flash"),
    # Chinese greetings
    (re.compile(r"^(你好|您好|谢谢|再见|早|嗯|哦|好|行|对|是的|不是)[！!。.]*$"), "flash"),
    # Trivial definition lookups — flash is enough
    (re.compile(r"^(what|who)\s+(is|are|was|were)\s+(a|an|the)\s+\w+[?!.]*$", re.IGNORECASE), "flash"),
    # Single-word queries (max 15 chars — longer is not a trivial query)
    (re.compile(r"^[a-zA-Z]{1,15}$"), "flash"),
    # Chinese single-character queries
    (re.compile(r"^[一-鿿]{1,4}$"), "flash"),
]

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

    # ------------------------------------------------------------------
    # Fast-path: avoid expensive embedding for trivially-classifiable queries
    # ------------------------------------------------------------------

    @staticmethod
    def _fast_path(query: str, current_tier: Optional[str] = None) -> Optional[dict]:
        """Try to classify without embedding.

        Returns a routing dict if the query matches a fast-path pattern,
        otherwise None (caller should proceed with full classification).

        The returned dict has the same shape as ``resolve()`` output:
        ``{tier, model, ollama_ready, needs_switch, reason, recommendation}``
        but ``model`` and ``ollama_ready`` are filled in by the caller.
        """
        stripped = query.strip()

        # 1. Pattern match first: greetings, single words, definitions.
        #    These are trivially answerable by flash regardless of length.
        for pattern, recommended_tier in _FAST_PATH_PATTERNS:
            if pattern.match(stripped.lower()):
                needs_switch = False
                direction = None
                if current_tier is not None:
                    rec_idx = _TIER_ORDER.get(recommended_tier, _TIER_ORDER[DEFAULT_TIER])
                    cur_idx = _TIER_ORDER.get(current_tier, _TIER_ORDER[DEFAULT_TIER])
                    needs_switch = rec_idx != cur_idx
                    direction = "downgrade" if rec_idx < cur_idx else "upgrade"

                reason = (
                    f"Fast-path: pattern match — {recommended_tier} tier is sufficient "
                    f"for this query"
                )
                if needs_switch and direction:
                    reason += f" ({direction} from {current_tier})"

                return {
                    "tier": recommended_tier,
                    "needs_switch": needs_switch,
                    "reason": reason,
                    "recommendation": None,
                }

        # 2. Length check: queries under SHORT_QUERY_THRESHOLD chars skip embedding.
        #    Patterns already checked — whatever's left short is probably a typo
        #    or abbreviation.  Keep current tier to be safe.
        if len(stripped) < SHORT_QUERY_THRESHOLD:
            tier = current_tier or DEFAULT_TIER
            return {
                "tier": tier,
                "needs_switch": False,
                "reason": (
                    f"Fast-path: short query ({len(stripped)} chars) — "
                    f"keeping current {tier} tier, no routing needed"
                ),
                "recommendation": None,
            }

        return None  # Not fast-path eligible

    # ------------------------------------------------------------------
    # Recommendation builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_recommendation(
        tier: str,
        current_tier: Optional[str],
        needs_switch: bool,
        direction: Optional[str],
    ) -> Optional[str]:
        """Build a ready-to-paste switch recommendation string.

        Returns None if no switch is needed — the agent should say nothing.
        When a switch IS needed, returns a one-liner the agent drops into
        its response verbatim.  Zero formatting work for the agent.
        """
        if not needs_switch or current_tier is None:
            return None

        tier_config = TIERS.get(tier, TIERS[DEFAULT_TIER])
        model = tier_config["models"]
        provider = model["provider"]
        model_name = model["model"]

        if direction == "downgrade":
            return (
                f"💡 Downgrade to {tier}: `/model {provider} {model_name}` "
                f"— save costs, this query doesn't need {current_tier}"
            )
        elif direction == "upgrade":
            return (
                f"💡 Switch to {tier}: `/model {provider} {model_name}` "
                f"— {tier_config.get('description', 'better quality for complex query')}"
            )
        else:
            return f"💡 Switch to {tier}: `/model {provider} {model_name}`"

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
        stripped = query.strip()

        # Fast-path: skip embedding for short queries and trivial patterns.
        fast = self._fast_path(stripped)
        if fast is not None:
            tier = fast["tier"]
            logger.debug(f"Fast-path routed '{stripped[:40]}...' → {tier}")
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
                "recommendation": str | None,   # Ready-to-paste switch hint
            }
        """
        stripped = query.strip()

        # 1. Fast-path: skip embedding for trivially-classifiable queries.
        fast = self._fast_path(stripped, current_tier=current_tier)
        if fast is not None:
            tier = fast["tier"]
            needs_switch = fast["needs_switch"]
            reason = fast["reason"]
            # Build recommendation even for fast-path hits (e.g. pro user says "hi")
            direction: Optional[str] = None
            if needs_switch and current_tier is not None:
                rec_idx = _TIER_ORDER.get(tier, _TIER_ORDER[DEFAULT_TIER])
                cur_idx = _TIER_ORDER.get(current_tier, _TIER_ORDER[DEFAULT_TIER])
                direction = "upgrade" if rec_idx > cur_idx else "downgrade"
            recommendation = self._build_recommendation(tier, current_tier, needs_switch, direction)
        else:
            # 2. Full semantic classification.
            tier = self.classify(stripped)

            # 3. Compute needs_switch and direction.
            needs_switch = False
            direction: Optional[str] = None
            if current_tier is not None:
                recommended_idx = _TIER_ORDER.get(tier, _TIER_ORDER[DEFAULT_TIER])
                current_idx = _TIER_ORDER.get(current_tier, _TIER_ORDER[DEFAULT_TIER])
                needs_switch = recommended_idx != current_idx
                if needs_switch:
                    direction = "upgrade" if recommended_idx > current_idx else "downgrade"

            # 4. Build enriched reason.
            tier_config = TIERS.get(tier, TIERS.get(DEFAULT_TIER, {}))
            tier_desc = tier_config.get("description", "")

            if needs_switch and direction:
                if direction == "downgrade":
                    reason = (
                        f"Downgrade from {current_tier} to {tier} — "
                        f"{tier_desc} (save costs — this query doesn't need {current_tier})"
                    )
                else:
                    reason = (
                        f"Upgrade from {current_tier} to {tier} — "
                        f"{tier_desc} (matched via semantic routing)"
                    )
            else:
                verb = "keeping" if current_tier == tier else "classified as"
                reason = f"Current {tier} tier is appropriate — {tier_desc}"

            recommendation = self._build_recommendation(tier, current_tier, needs_switch, direction)

        # 5. Lifecycle: ensure Ollama is running when routing to local.
        ollama_ready: Optional[bool] = None
        if self._ollama is not None and tier == "local":
            ollama_ready = self._ollama.ensure_running()

        return {
            "tier": tier,
            "model": self._get_model_for_tier(tier),
            "ollama_ready": ollama_ready,
            "needs_switch": needs_switch,
            "reason": reason,
            "recommendation": recommendation,
        }

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
