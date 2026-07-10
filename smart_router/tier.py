"""
Tier definitions, model configurations, and example utterances for each tier.

Each tier has:
- models: provider/model config dict
- utterances: example queries that should route to this tier
- description: human-readable usage description

Tiers can be overridden in two ways:
1. ``config.local.yaml`` — the author's personal override (gitignored, auto-loaded)
2. ``smart_router.tiers`` in Hermes config.yaml — user configuration

Generic defaults ship in the repo. The author's personal config lives in
config.local.yaml and overrides these at import time.
"""

import logging
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default tiers — vendor-neutral, generic models suitable for any user.
# Override via config.local.yaml (author) or Hermes config.yaml (everyone).
DEFAULT_TIERS = {
    "local": {
        "models": {"provider": "custom", "model": "llama3.2:3b", "base_url": "http://localhost:11434/v1"},
        "utterances": [
            # Simple Q&A / facts
            "What is the capital of France?",
            "How many days are in a year?",
            "What color is the sky?",
            "When was Einstein born?",
            "What is 2 plus 2?",
            # Greetings / chitchat
            "Hello, how are you?",
            "Good morning!",
            "Thanks for your help",
            # Translation
            "Translate hello to Chinese",
            "What does 'bonjour' mean?",
            "Say good morning in Spanish",
            # Time / basic info
            "What time is it?",
            "What day is it today?",
            "What's the weather like?",
            # Simple commands
            "Write a short greeting message",
            "Create a simple todo list",
            "Format this text",
            # Chinese simple queries
            "你好吗",
            "今天天气怎么样",
            "现在是几点了",
            "巴黎是哪个国家的首都",
        ],
        "description": "Simple lookups, translations, greetings, basic Q&A — safe for small local models",
    },
    "flash": {
        "models": {"provider": "openrouter", "model": "google/gemini-flash-1.5"},
        "utterances": [
            # General knowledge
            "Explain how DNS works",
            "What is the difference between TCP and UDP?",
            "How does garbage collection work in Python?",
            "What is REST API design best practices?",
            # Coding tasks (simple, single-function, self-contained)
            "Write a short Python function to check if a string is a palindrome",
            "How do I parse a CSV file in Python?",
            "Write a simple regex to validate email addresses",
            "Write a SQL query to find the top 5 customers by order count",
            "Create a basic HTML form with name and email fields",
            "Fix this off-by-one error in my for loop",
            "What is the correct way to read a file line by line in Python?",
            "How do I add a CSS class to an element in vanilla JavaScript?",
            # Summarization / analysis
            "Summarize this article for me",
            "What are the key points in this document?",
            "Compare Python and JavaScript for web development",
            # Explanations
            "What is machine learning in simple terms?",
            "Explain how HTTPS works",
            "What is the difference between SQL and NoSQL?",
            # Chinese general queries
            "请解释一下区块链的工作原理",
            "Python和Java有什么区别",
            "如何搭建一个个人博客网站",
        ],
        "description": "General knowledge, casual coding, explanations — default tier for most queries",
    },
    "pro": {
        "models": {"provider": "anthropic", "model": "claude-sonnet-4"},
        "utterances": [
            # Complex architecture
            "Design a microservice architecture for an e-commerce platform",
            "Design a fault-tolerant distributed database system",
            "Architect a real-time chat system for 10 million users",
            # Complex debugging
            "Debug this distributed system race condition",
            "Analyze this complex error trace from a production outage",
            "Find the memory leak in this multi-threaded application",
            # Advanced coding (complex "Write/Implement/Build" — scope distinguishes from flash)
            "Write a complete authentication system with JWT, refresh tokens, and OAuth2",
            "Implement a custom memory allocator in C",
            "Build a full-text search engine from scratch",
            "Write a distributed consensus algorithm (Raft) implementation",
            "Write a production-grade rate limiter with sliding window algorithm and Redis backend",
            "Implement a thread-safe connection pool with retry logic and circuit breaker",
            "Build a real-time collaborative editing engine with operational transforms and CRDT",
            "Write a complete compiler from a subset of Python to WebAssembly",
            "Implement a distributed task scheduler with priority queues and dead letter handling",
            "Build an event sourcing and CQRS system for a banking ledger",
            # Math / algorithms
            "Prove the Riemann hypothesis implications for prime number distribution",
            "Design an algorithm for real-time fraud detection at scale",
            "Optimize this NP-hard problem with approximation algorithms",
            # Complex analysis
            "Analyze this business case and provide a detailed financial model",
            "Perform a thorough security audit of this authentication flow",
            # Multi-step reasoning
            "I need to migrate from monolith to microservices, plan the entire migration",
            "Design a CI/CD pipeline with blue-green deployment for a Kubernetes cluster",
            # Chinese complex queries
            "设计一个支持十亿用户的高并发秒杀系统",
            "分析这段分布式系统的死锁问题并提供修复方案",
            "实现一个完整的Raft共识算法，包括领导者选举和日志复制",
        ],
        "description": "Complex code, architecture, multi-step reasoning, advanced math — highest quality model needed",
    },
}

# The fallback tier when confidence is low or classification fails
DEFAULT_TIER = "flash"

# Confidence threshold below which we fall back to default
MIN_CONFIDENCE = 0.6

# Queries shorter than this skip embedding and return DEFAULT_TIER.
# Greetings, one-liners, and simple lookups don't need semantic routing.
SHORT_QUERY_THRESHOLD = 20

# Active configuration — runtime-mutable copy, overridden by config.local.yaml
# and smart_router.tiers. Deep copy prevents mutations from leaking into
# DEFAULT_TIERS (which must stay pristine as the reference template).
CONFIG = deepcopy(DEFAULT_TIERS)


def set_tiers(tiers: dict) -> None:
    """Override tier definitions with user config from Hermes config.yaml.

    Args:
        tiers: A dict matching the shape of DEFAULT_TIERS.
               Only the keys provided will be replaced.
               Set a tier to None to disable it.
    """
    global CONFIG
    if not tiers:
        return
    for tier_name, tier_config in tiers.items():
        if tier_config is None:
            CONFIG.pop(tier_name, None)
        else:
            CONFIG[tier_name] = tier_config


def get_tiers() -> dict:
    """Return the currently active tier configuration."""
    return dict(CONFIG)


def get_tier(name: str) -> dict | None:
    """Get a single tier config by name. Returns None if not found."""
    return CONFIG.get(name)


def reset_tiers() -> None:
    """Reset to default tier configuration. Mutates in place so TIERS alias stays valid."""
    global CONFIG
    CONFIG.clear()
    CONFIG.update(deepcopy(DEFAULT_TIERS))


# Convenience alias — router.py imports this. Since dict is mutable, mutations
# via set_tiers() are visible through TIERS. reset_tiers() mutates in place
# so the alias stays valid.
TIERS = CONFIG


# ---------------------------------------------------------------------------
# Local model auto-detection
# ---------------------------------------------------------------------------

def _detect_ollama_model() -> Optional[str]:
    """Return the first available Ollama chat model, skipping embedding models."""
    # Embedding models to skip — these aren't chat models
    _embedding_models = {"nomic-embed-text", "mxbai-embed-large", "all-minilm"}

    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return None
        lines = result.stdout.strip().split("\n")
        if len(lines) < 2:  # header only, no models
            return None
        for line in lines[1:]:
            parts = line.split()
            if parts and parts[0] not in _embedding_models:
                return parts[0]
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _apply_auto_detection() -> None:
    """Auto-detect local Ollama model and fill in the local tier config.

    Called once at import time. If the user hasn't explicitly configured
    a local model, probes ``ollama list`` and uses whatever is available.
    Does nothing if ollama is not installed or no models are found.
    """
    detected = _detect_ollama_model()
    if detected and "local" in CONFIG:
        current_model = CONFIG["local"]["models"]["model"]
        # Only auto-detect if the model is still the generic default
        # (a user override or config.local.yaml takes precedence)
        if current_model == DEFAULT_TIERS["local"]["models"]["model"]:
            CONFIG["local"]["models"]["model"] = detected
            logger.info(f"Auto-detected local Ollama model: {detected}")


# ---------------------------------------------------------------------------
# config.local.yaml loading (author's personal override)
# ---------------------------------------------------------------------------

def _load_local_config() -> None:
    """Load config.local.yaml if it exists.

    This is the mechanism that lets contributors keep their personal config
    on disk without it ever reaching GitHub. The file is .gitignored —
    strangers who clone see only the generic defaults.
    """
    local_path = Path(__file__).resolve().parent.parent / "config.local.yaml"
    if not local_path.exists():
        return

    try:
        import yaml
    except ImportError:
        logger.debug("PyYAML not installed — skipping config.local.yaml")
        return

    try:
        with open(local_path) as f:
            config = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        logger.warning(f"Failed to read config.local.yaml: {e}")
        return

    if not config or "tiers" not in config:
        return

    # Merge each tier override into CONFIG
    for tier_name, tier_config in config["tiers"].items():
        if tier_config is None:
            CONFIG.pop(tier_name, None)
            logger.info(f"config.local.yaml: disabled tier '{tier_name}'")
        elif isinstance(tier_config, dict) and "provider" in tier_config:
            # config.local.yaml uses flat {provider, model, base_url?}
            CONFIG[tier_name]["models"] = {
                "provider": tier_config["provider"],
                "model": tier_config["model"],
            }
            if "base_url" in tier_config:
                CONFIG[tier_name]["models"]["base_url"] = tier_config["base_url"]
            logger.info(
                f"config.local.yaml: tier '{tier_name}' → "
                f"{tier_config['provider']}/{tier_config['model']}"
            )


_config_loaded = False


def ensure_config_loaded() -> None:
    """Apply local config overrides and auto-detection, once per process.

    Called lazily by ModelRouter on first use — not at import time.
    This keeps DEFAULT_TIERS pristine for tests that import tier.py
    directly, while still applying the author's config.local.yaml when
    the router actually runs.
    """
    global _config_loaded
    if _config_loaded:
        return
    _config_loaded = True
    _load_local_config()
    _apply_auto_detection()
