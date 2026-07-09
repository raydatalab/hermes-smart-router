---
name: hermes-smart-router
description: Intelligent model tier routing for Hermes — auto-selects local, flash, or pro per query
version: 0.1.0
author: raydatalab
license: MIT
platforms: [linux, macos, wsl]
metadata:
  hermes:
    tags: [routing, model, cost-optimization, ollama]
    homepage: https://github.com/raydatalab/hermes-smart-router
    related_skills: [hermes-cost-optimization]
---

# Smart Router

Intelligent model tier routing for Hermes Agent — picks the right model for every query based on semantic complexity.

| Tier | Typical Model | Use When |
|------|--------------|----------|
| `local` | Your Ollama model | Simple Q&A, translations, offline, private |
| `flash` | Affordable API (e.g. Gemini Flash) | General knowledge, casual coding (default) |
| `pro` | Frontier API (e.g. Claude Sonnet) | Complex reasoning, architecture, multi-step code |

## How It Works

1. User sends a query
2. Smart Router classifies it using local Ollama embeddings (no API calls, no API keys)
3. Routes to the appropriate tier — auto-starts Ollama for local, manages idle timeout
4. Agent responds using the selected model

## Prerequisites

- Hermes Agent v0.17+
- Ollama installed (the skill auto-detects models from `ollama list`; embedding model is pulled on first use)
- Python packages (install once, see testing section below):
  - `semantic-router[ollama]`
  - `smart_router` (from this repo)
- For flash/pro tiers: configure providers via `hermes model`

## Configuration

Configure via `hermes config set` for each tier or add this block under `smart_router:` in config.yaml:

```yaml
smart_router:
  enabled: true
  default_tier: flash
  encoder_model: nomic-embed-text
  tiers:
    local:
      provider: custom
      model: llama3.2:3b
      base_url: http://localhost:11434/v1
    flash:
      provider: openrouter
      model: google/gemini-flash-1.5
    pro:
      provider: anthropic
      model: claude-sonnet-4
  ollama:
    auto_start: true
    idle_timeout: 300
```

## Agent Instructions

**Note:** Hermes does not auto-install pip packages when a skill loads.
Dependencies must be installed once per environment (see Testing section below).

Before responding to each user message, classify it and switch to the best model:

```python
from smart_router.router import ModelRouter
from smart_router.ollama import OllamaManager

ollama = OllamaManager()
router = ModelRouter(ollama_manager=ollama)

decision = router.resolve(user_message)
# → {"tier": "local", "model": {"provider": "...", "model": "..."}, "ollama_ready": true}
```

- **Switch to the recommended model** using `/model <provider> <model>` with the values from `decision["model"]`. For the local tier, `resolve()` already starts Ollama — check `decision["ollama_ready"]` first. When routing away from local, the idle timeout begins automatically.
- **If the switch doesn't take effect**, you must still prepend a short hint at the top of your reply: *"💡 This query would suit the {tier} tier ({model})."* Then answer with whichever model is active — the hint alone makes the skill useful.
- **When classification has low confidence or errors out**, the router falls back to `"flash"`. Treat this the same as any flash-tier response. If `ollama_ready` is `false`, skip the local tier.

### Ollama Lifecycle

| Action | Method | Behavior |
|--------|--------|----------|
| Start | `ollama_manager.ensure_running()` | Starts `ollama serve`, waits for port, pulls model if missing |
| Check | `ollama_manager.is_running` | Checks via `ollama ps` → systemd → pgrep |
| Idle | `ollama_manager.idle_seconds` | Seconds since last local-tier use |
| Kill | `ollama_manager.ensure_killed()` | SIGTERM (SIGKILL if forced) — skips systemd-managed |

## Slash Commands

| Command | Description |
|---------|-------------|
| `/route <query>` | Show which tier would be selected (dry run) |
| `/route-stats` | Show session routing statistics |
| `/ollama start` | Manually start Ollama |
| `/ollama stop` | Manually stop Ollama |
| `/ollama status` | Check Ollama process status |
| `/tier` | Show current active tier and model |

## Testing

```bash
# Install dependencies (one-time setup)
bash scripts/install.sh

# Test routing
~/.hermes/hermes-agent/venv/bin/python3 -m smart_router route "What is the capital of France?"
~/.hermes/hermes-agent/venv/bin/python3 -m smart_router chat
~/.hermes/hermes-agent/venv/bin/python3 -m pytest tests/
```

## Notes

- First use pulls `nomic-embed-text` via Ollama (~274MB, cached).
- All classification is local — zero API calls for routing.
- Skill auto-detects your Ollama models from `ollama list`.
- Model switches are session-scoped — your config.yaml is not modified.
