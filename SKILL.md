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

When this skill is loaded:

### 1. Classify Every Query

```python
from smart_router.router import ModelRouter
from smart_router.ollama import OllamaManager

ollama = OllamaManager()
router = ModelRouter(ollama_manager=ollama)

# Full lifecycle: classify + manage Ollama + return model config
decision = router.resolve(user_message)
# → {"tier": "local", "model": {...}, "ollama_ready": true}

# Quick classify-only (no lifecycle):
tier = router.classify(user_message)  # "local", "flash", or "pro"
```

### 2. Switch Model

If the recommended tier differs from the current model:

- **Target is `local`**: `resolve()` already calls `ensure_running()`. Check `ollama_ready`.
- **Leaving `local`**: idle timeout starts automatically via `check_idle_and_kill()`.
- **Switch**: use `/model <provider> <model>` or Hermes config.

### 3. Handling Failures

- If the recommended model fails, fall back to `flash` tier.
- If `ollama_ready` is `false`, skip local tier and use flash.
- On classification error, `classify()` always returns `"flash"` (the safe default).

### 4. Ollama Lifecycle

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
