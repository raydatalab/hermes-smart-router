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

Intelligent model tier routing for Hermes Agent — recommends the right model tier based on query complexity.

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

### When to Classify

**Do NOT classify every message.** The router is a tool you invoke when the
current model might not be the right fit. `needs_switch` fires in both
directions — upgrade when the model is too weak, downgrade when it's overkill.

| Situation | Action |
|-----------|--------|
| Current model is **local** and query involves reasoning, architecture, or multi-step code | Classify — likely needs flash or pro (upgrade) |
| Current model is **flash** and query is deep architecture, complex debugging, or math proofs | Classify — likely needs pro (upgrade) |
| Current model is **pro** and query is simple (greeting, translation, basic fact) | Classify — likely needs flash or local (downgrade) |
| Current model is **flash** and query is very simple (greeting, translation, basic fact) | Classify — likely needs local (downgrade) |
| Query complexity obviously matches current tier | Skip — no mismatch expected |
| User explicitly asks about tier or model choice | Classify — user wants the info |
| You're unsure | Classify — cheap check |

The rule: **classify when the current tier feels wrong for the query**, either
too weak or too expensive. Skip when it's obviously right.

### How to Classify

Use the module singleton — init cost is paid once per session:

```python
from smart_router.router import get_router

router = get_router()
decision = router.resolve(user_message, current_tier="flash")  # use your active tier
# → {"tier": "pro", "model": {...}, "ollama_ready": null, "needs_switch": true}
```

`current_tier` must be one of `"local"`, `"flash"`, or `"pro"` — match whatever `/tier` reports.

### How to Act on the Result

You **cannot execute `/model` yourself** — it is a user-side slash command.
Instead, prepend a one-line recommendation to your reply:

- **`needs_switch` is `false`** → say nothing, just answer.
- **`needs_switch` is `true` and tier is an upgrade** (local→flash, local→pro, flash→pro):
  *"💡 Switch to {tier}: `/model {provider} {model}`"*
- **`needs_switch` is `true` and tier is a downgrade** (pro→flash, flash→local, pro→local):
  *"💡 Downgrade to {tier}: `/model {provider} {model}`"*
- For local tier: check `decision["ollama_ready"]` first — if `false`, mention that Ollama isn't ready.
- **Classification errors out or times out** → skip, answer with current model. Do not retry.

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
