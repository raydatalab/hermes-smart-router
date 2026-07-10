---
name: hermes-smart-router
description: Intelligent model tier routing for Hermes — auto-selects local, flash, or pro per query
version: 0.1.5
author: raydatalab
license: MIT
platforms: [linux, macos, wsl]
triggers:
  - model switch
  - change model
  - /model
  - /tier
  - switch to
  - use local
  - use flash
  - use pro
  - too slow
  - wasting tokens
  - cheaper model
  - downgrade
  - upgrade model
  - routing
  - tier selection
  - local model
  - what model
  - which model
  - speed up
  - save cost
  - 切换模型
  - 太慢了
  - 换模型
  - 用什么模型
metadata:
  hermes:
    tags: [routing, model, cost-optimization, ollama]
    homepage: https://github.com/raydatalab/hermes-smart-router
    related_skills: [hermes-cost-optimization]
---

# Smart Router

Intelligent model tier routing for Hermes Agent — recommends the right model tier based on query complexity.

## Why Use Me

Every query you send to the wrong model tier wastes tokens (pro for a greeting)
or produces weak output (local for architecture). Smart Router catches these
mismatches automatically:

- **Save money** — downgrade from pro/flash to local on simple queries
- **Get quality** — upgrade from local/flash to pro on complex reasoning
- **100% local, free, private** — all classification runs on your machine via
  Ollama embeddings. Zero API calls for routing. Zero setup after install.

The router is a bonus — when called, it saves tokens and improves quality.
When skipped, you never notice. There is no penalty for not using it.

## CRITICAL — Read Before Responding

Before responding to any non-trivial query:

1. **Decide if routing is relevant.**
   - Route if the response needs 3+ tool calls, complex reasoning,
     architecture, multi-step code, or you're unsure about the current tier.
   - Skip for greetings, thanks, translations, one-liners, basic facts.

2. **Run the router** (via execute_code):
   ```python
   from smart_router.router import get_router

   router = get_router()
   decision = router.resolve(user_message, current_tier=ACTIVE_TIER)
   # → {"tier": "pro", "model": {...}, "ollama_ready": null,
   #    "needs_switch": true, "reason": "Query classified as 'pro' — ..."}
   ```

3. **Act on the result:**
   - `needs_switch == false` → answer normally, don't mention routing.
   - `needs_switch == true` → prepend a switch recommendation (see below).
   - If the call errors or times out → skip, answer with current model.

**ACTIVE_TIER** is your current tier (local/flash/pro). Check with `/tier`
or ask the user if you don't know. Fast queries under 20 characters skip
embedding and return instantly — it is safe to call on every message.

## When to Route (Two Heuristics)

**Heuristic 1:** Before responding to any query that requires 3+ tool calls,
complex reasoning, architecture design, multi-step code, or debugging — run
the classifier. Also route when you are unsure whether the current tier fits.

**Heuristic 2:** Skip routing for simple greetings, translations, one-liner
facts, single-file lookups, and queries that trivially fit the current tier.

That's it. Two rules. If in doubt, route — the first call takes ~3 seconds,
subsequent calls return in microseconds.

## How to Route

Use the module singleton — init cost is paid once per session:

```python
from smart_router.router import get_router

router = get_router()
decision = router.resolve(user_message, current_tier="flash")  # your active tier
# → {"tier": "pro", "model": {...}, "ollama_ready": null,
#    "needs_switch": true, "reason": "Query classified as 'pro' — Complex code..."}
```

`current_tier` must be one of `"local"`, `"flash"`, or `"pro"` — match
whatever `/tier` reports. The `reason` field explains why the tier was chosen
and includes direction ("upgrade from flash") when a switch is needed.

## How to Act on the Result

You **cannot execute `/model` yourself** — it is a user-side slash command.
Instead, prepend a one-line recommendation to your reply:

- **`needs_switch` is `false`** → say nothing, just answer.
- **`needs_switch` is `true` and tier is an upgrade** (local→flash, local→pro, flash→pro):
  *"💡 Switch to {tier}: `/model {provider} {model}`"*
- **`needs_switch` is `true` and tier is a downgrade** (pro→flash, flash→local, pro→local):
  *"💡 Downgrade to {tier}: `/model {provider} {model}`"*
- For local tier: check `decision["ollama_ready"]` first — if `false`, mention that Ollama isn't ready.
- **Classification errors out or times out** → skip, answer with current model. Do not retry.

Check `decision["reason"]` for context — it tells you why the tier was chosen
and whether it's an upgrade or downgrade.

## Tier Reference

| Tier | Typical Model | Use When |
|------|--------------|----------|
| `local` | Your Ollama model | Simple Q&A, translations, offline, private |
| `flash` | Affordable API (e.g. Gemini Flash) | General knowledge, casual coding (default) |
| `pro` | Frontier API (e.g. Claude Sonnet) | Complex reasoning, architecture, multi-step code |

### How It Works

1. User sends a query
2. Smart Router classifies it using local Ollama embeddings (no API calls, no API keys)
3. Routes to the appropriate tier — auto-starts Ollama for local, manages idle timeout
4. Agent responds using the selected model

### Ollama Lifecycle

| Action | Method | Behavior |
|--------|--------|----------|
| Start | `ollama_manager.ensure_running()` | Starts `ollama serve`, waits for port, pulls model if missing |
| Check | `ollama_manager.is_running` | Checks via `ollama ps` → systemd → pgrep |
| Idle | `ollama_manager.idle_seconds` | Seconds since last local-tier use |
| Kill | `ollama_manager.ensure_killed()` | SIGTERM (SIGKILL if forced) — skips systemd-managed |

## Prerequisites

- Hermes Agent v0.17+
- Ollama installed (the skill auto-detects models from `ollama list`; embedding model is pulled on first use)
- Python packages (install once, see Testing section below):
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
- Fast-path: queries under 20 characters skip embedding and return the default tier instantly.
