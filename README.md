# Hermes Smart Router

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Hermes-v0.17+-purple.svg" alt="Hermes">
</p>

Intelligent model tier routing for Hermes Agent — automatically selects the appropriate model for each query.

## The Problem

A user may have access to multiple model tiers: a local model (fast, free, offline), a low-cost cloud model, and a more capable frontier model. Without routing, every query goes to the same target, incurring unnecessary cost on simple tasks and under-serving complex ones.

## The Solution

A Hermes skill that classifies each query and routes it to the appropriate model tier. Classification runs entirely on the local machine:

```
"Translate hello to German"           → local (Ollama, free)
"Explain how DNS works"               → flash (low-cost API)
"Design a multi-region database"      → pro (highest capability)
```

Routing uses [semantic-router](https://github.com/aurelio-labs/semantic-router) with local Ollama embeddings — zero API calls, zero cost. After the one-time embedding model pull, classification runs entirely offline.

## Tiers

| Tier | Purpose | Provider examples |
|------|---------|-------------------|
| **local** | Free, offline, private — simple lookups, translations, formatting | Ollama (llama3.2, qwen3, mistral), any local model |
| **flash** | Fast, low-cost — everyday coding, explanations, general Q&A | DeepSeek Flash, Gemini Flash, GPT-4o-mini, Claude Haiku |
| **pro** | Highest capability — complex architecture, debugging, multi-step reasoning | DeepSeek Pro, Gemini Pro, GPT-4o, Claude Sonnet |

One API token, different models for different complexity. There is no requirement for multiple providers — a single provider such as OpenRouter or DeepSeek provides both flash and pro tiers.

## Features

- 3-tier routing: local / flash / pro, auto-selected per query
- 100% local classification via Ollama embeddings — no cloud dependencies
- Ollama lifecycle: auto-start when needed, auto-kill when idle
- No GPU required — runs on CPU
- Queries never leave the local machine during routing decisions

## Installation

**Prerequisites:** Hermes Agent v0.17+, Python 3.10+, Ollama installed.

### Via Hermes (recommended)

```bash
hermes skills install hermes-smart-router
```

If the short form is unavailable, use the repo path:

```bash
hermes skills install raydatalab/hermes-smart-router/hermes-smart-router
```

On first use, run the install script to set up dependencies — `pip install` + embedding model pull. This takes ~30 seconds once per environment.

### Manual

```bash
git clone https://github.com/raydatalab/hermes-smart-router.git
cd hermes-smart-router && bash scripts/install.sh
hermes skills install SKILL.md
```

On first use, Smart Router detects available Ollama models from `ollama list` and adapts. If no cloud providers are configured, the skill operates in local-only mode — no errors, no missing-config warnings.

## Configuration

Configure via `hermes config set` for each tier, or add a `smart_router:` block in config.yaml with the following structure:

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
      provider: openrouter
      model: anthropic/claude-sonnet-4
  ollama:
    auto_start: true
    idle_timeout: 300
```

Providers are configured via `hermes model` — no manual `.env` editing is required.

## Provider Support

| Tier | Ollama | OpenRouter | DeepSeek | Anthropic | OpenAI | Custom endpoint |
|------|--------|------------|----------|-----------|--------|-----------------|
| local | auto-detect | — | — | — | — | configurable |
| flash | — | ✓ | ✓ | ✓ | ✓ | ✓ |
| pro | — | ✓ | ✓ | ✓ | ✓ | ✓ |

The local tier auto-detects whichever Ollama model is available. Flash and pro tiers work with any provider Hermes supports — the `provider` and `model` fields in the tier config determine the target.

## Usage

Once loaded, the skill evaluates each query before responding:

```
User: "What's the capital of France?"
→ local tier (Ollama)
Agent: The capital of France is Paris.

User: "Design a fault-tolerant payment system"
→ pro tier (Claude Sonnet)
Agent: [detailed architecture answer]
```

### Python API

```python
from smart_router.router import ModelRouter
from smart_router.ollama import OllamaManager

router = ModelRouter(ollama_manager=OllamaManager())
decision = router.resolve("Explain how DNS works")
# → {"tier": "flash", "model": {...}, "ollama_ready": null}
```

### CLI

```bash
python -m smart_router route "What is the capital of France?"
python -m smart_router chat          # interactive mode with stats
python -m smart_router tiers         # list configured tiers
python -m smart_router ollama status # check Ollama
```

## License

MIT
