# Hermes Smart Router

Intelligent model tier routing for Hermes Agent — automatically picks the right model for every query.

## The Problem

You have access to multiple model tiers — a fast local model, an affordable cloud model, and a powerful frontier model. But every query goes to the same place, wasting money on simple tasks and under-serving complex ones.

## The Solution

A Hermes skill that routes queries to the optimal tier using semantic classification that runs entirely on your machine:

```
"Translate hello to German"           → local (Ollama, free)
"Explain how DNS works"               → flash (affordable)
"Design a multi-region database"      → pro (best quality)
```

Routing uses [semantic-router](https://github.com/aurelio-labs/semantic-router) with local Ollama embeddings — zero API calls, zero cost, zero network.

## Tiers

| Tier | Purpose | Provider examples |
|------|---------|-------------------|
| **local** | Free, offline, private — simple lookups, translations, formatting | Ollama (llama3.2, qwen3, mistral), any local model |
| **flash** | Fast, affordable — everyday coding, explanations, general Q&A | DeepSeek Flash, Gemini Flash, GPT-4o-mini, Claude Haiku |
| **pro** | Best quality — complex architecture, debugging, multi-step reasoning | DeepSeek Pro, Gemini Pro, GPT-4o, Claude Sonnet |

One API token, different models for different complexity. You don't need multiple providers — even a single provider like OpenRouter or DeepSeek gives you both flash and pro tiers.

## Features

- 3-tier routing: local / flash / pro, auto-selected per query
- 100% local classification via Ollama embeddings — no cloud dependencies
- Ollama lifecycle: auto-start when needed, auto-kill when idle
- No GPU needed — runs on CPU
- Your queries never leave your machine for routing decisions

## Installation

**Prerequisites:** Hermes Agent v0.17+, Python 3.10+, Ollama with at least one model pulled.

```bash
git clone https://github.com/raydatalab/hermes-smart-router.git
cd hermes-smart-router
pip install -r requirements.txt
hermes skills install SKILL.md
```

On first use, Smart Router detects your Ollama models from `ollama list` and adapts. If no cloud providers are configured, it stays on local tier — no errors, no missing-config warnings.

## Configuration

Add a `smart_router` section to `~/.hermes/config.yaml`. Here's a typical setup using OpenRouter (one API key, both flash and pro):

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

Set up providers with `hermes model` — no need to edit `.env` files manually.

## Provider Support

| Tier | Ollama | OpenRouter | DeepSeek | Anthropic | OpenAI | Custom endpoint |
|------|--------|------------|----------|-----------|--------|-----------------|
| local | auto-detect | — | — | — | — | configurable |
| flash | — | ✓ | ✓ | ✓ | ✓ | ✓ |
| pro | — | ✓ | ✓ | ✓ | ✓ | ✓ |

Local tier auto-detects whatever Ollama model you have pulled. Flash and pro work with any provider Hermes supports — just configure the `provider` and `model` in your tier config.

## Usage

Once loaded, the skill evaluates each query before responding:

```
You: "What's the capital of France?"
→ local tier (Ollama)
Hermes: The capital of France is Paris.

You: "Design a fault-tolerant payment system"
→ pro tier (Claude Sonnet)
Hermes: [detailed architecture answer]
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
