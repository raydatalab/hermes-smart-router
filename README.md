# Hermes Smart Router

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/version-0.2.2-green.svg" alt="Version">
</p>

```
"Translate hello to German"       → local   $0/M tok
"Explain how DNS works"           → flash   $0.15/M tok  (GPT-4o-mini)
"Design a distributed database"   → pro     $3/M tok  (Claude Sonnet)
```

Smart Router knows which model to use — every query, automatically.
Simple questions don't pay pro prices. Hard ones get the power they need.

---

## How It Works

Smart Router classifies every query locally (Ollama + semantic-router) and tells
the Hermes agent which tier to use. No API calls for routing. No setup after install.

```
You: "What's the capital of France?"
Agent: [Smart Router: local]
Agent: Paris.

You: "Design a rate limiter for a distributed system."
Agent: [Smart Router: pro]
Agent: (production-ready answer using Claude Sonnet)
```

Same prompt. Three possible models. You pay for the model you need — not the one
your config happens to be on.

---

## Why Smart Router

**Save money.** Every query routed from pro → flash saves ~92%. Flash → local saves 100%.

**Get quality.** Complex reasoning stays on pro where it belongs. No more "local model
hallucinated a database architecture."

**100% local routing.** Classification runs on Ollama embeddings on your machine.
Zero API calls for routing decisions. Zero API keys needed for classification.

**No penalty.** When the router isn't triggered, you never notice. When it is,
it saves you tokens and improves output quality. There's no downside.

---

## Install

```bash
hermes skills install raydatalab/hermes-smart-router
```

Requires: Hermes Agent v0.17+, Ollama installed, `semantic-router[ollama]`.

First run pulls `nomic-embed-text` via Ollama (~274MB).

## Slash Commands

| Command | Description |
|---------|-------------|
| `/route <query>` | Show tier selection for a query (dry run) |
| `/route-stats` | Session routing stats |
| `/ollama start / stop / status` | Ollama lifecycle |

## Related

- [TokenSave](https://github.com/raydatalab/tokensave) — token waste analyzer
- [hermes-cost-optimization](https://clawhub.ai/raydatalab/skills/hermes-cost-optimization) — headroom compression + light model fallback
- [API pricing](https://openai.com/api/pricing/)
