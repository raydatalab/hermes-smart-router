# Smart Router — Usage Examples

## CLI Quick Start

```bash
# Activate venv
source .venv/bin/activate

# Route a simple question → local (Ollama, free)
python -m smart_router route "What is the capital of France?"
# Output:
#   Query:    What is the capital of France?
#   Tier:     local
#   Model:    custom/llama3.2:3b
#   Summary:  Simple lookups, translations, greetings, basic Q&A

# Route a moderate question → flash (cheap API)
python -m smart_router route "Explain how DNS works"
# Output:
#   Tier:     flash
#   Model:    openrouter/google/gemini-flash-1.5

# Route a complex question → pro (best quality)
python -m smart_router route "Design a microservice architecture for e-commerce"
# Output:
#   Tier:     pro
#   Model:    anthropic/claude-sonnet-4

# JSON output for scripting
python -m smart_router route --json "What is Python?"
```

## Interactive Mode

```bash
python -m smart_router chat
# Type queries interactively. Commands:
#   :stats   — show routing statistics
#   :tiers   — list configured tiers
#   :reset   — reset statistics
#   quit     — exit
```

## Ollama Management

```bash
# Check Ollama status
python -m smart_router ollama status

# Start Ollama (auto-pulls model if needed)
python -m smart_router ollama start

# Stop Ollama
python -m smart_router ollama stop
```

## View Config

```bash
# List all configured tiers
python -m smart_router tiers
python -m smart_router tiers --json
```

## Python API

```python
from smart_router.router import get_router

# Get the module singleton — init cost paid once, reused across calls
router = get_router()

# Classify only
tier = router.classify("What is the capital of France?")
# → "local"

model = router.get_model("Write a Python script to parse JSON")
# → {"provider": "openrouter", "model": "google/gemini-flash-1.5"}

# Full resolution with tier comparison
decision = router.resolve("Explain how DNS works", current_tier="flash")
# → {"tier": "flash", "model": {...}, "ollama_ready": null, "needs_switch": false}

decision = router.resolve("Design a distributed database", current_tier="flash")
# → {"tier": "pro", "model": {...}, "ollama_ready": null, "needs_switch": true}

# Ollama lifecycle requires an OllamaManager — pass it once when creating the singleton
from smart_router.router import get_router, reset_router
from smart_router.ollama import OllamaManager

reset_router()  # clear cached singleton
ollama = OllamaManager()
router = get_router(ollama_manager=ollama)

decision = router.resolve("Hello, how are you?", current_tier="flash")
# → {"tier": "local", "model": {...}, "ollama_ready": true, "needs_switch": true}
# (ollama_ready=true means Ollama is running and model is loaded)

# Debug: get full routing info
info = router.route_info("Design a distributed database")
# → {"query": "...", "tier": "pro", "model": {...}, "description": "..."}
```

## Customizing Tiers

Override tiers at runtime:

```python
from smart_router.tier import set_tiers, reset_tiers

# Disable local tier (API-only mode)
set_tiers({"local": None})

# Add a new tier
set_tiers({
    "ultra": {
        "models": {"provider": "openai", "model": "gpt-4o-mini"},
        "utterances": ["quick fact check", "simple yes/no question"],
    }
})

# Reset to defaults
reset_tiers()
```

Or configure permanently via config.yaml (under `smart_router:`):

```yaml
smart_router:
  enabled: true
  default_tier: flash
  encoder_model: nomic-embed-text
  tiers:
    local:
      provider: custom
      model: llama3.1:8b
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

## Environment Variables

For flash/pro tiers, set API keys via `hermes model` or your shell environment:

```bash
export DEEPSEEK_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```
