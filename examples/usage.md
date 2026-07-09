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
#   Model:    custom/qwen3:14b
#   Summary:  Simple lookups, translations, greetings, basic Q&A

# Route a moderate question → flash (cheap API)
python -m smart_router route "Explain how DNS works"
# Output:
#   Tier:     flash
#   Model:    deepseek/deepseek-v4-flash

# Route a complex question → pro (best quality)
python -m smart_router route "Design a microservice architecture for e-commerce"
# Output:
#   Tier:     pro
#   Model:    deepseek/deepseek-v4-pro

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
from smart_router.router import ModelRouter
from smart_router.ollama import OllamaManager

# Simple: classify only (no lifecycle management)
router = ModelRouter()
tier = router.classify("What is the capital of France?")
# → "local"

model = router.get_model("Write a Python script to parse JSON")
# → {"provider": "deepseek", "model": "deepseek-v4-flash"}

# Full lifecycle: classify + manage Ollama
ollama = OllamaManager()
router = ModelRouter(ollama_manager=ollama)

decision = router.resolve("Explain how DNS works")
# → {"tier": "flash", "model": {...}, "ollama_ready": null}

decision = router.resolve("Hello, how are you?")
# → {"tier": "local", "model": {...}, "ollama_ready": true}
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
