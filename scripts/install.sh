#!/bin/bash
# install.sh — one-command Smart Router setup
# Usage: curl -fsSL <raw-url>/scripts/install.sh | bash
#   or: ./scripts/install.sh [--no-ollama-pull]
set -e

# Colors
BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

echo -e "${BOLD}Hermes Smart Router — Installer${RESET}"
echo ""

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# --- Find the right Python ---
# Hermes ships its own venv.  We must install into that venv so the skill
# can import smart_router at runtime.  Fall back to python3 on PATH if
# Hermes is not installed (standalone / dev use).
HERMES_PYTHON="${HERMES_PYTHON:-$HOME/.hermes/hermes-agent/venv/bin/python3}"

if [ -x "$HERMES_PYTHON" ]; then
    PYTHON="$HERMES_PYTHON"
    echo -e "${GREEN}✓${RESET} Hermes venv detected: ${PYTHON/$HOME/~}"
else
    PYTHON="python3"
    echo -e "${YELLOW}⚠ Hermes venv not found — using system python3${RESET}"
    echo "  The skill will work from the CLI but Hermes won't see it."
    echo "  Install Hermes first: https://github.com/hermes/hermes-agent"
fi

# --- Prerequisites check ---
if ! command -v "$PYTHON" &>/dev/null; then
    echo -e "${RED}✗ $PYTHON not found. Install Python 3.10+ first.${RESET}"
    exit 1
fi
echo -e "${GREEN}✓${RESET} python: $($PYTHON --version)"

# --- Install package + dependencies ---
echo ""
echo "Installing smart-router into $($PYTHON -c 'import sys; print(sys.prefix)')..."
"$PYTHON" -m pip install --quiet --upgrade pip
"$PYTHON" -m pip install --quiet "$PROJECT_DIR"
echo -e "${GREEN}✓${RESET} smart-router + dependencies installed"

# --- Ollama check ---
echo ""
if command -v ollama &>/dev/null; then
    echo -e "${GREEN}✓${RESET} ollama found"

    # Pull embedding model if not already present
    NO_PULL="${1:-}"
    if [ "$NO_PULL" != "--no-ollama-pull" ]; then
        if ollama list 2>/dev/null | grep -q "nomic-embed-text"; then
            echo -e "${GREEN}✓${RESET} nomic-embed-text already pulled"
        else
            echo "Pulling embedding model: nomic-embed-text..."
            ollama pull nomic-embed-text
            echo -e "${GREEN}✓${RESET} nomic-embed-text pulled"
        fi
    fi
else
    echo -e "${YELLOW}⚠ ollama not found — local tier unavailable${RESET}"
    echo "  Install: curl -fsSL https://ollama.com/install.sh | sh"
    echo "  Then:    ollama pull nomic-embed-text"
fi

# --- Done ---
echo ""
echo -e "${GREEN}${BOLD}Smart Router installed!${RESET}"
echo ""
echo "Quick test:"
echo "  $PYTHON -m smart_router route 'What is the capital of France?'"
echo "  $PYTHON -m smart_router chat"
echo ""
echo "Install as Hermes skill:"
echo "  hermes skills install $PROJECT_DIR/SKILL.md"
echo ""

# --- Post-install: agent memory instruction ---
echo -e "${BOLD}━━━ For Best Results ━━${RESET}"
echo ""
echo "Hermes agents evaluate skills per-message. The router works best when"
echo "the agent is explicitly told to use it. After installing the skill,"
echo "send this message to your agent to create a lasting instruction:"
echo ""
echo -e "  ${YELLOW}From now on, before every response that needs more than a"
echo -e "  one-sentence answer, run: router = get_router();"
echo -e "  r = router.resolve(user_message, current_tier=ACTIVE_TIER);"
echo -e "  if r['recommendation']: paste it verbatim. Skip only for"
echo -e "  greetings, translations, and single-fact lookups.${RESET}"
echo ""
echo "This one-time instruction significantly increases routing frequency"
echo "and cost savings. See README.md for more details."
echo ""
