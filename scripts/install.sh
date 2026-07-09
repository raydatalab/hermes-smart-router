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

# --- Prerequisites check ---

# Python
if ! command -v python3 &>/dev/null; then
    echo -e "${RED}✗ python3 not found. Install Python 3.10+ first.${RESET}"
    exit 1
fi
echo -e "${GREEN}✓${RESET} python3: $(python3 --version)"

# Pip
if ! python3 -m pip --version &>/dev/null 2>&1; then
    echo -e "${RED}✗ pip not found. Install pip first.${RESET}"
    exit 1
fi
echo -e "${GREEN}✓${RESET} pip: $(python3 -m pip --version | cut -d' ' -f2)"

# --- Python venv ---
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$PROJECT_DIR/.venv"

if [ ! -d "$VENV" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv "$VENV"
fi

source "$VENV/bin/activate"
echo -e "${GREEN}✓${RESET} venv: $VENV"

# --- Dependencies ---
echo ""
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r "$PROJECT_DIR/requirements.txt"
echo -e "${GREEN}✓${RESET} dependencies installed"

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
echo "  source $VENV/bin/activate"
echo "  python -m smart_router route 'What is the capital of France?'"
echo "  python -m smart_router chat"
echo ""
echo "Install as Hermes skill:"
echo "  hermes skills install $PROJECT_DIR/SKILL.md"
