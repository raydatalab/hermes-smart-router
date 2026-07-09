#!/bin/bash
# smart-router.sh — route a query from the command line using Smart Router
# Usage: ./scripts/smart-router.sh "your question here"
set -e

cd "$(dirname "$0")/.."

# Find and activate the Python venv
for VENV in "$HOME/.hermes-router-env" "$PWD/.venv"; do
    if [ -f "$VENV/bin/activate" ]; then
        source "$VENV/bin/activate"
        break
    fi
done

export PYTHONPATH="$PWD"
python -m smart_router route "$@"
