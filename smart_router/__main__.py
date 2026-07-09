"""
CLI entry point for Smart Router.

Usage:
    python -m smart_router route [--json] [--verbose] <query>
    python -m smart_router ollama start|stop|status [--json]
    python -m smart_router chat [--verbose]
    python -m smart_router tiers              # list configured tiers

Examples:
    python -m smart_router route "What is the capital of France?"
    python -m smart_router route --json "Design a microservice architecture"
    python -m smart_router ollama status
    python -m smart_router chat
"""

import json
import sys
import logging
from typing import Optional

logger = logging.getLogger(__name__)

SEPARATOR = "─" * 50


def _setup_logging(verbose: bool) -> None:
    """Configure logging level based on verbosity flag."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _parse_flags(args: list[str]) -> tuple[bool, bool, list[str]]:
    """Extract --json and --verbose from args, returning (json_mode, verbose, remaining)."""
    json_mode = False
    verbose = False
    remaining: list[str] = []
    for arg in args:
        if arg == "--json":
            json_mode = True
        elif arg == "--verbose":
            verbose = True
        else:
            remaining.append(arg)
    return json_mode, verbose, remaining


# ---------------------------------------------------------------------------
# Route command
# ---------------------------------------------------------------------------

def cmd_route(args: list[str]) -> None:
    """Classify a query and print the routing decision."""
    json_mode, verbose, remaining = _parse_flags(args)
    _setup_logging(verbose)

    query = " ".join(remaining) if remaining else "Hello, how are you?"

    from .router import ModelRouter

    router = ModelRouter()

    try:
        info = router.route_info(query)
    except Exception as e:
        if json_mode:
            print(json.dumps({"error": str(e)}, ensure_ascii=False))
        else:
            print(f"Error: {e}")
        sys.exit(1)

    if json_mode:
        print(json.dumps(info, indent=2, ensure_ascii=False))
    else:
        model = info["model"]
        provider_model = f"{model['provider']}/{model['model']}"
        base_url = model.get("base_url", "")
        url_info = f"\n  Base URL:    {base_url}" if base_url else ""

        print(f"Query:    {info['query']}")
        print(f"Tier:     {info['tier']}")
        print(f"Model:    {provider_model}{url_info}")
        print(f"Summary:  {info['description']}")


# ---------------------------------------------------------------------------
# Ollama commands
# ---------------------------------------------------------------------------

def cmd_ollama(args: list[str]) -> None:
    """Manage Ollama lifecycle: start, stop, status."""
    json_mode, verbose, remaining = _parse_flags(args)
    _setup_logging(verbose)

    from .ollama import OllamaManager

    mgr = OllamaManager()

    if not remaining or remaining[0] == "status":
        status = mgr.status()
        if json_mode:
            print(json.dumps(status, indent=2))
        else:
            print("Ollama Status")
            print(SEPARATOR)
            print(f"  Running:       {'✓ yes' if status['running'] else '✗ no'}")
            print(f"  Binary:        {'✓ found' if status['binary_exists'] else '✗ not found'}")
            print(f"  Model:         {status['model'] or '(none)'}")
            print(f"  Model loaded:  {'✓ yes' if status['model_loaded'] else '✗ no'}")
            print(f"  Model pulled:  {'✓ yes' if status['model_pulled'] else '✗ no'}")
            print(f"  Idle:          {status['idle_seconds']}s (timeout: {status['idle_timeout']}s)")
            print(f"  Managed PID:   {status['our_pid'] or 'none'}")
            print(f"  WSL:           {'✓ yes' if status['wsl'] else '✗ no'}")
        return

    action = remaining[0]
    if action == "start":
        ok = mgr.ensure_running()
        if json_mode:
            print(json.dumps({"action": "start", "success": ok}))
        else:
            print(f"Ollama start: {'✓ ready' if ok else '✗ failed'}")
            if not ok:
                print("  Is ollama installed? Try: curl -fsSL https://ollama.com/install.sh | sh")

    elif action == "stop":
        ok = mgr.ensure_killed(force=True)
        if json_mode:
            print(json.dumps({"action": "stop", "success": ok}))
        else:
            print(f"Ollama stop: {'✓ killed' if ok else '✗ not killed (may be systemd-managed)'}")

    else:
        print(f"Unknown action: {action}. Use: start, stop, status", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Tiers command
# ---------------------------------------------------------------------------

def cmd_tiers(args: list[str]) -> None:
    """List configured tiers with model info."""
    json_mode, verbose, remaining = _parse_flags(args)
    _setup_logging(verbose)

    from .tier import get_tiers, DEFAULT_TIER

    tiers = get_tiers()

    if json_mode:
        print(json.dumps({"tiers": tiers, "default": DEFAULT_TIER}, indent=2, ensure_ascii=False))
        return

    print("Smart Router — Configured Tiers")
    print(SEPARATOR)
    for name, config in tiers.items():
        marker = " ★ DEFAULT" if name == DEFAULT_TIER else ""
        m = config["models"]
        provider_model = f"{m['provider']}/{m['model']}"
        print(f"  [{name}]{marker}")
        print(f"    Model:       {provider_model}")
        if "base_url" in m:
            print(f"    Base URL:    {m['base_url']}")
        print(f"    Utterances:  {len(config['utterances'])} examples")
        print(f"    Description: {config.get('description', '')}")
        print()


# ---------------------------------------------------------------------------
# Chat (interactive) command
# ---------------------------------------------------------------------------

def cmd_chat(args: list[str]) -> None:
    """Interactive routing test with session statistics."""
    json_mode, verbose, remaining = _parse_flags(args)
    _setup_logging(verbose)

    from .router import ModelRouter

    router = ModelRouter()

    # Session routing statistics
    stats: dict[str, int] = {"local": 0, "flash": 0, "pro": 0}

    print("╔══════════════════════════════════════════════════╗")
    print("║        Smart Router — Interactive Test           ║")
    print("╠══════════════════════════════════════════════════╣")
    print("║  Type a query to see tier + model routing.       ║")
    print("║  Commands: :stats  :reset  :help  quit          ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    while True:
        try:
            query = input("query ▸ ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            continue

        # Handle special commands
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if query.startswith(":"):
            _handle_chat_command(query, stats, router)
            continue

        # Route the query
        info = router.route_info(query)
        tier = info["tier"]
        stats[tier] += 1
        model = info["model"]
        provider_model = f"{model['provider']}/{model['model']}"

        # Display routing decision
        tier_symbols = {"local": "🖥 ", "flash": "⚡", "pro": "🧠"}
        symbol = tier_symbols.get(tier, "→")

        print(f"  {symbol}  Tier:  {tier}")
        print(f"     Model: {provider_model}")
        if "base_url" in model:
            print(f"     URL:   {model['base_url']}")
        print(f"     {info['description']}")
        print()


def _handle_chat_command(cmd: str, stats: dict[str, int], router) -> None:
    """Process in-chat meta-commands (prefixed with ':')."""
    cmd = cmd.lower()

    if cmd == ":stats":
        total = sum(stats.values())
        print(f"  Session Stats ({total} queries)")
        print(f"  {'─' * 28}")
        if total == 0:
            print("    No queries yet.")
        else:
            for tier, count in stats.items():
                pct = (count / total * 100) if total > 0 else 0
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                print(f"    {tier:8s}  {bar}  {count:3d} ({pct:3.0f}%)")

    elif cmd == ":reset":
        for key in stats:
            stats[key] = 0
        print("  Stats reset.")

    elif cmd == ":help":
        print("  Commands:")
        print("    :stats   Show session routing statistics")
        print("    :reset   Reset session statistics")
        print("    :help    Show this help")
        print("    :tiers   List configured tiers")
        print("    quit     Exit interactive mode")

    elif cmd == ":tiers":
        from .tier import get_tiers, DEFAULT_TIER
        tiers = get_tiers()
        for name, config in tiers.items():
            marker = " (default)" if name == DEFAULT_TIER else ""
            m = config["models"]
            print(f"    [{name}]{marker}: {m['provider']}/{m['model']}")

    else:
        print(f"  Unknown command: {cmd}. Type :help for available commands.")

    print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse subcommand and dispatch."""

    usage_text = (
        "Smart Router CLI — intelligent model tier routing\n"
        "\n"
        "Usage:\n"
        "  python -m smart_router route [--json] [--verbose] <query>\n"
        "  python -m smart_router ollama start|stop|status [--json]\n"
        "  python -m smart_router chat  [--verbose]\n"
        "  python -m smart_router tiers [--json]\n"
        "\n"
        "Examples:\n"
        '  python -m smart_router route "What is the capital of France?"\n'
        '  python -m smart_router route --json "Design a system"\n'
        "  python -m smart_router ollama status\n"
    )

    if len(sys.argv) < 2:
        print(usage_text)
        sys.exit(0)

    command = sys.argv[1]
    args = sys.argv[2:]

    if command == "route":
        cmd_route(args)
    elif command == "ollama":
        cmd_ollama(args)
    elif command == "chat":
        cmd_chat(args)
    elif command == "tiers":
        cmd_tiers(args)
    elif command in ("--help", "-h", "help"):
        print(usage_text)
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Available: route, ollama, chat, tiers", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
