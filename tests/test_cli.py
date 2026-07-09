"""
Tests for the CLI entry point.

Covers: route, ollama, chat, tiers commands, --json/--verbose flags,
unknown commands, and help output.
"""

import json
import sys
from io import StringIO
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from smart_router.__main__ import main, cmd_route, cmd_ollama, cmd_tiers, cmd_chat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_main(argv: list[str]) -> tuple[int, str, str]:
    """Run main() with given argv and capture stdout/stderr + exit code."""
    stdout = StringIO()
    stderr = StringIO()
    exit_code = 0

    with patch.object(sys, "argv", ["smart_router"] + argv):
        with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
            try:
                main()
            except SystemExit as e:
                exit_code = e.code if isinstance(e.code, int) else 0

    return exit_code, stdout.getvalue(), stderr.getvalue()


# ---------------------------------------------------------------------------
# Help / Usage
# ---------------------------------------------------------------------------

class TestHelp:
    def test_no_args_shows_usage(self):
        exit_code, stdout, stderr = _run_main([])
        assert "Usage:" in stdout
        assert "route" in stdout
        assert exit_code == 0

    def test_help_flag(self):
        for flag in ("--help", "-h", "help"):
            exit_code, stdout, stderr = _run_main([flag])
            assert "Usage:" in stdout
            assert exit_code == 0

    def test_unknown_command(self):
        exit_code, stdout, stderr = _run_main(["bogus"])
        assert "Unknown command" in stderr
        assert exit_code == 1


# ---------------------------------------------------------------------------
# Route command
# ---------------------------------------------------------------------------

class TestRouteCommand:
    def test_route_basic(self):
        with patch("smart_router.router.ModelRouter") as MockRouter:
            mock = MagicMock()
            mock.route_info.return_value = {
                "query": "hello world",
                "tier": "local",
                "model": {"provider": "custom", "model": "qwen3:14b", "base_url": "http://localhost:11434/v1"},
                "description": "Simple queries",
            }
            MockRouter.return_value = mock

            exit_code, stdout, stderr = _run_main(["route", "hello world"])
            assert "local" in stdout
            assert "custom/qwen3:14b" in stdout
            assert exit_code == 0

    def test_route_json(self):
        with patch("smart_router.router.ModelRouter") as MockRouter:
            mock = MagicMock()
            mock.route_info.return_value = {
                "query": "design a system",
                "tier": "pro",
                "model": {"provider": "deepseek", "model": "deepseek-v4-pro"},
                "description": "Complex tasks",
            }
            MockRouter.return_value = mock

            exit_code, stdout, stderr = _run_main(["route", "--json", "design a system"])
            data = json.loads(stdout)
            assert data["tier"] == "pro"
            assert data["model"]["provider"] == "deepseek"
            assert exit_code == 0

    def test_route_default_query(self):
        """When no query is provided, use a default."""
        with patch("smart_router.router.ModelRouter") as MockRouter:
            mock = MagicMock()
            mock.route_info.return_value = {
                "query": "Hello, how are you?",
                "tier": "local",
                "model": {"provider": "custom", "model": "qwen3:14b"},
                "description": "",
            }
            MockRouter.return_value = mock

            exit_code, stdout, stderr = _run_main(["route"])
            assert "local" in stdout
            assert exit_code == 0

    def test_route_error_handling(self):
        with patch("smart_router.router.ModelRouter") as MockRouter:
            MockRouter.return_value.route_info.side_effect = RuntimeError("model download failed")

            exit_code, stdout, stderr = _run_main(["route", "test"])
            assert "Error:" in stdout or exit_code == 1


# ---------------------------------------------------------------------------
# Ollama command
# ---------------------------------------------------------------------------

class TestOllamaCommand:
    def test_ollama_status(self):
        with patch("smart_router.ollama.OllamaManager") as MockMgr:
            mock = MagicMock()
            mock.status.return_value = {
                "running": True,
                "binary_exists": True,
                "model": "qwen3:14b",
                "model_loaded": True,
                "model_pulled": True,
                "idle_seconds": 42,
                "idle_timeout": 300,
                "our_pid": None,
                "wsl": True,
            }
            MockMgr.return_value = mock

            exit_code, stdout, stderr = _run_main(["ollama", "status"])
            assert "Running:" in stdout
            assert "qwen3:14b" in stdout
            assert exit_code == 0

    def test_ollama_status_json(self):
        with patch("smart_router.ollama.OllamaManager") as MockMgr:
            mock = MagicMock()
            mock.status.return_value = {"running": False, "model": "", "idle_seconds": -1}
            MockMgr.return_value = mock

            exit_code, stdout, stderr = _run_main(["ollama", "status", "--json"])
            data = json.loads(stdout)
            assert data["running"] is False
            assert exit_code == 0

    def test_ollama_start(self):
        with patch("smart_router.ollama.OllamaManager") as MockMgr:
            mock = MagicMock()
            mock.ensure_running.return_value = True
            MockMgr.return_value = mock

            exit_code, stdout, stderr = _run_main(["ollama", "start"])
            assert "ready" in stdout
            assert exit_code == 0

    def test_ollama_start_failure(self):
        with patch("smart_router.ollama.OllamaManager") as MockMgr:
            mock = MagicMock()
            mock.ensure_running.return_value = False
            MockMgr.return_value = mock

            exit_code, stdout, stderr = _run_main(["ollama", "start"])
            assert "failed" in stdout
            assert exit_code == 0

    def test_ollama_stop(self):
        with patch("smart_router.ollama.OllamaManager") as MockMgr:
            mock = MagicMock()
            mock.ensure_killed.return_value = True
            MockMgr.return_value = mock

            exit_code, stdout, stderr = _run_main(["ollama", "stop"])
            assert "killed" in stdout
            assert exit_code == 0

    def test_ollama_unknown_action(self):
        exit_code, stdout, stderr = _run_main(["ollama", "restart"])
        assert exit_code == 1
        assert "Unknown action" in stderr


# ---------------------------------------------------------------------------
# Tiers command
# ---------------------------------------------------------------------------

class TestTiersCommand:
    def test_tiers_basic(self):
        exit_code, stdout, stderr = _run_main(["tiers"])
        assert "[local]" in stdout
        assert "[flash]" in stdout
        assert "[pro]" in stdout
        assert "DEFAULT" in stdout
        assert exit_code == 0

    def test_tiers_json(self):
        exit_code, stdout, stderr = _run_main(["tiers", "--json"])
        data = json.loads(stdout)
        assert "tiers" in data
        assert "default" in data
        assert "local" in data["tiers"]
        assert exit_code == 0


# ---------------------------------------------------------------------------
# Chat command
# ---------------------------------------------------------------------------

class TestChatCommand:
    def test_chat_quit_immediately(self):
        with patch("builtins.input", side_effect=["quit"]):
            exit_code, stdout, stderr = _run_main(["chat"])
            assert "Interactive Test" in stdout
            assert exit_code == 0

    def test_chat_routes_single_query(self):
        with patch("smart_router.router.ModelRouter") as MockRouter:
            mock = MagicMock()
            mock.route_info.return_value = {
                "query": "hello",
                "tier": "local",
                "model": {"provider": "custom", "model": "qwen3:14b"},
                "description": "Simple queries",
            }
            MockRouter.return_value = mock

            with patch("builtins.input", side_effect=["hello", "quit"]):
                exit_code, stdout, stderr = _run_main(["chat"])
                assert "local" in stdout
                assert "qwen3:14b" in stdout
                assert exit_code == 0

    def test_chat_stats_command(self):
        with patch("builtins.input", side_effect=[":stats", "quit"]):
            exit_code, stdout, stderr = _run_main(["chat"])
            assert "Session Stats" in stdout or "Stats" in stdout
            assert exit_code == 0

    def test_chat_eof_handling(self):
        with patch("builtins.input", side_effect=EOFError):
            exit_code, stdout, stderr = _run_main(["chat"])
            assert exit_code == 0


# ---------------------------------------------------------------------------
# --verbose flag
# ---------------------------------------------------------------------------

class TestVerboseFlag:
    def test_verbose_enables_debug(self):
        exit_code, stdout, stderr = _run_main(["tiers", "--verbose", "--json"])
        assert exit_code == 0
