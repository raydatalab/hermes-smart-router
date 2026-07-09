"""
Tests for Ollama lifecycle management.

All subprocess calls are mocked — no actual ollama binary needed.
"""

import socket
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open, PropertyMock

import pytest

from smart_router.ollama import OllamaManager

# Re-import the module for patching module-level variables and helpers
from smart_router import ollama as ollama_module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_run():
    """Patch _run_cmd to return a successful CompletedProcess by default."""
    with patch("smart_router.ollama._run_cmd") as mock:
        mock.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        yield mock


@pytest.fixture
def mock_popen():
    """Patch subprocess.Popen."""
    with patch("subprocess.Popen") as mock:
        proc = MagicMock()
        proc.pid = 12345
        mock.return_value = proc
        yield mock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cproc(returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    """Create a CompletedProcess with the given values."""
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


# ---------------------------------------------------------------------------
# _is_wsl
# ---------------------------------------------------------------------------

class TestIsWsl:
    def test_wsl_detected(self):
        with patch("builtins.open", mock_open(read_data="Linux version ... microsoft ... WSL2")):
            assert ollama_module._is_wsl() is True

    def test_wsl_not_detected(self):
        with patch("builtins.open", mock_open(read_data="Linux version 6.1.0-100-generic")):
            assert ollama_module._is_wsl() is False

    def test_proc_version_missing(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            assert ollama_module._is_wsl() is False


# ---------------------------------------------------------------------------
# _run_cmd
# ---------------------------------------------------------------------------

class TestRunCmd:
    def test_successful_command(self):
        with patch("subprocess.run") as mock:
            mock.return_value = _make_cproc(returncode=0, stdout="output")
            result = ollama_module._run_cmd(["echo", "hello"])
            assert result.returncode == 0
            assert result.stdout == "output"

    def test_timeout(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["sleep"], timeout=10)):
            result = ollama_module._run_cmd(["sleep", "100"], timeout=1)
            assert result.returncode == -1
            assert "timeout" in result.stderr

    def test_command_not_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = ollama_module._run_cmd(["nonexistent"])
            assert result.returncode == -1
            assert "not found" in result.stderr


# ---------------------------------------------------------------------------
# OllamaManager.__init__
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_model_auto_detect(self, mock_run):
        mock_run.return_value = _make_cproc(
            returncode=0,
            stdout="NAME          ID              SIZE      MODIFIED\nqwen3:14b     abc123          14 GB     2 days ago\n",
        )
        mgr = OllamaManager()
        assert mgr.model == "qwen3:14b"

    def test_explicit_model(self):
        mgr = OllamaManager(model="llama3.1:8b")
        assert mgr.model == "llama3.1:8b"

    def test_custom_idle_timeout(self, mock_run):
        mock_run.return_value = _make_cproc(
            returncode=0,
            stdout="NAME          ID              SIZE      MODIFIED\nqwen3:14b     abc123          14 GB     2 days ago\n",
        )
        mgr = OllamaManager(idle_timeout=600)
        assert mgr.idle_timeout == 600

    def test_no_models_available(self, mock_run):
        mock_run.return_value = _make_cproc(returncode=0, stdout="NAME  ID  SIZE  MODIFIED\n")
        mgr = OllamaManager()
        assert mgr.model == ""

    def test_ollama_list_fails(self, mock_run):
        mock_run.return_value = _make_cproc(returncode=1, stderr="connection refused")
        mgr = OllamaManager()
        assert mgr.model == ""


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------

class TestListModels:
    def test_parse_valid_output(self, mock_run):
        mock_run.return_value = _make_cproc(
            returncode=0,
            stdout="NAME          ID              SIZE      MODIFIED\nqwen3:14b     abc123          14 GB     2 days ago\nllama3:8b     def456          8 GB      1 week ago\n",
        )
        mgr = OllamaManager(model="qwen3:14b")
        models = mgr.list_models()
        assert len(models) == 2
        assert models[0]["name"] == "qwen3:14b"
        assert models[0]["id"] == "abc123"
        assert models[0]["size"] == "14"
        assert models[1]["name"] == "llama3:8b"

    def test_empty_output(self, mock_run):
        mock_run.return_value = _make_cproc(returncode=0, stdout="")
        mgr = OllamaManager(model="qwen3:14b")
        assert mgr.list_models() == []

    def test_command_fails(self, mock_run):
        mock_run.return_value = _make_cproc(returncode=1)
        mgr = OllamaManager(model="qwen3:14b")
        assert mgr.list_models() == []


# ---------------------------------------------------------------------------
# is_running
# ---------------------------------------------------------------------------

class TestIsRunning:
    def test_ollama_ps_succeeds(self, mock_run):
        """ollama ps returns 0 → running."""
        mock_run.return_value = _make_cproc(returncode=0, stdout="NAME      ID\nqwen3:14b abc123\n")
        mgr = OllamaManager(model="qwen3:14b")
        assert mgr.is_running is True

    def test_systemd_active(self, mock_run):
        """ollama ps fails but systemd says active."""
        responses = {
            ("ollama", "ps"): _make_cproc(returncode=1),
            ("systemctl", "--user", "is-active", "ollama"): _make_cproc(returncode=0, stdout="active"),
        }

        def _side_effect(cmd, **kwargs):
            key = tuple(cmd)
            return responses.get(key, _make_cproc(returncode=1))

        mock_run.side_effect = _side_effect
        mgr = OllamaManager(model="qwen3:14b")
        assert mgr.is_running is True

    def test_pgrep_finds_process(self, mock_run):
        """ollama ps and systemd fail, but pgrep finds it."""
        responses = {
            ("ollama", "ps"): _make_cproc(returncode=1),
            ("systemctl", "--user", "is-active", "ollama"): _make_cproc(returncode=1),
            ("pgrep", "-x", "ollama"): _make_cproc(returncode=0),
        }

        def _side_effect(cmd, **kwargs):
            key = tuple(cmd)
            return responses.get(key, _make_cproc(returncode=1))

        mock_run.side_effect = _side_effect
        mgr = OllamaManager(model="qwen3:14b")
        assert mgr.is_running is True

    def test_all_checks_fail(self, mock_run):
        """All three checks fail → not running."""
        mock_run.return_value = _make_cproc(returncode=1)
        mgr = OllamaManager(model="qwen3:14b")
        assert mgr.is_running is False


# ---------------------------------------------------------------------------
# is_model_loaded / has_model
# ---------------------------------------------------------------------------

class TestModelState:
    def test_is_model_loaded_true(self, mock_run):
        mock_run.return_value = _make_cproc(returncode=0, stdout="NAME      ID\nqwen3:14b abc123\n")
        mgr = OllamaManager(model="qwen3:14b")
        assert mgr.is_model_loaded is True

    def test_is_model_loaded_false(self, mock_run):
        mock_run.return_value = _make_cproc(returncode=0, stdout="NAME      ID\nllama3:8b def456\n")
        mgr = OllamaManager(model="qwen3:14b")
        assert mgr.is_model_loaded is False

    def test_is_model_loaded_not_running(self, mock_run):
        mock_run.return_value = _make_cproc(returncode=1)
        mgr = OllamaManager(model="qwen3:14b")
        assert mgr.is_model_loaded is False

    def test_has_model_true(self, mock_run):
        mock_run.return_value = _make_cproc(returncode=0, stdout="qwen3:14b abc123 14 GB")
        mgr = OllamaManager(model="qwen3:14b")
        assert mgr.has_model is True

    def test_has_model_false(self, mock_run):
        mock_run.return_value = _make_cproc(returncode=0, stdout="llama3:8b def456 8 GB")
        mgr = OllamaManager(model="qwen3:14b")
        assert mgr.has_model is False

    def test_has_model_not_running(self, mock_run):
        mock_run.return_value = _make_cproc(returncode=1)
        mgr = OllamaManager(model="qwen3:14b")
        assert mgr.has_model is False


# ---------------------------------------------------------------------------
# idle tracking
# ---------------------------------------------------------------------------

class TestIdleTracking:
    def test_idle_seconds_no_marker(self, tmp_path):
        with patch.object(ollama_module, "IDLE_MARKER", tmp_path / "marker"):
            mgr = OllamaManager(model="qwen3:14b")
            assert mgr.idle_seconds == -1

    def test_idle_seconds_with_marker(self, tmp_path):
        marker = tmp_path / "marker"
        marker.write_text("0")
        # Set mtime to 100 seconds ago
        past = time.time() - 100
        marker.touch()
        import os
        os.utime(str(marker), (past, past))

        with patch.object(ollama_module, "IDLE_MARKER", marker):
            mgr = OllamaManager(model="qwen3:14b")
            idle = mgr.idle_seconds
            assert 99 <= idle <= 101  # allow 1s skew

    def test_mark_used_creates_file(self, tmp_path):
        marker = tmp_path / "marker"
        with patch.object(ollama_module, "IDLE_MARKER", marker):
            mgr = OllamaManager(model="qwen3:14b")
            mgr.mark_used()
            assert marker.exists()
            assert float(marker.read_text()) > 0

    def test_mark_used_updates_existing(self, tmp_path):
        marker = tmp_path / "marker"
        marker.write_text("1.0")
        with patch.object(ollama_module, "IDLE_MARKER", marker):
            mgr = OllamaManager(model="qwen3:14b")
            mgr.mark_used()
            assert float(marker.read_text()) > 1.0


# ---------------------------------------------------------------------------
# ensure_running
# ---------------------------------------------------------------------------

class TestEnsureRunning:
    def test_already_running_model_ready(self, mock_run):
        """Ollama running, model pulled and loaded."""
        mock_run.return_value = _make_cproc(returncode=0, stdout="qwen3:14b abc123")
        mgr = OllamaManager(model="qwen3:14b")
        result = mgr.ensure_running()
        assert result is True

    def test_start_fresh(self, mock_run, mock_popen):
        """Not running → start serve → port ready → model ready."""
        # is_running: ollama ps fails, systemd fails, pgrep fails
        mock_run.return_value = _make_cproc(returncode=1)
        mgr = OllamaManager(model="qwen3:14b")

        with patch.object(mgr, "_check_port_ready", return_value=True):
            with patch.object(OllamaManager, "has_model", PropertyMock(return_value=True)):
                result = mgr.ensure_running(wait_seconds=1)
                assert result is True
                mock_popen.assert_called_once_with(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                assert mgr._our_pid == 12345

    def test_ollama_binary_not_found(self, mock_run):
        mock_run.return_value = _make_cproc(returncode=1)
        mgr = OllamaManager(model="qwen3:14b")

        with patch("subprocess.Popen", side_effect=FileNotFoundError):
            result = mgr.ensure_running(wait_seconds=1)
            assert result is False

    def test_timeout_waiting_for_port(self, mock_run, mock_popen):
        """Port never becomes ready."""
        mock_run.return_value = _make_cproc(returncode=1)
        mgr = OllamaManager(model="qwen3:14b")

        with patch.object(mgr, "_check_port_ready", return_value=False):
            result = mgr.ensure_running(wait_seconds=1)
            assert result is False

    def test_pulls_model_when_missing(self, mock_run, mock_popen):
        """Model not pulled → pull it."""
        mgr = OllamaManager(model="qwen3:14b")

        # has_model returns False first time (trigger pull), True after
        call_count = [0]

        def has_model_side_effect():
            call_count[0] += 1
            return call_count[0] > 1  # False first time, True after

        with patch.object(OllamaManager, "has_model", PropertyMock(side_effect=has_model_side_effect)):
            with patch.object(mgr, "_check_port_ready", return_value=True):
                with patch.object(OllamaManager, "is_running", PropertyMock(return_value=True)):
                    mock_run.return_value = _make_cproc(returncode=0)  # pull succeeds
                    result = mgr.ensure_running()
                    assert result is True
                    # Verify pull was called
                    pull_calls = [c for c in mock_run.call_args_list if c[0][0][:2] == ["ollama", "pull"]]
                    assert len(pull_calls) == 1


# ---------------------------------------------------------------------------
# _check_port_ready
# ---------------------------------------------------------------------------

class TestCheckPortReady:
    def test_port_open(self):
        mgr = OllamaManager(model="qwen3:14b")
        with patch("socket.create_connection") as mock_conn:
            mgr._check_port_ready()
            mock_conn.assert_called_once_with(("localhost", 11434), timeout=2)

    def test_port_closed(self):
        mgr = OllamaManager(model="qwen3:14b")
        with patch("socket.create_connection", side_effect=ConnectionRefusedError):
            assert mgr._check_port_ready() is False

    def test_port_error(self):
        mgr = OllamaManager(model="qwen3:14b")
        with patch("socket.create_connection", side_effect=OSError):
            assert mgr._check_port_ready() is False


# ---------------------------------------------------------------------------
# ensure_killed
# ---------------------------------------------------------------------------

class TestEnsureKilled:
    def test_already_dead(self, mock_run):
        mock_run.return_value = _make_cproc(returncode=1)
        mgr = OllamaManager(model="qwen3:14b")
        assert mgr.ensure_killed() is True

    def test_systemd_managed_refuses(self, mock_run):
        """Ollama is systemd-managed — should not kill."""
        responses = {
            ("ollama", "ps"): _make_cproc(returncode=0),  # is_running → True
            ("systemctl", "--user", "is-active", "ollama"): _make_cproc(returncode=0, stdout="active"),
        }

        def _side_effect(cmd, **kwargs):
            key = tuple(cmd)
            return responses.get(key, _make_cproc(returncode=1))

        mock_run.side_effect = _side_effect
        mgr = OllamaManager(model="qwen3:14b")
        result = mgr.ensure_killed()
        assert result is False  # Refused to kill

    def test_pkill_success(self, mock_run):
        """Not systemd-managed, pkill succeeds."""
        responses = {
            ("ollama", "ps"): _make_cproc(returncode=0),  # is_running → True
            ("systemctl", "--user", "is-active", "ollama"): _make_cproc(returncode=1),  # not systemd
            ("pgrep", "-x", "ollama"): _make_cproc(returncode=0),
        }
        # After the systemd check in ensure_killed itself:
        # ensure_killed calls _run_cmd(["systemctl", ...]) again directly

        def _side_effect(cmd, **kwargs):
            key = tuple(cmd)
            if key == ("pkill", "-SIGTERM", "ollama"):
                return _make_cproc(returncode=0)  # kill succeeds
            return responses.get(key, _make_cproc(returncode=1))

        mock_run.side_effect = _side_effect
        mgr = OllamaManager(model="qwen3:14b")
        result = mgr.ensure_killed()
        assert result is True

    def test_force_kill_uses_sigkill(self, mock_run):
        """force=True → SIGKILL."""
        responses = {
            ("ollama", "ps"): _make_cproc(returncode=0),  # is_running → True
            ("systemctl", "--user", "is-active", "ollama"): _make_cproc(returncode=1),
            ("pgrep", "-x", "ollama"): _make_cproc(returncode=0),
            ("pkill", "-SIGKILL", "ollama"): _make_cproc(returncode=0),
        }

        def _side_effect(cmd, **kwargs):
            key = tuple(cmd)
            return responses.get(key, _make_cproc(returncode=1))

        mock_run.side_effect = _side_effect
        mgr = OllamaManager(model="qwen3:14b")
        result = mgr.ensure_killed(force=True)
        assert result is True


# ---------------------------------------------------------------------------
# check_idle_and_kill
# ---------------------------------------------------------------------------

class TestCheckIdleAndKill:
    def test_never_used(self, tmp_path):
        marker = tmp_path / "marker"
        with patch.object(ollama_module, "IDLE_MARKER", marker):
            mgr = OllamaManager(model="qwen3:14b")
            assert mgr.check_idle_and_kill() is False

    def test_idle_not_exceeded(self, tmp_path):
        marker = tmp_path / "marker"
        marker.write_text(str(time.time()))  # just now
        with patch.object(ollama_module, "IDLE_MARKER", marker):
            mgr = OllamaManager(model="qwen3:14b", idle_timeout=300)
            assert mgr.check_idle_and_kill() is False

    def test_idle_exceeded_kills(self, tmp_path):
        marker = tmp_path / "marker"
        marker.write_text("1.0")  # very old
        past = time.time() - 500
        import os
        os.utime(str(marker), (past, past))

        with patch.object(ollama_module, "IDLE_MARKER", marker):
            mgr = OllamaManager(model="qwen3:14b", idle_timeout=300)
            with patch.object(mgr, "ensure_killed", return_value=True) as mock_kill:
                result = mgr.check_idle_and_kill()
                assert result is True
                mock_kill.assert_called_once()


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

class TestStatus:
    def test_status_has_all_keys(self, mock_run):
        mock_run.return_value = _make_cproc(returncode=1)  # not running
        mgr = OllamaManager(model="qwen3:14b")
        status = mgr.status()
        assert set(status.keys()) == {
            "running", "binary_exists", "model", "model_loaded",
            "model_pulled", "idle_seconds", "idle_timeout", "our_pid", "wsl",
        }
        assert status["running"] is False
        assert status["model"] == "qwen3:14b"
        assert status["idle_timeout"] == 300

    def test_status_when_running(self, mock_run):
        mock_run.return_value = _make_cproc(returncode=0, stdout="qwen3:14b abc123")
        mgr = OllamaManager(model="qwen3:14b")
        status = mgr.status()
        assert status["running"] is True
        assert status["model_loaded"] is True
        assert status["model_pulled"] is True


# ---------------------------------------------------------------------------
# model setter
# ---------------------------------------------------------------------------

class TestModelSetter:
    def test_change_model(self, mock_run):
        mock_run.return_value = _make_cproc(returncode=1)
        mgr = OllamaManager(model="qwen3:14b")
        assert mgr.model == "qwen3:14b"
        mgr.model = "llama3.1:8b"
        assert mgr.model == "llama3.1:8b"
