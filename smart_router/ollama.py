"""
Ollama lifecycle management for local tier routing.

Handles:
- Detecting if Ollama is running
- Starting Ollama on demand
- Pulling models when missing
- Auto-killing Ollama after idle timeout
- WSL2 detection and compatibility
"""

import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

OLLAMA_PORT = 11434
OLLAMA_BIN = "ollama"
IDLE_MARKER = Path("/tmp/smart-router-ollama-last-use")


def _is_wsl() -> bool:
    """Detect if running inside WSL2."""
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except (FileNotFoundError, OSError):
        return False


def _run_cmd(cmd: list[str], timeout: int = 10) -> subprocess.CompletedProcess:
    """Run a shell command safely and return the result."""
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"Command timed out: {' '.join(cmd)}")
        return subprocess.CompletedProcess(cmd, returncode=-1, stdout="", stderr="timeout")
    except FileNotFoundError:
        logger.error(f"Command not found: {cmd[0]}")
        return subprocess.CompletedProcess(cmd, returncode=-1, stdout="", stderr="not found")


class OllamaManager:
    """Manage Ollama process lifecycle.

    Supports:
    - Auto-detection of available models via `ollama list`
    - WSL2 detection and compatibility (systemd + pgrep fallback)
    - Starting/killing ollama serve
    - Idle timeout auto-kill
    """

    def __init__(self, model: str = "", idle_timeout: int = 300):
        self.idle_timeout = idle_timeout
        self._our_pid: Optional[int] = None  # PID if WE started it

        # Resolve model: if empty, auto-detect from available models
        self._model = model or self._detect_best_model()

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str):
        self._model = value

    # Embedding models — never pick these as the chat model
    _EMBEDDING_MODELS = {"nomic-embed-text", "mxbai-embed-large", "all-minilm"}

    def _detect_best_model(self) -> str:
        """Auto-detect the best available Ollama chat model.

        Skips embedding models (nomic-embed-text etc.). Returns the first
        chat-capable model name, or empty string if none found.
        """
        result = _run_cmd(["ollama", "list"])
        if result.returncode != 0 or not result.stdout.strip():
            logger.warning("No Ollama models found")
            return ""

        models = self.list_models()
        if not models:
            return ""

        for model in models:
            if model["name"] not in self._EMBEDDING_MODELS:
                logger.info(f"Auto-detected Ollama model: {model['name']}")
                return model["name"]

        logger.warning("Only embedding models found — no chat model available")
        return ""

    def list_models(self) -> list[dict]:
        """Return list of available Ollama models with details.

        Returns:
            [{name, id, size, modified}, ...] or empty list.
        """
        result = _run_cmd(["ollama", "list"])
        if result.returncode != 0 or not result.stdout.strip():
            return []

        models = []
        lines = result.stdout.strip().split("\n")
        # Skip header line
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 4:
                models.append({
                    "name": parts[0],
                    "id": parts[1],
                    "size": parts[2],
                    "modified": " ".join(parts[3:]),
                })
        return models

    @property
    def is_running(self) -> bool:
        """Check if ollama serve process is alive.

        Tries three methods in order:
        1. ``ollama ps`` — most reliable, checks the API directly
        2. ``systemctl --user is-active ollama`` — WSL2 systemd / native Linux
        3. ``pgrep -x ollama`` — fallback for manually-started ollama
        """
        # 1. Try ollama ps first (talks to the API — definitive)
        result = _run_cmd(["ollama", "ps"])
        if result.returncode == 0:
            return True

        # 2. Try systemd (WSL2 with systemd, or native Linux)
        result = _run_cmd(["systemctl", "--user", "is-active", "ollama"])
        if result.returncode == 0 and result.stdout.strip() == "active":
            return True

        # 3. Fall back to pgrep
        result = _run_cmd(["pgrep", "-x", "ollama"])
        return result.returncode == 0

    @property
    def is_model_loaded(self) -> bool:
        """Check if the target model is loaded in Ollama."""
        if not self.is_running or not self._model:
            return False
        result = _run_cmd(["ollama", "ps"])
        return self._model in result.stdout

    @property
    def has_model(self) -> bool:
        """Check if the model has been pulled."""
        if not self.is_running or not self._model:
            return False
        result = _run_cmd(["ollama", "list"])
        return self._model in result.stdout

    @property
    def idle_seconds(self) -> int:
        """Seconds since last local-tier use. Returns -1 if never tracked."""
        if not IDLE_MARKER.exists():
            return -1
        last_use = IDLE_MARKER.stat().st_mtime
        return int(time.time() - last_use)

    def mark_used(self):
        """Record that local tier was just used (resets idle timer)."""
        IDLE_MARKER.parent.mkdir(parents=True, exist_ok=True)
        IDLE_MARKER.write_text(str(time.time()))
        logger.debug("Local tier used — idle timer reset")

    def ensure_running(self, wait_seconds: int = 30) -> bool:
        """
        Start ollama serve if not running. Pull model if missing.

        Args:
            wait_seconds: Max seconds to wait for ollama to be ready.

        Returns:
            True if ollama is ready, False otherwise.
        """
        if self.is_running:
            logger.info("Ollama already running")
            return self._ensure_model_ready()

        logger.info("Starting ollama serve...")

        # Start ollama in background
        try:
            proc = subprocess.Popen(
                [OLLAMA_BIN, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._our_pid = proc.pid
        except FileNotFoundError:
            logger.error("ollama binary not found. Install: curl -fsSL https://ollama.com/install.sh | sh")
            return False

        # Wait for port to be ready
        for i in range(wait_seconds):
            time.sleep(1)
            if self._check_port_ready():
                logger.info(f"Ollama ready after {i + 1}s")
                return self._ensure_model_ready()

        logger.error(f"Ollama not ready after {wait_seconds}s")
        return False

    def _check_port_ready(self) -> bool:
        """Check if Ollama API port is accepting connections (pure Python, no curl)."""
        import socket

        try:
            sock = socket.create_connection(("localhost", OLLAMA_PORT), timeout=2)
            sock.close()
            return True
        except (ConnectionRefusedError, OSError):
            return False

    def _ensure_model_ready(self) -> bool:
        """Pull model if missing. Returns True if model is available."""
        if self.has_model:
            if not self.is_model_loaded:
                logger.info(f"Model {self.model} available but not loaded — will load on first use")
            return True

        logger.info(f"Pulling model {self.model} (first time, may take minutes)...")
        result = _run_cmd(["ollama", "pull", self.model], timeout=600)
        if result.returncode == 0:
            logger.info(f"Model {self.model} pulled successfully")
            return True
        else:
            logger.error(f"Failed to pull model: {result.stderr}")
            return False

    def ensure_killed(self, force: bool = False) -> bool:
        """
        Kill ollama process. Respects systemd services (doesn't kill them).

        Args:
            force: If True, use SIGKILL instead of SIGTERM.

        Returns:
            True if killed or already dead, False if we shouldn't kill (systemd managed).
        """
        if not self.is_running:
            return True

        # Systemd managed — don't kill (applies on all platforms, including WSL2)
        result = _run_cmd(["systemctl", "--user", "is-active", "ollama"])
        if result.returncode == 0 and result.stdout.strip() == "active":
            logger.info("Ollama is systemd-managed — not killing")
            return False

        sig = "-SIGKILL" if force else "-SIGTERM"
        logger.info(f"Sending {sig} to ollama...")
        result = _run_cmd(["pkill", sig, "ollama"])
        if result.returncode == 0:
            logger.info("Ollama killed")
            return True
        else:
            logger.warning("Failed to kill ollama (may already be dead)")
            return False

    def check_idle_and_kill(self) -> bool:
        """
        Auto-kill if idle past timeout. Called before switching away from local tier.
        Returns True if killed, False if still running or shouldn't kill.
        """
        idle = self.idle_seconds
        if idle < 0:
            return False
        if idle >= self.idle_timeout:
            logger.info(f"Ollama idle for {idle}s (timeout: {self.idle_timeout}s) — killing")
            return self.ensure_killed()
        return False

    def status(self) -> dict:
        """Return full status dict for display."""
        running = self.is_running
        return {
            "running": running,
            "binary_exists": bool(subprocess.run(["which", "ollama"], capture_output=True).returncode == 0),
            "model": self.model,
            "model_loaded": self.is_model_loaded if running else False,
            "model_pulled": self.has_model if running else False,
            "idle_seconds": self.idle_seconds,
            "idle_timeout": self.idle_timeout,
            "our_pid": self._our_pid,
            "wsl": _is_wsl(),
        }
