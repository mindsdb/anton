"""
Shared utilities for Anton E2E tests.

Provides:
  run_anton()       run the real CLI as a subprocess
  base_env()        baseline env vars (stub URL, no consent/analytics/memory)
  assert_output()   pattern matching on captured stdout/stderr
  find_history_files()
  E2EConfig         stub vs. live mode configuration
  LiveProvider      no-op stub replacement for live provider runs
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from tests.e2e.stub_server import StubServer

@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    duration: float
    timed_out: bool = False
    cmd: list[str] = field(default_factory=list)
    upstream_asyncgen_noise: bool = False

    def __str__(self) -> str:
        status = "TIMEOUT" if self.timed_out else f"exit {self.returncode}"
        return (
            f"RunResult({status}, {self.duration:.1f}s)\n"
            f"  stdout: {self.stdout[:300]!r}\n"
            f"  stderr: {self.stderr[:300]!r}"
        )


class LiveProvider:
    """Drop-in replacement for StubServer in live mode.

    Queue methods are no-ops; the real provider is contacted directly.
    base_url returns None so base_env() does not override ANTON_OPENAI_BASE_URL.
    """

    def queue_text(self, text: str) -> "LiveProvider":
        return self

    def queue_tool_call(self, name: str, arguments: dict) -> "LiveProvider":
        return self

    def queue_verification_ok(self) -> "LiveProvider":
        return self

    def queue_verification_incomplete(self, reason: str = "") -> "LiveProvider":
        return self

    def queue_verification_stuck(self, reason: str = "") -> "LiveProvider":
        return self

    def queue_summary(self, text: str = "") -> "LiveProvider":
        return self

    @property
    def base_url(self) -> None:  # type: ignore[override]
        return None

    @property
    def requests(self) -> list[dict]:
        return []

    @property
    def request_count(self) -> int:
        return 0

    def system_prompts(self) -> list[str]:
        return []

    def __enter__(self) -> "LiveProvider":
        return self

    def __exit__(self, *_) -> None:
        pass


@dataclass
class E2EConfig:
    """Configuration passed to tests via the cfg fixture.

    Attributes:
        live: When True, use the real LLM provider instead of the stub server.
    """

    live: bool = False

    def make_provider(self) -> StubServer | LiveProvider:
        """Return a fresh provider appropriate for this mode."""
        return LiveProvider() if self.live else StubServer()

    def timeout(self, base: float) -> float:
        """Scale timeouts for live mode (real LLM calls are much slower)."""
        return base * 5 if self.live else base



# httpcore2 raises "generator didn't stop after athrow()" while finalizing a
# response stream at interpreter exit (upstream, pydantic/httpx2). It is not
# anton output, so strip it before assertions rather than weakening them.
_UPSTREAM_ASYNCGEN_NOISE_START = "an error occurred during closing of asynchronous generator"
# The exception line, not the bare phrase: the block's traceback also contains
# `raise RuntimeError("generator didn't stop after athrow()")`, and ending the
# strip there would leave the final line behind.
_UPSTREAM_ASYNCGEN_NOISE_END = "RuntimeError: generator didn't stop after athrow()"


def _strip_upstream_asyncgen_noise(text: str) -> tuple[str, bool]:
    """Return (text without complete upstream blocks, whether one was removed).

    Only a block closed by the end marker is stripped. An unterminated block —
    upstream wording change, or stderr cut off by the TimeoutExpired path — is
    not provably noise, and dropping the rest of stderr there would let a real
    traceback vanish before the assertions see it.
    """
    if _UPSTREAM_ASYNCGEN_NOISE_START not in text:
        return text, False
    kept: list[str] = []
    block: list[str] = []  # candidate block, restored if it never closes
    skipping, found = False, False
    for line in text.splitlines(keepends=True):
        if not skipping and _UPSTREAM_ASYNCGEN_NOISE_START in line:
            skipping, block = True, [line]
            continue
        if skipping:
            block.append(line)
            if _UPSTREAM_ASYNCGEN_NOISE_END in line:
                skipping, block, found = False, [], True
            continue
        kept.append(line)
    kept.extend(block)
    return "".join(kept), found


def wheel_python() -> str | None:
    """Interpreter for an installed-wheel run, or None to use the source tree."""
    return os.environ.get("ANTON_E2E_WHEEL_PYTHON") or None


def run_anton(
    cmd_args: list[str],
    input_lines: list[str],
    *,
    env: dict[str, str],
    cwd: str | Path | None = None,
    timeout: float = 30.0,
) -> RunResult:
    """Run the Anton CLI as a subprocess and capture all output."""
    target = wheel_python()
    cmd = [target or sys.executable, "-m", "anton"] + cmd_args
    # `-m` puts cwd on sys.path; the repo root there would shadow the wheel.
    if target:
        repo = Path(__file__).parents[2].resolve()
        if cwd is None:
            cwd = tempfile.gettempdir()
        elif Path(cwd).resolve() == repo or repo in Path(cwd).resolve().parents:
            raise ValueError(
                f"wheel run would import anton from the checkout via cwd={cwd}"
            )
    stdin_bytes = "\n".join(input_lines).encode() + b"\n"

    t0 = time.monotonic()
    timed_out = False
    try:
        result = subprocess.run(
            cmd,
            input=stdin_bytes,
            capture_output=True,
            timeout=timeout,
            env=env,
            cwd=str(cwd) if cwd else None,
        )
        returncode = result.returncode
        stdout = result.stdout.decode(errors="replace")
        stderr = result.stderr.decode(errors="replace")
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = -1
        stdout = (exc.stdout or b"").decode(errors="replace")
        stderr = (exc.stderr or b"").decode(errors="replace")

    stderr, upstream_noise = _strip_upstream_asyncgen_noise(stderr)

    return RunResult(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        duration=time.monotonic() - t0,
        timed_out=timed_out,
        cmd=cmd,
        upstream_asyncgen_noise=upstream_noise,
    )


def base_env(
    provider: StubServer | LiveProvider,
    *,
    memory_enabled: bool = False,
) -> dict[str, str]:
    """Return an env dict that boots Anton against the given provider.

    Stub mode: fully controlled env pointing at the local stub server.
    Live mode: inherits the real process env, applies test-specific overrides
               (disable analytics/autoupdates, suppress colour output).
    """
    common: dict[str, str] = {
        "ANTON_TERMS_CONSENT": "true",
        "ANTON_ANALYTICS_ENABLED": "false",
        "ANTON_DISABLE_AUTOUPDATES": "true",
        "ANTON_MINDS_ENABLED": "false",
        "ANTON_MEMORY_ENABLED": "true" if memory_enabled else "false",
        "ANTON_MEMORY_MODE": "autopilot" if memory_enabled else "off",
        "ANTON_EPISODIC_MEMORY": "true" if memory_enabled else "false",
        "NO_COLOR": "1",
        "TERM": "dumb",
    }
    # Source runs need the repo importable; a wheel run must not see it.
    if not wheel_python():
        common["PYTHONPATH"] = str(Path(__file__).parents[2])

    if isinstance(provider, LiveProvider):
        env = dict(os.environ)
        env.update(common)
        if wheel_python():
            env.pop("PYTHONPATH", None)
        return env

    return {
        **common,
        "ANTON_PLANNING_PROVIDER": "openai-compatible",
        "ANTON_CODING_PROVIDER": "openai-compatible",
        "ANTON_OPENAI_BASE_URL": provider.base_url,
        "ANTON_OPENAI_API_KEY": "test-key-e2e",
        "ANTON_PLANNING_MODEL": "gpt-test",
        "ANTON_CODING_MODEL": "gpt-test",
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
    }



def assert_output(result: RunResult, *patterns: str) -> None:
    """Assert that all patterns appear somewhere in stdout + stderr."""
    combined = result.stdout + result.stderr
    for p in patterns:
        assert p in combined, f"Pattern not found in output: {p!r}\n{result}"


def assert_not_output(result: RunResult, *patterns: str) -> None:
    combined = result.stdout + result.stderr
    for p in patterns:
        assert p not in combined, f"Unexpected pattern found in output: {p!r}\n{result}"


def assert_exit_ok(result: RunResult) -> None:
    assert not result.timed_out, f"Command timed out\n{result}"
    assert result.returncode == 0, f"Expected exit 0, got {result.returncode}\n{result}"
    # The stripped block is still a regression: it means a provider HTTP pool
    # reached interpreter exit unclosed. This is the only assertion that would
    # catch the original symptom coming back.
    assert not result.upstream_asyncgen_noise, (
        f"asyncgen finalization noise on stderr — a provider pool was not closed\n{result}"
    )


def assert_exit_fail(result: RunResult) -> None:
    assert not result.timed_out, f"Command timed out\n{result}"
    assert result.returncode != 0, f"Expected non-zero exit, got 0\n{result}"


def find_history_files(ws: Path) -> list[Path]:
    """Return all session history JSON files in the workspace."""
    episodes_dir = ws / ".anton" / "episodes"
    if not episodes_dir.exists():
        return []
    return sorted(episodes_dir.glob("*_history.json"))
