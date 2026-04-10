"""Diagnostic information collection for bug reports."""

from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from anton import __version__
from anton.data_vault import DataVault

if TYPE_CHECKING:
    from anton.chat_session import ChatSession
    from anton.config.settings import AntonSettings
    from anton.memory.cortex import Cortex
    from anton.workspace import Workspace


def collect_diagnostics(
    settings: AntonSettings,
    session: ChatSession | None = None,
    workspace: Workspace | None = None,
    cortex: Cortex | None = None,
) -> dict[str, Any]:
    """Collect comprehensive diagnostic information for bug reports.

    Returns a dictionary with:
    - system_info: OS, Python, Anton versions
    - packages: Installed package versions
    - config: Sanitized configuration
    - datasources: Connected datasource names
    - workspace: Current workspace info
    - memory: Memory state if enabled
    - conversation: Current conversation history
    - logs: Recent log entries
    """
    diagnostics = {
        "timestamp": datetime.now(UTC).isoformat(),
        "anton_version": __version__,
    }

    # System information
    diagnostics["system_info"] = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "os_name": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
    }

    # Installed packages
    try:
        import importlib.metadata

        packages = {}
        for pkg in ["anthropic", "openai", "pydantic", "prompt_toolkit", "rich", "typer"]:
            try:
                packages[pkg] = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:
                packages[pkg] = "not installed"
        diagnostics["packages"] = packages
    except Exception:
        diagnostics["packages"] = {}

    # Sanitized configuration
    config_dict = settings.model_dump()
    # Remove sensitive fields
    sensitive_fields = [
        "anthropic_api_key",
        "openai_api_key",
        "minds_api_key",
        "google_api_key",
        "groq_api_key",
        "aws_access_key_id",
        "aws_secret_access_key",
    ]
    for field in sensitive_fields:
        if field in config_dict:
            config_dict[field] = "***REDACTED***" if config_dict[field] else None
    diagnostics["config"] = config_dict

    # Connected datasources (names only)
    try:
        vault = DataVault()
        connections = vault.list_connections()
        diagnostics["datasources"] = [
            {"engine": c["engine"], "name": c["name"]} for c in connections
        ]
    except Exception as e:
        diagnostics["datasources"] = f"Error collecting: {str(e)}"

    # Workspace information
    if workspace:
        diagnostics["workspace"] = {
            "base": str(workspace.base),
            "name": workspace.name,
            "has_git": workspace.git_root is not None,
        }
    else:
        diagnostics["workspace"] = None

    # Memory state
    if cortex:
        try:
            diagnostics["memory"] = {
                "enabled": cortex.enabled,
                "mode": settings.memory_mode,
                "episodic_enabled": cortex._episodic.enabled if cortex._episodic else False,
            }
        except Exception as e:
            diagnostics["memory"] = f"Error collecting: {str(e)}"
    else:
        diagnostics["memory"] = {"enabled": False}

    # Current conversation history
    if session:
        try:
            # Get conversation history without tool outputs for brevity
            history = []
            for msg in session._history:
                if msg.get("role") in ["user", "assistant"]:
                    entry = {"role": msg["role"]}
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        # Truncate very long messages
                        entry["content"] = (
                            content[:1000] + "..." if len(content) > 1000 else content
                        )
                    else:
                        entry["content"] = "[complex content]"
                    history.append(entry)
            diagnostics["conversation"] = {
                "turn_count": session._turn_count,
                "history_length": len(session._history),
                "history_sample": history[-10:],  # Last 10 messages
            }
        except Exception as e:
            diagnostics["conversation"] = f"Error collecting: {str(e)}"
    else:
        diagnostics["conversation"] = None

    # Recent logs
    try:
        log_dir = Path(settings.workspace_path) / ".anton" / "logs"
        if log_dir.exists():
            log_files = sorted(log_dir.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)
            if log_files:
                # Read last 100 lines from most recent log
                recent_log = log_files[0]
                lines = recent_log.read_text().splitlines()
                diagnostics["recent_logs"] = {
                    "log_file": recent_log.name,
                    "last_lines": lines[-100:] if len(lines) > 100 else lines,
                }
            else:
                diagnostics["recent_logs"] = None
        else:
            diagnostics["recent_logs"] = None
    except Exception as e:
        diagnostics["recent_logs"] = f"Error collecting: {str(e)}"

    return diagnostics


def save_diagnostics_file(diagnostics: dict[str, Any], output_dir: Path) -> Path:
    """Save diagnostics to a JSON file in the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    filename = f"bug_report_{timestamp}.json"
    filepath = output_dir / filename

    with open(filepath, "w") as f:
        json.dump(diagnostics, f, indent=2, default=str)

    return filepath
