"""Chat history persistence — save/load full conversation history for resume.

Stores conversation history as JSON files alongside episodic JSONL files
in the `.anton/episodes/` directory.  Fire-and-forget writes (never raises).
"""

from __future__ import annotations

import ast
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anton.core.memory.episodes import EpisodicMemory


class HistoryStore:
    """Persist and retrieve full chat history for session resume."""

    def __init__(self, episodes_dir: Path) -> None:
        self._dir = episodes_dir

    def save(self, session_id: str, history: list[dict]) -> None:
        """Atomically write history to ``{session_id}_history.json``.

        Fire-and-forget: silently ignores errors to avoid disrupting chat.
        """
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            target = self._dir / f"{session_id}_history.json"
            fd, tmp = tempfile.mkstemp(
                dir=str(self._dir), suffix=".tmp", prefix=".hist_"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(history, f, ensure_ascii=False)
                os.replace(tmp, str(target))
            except Exception:
                # Clean up temp file on failure
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        except Exception:
            pass  # Fire-and-forget

    def load(self, session_id: str) -> list[dict] | None:
        """Load history for *session_id*.  Returns ``None`` on missing/corrupt."""
        path = self._dir / f"{session_id}_history.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
            return None
        except Exception:
            return None

    @staticmethod
    def episodes_to_api_history(episodes: list[dict]) -> list[dict]:
        """Convert episodic episode list to Anthropic API message format for HistoryStore.

        Processes episodes sequentially:
          user -> {"role":"user","content":text}
          tool_call -> {"role":"assistant","content":[tool_use block]}  (generates id)
          scratchpad -> skipped (content captured in tool_result)
          tool_result -> {"role":"user","content":[tool_result block]}  (uses id from preceding tool_call)
          assistant -> {"role":"assistant","content":text}
        """
        history: list[dict] = []
        i = 0
        while i < len(episodes):
            ep = episodes[i]
            role = ep.get("role", "")

            if role == "user":
                history.append({"role": "user", "content": ep["content"]})
                i += 1

            elif role == "tool_call":
                tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
                tool_name = ep.get("meta", {}).get("tool", "unknown")
                content_str = ep.get("content", "{}")
                try:
                    tool_input = json.loads(content_str)
                except Exception:
                    try:
                        tool_input = ast.literal_eval(content_str)
                    except Exception:
                        tool_input = {"raw": content_str}

                history.append({
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": tool_id, "name": tool_name, "input": tool_input}],
                })
                i += 1

                # Skip optional scratchpad episode
                if i < len(episodes) and episodes[i].get("role") == "scratchpad":
                    i += 1

                # Consume matching tool_result
                if i < len(episodes) and episodes[i].get("role") == "tool_result":
                    history.append({
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": episodes[i]["content"]}],
                    })
                    i += 1

            elif role == "assistant":
                history.append({"role": "assistant", "content": ep["content"]})
                i += 1

            else:
                i += 1

        return history

    def rebuild_from_episodic(self, episodic: "EpisodicMemory") -> list[dict]:
        """Rebuild and persist API history from current episodic session.

        Reads episodes via get_conversation(), converts to API format,
        saves to HistoryStore, and returns the result.
        """
        from dataclasses import asdict
        episodes = [asdict(ep) for ep in episodic.get_conversation()]
        history = self.episodes_to_api_history(episodes)
        if episodic.session_id:
            self.save(episodic.session_id, history)
        return history

    def list_sessions(self, limit: int = 20) -> list[dict]:
        """List recent sessions with history, newest-first.

        Returns a list of dicts with keys:
        ``session_id``, ``date``, ``turns``, ``preview``.
        """
        if not self._dir.is_dir():
            return []

        files = sorted(self._dir.glob("*_history.json"), reverse=True)
        results: list[dict] = []
        for path in files:
            if len(results) >= limit:
                break
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(data, list) or not data:
                continue

            session_id = path.stem.removesuffix("_history")

            # Count user turns
            turns = sum(1 for m in data if m.get("role") == "user")
            if turns == 0:
                continue

            # Extract date from session_id (format: YYYYMMDD_HHMMSS)
            try:
                dt = datetime.strptime(session_id, "%Y%m%d_%H%M%S").replace(
                    tzinfo=timezone.utc
                )
                date_str = dt.strftime("%Y-%m-%d %H:%M")
            except ValueError:
                date_str = session_id

            # First user message as preview
            preview = ""
            for m in data:
                if m.get("role") == "user":
                    content = m.get("content", "")
                    if isinstance(content, str):
                        preview = content.strip()
                    elif isinstance(content, list):
                        # Multimodal content — find first text block
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                preview = block.get("text", "").strip()
                                break
                    break
            if len(preview) > 60:
                preview = preview[:57] + "..."

            results.append({
                "session_id": session_id,
                "date": date_str,
                "turns": turns,
                "preview": preview,
            })

        return results
