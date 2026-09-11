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
import logging as _logging
from pathlib import Path

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anton.core.memory.episodes import EpisodicMemory


def is_user_turn(message: dict) -> bool:
    """True for a genuine user turn.

    A user-role message whose content is only ``tool_result`` blocks is the
    reply to a tool call, not a new turn. Counting it would inflate turn
    totals once tool activity is replayed back into history.
    """
    if message.get("role") != "user":
        return False
    content = message.get("content")
    if not isinstance(content, list):
        return True  # string / multimodal input is always a real turn
    return not any(
        isinstance(block, dict) and block.get("type") == "tool_result"
        for block in content
    )


def repair_replayed_tool_ids(history: list[dict]) -> tuple[list[dict], int]:
    """Re-key replayed tool blocks whose id never survived generation.

    Returns ``(history, repaired_block_count)``; the input list is never
    mutated (messages that need a change are copied first, so a host that
    holds the same dicts is not edited underneath it).

    ENG-2420 wrote `tool_use` blocks with ``id: ""`` into stored
    conversations. Replaying one makes the provider reject the entire request
    — ``400 Invalid 'input[N].call_id': empty string`` — so every later turn
    of that conversation fails and it never recovers. Guarding the readers
    stops NEW ones; this repairs the ones already in users' stored histories,
    which is why it runs at the single ingress every surface shares
    (`ChatSessionConfig.initial_history`) rather than in one host.

    **Count-preserving by contract, not by taste.** `ChatSession.last_compaction`
    reports ``covered_through`` as a positional count into `initial_history`,
    and the host maps that count onto its own message list (cowork-server's
    `_persist_history_compaction` does ``tail_start + covered - 1``). Dropping
    a message here would shift a saved compaction cutoff onto the wrong
    message and silently drop or duplicate real history on the next turn —
    a worse bug than the one being fixed, and one that would be blamed on
    compaction. So blocks are rewritten in place and no message is ever added
    or removed.

    A tool-reply user message also keeps at least one `tool_result` block, so
    `is_user_turn` still reads it as a reply rather than a new turn. Rewriting
    it to plain text would inflate the session's turn count and — in exactly
    the shape being repaired, where the poisoned turn died and the next
    message is a real user turn — leave two consecutive user messages, which
    `_seed_history` and `_validate_history_for_provider` both treat as a
    provider hazard.

    Pairing uses the adjacent (assistant, user) model because that is the only
    shape anton writes: a `tool_result` block is only ever built from the
    `tc.id` of a call in the immediately preceding assistant message. Anything
    that does not fit it is rewritten to text rather than guessed at.
    """
    from anton.core.llm.provider import UNNAMED_REPLAYED_TOOL, usable_call_id

    seen: dict[str, int] = {}

    def _mint(block: dict | None = None) -> str:
        """A DETERMINISTIC id, derived from what the block already says.

        Not `uuid4`, and that is the whole point. This repair is re-applied on
        every load and is never written back: cowork-server leaves
        `history_store` unset, and it persists only `session.history[seed_len:]`
        — this turn's messages — so the repaired seed never reaches the
        `messages` table. A random id would therefore differ on every turn,
        changing the replayed prefix each time and missing the prompt cache on
        the WHOLE conversation, forever, for exactly the conversations being
        healed. Byte-stability of the history prefix is an explicit design goal
        of the host that builds it (see `_stamp_message`, "cache-safe").

        Keyed on content rather than position so a compaction summary being
        prepended — which shifts every index — does not change the ids of the
        tail it kept. The occurrence counter disambiguates a conversation that
        made the identical call twice.
        """
        import hashlib
        import json as _json

        if block is None:
            basis = "orphan"
        else:
            basis = _json.dumps(
                [block.get("name") or "", block.get("input")],
                sort_keys=True, default=str,
            )
        seen[basis] = seen.get(basis, 0) + 1
        digest = hashlib.sha256(f"{basis}|{seen[basis]}".encode()).hexdigest()
        return f"call_repaired_{digest[:24]}"

    def _blocks(msg) -> list | None:
        content = msg.get("content") if isinstance(msg, dict) else None
        return content if isinstance(content, list) else None

    def _bad(block, field) -> bool:
        return isinstance(block, dict) and not usable_call_id(block.get(field))

    out = [m for m in history]
    repaired = 0

    for i, msg in enumerate(out):
        blocks = _blocks(msg)
        if blocks is None or msg.get("role") != "assistant":
            continue
        broken = [b for b in blocks
                  if isinstance(b, dict) and b.get("type") == "tool_use" and _bad(b, "id")]
        if not broken:
            continue

        nxt = out[i + 1] if i + 1 < len(out) else None
        reply_blocks = _blocks(nxt) if nxt is not None and nxt.get("role") == "user" else None
        pairable = reply_blocks is not None and any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in reply_blocks
        )

        new_blocks = [dict(b) if isinstance(b, dict) else b for b in blocks]
        if not pairable:
            # Nothing to pair with, so a minted id would be an orphan
            # `tool_use` — which is its own 400 on Anthropic. Keep the message
            # and the block COUNT, drop only the call.
            for pos, b in enumerate(new_blocks):
                if isinstance(b, dict) and b.get("type") == "tool_use" and _bad(b, "id"):
                    new_blocks[pos] = {
                        "type": "text",
                        "text": "[a tool call was recorded here without an id and has been removed]",
                    }
                    repaired += 1
            out[i] = {**msg, "content": new_blocks}
            continue

        minted: list[str] = []
        for b in new_blocks:
            if isinstance(b, dict) and b.get("type") == "tool_use" and _bad(b, "id"):
                b["id"] = _mint(b)
                if not isinstance(b.get("name"), str) or not b["name"]:
                    b["name"] = UNNAMED_REPLAYED_TOOL
                minted.append(b["id"])
                repaired += 1
        out[i] = {**msg, "content": new_blocks}

        queue = list(minted)
        new_reply = [dict(b) if isinstance(b, dict) else b for b in reply_blocks]
        for b in new_reply:
            if not queue:
                break
            if isinstance(b, dict) and b.get("type") == "tool_result" and _bad(b, "tool_use_id"):
                b["tool_use_id"] = queue.pop(0)
                repaired += 1
        for orphan in queue:
            # A block, not a message: still count-preserving.
            new_reply.append({
                "type": "tool_result",
                "tool_use_id": orphan,
                "content": "[no result was recorded for this call]",
            })
            repaired += 1
        out[i + 1] = {**nxt, "content": new_reply}

    # A bad `tool_result` the pass above never claimed has no matching call in
    # the message before it. Re-keying it alone would orphan it, so it is
    # rewritten to text — the one case that can change `is_user_turn`, which is
    # why it is logged rather than done quietly.
    for i, msg in enumerate(out):
        blocks = _blocks(msg)
        if blocks is None or msg.get("role") != "user":
            continue
        if not any(isinstance(b, dict) and b.get("type") == "tool_result"
                   and _bad(b, "tool_use_id") for b in blocks):
            continue
        new_blocks = []
        for b in blocks:
            if isinstance(b, dict) and b.get("type") == "tool_result" and _bad(b, "tool_use_id"):
                new_blocks.append({"type": "text",
                                   "text": "[a tool result with no matching call was removed]"})
                repaired += 1
            else:
                new_blocks.append(b)
        out[i] = {**msg, "content": new_blocks}
        _logging.getLogger(__name__).warning(
            "ENG-2420: replayed history had a tool_result with no pairable call "
            "at index %d; rewrote it as text (this message may now count as a "
            "user turn).", i,
        )

    if repaired:
        _logging.getLogger(__name__).warning(
            "ENG-2420: repaired %d unreplayable tool block(s) in replayed "
            "history; the conversation would otherwise have been rejected by "
            "the provider on every turn.", repaired,
        )
    return out, repaired


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

            # Count user turns (tool_result replies are not turns)
            turns = sum(1 for m in data if is_user_turn(m))
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
