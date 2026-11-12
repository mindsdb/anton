"""Messages the user adds to a running cloud turn (ENG-4194).

In hosted mode the whole agent loop runs in this pod, so a message the user
sends while Anton works reaches the turn the way an ask_user answer does: a
line on stdin (``cloud_turn/stdin.py`` hands it to ``CloudAddons.deliver``).
The pod is the only place that knows whether the turn took the message, so
every outcome goes back on the wire as an ``addon`` event naming the
message's ``addon_id``; the controller relays it as a ``turn_step`` reply,
which cowork-server both matches by ``addon_id`` and relays to the app as a
``response.addon`` event.

Once the turn is taking messages, the pod says so once with a ``ready`` event
(at turn start, before any model call), so cowork-server knows a line it sends
from then on reaches the turn.

A line past this turn's limits (``MAX_ADDONS_PER_TURN`` messages,
``MAX_ADDON_CHARS`` characters each, ``MAX_ADDON_TURN_BYTES`` in all) is
refused, as is one that arrives before the session is built, or before the
turn starts or after it ends: ``ChatSession.add_user_input`` takes text only
while a turn runs, and a refused message is one cowork-server keeps as the
next task, so nothing is lost. That is simpler than buffering, and a buffer
would have to be returned anyway when the turn never starts.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

EVENT_KIND = "addon"
EVENT_TYPE = "response.addon"

# Per-turn limits, the same as ChatSession.add_user_input's; checked here too
# so a session that predates them can't be flooded.
MAX_ADDONS_PER_TURN = 20
MAX_ADDON_CHARS = 4000
MAX_ADDON_TURN_BYTES = 64 * 1024


class CloudAddons:
    def __init__(self, emit: Callable[[dict], None]) -> None:
        self._emit = emit
        self._session: Any = None
        # addon_id -> {"text", "decided_by", "state"} for every message the
        # turn took, so later events can carry its text.
        self._addons: dict[str, dict] = {}
        # Ids refused this turn: a repeat of one is refused again, never added.
        self._refused: set[str] = set()
        self._bytes = 0
        self._ready_sent = False

    def attach(self, session: Any, *, announce: bool = False) -> None:
        """Start taking messages for *session*'s turn. With *announce* (the
        stdin reader is running), emit ``ready`` once the turn starts taking
        them: the session calls ``on_user_input_ready`` at turn start, before
        any model call."""
        self._session = session
        if callable(getattr(session, "add_user_input", None)):
            session.on_user_input_read = self.mark_read
            if announce:
                session.on_user_input_ready = self.announce_ready

    def announce_ready(self) -> None:
        """Tell cowork-server the turn now takes added messages. At most once
        (contract.py)."""
        if self._ready_sent:
            return
        self._ready_sent = True
        self._emit({"kind": EVENT_KIND, "type": EVENT_TYPE, "state": "ready"})

    def deliver(self, line: dict) -> None:
        """Apply one addon or addon_withdraw line. Runs on the event loop thread."""
        if line["kind"] == "addon":
            self._add(line)
        elif line["kind"] == "addon_withdraw":
            self._withdraw(line)

    def mark_read(self, ids: list[str]) -> None:
        """``on_user_input_read``: the model has now seen these messages."""
        for addon_id in ids:
            addon = self._addons.get(addon_id)
            if addon is not None:
                addon["state"] = "read"
            self._event(addon_id, "read", **self._details(addon_id))

    def return_unread(self) -> None:
        """Report every message the model never saw as returned. Called once
        the turn is over, before the terminal event, on every path."""
        take = getattr(self._session, "take_unread_user_inputs", None)
        if not callable(take):
            return
        try:
            unread = take()
        except Exception:
            logger.warning("cloud turn: could not take the unread added messages", exc_info=True)
            return
        for item in unread:
            addon_id = item["id"]
            addon = self._addons.get(addon_id)
            if addon is not None:
                addon["state"] = "returned"
            decided_by = addon["decided_by"] if addon is not None else None
            self._event(addon_id, "returned", text=item["text"], decided_by=decided_by)

    def _add(self, line: dict) -> None:
        addon_id = line["addon_id"]
        known = self._addons.get(addon_id)
        if known is not None:
            # The same line delivered twice: report where the message is now
            # instead of handing the turn a second copy.
            self._event(addon_id, known["state"], **self._details(addon_id))
            return
        if addon_id in self._refused:
            # Refused stays refused: cowork-server already keeps it as the next task.
            self._event(addon_id, "refused", text=line["text"], decided_by=line["decided_by"])
            return
        added = False
        size = len(line["text"].encode("utf-8", "replace"))
        add = getattr(self._session, "add_user_input", None)
        if (
            len(line["text"]) > MAX_ADDON_CHARS
            or len(self._addons) >= MAX_ADDONS_PER_TURN
            or self._bytes + size > MAX_ADDON_TURN_BYTES
        ):
            add = None
        if callable(add):
            try:
                added = add(addon_id, line["text"]) is True
            except Exception:
                logger.warning("cloud turn: add_user_input failed", exc_info=True)
        state = "added" if added else "refused"
        logger.info("addon %s: %s", addon_id, state)
        if added:
            self._bytes += size
            self._addons[addon_id] = {"text": line["text"],
                                      "decided_by": line["decided_by"], "state": state}
        else:
            self._refused.add(addon_id)
        self._event(addon_id, state, text=line["text"], decided_by=line["decided_by"])

    def _withdraw(self, line: dict) -> None:
        addon_id = line["addon_id"]
        withdrawn = False
        withdraw = getattr(self._session, "withdraw_user_input", None)
        if callable(withdraw):
            try:
                withdrawn = withdraw(addon_id) is True
            except Exception:
                logger.warning("cloud turn: withdraw_user_input failed", exc_info=True)
        known = self._addons.get(addon_id)
        if withdrawn:
            state = "withdrawn"
        elif known is not None and known["state"] == "returned":
            # Too late to take back, but not read either: it is the next task.
            state = "returned"
        else:
            state = "already_read"
        logger.info("addon %s withdraw: %s", addon_id, state)
        if withdrawn and known is not None:
            known["state"] = state
        self._event(addon_id, state, request_id=line["request_id"])

    def _details(self, addon_id: str) -> dict:
        addon = self._addons.get(addon_id)
        if addon is None:
            return {}
        return {"text": addon["text"], "decided_by": addon["decided_by"]}

    def _event(self, addon_id: str, state: str, **fields: Any) -> None:
        self._emit({"kind": EVENT_KIND, "type": EVENT_TYPE,
                    "addon_id": addon_id, "state": state, **fields})
