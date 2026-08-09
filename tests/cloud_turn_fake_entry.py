"""Deterministic subprocess entrypoint for cloud-turn E2E / FD tests.

Run as ``python tests/cloud_turn_fake_entry.py`` — it installs a deterministic
seam, then calls the REAL ``anton.cloud_turn.__main__.main()`` so the full
process boundary (FD isolation, stdin parse, runner lifecycle, scratchpad) runs
for real. NOT collected by pytest (no ``test_`` prefix).

Modes (env ``CLOUD_TURN_FAKE_MODE``):
* ``model`` — real ChatSession + real scratchpad, but the LLM is a fake provider
  scripted by ``CLOUD_TURN_FAKE_SCRIPT`` (JSON list of steps). No network.
* ``stray`` — replace the session builder with one whose turn writes stray
  output to stdout/FD 1/logging, then completes (proves FD isolation).
* ``stray_fail`` — like ``stray`` but the turn raises after the stray output.
"""

from __future__ import annotations

import json
import logging
import os
import sys


def _install_fake_model() -> None:
    import anton.core.llm.client as client_mod
    from anton.core.llm.client import LLMClient
    from anton.core.llm.provider import (
        LLMProvider,
        LLMResponse,
        ProviderConnectionInfo,
        ToolCall,
        Usage,
    )

    script = json.loads(os.environ.get("CLOUD_TURN_FAKE_SCRIPT", "[]"))

    class _FakeProvider(LLMProvider):
        name = "fake"

        def __init__(self) -> None:
            self._i = 0

        async def complete(self, *, model, system, messages, tools=None,
                           tool_choice=None, max_tokens=4096, native_web_tools=None):
            step = script[min(self._i, len(script) - 1)] if script else {"text": ""}
            self._i += 1
            tool_calls = []
            if "tool" in step:
                t = step["tool"]
                tool_calls = [ToolCall(id=t.get("id", "t1"), name=t["name"], input=t["input"])]
            return LLMResponse(
                content=step.get("text", ""),
                tool_calls=tool_calls,
                usage=Usage(context_pressure=0.0),
            )

        def export_connection_info(self):
            return ProviderConnectionInfo(provider="fake", api_key="fake")

    def _fake_from_settings(cls, settings):
        prov = _FakeProvider()
        return LLMClient(
            planning_provider=prov, planning_model="fake-model",
            coding_provider=prov, coding_model="fake-model",
        )

    client_mod.LLMClient.from_settings = classmethod(_fake_from_settings)


def _install_stray_session(fail: bool) -> None:
    import anton.cloud_turn.__main__ as entry_mod

    class _StraySession:
        def __init__(self) -> None:
            self.history = []
            self.closed = False

        async def turn_stream(self, user_input, **kwargs):
            # Every channel that must NOT reach the protocol stream:
            print("STRAY via print()")                       # Python stdout
            sys.stdout.write("STRAY via sys.stdout.write\n")  # Python stdout
            os.write(1, b"STRAY via os.write(1)\n")           # direct FD 1 (native-style)
            logging.getLogger("some.library").warning("STRAY via logging")
            if fail:
                raise RuntimeError("boom after stray output")
            if False:  # make this an async generator
                yield

        def close(self):
            self.closed = True

    entry_mod.build_cloud_chat_session = lambda request: _StraySession()


def main() -> int:
    mode = os.environ.get("CLOUD_TURN_FAKE_MODE", "model")
    if mode == "model":
        _install_fake_model()
    elif mode == "stray":
        _install_stray_session(fail=False)
    elif mode == "stray_fail":
        _install_stray_session(fail=True)
    else:
        raise SystemExit(f"unknown CLOUD_TURN_FAKE_MODE={mode!r}")

    from anton.cloud_turn.__main__ import main as real_main

    return real_main([])


if __name__ == "__main__":
    sys.exit(main())
