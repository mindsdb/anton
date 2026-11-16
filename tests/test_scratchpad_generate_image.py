"""The scratchpad's `generate_image()` helper (ENG-4024), in a real pad.

A real pad process calls a stub images API on localhost, so these cover what
only the subprocess can show: the helper is injected next to `get_llm()`,
picks its model from the env the host sets, saves the picture and returns its
path, sends an edit as multipart, and is held to the turn's image allowance the
host sends with each cell, booking its calls in the host's turn budget.
"""

from __future__ import annotations

import base64
import json
import struct
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from anton.core.backends.local import LocalScratchpadRuntime
from anton.core.llm.images import TurnImageBudget, arm_turn_images, disarm_turn_images

pytestmark = pytest.mark.slow

_PNG = (
    b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 64, 32)
    + b"\x08\x06\x00\x00\x00" + b"\x00" * 68
)


class _Gateway(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.requests: list[tuple[str, str, bytes]] = []
        self.bearers: list[str | None] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}/v1"


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _Gateway

    def log_message(self, format: str, *args: object) -> None:
        """Silent: the test reads ``server.requests``."""

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.server.requests.append((self.path, self.headers.get("Content-Type", ""), body))
        self.server.bearers.append(self.headers.get("Authorization"))
        answer = {
            "created": 1,
            "data": [{"b64_json": base64.b64encode(_PNG).decode()}],
            "usage": {
                "input_tokens": 16,
                "input_tokens_details": {"image_tokens": 0, "text_tokens": 16},
                "output_tokens": 824,
                "output_tokens_details": {"image_tokens": 824, "text_tokens": 0},
                "total_tokens": 840,
            },
        }
        raw = json.dumps(answer).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@contextmanager
def _serving() -> Iterator[_Gateway]:
    server = _Gateway()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def _pad(gateway: _Gateway, workspace: Path, *, image_model: str | None) -> LocalScratchpadRuntime:
    return LocalScratchpadRuntime(
        name="images",
        coding_provider="openai-compatible",
        coding_model="stub-model",
        coding_api_key="pad-key",
        coding_base_url=gateway.url,
        workspace_path=workspace,
        image_model=image_model,
    )


class _Turn:
    """The session's side of a turn: an image cap the tool and the pad share."""

    def __init__(self, limit: int, spend_limited: bool = False) -> None:
        self.limit = limit
        self.spend_limited = spend_limited
        self.taken = 0
        self.usage: list = []

    def budget(self) -> TurnImageBudget:
        return TurnImageBudget(
            allowance=lambda: max(0, self.limit - self.taken),
            take=self._take,
            record_usage=lambda model, usage: self.usage.append((model, usage)),
            spend_limited=lambda: self.spend_limited,
        )

    def _take(self) -> None:
        self.taken += 1


@contextmanager
def _turn(limit: int = 8, spend_limited: bool = False) -> Iterator[_Turn]:
    turn = _Turn(limit, spend_limited)
    token = arm_turn_images(turn.budget())
    try:
        yield turn
    finally:
        disarm_turn_images(token)


@pytest.fixture
def host_env(monkeypatch):
    """The host's environment as a Cowork server leaves it: no MindsHub key of
    its own for the pad (the pad gets the coding credential the runtime sets)."""
    for name in ("ANTON_MINDS_API_KEY", "ANTON_MINDS_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL",
                 "ANTON_OPENAI_API_KEY", "ANTON_OPENAI_BASE_URL", "ANTON_IMAGE_MODEL",
                 "ANTON_SCRATCHPAD_IMAGE_MODEL", "ANTON_CLOUD_ARTIFACTS_ROOT"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


async def test_the_helper_saves_the_image_and_returns_its_path(tmp_path, host_env):
    (tmp_path / "logo.png").write_bytes(_PNG)
    with _serving() as gateway, _turn() as turn:
        pad = _pad(gateway, tmp_path, image_model="gpt-image-2.6-sunburst")
        await pad.start()
        try:
            cell = await pad.execute(
                "p = generate_image('a red bicycle on a beach', size='1536x1024', quality='low')\n"
                "print(p)\n"
                "q = generate_image('the same, at night', reference_images=['logo.png'], path='slides/night.png')\n"
                "print(q)\n"
            )
        finally:
            await pad.close()
    assert cell.error is None, cell.error
    first, second = (Path(line) for line in cell.stdout.split())
    # No path: an image artifact of its own, in the project's artifacts folder.
    assert first.name == "a-red-bicycle-on-a-beach.png"
    assert first.parent.parent == (tmp_path / ".anton" / "artifacts").resolve()
    metadata = json.loads((first.parent / "metadata.json").read_text())
    assert (metadata["type"], metadata["primary"]) == ("image", first.name)
    assert second == (tmp_path / "slides" / "night.png").resolve()
    # The pad reached the images API with the credential the runtime gave it.
    assert gateway.bearers == ["Bearer pad-key", "Bearer pad-key"]
    assert first.read_bytes() == second.read_bytes() == _PNG
    # The cell's output names the files, never the pixels.
    assert base64.b64encode(_PNG).decode() not in cell.stdout
    # Both calls were counted against the turn and booked in its cost.
    assert turn.taken == 2
    assert [(m, u.input_tokens, u.image_output_tokens) for m, u in turn.usage] == [
        ("gpt-image-2.6-sunburst", 16, 824), ("gpt-image-2.6-sunburst", 16, 824),
    ]

    (gen_path, gen_type, gen_body), (edit_path, edit_type, edit_body) = gateway.requests
    assert gen_path == "/v1/images/generations" and gen_type == "application/json"
    assert json.loads(gen_body) == {
        "model": "gpt-image-2.6-sunburst", "prompt": "a red bicycle on a beach",
        "size": "1536x1024", "quality": "low",
    }
    assert edit_path == "/v1/images/edits" and edit_type.startswith("multipart/form-data")
    assert b'name="image"; filename="logo.png"' in edit_body
    assert b'name="output_format"\r\n\r\npng' in edit_body


async def test_the_cap_is_the_turns_across_cells(tmp_path, host_env):
    # A pad started with no image model of its own takes the host's
    # ANTON_IMAGE_MODEL, under the pad's name, never a stale pad value.
    host_env.setenv("ANTON_SCRATCHPAD_IMAGE_MODEL", "stale-model")
    host_env.setenv("ANTON_IMAGE_MODEL", "gpt-image-2.6-sunburst")
    with _serving() as gateway, _turn(limit=2) as turn:
        pad = _pad(gateway, tmp_path, image_model=None)
        await pad.start()
        try:
            first = await pad.execute("generate_image('one')\n")
            capped = await pad.execute("generate_image('two')\ngenerate_image('three')\n")
        finally:
            await pad.close()
    assert first.error is None, first.error
    assert capped.error is not None and "used up its images" in capped.error
    assert "no request was sent" in capped.error
    assert len(gateway.requests) == 2 and turn.taken == 2
    assert json.loads(gateway.requests[0][2])["model"] == "gpt-image-2.6-sunburst"


async def test_a_spent_turn_says_its_spend_limit_was_reached(tmp_path, host_env):
    with _serving() as gateway, _turn(limit=0, spend_limited=True):
        pad = _pad(gateway, tmp_path, image_model="m")
        await pad.start()
        try:
            cell = await pad.execute("generate_image('a fox')\n")
        finally:
            await pad.close()
    assert cell.error is not None and "spend limit" in cell.error
    assert "ANTON_MAX_IMAGES_PER_TURN" not in cell.error
    assert gateway.requests == []


async def test_the_helper_is_withheld_when_the_turn_does_not_offer_it(tmp_path, host_env):
    with _serving() as gateway:
        pad = _pad(gateway, tmp_path, image_model="m")
        await pad.start()
        try:
            with _turn():
                kept = await pad.execute("g = generate_image\n")
            # No budget armed: the host offers no images this turn.
            missing = await pad.execute("print('generate_image' in globals())\n")
            stale = await pad.execute("g('a fox')\n")
        finally:
            await pad.close()
    assert kept.error is None, kept.error
    assert missing.stdout.strip() == "False"
    assert stale.error is not None and "not available" in stale.error
    assert gateway.requests == []


async def test_on_a_cloud_pod_artifact_paths_go_to_the_mounted_artifacts_folder(tmp_path, host_env):
    mount = tmp_path / "project-artifacts"
    (mount / "deck").mkdir(parents=True)
    (mount / "deck" / "logo.png").write_bytes(_PNG)
    host_env.setenv("ANTON_CLOUD_ARTIFACTS_ROOT", str(mount))
    workspace = tmp_path / "conversation"
    workspace.mkdir()
    with _serving() as gateway, _turn():
        pad = _pad(gateway, workspace, image_model=None)
        await pad.start()
        try:
            cell = await pad.execute(
                "print(generate_image('a fox'))\n"
                "print(generate_image('a fox, framed', reference_images=['.anton/artifacts/deck/logo.png'],"
                " path='.anton/artifacts/deck/hero.png'))\n"
            )
        finally:
            await pad.close()
    assert cell.error is None, cell.error
    default, hero = (Path(line) for line in cell.stdout.split())
    assert default.parent.parent == mount.resolve() and default.read_bytes() == _PNG
    assert hero == (mount / "deck" / "hero.png").resolve()
    assert not (workspace / ".anton" / "artifacts").exists()
    # Unset everywhere, endpoint not MindsHub: the built-in default.
    assert json.loads(gateway.requests[0][2])["model"] == "gpt-image-2.6-flare"


async def test_threads_racing_in_one_cell_cannot_pass_the_cap_together(tmp_path, host_env):
    """The slot is taken under a lock, so six threads against an allowance of
    two send two requests."""
    with _serving() as gateway, _turn(limit=2):
        pad = _pad(gateway, tmp_path, image_model="m")
        await pad.start()
        try:
            cell = await pad.execute(
                "from concurrent.futures import ThreadPoolExecutor\n"
                "def one(i):\n"
                "    try:\n"
                "        return generate_image(f'image {i}', path=f'out/{i}.png')\n"
                "    except RuntimeError as exc:\n"
                "        return 'refused' if 'no request was sent' in str(exc) else repr(exc)\n"
                "with ThreadPoolExecutor(6) as pool:\n"
                "    results = list(pool.map(one, range(6)))\n"
                "print(sum(r == 'refused' for r in results), sum(r.endswith('.png') for r in results))\n"
            )
        finally:
            await pad.close()
    assert cell.error is None, cell.error
    assert cell.stdout.split() == ["4", "2"]
    assert len(gateway.requests) == 2


async def test_a_bad_argument_uses_no_slot(tmp_path, host_env):
    (tmp_path / "notes.txt").write_text("not an image")
    with _serving() as gateway, _turn(limit=1) as turn:
        pad = _pad(gateway, tmp_path, image_model="m")
        await pad.start()
        try:
            cell = await pad.execute(
                "try:\n"
                "    generate_image('x', reference_images=['notes.txt'])\n"
                "except ValueError as exc:\n"
                "    print('refused:', 'not a PNG' in str(exc))\n"
                "print(generate_image('y', path='y.png'))\n"
            )
        finally:
            await pad.close()
    assert cell.error is None, cell.error
    assert cell.stdout.splitlines()[0] == "refused: True"
    assert len(gateway.requests) == 1 and turn.taken == 1


async def test_a_failed_save_leaves_no_empty_image_artifact(tmp_path, host_env):
    with _serving() as gateway, _turn():
        pad = _pad(gateway, tmp_path, image_model="m")
        await pad.start()
        try:
            cell = await pad.execute(
                "from anton.core.llm import images as _m\n"
                "def _boom(*a, **k):\n"
                "    raise OSError('disk full')\n"
                "_m.write_new_image_file = _boom\n"
                "generate_image('a fox')\n"
            )
        finally:
            await pad.close()
    assert cell.error is not None and "disk full" in cell.error
    artifacts = tmp_path / ".anton" / "artifacts"
    assert not artifacts.exists() or not [p for p in artifacts.iterdir() if p.is_dir()]


async def test_the_default_artifact_names_the_cells_conversation_first(tmp_path, host_env):
    """The provenance stamp is how the desktop host (track_edits_since) and
    cowork-server's cloud path claim a pad-made artifact as the turn's."""
    from anton.core.artifacts import ArtifactStore
    from anton.core.llm.tracing import TraceContext, reset_trace_context, set_trace_context

    conversation = "3f2b8c1e-5d6a-4e7f-9a0b-1c2d3e4f5a6b"
    with _serving() as gateway, _turn():
        pad = _pad(gateway, tmp_path, image_model="m")
        await pad.start()
        token = set_trace_context(TraceContext(session_id=conversation, turn_id=3))
        try:
            cell = await pad.execute("print(generate_image('a fox'))\n")
        finally:
            reset_trace_context(token)
            await pad.close()
    assert cell.error is None, cell.error
    slug = Path(cell.stdout.strip()).parent.name
    artifact = ArtifactStore(tmp_path / ".anton" / "artifacts").read_metadata(slug)
    first = artifact.provenance[0]
    assert first.conversation == conversation and first.turns[0].index == 3
