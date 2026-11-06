"""The scratchpad's `generate_image()` helper (ENG-4024), in a real pad.

A real pad process calls a stub images API on localhost, so these cover what
only the subprocess can show: the helper is injected next to `get_llm()`,
picks its model from the env the host sets, saves the picture and returns its
path, sends an edit as multipart, and refuses past its per-cell cap without
sending anything.
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
    with _serving() as gateway:
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

    (gen_path, gen_type, gen_body), (edit_path, edit_type, edit_body) = gateway.requests
    assert gen_path == "/v1/images/generations" and gen_type == "application/json"
    assert json.loads(gen_body) == {
        "model": "gpt-image-2.6-sunburst", "prompt": "a red bicycle on a beach",
        "size": "1536x1024", "quality": "low",
    }
    assert edit_path == "/v1/images/edits" and edit_type.startswith("multipart/form-data")
    assert b'name="image"; filename="logo.png"' in edit_body
    assert b'name="output_format"\r\n\r\npng' in edit_body


async def test_the_helper_refuses_past_its_cap_until_the_next_cell(tmp_path, host_env):
    # The host reads the cap from its settings when it starts the pad.
    host_env.setenv("ANTON_SCRATCHPAD_MAX_IMAGES", "1")
    # A pad started with no image model of its own takes the host's
    # ANTON_IMAGE_MODEL, under the pad's name, never a stale pad value.
    host_env.setenv("ANTON_SCRATCHPAD_IMAGE_MODEL", "stale-model")
    host_env.setenv("ANTON_IMAGE_MODEL", "gpt-image-2.6-sunburst")
    with _serving() as gateway:
        pad = _pad(gateway, tmp_path, image_model=None)
        await pad.start()
        try:
            capped = await pad.execute(
                "generate_image('one')\n"
                "generate_image('two')\n"
            )
            next_cell = await pad.execute("print(generate_image('three'))\n")
        finally:
            await pad.close()
    assert capped.error is not None and "ANTON_SCRATCHPAD_MAX_IMAGES=1" in capped.error
    assert "No request was sent" in capped.error
    assert next_cell.error is None, next_cell.error
    assert len(gateway.requests) == 2
    assert json.loads(gateway.requests[0][2])["model"] == "gpt-image-2.6-sunburst"


async def test_on_a_cloud_pod_the_default_goes_to_the_mounted_artifacts_folder(tmp_path, host_env):
    mount = tmp_path / "project-artifacts"
    mount.mkdir()
    host_env.setenv("ANTON_CLOUD_ARTIFACTS_ROOT", str(mount))
    workspace = tmp_path / "conversation"
    workspace.mkdir()
    with _serving() as gateway:
        pad = _pad(gateway, workspace, image_model=None)
        await pad.start()
        try:
            cell = await pad.execute("print(generate_image('a fox'))\n")
        finally:
            await pad.close()
    assert cell.error is None, cell.error
    saved = Path(cell.stdout.strip())
    assert saved.parent.parent == mount.resolve() and saved.read_bytes() == _PNG
    # Unset everywhere, endpoint not MindsHub: the built-in default.
    assert json.loads(gateway.requests[0][2])["model"] == "gpt-image-2.6-flare"


async def test_threads_racing_in_one_cell_cannot_pass_the_cap_together(tmp_path, host_env):
    """The slot is taken before the helper's blocking work, so six threads
    against a cap of two send two requests."""
    host_env.setenv("ANTON_SCRATCHPAD_MAX_IMAGES", "2")
    with _serving() as gateway:
        pad = _pad(gateway, tmp_path, image_model="m")
        await pad.start()
        try:
            cell = await pad.execute(
                "from concurrent.futures import ThreadPoolExecutor\n"
                "def one(i):\n"
                "    try:\n"
                "        return generate_image(f'image {i}', path=f'out/{i}.png')\n"
                "    except RuntimeError as exc:\n"
                "        return 'refused' if 'No request was sent' in str(exc) else repr(exc)\n"
                "with ThreadPoolExecutor(6) as pool:\n"
                "    results = list(pool.map(one, range(6)))\n"
                "print(sum(r == 'refused' for r in results), sum(r.endswith('.png') for r in results))\n"
            )
        finally:
            await pad.close()
    assert cell.error is None, cell.error
    assert cell.stdout.split() == ["4", "2"]
    assert len(gateway.requests) == 2


async def test_a_call_that_fails_before_sending_gives_its_slot_back(tmp_path, host_env):
    host_env.setenv("ANTON_SCRATCHPAD_MAX_IMAGES", "1")
    (tmp_path / "notes.txt").write_text("not an image")
    with _serving() as gateway:
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
    assert len(gateway.requests) == 1


async def test_the_default_artifact_names_the_cells_conversation_first(tmp_path, host_env):
    """The provenance stamp is how the desktop host (track_edits_since) and
    cowork-server's cloud path claim a pad-made artifact as the turn's."""
    from anton.core.artifacts import ArtifactStore
    from anton.core.llm.tracing import TraceContext, reset_trace_context, set_trace_context

    conversation = "3f2b8c1e-5d6a-4e7f-9a0b-1c2d3e4f5a6b"
    with _serving() as gateway:
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
