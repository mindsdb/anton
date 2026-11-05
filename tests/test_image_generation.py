"""ENG-4024: image generation — the image role, the `generate_image` tool and
the image role in the turn cost.

Every images API call here goes to an in-process stub transport (an
``httpx2.MockTransport`` under a real ``AsyncOpenAI`` client), so the requests
are the SDK's real wire format and nothing reaches the network.
"""

from __future__ import annotations

import base64
import json
import logging
import struct
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
import openai
import pytest

from anton.config.settings import AntonSettings
from anton.core.llm import images
from anton.core.llm.client import LLMClient
from anton.core.llm.openai import OpenAIProvider
from anton.core.llm.provider import ProviderAuthError, Usage
from anton.core.session import ChatSession, ChatSessionConfig
from anton.core.settings import CoreSettings
from anton.core.tools import tool_handlers as th
from anton.core.tools.progress import ToolProgress
from anton.core.tools.registry import ToolOutcome, ToolRegistry
from anton.core.tools.tool_defs import GENERATE_IMAGE_TOOL
from anton.core.turn_cost import EVENT_ROLES, TurnCost
from anton.workspace import Workspace
from tests.conftest import make_mock_llm, run_turn

MINDSHUB = "https://api.mindshub.ai/v1"


def _png(width: int, height: int) -> bytes:
    """A PNG header with these dimensions: enough for `sniff_image`."""
    ihdr = struct.pack(">II", width, height) + b"\x08\x06\x00\x00\x00"
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + ihdr + b"\x00\x00\x00\x00" + b"\x00" * 64


PNG = _png(1536, 1024)
PNG_B64 = base64.b64encode(PNG).decode()

# The usage shape observed live on gpt-image-2.6-flare (CONTRACT.md).
LIVE_USAGE = {
    "input_tokens": 16,
    "input_tokens_details": {"image_tokens": 0, "text_tokens": 16},
    "output_tokens": 824,
    "output_tokens_details": {"image_tokens": 824, "text_tokens": 0},
    "total_tokens": 840,
}


class Gateway:
    """The images API on a mock transport. Records each request."""

    def __init__(self, *, status: int = 200, body: dict | None = None, headers: dict | None = None):
        self.status = status
        self.body = body if body is not None else {
            "created": 1, "data": [{"b64_json": PNG_B64}], "output_format": "png", "usage": LIVE_USAGE,
        }
        self.headers = headers or {}
        self.requests: list[httpx2.Request] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return httpx2.Response(self.status, json=self.body, headers=self.headers)

    def json_body(self, index: int = -1) -> dict:
        return json.loads(self.requests[index].content)


def _provider(gateway: Gateway, *, base_url: str = MINDSHUB,
              flavor: str = OpenAIProvider.FLAVOR_MINDS_PASSTHROUGH) -> OpenAIProvider:
    provider = OpenAIProvider(api_key="test-key", base_url=base_url, flavor=flavor)
    provider._client = openai.AsyncOpenAI(
        api_key="test-key",
        base_url=base_url,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(gateway.handler)),
        max_retries=0,
    )
    return provider


def _client(gateway: Gateway, *, image_model: str | None = "gpt-image-2.6-flare", **kw) -> LLMClient:
    provider = _provider(gateway, **kw)
    return LLMClient(
        planning_provider=provider, planning_model="sonnet",
        coding_provider=provider, coding_model="haiku",
        image_model=image_model,
    )


@pytest.fixture(autouse=True)
def _fresh_catalog_cache():
    images._catalog_defaults.clear()
    yield
    images._catalog_defaults.clear()


# ── Settings ─────────────────────────────────────────────────────────


def test_image_model_setting_reads_its_env_var(monkeypatch):
    monkeypatch.setenv("ANTON_IMAGE_MODEL", "gpt-image-2.6-sunburst")
    assert AntonSettings(_env_file=None).image_model == "gpt-image-2.6-sunburst"
    monkeypatch.delenv("ANTON_IMAGE_MODEL")
    assert AntonSettings(_env_file=None).image_model is None


def test_from_settings_carries_the_image_model():
    settings = AntonSettings(
        _env_file=None, planning_provider="openai", coding_provider="openai",
        openai_api_key="k", image_model="gpt-image-2.6-sunburst",
    )
    assert LLMClient.from_settings(settings).image_model == "gpt-image-2.6-sunburst"


def test_image_caps_default_to_eight_and_read_their_env_vars(monkeypatch):
    assert CoreSettings().max_images_per_turn == 8
    assert CoreSettings().scratchpad_max_images == 8
    monkeypatch.setenv("ANTON_MAX_IMAGES_PER_TURN", "3")
    monkeypatch.setenv("ANTON_SCRATCHPAD_MAX_IMAGES", "2")
    assert CoreSettings().max_images_per_turn == 3
    assert CoreSettings().scratchpad_max_images == 2


# ── Which model ──────────────────────────────────────────────────────

_CATALOG = {
    "data": [
        {"id": "sonnet", "kind": "chat", "default_for": ["planning"]},
        {"id": "gpt-image-2.6-flare", "kind": "image"},
        {"id": "gpt-image-2.6-sunburst", "kind": "image", "default_for": ["image"]},
    ]
}


def _catalog(payload: dict = _CATALOG):
    return patch("anton.minds_client.minds_request", return_value=json.dumps(payload).encode())


def test_a_configured_model_wins_without_reading_the_catalogue():
    with patch("anton.minds_client.minds_request", side_effect=AssertionError("read")):
        assert images.resolve_image_model(
            "my-image", base_url=MINDSHUB, api_key="k", mindshub=True
        ) == "my-image"


def test_unset_on_mindshub_takes_the_catalogue_image_default():
    with _catalog() as request:
        model = images.resolve_image_model(None, base_url=MINDSHUB, api_key="k", mindshub=True)
    assert model == "gpt-image-2.6-sunburst"
    assert request.call_args.args[0] == f"{MINDSHUB}/models"


def test_a_role_default_on_a_chat_row_is_not_an_image_model():
    payload = {"data": [{"id": "sonnet", "kind": "chat", "default_for": ["image"]}]}
    with _catalog(payload):
        model = images.resolve_image_model(None, base_url=MINDSHUB, api_key="k", mindshub=True)
    assert model == images.DEFAULT_IMAGE_MODEL == "gpt-image-2.6-flare"


def test_off_mindshub_the_default_is_the_constant():
    with patch("anton.minds_client.minds_request", side_effect=AssertionError("read")):
        assert images.resolve_image_model(
            None, base_url="https://api.openai.com/v1", api_key="k", mindshub=False
        ) == "gpt-image-2.6-flare"


def test_an_unreadable_catalogue_falls_back_and_is_asked_once():
    with patch("anton.minds_client.minds_request", side_effect=OSError("down")) as request:
        for _ in range(3):
            assert images.resolve_image_model(
                None, base_url=MINDSHUB, api_key="k", mindshub=True
            ) == images.DEFAULT_IMAGE_MODEL
    assert request.call_count == 1


async def test_the_client_resolves_the_default_with_its_mindshub_key():
    client = _client(Gateway(), image_model=None)
    with _catalog() as request:
        assert await client.resolve_image_model() == "gpt-image-2.6-sunburst"
    assert request.call_args.args[1] == "test-key"


def test_role_default_model_skips_rows_the_key_may_not_use():
    from anton.minds_client import role_default_model

    rows = [
        {"id": "a", "kind": "image", "default_for": ["image"], "enabled": False},
        {"id": "b", "kind": "image", "default_for": ["image"]},
    ]
    assert role_default_model(rows, "image", kind="image") == "b"
    assert role_default_model(rows, "planning") is None


# ── The image provider ───────────────────────────────────────────────


def test_an_anthropic_only_client_has_no_image_provider():
    client = LLMClient(
        planning_provider=AsyncMock(), planning_model="sonnet",
        coding_provider=AsyncMock(), coding_model="haiku",
    )
    assert client.image_provider is None


def test_a_role_on_mindshub_is_preferred_for_images():
    openai_direct = _provider(Gateway(), base_url="https://api.openai.com/v1", flavor=OpenAIProvider.FLAVOR_OPENAI)
    mindshub = _provider(Gateway())
    client = LLMClient(
        planning_provider=openai_direct, planning_model="gpt",
        coding_provider=mindshub, coding_model="haiku",
    )
    assert client.image_provider is mindshub


# ── The call ─────────────────────────────────────────────────────────


async def test_a_generation_posts_json_and_books_the_image_role():
    gateway = Gateway()
    client = _client(gateway)
    seen: list = []
    client.usage_listener = lambda role, model, usage: seen.append((role, model, usage))

    result = await client.generate_image(prompt="a lighthouse at dusk", size="1536x1024", quality="low")

    request = gateway.requests[0]
    assert request.url.path == "/v1/images/generations"
    assert request.headers["authorization"] == "Bearer test-key"
    body = gateway.json_body()
    assert body == {
        "model": "gpt-image-2.6-flare", "prompt": "a lighthouse at dusk",
        "size": "1536x1024", "quality": "low",
    }
    assert (result.mime_type, result.extension, result.width, result.height) == ("image/png", "png", 1536, 1024)
    assert result.data == PNG
    [(role, model, usage)] = seen
    assert (role, model) == ("image", "gpt-image-2.6-flare")
    # Text tokens stay in input/output; image tokens are their own components.
    assert (usage.input_tokens, usage.output_tokens) == (16, 0)
    assert (usage.image_input_tokens, usage.image_output_tokens) == (0, 824)


async def test_reference_images_make_a_multipart_edit():
    gateway = Gateway()
    client = _client(gateway)
    ref = images.ReferenceImage("photo.png", PNG, "image/png")

    await client.generate_image(prompt="make the sky blue", reference_images=[ref])
    await client.generate_image(prompt="combine them", reference_images=[ref, ref._replace(name="b.png")])

    single, double = gateway.requests
    assert single.url.path == double.url.path == "/v1/images/edits"
    assert single.headers["content-type"].startswith("multipart/form-data")
    assert b'name="image"; filename="photo.png"' in single.content
    assert b"Content-Type: image/png" in single.content
    assert b'name="model"\r\n\r\ngpt-image-2.6-flare' in single.content
    assert b'name="prompt"\r\n\r\nmake the sky blue' in single.content
    assert double.content.count(b'name="image[]"') == 2


@pytest.mark.parametrize("status, body, headers, code, phrase", [
    (400, {"error": {"code": "moderation_blocked", "message": "Your request was rejected by the safety system."}},
     {}, "moderation_blocked", "safety system refused"),
    (400, {"error": {"code": "unsupported_model_kind", "message": "not an image model"}},
     {}, "unsupported_model_kind", "is not an image model"),
    (402, {"error": {"code": "wallet_empty", "message": "Wallet is empty"}},
     {}, "wallet_empty", "No image was made"),
    (429, {"error": {"code": "rate_limited", "message": "slow down"}},
     {"retry-after": "20"}, "rate_limited", "wait 20 s"),
    (503, {"error": {"message": "upstream unavailable"}}, {}, "provider_error", "HTTP 503"),
    (404, {"error": {"code": "model_not_found", "message": "no such model"}}, {}, "model_not_found", "HTTP 404"),
])
async def test_failures_come_back_as_messages_to_relay(status, body, headers, code, phrase):
    gateway = Gateway(status=status, body=body, headers=headers)
    with pytest.raises(images.ImageGenerationError) as caught:
        await _client(gateway).generate_image(prompt="x")
    assert caught.value.code == code
    assert caught.value.status_code == status
    assert phrase in str(caught.value)


async def test_a_rejected_key_is_an_auth_error_after_one_confirmation_retry():
    gateway = Gateway(status=401, body={"error": {"message": "bad key"}})
    with pytest.raises(ProviderAuthError):
        await _client(gateway).generate_image(prompt="x")
    assert len(gateway.requests) == 2


async def test_a_reply_without_an_image_is_an_error():
    gateway = Gateway(body={"created": 1, "data": []})
    with pytest.raises(images.ImageGenerationError) as caught:
        await _client(gateway).generate_image(prompt="x")
    assert caught.value.code == "no_image"


async def test_no_image_role_without_an_openai_endpoint():
    client = LLMClient(
        planning_provider=AsyncMock(), planning_model="sonnet",
        coding_provider=AsyncMock(), coding_model="haiku",
    )
    with pytest.raises(images.ImageGenerationError) as caught:
        await client.generate_image(prompt="x")
    assert caught.value.code == "unavailable"


def test_sniff_reads_jpeg_and_webp_sizes():
    jpeg = (b"\xff\xd8\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x00" * 9
            + b"\xff\xc0" + struct.pack(">HBHH", 17, 8, 600, 800) + b"\x00" * 16)
    assert images.sniff_image(jpeg) == ("jpeg", 800, 600)
    webp = b"RIFF" + b"\x00" * 4 + b"WEBPVP8X" + b"\x00" * 8 + (639).to_bytes(3, "little") + (479).to_bytes(3, "little")
    assert images.sniff_image(webp) == ("webp", 640, 480)
    assert images.sniff_image(b"not an image") == (None, None, None)


# ── The tool ─────────────────────────────────────────────────────────


def _session(tmp_path: Path, gateway: Gateway, **kw) -> SimpleNamespace:
    workspace = Workspace(tmp_path)
    workspace.initialize()
    return SimpleNamespace(
        _workspace=workspace,
        _llm=_client(gateway),
        _artifacts_touched=set(),
        _images_this_turn=0,
        _max_images_per_turn=8,
        _session_id=None,
        _scratchpads=None,
        **kw,
    )


async def _run(session, tc_input: dict) -> tuple[list[ToolProgress], ToolOutcome]:
    progress: list[ToolProgress] = []
    outcome = None
    async for item in th.handle_generate_image(session, tc_input):
        if isinstance(item, ToolProgress):
            progress.append(item)
        else:
            outcome = item
    assert isinstance(outcome, ToolOutcome)
    return progress, outcome


async def test_the_tool_saves_an_image_artifact_and_returns_its_path_not_the_pixels(tmp_path):
    gateway = Gateway()
    session = _session(tmp_path, gateway)

    progress, outcome = await _run(session, {"prompt": "A watercolor fox in a snowy forest", "size": "1536x1024"})

    assert outcome.ok is True, outcome.content
    # The model never sees the picture: no base64, and a small result.
    assert PNG_B64 not in outcome.content and len(outcome.content) < 2000
    payload = json.loads(outcome.content)
    saved = Path(payload["absolute_path"])
    assert saved.read_bytes() == PNG
    assert saved.parent.parent == Path(session._workspace.artifacts_dir)
    # `path` is relative to the project root: what Cowork's files-raw route takes.
    assert payload["path"] == f".anton/artifacts/{payload['artifact_slug']}/{saved.name}"
    assert (tmp_path / payload["path"]).read_bytes() == PNG
    assert (payload["mime_type"], payload["width"], payload["height"]) == ("image/png", 1536, 1024)
    assert payload["model"] == "gpt-image-2.6-flare"
    assert payload["usage"] == {
        "text_input_tokens": 16, "image_input_tokens": 0,
        "text_output_tokens": 0, "image_output_tokens": 824,
    }
    metadata = json.loads((saved.parent / "metadata.json").read_text())
    assert (metadata["type"], metadata["primary"]) == ("image", saved.name)
    assert saved.name == "a-watercolor-fox-in-a-snowy.png"
    # The host builds the turn's artifact cards from this set.
    assert session._artifacts_touched == {payload["artifact_slug"]}
    assert session._images_this_turn == 1
    assert [p.text for p in progress] == ["Generating an image with gpt-image-2.6-flare"]


async def test_the_filename_names_the_file_and_picks_the_format(tmp_path):
    gateway = Gateway()
    _, outcome = await _run(_session(tmp_path, gateway), {"prompt": "x", "filename": "Hero Banner.jpg"})
    assert gateway.json_body()["output_format"] == "jpeg"
    # The stub answered PNG, so the file is named for what it holds.
    assert Path(json.loads(outcome.content)["path"]).name == "hero-banner.png"


async def test_a_filename_path_saves_into_an_existing_artifact_folder(tmp_path):
    from anton.core.artifacts import ArtifactStore

    session = _session(tmp_path, Gateway())
    deck = ArtifactStore(session._workspace.artifacts_dir).create(
        name="Pitch deck", description="d", type="document", primary="deck.md",
    )
    _, outcome = await _run(
        session, {"prompt": "x", "filename": f".anton/artifacts/{deck.slug}/slides/hero.png"}
    )
    assert outcome.ok is True, outcome.content
    payload = json.loads(outcome.content)
    folder = Path(session._workspace.artifacts_dir) / deck.slug
    assert (folder / "slides" / "hero.png").read_bytes() == PNG
    assert payload["path"] == f".anton/artifacts/{deck.slug}/slides/hero.png"
    assert payload["artifact_slug"] == deck.slug
    assert "slides/hero.png" in payload["note"]
    # No artifact of its own: only the deck's folder holds a metadata file.
    assert [m.parent.name for m in folder.parent.glob("*/metadata.json")] == [deck.slug]
    assert session._artifacts_touched == {deck.slug}


async def test_a_filename_path_saves_elsewhere_in_the_project_without_overwriting(tmp_path):
    session = _session(tmp_path, Gateway())
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "fox.png").write_bytes(b"theirs")
    _, outcome = await _run(session, {"prompt": "x", "filename": "images/fox.png"})
    payload = json.loads(outcome.content)
    assert payload["path"] == "images/fox-2.png" and payload["artifact_slug"] is None
    assert (tmp_path / "images" / "fox.png").read_bytes() == b"theirs"
    assert not any(Path(session._workspace.artifacts_dir).glob("*/metadata.json"))


@pytest.mark.parametrize("filename, phrase", [
    ("../outside.png", "'..'"),
    ("/etc/outside.png", "inside the project"),
    (".git/hooks.png", "dot-folder"),
    (".anton/artifacts/no-such-artifact/x.png", "no artifact 'no-such-artifact'"),
    (".anton/artifacts/x.png", "an artifact's folder and a file"),
])
async def test_a_filename_path_outside_the_allowed_places_sends_nothing(tmp_path, filename, phrase):
    gateway = Gateway()
    _, outcome = await _run(_session(tmp_path, gateway), {"prompt": "x", "filename": filename})
    assert outcome.ok is False and outcome.reason == "invalid_path"
    assert phrase in outcome.content
    assert gateway.requests == []


async def test_on_a_cloud_pod_artifact_paths_are_still_project_paths(tmp_path):
    """A pod's artifacts root is a mount outside its workspace; the project
    keeps it at .anton/artifacts, which is the path a host resolves."""
    from anton.core.artifacts import ArtifactStore

    workspace = SimpleNamespace(base=tmp_path / "conversation", artifacts_dir=tmp_path / "project-artifacts")
    workspace.base.mkdir()
    session = _session(tmp_path, Gateway())
    session._workspace = workspace
    _, outcome = await _run(session, {"prompt": "a fox"})
    payload = json.loads(outcome.content)
    assert payload["path"] == f".anton/artifacts/{payload['artifact_slug']}/a-fox.png"
    assert Path(payload["absolute_path"]).parent.parent == workspace.artifacts_dir

    deck = ArtifactStore(workspace.artifacts_dir).create(name="Deck", description="d", type="document")
    _, outcome = await _run(session, {"prompt": "x", "filename": f".anton/artifacts/{deck.slug}/hero.png"})
    assert (workspace.artifacts_dir / deck.slug / "hero.png").exists(), outcome.content


async def test_reference_images_are_read_from_the_project_and_sent_as_an_edit(tmp_path):
    (tmp_path / "photo.png").write_bytes(PNG)
    gateway = Gateway()
    progress, outcome = await _run(
        _session(tmp_path, gateway), {"prompt": "make it night", "reference_images": ["photo.png"]}
    )
    assert outcome.ok is True, outcome.content
    assert gateway.requests[0].url.path == "/v1/images/edits"
    assert json.loads(outcome.content)["edited_from"] == ["photo.png"]
    assert progress[0].text.startswith("Editing an image")


async def test_a_reference_that_is_not_an_image_sends_nothing(tmp_path):
    (tmp_path / "notes.txt").write_text("hello")
    gateway = Gateway()
    session = _session(tmp_path, gateway)
    _, outcome = await _run(session, {"prompt": "x", "reference_images": ["notes.txt"]})
    assert outcome.ok is False and "not a PNG, JPEG or WebP" in outcome.content
    assert gateway.requests == [] and session._images_this_turn == 0


async def test_past_the_turn_cap_no_request_is_sent(tmp_path):
    gateway = Gateway()
    session = _session(tmp_path, gateway, )
    session._images_this_turn = session._max_images_per_turn = 2
    _, outcome = await _run(session, {"prompt": "one more"})
    assert outcome.ok is False and outcome.reason == "image_limit"
    assert "ANTON_MAX_IMAGES_PER_TURN=2" in outcome.content and "no request was sent" in outcome.content
    assert gateway.requests == []


async def test_a_safety_refusal_is_relayed_and_leaves_no_artifact(tmp_path):
    gateway = Gateway(status=400, body={"error": {"code": "moderation_blocked", "message": "blocked"}})
    session = _session(tmp_path, gateway)
    _, outcome = await _run(session, {"prompt": "something flagged"})
    assert outcome.ok is False and outcome.reason == "image_refused"
    assert "safety system refused" in outcome.content
    assert not any(Path(session._workspace.artifacts_dir).glob("*/metadata.json"))
    assert session._images_this_turn == 1  # a refused call may still be billed


@pytest.mark.parametrize("tc_input, reason", [
    ({}, "missing_name"),
    ({"prompt": "x", "size": "huge"}, "invalid_type"),
    ({"prompt": "x", "quality": "ultra"}, "invalid_type"),
    ({"prompt": "x", "filename": "a.gif"}, "invalid_type"),
])
async def test_bad_arguments_are_refused_before_any_request(tmp_path, tc_input, reason):
    gateway = Gateway()
    _, outcome = await _run(_session(tmp_path, gateway), tc_input)
    assert outcome.ok is False and outcome.reason == reason
    assert gateway.requests == []


def test_every_image_failure_reason_has_a_tier():
    from anton.core.root_cause import _SENTINEL_REASONS

    assert set(th._IMAGE_FAILURE_REASONS.values()) | {"image_failed", "image_limit"} <= set(_SENTINEL_REASONS)


async def test_the_registry_dispatches_the_streaming_handler(tmp_path):
    registry = ToolRegistry()
    registry.register_tool(GENERATE_IMAGE_TOOL)
    outcome = await registry.dispatch_tool(_session(tmp_path, Gateway()), "generate_image", {"prompt": "x"})
    assert outcome.ok is True and json.loads(outcome.content)["status"] == "generated"


def test_the_schema_offers_the_contract_fields():
    props = GENERATE_IMAGE_TOOL.input_schema["properties"]
    assert set(props) == {"prompt", "reference_images", "size", "quality", "filename"}
    assert GENERATE_IMAGE_TOOL.input_schema["required"] == ["prompt"]


# ── The session ──────────────────────────────────────────────────────


def _image_tool(outcome: ToolOutcome):
    async def handler(_session, _input):
        yield ToolProgress("Generating an image with m")
        yield outcome

    return GENERATE_IMAGE_TOOL.__class__(
        name="generate_image", description="stub", input_schema={"type": "object", "properties": {}},
        handler=handler,
    )


async def _turn_events(tool_outcome: ToolOutcome) -> list:
    from anton.core.llm.provider import LLMResponse, StreamComplete, ToolCall

    llm = make_mock_llm()
    responses = [
        LLMResponse(content="", tool_calls=[ToolCall(id="toolu_1", name="generate_image", input={"prompt": "a fox"})],
                    usage=Usage(input_tokens=10, output_tokens=5), stop_reason="tool_use"),
        LLMResponse(content="Here it is.", usage=Usage(input_tokens=10, output_tokens=5), stop_reason="end_turn"),
    ]

    def plan_stream(**_kw):
        async def gen():
            yield StreamComplete(response=responses.pop(0))
        return gen()

    llm.plan_stream = plan_stream
    session = ChatSession(ChatSessionConfig(llm_client=llm, workspace=MagicMock(base=Path.cwd())))
    session.tool_registry.register_tool(_image_tool(tool_outcome))
    try:
        return [event async for event in session.turn_stream("draw a fox")]
    finally:
        await session.close()


async def test_a_saved_image_reaches_the_host_as_a_tool_result_after_tool_done():
    from anton.core.llm.provider import StreamTaskProgress, StreamToolResult

    result = json.dumps({"status": "generated", "path": ".anton/artifacts/fox-1/fox.png", "width": 1024})
    events = await _turn_events(ToolOutcome(content=result, ok=True))
    [relayed] = [e for e in events if isinstance(e, StreamToolResult)]
    assert (relayed.name, relayed.id, relayed.content) == ("generate_image", "toolu_1", result)
    done = next(i for i, e in enumerate(events)
                if isinstance(e, StreamTaskProgress) and e.phase == "tool_done" and e.id == "toolu_1")
    assert events.index(relayed) > done


async def test_a_failed_image_sends_no_tool_result():
    from anton.core.llm.provider import StreamToolResult

    events = await _turn_events(ToolOutcome(content="Error: blocked", ok=False, reason="image_refused"))
    assert not [e for e in events if isinstance(e, StreamToolResult)]



async def _first_request(tmp_path: Path, llm) -> tuple[str, set[str]]:
    workspace = Workspace(tmp_path / "workspace")
    workspace.initialize()
    session = ChatSession(ChatSessionConfig(llm_client=llm, workspace=workspace))
    try:
        await run_turn(session, "Draw me a cat.")
    finally:
        await session.close()
    kwargs = llm.plan.call_args_list[0].kwargs
    return kwargs["system"], {tool["name"] for tool in kwargs["tools"]}


@pytest.mark.parametrize("flavor, base_url, image_model, offered", [
    (OpenAIProvider.FLAVOR_MINDS_PASSTHROUGH, MINDSHUB, None, True),
    (OpenAIProvider.FLAVOR_OPENAI, "https://api.openai.com/v1", None, True),
    (OpenAIProvider.FLAVOR_OPENAI_COMPATIBLE_GENERIC, "http://localhost:11434/v1", None, False),
    (OpenAIProvider.FLAVOR_OPENAI_COMPATIBLE_GENERIC, "http://localhost:11434/v1", "local-image", True),
])
async def test_the_tool_and_its_prompt_are_offered_where_an_images_api_is(
    tmp_path, flavor, base_url, image_model, offered,
):
    llm = make_mock_llm()
    llm.image_provider = OpenAIProvider(api_key="k", base_url=base_url, flavor=flavor)
    llm.image_model = image_model
    system, tools = await _first_request(tmp_path, llm)
    assert ("generate_image" in tools) is offered
    assert (GENERATE_IMAGE_TOOL.prompt in system) is offered


async def test_an_anthropic_only_session_has_no_image_tool(tmp_path):
    system, tools = await _first_request(tmp_path, make_mock_llm())
    assert "generate_image" not in tools
    assert "IMAGES:" not in system


async def test_the_session_hands_its_image_model_to_a_host_built_client_and_its_pads(tmp_path):
    client = _client(Gateway(), image_model=None)
    settings = AntonSettings(_env_file=None, image_model="gpt-image-2.6-sunburst", max_images_per_turn=3)
    session = ChatSession(ChatSessionConfig(llm_client=client, settings=settings))
    try:
        assert client.image_model == "gpt-image-2.6-sunburst"
        assert session._scratchpads._image_model == "gpt-image-2.6-sunburst"
        assert session._max_images_per_turn == 3
    finally:
        await session.close()


class _Pad:
    cell_sandbox = None

    async def start(self):
        pass


def _factory(seen: dict, *, takes_image_model: bool):
    if takes_image_model:
        def factory(*, name, cells, coding_provider, coding_model, coding_api_key,
                    coding_base_url, workspace_path, image_model=None):
            seen["image_model"] = image_model
            return _Pad()
    else:
        def factory(*, name, cells, coding_provider, coding_model, coding_api_key,
                    coding_base_url, workspace_path):
            seen["called"] = True
            return _Pad()
    return factory


@pytest.mark.parametrize("image_model, takes, expected", [
    ("gpt-image-2.6-sunburst", True, {"image_model": "gpt-image-2.6-sunburst"}),
    (None, True, {"image_model": None}),
    ("gpt-image-2.6-sunburst", False, {"called": True}),
])
async def test_the_manager_hands_the_image_model_to_a_factory_that_takes_it(image_model, takes, expected):
    from anton.core.backends.manager import ScratchpadManager

    seen: dict = {}
    manager = ScratchpadManager(
        runtime_factory=_factory(seen, takes_image_model=takes),
        coding_provider="openai", coding_model="m", coding_api_key="k",
        coding_base_url=MINDSHUB, image_model=image_model,
    )
    await manager.get_or_create("p")
    assert seen == expected


async def test_the_turn_cap_resets_each_turn(tmp_path):
    workspace = Workspace(tmp_path)
    workspace.initialize()
    session = ChatSession(ChatSessionConfig(llm_client=make_mock_llm(), workspace=workspace))
    try:
        session._images_this_turn = 8
        await run_turn(session, "hi")
        assert session._images_this_turn == 0
    finally:
        await session.close()


# ── The turn cost ────────────────────────────────────────────────────


def test_image_calls_are_their_own_role_with_image_tokens_visible():
    assert "image" in EVENT_ROLES
    tc = TurnCost()
    tc.add("planning", "sonnet", Usage(input_tokens=1000, output_tokens=100))
    tc.add("image", "gpt-image-2.6-flare", Usage(input_tokens=16, image_input_tokens=300, image_output_tokens=824))
    assert (tc.image_input_tokens, tc.image_output_tokens) == (300, 824)
    assert tc.input_tokens == 1016  # text only
    assert tc.by_role["image"].model == "gpt-image-2.6-flare"
    assert tc.by_role["image"].tokens == 1140
    assert tc.total_tokens == 2240
    assert sum(s.tokens for s in tc.by_role.values()) == tc.total_tokens
    assert "unknown" not in tc.by_role


def test_the_turn_event_carries_the_image_role(caplog):
    s = ChatSession.__new__(ChatSession)
    s._turn_cost = TurnCost()
    s._turn_cost.add("image", "gpt-image-2.6-flare", Usage(input_tokens=16, image_output_tokens=824))
    s._llm = MagicMock(planning_model="planner", coding_model="coder")
    for name, value in (("_session_id", "conv-1"), ("_harness", "cowork"), ("_surface", "desktop"),
                        ("_user_id", None), ("_organization_id", None), ("_turn_count", 0),
                        ("_settings", None)):
        setattr(s, name, value)
    s._cancel_event = MagicMock(is_set=lambda: False)
    with patch("anton.analytics.send_event") as send:
        with caplog.at_level(logging.INFO, logger="anton.core.session"):
            s._emit_turn_cost()
    kw = send.call_args.kwargs
    assert (kw["image_model"], kw["image_tokens"], kw["image_calls"]) == ("gpt-image-2.6-flare", "840", "1")
    assert (kw["image_input_tokens"], kw["image_output_tokens"]) == ("0", "824")
    assert kw["tokens_total"] == "840"
    line = next(r.getMessage() for r in caplog.records if "turn_cost" in r.getMessage())
    assert "image=gpt-image-2.6-flare:840/1" in line and "image_output=824" in line


def test_a_chat_only_turn_line_is_unchanged(caplog):
    s = ChatSession.__new__(ChatSession)
    s._turn_cost = TurnCost()
    s._turn_cost.add("planning", "planner", Usage(input_tokens=10, output_tokens=1))
    s._llm = MagicMock(planning_model="planner", coding_model="coder")
    for name, value in (("_session_id", "conv-1"), ("_harness", "cowork"), ("_surface", "desktop"),
                        ("_user_id", None), ("_organization_id", None), ("_turn_count", 0),
                        ("_settings", None)):
        setattr(s, name, value)
    s._cancel_event = MagicMock(is_set=lambda: False)
    with patch("anton.analytics.send_event") as send:
        with caplog.at_level(logging.INFO, logger="anton.core.session"):
            s._emit_turn_cost()
    line = next(r.getMessage() for r in caplog.records if "turn_cost" in r.getMessage())
    assert "image_input=" not in line
    assert send.call_args.kwargs["image_calls"] == "0"


def test_the_usage_log_record_names_image_tokens_only_when_there_are_some():
    from anton.core.llm.provider import _usage_record, scratchpad_usage_record

    chat = _usage_record(role="planning", model="m", served_model=None, input_tokens=1,
                         output_tokens=1, cache_read=0, cache_creation=0)
    assert "image_output" not in chat
    image = scratchpad_usage_record({"role": "image", "model": "m", "input": 16, "image_output": 824})
    assert (image["role"], image["input"], image["image_output"]) == ("image", 16, 824)


# ── Images in artifacts ──────────────────────────────────────────────


def test_an_artifact_file_can_be_attached_to_a_page(tmp_path):
    from anton.core.tools.generate_artifact.attachments import resolve_attachments

    artifacts = tmp_path / ".anton" / "artifacts"
    image = artifacts / "fox-1234abcd" / "fox.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(PNG)
    hidden = artifacts / "fox-1234abcd" / ".published.json"
    hidden.write_text("{}")

    kept, dropped = resolve_attachments([str(image), str(hidden)], workspace=tmp_path)
    assert kept == [] and len(dropped) == 2  # without the root, `.anton/` is refused

    kept, dropped = resolve_attachments(
        [str(image), str(hidden)], workspace=tmp_path, owned_roots=(artifacts,)
    )
    assert [a.name for a in kept] == ["fox.png"]
    assert len(dropped) == 1 and ".published.json" in dropped[0]
