"""Image generation through the OpenAI images API (ENG-4024).

MindsHub serves OpenAI's own ``/v1/images/generations`` (JSON) and
``/v1/images/edits`` (multipart) on the base URL and key anton already sends
its chat calls to, so an image call goes out on the same ``AsyncOpenAI``
client: `OpenAIProvider.generate_image`. This module holds what the two
callers share, so they read a reply, its usage and its errors the same way:

- the ``generate_image`` tool, through `LLMClient.generate_image`, in the host;
- the scratchpad's ``generate_image()`` helper (`scratchpad_boot`), in the pad.

The picture never reaches the model. Both callers write the bytes to a file and
hand the agent the path; a base64 string in a tool result would cost a
megabyte of context for nothing the model can read.
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
import re
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import openai

from .provider import (
    ProviderAuthError,
    ProviderErrorBody,
    Usage,
    mindshub_billing_stop,
    mindshub_model_restriction,
    origin_is_known_third_party,
    retry_after_seconds,
)

logger = logging.getLogger(__name__)

#: The image model when nothing names one: the setting is unset and MindsHub's
#: catalogue named no default for the ``image`` role (or anton is not on
#: MindsHub). The same alias auth marks ``default_for: ["image"]``.
DEFAULT_IMAGE_MODEL = "gpt-image-2.6-flare"

#: The role image calls are counted and traced under (`TurnCost`, Langfuse).
IMAGE_ROLE = "image"

#: How many images one agent turn (the tool) or one scratchpad cell (the
#: helper) may make, unless configured otherwise.
DEFAULT_MAX_IMAGES = 8

QUALITIES = ("auto", "low", "medium", "high")
SIZES = ("auto", "1024x1024", "1536x1024", "1024x1536")
_SIZE_RE = re.compile(r"^[1-9]\d{1,4}x[1-9]\d{1,4}$")

# OpenAI's limits for an edit: at most 16 reference images, each under 50 MB.
MAX_REFERENCE_IMAGES = 16
MAX_REFERENCE_BYTES = 50 * 1024 * 1024

# What the images API returns, by the extension a file of it gets.
_FORMATS = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}
_SUFFIX_FORMATS = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp"}
# What an edit accepts as a reference image.
_REFERENCE_FORMATS = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}

_CATALOG_TTL_S = 3600.0
_catalog_defaults: dict[str, tuple[float, str | None]] = {}
_catalog_lock = threading.Lock()


class ReferenceImage(NamedTuple):
    """An image an edit starts from: its file name, bytes and media type."""

    name: str
    data: bytes
    mime_type: str


@dataclass
class GeneratedImage:
    """One image the API returned, decoded, and what the call cost."""

    data: bytes
    mime_type: str
    #: File extension for the bytes' real format, without the dot.
    extension: str
    width: int | None
    height: int | None
    model: str
    usage: Usage
    revised_prompt: str | None = None


class ImageGenerationError(Exception):
    """An image call that made no image, worded for the agent to tell the user.

    ``code`` is the short cause a caller can branch on: the provider's own
    (``moderation_blocked``, ``unsupported_model_kind``, a MindsHub billing
    code) or one of ours (``rate_limited``, ``provider_error``,
    ``connection_error``, ``model_not_found``, ``bad_request``, ``no_image``,
    ``unavailable``).
    """

    def __init__(self, message: str, *, code: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


# ── Arguments ────────────────────────────────────────────────────────


def normalize_size(value: object) -> str | None:
    """``size`` as the API takes it, or None to let the model choose. Raises ValueError."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    text = str(value).strip().lower().replace(" ", "").replace("×", "x")
    if text in SIZES or _SIZE_RE.match(text):
        return text
    raise ValueError(
        f"size must be one of {', '.join(SIZES)} (or another WIDTHxHEIGHT the model accepts), "
        f"not {value!r}"
    )


def normalize_quality(value: object) -> str | None:
    """``quality`` as the API takes it, or None for the model's default. Raises ValueError."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    text = str(value).strip().lower()
    if text in QUALITIES:
        return text
    raise ValueError(f"quality must be one of {', '.join(QUALITIES)}, not {value!r}")


def output_format_for(name: str | None) -> str | None:
    """The ``output_format`` a file name asks for (``png``/``jpeg``/``webp``).

    None for a name without a suffix, which gets the API's default (PNG).
    Raises ValueError for a suffix the API cannot produce.
    """
    if not name:
        return None
    suffix = Path(name).suffix.lower()
    if not suffix:
        return None
    if suffix not in _SUFFIX_FORMATS:
        raise ValueError(
            f"the images API writes .png, .jpg or .webp files, not {suffix!r}"
        )
    return _SUFFIX_FORMATS[suffix]


_STEM_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def file_stem(name: str | None, prompt: str) -> str:
    """A safe file name stem: the given name's, else the prompt's first words."""
    raw = Path(name).stem if name else " ".join(prompt.split()[:6])
    stem = _STEM_UNSAFE.sub("-", raw).strip("-._").lower()[:60].rstrip("-._")
    return stem or "image"


_TITLE_WORDS = 8


def image_title(name: str | None, prompt: str) -> str:
    """An image artifact's name: the file name's words, else the prompt's first ones."""
    if name:
        words = re.sub(r"[-_.]+", " ", Path(name).stem).strip()
        if words:
            return words[:80]
    words = prompt.split()
    title = " ".join(words[:_TITLE_WORDS])
    if len(words) > _TITLE_WORDS:
        title += "…"
    return title[:80] or "Generated image"


def load_reference_image(path: Path) -> ReferenceImage:
    """Read one reference image for an edit. Raises ValueError with the reason.

    The format is read from the bytes, not the name: the edit is sent as
    multipart with the media type, and a JPEG named ``.png`` must not go out
    labelled PNG.
    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc.strerror or exc}") from exc
    if not path.is_file():
        raise ValueError(f"{path} is not a file")
    if size > MAX_REFERENCE_BYTES:
        raise ValueError(f"{path.name} is larger than {MAX_REFERENCE_BYTES // (1024 * 1024)} MB")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc.strerror or exc}") from exc
    fmt = sniff_image(data)[0]
    if fmt not in _REFERENCE_FORMATS:
        raise ValueError(f"{path.name} is not a PNG, JPEG or WebP image")
    return ReferenceImage(name=path.name, data=data, mime_type=_REFERENCE_FORMATS[fmt])


# ── The image bytes ──────────────────────────────────────────────────


def sniff_image(data: bytes) -> tuple[str | None, int | None, int | None]:
    """(format, width, height) read from the bytes: ``png``, ``jpeg``, ``webp`` or ``gif``.

    Format None for anything else. A size it cannot read is None: the header
    parse is best-effort and never raises, since the bytes are still a usable
    image the user asked for.
    """
    try:
        if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
            width, height = struct.unpack(">II", data[16:24])
            return "png", width, height
        if data.startswith(b"\xff\xd8\xff"):
            return ("jpeg", *_jpeg_size(data))
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return ("webp", *_webp_size(data))
        if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
            width, height = struct.unpack("<HH", data[6:10])
            return "gif", width, height
    except struct.error:
        pass
    return None, None, None


def _jpeg_size(data: bytes) -> tuple[int | None, int | None]:
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7 or marker == 0xFF:
            i += 1 if marker == 0xFF else 2
            continue
        (length,) = struct.unpack(">H", data[i + 2:i + 4])
        # Start-of-frame markers carry the size; DHT, JPG and DAC share the range.
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height, width = struct.unpack(">HH", data[i + 5:i + 9])
            return width, height
        i += 2 + length
    return None, None


def _webp_size(data: bytes) -> tuple[int | None, int | None]:
    chunk = data[12:16]
    if chunk == b"VP8 " and len(data) >= 30:
        width, height = struct.unpack("<HH", data[26:30])
        return width & 0x3FFF, height & 0x3FFF
    if chunk == b"VP8L" and len(data) >= 25:
        bits = int.from_bytes(data[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if chunk == b"VP8X" and len(data) >= 30:
        width = int.from_bytes(data[24:27], "little") + 1
        height = int.from_bytes(data[27:30], "little") + 1
        return width, height
    return None, None


def write_new_image_file(
    data: bytes, directory: Path, stem: str, extension: str, *, root: Path | None = None
) -> Path:
    """Write ``data`` to a NEW file ``directory/stem.extension`` and return its path.

    Never overwrites: a taken name gets ``-2``, ``-3``, ... like artifact slugs
    do. With ``root``, every folder below it is opened without following a
    symlink (`open_nofollow`), the rule for a file the host process writes
    into a project agent code can plant links in.
    """
    from anton.utils.nofollow import open_nofollow

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    for attempt in range(1, 1000):
        name = f"{stem}.{extension}" if attempt == 1 else f"{stem}-{attempt}.{extension}"
        target = directory / name
        try:
            if root is not None:
                fd = open_nofollow(root, target, flags, 0o666, create_parents=True)
            else:
                directory.mkdir(parents=True, exist_ok=True)
                fd = os.open(target, flags, 0o666)
        except FileExistsError:
            continue
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        return target
    raise OSError(f"no free file name for {stem}.{extension} in {directory}")


# ── The call ─────────────────────────────────────────────────────────


def _field(obj: Any, name: str) -> Any:
    """``obj.name`` for an SDK model, ``obj[name]`` for a dict, else None."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _count(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def image_call_usage(raw: Any) -> Usage:
    """The images API's usage as a `Usage`: text tokens in ``input``/``output``,
    image tokens in ``image_input``/``image_output``.

    OpenAI reports ``input_tokens`` as text and reference images together and
    splits them in ``input_tokens_details``. ``output_tokens`` is the picture:
    GPT Image writes no text, and ``output_tokens_details`` may be absent.
    """
    if raw is None:
        return Usage()
    input_total = _count(_field(raw, "input_tokens"))
    input_details = _field(raw, "input_tokens_details")
    image_in = _count(_field(input_details, "image_tokens"))
    text_in = _field(input_details, "text_tokens")
    text_in = _count(text_in) if text_in is not None else max(0, input_total - image_in)
    output_total = _count(_field(raw, "output_tokens"))
    output_details = _field(raw, "output_tokens_details")
    if output_details is not None:
        image_out = _count(_field(output_details, "image_tokens"))
        text_out = _count(_field(output_details, "text_tokens"))
    else:
        image_out, text_out = output_total, 0
    return Usage(
        input_tokens=text_in,
        output_tokens=text_out,
        image_input_tokens=image_in,
        image_output_tokens=image_out,
    )


def parse_image_response(response: Any, model: str) -> GeneratedImage:
    """Decode the first image of an images API reply. Raises ImageGenerationError."""
    data = _field(response, "data") or []
    first = data[0] if data else None
    encoded = _field(first, "b64_json") if first is not None else None
    if not isinstance(encoded, str) or not encoded:
        raise ImageGenerationError(
            "The image provider answered without an image, so nothing was saved. "
            "Try once more; if it happens again, tell the user.",
            code="no_image",
        )
    try:
        raw = base64.b64decode(encoded)
    except (binascii.Error, ValueError) as exc:
        raise ImageGenerationError(
            "The image provider sent image data that could not be decoded, so nothing was saved.",
            code="no_image",
        ) from exc
    fmt, width, height = sniff_image(raw)
    if fmt not in _FORMATS:
        raise ImageGenerationError(
            "The image provider sent data that is not a PNG, JPEG or WebP image, so nothing was saved.",
            code="no_image",
        )
    revised = _field(first, "revised_prompt")
    return GeneratedImage(
        data=raw,
        mime_type=_FORMATS[fmt],
        extension="jpg" if fmt == "jpeg" else fmt,
        width=width,
        height=height,
        model=model,
        usage=image_call_usage(_field(response, "usage")),
        revised_prompt=revised if isinstance(revised, str) and revised.strip() else None,
    )


async def request_image(
    client: Any,
    *,
    model: str,
    prompt: str,
    size: str | None = None,
    quality: str | None = None,
    output_format: str | None = None,
    reference_images: tuple[ReferenceImage, ...] | list[ReferenceImage] = (),
    extra_headers: dict[str, str] | None = None,
) -> GeneratedImage:
    """One image from ``client`` (an ``AsyncOpenAI``): a generation, or an edit
    of ``reference_images`` sent as multipart. Raises ImageGenerationError, or
    ProviderAuthError for a rejected key so the caller's auth retry can run.
    """
    kwargs: dict[str, Any] = {"model": model, "prompt": prompt}
    if size:
        kwargs["size"] = size
    if quality:
        kwargs["quality"] = quality
    if output_format:
        kwargs["output_format"] = output_format
    if extra_headers:
        kwargs["extra_headers"] = extra_headers
    try:
        if reference_images:
            files = [(ref.name, ref.data, ref.mime_type) for ref in reference_images]
            # One file goes as `image`, several as `image[]`: the SDK picks the
            # field name from the shape, and MindsHub accepts both.
            response = await client.images.edit(image=files if len(files) > 1 else files[0], **kwargs)
        else:
            response = await client.images.generate(**kwargs)
    except openai.APIStatusError as exc:
        raise image_call_error(exc, model) from exc
    except openai.APIConnectionError as exc:
        raise ImageGenerationError(
            "Could not reach the image provider, so no image was made. Check the connection "
            "or try again in a moment.",
            code="connection_error",
        ) from exc
    return parse_image_response(response, model)


def _clip(text: str, limit: int = 300) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def image_call_error(exc: "openai.APIStatusError", model: str) -> Exception:
    """The error an images API failure becomes: ProviderAuthError for a 401,
    else an ImageGenerationError whose message the agent can relay as is.

    Billing and model-restriction copy comes from the same helpers the chat
    mappers use, origin-gated the same way, so an image call out of credits
    reads exactly like a chat call out of credits.
    """
    status = exc.status_code
    body = exc.body
    if isinstance(body, list) and body and isinstance(body[0], dict):
        body = body[0]
    parsed = ProviderErrorBody.parse(body)
    code = parsed.code if isinstance(parsed.code, str) else ""
    message = parsed.envelope.message or parsed.top.message or ""
    said = f' The provider said: "{_clip(message)}"' if message else ""

    if status == 401:
        return ProviderAuthError(
            "Invalid API key — the image endpoint refused the key anton sends its model calls with."
        )
    if code == "moderation_blocked" or "safety system" in message.lower():
        return ImageGenerationError(
            "The image provider's safety system refused this request (moderation_blocked), so no "
            "image was made. Tell the user. If a reworded prompt would still serve what they asked "
            "for without the flagged content, you may offer one; do not retry the same prompt.",
            code="moderation_blocked", status_code=status,
        )
    if code == "unsupported_model_kind":
        return ImageGenerationError(
            f"'{model}' is not an image model, so the images API refused it "
            f"(unsupported_model_kind). The image model setting must name an image model such as "
            f"{DEFAULT_IMAGE_MODEL}: tell the user to pick one in Settings.",
            code="unsupported_model_kind", status_code=status,
        )
    billing = mindshub_billing_stop(exc=exc, body=body, status_code=status)
    if billing is not None:
        return ImageGenerationError(
            f"{billing} No image was made.", code=billing.reason, status_code=status,
        )
    restriction = mindshub_model_restriction(exc=exc, body=body, status_code=status, model=model)
    if restriction is not None:
        return ImageGenerationError(str(restriction), code="model_restricted", status_code=status)
    if status == 403 and code in ("model_access_denied", "model_disabled") and not origin_is_known_third_party(exc):
        return ImageGenerationError(
            f"The image model '{model}' isn't available on the user's MindsHub plan, or is "
            "disabled right now, so no image was made. Tell the user; they can pick another image "
            "model in Settings or upgrade at https://console.mindshub.ai.",
            code=code, status_code=status,
        )
    if status == 404:
        return ImageGenerationError(
            f"The image model '{model}' or the images API was not found at this endpoint (HTTP 404), "
            f"so no image was made.{said}",
            code="model_not_found", status_code=status,
        )
    if status == 429:
        wait = retry_after_seconds(exc)
        hint = f" It asked to wait {wait:.0f} s." if wait else ""
        return ImageGenerationError(
            f"The image provider is rate limiting requests (HTTP 429), so no image was made.{hint} "
            "Wait before trying again, or tell the user.",
            code="rate_limited", status_code=status,
        )
    if status == 402:
        return ImageGenerationError(
            f"The image provider refused the request for payment reasons (HTTP 402), so no image was "
            f"made.{said} Tell the user.",
            code=code or "payment_required", status_code=status,
        )
    if status >= 500:
        return ImageGenerationError(
            f"The image provider failed (HTTP {status}), so no image was made. This is usually "
            "temporary: try once more, and tell the user if it fails again.",
            code="provider_error", status_code=status,
        )
    return ImageGenerationError(
        f"The image request was rejected (HTTP {status}{', ' + code if code else ''}), so no image "
        f"was made.{said}",
        code=code or "bad_request", status_code=status,
    )


# ── Which model ──────────────────────────────────────────────────────


def catalog_image_model(base_url: str, api_key: str, *, verify: bool = True) -> str | None:
    """The alias MindsHub's ``/v1/models`` marks ``default_for: ["image"]``, or None.

    Read through `anton.minds_client` — the catalogue reader setup already
    uses — and kept per base URL for an hour, so a host that builds a client
    per turn asks once. A catalogue that cannot be read is remembered as None
    for the same hour: the caller then falls back to `DEFAULT_IMAGE_MODEL`,
    which is the same alias, rather than paying a timeout on every call.
    Blocking: call it off the event loop.
    """
    from anton.minds_client import fetch_model_catalog, minds_v1_base, role_default_model

    key = minds_v1_base(base_url)
    now = time.monotonic()
    with _catalog_lock:
        cached = _catalog_defaults.get(key)
    if cached is not None and now - cached[0] < _CATALOG_TTL_S:
        return cached[1]
    try:
        entries = fetch_model_catalog(base_url, api_key, verify=verify, timeout=5)
        found = role_default_model(entries, IMAGE_ROLE, kind="image")
    except Exception as exc:
        logger.info("image model: MindsHub catalogue unreadable (%s); using %s",
                    type(exc).__name__, DEFAULT_IMAGE_MODEL)
        found = None
    with _catalog_lock:
        _catalog_defaults[key] = (now, found)
    return found


def resolve_image_model(
    configured: str | None,
    *,
    base_url: str | None,
    api_key: str | None,
    verify: bool = True,
    mindshub: bool = False,
) -> str:
    """The model an image call asks for: the setting, else MindsHub's ``image``
    role default, else `DEFAULT_IMAGE_MODEL`. Blocking when it reads the catalogue."""
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    if mindshub and base_url and api_key:
        found = catalog_image_model(base_url, api_key, verify=verify)
        if found:
            return found
    return DEFAULT_IMAGE_MODEL


def is_mindshub_endpoint(flavor: str | None, base_url: str | None) -> bool:
    """Whether a provider with this flavor and base URL talks to MindsHub."""
    from urllib.parse import urlparse

    from .openai import OpenAIProvider
    from .provider import is_mindshub_host

    if flavor == OpenAIProvider.FLAVOR_MINDS_PASSTHROUGH:
        return True
    try:
        return is_mindshub_host(urlparse(base_url or "").hostname)
    except ValueError:
        return False


def usage_summary(usage: Usage) -> dict[str, int]:
    """An image call's tokens as the tool result and the helper report them."""
    return {
        "text_input_tokens": usage.input_tokens,
        "image_input_tokens": usage.image_input_tokens,
        "text_output_tokens": usage.output_tokens,
        "image_output_tokens": usage.image_output_tokens,
    }
