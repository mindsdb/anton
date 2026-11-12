from __future__ import annotations

import re
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, SecretStr, field_validator, model_validator

#: A browser session or project key the hosts accept: 1-64 letters, digits,
#: '-' or '_'. The hosts check session ids against the same pattern.
_KEY = re.compile(r"[A-Za-z0-9_-]{1,64}")

#: The only address the Cowork desktop app's own browser listens on.
LOOPBACK_HOST = "127.0.0.1"

#: A launch key is a bearer header value: visible ASCII, no spaces.
_TOKEN = re.compile(r"[\x21-\x7e]+")


class DecisionsTarget(BaseModel):
    """A decision model of the host's own that picks the browser's steps,
    instead of Jev on the session's MindsHub connection.

    ``url`` is the endpoint's full URL, which takes the decision body
    ``anton.core.llm.jev.decide`` sends (model, state, questions): hosts serve
    it at different paths (TypeSafe's ``/v1/systemone``, a gateway's
    ``/v1/decisions``), so nothing is appended to it. ``model`` is the name
    that endpoint expects. Only a host builds one, in its own process, as it
    does the in-app browser's launch key: ``from_dict`` never reads one and
    ``to_dict`` refuses one. ``repr`` and validation errors leave the key out.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    url: str
    api_key: SecretStr
    model: str = "jev"

    @field_validator("url")
    @classmethod
    def _url_is_http(cls, value: str) -> str:
        url = urlsplit(value)
        if url.scheme not in ("http", "https") or not url.hostname or "@" in url.netloc:
            raise ValueError("the decision endpoint must be an http(s) URL with a host and no user")
        return value

    @field_validator("api_key")
    @classmethod
    def _key_is_set(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("the decision endpoint needs its API key")
        return value

    @field_validator("model")
    @classmethod
    def _model_is_set(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("the decision endpoint needs a model name")
        return value.strip()


class DecisionsProxy(BaseModel):
    """A decision model of the host's own, reached through the host: the
    form a turn request can carry, since it holds no key.

    A cloud pod must not hold the organization's decision model key, so the
    host serves a route that forwards each decision to that model with the
    key it keeps. ``url`` is that route's full URL (https, no user, query or
    fragment), and the tool calls it with the session's MindsHub credential,
    the turn key the router takes too, which the host checks and maps to the
    organization. ``model`` is the name the organization's endpoint expects.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    url: str
    model: str = "jev"

    @field_validator("url")
    @classmethod
    def _url_is_https(cls, value: str) -> str:
        url = urlsplit(value)
        if url.scheme != "https" or not url.hostname or "@" in url.netloc or url.query or url.fragment:
            raise ValueError("the decisions proxy must be https://<host>[/path], with no user, query or fragment")
        return value

    @field_validator("model")
    @classmethod
    def _model_is_set(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("the decisions proxy needs a model name")
        return value.strip()

    @classmethod
    def from_dict(cls, raw: object) -> DecisionsProxy | None:
        """A turn request's ``{"url", "model"?}``, or None when unusable. A
        block with any other key, a ``api_key`` above all, is refused whole."""
        if not isinstance(raw, dict) or not set(raw) <= {"url", "model"}:
            return None
        url, model = raw.get("url"), raw.get("model", "jev")
        if not isinstance(url, str) or not isinstance(model, str):
            return None
        try:
            return cls(url=url, model=model)
        except ValueError:
            return None


class BrowserConfig(BaseModel):
    """Where the browser the tool drives is, and which session to drive.

    Exactly two forms, checked whenever one is built:

    - The MindsHub browser router: ``base_url`` is ``https://<host>`` with an
      optional path (``https://app.example.test/browser``), and ``token`` is
      None. The credential is resolved per call from the session's MindsHub
      connection (the turn key in a cloud pod), because a turn key can be
      rotated mid-session. This is the only form ``from_dict`` reads, so it is
      the only one a turn request can carry.
    - The Cowork desktop app's own browser: ``base_url`` is exactly
      ``http://127.0.0.1:<port>`` and ``token`` is the key the app made for
      this launch. Only the host builds it, in its own process, with this
      constructor; it never comes from a dict, a setting or the environment.

    ``profile`` is the browser session this conversation drives. ``project``
    keys the tabs saved for the conversation's project; empty when there is
    none, and saved tabs are then unavailable.

    ``decisions`` is the host's own decision model for the browser's steps
    (``DecisionsTarget``), for either form. ``decisions_proxy`` is the same
    model reached through the host with the session's MindsHub credential
    (``DecisionsProxy``), for the router form: the one a turn request
    carries, since it holds no key. With neither, Jev on the session's
    MindsHub connection picks them, where there is one.

    Validation errors never echo the input, so a refused config can be
    logged without its key. ``repr`` masks the key too.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    base_url: str
    profile: str
    project: str = ""
    token: SecretStr | None = None
    decisions: DecisionsTarget | None = None
    decisions_proxy: DecisionsProxy | None = None

    @field_validator("base_url", mode="before")
    @classmethod
    def _without_trailing_slash(cls, value: object) -> object:
        return value.rstrip("/") if isinstance(value, str) else value

    @field_validator("profile")
    @classmethod
    def _profile_is_a_key(cls, value: str) -> str:
        if not _KEY.fullmatch(value):
            raise ValueError("profile must be 1-64 letters, digits, '-' or '_'")
        return value

    @field_validator("project")
    @classmethod
    def _project_is_a_key(cls, value: str) -> str:
        if value and not _KEY.fullmatch(value):
            raise ValueError("project must be empty or 1-64 letters, digits, '-' or '_'")
        return value

    @model_validator(mode="after")
    def _one_of_two_forms(self) -> BrowserConfig:
        url = urlsplit(self.base_url)
        if url.scheme == "https":
            if not url.hostname or "@" in url.netloc or "?" in self.base_url or "#" in self.base_url:
                raise ValueError("the router's base_url must be https://<host>[/path], with no user, query or fragment")
            if self.token is not None:
                raise ValueError("the router takes the session's MindsHub credential, never a token")
            return self
        if url.scheme == "http" and url.hostname == LOOPBACK_HOST:
            try:
                port = url.port
            except ValueError:
                port = None
            if not port or self.base_url != f"http://{LOOPBACK_HOST}:{port}":
                raise ValueError(f"the in-app browser's base_url must be exactly http://{LOOPBACK_HOST}:<port>")
            if self.token is None or not _TOKEN.fullmatch(self.token.get_secret_value()):
                raise ValueError("the in-app browser needs its launch key as the token")
            if self.decisions_proxy is not None:
                raise ValueError("the in-app browser's host hands it its decision model directly")
            return self
        raise ValueError(f"base_url must be the https router or http://{LOOPBACK_HOST}:<port>")

    @property
    def in_app(self) -> bool:
        """The Cowork desktop app's own browser, reached with its launch key."""
        return self.token is not None

    @classmethod
    def from_dict(cls, raw: object) -> BrowserConfig | None:
        """Read a turn request's router block, ``{"base_url", "profile", "project"?,
        "decisions_proxy"?}``.

        None when the block is absent or unusable: not a dict, a ``token``
        key of any value, a base_url that isn't an https router, or a
        missing or malformed ``profile``. A malformed ``project`` reads as
        none, and so does a malformed ``decisions_proxy``
        (``DecisionsProxy.from_dict``); a ``decisions`` key is never read. So
        nothing a request carries can point the tool at a browser that takes
        its own key, or hand it a decision model's key.
        """
        if not isinstance(raw, dict) or "token" in raw:
            return None
        base_url, profile, project = raw.get("base_url"), raw.get("profile"), raw.get("project")
        if not isinstance(base_url, str) or not isinstance(profile, str):
            return None
        if not isinstance(project, str) or not _KEY.fullmatch(project):
            project = ""
        try:
            if urlsplit(base_url).scheme != "https":
                return None
            return cls(
                base_url=base_url, profile=profile, project=project,
                decisions_proxy=DecisionsProxy.from_dict(raw.get("decisions_proxy")),
            )
        except ValueError:
            return None

    def to_dict(self) -> dict:
        """The router block a host puts in a turn request: ``base_url`` and
        ``profile``, plus ``project`` and ``decisions_proxy`` (``{"url",
        "model"}``, no key) when there are. Never a ``token`` or
        ``decisions`` key, so ``from_dict`` reads it back as this same config.

        The in-app browser's config has no dict form: its launch key stays in
        the host's process. Nor has one with a decision model of the host's
        own (``decisions``), for its key: a host hands a turn elsewhere the
        same model as a ``decisions_proxy``. Raises ValueError for either.
        """
        if self.in_app:
            raise ValueError("the in-app browser's config stays in the host's process")
        if self.decisions is not None:
            raise ValueError("a config with the host's own decision model stays in the host's process")
        block = {"base_url": self.base_url, "profile": self.profile}
        if self.project:
            block["project"] = self.project
        if self.decisions_proxy is not None:
            block["decisions_proxy"] = {"url": self.decisions_proxy.url, "model": self.decisions_proxy.model}
        return block
