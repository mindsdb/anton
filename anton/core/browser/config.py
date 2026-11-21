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

    Validation errors never echo the input, so a refused config can be
    logged without its key. ``repr`` masks the key too.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    base_url: str
    profile: str
    project: str = ""
    token: SecretStr | None = None

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
            return self
        raise ValueError(f"base_url must be the https router or http://{LOOPBACK_HOST}:<port>")

    @property
    def in_app(self) -> bool:
        """The Cowork desktop app's own browser, reached with its launch key."""
        return self.token is not None

    @classmethod
    def from_dict(cls, raw: object) -> BrowserConfig | None:
        """Read a turn request's router block, ``{"base_url", "profile", "project"?}``.

        None when the block is absent or unusable: not a dict, a ``token``
        key of any value, a base_url that isn't an https router, or a
        missing or malformed ``profile``. A malformed ``project`` reads as
        none. So nothing a request carries can point the tool at a browser
        that takes its own key.
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
            return cls(base_url=base_url, profile=profile, project=project)
        except ValueError:
            return None

    def to_dict(self) -> dict[str, str]:
        """The router block a host puts in a turn request: ``base_url`` and
        ``profile``, plus ``project`` when there is one. Never a ``token``
        key, so ``from_dict`` reads it back as this same config.

        The in-app browser's config has no dict form: its launch key stays in
        the host's process. Raises ValueError for it.
        """
        if self.in_app:
            raise ValueError("the in-app browser's config stays in the host's process")
        block = {"base_url": self.base_url, "profile": self.profile}
        if self.project:
            block["project"] = self.project
        return block
