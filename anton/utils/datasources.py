from __future__ import annotations

import base64
import bisect
import functools
import html
import json
import logging
import os
import re
import shutil
import urllib.parse
import uuid
import yaml
from collections.abc import Collection, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from anton.core.datasources.data_vault import (
    DataVault,
    LocalDataVault,
    _slug_env_prefix,
    _sweep_ds_env_vars,
    is_secret_key,
)
from anton.core.datasources.datasource_registry import DatasourceRegistry, _YAML_BLOCK_RE

if TYPE_CHECKING:
    from anton.core.datasources.datasource_registry import DatasourceEngine, DatasourceField

logger = logging.getLogger(__name__)


class _ContextScopedSet:
    """A set[str] scoped to the turn that opened it.

    The ContextVar holds a MUTABLE set, opened once per turn and mutated in
    place after that. Mutating in place is what makes a tool handler's
    registration visible to the turn that dispatched it: every tool call runs
    in its own task, and `asyncio.create_task` copies the context, so a child
    task's `ContextVar.set()` is discarded when it finishes — but a change to
    the set the parent already holds is not.
    """

    def __init__(self, name: str) -> None:
        self._cv: ContextVar[set[str] | None] = ContextVar(name, default=None)

    def add(self, item: str) -> None:
        current = self._cv.get()
        if current is None:
            current = set()
            self._cv.set(current)
        current.add(item)

    def discard(self, item: str) -> None:
        current = self._cv.get()
        if current is not None:
            current.discard(item)

    def clear(self) -> None:
        # Open a scope only when this context has none; otherwise empty the one
        # already open, so a mid-turn rebuild stays visible to its own turn.
        current = self._cv.get()
        if current is None:
            self._cv.set(set())
        else:
            current.clear()

    def begin_scope(self) -> None:
        """Open this turn's own container, carrying over what is visible now.

        Opened in the turn's context, so a tool task's in-place write reaches a
        container the turn still holds instead of a copy that is discarded.
        """
        self._cv.set(set(self._cv.get() or ()))

    def replace(self, items: set[str]) -> None:
        """Swap the contents in place, keeping the container this turn holds."""
        self.clear()
        current = self._cv.get()
        if current is not None:
            current.update(items)

    def reset(self) -> None:
        """Drop the scope entirely. For teardown, not for a mid-turn rebuild."""
        self._cv.set(None)

    def __contains__(self, item: object) -> bool:
        return item in (self._cv.get() or ())

    def __iter__(self):
        # A snapshot: scrub_credentials iterates while a concurrent tool task
        # may still be registering into the same set.
        return iter(tuple(self._cv.get() or ()))

    def __len__(self) -> int:
        return len(self._cv.get() or ())


class _ContextScopedEnvValues:
    """Per-turn DS_* name -> value map, used to scrub credential values.

    A turn that has called set() is authoritative: a key missing from its map
    has no known value for this turn, and os.environ is NOT consulted, because
    another turn may hold a different value under the same name and redacting
    with it would both miss this turn's secret and disclose the other's.
    Callers that never call set() (the CLI, tests writing os.environ directly)
    keep reading os.environ exactly as before.
    """

    def __init__(self, name: str) -> None:
        # None means "this context never opened a map", which separates a turn
        # with no connections (an empty map) from a plain os.environ user.
        self._cv: ContextVar[dict[str, str] | None] = ContextVar(name, default=None)

    def set(self, values: dict[str, str]) -> None:
        # In place when a map is already open, for the same reason
        # _ContextScopedSet mutates in place: a tool call runs in its own task.
        current = self._cv.get()
        if current is None:
            self._cv.set(dict(values))
        else:
            current.clear()
            current.update(values)

    def begin_scope(self) -> None:
        """Open this turn's own map, seeded from the ambient DS_*.

        Re-seeded every turn rather than only the first: without a vault
        nothing else refreshes the map, so a credential the host rotated
        between turns would be scrubbed against the previous value. Called
        once per turn, so no mid-turn registration is lost.
        """
        self._cv.set({k: v for k, v in os.environ.items() if k.startswith("DS_")})

    def snapshot(self) -> dict[str, str] | None:
        """A copy of the open map, or None when no scope is open."""
        current = self._cv.get()
        return dict(current) if current is not None else None

    def reset(self) -> None:
        """Back to "never opened", so get() reads os.environ again."""
        self._cv.set(None)

    def get(self, key: str, default: str = "") -> str:
        overrides = self._cv.get()
        if overrides is None:
            return os.environ.get(key, default)
        return overrides.get(key, default)

    def items(self):
        # os.environ stays in the union on purpose, for the coarse unknown-DS_*
        # net. Over-redacting is harmless; emitting a value is what get() stops.
        merged = {k: v for k, v in os.environ.items() if k.startswith("DS_")}
        merged.update(self._cv.get() or {})
        return merged.items()


# DS_* var names whose values are known to be secret (passwords, tokens, keys).
# Populated at startup and after each successful connect.
_DS_SECRET_VARS = _ContextScopedSet("ds_secret_vars")

# DS_* var names for **ALL** fields of registered engines.
_DS_KNOWN_VARS = _ContextScopedSet("ds_known_vars")

# DS_* name -> value overrides for the current turn (see _ContextScopedEnvValues).
_ds_env_values = _ContextScopedEnvValues("ds_env_values")


def begin_ds_turn_scope() -> None:
    """Open the per-turn DS_* scope. Call once at the start of a turn.

    Without it a host that never opened one loses any registration a tool
    handler makes, because tool calls are dispatched in their own task.
    """
    _DS_SECRET_VARS.begin_scope()
    _DS_KNOWN_VARS.begin_scope()
    _ds_env_values.begin_scope()
    _held_exact_secrets.set({})


def set_ds_env_values(values: dict[str, str]) -> None:
    """Record this turn's DS_* name -> value map, used to scrub credentials.

    Scoped to the calling context, so a concurrently running turn's map is
    untouched.
    """
    _ds_env_values.set(values)
    _hold_exact_secrets()


# Provider credential env vars injected into the scratchpad execution env for
# the code to USE. Their values must never reach the LLM (ENG-463): a model can
# only emit a secret it can see, and the agent was caught writing a raw `mdb_`
# key into generated code. Scrubbed by exact value (labeled) from anything that
# re-enters model context.
_PROVIDER_SECRET_VARS: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "ANTON_OPENAI_API_KEY",
    "ANTON_ANTHROPIC_API_KEY",
    "ANTON_GEMINI_API_KEY",
    "ANTON_MINDS_API_KEY",
)

# Well-known provider key formats — scrubbed by shape so a key is redacted even
# when it didn't come from one of our env vars (e.g. the model already emitted
# it, or it arrived via an unexpected path). Length floors keep this from
# matching short incidental `sk-`/`mdb_` strings. A letter or digit right before
# the prefix means a longer word, such as `task-completion-rate` or
# `risk-assessment-panel`, not a key.
_SECRET_KEY_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"mdb_[A-Za-z0-9._-]{10,}"   # MindsHub
    r"|sk-[A-Za-z0-9_-]{20,}"    # OpenAI / Anthropic (sk-, sk-proj-, sk-ant-)
    r"|AIza[A-Za-z0-9_-]{30,}"   # Google standard key
    r"|AQ\.[A-Za-z0-9._-]{50}"   # Google authorization key (AI Studio, 2026+)
    r")"
)

# Output only: user input and published artifacts keep _SECRET_KEY_PATTERN,
# since these shapes also match pasted connect credentials and token code.
_TOKEN_SHAPE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]*"  # JWT
    r"|ya29\.[A-Za-z0-9_-]{20,}"  # Google OAuth access token
)
_TOKEN_KEY = (
    r"(?<![A-Za-z0-9])"
    r"(?i:access[-_]?token|refresh[-_]?token|id[-_]?token|client[-_]?secret)"
)
# A marker this module writes in place of a secret: `[DS_...]` and provider labels
# from `scrub_credentials`, `[REDACTED_API_KEY]` and `[REDACTED_TOKEN]`. Every one
# holds an underscore, so a bracketed run without one, such as the `[Q]` in a
# password `x7[Q]abc`, is not a marker. `SCRUBBED_VALUE_RE` in `anton/tools.py`
# is built from it.
_SCRUB_MARKER = r"\[(?:DS_\w+|[A-Z][A-Z0-9]*_[A-Z0-9_]*)\]"
# A quoted value with no quote, backslash or space, unless it is a whole marker; for
# Bearer values and code-like token values. Clipped output may lack the closing quote.
_NARROW_QUOTED_VALUE = (
    r"(?!(?:\$?\{[^\"'\\\s{}]*\}|" + _SCRUB_MARKER + r")(?![^\"'\\\s]))"
    r"(?:[^\"'\\\s]|\\/|\\u[0-9a-fA-F]{4}){16,}"
)
# A Bearer value cannot hold a space or a quote, so prose after `"Bearer ` stays.
_QUOTED_BEARER_VALUE_PATTERN = re.compile(r"(\\?[\"'](?i:bearer)[ \t]+)" + _NARROW_QUOTED_VALUE)
_NARROW_QUOTED_VALUE_PATTERN = re.compile(_NARROW_QUOTED_VALUE)
# Group 1 is the backslashes before the value's quote, which its closing quote
# repeats; 31 covers five levels of JSON in JSON.
_QUOTED_TOKEN_OPENER_PATTERN = re.compile(_TOKEN_KEY + r"\\*[\"']?\s*[:=]\s*(\\{0,31})([\"'])")
# Scanned length of one value; longer than any token or secret a cell prints.
_QUOTED_VALUE_SCAN_MAX = 16_384
# A template expression (Helm, Jinja, GitHub Actions) is a whole marker too.
_WHOLE_MARKER_PATTERN = re.compile(
    r"\$?\{[^\"'\\\s{}]*\}|" + _SCRUB_MARKER + r"|\$?\{\{[^{}\r\n]*\}\}"
)
# PHP, Lua and SQL join strings with ` . `, ` .. ` and ` || `.
_CODE_OPERATOR_AFTER = re.compile(r"\s*(?:[+,()%]|(?:if|else|or|and)\b)|\s+(?:\.\.?|\|\|)\s")
_CODE_OPERATOR_BEFORE = re.compile(r"(?:[+,()%]|\b(?:if|else|or|and))\s*$|\s(?:\.\.?|\|\|)\s*$")
_CODE_OPERATOR_BEFORE_WINDOW = 64
_CODE_LEADING_OPERATOR = re.compile(r"\s*(?:[+,)]|\.\.?|\|\||[=!]=|(?:not\s+)?in|is)(?:\s|$)")
_QUOTE_OR_LINE_BREAK_PATTERN = re.compile(r"(?<!\\)(\\*)([\"'])|[\r\n]")
_LINE_BREAK_PATTERN = re.compile(r"[\r\n]")
_WORD_BEFORE_PATTERN = re.compile(r"[A-Za-z0-9]+$")
# Lowercase Python string prefixes; `b'`, `f'` and `rb'` open a literal, not prose.
_STRING_PREFIXES = frozenset({"b", "r", "u", "f", "rb", "br", "fr", "rf"})
# An f-string is formatted before it prints, so one after a key is source code.
_FORMAT_STRING_PREFIXES = frozenset(prefix for prefix in _STRING_PREFIXES if "f" in prefix)
# An unquoted Bearer value, on its line or the next; a lowercase word after
# `Bearer` is prose.
_BEARER_VALUE_PATTERN = re.compile(
    r"((?i:\bbearer)(?:[ \t]+|[ \t]*\r?\n[ \t]*)"
    r"(?![a-z_/-]+(?![A-Za-z0-9._~+/=%!-])))"
    r"(?!!)[A-Za-z0-9._~+/=%!-]{16,}"
    r"(?![A-Za-z0-9._~+/=%!(-])"
)
# The characters that open a marker, reference, placeholder or mask. A value
# starting with one stays readable only as `_unquoted_value_is_placeholder` allows.
_PLACEHOLDER_OPENERS = "$[{<*"
# Group 1 is the `:` or `=` before the value; spaces alone separate a printed
# table column. The value starts like a token or with a placeholder opener; a
# quoted value is left to the quoted pass.
_UNQUOTED_TOKEN_OPENER_PATTERN = re.compile(
    _TOKEN_KEY + r"\\?[\"']?(?:[ \t]*([:=])[ \t]*|[ \t]+)"
    r"(?=[A-Za-z0-9._~+/%" + re.escape(_PLACEHOLDER_OPENERS) + r"-])"
)
# An unquoted value ends at a line break or at `&`, `,` or `;`, which separate
# query parameters and list items. In a column or a URL a space ends it too;
# elsewhere it runs on through spaces up to the next `key=value` pair. Base64
# padding at a value's end is no pair. A run of spaces is tried from its first
# space only, so a long run costs one pass; a value never starts on a space.
_WORD_VALUE_END_PATTERN = re.compile(r"[\s&,;]")
_SPACED_VALUE_END_PATTERN = re.compile(r"[\r\n&,;]|(?<![ \t])[ \t]+(?=[A-Za-z_][\w.-]*=[^\s=])")
# A PHC password hash (`$argon2id$v=19$m=65536,t=3,p=4$<salt>$<hash>`) separates
# its parameters with commas, so they do not end the value.
_HASH_PARAMETERS_PATTERN = re.compile(
    r"\$[a-z0-9-]+(?:\$v=[0-9]+)?\$[a-z0-9-]+=[A-Za-z0-9/+.-]*"
    r"(?:,[a-z0-9-]+=[A-Za-z0-9/+.-]*)+(?=\$)"
)
_IDENTIFIER_PATH = r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"
# A value start that cannot be a secret: a scrub marker; a shell, env or JS
# template reference (`$NAME`, `${NAME}`, `${session.token}`, `$(command)`, or
# `${NAME:-default}` with a default under the 16-character floor); a template
# placeholder of a name (`{{ access_token }}`, `${{ secrets.TOKEN }}`,
# `{{ .Values.token | b64enc }}`, `{token}`, `{access_token!r}`); an angle-bracket
# placeholder (`<your-access-token>`, `<your access token>`); or asterisks with at
# most four characters after them (`****abcd`). A `$` before a digit or a `$name$`
# chain is a password hash, which none of these matches whole.
_PLACEHOLDER_PATTERN = re.compile(
    _SCRUB_MARKER
    + r"|\$(?:[A-Za-z_]\w*|\([^()\r\n]*\)"
    + r"|\{" + _IDENTIFIER_PATH + r"(?::?[-=?+][^\s\"'\\{}]{0,15})?\})"
    + r"|\$?\{\{[ \t]*\.?" + _IDENTIFIER_PATH + r"(?:[ \t]*\|[ \t]*[A-Za-z_]\w*)*[ \t]*\}\}"
    + r"|\{" + _IDENTIFIER_PATH + r"(?:![rsa])?\}"
    + r"|<[A-Za-z_][\w-]*(?: [\w-]+)*>"
    + r"|\*+[^\s*]{0,4}"
)
# What ends a placeholder in a string literal: the quote that closes the string
# or an escaped line break, after any spaces, as in `f"access_token: {token}\n"`.
_LITERAL_END_PATTERN = re.compile(r"[ \t]*(?:[\"'`]|\\[rn])")
# Quotes and closing brackets at the end of a value close what holds it.
_VALUE_TRAILER = " \t\"'`\\)]}>"
_TOKEN_CHARS_PATTERN = re.compile(r"[A-Za-z0-9._~+/=%!-]*")
# Unquoted values can be source code the model reads and may write back: a
# lowercase-led dotted path, all-lowercase words, a PascalCase type or an
# UPPER_SNAKE constant is code or prose. PascalCase words need three letters and
# UPPER_SNAKE parts are letters with at most one trailing digit.
_CODE_VALUE_PATTERN = re.compile(
    r"[a-z_/-]+(?:[ \t]+[a-z_/-]+)*|[a-z_]\w*(?:\.[A-Za-z_]\w*)+"
    r"|(?:[A-Z][a-z]{2,}){2,}|[A-Z]{2,}[0-9]?(?:_[A-Z]{2,}[0-9]?)+"
)
# What follows a value's first word in code rather than in a secret: an index,
# a closing bracket or a `::` path, or after a space an operator, a comment, a
# call or a keyword.
_CODE_AFTER_WORD_PATTERN = re.compile(
    r"[\[)\]}]|::"
    r"|[ \t]+(?:#|//|\(|[A-Za-z_][\w.]*\(|[-+*/%=!<>|?:.]+(?![^ \t])|(?:if|else|or|and|not|in|is)\b)"
)


def _reset_registered_ds_vars() -> None:
    """Drop all DS_* scope, so nothing survives into a turn with no vault."""
    _DS_SECRET_VARS.reset()
    _DS_KNOWN_VARS.reset()
    _ds_env_values.reset()
    _held_exact_secrets.set(None)


def parse_connection_slug(
    slug: str,
    known_engines: list[str],
    *,
    vault: DataVault | None = None,
) -> tuple[str, str] | None:
    """Split a connection slug into (engine, name) using longest-prefix matching.

    First tries each known registry engine longest-first so that 'sql-server-prod-db' is
    correctly parsed as engine='sql-server', name='prod-db' rather than
    engine='sql', name='server-prod-db'.

    If nothing matches and a vault is supplied, falls back to scanning vault
    connections for an exact slug match — handles custom/unregistered engines.

    Returns None if no match found or name part is empty.
    """
    for engine in sorted(known_engines, key=len, reverse=True):
        prefix = engine + "-"
        if slug.startswith(prefix) and len(slug) > len(prefix):
            return (engine, slug[len(prefix):])

    if vault is not None:
        for conn in vault.list_connections():
            if f"{conn['engine']}-{conn['name']}" == slug:
                return (conn["engine"], conn["name"])

    return None


def register_secret_vars(
    engine_def: "DatasourceEngine", *, engine: str = "", name: str = ""
) -> None:
    """Record which DS_* var names correspond to known/secret fields for engine_def.

    If engine and name are given, registers namespaced vars (DS_ENGINE_NAME__FIELD).
    Otherwise registers flat vars (DS_FIELD) — for temporary test_snippet execution.
    """
    all_fields = list(engine_def.fields)
    for am in engine_def.auth_methods or []:
        all_fields.extend(am.fields)
    for f in all_fields:
        if engine and name:
            prefix = _slug_env_prefix(engine, name)
            key = f"{prefix}__{f.name.upper()}"
        else:
            key = f"DS_{f.name.upper()}"
        _DS_KNOWN_VARS.add(key)
        if f.secret:
            _DS_SECRET_VARS.add(key)


def _register_unregistered_connection_vars(vault: DataVault, engine: str, name: str) -> None:
    """Register DS_* var names for a connection whose engine isn't in the
    registry (custom engines, connector-spec saves).

    Without this, every field of such a connection falls into the
    conservative unknown-DS_* scrub — so harmless values like base_url
    surfaced as `[DS_..._BASE_URL]` markers in user-facing output (ENG-688).
    Classification: the record's stored ``secure_keys`` when present, else
    the vault's canonical legacy secret-name heuristic. `_`-prefixed
    bookkeeping fields are skipped entirely — they're never injected as env
    vars (see `env_for()`), so registering them here would be dead weight.
    """
    record = vault.read_record(engine, name) if hasattr(vault, "read_record") else None
    fields = (record or {}).get("fields") or vault.load(engine, name) or {}
    secure_keys = (record or {}).get("secure_keys")
    prefix = _slug_env_prefix(engine, name)
    for field_name in fields:
        if field_name.startswith("_"):
            continue
        key = f"{prefix}__{field_name.upper()}"
        _DS_KNOWN_VARS.add(key)
        if is_secret_key(field_name, secure_keys):
            _DS_SECRET_VARS.add(key)


# A saved value this long or shorter is hidden only as a whole word and in its
# own case, so a short password does not hit ordinary text; a longer one is
# hidden anywhere, ASCII case folded. Unknown DS_* and provider values this
# short are not hidden at all.
_SHORT_SECRET_MAX = 8
# A value of at least this many UTF-8 bytes is hidden in base64 too; its
# shortest base64 form is then 6 characters.
_BASE64_MIN_BYTES = 5
# A literal run shorter than this is in most text, so checking for it first
# would only add work; the pattern runs instead.
_ANCHOR_MIN_LENGTH = 4
# Runs of characters every supported encoder prints as they are.
_PLAIN_RUN_PATTERN = re.compile(r"[A-Za-z0-9._-]+")
# A run of backslashes in a value doubles at each level of escaping, so it
# matches a run of any length. It takes every backslash in a row, so the
# escape of the character after it (JSON's `\u00e9` after an escaped
# backslash) may start with none left.
_VALUE_UNIT_PATTERN = re.compile(r"\\+|.", re.DOTALL)
_BACKSLASH_RUN = r"(?:\\++|(?:%5[Cc])++)"
_ESCAPE = r"\\++"
_ESCAPE_AFTER_BACKSLASH_RUN = r"\\*+"
_CONTROL_ESCAPES = {"\n": "n", "\r": "r", "\t": "t", "\b": "b", "\f": "f"}
_HTML_ENTITY_NAMES = {"&": "amp", "<": "lt", ">": "gt", '"': "quot", "'": "apos"}
# The two characters that differ between the standard and URL-safe alphabets.
_BASE64_ALPHABET_CHARS = {"+": r"(?:%2[Bb]|[+-])", "/": r"(?:%2[Ff]|\\*+/|_)"}
_BASE64_ANCHOR_PATTERN = re.compile(r"[A-Za-z0-9]+")
_BASE64_CHAR = r"[A-Za-z0-9+/_-]"
_BASE64_PADDING = r"(?:=|%3[Dd]){0,2}"
_TO_URLSAFE_BASE64 = str.maketrans("+/", "-_")
_SCRUB_MARKER_PATTERN = re.compile(_SCRUB_MARKER)


@dataclass(frozen=True)
class _ExactSecret:
    """A value `scrub_credentials` hides wherever it appears, and the label shown in its place."""

    label: str
    value: str


# Every value this turn has hidden by value, keyed by value. A pad keeps the DS_*
# values it started with until the turn ends, so a rebuild that no longer lists one
# (its connection removed or edited mid-turn) must not stop hiding it. Opened by
# begin_ds_turn_scope and mutated in place, like the registries above.
_held_exact_secrets: ContextVar[dict[str, _ExactSecret] | None] = ContextVar(
    "held_exact_secrets", default=None
)


def _hold_exact_secrets() -> None:
    """Keep hiding, until this turn ends, every value the turn hides by value now."""
    held = _held_exact_secrets.get()
    if held is not None:
        for secret in _exact_secrets():
            held.setdefault(secret.value, secret)


@dataclass(frozen=True)
class _SecretForms:
    """The matcher for one saved value in every form `scrub_credentials` hides.

    Every match of `pattern` contains one of `anchors`, lowercase when
    `folded`; `anchors` is empty when one is shorter than `_ANCHOR_MIN_LENGTH`.
    `literal_forms` are the concrete encoded strings, for the cut-edge helpers.
    """

    pattern: re.Pattern[str]
    anchors: tuple[str, ...]
    folded: bool
    literal_forms: tuple[str, ...]


class _MarkerSpans:
    """Where the scrub markers in one text start and end.

    A saved value can turn up inside a marker's label in another case, such
    as `custom_prod` in `[DS_CUSTOM_PROD__PASSWORD]`; such a match is part of
    the marker, not a secret. A match that holds a whole marker-shaped run,
    such as a password `p[A_B]ss`, is still the secret.
    """

    def __init__(self, text: str) -> None:
        """Find every marker in `text`; markers never overlap one another."""
        spans = [found.span() for found in _SCRUB_MARKER_PATTERN.finditer(text)]
        self.starts = [start for start, _ in spans]
        self.ends = [end for _, end in spans]

    def cuts(self, start: int, end: int) -> bool:
        """Return whether `text[start:end]` starts or ends strictly inside a marker, short of its whole label.

        A match that is a marker's whole label is a secret printed in brackets,
        such as `[PROD_DB_PASS_2024]`: a label is a variable name, which a
        saved value almost never equals.
        """
        for position in (start, end):
            index = bisect.bisect_left(self.starts, position) - 1
            if index >= 0 and position < self.ends[index]:
                return (start, end) != (self.starts[index] + 1, self.ends[index] - 1)
        return False


def _redact_outside_markers(
    pattern: re.Pattern[str], replacement: str, text: str, markers: _MarkerSpans
) -> tuple[str, int]:
    """Replace each match of `pattern` in `text` with `replacement`, except one that cuts into a marker.

    `markers` are the markers in `text`, from an earlier scrub or from this
    one. Returns the new text and the number of matches, replaced or not.
    """
    return pattern.subn(
        lambda found: found.group() if markers.cuts(*found.span()) else replacement, text
    )


def _hex_pattern(number: int, width: int) -> str:
    """Return a pattern for `number` written in at least `width` hex digits of either case."""
    return "".join(
        f"[{digit}{digit.lower()}]" if digit.isalpha() else digit for digit in f"{number:0{width}X}"
    )


def _escaped_char_patterns(char: str, escape: str) -> list[str]:
    """Return patterns for `char` written as a backslash escape, at any depth of escaping.

    JSON writes `\\n` or `\\uXXXX` (two for a character past U+FFFF); Python's
    `ascii()` writes `\\xXX`, `\\uXXXX` or `\\UXXXXXXXX`; a bytes repr writes
    each UTF-8 byte as `\\xXX`. `escape` is the pattern for the backslashes.
    """
    code = ord(char)
    patterns = []
    if code < 0x20 or code > 0x7E or char in _HTML_ENTITY_NAMES:
        utf16 = char.encode("utf-16-be")
        patterns.append("".join(
            escape + "u" + _hex_pattern(int.from_bytes(utf16[i : i + 2], "big"), 4)
            for i in range(0, len(utf16), 2)
        ))
    if code > 0xFFFF:
        patterns.append(escape + "U" + _hex_pattern(code, 8))
    if code > 0x7F:
        patterns.append("".join(escape + "x" + _hex_pattern(byte, 2) for byte in char.encode()))
    if code < 0x20 or 0x7F <= code <= 0xFF:
        patterns.append(escape + "x" + _hex_pattern(code, 2))
    if char in _CONTROL_ESCAPES:
        patterns.append(escape + _CONTROL_ESCAPES[char])
    return patterns


def _char_pattern(char: str, escape: str) -> str:
    """Return a pattern for one value character as printed or as a supported encoder writes it.

    A plain character stands for itself. Any other also matches a backslash
    escape, an HTML entity, its UTF-8 `%XX` bytes, `+` for a space, and
    itself after any backslashes. Longer forms come first, so `&amp;` is
    taken whole rather than as `&`.
    """
    if _PLAIN_RUN_PATTERN.fullmatch(char):
        return re.escape(char)
    alternatives = _escaped_char_patterns(char, escape)
    if char in _HTML_ENTITY_NAMES:
        alternatives.append(f"&{_HTML_ENTITY_NAMES[char]};")
    if char in _HTML_ENTITY_NAMES or ord(char) > 0x7E:
        alternatives.append(f"&#0*{ord(char)};|&#[xX]0*{_hex_pattern(ord(char), 1)};")
    alternatives.append("".join("%" + _hex_pattern(byte, 2) for byte in char.encode()))
    if char == " ":
        alternatives.append(r"\+")
    alternatives.append(r"\\*+" + re.escape(char))
    return "(?:" + "|".join(alternatives) + ")"


def _value_pattern(value: str) -> str:
    """Return a pattern for `value` as printed, URL-encoded, or JSON-, HTML- or backslash-escaped."""
    parts = []
    escape = _ESCAPE
    for unit in _VALUE_UNIT_PATTERN.findall(value):
        if unit[0] == "\\":
            parts.append(_BACKSLASH_RUN)
            escape = _ESCAPE_AFTER_BACKSLASH_RUN
        else:
            parts.append(_char_pattern(unit, escape))
            escape = _ESCAPE
    return "".join(parts)


def _base64_cores(raw: bytes) -> list[str]:
    """Return, for each byte offset modulo 3, the base64 characters that encode only `raw`.

    A value inside a longer base64 string, such as the password in an HTTP
    Basic header, starts at an offset set by what comes before it.
    """
    cores = []
    for offset in range(3):
        encoded = base64.b64encode(bytes(offset) + raw).decode()
        cores.append(encoded[(4 * offset + 2) // 3 : 8 * (offset + len(raw)) // 6])
    return cores


def _base64_pattern(core: str, offset: int) -> str:
    """Return a pattern for `core` in either alphabet, padded or not, percent-encoded or not.

    It takes the character at each end that mixes the value's bits with a
    neighbor's too, so none of the value's bits stay readable.
    """
    lead = _BASE64_CHAR + "?" if offset else ""
    body = "".join(_BASE64_ALPHABET_CHARS.get(char, char) for char in core)
    return lead + body + _BASE64_CHAR + "?" + _BASE64_PADDING


@functools.lru_cache(maxsize=1024)
def _secret_forms(value: str) -> _SecretForms:
    """Build the matcher for `value` in every form `scrub_credentials` hides.

    The forms are the value as printed, URL-encoded (`quote`, `quote_plus`, hex
    of either case), JSON-, HTML- and backslash-escaped at any depth, and,
    from `_BASE64_MIN_BYTES`, base64 at each byte offset in either alphabet.
    Cached per value, since the same values come back on every call and every turn.
    """
    folded = len(value) > _SHORT_SECRET_MAX
    body = _value_pattern(value)
    if not folded:
        # Only an edge that is a word character needs a non-word neighbor.
        if re.match(r"\w", value[0]):
            body = r"(?<!\w)" + body
        if re.match(r"\w", value[-1]):
            body += r"(?!\w)"
    branches = [body]
    anchors = [max(_PLAIN_RUN_PATTERN.findall(value), key=len, default="")]
    literal_forms = [
        value,
        urllib.parse.quote(value, safe=""),
        urllib.parse.quote_plus(value),
        json.dumps(value)[1:-1],
        html.escape(value),
    ]
    raw = value.encode()
    if len(raw) >= _BASE64_MIN_BYTES:
        for offset, core in enumerate(_base64_cores(raw)):
            branches.append(_base64_pattern(core, offset))
            anchors.append(max(_BASE64_ANCHOR_PATTERN.findall(core), key=len, default=""))
            literal_forms += [core, core.translate(_TO_URLSAFE_BASE64)]
    if folded:
        anchors = [anchor.lower() for anchor in anchors]
    if min(len(anchor) for anchor in anchors) < _ANCHOR_MIN_LENGTH:
        anchors = []
    return _SecretForms(
        pattern=re.compile("|".join(branches), re.IGNORECASE | re.ASCII if folded else 0),
        anchors=tuple(anchors),
        folded=folded,
        literal_forms=tuple(dict.fromkeys(literal_forms)),
    )


def scrub_credentials(text: str) -> str:
    """Remove secret values from scratchpad/tool output before it reaches the LLM.

    Redacts, longest value first:
      * DS_* values registered as secret via register_secret_vars (driven by
        DatasourceField.secret=true). Non-secret fields of known engines
        (DS_HOST, DS_PORT, DS_BASE_URL, …) stay readable so the LLM can reason
        about connection errors.
      * Unknown DS_* vars (custom engines not yet in the registry) — any long
        value, conservatively.
      * Provider credentials (ENG-463): the values of _PROVIDER_SECRET_VARS.
    Each value is matched in every form `_secret_forms` builds: as printed,
    URL-encoded, escaped, or in base64, but never inside a marker already in
    the text. Then anything matching a well-known key shape
    (_SECRET_KEY_PATTERN) is redacted, so a raw `mdb_`/`sk-`/`AIza`/`AQ.` key
    can't reach model context via a tool result, traceback, or settings echo.
    """
    lowered: str | None = None
    markers: _MarkerSpans | None = None
    for secret in _exact_secrets():
        forms = _secret_forms(secret.value)
        if forms.anchors:
            if forms.folded and lowered is None:
                lowered = text.lower()
            haystack = lowered if forms.folded else text
            if not any(anchor in haystack for anchor in forms.anchors):
                continue
        if markers is None:
            markers = _MarkerSpans(text)
        text, count = _redact_outside_markers(forms.pattern, f"[{secret.label}]", text, markers)
        if count:
            lowered = markers = None
    text = _SECRET_KEY_PATTERN.sub("[REDACTED_API_KEY]", text)
    return text


def _exact_secrets() -> list[_ExactSecret]:
    """Return the values `scrub_credentials` hides by value this turn, longest first.

    Every registered secret, unknown DS_* values over `_SHORT_SECRET_MAX`
    characters, provider keys over it, and any value the turn held earlier and
    the current map no longer has. Longest first, so a value that holds a
    shorter one goes whole; values of one length keep that order, so a
    registered secret's label wins.
    """
    secrets = [_ExactSecret(key, _ds_env_values.get(key)) for key in _DS_SECRET_VARS]
    # Unknown DS_* vars are matched heuristically, so a short value such as
    # "on" or "true" in a DS_ENABLE_X var stays.
    secrets += [
        _ExactSecret(key, value) for key, value in _ds_env_values.items()
        if key not in _DS_KNOWN_VARS and len(value) > _SHORT_SECRET_MAX
    ]
    for key in _PROVIDER_SECRET_VARS:
        value = os.environ.get(key, "")
        if len(value) > _SHORT_SECRET_MAX:
            secrets.append(_ExactSecret(key, value))
    live = {secret.value for secret in secrets}
    secrets += [
        secret for value, secret in (_held_exact_secrets.get() or {}).items()
        if value not in live
    ]
    return sorted(
        (secret for secret in secrets if secret.value),
        key=lambda secret: len(secret.value),
        reverse=True,
    )


def _exact_secret_forms() -> list[str]:
    """Return the concrete forms, as printed and encoded, of every value `scrub_credentials` hides by value."""
    return [form for secret in _exact_secrets() for form in _secret_forms(secret.value).literal_forms]


def longest_exact_secret() -> int:
    """Return the length of the longest form of any value `scrub_credentials` hides by value this turn.

    A caller that cuts output before scrubbing it needs this much text past the
    cut for such a form to match whole. Counts only what the scrub matches, so
    a long non-secret field does not hold output back. Returns 0 when there is none.
    """
    return max((len(form) for form in _exact_secret_forms()), default=0)


def exact_secret_prefix_at_end(text: str) -> int:
    """Return the length of the longest end of `text` that starts, but does not complete, an exact-match secret.

    Output cut mid-write can end partway through a secret, or an encoded form
    of one, which the exact-value scrub cannot see. Call it on scrubbed text,
    so a whole secret at the end is already a placeholder. Returns 0 when the
    end starts no such form.
    """
    longest = 0
    for form in _exact_secret_forms():
        for length in range(min(len(form) - 1, len(text)), longest, -1):
            if text.endswith(form[:length]):
                longest = length
                break
    return longest


def exact_secret_suffix_at_start(text: str) -> int:
    """Return the length of the longest start of `text` that ends, but does not complete, an exact-match secret.

    The start-of-text counterpart of `exact_secret_prefix_at_end`, for output
    whose beginning was dropped. Call it on scrubbed text. Returns 0 when the
    start ends no such form.
    """
    longest = 0
    for form in _exact_secret_forms():
        for length in range(min(len(form) - 1, len(text)), longest, -1):
            if text.startswith(form[-length:]):
                longest = length
                break
    return longest


@functools.cache
def _value_end_pattern(quote: str, depth: int) -> re.Pattern[str]:
    """Return the pattern for what ends a value opened by `quote` after `depth` backslashes.

    That is the same quote after exactly `depth` backslashes, or a line break
    for a value cut off early.
    """
    return re.compile(r"(?<!\\)\\{%d}%s|[\r\n]" % (depth, quote))


class _OpenLiterals:
    """Track which string literal is open at each escape depth on the current line.

    A quote opens a literal at its depth, and only the same kind of quote
    closes it; the other kind inside is content. After a redacted value the
    rest of its line reads as data, since a value that ran past its real
    end leaves the state wrong. A line break resets both.
    """

    def __init__(self, text: str) -> None:
        """Start at the beginning of `text` with no literal open."""
        self.text = text
        self.position = 0
        self.open: dict[int, str] = {}
        self.trusted = True

    def advance(self, end: int) -> None:
        """Count every quote and line break from the current position up to `end`."""
        for found in _QUOTE_OR_LINE_BREAK_PATTERN.finditer(self.text, self.position, end):
            quote, depth = found.group(2), len(found.group(1) or "")
            if quote is None:
                self.open.clear()
                self.trusted = True
            elif depth in self.open:
                if self.open[depth] == quote:
                    del self.open[depth]
            elif not _is_apostrophe(self.text, found):
                self.open[depth] = quote
        self.position = max(self.position, end)

    def is_open(self, quote: str, depth: int) -> bool:
        """Return whether a `quote` literal at `depth` is open and the line can still be read as code."""
        return self.trusted and self.open.get(depth) == quote

    def skip_redacted(self, end: int) -> None:
        """Pass over a redacted value; the rest of its line is data."""
        self.trusted = False
        self.position = end


def _is_apostrophe(text: str, found: re.Match[str]) -> bool:
    """Return whether a matched `'` is prose, as in `it's`, `90's` or `users'`, and so opens no literal.

    A lowercase string prefix right before it (`b'`, `f'`, `rb'`) opens a
    literal. Otherwise a word right before it of three or more characters,
    or a letter right after it, marks prose.
    """
    if found.group(2) != "'" or found.group(1):
        return False
    word = _WORD_BEFORE_PATTERN.search(text, max(0, found.start() - 3), found.start())
    if word is None or word.group() in _STRING_PREFIXES:
        return False
    return len(word.group()) >= 3 or (found.end() < len(text) and text[found.end()].isalpha())


def _quoted_value_end(text: str, start: int, quote: str, depth: int) -> int:
    """Return where a quoted value starting at `start` ends.

    It ends at the first `quote` preceded by exactly `depth` backslashes, the
    escaping its opening quote had; a quote with more backslashes is part of
    the value. A line break or the end of `text` ends a value cut off early.
    A value with no end within `_QUOTED_VALUE_SCAN_MAX` runs to the end of
    its line, so nothing past the scan is shown.
    """
    limit = min(len(text), start + _QUOTED_VALUE_SCAN_MAX)
    found = _value_end_pattern(quote, depth).search(text, start, limit)
    if found:
        return found.start()
    line_break = _LINE_BREAK_PATTERN.search(text, limit)
    return line_break.start() if line_break else len(text)


def _looks_like_code(
    text: str, start: int, end: int, quote: str, depth: int, literals: _OpenLiterals
) -> bool:
    """Return whether `text[start:end]`, a quoted value, is really code between two string literals.

    Only a quote that closes a literal opened before the key is a code
    boundary: the opening quote itself, when an operator follows it, or a
    quote inside the value next to an operator. Inside a literal the key
    opened, every quote and operator is part of the value.
    """
    if literals.is_open(quote, depth) and _CODE_LEADING_OPERATOR.match(text, start, end):
        return True
    other = "'" if quote == '"' else '"'
    for boundary in _QUOTE_OR_LINE_BREAK_PATTERN.finditer(text, start, end):
        boundary_quote, boundary_depth = boundary.group(2), len(boundary.group(1))
        if not (
            (boundary_quote == other and boundary_depth == depth)
            or (boundary_quote == quote and boundary_depth < depth)
        ):
            continue
        if not literals.is_open(boundary_quote, boundary_depth):
            continue
        if _CODE_OPERATOR_AFTER.match(text, boundary.end(), end):
            return True
        window_start = max(start, boundary.start() - _CODE_OPERATOR_BEFORE_WINDOW)
        if _CODE_OPERATOR_BEFORE.search(text, window_start, boundary.start()):
            return True
    return False


def _redact_quoted_token_values(text: str) -> str:
    """Redact each quoted value after a token key, up to its real closing quote.

    The value may hold quotes, spaces and escapes at any nesting depth. A
    value that looks like code falls back to the narrow grammar, so source
    lines read as written. Returns `text` with each redacted value replaced.
    """
    parts: list[str] = []
    position = 0
    literals = _OpenLiterals(text)
    for opener in _QUOTED_TOKEN_OPENER_PATTERN.finditer(text):
        if opener.start() < position:
            continue
        literals.advance(opener.start(1))
        start = opener.end()
        depth, quote = len(opener.group(1)), opener.group(2)
        end = _quoted_value_end(text, start, quote, depth)
        scanned_whole = end - start < _QUOTED_VALUE_SCAN_MAX
        if scanned_whole and _looks_like_code(text, start, end, quote, depth, literals):
            narrow = _NARROW_QUOTED_VALUE_PATTERN.match(text, start)
            if narrow is None:
                continue
            end = narrow.end()
        elif end - start < 16 or _WHOLE_MARKER_PATTERN.fullmatch(text, start, end):
            continue
        parts.append(text[position:start])
        parts.append("[REDACTED_TOKEN]")
        literals.skip_redacted(end)
        position = end
    parts.append(text[position:])
    return "".join(parts)


def _unquoted_value_reads_as_code(text: str, start: int, end: int) -> bool:
    """Return whether `text[start:end]`, an unquoted value, reads as code or prose rather than a secret.

    It does when the whole value is code-shaped, when its first word is called
    or opens an f-string, or when a first word that is short or code-shaped is
    followed by code. A quote, `@`, `#` or a plain word after the first word
    continues a secret.
    """
    if _CODE_VALUE_PATTERN.fullmatch(text, start, end):
        return True
    word_end = _TOKEN_CHARS_PATTERN.match(text, start, end).end()
    if word_end == end:
        return False
    if text[word_end] == "(":
        return True
    if text[word_end] in "\"'" and text[start:word_end] in _FORMAT_STRING_PREFIXES:
        return True
    word_is_code = word_end - start < 16 or _CODE_VALUE_PATTERN.fullmatch(text, start, word_end)
    return bool(word_is_code and _CODE_AFTER_WORD_PATTERN.match(text, word_end, end))


def _unquoted_value_is_placeholder(text: str, start: int, end: int, raw_end: int) -> bool:
    """Return whether `text[start:end]`, an unquoted value, is a marker, reference, placeholder or mask.

    `raw_end` is where the value ended before its trailing quotes and brackets
    were dropped, since a marker or placeholder closes with one. The value
    stays when the placeholder is all of it, or when code, a closing quote or
    an escaped line break follows it. A plain word after it continues a secret.
    """
    placeholder = _PLACEHOLDER_PATTERN.match(text, start, raw_end)
    if placeholder is None:
        return False
    rest = placeholder.end()
    return (
        rest >= end
        or _CODE_AFTER_WORD_PATTERN.match(text, rest, end) is not None
        or _LITERAL_END_PATTERN.match(text, rest, end) is not None
    )


class _NextMatch:
    """Find the first match of a pattern at or after a position in one text, reusing the last search.

    `Pattern.search(text, pos)` tries each position the same way whatever
    `pos` is, so a match found from an earlier position is still the first
    one from any later position up to its start.
    """

    def __init__(self, pattern: re.Pattern[str], text: str) -> None:
        """Search `text` for `pattern`, with nothing searched yet."""
        self.pattern = pattern
        self.text = text
        self.searched_from = len(text) + 1
        self.found: re.Match[str] | None = None

    def at_or_after(self, position: int) -> re.Match[str] | None:
        """Return the first match that starts at or after `position`, or None."""
        still_ahead = self.found is None or self.found.start() >= position
        if not (self.searched_from <= position and still_ahead):
            self.found = self.pattern.search(self.text, position)
            self.searched_from = position
        return self.found


def _trim_value_trailer(text: str, start: int, end: int) -> int:
    """Return where `text[start:end]` ends once its trailing spaces, quotes and closing brackets are dropped."""
    while end > start and text[end - 1] in _VALUE_TRAILER:
        end -= 1
    return end


def _redact_unquoted_token_values(text: str) -> str:
    """Redact each unquoted value after a token key, through the end of the value.

    The value may hold quotes, `@` and `#`, and spaces after `:` or `=` outside
    a URL. A value under 16 characters stays, and so does one that reads as
    code or prose or, when it starts with `$`, `[`, `{`, `<` or `*`, one that
    is a marker, reference, placeholder or mask. Returns `text` with each
    redacted value replaced.
    """
    parts: list[str] = []
    position = 0
    # A value that stays readable is not skipped past, so the values after it
    # overlap it; reusing each end search and trailer trim scans the text once.
    word_value_ends = _NextMatch(_WORD_VALUE_END_PATTERN, text)
    spaced_value_ends = _NextMatch(_SPACED_VALUE_END_PATTERN, text)
    trimmed_raw_end = trimmed_end = -1
    for opener in _UNQUOTED_TOKEN_OPENER_PATTERN.finditer(text):
        if opener.start() < position:
            continue
        start = opener.end()
        in_url = opener.start() > 0 and text[opener.start() - 1] in "?&#"
        spaced = opener.group(1) is not None and not in_url
        hash_parameters = _HASH_PARAMETERS_PATTERN.match(text, start)
        search_from = hash_parameters.end() if hash_parameters else start
        found = (spaced_value_ends if spaced else word_value_ends).at_or_after(search_from)
        raw_end = found.start() if found else len(text)
        if raw_end != trimmed_raw_end:
            trimmed_raw_end, trimmed_end = raw_end, _trim_value_trailer(text, start, raw_end)
        end = max(start, trimmed_end)
        if end - start < 16:
            continue
        if text[start] in _PLACEHOLDER_OPENERS:
            if _unquoted_value_is_placeholder(text, start, end, raw_end):
                continue
        elif _unquoted_value_reads_as_code(text, start, end):
            continue
        parts.append(text[position:start])
        parts.append("[REDACTED_TOKEN]")
        position = end
    parts.append(text[position:])
    return "".join(parts)


def scrub_output(text: str) -> str:
    """Scrub tool and scratchpad output before it reaches the LLM, a stream, or storage.

    Runs `scrub_credentials`, then redacts OAuth tokens that no configured
    value names: JWTs, Google `ya29.` tokens, `Bearer` values, and the values
    of access/refresh/id token and client secret keys. Use it on output only;
    inbound user text keeps `scrub_credentials` so pasted connect
    credentials still reach the vault.
    """
    text = scrub_credentials(text)
    text = _TOKEN_SHAPE_PATTERN.sub("[REDACTED_TOKEN]", text)
    text = _redact_quoted_token_values(text)
    text = _QUOTED_BEARER_VALUE_PATTERN.sub(r"\1[REDACTED_TOKEN]", text)
    text = _BEARER_VALUE_PATTERN.sub(r"\1[REDACTED_TOKEN]", text)
    return _redact_unquoted_token_values(text)


def unscrub_marker(vault: "DataVault", marker: str) -> str | None:
    """Reverse of `scrub_credentials` for a single `[DS_...]` marker.

    Scoped to `vault`'s own saved connections, not the global
    `_DS_SECRET_VARS`/`os.environ` — those can hold another session's
    in-flight test credentials or flat leftovers. Returns None if `marker`
    isn't `[...]`-bracketed or matches no field in `vault`.
    """
    if not (marker.startswith("[") and marker.endswith("]")):
        return None
    key = marker[1:-1]
    for conn in vault.list_connections():
        env = vault.env_for(conn["engine"], conn["name"])
        if env and key in env:
            return env[key]
    return None


def _parse_picked_files(raw: str | None) -> list[dict]:
    """Parse a `_picked_files` JSON field, dropping malformed entries so they
    can't crash a later f.get(...) call or embed "None" as a fileId."""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [f for f in parsed if isinstance(f, dict) and f.get("id")]


def _labels_by_connection(vault: "DataVault") -> dict[tuple[str, str], str]:
    """Every connection's non-empty label, keyed by (engine, name).

    Prefers `_user_label`; falls back to the legacy `_label` field so an
    old connection's already-visible label still counts as "taken" when
    computing a new one.
    """
    out: dict[tuple[str, str], str] = {}
    for c in vault.list_connections():
        fields = vault.load(c["engine"], c["name"]) or {}
        label = (fields.get("_user_label") or fields.get("_label") or "").strip()
        if label:
            out[(c["engine"], c["name"])] = label
    return out


def ensure_unique_user_label(
    vault: "DataVault", candidate: str, *, exclude: tuple[str, str] | None = None
) -> str:
    """Return `candidate`, or `candidate` with a numeric suffix if it's already
    used by another connection anywhere in the vault (global uniqueness, not
    scoped per engine).

    `exclude=(engine, name)` removes that one connection's own label from the
    collision set first — pass the connection being edited/re-saved so keeping
    its label unchanged doesn't bump it to "label 2".
    """
    by_conn = _labels_by_connection(vault)
    if exclude:
        by_conn.pop(exclude, None)
    existing = set(by_conn.values())
    if candidate not in existing:
        return candidate
    n = 2
    while f"{candidate} {n}" in existing:
        n += 1
    return f"{candidate} {n}"


def generate_unique_conn_name(vault: "DataVault", engine: str) -> str:
    """Return a random 8-hex-char connection name not already used by `engine`.

    Shared by every /connect path that creates a brand-new connection.
    Previously duplicated inline at three call sites, one of which (the
    custom-datasource path) had drifted to skip the collision check —
    consolidating here closes that gap.
    """
    name = uuid.uuid4().hex[:8]
    while vault.load(engine, name) is not None:
        name = uuid.uuid4().hex[:8]
    return name


def default_user_label(vault: "DataVault", engine: str) -> str:
    """Default label for a brand-new connection to `engine` — the engine id,
    de-duplicated against every existing connection's label (any engine)."""
    return ensure_unique_user_label(vault, engine)


def _resolve_usage_notes(
    engine: str,
    usage_notes: "Mapping[str, str] | None",
    engine_def: "DatasourceEngine | None",
) -> str:
    """The usage notes to render for `engine`, or "": from the host's map, or
    from anton's registry entry `engine_def` only when the host sent None
    (see ChatSessionConfig.connector_usage_notes)."""
    if usage_notes is None:
        text = engine_def.usage_notes if engine_def else ""
    else:
        text = usage_notes.get(engine, "")
    return text.strip() if isinstance(text, str) else ""


# The section heading; the artifact prompts quote it, so it lives here once.
DATASOURCES_HEADER = "## Connected Data Sources"
_DATASOURCES_HEADING = "\n\n" + DATASOURCES_HEADER
_DATASOURCES_INTRO = (
    "Each connection has a Slug (a stable machine identifier, e.g. "
    "`postgres-7e8971c3`) and a Label (a human-readable name the user "
    "gave it, e.g. \"prod-db\"; may be unset). "
    "Credentials are pre-injected as namespaced DS_<ENGINE_NAME>__<FIELD> "
    "environment variables. Use them directly in scratchpad code "
    "(e.g. DS_POSTGRES_PROD_DB__HOST). "
    "Never read the data vault files directly.\n"
    "If you see `[DS_<NAME>]` patterns in scratchpad output, those are "
    "scrub-markers where a secret value was redacted before returning "
    "text to you — the actual value IS injected in the env var. Reference "
    "it by slug; never treat the bracket form as a literal credential "
    "or pass it back as a value to any tool.\n"
)
_GOOGLE_DRIVE_PARAGRAPH = (
    "\nConnected Google Drive accounts are available through Google OAuth credentials "
    "in the injected `DS_GOOGLE_DRIVE_<CONNECTION>__...` environment variables. "
    "Only claim Google Drive access if you can actually use those credentials successfully."
)


@dataclass(frozen=True)
class CatalogConnection:
    """One connection of a `DatasourceCatalog`: the names a caller can match
    against, and the connection's rendered block."""

    slug: str
    engine: str
    # The registry's display name, or the engine id when the registry has none.
    engine_name: str
    # "" when the user gave none.
    label: str
    block: str
    # A paragraph rendered once after all usage notes when any kept
    # connection carries it (the Google Drive availability paragraph), or "".
    footer: str = ""


# eq=False: `notes` is a dict, so a generated __hash__ would raise; a catalog
# is an identity object and nothing compares two by value.
@dataclass(frozen=True, eq=False)
class DatasourceCatalog:
    """The `## Connected Data Sources` section as data, so it can be rendered
    for a subset of its connections."""

    connections: tuple[CatalogConnection, ...]
    # engine -> its rendered usage-notes section; only engines that have notes.
    notes: Mapping[str, str]

    def render(self, slugs: Collection[str] | None = None) -> str:
        """The section for the connections in `slugs` (all when None).

        Usage notes follow the connection blocks, once per engine of a kept
        connection; footers come last, each once.
        """
        _reject_bare_string(slugs)
        kept = [c for c in self.connections if slugs is None or c.slug in slugs]
        lines = [_DATASOURCES_HEADING, _DATASOURCES_INTRO]
        lines.extend(c.block for c in kept)
        for engine in dict.fromkeys(c.engine for c in kept):
            if engine in self.notes:
                lines.append(self.notes[engine])
        lines.extend(dict.fromkeys(c.footer for c in kept if c.footer))
        return "\n".join(lines)


def _reject_bare_string(slugs: Collection[str] | None) -> None:
    # A str is a Collection[str], so `slug in slugs` would test substrings.
    if isinstance(slugs, str):
        raise TypeError("slugs must be a collection of slugs, not a single str")


def collect_datasource_catalog(
    vault: DataVault | None,
    *,
    usage_notes: "Mapping[str, str] | None" = None,
    slugs: Collection[str] | None = None,
) -> DatasourceCatalog | None:
    """The vault's connections as a `DatasourceCatalog` (DS_* names, never
    values), or None when there are none or the vault cannot list them.

    `usage_notes`: see ChatSessionConfig.connector_usage_notes. A value that
    is not a mapping is treated as an empty map: the host did send something,
    so anton's registry must not stand in.

    `slugs`, when given, leaves every other connection out before its record
    is read: a remote vault fetches each record over the network.

    An unnamed connection's block shows its engine as the label, suffixed to
    stay unique within the returned catalog; its `label` field stays "".
    """
    _reject_bare_string(slugs)
    try:
        vault = vault or LocalDataVault()
        conns = vault.list_connections()
    except Exception:
        return None
    if not conns:
        return None
    if usage_notes is not None and not isinstance(usage_notes, Mapping):
        logger.warning(
            "connector usage notes are a %s, not a mapping; rendering none",
            type(usage_notes).__name__,
        )
        usage_notes = {}
    # Google Drive's drive.file OAuth scope only covers files the app created
    # itself, plus files explicitly granted via the Google Picker (persisted
    # as a `_picked_files` vault field).
    #
    # Those granted files are deliberately NOT listed here. Listing them needs
    # the *current project* to scope by, and `projects` is a cowork concept
    # this function has no access to — so the list rendered here was
    # unscoped, and named files from other projects. cowork-server renders the
    # scoped list itself, via ConnectionsService.picked_files_by_project() ->
    # the harness prompt suffix, which is the single renderer. All we track
    # here is *whether* any exist, so the availability paragraph still fires
    # for a Picker-only connection exactly as it always has.
    #
    # One registry per call, and one lookup per engine: DatasourceRegistry()
    # re-reads the built-in and user datasources.md and builds every engine on
    # each construction (only the YAML parse is cached), and this runs on
    # every chat turn.
    registry = DatasourceRegistry()
    engine_names: dict[str, str] = {}
    notes: dict[str, str] = {}
    connections: list[CatalogConnection] = []
    created_at: list[str] = []
    for c in conns:
        engine = c["engine"]
        slug = f"{engine}-{c['name']}"
        if slugs is not None and slug not in slugs:
            continue
        if engine not in engine_names:
            engine_def = registry.get(engine)
            engine_names[engine] = engine_def.display_name if engine_def else engine
            text = _resolve_usage_notes(engine, usage_notes, engine_def)
            if text:
                notes[engine] = (
                    f"\n### Usage notes: {engine_names[engine]} (engine `{engine}`)\n{text}"
                )
        engine_name = engine_names[engine]
        # read_record gives fields + secure_keys in one call; fall back to load
        # for any vault backend that doesn't implement it.
        if hasattr(vault, "read_record"):
            record = vault.read_record(engine, c["name"]) or {}
            fields = record.get("fields", {}) or {}
            secure_keys = record.get("secure_keys")
        else:
            fields = vault.load(engine, c["name"]) or {}
            secure_keys = None
        prefix = _slug_env_prefix(engine, c["name"])
        # Skip `_`-prefixed bookkeeping (`_connector_id`, `_method`, `_label`,
        # `_user_label`) — they're not credential env vars the agent should
        # reference.
        var_names = [
            f"{prefix}__{k.upper()}" for k in fields if not k.startswith("_")
        ]
        user_label = str(fields.get("_user_label", "")).strip() or str(
            fields.get("_label", "")
        ).strip()

        # The slug/label header is prepended below, once every stored label is known.
        lines = [f"Engine: {engine_name}"]
        # Same non-secret allowlist `_connection_identity()` encodes
        # (host/database/email, secure_keys-gated) — inlined here as
        # independent lines instead of one collapsed string.
        host = str(fields.get("host", "")).strip()
        database = str(fields.get("database", "")).strip()
        email = str(fields.get("email", "")).strip() or str(fields.get("account_email", "")).strip()
        if host and "host" not in (secure_keys or []):
            lines.append(f"Host: {host}")
        if database and "database" not in (secure_keys or []):
            lines.append(f"Database: {database}")
        if email and "email" not in (secure_keys or []) and "account_email" not in (secure_keys or []):
            lines.append(f"Account: {email}")
        lines.append("Credential env vars:")
        for var in var_names:
            lines.append(f" - {var}")

        # Parsed, not merely truthy-checked: a `_picked_files` holding only
        # malformed entries must read as "none", the same way it did when
        # this drove the (now removed) listing.
        google_drive_access = engine == "google_drive" and (
            fields.get("auth_type") == "oauth"
            or bool(_parse_picked_files(fields.get("_picked_files")))
        )
        created_at.append(str(c.get("created_at") or ""))
        connections.append(
            CatalogConnection(
                slug=slug,
                engine=engine,
                engine_name=engine_name,
                label=user_label,
                block="\n".join(lines),
                footer=_GOOGLE_DRIVE_PARAGRAPH if google_drive_access else "",
            )
        )
    # An unnamed connection shows its engine, not "(none)", since "(none)" made
    # the model estimate figures it lacked. The suffix rule matches
    # ensure_unique_user_label, against rendered connections only, because
    # `slugs` must not trigger reads of records it leaves out. Oldest first,
    # so adding a connection never renames one the model already knows.
    labels = [conn.label for conn in connections]
    taken = {label for label in labels if label}
    unnamed = [i for i, label in enumerate(labels) if not label]
    for i in sorted(unnamed, key=lambda i: (created_at[i], connections[i].slug)):
        base = connections[i].engine or connections[i].slug
        label, n = base, 2
        while label in taken:
            label, n = f"{base} {n}", n + 1
        taken.add(label)
        labels[i] = label
    connections = [
        replace(conn, block=f"\n### Slug: `{conn.slug}` — Label: {label}\n{conn.block}")
        for conn, label in zip(connections, labels)
    ]
    return DatasourceCatalog(connections=tuple(connections), notes=notes)


def build_datasource_context(
    vault: DataVault,
    active_only: str | None = None,
    *,
    usage_notes: "Mapping[str, str] | None" = None,
) -> str:
    """Build a system-prompt section listing available DS_* env vars by name.

    Shows the LLM what data sources are connected and which environment
    variable names to use — without exposing any credential values.

    If active_only is set, only the matching slug is included, with only its
    engine's usage notes. `usage_notes`: see `collect_datasource_catalog`.
    """
    catalog = collect_datasource_catalog(
        vault, usage_notes=usage_notes, slugs=[active_only] if active_only else None
    )
    return catalog.render() if catalog is not None else ""


def _connection_identity(fields: dict, secure_keys: list | None = None) -> str | None:
    """A short, non-secret label so the LLM can tell connections apart — the
    account email, or the database host (+ name).

    Scoped to these inherently non-secret fields on purpose: it is NOT a dump of
    every field (which would surface opaque ``client_id`` / config like
    ``ssl_mode``) and never a secret. Lets the agent pick the right account
    ("send from my support email") even when the connection slug is an old random
    one. Any field a record explicitly marks secret via ``secure_keys`` is
    skipped, defensively, even though email/host are not normally secret.
    """
    secure = set(secure_keys or [])
    # `email` is collected by credential forms; `account_email` is stored by the
    # OAuth flows (from userinfo). Either identifies the account.
    for key in ("email", "account_email"):
        val = str(fields.get(key, "")).strip()
        if val and key not in secure:
            return val
    host = str(fields.get("host", "")).strip()
    if host and "host" not in secure:
        database = str(fields.get("database", "")).strip()
        if database and "database" not in secure:
            return f"{host}/{database}"
        return host
    return None


def _is_oauth_connection(vault: DataVault, engine: str, name: str) -> bool:
    """True when this connection's own stored fields say auth_type=oauth.

    An OAuth connection's engine name can collide with an unrelated registry
    entry of the same name (gmail's registry entry is the legacy IMAP
    connector) — its field schema never matches, so it must not be
    classified against that registry entry. Cheap: TurnKeyDataVault caches
    per (engine, name), so this never costs a second network fetch.
    """
    record = vault.read_record(engine, name) if hasattr(vault, "read_record") else None
    fields = (record or {}).get("fields") or {}
    return fields.get("auth_type") == "oauth"


def clear_ds_env() -> None:
    """Unconditionally clear every DS_* env var and reset the known/secret
    registries — call this even on a turn with nothing to reinject (e.g. no
    oauth block this turn), so a var from a prior call can never survive into
    a turn that has no vault of its own to reset it. This is the version
    callers with no vault instance in hand can use; callers that do have one
    should call `restore_namespaced_env()` so any vault-specific override of
    `clear_ds_env()` (e.g. cache invalidation) still runs."""
    _reset_registered_ds_vars()
    _sweep_ds_env_vars()


def restore_namespaced_env(vault: DataVault) -> None:
    """Rebuild this turn's DS_* var registries and value map from the vault.

    Touches no process state: the values are recorded for the calling context
    only, and reach a subprocess through its own env (the scratchpad manager
    derives them from the same vault). Two turns on different vaults can run
    at once without seeing each other's credentials, and a connection this
    turn has disabled cannot be reinstated by another turn. A value the turn
    already hid stays hidden until the turn ends, even when the vault no
    longer lists it, because the turn's pads still hold it.
    """
    # Kept so a failed rebuild restores what the turn had rather than leaving
    # it with nothing, which would silently turn scrubbing off for the turn.
    previous = (set(_DS_SECRET_VARS), set(_DS_KNOWN_VARS), _ds_env_values.snapshot())
    try:
        _rebuild_ds_registries(vault)
    except Exception:
        _DS_SECRET_VARS.replace(previous[0])
        _DS_KNOWN_VARS.replace(previous[1])
        if previous[2] is not None:
            _ds_env_values.set(previous[2])
        raise


def _rebuild_ds_registries(vault: DataVault) -> None:
    # Emptied in place rather than via _reset_registered_ds_vars(), which drops
    # the scope: that would hide a mid-turn rebuild from its own turn.
    _DS_SECRET_VARS.clear()
    _DS_KNOWN_VARS.clear()
    dreg = DatasourceRegistry()
    env_values: dict[str, str] = {}
    for conn in vault.list_connections():
        engine, name = conn.get("engine"), conn.get("name")
        if not (engine and name):
            continue
        # Per connection, because a pad's env is built that way too: a record
        # that aborted the whole rebuild would still be injected there.
        try:
            edef = dreg.get(engine)
            oauth = _is_oauth_connection(vault, engine, name)
        except Exception:
            # Unclassifiable, so take the unknown-field path: its coarse net
            # over-redacts, which is the safe direction.
            edef, oauth = None, False
            logger.warning("Could not classify %s/%s", engine, name, exc_info=True)
        try:
            if edef is not None and not oauth:
                register_secret_vars(edef, engine=engine, name=name)
            else:
                _register_unregistered_connection_vars(vault, engine, name)
        except Exception:
            logger.warning(
                "Could not register vars for %s/%s; its credentials will not be "
                "scrubbed from this turn's output",
                engine, name, exc_info=True,
            )
        try:
            env_values.update(vault.env_for(engine, name) or {})
        except Exception:
            # Must not block the rest, but it leaves that connection
            # unscrubbable for this turn, so say so. Names only, never a value.
            logger.warning(
                "Could not resolve values for %s/%s; its credentials will not be "
                "scrubbed from this turn's output",
                engine, name, exc_info=True,
            )
    _ds_env_values.set(env_values)
    _hold_exact_secrets()


def find_matching_connection(
    vault: DataVault,
    engine_def: "DatasourceEngine",
    credentials: dict[str, str],
) -> str | None:
    """Return the name of an existing connection with matching identity fields.

    Uses the engine's ``name_from`` declaration as the identity signature.
    If those fields exist in both the incoming credentials and a stored
    connection of the same engine (and all values match), the stored name
    is returned so the caller can update in place instead of creating a
    duplicate. Returns None when no match is found, or when the engine has
    no ``name_from`` (custom/ad-hoc engines).
    """
    name_from = engine_def.name_from
    if not name_from:
        return None
    if isinstance(name_from, str):
        name_from = [name_from]
    incoming_sig = tuple(credentials.get(f, "") for f in name_from)
    if not any(incoming_sig):
        return None
    for conn in vault.list_connections():
        if conn["engine"] != engine_def.engine:
            continue
        existing_fields = vault.load(conn["engine"], conn["name"]) or {}
        existing_sig = tuple(existing_fields.get(f, "") for f in name_from)
        if existing_sig == incoming_sig:
            return conn["name"]
    return None


def save_connection(
    vault: DataVault,
    engine_def: "DatasourceEngine",
    name: str,
    credentials: dict[str, str],
    *,
    secure_keys: list[str] | None = None,
) -> str:
    """Persist credentials and refresh DS_* env vars. Returns the connection slug.

    Shared save path used by both /connect and connect_new_datasource tool.

    `secure_keys` is forwarded to the vault so future reads can
    classify fields without falling back to the name-matching
    heuristic. Optional for backward compatibility — callers that
    don't have the connector spec handy (the chat-side tool path,
    for example) can still call this without supplying the list.
    """
    vault.save(engine_def.engine, name, credentials, secure_keys=secure_keys)
    restore_namespaced_env(vault)
    register_secret_vars(engine_def, engine=engine_def.engine, name=name)
    return f"{engine_def.engine}-{name}"


def persist_custom_engine(
    registry: DatasourceRegistry,
    display_name: str,
    fields: list["DatasourceField"],
    test_snippet: str = "",
    pip: str = "",
) -> "DatasourceEngine | None":
    """Append a YAML block for a custom engine to ``~/.anton/datasources.md``.

    Reloads the registry and returns the parsed DatasourceEngine on success.
    Returns None if the newly-written block fails parse validation (caller
    may still fall back to an in-memory engine).
    """
    from anton.core.datasources.datasource_registry import DatasourceEngine

    slug = re.sub(r"[^\w]", "_", display_name.lower()).strip("_")
    field_lines = "\n".join(
        f"  - {{ name: {f.name}, required: {str(f.required).lower()}, "
        f"secret: {str(f.secret).lower()}, "
        f"description: \"{f.description}\" }}"
        for f in fields
    )
    test_snippet_yaml = ""
    if test_snippet:
        indented = "\n".join(f"  {line}" for line in test_snippet.splitlines())
        test_snippet_yaml = f"test_snippet: |\n{indented}\n"

    yaml_block = (
        f"\n---\n\n## {display_name}\n"
        "```yaml\n"
        f"engine: {slug}\n"
        f"display_name: {display_name}\n"
        + (f"pip: {pip}\n" if pip else "")
        + f"fields:\n{field_lines}\n"
        + test_snippet_yaml
        + "```\n"
    )
    user_ds_path = Path("~/.anton/datasources.md").expanduser()
    tmp_path = user_ds_path.with_suffix(".tmp")

    existing = (
        user_ds_path.read_text(encoding="utf-8")
        if user_ds_path.is_file()
        else ""
    )
    existing = remove_engine_block(existing, slug)

    user_ds_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(existing + yaml_block, encoding="utf-8")

    parsed = registry.validate_file(tmp_path)
    if slug not in parsed:
        tmp_path.unlink(missing_ok=True)
        return None

    shutil.move(str(tmp_path), str(user_ds_path))
    registry.reload()
    engine_def = registry.get(slug)
    if engine_def is None:
        engine_def = DatasourceEngine(
            engine=slug,
            display_name=display_name,
            pip=pip,
            fields=list(fields),
            test_snippet=test_snippet,
            custom=True,
        )
    return engine_def


def remove_engine_block(text: str, slug: str) -> str:
    """Return *text* with any YAML datasource block for *slug* removed."""
    cleaned = []
    prev = 0
    for m in _YAML_BLOCK_RE.finditer(text):
        try:
            data = yaml.safe_load(m.group(3))
            is_dup = isinstance(data, dict) and str(data.get("engine", "")) == slug
        except Exception:
            is_dup = False
        if is_dup:
            pre = text[prev : m.start()].rstrip()
            pre = re.sub(r"\n---\s*$", "", pre)
            cleaned.append(pre)
        else:
            cleaned.append(text[prev : m.end()])
        prev = m.end()
    cleaned.append(text[prev:])
    return "".join(cleaned)
