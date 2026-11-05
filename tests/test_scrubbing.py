from __future__ import annotations

import asyncio
import base64
import dataclasses
import html
import json
import os
import re
import time
import urllib.parse
from unittest.mock import patch

import pytest

from anton.core.datasources.data_vault import _slug_env_prefix
from anton.core.session import _scrub_user_input
from anton.tools import SCRUBBED_VALUE_RE
from anton.utils import datasources
from anton.utils.datasources import (
    _DS_KNOWN_VARS,
    _DS_SECRET_VARS,
    _PROVIDER_SECRET_VARS,
    _reset_registered_ds_vars,
    exact_secret_prefix_at_end,
    exact_secret_suffix_at_start,
    longest_exact_secret,
    scrub_credentials,
    scrub_output,
    set_ds_env_values,
)


# One sample per Google key format currently issued. The `_`/`-` variant is
# derived because a realistic literal trips GitHub push protection.
_AUTH_KEY = "AQ.AbCdEfGhIjKlMnOpQrStUvWxYz0123456789AbCdEfGhIjKlMn"
GOOGLE_KEY_SAMPLES = [
    "AIzaSyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q",  # standard key
    _AUTH_KEY,  # authorization key
    _AUTH_KEY[:30] + "_" + _AUTH_KEY[31:45] + "-" + _AUTH_KEY[46:],  # with _ and -
]


@pytest.fixture(autouse=True)
def clean_ds_state():
    """Clear _DS_SECRET_VARS, _DS_KNOWN_VARS, and all DS_* env vars around each test."""
    def _clean():
        _reset_registered_ds_vars()
        for k in list(os.environ):
            if k.startswith("DS_"):
                del os.environ[k]

    _clean()
    yield
    _clean()


class TestLongestExactSecret:
    """Counts the longest form of the values scrub_credentials matches by exact value."""

    def test_counts_secret_unknown_and_provider_values_only(self, monkeypatch):
        """Secret fields, unknown DS_* over 8 chars and provider keys over 8
        chars count; known non-secret fields and short values do not. A plain
        value's longest form is its base64, a third longer."""
        _DS_SECRET_VARS.add("DS_PG_PROD__PASSWORD")
        _DS_KNOWN_VARS.add("DS_PG_PROD__PASSWORD")
        _DS_KNOWN_VARS.add("DS_PG_PROD__HOST")
        set_ds_env_values({
            "DS_PG_PROD__PASSWORD": "p" * 12,
            "DS_PG_PROD__HOST": "h" * 50,
            "DS_CUSTOM__TOKEN": "u" * 30,
            "DS_CUSTOM__FLAG": "on",
        })
        for key in _PROVIDER_SECRET_VARS:
            monkeypatch.delenv(key, raising=False)
        assert longest_exact_secret() == len(base64.b64encode(b"u" * 30))

        monkeypatch.setenv("OPENAI_API_KEY", "k" * 42)
        assert longest_exact_secret() == len(base64.b64encode(b"k" * 42))

        monkeypatch.setenv("OPENAI_API_KEY", "k" * 8)
        assert longest_exact_secret() == len(base64.b64encode(b"u" * 30))

    def test_a_short_secret_field_counts(self, monkeypatch):
        """Secret fields match at any length, unlike unknown and provider values.
        Five bytes print as six whole base64 characters."""
        _DS_SECRET_VARS.add("DS_PG_PROD__PASSWORD")
        _DS_KNOWN_VARS.add("DS_PG_PROD__PASSWORD")
        set_ds_env_values({"DS_PG_PROD__PASSWORD": "p" * 5})
        for key in _PROVIDER_SECRET_VARS:
            monkeypatch.delenv(key, raising=False)
        assert longest_exact_secret() == 6

    def test_counts_a_url_encoded_form_longer_than_the_value(self, monkeypatch):
        """A cut must leave room for the longest form the scrub matches, here `%2F` per `/`."""
        _DS_SECRET_VARS.add("DS_PG_PROD__PASSWORD")
        _DS_KNOWN_VARS.add("DS_PG_PROD__PASSWORD")
        set_ds_env_values({"DS_PG_PROD__PASSWORD": "/" * 12})
        for key in _PROVIDER_SECRET_VARS:
            monkeypatch.delenv(key, raising=False)
        assert longest_exact_secret() == len("%2F" * 12)

    def test_is_zero_with_nothing_to_match(self, monkeypatch):
        """A known non-secret field, and unknown or provider values of 8 chars
        or fewer, set nothing to hold back."""
        _DS_KNOWN_VARS.add("DS_PG_PROD__HOST")
        set_ds_env_values({"DS_PG_PROD__HOST": "h" * 50, "DS_CUSTOM__FLAG": "u" * 8})
        for key in _PROVIDER_SECRET_VARS:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "k" * 8)
        assert longest_exact_secret() == 0


class TestScrubCredentials:
    """Focused regression tests for _scrub_credentials short-secret handling."""

    def test_registered_6char_secret_scrubbed(self, monkeypatch):
        """A 6-character registered secret is scrubbed regardless of length."""
        _DS_SECRET_VARS.add("DS_PASSWORD")
        monkeypatch.setenv("DS_PASSWORD", "abc123")
        result = scrub_credentials("auth failed: abc123")
        assert "abc123" not in result
        assert "[DS_PASSWORD]" in result

    def test_registered_8char_secret_scrubbed(self, monkeypatch):
        """An 8-character registered secret is scrubbed (was at the old threshold)."""
        _DS_SECRET_VARS.add("DS_API_KEY")
        monkeypatch.setenv("DS_API_KEY", "tok12345")
        result = scrub_credentials("token=tok12345 rejected")
        assert "tok12345" not in result
        assert "[DS_API_KEY]" in result

    def test_registered_1char_secret_scrubbed(self, monkeypatch):
        """A 1-character registered secret is scrubbed."""
        _DS_SECRET_VARS.add("DS_SECRET")
        monkeypatch.setenv("DS_SECRET", "x")
        result = scrub_credentials("value=x here")
        assert "=x " not in result
        assert "[DS_SECRET]" in result

    def test_non_secret_var_not_scrubbed(self, monkeypatch):
        """A known but non-secret DS_* var (e.g. DS_HOST) stays readable."""
        _DS_KNOWN_VARS.add("DS_HOST")
        monkeypatch.setenv("DS_HOST", "mydbhostname")
        result = scrub_credentials("host=mydbhostname")
        assert "mydbhostname" in result

    def test_unknown_short_ds_var_not_scrubbed(self, monkeypatch):
        """Unknown DS_* vars with short values are NOT scrubbed (heuristic threshold)."""
        monkeypatch.setenv("DS_ENABLE_FEATURE", "on")
        result = scrub_credentials("flag=on active")
        assert "on" in result


# A saved password with a character for each encoder to change: `&`, a space,
# quotes, `/`, `<`, `>`, a backslash and a non-ASCII letter.
ENCODED_SECRET = 'Tr0ub4dor&3 "horse"/<staple>\\é'
SECRET_LABEL = "DS_PG_PROD__PASSWORD"

# Each returns the encoded value alone, without the quotes around a literal.
SECRET_ENCODERS = {
    "quote": urllib.parse.quote,
    "quote-everything": lambda value: urllib.parse.quote(value, safe=""),
    "quote-plus": urllib.parse.quote_plus,
    "lowercase-hex": lambda value: re.sub(
        r"%[0-9A-F]{2}", lambda found: found.group().lower(), urllib.parse.quote(value, safe="")
    ),
    "json": lambda value: json.dumps(value)[1:-1],
    "json-unicode": lambda value: json.dumps(value, ensure_ascii=False)[1:-1],
    "json-in-json": lambda value: json.dumps(json.dumps(value))[3:-3],
    "html": html.escape,
    "repr": lambda value: repr(value)[1:-1],
    "ascii": lambda value: ascii(value)[1:-1],
    "bytes-repr": lambda value: repr(value.encode())[2:-1],
    "backslash-before-each-symbol": lambda value: re.sub(r"([^A-Za-z0-9._-])", r"\\\1", value),
}

# Output that holds none of the saved values below, only near misses of them.
ORDINARY_OUTPUT = """\
rows = df.query("status == 'active'").head(20)
{"user": "analyst", "host": "db.example.com", "port": 5432, "note": "tab\\there \\"quoted\\" \\u00e9"}
<a href="/reports?q=revenue%20by%20region&amp;year=2024">Revenue &amp; costs &lt;draft&gt;</a>
  File "C:\\\\Users\\\\analyst\\\\cell.py", line 3, in <module>
ValueError: could not convert string to float: 'n/a'
hint: Tr0ub4dor&3 is only its start; abc1234, xabc123, ABC123 and Hunter2 are other ids
data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==
"""


def _save_secrets(values: dict[str, str]) -> None:
    """Register each of `values` as this turn's saved secret under its key."""
    for key in values:
        _DS_SECRET_VARS.add(key)
        _DS_KNOWN_VARS.add(key)
    set_ds_env_values(values)


def _secret_only_runs(blob: str, start: int, length: int) -> list[str]:
    """Return each 6-character run of base64 `blob` that encodes only its bytes `start` to `start + length`."""
    first, end = -(-8 * start // 6), 8 * (start + length) // 6
    return [blob[index : index + 6] for index in range(first, end - 5)]


class _CountingPattern:
    """A compiled pattern that counts how often `subn` runs it."""

    def __init__(self, pattern: re.Pattern[str]) -> None:
        self.pattern = pattern
        self.calls = 0

    def subn(self, replacement: str, text: str) -> tuple[str, int]:
        self.calls += 1
        return self.pattern.subn(replacement, text)


class TestEncodedSavedSecrets:
    """A saved value goes in the forms encoders print it in, not only as typed (ENG-3428)."""

    @pytest.fixture(autouse=True)
    def no_provider_keys(self, monkeypatch):
        for key in _PROVIDER_SECRET_VARS:
            monkeypatch.delenv(key, raising=False)

    @pytest.fixture
    def pattern_runs(self, monkeypatch) -> dict[str, _CountingPattern]:
        """Count, per saved value, how often scrub_credentials runs its pattern."""
        counters: dict[str, _CountingPattern] = {}
        build = datasources._secret_forms

        def counting_forms(value: str):
            forms = build(value)
            counter = counters.setdefault(value, _CountingPattern(forms.pattern))
            return dataclasses.replace(forms, pattern=counter)

        monkeypatch.setattr(datasources, "_secret_forms", counting_forms)
        return counters

    @pytest.mark.parametrize("encode", list(SECRET_ENCODERS.values()), ids=list(SECRET_ENCODERS))
    def test_an_encoded_saved_secret_is_hidden_whole(self, encode):
        _save_secrets({SECRET_LABEL: ENCODED_SECRET})
        assert scrub_credentials(f"pw={encode(ENCODED_SECRET)};") == f"pw=[{SECRET_LABEL}];"

    @pytest.mark.parametrize("tail", [b"@db.example.com:5432", b""], ids=["inside", "at-the-end"])
    @pytest.mark.parametrize("offset", [0, 1, 2])
    @pytest.mark.parametrize("encode", [
        base64.b64encode,
        base64.urlsafe_b64encode,
        lambda data: base64.urlsafe_b64encode(data).rstrip(b"="),
    ], ids=["standard", "urlsafe", "urlsafe-unpadded"])
    def test_a_saved_secret_inside_a_longer_base64_string_is_hidden(self, encode, offset, tail):
        """As in an HTTP Basic header, where the user name sets which of three byte offsets the password starts at."""
        _save_secrets({SECRET_LABEL: ENCODED_SECRET})
        head = b"u" * (2 + offset) + b":"
        blob = encode(head + ENCODED_SECRET.encode() + tail).decode()
        result = scrub_credentials(f"Authorization: Basic {blob}")

        for run in _secret_only_runs(blob, len(head), len(ENCODED_SECRET.encode())):
            assert run not in result
        assert result.startswith(f"Authorization: Basic {blob[:4]}")
        assert result.endswith(blob[-4:] if tail else f"[{SECRET_LABEL}]")

    def test_base64_percent_encoded_in_a_url_is_hidden(self):
        _save_secrets({SECRET_LABEL: ENCODED_SECRET})
        blob = base64.b64encode(b"zz" + ENCODED_SECRET.encode()).decode()
        url = f"https://api.example.com/v1?auth={urllib.parse.quote(blob, safe='')}&page=2"
        result = scrub_credentials(url)

        for run in _secret_only_runs(blob, 2, len(ENCODED_SECRET.encode())):
            assert run not in urllib.parse.unquote(result)
        assert result.startswith("https://api.example.com/v1?auth=")
        assert result.endswith(f"[{SECRET_LABEL}]&page=2")

    @pytest.mark.parametrize("change_case", [str.upper, str.lower, str.swapcase])
    def test_a_long_saved_secret_is_hidden_in_any_ascii_case(self, change_case):
        _save_secrets({SECRET_LABEL: "Tr0ub4dor&3horse"})
        assert scrub_credentials(f"pw={change_case('Tr0ub4dor&3horse')};") == f"pw=[{SECRET_LABEL}];"

    def test_a_long_saved_secret_is_hidden_inside_a_word(self):
        _save_secrets({SECRET_LABEL: "Tr0ub4dor&3horse"})
        assert scrub_credentials("xTr0ub4dor&3horse_2") == f"x[{SECRET_LABEL}]_2"

    def test_a_short_saved_secret_is_hidden_only_as_a_whole_word_in_its_own_case(self):
        """Matching `abc123` inside other words or in another case would hit ordinary text."""
        _save_secrets({SECRET_LABEL: "abc123"})
        assert scrub_credentials("auth abc123 failed") == f"auth [{SECRET_LABEL}] failed"
        for text in ["abc123_x", "xabc123", "abc1234", "ABC123", "Abc123"]:
            assert scrub_credentials(text) == text

    def test_a_short_saved_secret_ending_in_a_symbol_is_hidden_before_a_letter(self):
        """Only an edge that is a word character needs a non-word neighbor."""
        _save_secrets({SECRET_LABEL: "Hunter2!"})
        assert scrub_credentials("pw=Hunter2!x") == f"pw=[{SECRET_LABEL}]x"

    @pytest.mark.parametrize("encode", [urllib.parse.quote, urllib.parse.quote_plus], ids=["quote", "quote-plus"])
    def test_a_short_saved_secret_is_hidden_in_both_url_forms(self, encode):
        _save_secrets({SECRET_LABEL: "p@s w!"})
        result = scrub_credentials(f"postgresql://app:{encode('p@s w!')}@db:5432/prod")
        assert result == f"postgresql://app:[{SECRET_LABEL}]@db:5432/prod"

    def test_base64_is_matched_from_five_bytes(self):
        """Four bytes are five whole base64 characters, too few to tell from other text."""
        _save_secrets({SECRET_LABEL: "abcd"})
        assert scrub_credentials("id YWJjZA==") == "id YWJjZA=="
        _save_secrets({SECRET_LABEL: "abc12"})
        assert scrub_credentials("id YWJjMTI=") == f"id [{SECRET_LABEL}]"

    def test_a_saved_value_holding_a_shorter_one_is_hidden_whole(self):
        """Longest first, so the short secret does not cut the long one and leave its tail."""
        _DS_SECRET_VARS.add("DS_A_PROD__PASSWORD")
        _DS_KNOWN_VARS.add("DS_A_PROD__PASSWORD")
        set_ds_env_values({"DS_A_PROD__PASSWORD": "abc", "DS_CUSTOM_PROD__TOKEN": "abc-defghijk"})
        assert scrub_credentials("tok=abc-defghijk") == "tok=[DS_CUSTOM_PROD__TOKEN]"

    @pytest.mark.parametrize("encode", [
        urllib.parse.quote,
        urllib.parse.quote_plus,
        lambda value: base64.b64encode(value.encode()).decode(),
    ], ids=["quote", "quote-plus", "base64"])
    @pytest.mark.parametrize(
        "label",
        ["DS_PG_PROD__PASSWORD", "DS_CUSTOM_PROD__TOKEN", "ANTON_MINDS_API_KEY"],
        ids=["saved-secret", "unknown-ds-var", "provider-key"],
    )
    def test_every_kind_of_exact_value_is_hidden_encoded(self, label, encode, monkeypatch):
        """`quote` keeps `/` and writes a space as `%20`; `quote_plus` encodes `/` and writes `+`."""
        value = "Zt7/Q8+Q8 Q8Q8Q8Q8"
        if label == "ANTON_MINDS_API_KEY":
            monkeypatch.setenv(label, value)
        elif label == "DS_PG_PROD__PASSWORD":
            _save_secrets({label: value})
        else:
            set_ds_env_values({label: value})
        assert scrub_credentials(f"auth={encode(value)}&page=2") == f"auth=[{label}]&page=2"

    def test_a_saved_value_is_not_matched_inside_a_marker_this_scrub_wrote(self):
        """The password goes first; `custom_prod` then folds onto `CUSTOM_PROD` in its marker's label."""
        _DS_SECRET_VARS.add("DS_CUSTOM_PROD__PASSWORD")
        _DS_KNOWN_VARS.add("DS_CUSTOM_PROD__PASSWORD")
        set_ds_env_values({"DS_CUSTOM_PROD__PASSWORD": "Tr0ub4dor&3horse", "DS_CUSTOM_PROD__DATABASE": "custom_prod"})
        result = scrub_credentials("connecting to custom_prod with Tr0ub4dor&3horse")
        assert result == "connecting to [DS_CUSTOM_PROD__DATABASE] with [DS_CUSTOM_PROD__PASSWORD]"

    def test_a_marker_from_an_earlier_scrub_is_left_alone(self):
        """Replayed history already holds markers; a value folding onto one's label is not in it."""
        set_ds_env_values({"DS_CUSTOM_PROD__DATABASE": "custom_prod"})
        text = "earlier: [DS_CUSTOM_PROD__PASSWORD] and [REDACTED_TOKEN]; now custom_prod"
        assert scrub_credentials(text) == (
            "earlier: [DS_CUSTOM_PROD__PASSWORD] and [REDACTED_TOKEN]; now [DS_CUSTOM_PROD__DATABASE]"
        )

    def test_a_secret_holding_a_marker_shaped_run_is_still_hidden(self):
        """`p[A_B]ss` holds what looks like a marker; a match that holds it whole is the secret."""
        _save_secrets({SECRET_LABEL: "p[A_B]ss-w0rd!x"})
        assert scrub_credentials("pw=p[A_B]ss-w0rd!x;") == f"pw=[{SECRET_LABEL}];"

    def test_a_password_holding_a_bracketed_capital_is_hidden(self):
        """`[Q]` has no underscore, so it is part of the password, not a marker."""
        _save_secrets({SECRET_LABEL: "x7[Q]abc"})
        assert scrub_credentials("pw x7[Q]abc ok") == f"pw [{SECRET_LABEL}] ok"

    @pytest.mark.parametrize("secret", ["HUNTER2X9Q", "PROD_DB_PASS_2024"], ids=["no-underscore", "marker-shaped"])
    def test_a_secret_printed_inside_brackets_is_hidden(self, secret):
        """`[HUNTER2X9Q]` is no marker. `[PROD_DB_PASS_2024]` is marker-shaped, but the match is its
        whole label, and a label is a variable name, not a saved value."""
        _save_secrets({SECRET_LABEL: secret})
        assert scrub_credentials(f"login [{secret}] failed") == f"login [[{SECRET_LABEL}]] failed"

    def test_every_marker_the_scrub_writes_is_a_marker(self, monkeypatch):
        """Each label the scrub writes holds an underscore, so the narrower grammar still knows it."""
        _save_secrets({"DS_POSTGRES_PROD__PASSWORD": "Tr0ub4dor&3horse", "DS_PASSWORD": "flat-secret-1"})
        provider_values = {key: f"provkey-{index}-Zt7Q8Q8Q8" for index, key in enumerate(_PROVIDER_SECRET_VARS)}
        for key, value in provider_values.items():
            monkeypatch.setenv(key, value)
        text = " | ".join([
            "Tr0ub4dor&3horse",
            "flat-secret-1",
            *provider_values.values(),
            "mdb_AAAAAAAAAA.BBBBBBBBBBBBCCCC",
            f"access_token={OPAQUE_TOKEN}",
        ])
        markers = re.findall(r"\[[^\[\]\s]+\]", scrub_output(text))

        assert set(markers) == {
            "[DS_POSTGRES_PROD__PASSWORD]",
            "[DS_PASSWORD]",
            *(f"[{key}]" for key in _PROVIDER_SECRET_VARS),
            "[REDACTED_API_KEY]",
            "[REDACTED_TOKEN]",
        }
        for marker in markers:
            assert SCRUBBED_VALUE_RE.fullmatch(marker)

    def test_ordinary_output_and_code_are_left_alone(self):
        _save_secrets({
            SECRET_LABEL: ENCODED_SECRET,
            "DS_API_PROD__TOKEN": "abc123",
            "DS_MAIL_PROD__PASSWORD": "Hunter2!",
        })
        assert scrub_credentials(ORDINARY_OUTPUT) == ORDINARY_OUTPUT

    def test_output_is_unchanged_with_no_saved_secrets(self):
        text = (
            ORDINARY_OUTPUT
            + urllib.parse.quote(ENCODED_SECRET)
            + "\n"
            + base64.b64encode(ENCODED_SECRET.encode()).decode()
        )
        assert scrub_credentials(text) == text

    def test_a_value_none_of_whose_anchors_appear_is_not_run(self, pattern_runs):
        """The prefilter skips a value when no literal run of it or of its base64 is in the text."""
        _save_secrets({SECRET_LABEL: ENCODED_SECRET})
        unrelated = "SELECT count(*) FROM users WHERE active;\n" * 50
        assert scrub_credentials(unrelated) == unrelated
        assert pattern_runs[ENCODED_SECRET].calls == 0

        near_miss = "TR0UB4DOR, but not the rest of it"
        assert scrub_credentials(near_miss) == near_miss
        assert pattern_runs[ENCODED_SECRET].calls == 1

    def test_a_value_with_no_long_plain_run_is_always_run(self, pattern_runs):
        """`a!b@c#d$e%f^` has no plain run of 4 characters, and a shorter run is in most text."""
        _save_secrets({SECRET_LABEL: "a!b@c#d$e%f^"})
        scrub_credentials("SELECT 1;")
        assert pattern_runs["a!b@c#d$e%f^"].calls == 1

    def test_output_ending_partway_through_a_base64_form_is_held_back(self):
        _save_secrets({SECRET_LABEL: ENCODED_SECRET})
        printed = base64.b64encode(ENCODED_SECRET.encode()).decode()
        assert exact_secret_prefix_at_end(scrub_credentials("auth " + printed[:20])) == 20

    def test_output_starting_partway_through_a_url_encoded_form_is_held_back(self):
        _save_secrets({SECRET_LABEL: ENCODED_SECRET})
        printed = urllib.parse.quote(ENCODED_SECRET, safe="")
        assert exact_secret_suffix_at_start(scrub_credentials(printed[-20:] + " done")) == 20


class TestScrubProviderKeys:
    """Provider API keys must never reach model context (ENG-463)."""

    MINDS_KEY = "mdb_dI2OzIgO.5t7QUxqGPdgrdg2wNwvFFDTUHPyYUZRH"

    def test_provider_key_value_scrubbed_with_label(self, monkeypatch):
        """A live provider key present in env is redacted with its var label."""
        monkeypatch.setenv("ANTON_MINDS_API_KEY", self.MINDS_KEY)
        result = scrub_credentials(f'api_key = "{self.MINDS_KEY}"')
        assert self.MINDS_KEY not in result
        assert "[ANTON_MINDS_API_KEY]" in result

    def test_openai_key_value_scrubbed(self, monkeypatch):
        key = "sk-proj-abcDEF1234567890abcDEF1234567890"
        monkeypatch.setenv("OPENAI_API_KEY", key)
        result = scrub_credentials(f"OPENAI_API_KEY={key}")
        assert key not in result
        assert "[OPENAI_API_KEY]" in result

    def test_mdb_key_scrubbed_by_pattern_without_env(self):
        """A key the model already emitted (not in any env var) is caught by shape."""
        result = scrub_credentials("here it is: mdb_AAAAAAAAAA.BBBBBBBBBBBBCCCC")
        assert "mdb_AAAAAAAAAA" not in result
        assert "[REDACTED_API_KEY]" in result

    def test_sk_key_scrubbed_by_pattern(self):
        result = scrub_credentials("k1=sk-ant-api03-abcdefghij1234567890XYZ")
        assert "sk-ant-api03" not in result

    @pytest.mark.parametrize("key", GOOGLE_KEY_SAMPLES)
    def test_every_issued_google_key_format_scrubbed_bare(self, key):
        result = scrub_credentials(f"k2={key}")
        assert key not in result
        assert "[REDACTED_API_KEY]" in result

    @pytest.mark.parametrize("key", GOOGLE_KEY_SAMPLES)
    def test_every_issued_google_key_format_scrubbed_in_url_query(self, key):
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}&pageSize=100"
        result = scrub_credentials(url)
        assert key not in result
        assert "&pageSize=100" in result

    @pytest.mark.parametrize(
        "text",
        [
            '<div id="task-completion-rate-chart" class="risk-assessment-summary-panel">',
            "getElementById('disk-usage-percentage-chart')",
            ".mask-image-gradient-overlay { opacity: 0.5; }",
            "desk-booking-calendar-widget",
            "@app.get('/api/task-assignments-by-owner')",
        ],
    )
    def test_a_word_running_into_a_key_prefix_is_not_a_key(self, text):
        """`task-`, `risk-`, `desk-`: a letter before `sk-` means a longer word."""
        assert scrub_credentials(text) == text

    @pytest.mark.parametrize("before", ['"', "'", "=", "(", " ", "\n", "Bearer ", "key:"])
    def test_a_key_after_punctuation_or_space_is_still_scrubbed(self, before):
        key = "sk-proj-abcdefghij1234567890XYZ"
        result = scrub_credentials(f"{before}{key}")
        assert key not in result
        assert "[REDACTED_API_KEY]" in result

    def test_short_sk_and_base_url_left_readable(self, monkeypatch):
        """Short `sk-` strings and non-secret base URLs are not over-redacted."""
        monkeypatch.setenv("ANTON_OPENAI_BASE_URL", "https://api.openai.com/v1")
        result = scrub_credentials("sk-abc connecting to https://api.openai.com/v1")
        assert "sk-abc" in result
        assert "https://api.openai.com/v1" in result


def _b64url(payload: dict) -> str:
    """Encode ``payload`` as one unpadded base64url JWT segment."""
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()


# Built at runtime: a realistic token literal trips GitHub push protection.
FAKE_JWT = ".".join([
    _b64url({"alg": "HS256", "typ": "JWT"}),
    _b64url({"sub": "scrub-test", "exp": 0}),
    "s" * 43,
])
OPAQUE_TOKEN = "opq" + "Z9" * 15
B64URL_TOKEN = "ab_" + "Cd-Ef.Gh9" * 3
TOKEN_SAMPLES = [OPAQUE_TOKEN, B64URL_TOKEN]
BCRYPT_HASH = "$2b$12$" + "AbCdEfGhIjKlMnOpQrStUv" + "wXyZ0123456789AbCdEfGhIjKlMnOpq"
ARGON2_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$RdescudvJCsgt3ub+b+dWRWJTmaaJObG"
# A connection name with a non-ASCII letter keeps it, uppercased, in its DS_* marker.
UNICODE_DS_KEY = _slug_env_prefix("postgres", "données") + "__ACCESS_TOKEN"
UNICODE_DS_MARKER = f"[{UNICODE_DS_KEY}]"


class _CountingSearches:
    """Wrap a compiled pattern and count the characters its searches cover."""

    def __init__(self, pattern: re.Pattern[str]) -> None:
        self.pattern = pattern
        self.covered = 0

    def search(self, text: str, pos: int = 0) -> re.Match[str] | None:
        found = self.pattern.search(text, pos)
        self.covered += (found.start() if found else len(text)) - pos
        return found


class TestScrubOutput:
    """OAuth tokens a cell fetches at runtime must not leave the scratchpad readable."""

    def test_printed_token_response_keeps_keys_and_drops_the_token(self):
        """A printed token endpoint response keeps every key and loses only the token."""
        body = json.dumps({"access_token": OPAQUE_TOKEN, "token_type": "Bearer", "expires_in": 3600})
        result = scrub_output(body)
        assert OPAQUE_TOKEN not in result
        assert '"access_token": "[REDACTED_TOKEN]"' in result
        assert '"token_type": "Bearer"' in result
        assert '"expires_in": 3600' in result

    @pytest.mark.parametrize("token", TOKEN_SAMPLES)
    @pytest.mark.parametrize("key", [
        "access_token", "refresh_token", "id_token", "client_secret",
        "accessToken", "refreshToken", "clientSecret", "ACCESS_TOKEN",
        "client-secret", "access-token",
    ])
    def test_every_token_key_form_is_redacted(self, key, token):
        """Snake, camel, upper and kebab key spellings all redact their value."""
        result = scrub_output(f"{key}={token}")
        assert result == f"{key}=[REDACTED_TOKEN]"

    def test_whitespace_separated_column_is_redacted(self):
        """A printed table row with no separator but spaces still redacts the value."""
        result = scrub_output(f"access_token    {OPAQUE_TOKEN}\nexpires_in      3600")
        assert result == "access_token    [REDACTED_TOKEN]\nexpires_in      3600"

    def test_url_encoded_value_is_redacted_whole(self):
        """Percent-encoded characters do not split a token into a visible tail."""
        token = "abc%2FdefGhijk%3D" + OPAQUE_TOKEN
        assert scrub_output(f"'access_token': '{token}'") == "'access_token': '[REDACTED_TOKEN]'"

    @pytest.mark.parametrize("token", [
        "00Dxx0000001gPL!AR8AQJXg5oj8ZnP9kQ2wRtY",
        "AbCdEfGh.IjKlMnOpQr",
        "abcdefabcdefabcdefabcd",
        "Ab*Cd$Ef9Gh*Ij$Kl9Mn",
    ])
    def test_any_quoted_value_after_a_token_key_is_redacted(self, token):
        """A quoted value is a literal, so its characters never decide whether it is a token."""
        result = scrub_output(json.dumps({"access_token": token, "expires_in": 3600}))
        assert result == '{"access_token": "[REDACTED_TOKEN]", "expires_in": 3600}'

    @pytest.mark.parametrize("text", [
        f'{{"access_token": "{OPAQUE_TOKEN}',
        f"'refresh_token': '{B64URL_TOKEN}",
        f'\\"access_token\\": \\"{OPAQUE_TOKEN}',
    ])
    def test_a_quoted_value_cut_off_before_its_closing_quote_is_redacted(self, text):
        """Clipped or killed-cell output can end mid-value; the value is still redacted."""
        result = scrub_output(text)
        assert OPAQUE_TOKEN not in result and B64URL_TOKEN not in result
        assert result.endswith("[REDACTED_TOKEN]")

    def test_value_on_the_line_after_its_key_is_redacted(self):
        """Pretty-printed JSON can put a long value on the line after its key."""
        result = scrub_output(f'"access_token":\n  "{OPAQUE_TOKEN}"')
        assert result == '"access_token":\n  "[REDACTED_TOKEN]"'

    @pytest.mark.parametrize("text", [
        '{"access_token":"AAAAAAAAAAAAAAAAAAAAAA\\/BBBBBBBBBBBBB"}',
        '{"access_token":"1\\/\\/0gAbCdEfGhIjKlMnOp"}',
    ])
    def test_json_escaped_slashes_do_not_leave_a_tail(self, text):
        """Some JSON encoders escape `/`; the whole value is still one token."""
        assert scrub_output(text) == '{"access_token":"[REDACTED_TOKEN]"}'

    @pytest.mark.parametrize("text", [
        f'{{"access_token": "[X]{OPAQUE_TOKEN}"}}',
        f'{{"access_token": "${OPAQUE_TOKEN}"}}',
        f'{{"access_token": "{{a}}{OPAQUE_TOKEN}"}}',
        f'{{"access_token": "{{x}} {OPAQUE_TOKEN}"}}',
        f'{{"access_token": "[DS_A] {OPAQUE_TOKEN}"}}',
        f"'Bearer {{x}}{OPAQUE_TOKEN}'",
    ])
    def test_a_value_that_only_starts_like_a_marker_or_placeholder_is_redacted(self, text):
        """Only a whole marker or template placeholder is kept, never a prefix of a token."""
        assert OPAQUE_TOKEN not in scrub_output(text)

    def test_python_dict_repr_is_redacted(self):
        """A printed dict uses single quotes around keys and values."""
        result = scrub_output(str({"refresh_token": OPAQUE_TOKEN}))
        assert result == "{'refresh_token': '[REDACTED_TOKEN]'}"

    def test_token_inside_escaped_json_is_redacted(self):
        """A cell result is JSON inside a JSON string, so its quotes are escaped."""
        cell_json = json.dumps({"stdout": json.dumps({"access_token": OPAQUE_TOKEN})})
        result = scrub_output(cell_json)
        assert OPAQUE_TOKEN not in result
        assert '\\"access_token\\": \\"[REDACTED_TOKEN]\\"' in result

    def test_query_string_value_stops_at_the_next_parameter(self):
        """Only the token is redacted; the parameters after it stay readable."""
        result = scrub_output(f"https://idp.example/cb?access_token={OPAQUE_TOKEN}&scope=read")
        assert result == "https://idp.example/cb?access_token=[REDACTED_TOKEN]&scope=read"

    def test_unquoted_value_with_a_bang_is_redacted_whole(self):
        """Tokens such as Salesforce's contain `!`; nothing after it may show."""
        result = scrub_output("access_token=AbCdEfGhIjKlMnOpQr!secretTailXYZ")
        assert result == "access_token=[REDACTED_TOKEN]"

    @pytest.mark.parametrize("glue", ['"', "'", " ", "@", "#"], ids=["quote", "apostrophe", "space", "at", "hash"])
    @pytest.mark.parametrize("key", ["access_token", "refresh_token", "client_secret"])
    def test_an_unquoted_value_is_redacted_past_a_quote_space_at_or_hash(self, key, glue):
        """The tail after a quote, space, `@` or `#` is part of the value."""
        result = scrub_output(f"{key}={OPAQUE_TOKEN}{glue}tailGhIjKlMn")
        assert result == f"{key}=[REDACTED_TOKEN]"

    @pytest.mark.parametrize("glue", ['"', "'", " ", "@", "#"], ids=["quote", "apostrophe", "space", "at", "hash"])
    @pytest.mark.parametrize("key", ["access_token", "refresh_token", "client_secret"])
    def test_a_short_first_part_does_not_let_the_rest_of_a_token_through(self, key, glue):
        """A first part under 16 characters is judged with the rest of the value, not alone."""
        result = scrub_output(f"{key}=ab{glue}{OPAQUE_TOKEN}")
        assert result == f"{key}=[REDACTED_TOKEN]"

    @pytest.mark.parametrize("padding", ["=", "=="])
    def test_base64_padding_after_a_space_is_not_the_next_pair(self, padding):
        """`defGhIjKl=` at the end of a value is padding, not a `key=value` pair that ends it."""
        result = scrub_output(f"client_secret=abc defGhIjKlMnOp{padding}")
        assert result == "client_secret=[REDACTED_TOKEN]"

    @pytest.mark.parametrize(("text", "expected"), [
        (f'access_token=ACCESS_TOKEN_NAME"{OPAQUE_TOKEN}', "access_token=[REDACTED_TOKEN]"),
        (f"access_token=ACCESS_TOKEN_NAME {OPAQUE_TOKEN}", "access_token=[REDACTED_TOKEN]"),
        (f"client_secret=load_client_secret#{OPAQUE_TOKEN}", "client_secret=[REDACTED_TOKEN]"),
        (f"refresh_token: AccessTokenResponse@{OPAQUE_TOKEN}", "refresh_token: [REDACTED_TOKEN]"),
        (f"client_secret=b'{OPAQUE_TOKEN}", "client_secret=[REDACTED_TOKEN]"),
    ], ids=["constant-quote", "constant-space", "lowercase-hash", "type-at", "bytes-literal"])
    def test_a_code_shaped_first_part_does_not_let_the_rest_of_a_token_through(self, text, expected):
        """A name, type or constant is code only as the whole value or with code after it."""
        assert scrub_output(text) == expected

    @pytest.mark.parametrize(("text", "expected"), [
        (
            f'a=1&access_token=ab"{OPAQUE_TOKEN}&b=2',
            "a=1&access_token=[REDACTED_TOKEN]&b=2",
        ),
        (
            f'{{"user": "bob", "access_token": ab"{OPAQUE_TOKEN}, "scope": "read write", "expires_in": 3600}}',
            '{"user": "bob", "access_token": [REDACTED_TOKEN], "scope": "read write", "expires_in": 3600}',
        ),
        (
            f"{{'client_secret': ab'{OPAQUE_TOKEN}}}",
            "{'client_secret': [REDACTED_TOKEN]}",
        ),
        (
            f"refresh_token=abc {OPAQUE_TOKEN} user=bob status=ok",
            "refresh_token=[REDACTED_TOKEN] user=bob status=ok",
        ),
        (
            f'"GET /cb?code=x&access_token={OPAQUE_TOKEN} HTTP/1.1" 200 OK',
            '"GET /cb?code=x&access_token=[REDACTED_TOKEN] HTTP/1.1" 200 OK',
        ),
        (
            f'{{"url": "https://idp.example/cb?access_token={OPAQUE_TOKEN}", "status": 401}}',
            '{"url": "https://idp.example/cb?access_token=[REDACTED_TOKEN]", "status": 401}',
        ),
    ], ids=["query-string", "json-like", "dict-close", "log-pairs", "url-in-log", "url-in-json"])
    def test_an_unquoted_value_leaves_the_rest_of_its_line(self, text, expected):
        """A value ends at `&`, `,` or `;`, before the next `key=` pair, at a space
        in a URL, and before the quote or bracket that closes what holds it."""
        assert scrub_output(text) == expected

    @pytest.mark.parametrize("line", [
        "access_token = ACCESS_TOKEN_NAME or DEFAULT_TOKEN_NAME",
        "access_token = ACCESS_TOKEN_DEFAULT  # fallback when unset",
        'access_token: str = Field(default="", description="OAuth token")',
        "access_token: Optional[SecretStr] = None",
        'access_token: std::env::var("ACCESS_TOKEN").unwrap(),',
        "const accessToken = await getAccessToken(session);",
        "refresh_token: invalid_grant (Token has been expired or revoked.)",
        "access_token: expired for user bob",
        'access_token = f"{token_type} {token_value}"',
        "access_token for user Bob has expired",
    ])
    def test_code_and_prose_after_a_short_or_code_shaped_first_word_are_left_alone(self, line):
        """An operator, comment, bracket, call or keyword after the first word reads as
        code; lowercase words read as prose; a column value ends at its first space."""
        assert scrub_output(line) == line

    @pytest.mark.parametrize("password_hash", [BCRYPT_HASH, ARGON2_HASH], ids=["bcrypt", "argon2id"])
    @pytest.mark.parametrize(("template", "expected"), [
        ("client_secret={}", "client_secret=[REDACTED_TOKEN]"),
        ("client_secret: {} user=bob", "client_secret: [REDACTED_TOKEN] user=bob"),
        ("https://idp.example/cb?client_secret={}&scope=read", "https://idp.example/cb?client_secret=[REDACTED_TOKEN]&scope=read"),
        ("client_secret    {}\nexpires_in      3600", "client_secret    [REDACTED_TOKEN]\nexpires_in      3600"),
    ], ids=["key-value", "log-pairs", "query-string", "column"])
    def test_an_unquoted_password_hash_is_redacted_whole(self, template, expected, password_hash):
        """A `$` before a digit or a `$name$` chain is a hash, not a reference, and
        the commas between a PHC hash's parameters do not end it."""
        assert scrub_output(template.format(password_hash)) == expected

    @pytest.mark.parametrize(("value", "expected"), [
        ("[" + OPAQUE_TOKEN, "[REDACTED_TOKEN]"),
        ("{" + OPAQUE_TOKEN, "[REDACTED_TOKEN]"),
        ("<" + OPAQUE_TOKEN, "[REDACTED_TOKEN]"),
        ("*" + OPAQUE_TOKEN, "[REDACTED_TOKEN]"),
        ("****" + OPAQUE_TOKEN, "[REDACTED_TOKEN]"),
        ("$9" + OPAQUE_TOKEN, "[REDACTED_TOKEN]"),
        ("[DS_A]" + OPAQUE_TOKEN, "[REDACTED_TOKEN]"),
        ("[DS_A] " + OPAQUE_TOKEN, "[REDACTED_TOKEN]"),
        ("{x}" + OPAQUE_TOKEN, "[REDACTED_TOKEN]"),
        ("{x} " + OPAQUE_TOKEN, "[REDACTED_TOKEN]"),
        ("${X}" + OPAQUE_TOKEN, "[REDACTED_TOKEN]"),
        ("${OAUTH_SECRET:-" + OPAQUE_TOKEN + "}", "[REDACTED_TOKEN]}"),
        ("$(x)" + OPAQUE_TOKEN, "[REDACTED_TOKEN]"),
        ("$(x) " + OPAQUE_TOKEN + ")", "[REDACTED_TOKEN])"),
        ("$TOKEN@" + OPAQUE_TOKEN, "[REDACTED_TOKEN]"),
        ("<token>" + OPAQUE_TOKEN, "[REDACTED_TOKEN]"),
    ], ids=[
        "bracket", "brace", "angle", "asterisk", "asterisks", "dollar-digit", "marker-prefix",
        "marker-then-word", "placeholder-prefix", "placeholder-then-word", "reference-prefix",
        "long-default", "command-prefix", "command-then-word", "name-then-at", "angle-placeholder-prefix",
    ])
    @pytest.mark.parametrize("key", ["access_token", "client_secret"])
    def test_an_unquoted_secret_starting_with_a_placeholder_character_is_redacted(self, key, value, expected):
        """`$`, `[`, `{`, `<` and `*` keep a value readable only as a whole marker,
        reference, placeholder or mask, never as the start of a token. A closing
        bracket at the very end stays, as for any value."""
        assert scrub_output(f"{key}={value}") == f"{key}={expected}"

    @pytest.mark.parametrize("line", [
        "client_secret=[REDACTED_API_KEY]",
        "client_secret=[ANTON_OPENAI_API_KEY]",
        "access_token=[DS_HUBSPOT_MAIN__ACCESS_TOKEN]",
        "access_token: " + UNICODE_DS_MARKER,
        "{'client_secret': [REDACTED_API_KEY]}",
        "client_secret=$GOOGLE_CLIENT_SECRET",
        "client_secret=${GOOGLE_CLIENT_SECRET}",
        "client_secret=${GOOGLE_CLIENT_SECRET:-changeme}",
        "access_token=$(gcloud auth print-access-token)",
        "access_token=$(gcloud auth print-access-token) || exit 1",
        "const url = `https://idp.example/cb?access_token=${session.accessToken}`",
        "  client_secret: ${{ secrets.GOOGLE_CLIENT_SECRET }}",
        "  client_secret: {{ .Values.oauth.clientSecret | b64enc }}",
        "access_token={{ session.access_token }}",
        "access_token={self.access_token!r}",
        'url = f"https://idp.example/cb?access_token={self.access_token}&scope=read"',
        "access_token=<your-access-token>",
        "access_token=<SALESFORCE_ACCESS_TOKEN>",
        "access_token: <your access token>",
        "client_secret=********************",
        "client_secret=****************abcd",
        'text = f"client_secret={OPAQUE_TOKEN} and {FAKE_JWT}"',
        'assert "access_token=[REDACTED_TOKEN]" in cells[0].stdout',
        r'rest += "".join(f"access_token: {_LONG_TOKEN}\n" for _ in range(35))',
        r'result = scrub_output(f"access_token    {OPAQUE_TOKEN}\nexpires_in      3600")',
    ], ids=[
        "redacted-api-key", "provider-label", "ds-marker", "ds-marker-non-ascii", "marker-in-dict",
        "env-name", "env-braces", "env-default", "command", "command-then-operator",
        "js-template-path", "actions-secret", "helm-value-with-filter", "jinja", "format-conversion",
        "f-string-path", "angle-dashed", "angle-constant", "angle-words", "all-asterisks",
        "masked-tail", "placeholder-then-keyword", "marker-then-closing-quote",
        "placeholder-then-escaped-newline", "placeholder-then-escaped-newline-in-column",
    ])
    def test_an_unquoted_marker_reference_placeholder_or_mask_is_left_alone(self, line):
        """A value that is all marker, reference, placeholder or mask stays, and so
        does one followed by code, a closing quote or an escaped line break."""
        assert scrub_output(line) == line

    @pytest.mark.parametrize(("text", "expected"), [
        (f"client_secret={BCRYPT_HASH}", "client_secret=[REDACTED_TOKEN]"),
        (f"client_secret: {ARGON2_HASH} user=bob", "client_secret: [REDACTED_TOKEN] user=bob"),
        (f"access_token=[{OPAQUE_TOKEN}", "access_token=[REDACTED_TOKEN]"),
        (f"{{'client_secret': ab'{OPAQUE_TOKEN}}}", "{'client_secret': [REDACTED_TOKEN]}"),
        (f'a=1&access_token=ab"{OPAQUE_TOKEN}&b=2', "a=1&access_token=[REDACTED_TOKEN]&b=2"),
        (f"refresh_token=abc {OPAQUE_TOKEN} user=bob", "refresh_token=[REDACTED_TOKEN] user=bob"),
        (f"access_token    {OPAQUE_TOKEN}\nexpires_in  3600", "access_token    [REDACTED_TOKEN]\nexpires_in  3600"),
        (f'{{"access_token": "{OPAQUE_TOKEN}"}}', '{"access_token": "[REDACTED_TOKEN]"}'),
        (f"Authorization: Bearer {OPAQUE_TOKEN}", "Authorization: Bearer [REDACTED_TOKEN]"),
        (f"access_token={FAKE_JWT}&scope=read", "access_token=[REDACTED_TOKEN]&scope=read"),
        ("client_secret=sk-ant-api03-abcdefghij1234567890XYZ", "client_secret=[REDACTED_API_KEY]"),
        ("client_secret=sk-proj-" + "Kq7" * 10, "client_secret=[OPENAI_API_KEY]"),
        ("access_token=crm" + "Q7" * 10 + " user=bob", "access_token=[DS_CRM_PROD__ACCESS_TOKEN] user=bob"),
        ("access_token: pg" + "W4" * 10, "access_token: " + UNICODE_DS_MARKER),
    ], ids=[
        "bcrypt", "argon2id", "bracket-secret", "dict-close", "query-string", "log-pairs", "column",
        "json", "bearer", "jwt", "api-key-shape", "provider-label", "ds-marker", "ds-marker-non-ascii",
    ])
    def test_scrubbing_scrubbed_output_again_changes_nothing(self, monkeypatch, text, expected):
        """Every marker the scrub writes, for a configured secret, a provider key, a key
        shape or a token value, reads as a marker to the unquoted pass after it and when
        the output is scrubbed again."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-" + "Kq7" * 10)
        _DS_SECRET_VARS.add("DS_CRM_PROD__ACCESS_TOKEN")
        _DS_SECRET_VARS.add(UNICODE_DS_KEY)
        set_ds_env_values({"DS_CRM_PROD__ACCESS_TOKEN": "crm" + "Q7" * 10, UNICODE_DS_KEY: "pg" + "W4" * 10})

        once = scrub_output(text)

        assert once == expected
        assert scrub_output(once) == once

    def test_the_unquoted_pass_searches_each_part_of_a_line_once(self, monkeypatch):
        """A value that stays readable runs on over the values after it, so a fresh
        search for each value's end would rescan the rest of the line every time."""
        line = "access_token: abc and " * 5000
        counters = []
        for name in ("_WORD_VALUE_END_PATTERN", "_SPACED_VALUE_END_PATTERN"):
            counter = _CountingSearches(getattr(datasources, name))
            monkeypatch.setattr(datasources, name, counter)
            counters.append(counter)

        assert scrub_output(line) == line
        assert sum(counter.covered for counter in counters) <= 2 * len(line)

    @pytest.mark.parametrize("line", [
        "access_token: abc and " * 5000,
        "access_token: {x} and " * 5000,
        "access_token: x" + " " * 50_000 + "y",
        "access_token: abc and " * 2000 + ")" * 50_000,
    ], ids=["readable-values", "readable-placeholders", "long-space-run", "long-trailer-after-readable-values"])
    def test_a_long_line_scrubs_in_well_under_a_second(self, line):
        """Each line took 3 to 12 seconds when each value rescanned the rest of its
        line, a run of spaces was retried from each space, or each value trimmed
        the same trailer again; scanned once, each takes milliseconds."""
        started = time.perf_counter()
        scrub_output(line)
        assert time.perf_counter() - started < 1.0

    @pytest.mark.parametrize("token", TOKEN_SAMPLES)
    @pytest.mark.parametrize("word", ["Bearer", "bearer", "BEARER"])
    def test_bearer_value_is_redacted_and_the_word_kept(self, word, token):
        """Any casing of `Bearer` redacts the value and keeps the word."""
        result = scrub_output(f"Authorization: {word} {token}")
        assert result == f"Authorization: {word} [REDACTED_TOKEN]"

    @pytest.mark.parametrize("text, expected", [
        ("{'Authorization': 'Bearer AbCdEfGh.IjKlMnOpQr'}",
         "{'Authorization': 'Bearer [REDACTED_TOKEN]'}"),
        ("Authorization: Bearer 00Dxx0000001gPL!AR8AQJXg5oj8ZnP9kQ2wRtY",
         "Authorization: Bearer [REDACTED_TOKEN]"),
        ("Bearer\n" + OPAQUE_TOKEN, "Bearer\n[REDACTED_TOKEN]"),
        ("Bearer\r\n" + OPAQUE_TOKEN, "Bearer\r\n[REDACTED_TOKEN]"),
        ('{"access_token":"AAAA\\u002fBBBBBBBBBBBBBBBBBBBBB"}', '{"access_token":"[REDACTED_TOKEN]"}'),
        ("client_secret = '$2b$12$AbCdEfGhIjKlMnOpQrStUv'", "client_secret = '[REDACTED_TOKEN]'"),
        ("access_token: Abc.Def.Ghi_jklmnopq", "access_token: [REDACTED_TOKEN]"),
        ("access_token=Ab9CdEfGhIjKlMnOpQr", "access_token=[REDACTED_TOKEN]"),
        ("access_token=ABCDEFGHIJKLMNOPQRST", "access_token=[REDACTED_TOKEN]"),
        ("access_token=AB_cdEFgh_IJklMNop", "access_token=[REDACTED_TOKEN]"),
        ("access_token=AbCdEfGhIjKlMnOpQr", "access_token=[REDACTED_TOKEN]"),
        ("refresh_token=A1B2C3D4_E5F6G7H8_I9J0", "refresh_token=[REDACTED_TOKEN]"),
        ("access_token=AB12_CD34_EF56_GH78", "access_token=[REDACTED_TOKEN]"),
    ])
    def test_tokens_that_resemble_code_are_still_redacted(self, text, expected):
        """Printed headers and token_type/token pairs are output, not source code."""
        assert scrub_output(text) == expected

    def test_bare_jwt_is_redacted(self):
        """A JWT is caught by its shape with no key or word in front of it."""
        result = scrub_output(f"token is {FAKE_JWT} now")
        assert result == "token is [REDACTED_TOKEN] now"

    def test_google_oauth_token_is_redacted(self):
        """Google `ya29.` access tokens are caught by their prefix."""
        token = "ya29." + "a0B1" * 8
        assert scrub_output(f"got {token}") == "got [REDACTED_TOKEN]"

    @pytest.mark.parametrize("text", [
        "send a Bearer token in the Authorization header",
        "access_token=short",
        '{"access_token": null, "token_type": "Bearer"}',
        "access_token=[DS_CRM_PROD__TOKEN]",
        '{"access_token": "[DS_CRM_PROD__ACCESS_TOKEN]"}',
        "valid_token=" + OPAQUE_TOKEN,
        "the bearer tokens/credentials are sent",
        "token_type    Bearer\nrefresh_token_expires_in    7200",
        "uses Bearer token-based-authentication-scheme",
        '"Bearer token is missing, sign in again"',
    ])
    def test_prose_short_values_and_markers_are_left_alone(self, text):
        """Prose, short values and existing scrub markers stay as they are."""
        assert scrub_output(text) == text

    @pytest.mark.parametrize("line", [
        'const accessToken = window.localStorage.getItem("t")',
        "refreshToken: response.data.refresh_token,",
        "access_token = self.config.access_token_value",
        "client_secret = load_client_secret_from_vault()",
        "accessToken = fetchAccessTokenNow(session)",
        "headers = {'Authorization': 'Bearer ' + access_token}",
        'headers = {"Authorization": f"Bearer {self.config.access_token_value}"}',
        "const auth = `Bearer ${process.env.API_ACCESS_TOKEN}`",
        "if access_token !=expected_access_token_value:",
        "def refresh(access_token: AccessTokenResponse) -> None:",
        "client_secret = CLIENT_SECRET_ENV_VAR",
        "    access_token=ACCESS_TOKEN_DEFAULT_NAME,",
        """print("access_token='" + access_token[:6] + "...'")""",
        """const body = '{"access_token": "' + accessToken + '"}'""",
        'print("access_token: ", response.headers.get("x-request-id"))',
        'log("client_secret=" + mask(client_secret) + " for tenant")',
        r'body = "{\"access_token\": \"" + access_token_value + "\"}"',
        'x = "access_token=\'" if quoted_output else "access_token="',
        r'"{\"client_secret\": \"%s\"}" % secret_name',
        'client_secret: "{{ .Values.oauth.clientSecret }}"',
        'client_secret: "${{ secrets.GOOGLE_CLIENT_SECRET }}"',
        '"access_token": "{{ access_token }}"',
        '$url = "x?access_token=" . $accessTokenValue . "&scope=a";',
        "local url = 'x?access_token=' .. access_token_value .. '&scope=a'",
        "SELECT 'access_token=' || access_token_value || '&x' FROM tokens",
        "body = b'client_secret=' + secret_bytes + b'&grant_type=client_credentials'",
        "url = f'https://x?access_token=' + access_token_value + '&scope=a'",
        'if "access_token=" in response_body:',
        'if "access_token=" not in body and "client_secret=" in other_body:',
    ])
    def test_source_code_the_model_reads_is_left_alone(self, line):
        """Code the model reads and may write back is never rewritten."""
        assert scrub_output(line) == line

    @pytest.mark.parametrize("value", [
        "abc'defGhIjKlMnOpQr",
        "abc defGhIjKlMnOpQrStu",
        'abc"defGhIjKlMnOpQrStu',
        "abc\\defGhIjKlMnOpQrStu",
    ], ids=["apostrophe", "space", "quote", "backslash"])
    @pytest.mark.parametrize("encode", [
        json.dumps,
        str,
        lambda body: json.dumps({"stdout": json.dumps(body)}),
        lambda body: repr(json.dumps(body)),
        lambda body: repr(json.dumps(body).encode()),
        lambda body: repr(str(body)),
    ], ids=["json", "dict-repr", "json-in-json", "repr-of-json", "bytes-repr-of-json", "repr-of-dict-repr"])
    def test_a_quoted_value_is_redacted_up_to_its_closing_quote(self, encode, value):
        """A value is a literal up to its real closing quote, whatever it holds,
        however many times its encoder escaped it."""
        result = scrub_output(encode({"client_secret": value, "expires_in": 3600}))

        for start in range(len(value) - 7):
            piece = value[start : start + 8]
            if piece.isalnum():
                assert piece not in result
        assert "client_secret" in result
        assert "[REDACTED_TOKEN]" in result
        assert "expires_in" in result and "3600" in result

    @pytest.mark.parametrize("text", [
        """client_secret='abc def\\'GhIjKlMnOpQr'""",
        'access_token="abc\\"defGhIjKlMnOpQrStu"',
        'access_token="abc defGhIjKlMnOpQrStu"',
    ], ids=["single-quoted", "double-quoted", "space"])
    def test_a_quoted_value_in_key_value_form_is_redacted_whole(self, text):
        """The .env and shell form escapes its quote with a backslash."""
        result = scrub_output(text)
        assert "GhIjKlMn" not in result
        assert result.endswith("[REDACTED_TOKEN]" + text[-1])

    @pytest.mark.parametrize("value", ["abc'defGhIjKlMnOpQr", "abc defGhIjKlMnOpQrStu", 'abc"defGhIjKlMnOpQrStu'])
    def test_a_value_with_a_quote_or_space_cut_off_before_its_closing_quote_is_redacted(self, value):
        """Clipped output ends mid-value; the value is redacted to the end."""
        result = scrub_output(json.dumps({"client_secret": value})[:-2])
        assert "GhIjKlMn" not in result
        assert result.endswith("[REDACTED_TOKEN]")

    @pytest.mark.parametrize("text", [
        """{"client_secret":"abc'defGhIjKlMnOpQr"}""",
        '{"access_token":"abc defGhIjKlMnOpQrStu"}',
        '{"access_token":"abc\\"defGhIjKlMnOpQrStu"}',
    ], ids=["apostrophe", "space", "quote"])
    def test_the_reported_examples_are_redacted_as_written(self, text):
        """Compact JSON, exactly as the report printed it."""
        assert scrub_output(text) == text.split(":")[0] + ':"[REDACTED_TOKEN]"}'

    @pytest.mark.parametrize(("text", "expected"), [
        ("""{"client_secret":"abc',defGhIjKlMnOpQrStu"}""", '{"client_secret":"[REDACTED_TOKEN]"}'),
        ('{"client_secret":"+ AbCdEfGhIjKlMnOpQrSt"}', '{"client_secret":"[REDACTED_TOKEN]"}'),
        ('client_secret="+ AbCdEfGhIjKlMnOpQrSt"', 'client_secret="[REDACTED_TOKEN]"'),
        (
            """It's here: {"client_secret":"abc',defGhIjKlMnOpQrStu"}""",
            """It's here: {"client_secret":"[REDACTED_TOKEN]"}""",
        ),
        (
            """users' tokens: {"client_secret":"abc', defGhIjKlMnOpQrStu"}""",
            """users' tokens: {"client_secret":"[REDACTED_TOKEN]"}""",
        ),
        (
            """Plan B's tokens: {"client_secret":"abc',defGhIjKlMnOpQrStu"}""",
            """Plan B's tokens: {"client_secret":"[REDACTED_TOKEN]"}""",
        ),
        (
            """{"msg":"rock 'n roll","client_secret":"abc',defGhIjKlMnOpQrStu"}""",
            """{"msg":"rock 'n roll","client_secret":"[REDACTED_TOKEN]"}""",
        ),
        (
            """{"msg":"90's","client_secret":"abc',defGhIjKlMnOpQrStu"}""",
            """{"msg":"90's","client_secret":"[REDACTED_TOKEN]"}""",
        ),
        (
            repr({"note": "users' data", "client_secret": "abc', defGhIjKlMnOpQrStu"}),
            """{'note': "users' data", 'client_secret': "[REDACTED_TOKEN]"}""",
        ),
    ], ids=[
        "quote-and-comma", "leading-plus", "key-value-leading-plus", "after-prose-apostrophe",
        "after-plural-possessive", "after-capital-letter-possessive", "apostrophe-in-earlier-value", "digit-apostrophe-in-earlier-value",
        "apostrophe-in-earlier-repr-value",
    ])
    def test_a_value_holding_code_operators_is_redacted_whole(self, text, expected):
        """Quotes and operators inside a literal the key opened are part of the value, not code."""
        assert scrub_output(text) == expected

    def test_a_value_after_one_that_ran_past_its_end_is_still_redacted(self):
        """A value ending in a backslash runs into the next key; what follows on the line is data."""
        text = json.dumps({"client_secret": "AbCdEfGhIjKlMnOp\\", "access_token": "+ defGhIjKlMnOpQrStu"})
        result = scrub_output(text)
        assert "AbCdEfGh" not in result
        assert "GhIjKlMn" not in result

    def test_a_value_nested_five_json_levels_deep_is_redacted(self):
        """Thirty-one backslashes before the quote is five levels of JSON in JSON."""
        text = {"access_token": "abc defGhIjKlMnOpQrStuVw"}
        for _ in range(5):
            text = json.dumps(text)
        result = scrub_output(text)
        assert "GhIjKlMn" not in result
        assert "[REDACTED_TOKEN]" in result

    @pytest.mark.parametrize("closing", ['"', ""], ids=["closed", "cut-off"])
    @pytest.mark.parametrize("tail", ["", " tailGhIjKlMnOpQrStu", '\\"tailGhIjKlMnOpQrStu'], ids=["none", "space", "escaped-quote"])
    def test_a_value_longer_than_the_scan_is_redacted_whole(self, closing, tail):
        """Past the scan limit, the rest of a long value is still redacted."""
        result = scrub_output('"access_token": "' + "A" * 20_000 + tail + closing)
        assert "AAAAAAAA" not in result
        assert "GhIjKlMn" not in result
        assert result.startswith('"access_token": "[REDACTED_TOKEN]')

    def test_a_value_ending_in_a_backslash_leaves_nothing_of_it(self):
        """JSON's escaped trailing backslash reads like a bytes repr's escaped
        quote; the scrub takes the reading that cannot leak, so the separator
        after the value may go with it."""
        result = scrub_output(json.dumps({"client_secret": "AbCdEfGhIjKlMnOp\\", "expires_in": 3600}))
        assert "AbCdEfGh" not in result
        assert result.startswith('{"client_secret": "[REDACTED_TOKEN]')

    def test_a_value_nested_two_escapes_deep_is_redacted(self):
        """The repr of a cell result escapes the inner quotes twice."""
        text = repr(json.dumps({"stdout": json.dumps({"access_token": OPAQUE_TOKEN})}))
        assert OPAQUE_TOKEN not in scrub_output(text)

    def test_a_value_starting_with_a_plus_is_still_a_value(self):
        """Standard base64 can start with `+`; only `+ ` reads as code."""
        result = scrub_output('{"client_secret": "+AbCd EfGhIjKlMnOpQrSt"}')
        assert result == '{"client_secret": "[REDACTED_TOKEN]"}'

    def test_marker_is_one_the_connect_flow_refuses_to_save(self):
        """An echoed marker is never saved as a real credential."""
        assert SCRUBBED_VALUE_RE.match("[REDACTED_TOKEN]")

    def test_known_credentials_are_still_scrubbed(self):
        """scrub_output keeps everything scrub_credentials already redacts."""
        result = scrub_output("key mdb_AAAAAAAAAA.BBBBBBBBBBBBCCCC")
        assert "[REDACTED_API_KEY]" in result

    def test_scrub_credentials_leaves_tokens_for_input_paths(self):
        """Input paths keep the narrower filter, so pasted connect credentials still save."""
        text = f"client_secret={OPAQUE_TOKEN} and {FAKE_JWT}"
        assert scrub_credentials(text) == text


class TestScrubUserInput:
    """User messages are scrubbed before entering session history (ENG-583)."""

    def test_string_input_key_redacted(self):
        result = _scrub_user_input(
            "use this key: sk-ant-api03-abcdefghij1234567890XYZ"
        )
        assert "sk-ant-api03" not in result
        assert "[REDACTED_API_KEY]" in result

    def test_plain_string_unchanged(self):
        text = "please connect me to my staging database"
        assert _scrub_user_input(text) == text

    def test_text_blocks_scrubbed_other_blocks_untouched(self):
        blocks = [
            {"type": "text", "text": "key is mdb_AAAAAAAAAA.BBBBBBBBBBBBCCCC"},
            {"type": "image", "source": {"type": "base64", "data": "aGk="}},
        ]
        result = _scrub_user_input(blocks)
        assert "mdb_AAAAAAAAAA" not in result[0]["text"]
        assert "[REDACTED_API_KEY]" in result[0]["text"]
        assert result[1] is blocks[1]

    def test_known_secret_env_value_redacted_with_label(self, monkeypatch):
        """A pasted value matching a stored provider secret gets its var label."""
        key = "sk-proj-abcDEF1234567890abcDEF1234567890"
        monkeypatch.setenv("OPENAI_API_KEY", key)
        result = _scrub_user_input(f"my key is {key}")
        assert key not in result
        assert "[OPENAI_API_KEY]" in result


class TestCustomEngineRegistration:
    """ENG-688: connections of engines not in the registry (custom engines,
    connector-spec saves) must register their fields so non-secret values
    (base_url, host, ...) stay readable instead of leaking as markers."""

    def _vault(self, tmp_path):
        from anton.core.datasources.data_vault import LocalDataVault

        return LocalDataVault(tmp_path / "vault")

    def test_custom_engine_base_url_readable_secret_scrubbed(self, tmp_path):
        from anton.utils.datasources import restore_namespaced_env

        vault = self._vault(tmp_path)
        vault.save(
            "acme_crm", "prod",
            {"base_url": "https://api.acme-crm.example", "token": "tok_1234567890abcdef"},
            secure_keys=["token"],
        )
        restore_namespaced_env(vault)

        result = scrub_credentials(
            "GET https://api.acme-crm.example failed with token tok_1234567890abcdef"
        )
        assert "https://api.acme-crm.example" in result
        assert "tok_1234567890abcdef" not in result
        assert "[DS_ACME_CRM_PROD__TOKEN]" in result

    def test_custom_engine_without_secure_keys_uses_name_heuristic(self, tmp_path):
        from anton.utils.datasources import restore_namespaced_env

        vault = self._vault(tmp_path)
        vault.save(
            "acme_crm", "legacy",
            {"base_url": "https://legacy.acme-crm.example", "api_key": "ak_1234567890abcdef"},
        )
        restore_namespaced_env(vault)

        result = scrub_credentials(
            "base https://legacy.acme-crm.example key ak_1234567890abcdef"
        )
        assert "https://legacy.acme-crm.example" in result
        assert "ak_1234567890abcdef" not in result
        assert "[DS_ACME_CRM_LEGACY__API_KEY]" in result

    def test_custom_engine_legacy_passphrase_is_scrubbed(self, tmp_path):
        from anton.utils.datasources import restore_namespaced_env

        vault = self._vault(tmp_path)
        passphrase = "correct horse battery staple"
        vault.save(
            "acme_crm",
            "legacy",
            {
                "base_url": "https://legacy.acme-crm.example",
                "passphrase": passphrase,
            },
        )
        restore_namespaced_env(vault)

        result = scrub_credentials(
            f"base https://legacy.acme-crm.example passphrase {passphrase}"
        )
        assert "https://legacy.acme-crm.example" in result
        assert passphrase not in result
        assert "[DS_ACME_CRM_LEGACY__PASSPHRASE]" in result


class TestConcurrentTurnIsolation:
    """The ticket's own acceptance criterion: two turns running concurrently
    on divergent vaults must never cross-contaminate scrub state."""

    def _vault(self, tmp_path, subdir: str):
        from anton.core.datasources.data_vault import LocalDataVault

        return LocalDataVault(tmp_path / subdir)

    async def test_two_concurrent_turns_scrub_only_their_own_secret(self, tmp_path):
        import asyncio

        from anton.utils.datasources import restore_namespaced_env

        vault_a = self._vault(tmp_path, "vault_a")
        vault_a.save("acme_crm", "prod", {"token": "secret-for-turn-a"}, secure_keys=["token"])

        vault_b = self._vault(tmp_path, "vault_b")
        vault_b.save("acme_crm", "prod", {"token": "secret-for-turn-b"}, secure_keys=["token"])

        async def turn(vault, own_secret: str, other_secret: str) -> str:
            restore_namespaced_env(vault)
            await asyncio.sleep(0)  # yield, so the other turn's setup interleaves here
            return scrub_credentials(f"own={own_secret} other={other_secret}")

        result_a, result_b = await asyncio.gather(
            turn(vault_a, "secret-for-turn-a", "secret-for-turn-b"),
            turn(vault_b, "secret-for-turn-b", "secret-for-turn-a"),
        )

        assert "secret-for-turn-a" not in result_a
        assert "[DS_ACME_CRM_PROD__TOKEN]" in result_a
        assert "secret-for-turn-b" not in result_b
        assert "[DS_ACME_CRM_PROD__TOKEN]" in result_b


class TestTurnKeyGoogleAdsDeveloperToken:
    def test_developer_token_is_scrubbed_and_login_customer_id_is_not(self, monkeypatch):
        from anton.core.datasources.data_vault import TurnKeyDataVault
        from anton.utils.datasources import restore_namespaced_env

        def fake(url, api_key, *, method="GET", payload=None, verify=True, timeout=30):
            return b'{"access_token": "ya29.live-token", "developer_token": "ABCDE-FGHIJ-KLMNO", "login_customer_id": "1234567890"}'

        monkeypatch.setattr("anton.minds_client.minds_request", fake)
        vault = TurnKeyDataVault({"turn_key": "tk_abc", "connections": [{"engine": "google_ads", "name": "primary"}]})
        restore_namespaced_env(vault)

        result = scrub_credentials("dev token ABCDE-FGHIJ-KLMNO for customer 1234567890")
        assert "ABCDE-FGHIJ-KLMNO" not in result
        assert "[DS_GOOGLE_ADS_PRIMARY__DEVELOPER_TOKEN]" in result
        assert "1234567890" in result


class TestOAuthEngineRegistryCollision:
    """A cloud gmail OAuth connection shares its engine name with the
    registry's legacy IMAP gmail connector (datasources.md) — its OAuth
    fields (access_token, account_email, ...) must classify against the
    vault's own secure_keys, not the IMAP entry's (email, app_password)."""

    def test_gmail_oauth_fields_classify_against_the_vault_not_the_imap_registry_entry(self, monkeypatch):
        from anton.core.datasources.data_vault import TurnKeyDataVault
        from anton.utils.datasources import restore_namespaced_env

        # Long enough (> 8 chars) to actually exercise the coarse "unknown
        # DS_* var" bucket if misclassified — a short value dodges it either
        # way and would pass this test for the wrong reason.
        email = "someone@example.com"

        def fake(url, api_key, *, method="GET", payload=None, verify=True, timeout=30):
            return f'{{"access_token": "ya29.live-token", "account_email": "{email}", "scope": "gmail.readonly"}}'.encode()

        monkeypatch.setattr("anton.minds_client.minds_request", fake)
        vault = TurnKeyDataVault({"turn_key": "tk_abc", "connections": [{"engine": "gmail", "name": "primary"}]})
        restore_namespaced_env(vault)

        result = scrub_credentials(f"token ya29.live-token for {email} scope gmail.readonly")
        assert "ya29.live-token" not in result
        assert "[DS_GMAIL_PRIMARY__ACCESS_TOKEN]" in result
        # Non-secret metadata must stay visible — before the fix, both fell
        # into the coarse "unknown DS_* var, len > 8" bucket instead, since
        # the IMAP registry entry's fields (email, app_password) never
        # registered these as known.
        assert email in result
        assert "gmail.readonly" in result


class TestTurnMapIsAuthoritative:
    """Once a turn has a value map, os.environ is not consulted for a missing
    key: another turn may hold a different value under the same name."""

    def test_a_missing_key_is_not_read_from_environ(self, monkeypatch):
        from anton.utils.datasources import set_ds_env_values

        key = "DS_ACME_CRM_PROD__TOKEN"
        _DS_SECRET_VARS.add(key)
        _DS_KNOWN_VARS.add(key)

        # Another turn's value for the same var name, still in the process env.
        monkeypatch.setenv(key, "another-turns-token")
        # This turn has a map, and the key is not in it (its lookup failed).
        set_ds_env_values({})

        result = scrub_credentials("log line mentioning another-turns-token")

        # Redacting here would both miss this turn's secret and confirm the
        # other turn's value by substituting it.
        assert "another-turns-token" in result

    def test_the_coarse_net_still_reads_environ_after_a_map_opens(self, monkeypatch):
        """Only the labeled lookup is gated. A DS_* var the vault never
        registered still gets caught by the unknown-DS_* net, which is what
        keeps an operator-exported credential out of model context.
        """
        from anton.utils.datasources import set_ds_env_values

        # In neither registry, the only reachable coarse-net state: every
        # register_* helper adds to KNOWN before SECRET.
        monkeypatch.setenv("DS_SHELL_SET__PASSWORD", "shell-value")

        set_ds_env_values({})
        result = scrub_credentials("pw shell-value")

        assert "shell-value" not in result
        assert "[DS_SHELL_SET__PASSWORD]" in result

    def test_a_caller_that_never_set_a_map_still_reads_environ(self, monkeypatch):
        """The CLI and os.environ-based tests keep working unchanged."""
        _DS_SECRET_VARS.add("DS_MANUAL__PASSWORD")
        monkeypatch.setenv("DS_MANUAL__PASSWORD", "shell-set-secret")

        result = scrub_credentials("pw shell-set-secret")

        assert "shell-set-secret" not in result
        assert "[DS_MANUAL__PASSWORD]" in result


class TestRegistrationFromInsideAToolCall:
    """Every tool call runs in its own task (`asyncio.create_task` in
    session.py), and that copies the context. A connect made mid-turn must
    still be scrubbed from the rest of the turn's output, so the per-turn
    state has to be mutated in place rather than reassigned.
    """

    async def test_a_mid_turn_connect_is_scrubbed_by_its_own_turn(self, tmp_path):
        from anton.core.datasources.data_vault import LocalDataVault
        from anton.utils.datasources import begin_ds_turn_scope, restore_namespaced_env

        vault = LocalDataVault(vault_dir=tmp_path / "vault")

        # What turn()/turn_stream() do before dispatching any tool.
        begin_ds_turn_scope()

        # Mid-turn the user hands the agent a password; the tool handler that
        # saves it runs in its own task.
        async def connect_tool() -> None:
            vault.save(
                "postgres", "prod", {"password": "given-mid-turn"}, secure_keys=["password"]
            )
            restore_namespaced_env(vault)

        await asyncio.create_task(connect_tool())

        # Back in the turn's own context, which is what scrubs cell output.
        assert "DS_POSTGRES_PROD__PASSWORD" in _DS_SECRET_VARS
        result = scrub_credentials("traceback: password=given-mid-turn")
        assert "given-mid-turn" not in result
        assert "[DS_POSTGRES_PROD__PASSWORD]" in result

    async def test_a_concurrent_turn_still_cannot_see_it(self, tmp_path):
        """Mutating in place must not reintroduce cross-turn bleed: a sibling
        turn opens its own scope and sees only its own vault."""
        from anton.core.datasources.data_vault import LocalDataVault
        from anton.utils.datasources import restore_namespaced_env, set_ds_env_values

        vault_a = LocalDataVault(vault_dir=tmp_path / "a")
        vault_a.save("postgres", "prod", {"password": "turn-a-pw"}, secure_keys=["password"])
        vault_b = LocalDataVault(vault_dir=tmp_path / "b")

        async def turn(vault) -> str:
            restore_namespaced_env(vault)
            await asyncio.sleep(0)
            return scrub_credentials("saw turn-a-pw")

        # Each turn runs as its own task, the way the streaming producers do.
        out_a, out_b = await asyncio.gather(
            asyncio.create_task(turn(vault_a)),
            asyncio.create_task(turn(vault_b)),
        )

        assert "turn-a-pw" not in out_a
        # Turn B has no such connection, so it has no value to redact with and
        # must not have acquired turn A's.
        assert "turn-a-pw" in out_b


class TestARemovedValueStaysHiddenForItsTurn:
    """ENG-3305: a pad keeps the DS_* values it started with until the turn
    ends. A connection removed or edited mid-turn, followed by a rebuild in
    the same turn, must not let the old value print in clear."""

    @pytest.mark.parametrize("change", ["delete", "edit"])
    async def test_a_value_held_at_turn_start_is_hidden_after_a_mid_turn_rebuild(
        self, tmp_path, change
    ):
        from anton.core.datasources.data_vault import LocalDataVault
        from anton.core.datasources.datasource_registry import DatasourceRegistry
        from anton.utils.datasources import (
            begin_ds_turn_scope,
            register_secret_vars,
            restore_namespaced_env,
            save_connection,
        )

        postgres = DatasourceRegistry().get("postgres")
        vault = LocalDataVault(vault_dir=tmp_path / "vault")
        vault.save("postgres", "x", {"host": "db", "password": "old-x-password"})

        # What turn_stream() does before dispatching any tool.
        begin_ds_turn_scope()
        restore_namespaced_env(vault)

        # In another tab, the user removes or edits X while the turn runs.
        if change == "delete":
            vault.delete("postgres", "x")
        else:
            vault.save("postgres", "x", {"host": "db", "password": "new-x-password"})

        # The agent connects another source: the connect tool's test, then its save.
        async def connect_tool() -> None:
            credentials = {"host": "db", "password": "y-password"}
            register_secret_vars(postgres)
            set_ds_env_values({f"DS_{k.upper()}": v for k, v in credentials.items()})
            restore_namespaced_env(vault)
            save_connection(vault, postgres, "y", credentials)

        await asyncio.create_task(connect_tool())

        result = scrub_output("a cell printed old-x-password")
        assert "old-x-password" not in result
        assert "[DS_POSTGRES_X__PASSWORD]" in result

        # The next turn starts from its own vault only.
        begin_ds_turn_scope()
        restore_namespaced_env(vault)
        assert "old-x-password" in scrub_output("a cell printed old-x-password")


class TestTheTurnBoundaryOpensTheScope:
    """A host that registers nothing up front still has to scrub a credential
    the user hands over mid-turn. The turn boundary is what guarantees it."""

    async def test_without_a_scope_a_mid_turn_connect_is_lost(self, tmp_path):
        """Documents why the turn boundary has to open one: the tool task's
        writes go into a context copy that is discarded."""
        from anton.core.datasources.data_vault import LocalDataVault
        from anton.utils.datasources import restore_namespaced_env

        vault = LocalDataVault(vault_dir=tmp_path / "vault")

        async def connect_tool() -> None:
            vault.save("postgres", "p", {"password": "no-scope-pw"}, secure_keys=["password"])
            restore_namespaced_env(vault)

        await asyncio.create_task(connect_tool())

        assert "no-scope-pw" in scrub_credentials("pw no-scope-pw")

    async def test_the_turn_entry_points_open_one(self):
        """turn_stream() — the only turn entry point — must call it."""
        import inspect

        from anton.core.session import ChatSession

        assert "_open_ds_turn_scope()" in inspect.getsource(ChatSession.turn_stream)

    async def test_opening_a_scope_does_not_change_what_a_reader_sees(self, monkeypatch):
        """It seeds from the ambient DS_*, so a host relying on the os.environ
        fallback keeps scrubbing exactly as before."""
        from anton.utils.datasources import begin_ds_turn_scope

        _DS_SECRET_VARS.add("DS_AMBIENT__PASSWORD")
        _DS_KNOWN_VARS.add("DS_AMBIENT__PASSWORD")
        monkeypatch.setenv("DS_AMBIENT__PASSWORD", "ambient-value")

        begin_ds_turn_scope()

        assert "ambient-value" not in scrub_credentials("pw ambient-value")


class TestTheTurnRebuildsFromItsOwnVault:
    """A host may build the session off the event loop thread — the pod uses
    run_in_executor so its heartbeat keeps firing. ContextVar writes made in
    that worker never reach the turn's task, so the turn cannot trust them.
    """

    async def test_state_built_in_an_executor_is_rebuilt_by_the_turn(self, tmp_path):
        from unittest.mock import MagicMock

        from anton.core.datasources.data_vault import LocalDataVault
        from anton.core.session import ChatSession, ChatSessionConfig
        from anton.utils.datasources import restore_namespaced_env
        from tests.conftest import make_mock_llm

        vault = LocalDataVault(vault_dir=tmp_path / "vault")
        vault.save(
            "gmail", "primary", {"access_token": "ya29.live-token"},
            secure_keys=["access_token"],
        )

        session = ChatSession(
            ChatSessionConfig(llm_client=make_mock_llm(), data_vault=vault)
        )

        # The host's registration happens in a worker thread and is lost.
        await asyncio.get_running_loop().run_in_executor(
            None, restore_namespaced_env, vault
        )
        assert "ya29.live-token" in scrub_credentials("printed ya29.live-token")

        # The turn boundary rebuilds it from the session's own vault.
        session._open_ds_turn_scope()

        result = scrub_credentials("printed ya29.live-token")
        assert "ya29.live-token" not in result
        assert "[DS_GMAIL_PRIMARY__ACCESS_TOKEN]" in result

    async def test_a_session_without_a_vault_still_opens_a_scope(self):
        from anton.core.session import ChatSession, ChatSessionConfig
        from tests.conftest import make_mock_llm

        session = ChatSession(ChatSessionConfig(llm_client=make_mock_llm()))
        session._open_ds_turn_scope()  # must not raise


class TestAFailedRebuildKeepsTheOldState:
    """Emptying the registries before rebuilding means a raise mid-rebuild
    would leave the turn scrubbing against nothing, which reads as success."""

    def test_a_raise_mid_rebuild_restores_the_previous_state(self, tmp_path, monkeypatch):
        from anton.core.datasources.data_vault import LocalDataVault
        from anton.utils.datasources import restore_namespaced_env

        vault = LocalDataVault(vault_dir=tmp_path / "vault")
        vault.save("postgres", "prod", {"password": "known-secret"}, secure_keys=["password"])
        restore_namespaced_env(vault)
        assert "known-secret" not in scrub_credentials("pw known-secret")

        # A user-written datasources.md that cannot be parsed, say.
        monkeypatch.setattr(
            "anton.utils.datasources.DatasourceRegistry",
            lambda *a, **k: (_ for _ in ()).throw(OSError("unreadable")),
        )
        with pytest.raises(OSError):
            restore_namespaced_env(vault)

        # The turn keeps what it had rather than silently scrubbing nothing.
        assert "known-secret" not in scrub_credentials("pw known-secret")
        assert "[DS_POSTGRES_PROD__PASSWORD]" in scrub_credentials("pw known-secret")


class TestOneBadConnectionCannotBlindTheWholeTurn:
    """A pad's env is built per connection and tolerates a bad record, so the
    registry rebuild has to degrade the same way. All-or-nothing there leaves
    the pad holding credentials this turn has no way to scrub."""

    def test_a_connection_that_cannot_be_classified_leaves_the_others_scrubbed(
        self, tmp_path, monkeypatch
    ):
        from anton.core.datasources.data_vault import LocalDataVault
        from anton.utils.datasources import restore_namespaced_env

        vault = LocalDataVault(vault_dir=tmp_path / "vault")
        vault.save(
            "postgres", "broken", {"password": "broken-secret"}, secure_keys=["password"]
        )
        vault.save(
            "mysql", "healthy", {"password": "healthy-secret"}, secure_keys=["password"]
        )

        real_read_record = vault.read_record

        def flaky_read_record(engine, name, *args, **kwargs):
            if name == "broken":
                raise OSError("unreadable record")
            return real_read_record(engine, name, *args, **kwargs)

        monkeypatch.setattr(vault, "read_record", flaky_read_record)

        restore_namespaced_env(vault)

        assert "healthy-secret" not in scrub_credentials("pw healthy-secret")


class TestTheNoVaultScopeFollowsAmbientChanges:
    """Without a vault nothing rebuilds the map, so the turn boundary is the
    only chance to notice a credential the host rotated between turns."""

    def test_a_rotated_ambient_credential_is_scrubbed_on_the_next_turn(self, monkeypatch):
        from anton.utils.datasources import begin_ds_turn_scope

        monkeypatch.setenv("DS_CUSTOM_THING__TOKEN", "old-secret-value")
        begin_ds_turn_scope()
        assert "old-secret-value" not in scrub_credentials("tok old-secret-value")

        monkeypatch.setenv("DS_CUSTOM_THING__TOKEN", "new-secret-value")
        begin_ds_turn_scope()
        assert "new-secret-value" not in scrub_credentials("tok new-secret-value")
