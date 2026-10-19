from __future__ import annotations

import asyncio
import base64
import json
import os
from unittest.mock import patch

import pytest

from anton.core.session import _scrub_user_input
from anton.tools import SCRUBBED_VALUE_RE
from anton.utils.datasources import (
    _DS_KNOWN_VARS,
    _DS_SECRET_VARS,
    _PROVIDER_SECRET_VARS,
    _reset_registered_ds_vars,
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
    """Counts exactly the values scrub_credentials matches by exact value."""

    def test_counts_secret_unknown_and_provider_values_only(self, monkeypatch):
        """Secret fields, unknown DS_* over 8 chars and provider keys over 8
        chars count; known non-secret fields and short values do not."""
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
        assert longest_exact_secret() == 30

        monkeypatch.setenv("OPENAI_API_KEY", "k" * 40)
        assert longest_exact_secret() == 40

        monkeypatch.setenv("OPENAI_API_KEY", "k" * 8)
        assert longest_exact_secret() == 30

    def test_a_short_secret_field_counts(self, monkeypatch):
        """Secret fields match at any length, unlike unknown and provider values."""
        _DS_SECRET_VARS.add("DS_PG_PROD__PASSWORD")
        _DS_KNOWN_VARS.add("DS_PG_PROD__PASSWORD")
        set_ds_env_values({"DS_PG_PROD__PASSWORD": "p" * 5})
        for key in _PROVIDER_SECRET_VARS:
            monkeypatch.delenv(key, raising=False)
        assert longest_exact_secret() == 5

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
