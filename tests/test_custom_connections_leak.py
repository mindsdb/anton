"""A custom API connection's values reach no log record, no prompt and no turn output,
whatever its fetch did."""

import io
import json
import logging
import threading
import urllib.error

import pytest

from anton.cloud_turn.contract import CustomApiBlockV1, CustomApiConnectionRefV1
from anton.core.datasources import data_vault as dv
from anton.core.datasources.data_vault import TurnKeyDataVault
from anton.utils import datasources as ds

SECRET = "acct-77f3c9e2b1d4-private"
CLIENT_REF = "cref-9a81b2c3d4e5-planted"
TURN_KEY = "mdb_turnkey.planted-secret"


class _Every(logging.Handler):
    """Every record any logger emits, formatted with its arguments and traceback."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())
        if record.exc_info:
            self.lines.append(logging.Formatter().formatException(record.exc_info))


@pytest.fixture()
def records():
    handler = _Every()
    loggers = [logging.getLogger()] + [
        lg for lg in logging.root.manager.loggerDict.values() if isinstance(lg, logging.Logger)
    ]
    saved = [(lg, lg.level) for lg in loggers]
    for lg in loggers:
        lg.addHandler(handler)
        lg.setLevel(logging.DEBUG)
    yield handler.lines
    for lg, level in saved:
        lg.removeHandler(handler)
        lg.setLevel(level)


def _ref(connection_id=7, connector_id="acme_crm"):
    return CustomApiConnectionRefV1(connection_id, connector_id, "Prod", 1)


def _body(ref):
    return json.dumps({
        "connector_id": ref.connector_id, "name": ref.name,
        "fields": {"api_key": SECRET, "client_ref": CLIENT_REF, "base_url": "https://api.acme.example"},
        "secure_keys": ["api_key", "client_ref"],
    }).encode()


def _http(code):
    # auth's error body could echo anything; the vault must not log it.
    return urllib.error.HTTPError("u", code, f"refused {SECRET}", {}, io.BytesIO(SECRET.encode()))


@pytest.mark.parametrize("outcome", ["pass", "404", "429", "503", "malformed", "timeout", "late", "colliding id"])
def test_a_planted_value_reaches_no_log_no_prompt_and_no_output(records, monkeypatch, outcome):
    from anton import minds_client

    ref = _ref(7, "shopify" if outcome == "colliding id" else "acme_crm")
    released = threading.Event()

    def fake_request(url, api_key, **kwargs):
        if outcome in ("pass", "colliding id"):
            return _body(ref)
        if outcome in ("404", "429", "503"):
            raise _http(int(outcome))
        if outcome == "malformed":
            return f'{{"connector_id": "other", "fields": {{"api_key": "{SECRET}"}}}}'.encode()
        if outcome == "timeout":
            raise TimeoutError(f"no answer for {SECRET}")
        released.wait(5)
        return _body(ref)

    monkeypatch.setattr(minds_client, "minds_request", fake_request)
    monkeypatch.setattr(dv, "_CUSTOM_FETCH_DEADLINE_SECONDS", 0.2)

    vault = TurnKeyDataVault(None, CustomApiBlockV1(TURN_KEY, (ref,)))
    released.set()
    ds.restore_namespaced_env(vault)
    prompt = ds.build_datasource_context(vault, usage_notes={"c7": "Call /v1/me first."})
    printed = ds.scrub_output(f"api_key={SECRET} ref={CLIENT_REF} key={TURN_KEY}")

    for planted in (SECRET, CLIENT_REF, TURN_KEY):
        assert not [line for line in records if planted in line], planted
        assert planted not in prompt
    if outcome in ("pass", "colliding id"):
        assert "DS_C7_" in prompt
        assert SECRET not in printed and CLIENT_REF not in printed
    else:
        assert vault.list_connections() == []


def test_every_variable_the_pad_gets_is_registered_for_scrubbing(monkeypatch):
    """A pad restores last turn's variables only when their names are still present,
    so a name must mean the same connection and be scrubbed whenever it exists."""
    from anton import minds_client

    refs = [_ref(1, "ab"), _ref(2, "ab_c1__k")]
    bodies = {
        1: {"connector_id": "ab", "name": "Prod", "fields": {"k_c2__z": "value-one-planted"}, "secure_keys": ["k_c2__z"]},
        2: {"connector_id": "ab_c1__k", "name": "Prod", "fields": {"z": "value-two-planted"}, "secure_keys": ["z"]},
    }
    monkeypatch.setattr(minds_client, "minds_request",
                        lambda url, key, **kw: json.dumps(bodies[int(url.rsplit("/", 2)[-2])]).encode())
    vault = TurnKeyDataVault(None, CustomApiBlockV1(TURN_KEY, tuple(refs)))

    ds.restore_namespaced_env(vault)
    env = {}
    for connection in vault.list_connections():
        env.update(vault.env_for(connection["engine"], connection["name"]))

    assert len(env) == 2
    for value in env.values():
        assert value not in ds.scrub_credentials(f"printed {value} here")


def test_fields_the_template_left_unflagged_are_hidden_and_kept_out_of_the_prompt(monkeypatch):
    """A template can mark a credential field `secret: false`; the turn hides it anyway."""
    from anton import minds_client

    ref = _ref()
    api_secret, account = "hunter2-prod-api-secret-77aa", "tok_9f8e7d6c5b4a39281706"
    body = {"connector_id": ref.connector_id, "name": ref.name, "secure_keys": ["api_key"],
            "fields": {"api_key": SECRET, "api_secret": api_secret, "email": account,
                       "base_url": "https://api.acme.example"}}
    monkeypatch.setattr(minds_client, "minds_request", lambda url, key, **kw: json.dumps(body).encode())
    vault = TurnKeyDataVault(None, CustomApiBlockV1(TURN_KEY, (ref,)))

    ds.restore_namespaced_env(vault)
    prompt = ds.build_datasource_context(vault, usage_notes={})
    printed = ds.scrub_output(f"secret={api_secret} email={account} key={SECRET}")

    for planted in (SECRET, api_secret, account):
        assert planted not in printed
        assert planted not in prompt
