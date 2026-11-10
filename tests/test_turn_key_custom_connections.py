"""A cloud turn's custom API connections in the turn-key vault: fetched once with the
turn key, named by auth id, scrubbed by auth's secure_keys, and nothing else."""

import io
import json
import threading
import time
import urllib.error

import pytest

from anton.cloud_turn.contract import CustomApiBlockV1, CustomApiConnectionRefV1
from anton.cloud_turn.session import _connector_usage_notes
from anton.core.datasources import data_vault as dv
from anton.core.datasources.data_vault import TurnKeyDataVault, custom_connection_key
from anton.utils import datasources as ds
from tests.test_cloud_turn_session import _build as _build_session

SECRET = "acct-77f3c9e2b1d4-private"
BASE = "https://api.acme.example"


def _ref(connection_id=7, connector_id="acme_crm", name="Prod"):
    return CustomApiConnectionRefV1(connection_id, connector_id, name, 1)


def _block(*refs):
    return CustomApiBlockV1("mdb_turnkey.secret", tuple(refs or (_ref(),)))


def _answer(ref, fields=None, secure_keys=("api_key",)):
    """Shaped like auth's CustomApiFieldsView answer."""
    return {"connector_id": ref.connector_id, "name": ref.name,
            "fields": fields if fields is not None else {"api_key": SECRET, "base_url": BASE},
            "secure_keys": list(secure_keys)}


@pytest.fixture()
def auth(monkeypatch):
    """Answer custom field fetches per connection id; record each call."""
    from anton import minds_client

    answers: dict[int, object] = {}
    calls: list[tuple[str, str]] = []

    def fake_request(url, api_key, **kwargs):
        calls.append((url, api_key))
        connection_id = int(url.rsplit("/", 2)[-2])
        answer = answers[connection_id]
        if callable(answer):
            return answer()
        if isinstance(answer, Exception):
            raise answer
        return json.dumps(answer).encode()

    monkeypatch.setattr(minds_client, "minds_request", fake_request)
    return answers, calls


def test_a_custom_connection_is_listed_by_auth_id_and_named_without_typed_text(auth):
    answers, calls = auth
    answers[7] = _answer(_ref())

    vault = TurnKeyDataVault(None, _block(_ref()))

    assert vault.list_connections() == [{"engine": "c7", "name": "acme_crm", "created_at": ""}]
    assert vault.env_for("c7", "acme_crm") == {
        "DS_C7_ACME_CRM__API_KEY": SECRET, "DS_C7_ACME_CRM__BASE_URL": BASE,
    }
    [(url, key)] = calls
    assert url.endswith("/v1/custom-connections/7/fields")
    assert key == "mdb_turnkey.secret"


def test_every_field_but_the_address_is_secure_whatever_the_template_said(auth):
    """auth's secure_keys follow the template's model-written `secret` flags, so the pod widens them."""
    answers, _ = auth
    fields = {"api_key": SECRET, "api_secret": "s2", "email": "tok_9f8e", "base_url": BASE}
    answers[7] = _answer(_ref(), fields=fields, secure_keys=["api_key"])

    record = TurnKeyDataVault(None, _block(_ref())).read_record("c7", "acme_crm")

    assert record["secure_keys"] == ["api_key", "api_secret", "email"]


def test_the_address_stays_hidden_when_auth_marks_it_secure(auth):
    answers, _ = auth
    answers[7] = _answer(_ref(), fields={"api_key": SECRET, "base_url": BASE}, secure_keys=["api_key", "base_url"])

    record = TurnKeyDataVault(None, _block(_ref())).read_record("c7", "acme_crm")

    assert record["secure_keys"] == ["api_key", "base_url"]


def test_the_label_is_the_members_name_on_one_line_without_control_characters(auth):
    answers, _ = auth
    name = "Prod\nline\u202etwo\x07"
    answers[7] = _answer(_ref(name=name))

    record = TurnKeyDataVault(None, _block(_ref(name=name))).read_record("c7", "acme_crm")

    assert record["fields"]["_user_label"] == "Prod linetwo (acme_crm)"


@pytest.mark.parametrize("answer", [
    urllib.error.HTTPError("u", 404, "not found", {}, io.BytesIO(b"")),
    urllib.error.HTTPError("u", 429, "limit", {}, io.BytesIO(b"")),
    urllib.error.HTTPError("u", 503, "off", {}, io.BytesIO(b"")),
    TimeoutError("slow"),
    lambda: b"not json",
    {"connector_id": "other_api", "name": "Prod", "fields": {"api_key": SECRET}, "secure_keys": ["api_key"]},
    {"connector_id": "acme_crm", "name": "Other", "fields": {"api_key": SECRET}, "secure_keys": ["api_key"]},
    {"connector_id": "acme_crm", "name": "Prod", "fields": {"API_KEY": SECRET}, "secure_keys": ["API_KEY"]},
    {"connector_id": "acme_crm", "name": "Prod", "fields": {"api_key": SECRET}, "secure_keys": ["missing"]},
    {"connector_id": "acme_crm", "name": "Prod", "fields": {"api_key": 5}, "secure_keys": []},
    {"connector_id": "acme_crm", "name": "Prod", "fields": {"api_key": SECRET}},
], ids=["404", "429", "503", "timeout", "not json", "other connector", "other name", "upper field",
        "secure key not a field", "non text value", "no secure keys"])
def test_a_connection_that_did_not_answer_in_shape_is_unknown_to_the_vault(auth, answer):
    answers, _ = auth
    answers[7] = answer

    vault = TurnKeyDataVault(None, _block(_ref()))

    assert vault.list_connections() == []
    assert vault.env_for("c7", "acme_crm") is None
    assert vault.load("c7", "acme_crm") is None
    assert vault.read_record("c7", "acme_crm") is None


def test_an_answer_after_the_deadline_is_discarded_and_never_kept(auth, monkeypatch):
    answers, calls = auth
    released = threading.Event()

    def slow():
        released.wait(5)
        return json.dumps(_answer(_ref())).encode()

    answers[7] = slow
    monkeypatch.setattr(dv, "_CUSTOM_FETCH_DEADLINE_SECONDS", 0.2)

    started = time.monotonic()
    vault = TurnKeyDataVault(None, _block(_ref()))
    assert time.monotonic() - started < 2
    released.set()
    time.sleep(0.2)

    assert vault.list_connections() == []
    assert vault.env_for("c7", "acme_crm") is None
    assert len(calls) == 1


def test_a_fetch_still_waiting_at_the_deadline_never_holds_the_pods_exit(auth, monkeypatch):
    answers, _ = auth
    released = threading.Event()
    answers[7] = lambda: released.wait(5) and json.dumps(_answer(_ref())).encode()
    monkeypatch.setattr(dv, "_CUSTOM_FETCH_DEADLINE_SECONDS", 0.1)

    TurnKeyDataVault(None, _block(_ref()))
    waiting = [thread for thread in threading.enumerate() if thread.name == "custom-fetch"]
    released.set()

    assert waiting and all(thread.daemon for thread in waiting)


def test_a_fetch_still_queued_at_the_deadline_never_starts(auth, monkeypatch):
    """Each fetch spends one of the grant's fetches and writes an audit row; a queued one is not worth it."""
    answers, calls = auth
    refs = [_ref(i, f"api_{i}", f"n{i}") for i in range(1, 11)]
    for ref in refs:
        answers[ref.connection_id] = (lambda r: lambda: time.sleep(0.3) or json.dumps(_answer(r)).encode())(ref)
    monkeypatch.setattr(dv, "_CUSTOM_FETCH_DEADLINE_SECONDS", 0.1)

    TurnKeyDataVault(None, _block(*refs))
    started_by_deadline = len(calls)
    time.sleep(1.0)

    assert started_by_deadline == 4
    assert len(calls) == 4


def test_at_most_four_fetches_run_at_once(auth):
    answers, _ = auth
    lock = threading.Lock()
    running = {"now": 0, "most": 0}

    def counted(ref):
        def answer():
            with lock:
                running["now"] += 1
                running["most"] = max(running["most"], running["now"])
            time.sleep(0.05)
            with lock:
                running["now"] -= 1
            return json.dumps(_answer(ref)).encode()
        return answer

    refs = [_ref(i, f"api_{i}", f"n{i}") for i in range(1, 11)]
    for ref in refs:
        answers[ref.connection_id] = counted(ref)

    vault = TurnKeyDataVault(None, _block(*refs))

    assert running["most"] == 4
    assert len(vault.list_connections()) == 10


def test_custom_connections_sit_beside_oauth_ones_and_survive_clearing_the_token_cache(auth):
    answers, _ = auth
    answers[7] = _answer(_ref())
    oauth = {"turn_key": "mdb_turnkey.secret", "connections": [{"engine": "hubspot", "name": "main"}]}

    vault = TurnKeyDataVault(oauth, _block(_ref()))
    vault.clear_ds_env()

    assert [(c["engine"], c["name"]) for c in vault.list_connections()] == [("hubspot", "main"), ("c7", "acme_crm")]
    assert vault.env_for("c7", "acme_crm")["DS_C7_ACME_CRM__API_KEY"] == SECRET


def test_a_connection_the_turn_did_not_list_is_never_fetched(auth):
    answers, calls = auth
    answers[7] = _answer(_ref())
    vault = TurnKeyDataVault(None, _block(_ref()))

    assert vault.load("c8", "acme_crm") is None
    assert len(calls) == 1


@pytest.mark.parametrize(("connector_id", "secret_field"), [
    ("myapi", "client_id"), ("shopify", "client_id"), ("chromadb", "host"), ("email", "email"),
])
def test_the_secret_is_hidden_by_auths_secure_keys_whatever_the_connector_id(auth, connector_id, secret_field):
    """An id that is also an anton engine (shopify, email, …) is still classified by auth's keys."""
    answers, _ = auth
    ref = _ref(7, connector_id)
    answers[7] = _answer(ref, fields={secret_field: SECRET, "base_url": BASE}, secure_keys=[secret_field])
    vault = TurnKeyDataVault(None, _block(ref))

    ds.restore_namespaced_env(vault)
    out = ds.scrub_credentials(f"value is {SECRET} at {BASE}")

    assert SECRET not in out
    assert BASE in out


@pytest.mark.parametrize("pair", [
    ((1, "ab", "k_c2__z"), (2, "ab_c1__k", "z")),
    ((5, "gmail_x", "k__access_token"), (6, "gmail", "access_token")),
], ids=["custom with custom", "a connector id shaped like another's prefix"])
def test_review_inputs_that_once_collided_now_give_distinct_names(auth, pair):
    answers, _ = auth
    refs = []
    for connection_id, connector_id, field in pair:
        ref = _ref(connection_id, connector_id, f"n{connection_id}")
        refs.append(ref)
        answers[connection_id] = _answer(ref, fields={field: f"v{connection_id}"}, secure_keys=[field])

    vault = TurnKeyDataVault(None, _block(*refs))
    names = [set(vault.env_for(*custom_connection_key(r.connection_id, r.connector_id))) for r in refs]

    assert names[0].isdisjoint(names[1])


def test_the_connectors_notes_follow_it_to_its_vault_engine():
    block = _block(_ref(7, "acme_crm"))

    notes = _connector_usage_notes({"acme_crm": {"usage_notes": "Call /v1/me first."}}, block)

    assert notes["c7"] == "Call /v1/me first."


def test_a_turn_with_only_custom_connections_gets_a_vault_and_their_notes(auth, tmp_path, monkeypatch):
    """No oauth block: the session still builds the turn-key vault for the custom one."""
    answers, _ = auth
    answers[7] = _answer(_ref())

    _, cfg = _build_session(
        tmp_path, monkeypatch, custom_api=_block(_ref()),
        connectors={"acme_crm": {"usage_notes": "Call /v1/me first."}},
    )

    assert isinstance(cfg.data_vault, TurnKeyDataVault)
    assert cfg.data_vault.list_connections() == [{"engine": "c7", "name": "acme_crm", "created_at": ""}]
    assert cfg.connector_usage_notes["c7"] == "Call /v1/me first."
