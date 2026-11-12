"""`call_minds_api`, the scratchpad's turn-key-bound helper for a member's custom API connections.

Each test runs the real boot in a child process, as a cloud turn does. Most
replace ``http.client.HTTPSConnection`` inside the cell with a recorder that
answers from a script; one sends to a local TLS server.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import os
import ssl
import subprocess
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from anton.core.backends.wire import (
    CELL_DELIM,
    INJECTED_HELPER_NAMES,
    RESULT_END,
    RESULT_START,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES = Path(__file__).parent / "fixtures"
_CONTRACT_FILE = _FIXTURES / "custom_api_request_v1.json"
#: Copied byte for byte from mindshub_inference's docs/contracts/custom-api-request-v1.json
#: at this commit. Copy again, and update both, when the gateway's file changes.
_CONTRACT_SOURCE = "da3f9371fc146a763c02193899e4699caa29466f"
_CONTRACT_SHA256 = "9340bcb30d0d7e78c0da3f38f170082efaf3a677df440e129e6f1a0d1ec1d3fe"
_CONTRACT = json.loads(_CONTRACT_FILE.read_text(encoding="utf-8"))
#: The gateway's 422 details that name a part; the merge gate checks them at the gateway's merge SHA.
_DETAILS = json.loads(
    (_FIXTURES / "custom_api_request_details_v1.json").read_text(encoding="utf-8")
)
_EXAMPLE = _CONTRACT["request"]
_TURN_KEY = "mdb_turn_custom_secret"
_GATEWAY_CODES = sorted(error["code"] for error in _CONTRACT["errors"])
_HELPER_CODES = [
    "invalid_custom_api_request",
    "gateway_unavailable",
    "custom_api_unavailable",
    "invalid_custom_api_response",
]


def _env(**overrides: str) -> dict[str, str]:
    """A cloud turn child's environment with the contract example's connection and correlation id."""
    env = os.environ.copy()
    for key in [k for k in env if k.startswith(("ANTON_CLOUD_", "DS_"))]:
        del env[key]
    env.update(
        {
            "ANTON_CLOUD_TURN": "1",
            "OPENAI_API_KEY": _TURN_KEY,
            "OPENAI_BASE_URL": "https://inference.internal/v1",
            "ANTON_OPENAI_API_KEY": "shared-provider-key",
            "ANTON_OPENAI_BASE_URL": "https://shared.invalid/v1",
            "ANTON_CLOUD_CUSTOM_API_CORRELATION_ID": _EXAMPLE["correlation_id"],
            "ANTON_CLOUD_CUSTOM_API_CONNECTIONS": json.dumps(
                [
                    {
                        "connection_id": _EXAMPLE["connection_id"],
                        "credential_version": _EXAMPLE["credential_version"],
                    }
                ]
            ),
            "ANTON_SCRATCHPAD_HEARTBEAT_INTERVAL": "0",
            "PYTHONPATH": str(REPO_ROOT),
        }
    )
    env.update(overrides)
    return env


def _run_cell(cell: str, tmp_path: Path, env: dict[str, str]) -> dict:
    """Run one cell in a fresh boot child; return its result record."""
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "anton" / "core" / "backends" / "scratchpad_boot.py"),
        ],
        input=cell + f"\n{CELL_DELIM}\n",
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env=env,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(
        completed.stdout.split(RESULT_START, 1)[1].split(RESULT_END, 1)[0]
    )
    assert result["error"] is None, result["error"]
    result["child_stderr"] = completed.stderr
    return result


#: Replaces the HTTPS connection with a recorder, runs each scenario's call against
#: its scripted answers, and prints what was sent and returned. The Authorization
#: header is compared here, so the key itself is never printed.
_HARNESS = """
import http.client, json, os

SCENARIOS = json.loads({scenarios!r})
sent = []
answers = []

def _body(spec):
    if spec is None:
        return b""
    if "json" in spec:
        return json.dumps(spec["json"]).encode()
    if "repeat" in spec:
        return spec["repeat"][0].encode() * spec["repeat"][1]
    return spec["text"].encode("utf-8")

class Response:
    def __init__(self, status, body):
        self.status = status
        self._body = body
    def read(self, amt=None):
        return self._body if amt is None else self._body[:amt]

class Connection:
    def __init__(self, host, port=None, timeout=None, context=None):
        self.target = (host, port, timeout, context is not None)
    def request(self, method, url, body=None, headers=None):
        headers = dict(headers or {{}})
        auth = headers.pop("Authorization", None)
        sent.append({{
            "target": list(self.target), "method": method, "url": url,
            "body": json.loads(body), "headers": headers,
            "bearer_is_turn_key": auth == "Bearer " + os.environ["OPENAI_API_KEY"],
        }})
    def getresponse(self):
        status, spec = answers.pop(0)
        if status is None:
            raise OSError("connection reset")
        return Response(status, _body(spec))
    def close(self):
        pass

http.client.HTTPSConnection = Connection
results = []
for scenario in SCENARIOS:
    sent.clear()
    answers[:] = [tuple(a) for a in scenario.get("answers", [])]
    result = call_minds_api(*scenario["args"], **scenario.get("kwargs", {{}}))
    key_in_result = os.environ["OPENAI_API_KEY"] in json.dumps(result)
    results.append({{"result": result, "sent": list(sent), "key_in_result": key_in_result}})
print(json.dumps(results))
"""


def _calls(tmp_path: Path, scenarios: list[dict], **env: str) -> list[dict]:
    """Run ``scenarios`` through ``call_minds_api`` in one child; each gets its result and what it sent."""
    record = _run_cell(
        _HARNESS.format(scenarios=json.dumps(scenarios)), tmp_path, _env(**env)
    )
    outcomes = json.loads(record["stdout"])
    # Every call, whatever it returns: no result carries the turn key.
    assert [outcome.pop("key_in_result") for outcome in outcomes] == [False] * len(
        outcomes
    )
    return outcomes


def _call(
    tmp_path: Path, args: list, kwargs: dict | None = None, answers: list | None = None
) -> dict:
    """One call's result and what it sent."""
    return _calls(
        tmp_path, [{"args": args, "kwargs": kwargs or {}, "answers": answers or []}]
    )[0]


_ID = _EXAMPLE["connection_id"]


def test_the_contract_copy_is_the_gateways_file_at_the_pinned_commit():
    assert (
        hashlib.sha256(_CONTRACT_FILE.read_bytes()).hexdigest() == _CONTRACT_SHA256
    ), (
        f"{_CONTRACT_FILE.name} differs from mindshub_inference's at {_CONTRACT_SOURCE}. "
        "Copy it again, unchanged, and update the pinned hash and source commit here."
    )


def test_the_request_is_the_contracts_example(tmp_path):
    call = _EXAMPLE["request"]
    outcome = _call(
        tmp_path,
        args=[_ID, call["path"]],
        kwargs={"query": call["query"], "headers": call["headers"]},
        answers=[[200, {"json": _CONTRACT["response"]}]],
    )

    assert outcome["sent"] == [
        {
            "target": ["inference.internal", 443, 30, True],
            "method": "POST",
            "url": "/v1/datasources/custom-api/request",
            "body": _EXAMPLE,
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            "bearer_is_turn_key": True,
        }
    ]


def test_a_dict_query_and_numbers_are_sent_as_string_pairs(tmp_path):
    scenarios = [
        {
            "args": [_ID, "/v2/eod"],
            "kwargs": {"query": {"symbols": "AAPL", "limit": 10, "ratio": 0.5}},
        },
        {"args": [_ID, "/v2/eod"]},
    ]
    for scenario in scenarios:
        scenario["answers"] = [[200, {"json": _CONTRACT["response"]}]]

    first, bare = _calls(tmp_path, scenarios)

    assert first["sent"][0]["body"]["request"]["query"] == [
        ["symbols", "AAPL"],
        ["limit", "10"],
        ["ratio", "0.5"],
    ]
    assert bare["sent"][0]["body"]["request"] == {
        "method": "GET",
        "path": "/v2/eod",
        "query": [],
        "headers": {},
    }


def test_nan_and_large_floats_are_sent_as_python_writes_them(tmp_path):
    cell = _HARNESS.format(scenarios="[]") + (
        "\nanswers[:] = [(200, {'json': SAMPLE})]\n"
        "r = call_minds_api(ID, '/v2/eod', query=[['x', float('nan')], ['y', 1e20]])\n"
        "print(json.dumps({'result': r, 'query': sent[0]['body']['request']['query']}))\n"
    )
    cell = f"SAMPLE = {_CONTRACT['response']!r}\nID = {_ID}\n" + cell
    record = _run_cell(cell, tmp_path, _env())

    last = json.loads(record["stdout"].strip().splitlines()[-1])
    assert last["query"] == [["x", "nan"], ["y", "1e+20"]]
    assert last["result"]["type"] == "response"


@pytest.mark.parametrize(
    "name",
    ["response", "binary_response", "provider_error_response", "client_error_response"],
)
def test_each_answer_example_comes_back_as_the_result(tmp_path, name):
    example = _CONTRACT[name]
    outcome = _call(tmp_path, [_ID, "/v2/quote"], answers=[[200, {"json": example}]])

    assert outcome["result"] == {
        "type": "response",
        "status": example["status"],
        "headers": example["headers"],
        "body": example["body"],
        "body_encoding": example["body_encoding"],
    }


@pytest.mark.parametrize("status", [100, 999])
def test_any_status_the_gateway_can_pass_through_is_accepted(tmp_path, status):
    answer = {**_CONTRACT["response"], "status": status}
    outcome = _call(tmp_path, [_ID, "/v2/quote"], answers=[[200, {"json": answer}]])

    assert outcome["result"]["status"] == status


@pytest.mark.parametrize(
    "change",
    [
        {"status": 99},
        {"status": 1000},
        {"status": True},
        {"status": "200"},
        {"connection_id": _ID + 1},
        {"credential_version": _EXAMPLE["credential_version"] + 1},
        {"protocol_version": 2},
        {"connection_id": str(_ID)},
        {"headers": {"content-type": "image/png"}},
        {"headers": {"content-type": "application/json", "set-cookie": "a=b"}},
        {"headers": {"Content-Type": "application/json"}},
        {"body": 1},
        {"body_encoding": "gzip"},
        {"body_encoding": "base64", "body": "aGk*="},
        {"truncated": False},
    ],
    ids=[
        "status-99",
        "status-1000",
        "status-bool",
        "status-text",
        "other-connection",
        "other-version",
        "other-protocol",
        "id-as-text",
        "unlisted-media-type",
        "extra-header",
        "header-case",
        "body-not-text",
        "unknown-encoding",
        "bad-base64",
        "extra-key",
    ],
)
def test_an_answer_outside_the_contract_is_refused(tmp_path, change):
    answer = {**_CONTRACT["response"], **change}
    outcome = _call(tmp_path, [_ID, "/v2/quote"], answers=[[200, {"json": answer}]])

    assert outcome["result"] == {
        "type": "error",
        "error_code": "invalid_custom_api_response",
    }


@pytest.mark.parametrize(
    "body",
    [{"text": "not json"}, {"json": ["a list"]}],
    ids=["not-json", "not-an-object"],
)
def test_an_unreadable_answer_is_refused(tmp_path, body):
    outcome = _call(tmp_path, [_ID, "/v2/quote"], answers=[[200, body]])

    assert outcome["result"] == {
        "type": "error",
        "error_code": "invalid_custom_api_response",
    }


def test_an_answer_past_the_read_cap_is_too_large(tmp_path):
    outcome = _call(
        tmp_path, [_ID, "/v2/quote"], answers=[[200, {"repeat": ["a", 8_388_609]}]]
    )

    assert outcome["result"] == {"type": "error", "error_code": "response_too_large"}


@pytest.mark.parametrize("error", _CONTRACT["errors"], ids=lambda e: e["code"])
def test_each_gateway_code_is_relayed_as_it_leaves_the_gateway(tmp_path, error):
    status = error.get("leaves_as", error["status"])
    body = {
        "code": error["code"],
        "detail": "Custom API request is invalid.",
        "request_id": "r1",
    }
    if status != error["status"]:
        body["upstream_status"] = error["status"]
    outcome = _call(tmp_path, [_ID, "/v2/quote"], answers=[[status, {"json": body}]])

    assert outcome["result"] == {"type": "error", "error_code": error["code"]}


@pytest.mark.parametrize("part", sorted(_DETAILS))
def test_a_refused_part_is_named_from_the_gateways_own_detail(tmp_path, part):
    body = {"code": "invalid_request", "detail": _DETAILS[part], "request_id": "r1"}
    outcome = _call(tmp_path, [_ID, "/v2/quote"], answers=[[422, {"json": body}]])

    assert outcome["result"] == {
        "type": "error",
        "error_code": "invalid_request",
        "part": part,
    }


@pytest.mark.parametrize(
    "detail",
    ["Custom API request is invalid.", "Custom API request path is refused.", None, 7],
    ids=["no-part", "reworded", "absent", "not-text"],
)
def test_a_detail_that_is_not_one_of_the_four_names_no_part(tmp_path, detail):
    body = {
        "code": "invalid_request",
        "request_id": "r1",
        **({"detail": detail} if detail is not None else {}),
    }
    outcome = _call(tmp_path, [_ID, "/v2/quote"], answers=[[422, {"json": body}]])

    assert outcome["result"] == {"type": "error", "error_code": "invalid_request"}


def test_a_part_comes_only_with_invalid_request(tmp_path):
    body = {
        "code": "invalid_connection",
        "detail": _DETAILS["query"],
        "request_id": "r1",
    }
    outcome = _call(tmp_path, [_ID, "/v2/quote"], answers=[[404, {"json": body}]])

    assert outcome["result"] == {"type": "error", "error_code": "invalid_connection"}


@pytest.mark.parametrize(
    "answer",
    [
        [503, {"json": {"code": "made_up_code"}}],
        [502, {"text": "<html>Bad gateway</html>"}],
        [404, {"text": "Not Found"}],
        [302, {"text": ""}],
        [503, {"repeat": ["a", 70_000]}],
        [None, None],
    ],
    ids=[
        "unknown-code",
        "proxy-page",
        "ingress-404",
        "redirect",
        "huge-error-body",
        "connection-reset",
    ],
)
def test_anything_else_is_the_custom_api_being_unavailable(tmp_path, answer):
    outcome = _call(tmp_path, [_ID, "/v2/quote"], answers=[answer])

    assert outcome["result"] == {
        "type": "error",
        "error_code": "custom_api_unavailable",
    }


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        ([_ID + 1, "/v2/quote"], {}),
        ([True, "/v2/quote"], {}),
        ([str(_ID), "/v2/quote"], {}),
        ([_ID, ""], {}),
        ([_ID, "v2/quote"], {}),
        ([_ID, None], {}),
        ([_ID, "/v2/quote"], {"query": "symbols=AAPL"}),
        ([_ID, "/v2/quote"], {"query": [["symbols"]]}),
        ([_ID, "/v2/quote"], {"query": [["symbols", "AAPL", "x"]]}),
        ([_ID, "/v2/quote"], {"query": {"symbols": True}}),
        ([_ID, "/v2/quote"], {"query": {"symbols": None}}),
        ([_ID, "/v2/quote"], {"query": {"symbols": ["AAPL"]}}),
        ([_ID, "/v2/quote"], {"query": [[1, "AAPL"]]}),
        ([_ID, "/v2/quote"], {"headers": [["Accept", "text/csv"]]}),
        ([_ID, "/v2/quote"], {"headers": {"Accept": 1}}),
    ],
    ids=[
        "unknown-connection",
        "bool-id",
        "text-id",
        "empty-path",
        "relative-path",
        "no-path",
        "query-as-text",
        "pair-of-one",
        "pair-of-three",
        "bool-value",
        "none-value",
        "list-value",
        "number-name",
        "headers-as-pairs",
        "number-header",
    ],
)
def test_an_argument_the_helper_refuses_sends_nothing(tmp_path, args, kwargs):
    outcome = _call(tmp_path, args, kwargs)

    assert outcome == {
        "result": {"type": "error", "error_code": "invalid_custom_api_request"},
        "sent": [],
    }


@pytest.mark.parametrize(
    "base_url",
    [
        "http://inference.internal/v1",
        "https://inference.internal/api/v1",
        "https://inference.internal",
        "https://u:p@inference.internal/v1",
        "https://inference.internal/v1?x=1",
        "",
    ],
    ids=["plaintext", "api-v1", "no-path", "credentials", "query", "unset"],
)
def test_a_base_the_bearer_cannot_go_to_sends_nothing(tmp_path, base_url):
    scenarios = [
        {
            "args": [_ID, "/v2/quote"],
            "answers": [[200, {"json": _CONTRACT["response"]}]],
        }
    ]
    (outcome,) = _calls(tmp_path, scenarios, OPENAI_BASE_URL=base_url)

    assert outcome == {
        "result": {"type": "error", "error_code": "gateway_unavailable"},
        "sent": [],
    }


def test_no_result_or_stderr_line_carries_the_turn_key(tmp_path):
    scenarios = [
        {"args": [_ID, "/v2/quote"], "answers": [[None, None]]},
        {"args": [_ID, "/v2/quote"], "answers": [[503, {"text": _TURN_KEY}]]},
        {
            "args": [_ID, f"/{_TURN_KEY}"],
            "answers": [
                [
                    422,
                    {
                        "json": {
                            "code": "invalid_request",
                            "detail": _DETAILS["turn_key"],
                        }
                    },
                ]
            ],
        },
        {"args": [_ID + 1, "/v2/quote"]},
    ]
    record = _run_cell(
        _HARNESS.format(scenarios=json.dumps(scenarios)), tmp_path, _env()
    )

    assert [outcome["key_in_result"] for outcome in json.loads(record["stdout"])] == [
        False
    ] * 4
    assert _TURN_KEY not in record["stderr"]
    assert _TURN_KEY not in record["child_stderr"]


def test_the_helper_codes_are_one_closed_list():
    assert len(_GATEWAY_CODES) == 12
    assert len(set(_GATEWAY_CODES) | set(_HELPER_CODES)) == 16
    assert "response_too_large" in _GATEWAY_CODES


@pytest.mark.parametrize(
    "removed",
    ["ANTON_CLOUD_CUSTOM_API_CONNECTIONS", "ANTON_CLOUD_TURN"],
    ids=["no-connection-refs", "not-a-cloud-turn"],
)
def test_no_helper_is_injected_outside_a_custom_api_cloud_turn(tmp_path, removed):
    env = _env()
    del env[removed]
    record = _run_cell("print('call_minds_api' in globals())", tmp_path, env)

    assert record["stdout"] == "False\n"


def test_malformed_references_inject_nothing_and_say_why(tmp_path):
    record = _run_cell(
        "print('call_minds_api' in globals())",
        tmp_path,
        _env(ANTON_CLOUD_CUSTOM_API_CONNECTIONS='[{"connection_id": 1}]'),
    )

    assert record["stdout"] == "False\n"
    assert "custom API helper not injected" in record["child_stderr"]


def test_the_write_check_knows_the_helper():
    assert "call_minds_api" in INJECTED_HELPER_NAMES


def _self_signed(tmp_path: Path) -> tuple[Path, Path]:
    """A certificate and key for localhost, valid for a day."""
    # Imported here: cryptography is optional for anton, and its test skips without it.
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.now(UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False
        )
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    cert_path = tmp_path / "cert.pem"
    key_path = tmp_path / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return cert_path, key_path


def test_the_wire_request_reaches_a_tls_gateway_as_the_contract_says(tmp_path):
    pytest.importorskip("cryptography")
    cert_path, key_path = _self_signed(tmp_path)
    seen: list[dict] = []

    class Gateway(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            seen.append(
                {
                    "line": self.requestline,
                    "authorization": self.headers["Authorization"],
                    "content_type": self.headers["Content-Type"],
                    "body": json.loads(self.rfile.read(length)),
                }
            )
            payload = json.dumps(_CONTRACT["response"]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Gateway)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        call = _EXAMPLE["request"]
        cell = (
            "import json\n"
            f"r = call_minds_api({_ID}, {call['path']!r}, query={call['query']!r}, headers={call['headers']!r})\n"
            "print(json.dumps(r))\n"
        )
        env = _env(
            OPENAI_BASE_URL=f"https://localhost:{server.server_address[1]}/v1",
            SSL_CERT_FILE=str(cert_path),
        )
        record = _run_cell(cell, tmp_path, env)
    finally:
        server.shutdown()
        server.server_close()

    assert seen == [
        {
            "line": "POST /v1/datasources/custom-api/request HTTP/1.1",
            "authorization": f"Bearer {_TURN_KEY}",
            "content_type": "application/json",
            "body": _EXAMPLE,
        }
    ]
    assert json.loads(record["stdout"])["type"] == "response"
