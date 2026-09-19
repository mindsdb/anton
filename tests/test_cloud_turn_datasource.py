from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_cloud_scratchpad_helper_uses_turn_bound_typed_request(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(
        {
            "ANTON_CLOUD_TURN": "1",
            "ANTON_CLOUD_DATASOURCE_TURN_KEY": "mdb_turn_secret",
            "ANTON_CLOUD_DATASOURCE_CORRELATION_ID": "corr-1",
            "ANTON_CLOUD_DATASOURCE_CONNECTIONS": json.dumps(
                [{"connection_id": 7, "credential_version": 3}]
            ),
            "ANTON_DATASOURCE_GATEWAY_URL": "https://datasource.internal/base",
            "ANTON_MINDS_DATASOURCE": "legacy-datasource",
            "ANTON_SCRATCHPAD_HEARTBEAT_INTERVAL": "0",
            "PYTHONPATH": str(repo_root),
        }
    )
    cell = """
import json
import urllib.request

seen = {}

class Response:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, size=-1):
        if seen["body"]["operation"]["sql"] == "SELECT wide":
            return json.dumps({
                "columns": [{"name": "value", "type_name": "text"}] * 257,
                "rows": [],
                "truncated": False,
            }).encode()
        return json.dumps({
            "columns": [{"name": "value", "type_name": "text"}],
            "rows": [["untrusted-value"]],
            "truncated": False,
        }).encode()

def open_url(request, timeout):
    seen["url"] = request.full_url
    seen["headers"] = dict(request.headers)
    seen["body"] = json.loads(request.data)
    return Response()

urllib.request.urlopen = open_url
result = query_minds_data(7, "SELECT 1", {"limit": 1})
assert result == {
    "type": "query",
    "columns": [{"name": "value", "type_name": "text"}],
    "data": [["untrusted-value"]],
    "truncated": False,
}
assert seen["url"] == "https://datasource.internal/base/v1/datasources/execute"
assert seen["headers"]["Authorization"] == "Bearer mdb_turn_secret"
assert seen["body"]["connection_id"] == 7
assert seen["body"]["credential_version"] == 3
assert seen["body"]["correlation_id"] == "corr-1"
assert "capability" not in seen["body"]
assert seen["body"]["operation"] == {
    "kind": "query",
    "sql": "SELECT 1",
    "parameters": {"limit": 1},
}
assert query_minds_data(8, "SELECT 1") == {
    "type": "error",
    "error_code": "invalid_datasource_request",
}
assert query_minds_data(7, "SELECT wide") == {
    "type": "error",
    "error_code": "result_too_large",
}
assert query_minds_data(7, "x" * 65537) == {
    "type": "error",
    "error_code": "query_too_large",
}
print("ok")
"""
    completed = subprocess.run(
        [sys.executable, str(repo_root / "anton" / "core" / "backends" / "scratchpad_boot.py")],
        input=cell + "\n__ANTON_CELL_END__\n",
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env=env,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "__ANTON_RESULT__" in completed.stdout
    assert '"error": null' in completed.stdout
    assert '"stdout": "ok\\n"' in completed.stdout
