from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import UUID

import pytest


@pytest.mark.asyncio
async def test_docker_backend_report_exists_and_get_report(monkeypatch):
    from minds.agents.anton_agent.anton.backends.docker import DockerScratchpadRuntime, NotFound

    runtime = DockerScratchpadRuntime(name="x", _extra_env={"ANTON_MINDS_CONVERSATION_ID": "c1"})

    monkeypatch.setattr(runtime, "_get_container", lambda: (_ for _ in ()).throw(NotFound("x")))
    ok = await runtime.report_exists(
        organization_id=UUID("00000000-0000-0000-0000-000000000002"),
        user_id=UUID("00000000-0000-0000-0000-000000000001"),
        conversation_id=UUID("00000000-0000-0000-0000-0000000000aa"),
        message_id=UUID("00000000-0000-0000-0000-0000000000bb"),
    )
    assert ok is False

    with pytest.raises(FileNotFoundError):
        await runtime.get_report(
            organization_id=UUID("00000000-0000-0000-0000-000000000002"),
            user_id=UUID("00000000-0000-0000-0000-000000000001"),
            conversation_id=UUID("00000000-0000-0000-0000-0000000000aa"),
            message_id=UUID("00000000-0000-0000-0000-0000000000bb"),
        )


@pytest.mark.asyncio
async def test_docker_backend_get_report_success(monkeypatch):
    import io
    import tarfile

    from minds.agents.anton_agent.anton.backends.docker import DockerScratchpadRuntime

    runtime = DockerScratchpadRuntime(name="x", _extra_env={"ANTON_MINDS_CONVERSATION_ID": "c1"})

    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as tar:
        payload = b"<html>ok</html>"
        info = tarfile.TarInfo(name="report.html")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    tar_bytes = data.getvalue()

    class Container:
        def get_archive(self, _path):
            return ([tar_bytes], None)

    monkeypatch.setattr(runtime, "_get_container", lambda: Container())
    html = await runtime.get_report(
        organization_id=UUID("00000000-0000-0000-0000-000000000002"),
        user_id=UUID("00000000-0000-0000-0000-000000000001"),
        conversation_id=UUID("00000000-0000-0000-0000-0000000000aa"),
        message_id=UUID("00000000-0000-0000-0000-0000000000bb"),
    )
    assert html == "<html>ok</html>"


@pytest.mark.asyncio
async def test_docker_backend_execute_streaming_happy_path(monkeypatch):
    from minds.agents.anton_agent.anton.backends.base import Cell
    from minds.agents.anton_agent.anton.backends.docker import DockerScratchpadRuntime

    rt = DockerScratchpadRuntime(name="x", _extra_env={"ANTON_MINDS_CONVERSATION_ID": "c1"})
    rt.sock = SimpleNamespace(_sock=object())

    class _Loop:
        async def sock_sendall(self, _sock, _data):
            return None

    monkeypatch.setattr("minds.agents.anton_agent.anton.backends.docker.asyncio.get_running_loop", lambda: _Loop())

    async def _read_result(**_kwargs):
        yield "progress 1"
        yield {"stdout": "out", "stderr": "", "error": None, "logs": ""}

    monkeypatch.setattr(rt, "_read_result", _read_result)

    items = [i async for i in rt.execute_streaming("print(1)", description="d", estimated_seconds=1)]
    assert "progress 1" in items
    cells = [x for x in items if isinstance(x, Cell)]
    assert cells and cells[0].stdout == "out"


@pytest.mark.asyncio
async def test_docker_backend_execute_streaming_timeout_recovers(monkeypatch):
    from minds.agents.anton_agent.anton.backends.docker import DockerScratchpadRuntime

    rt = DockerScratchpadRuntime(name="x", _extra_env={"ANTON_MINDS_CONVERSATION_ID": "c1"})
    rt.sock = SimpleNamespace(_sock=object())

    class _Loop:
        async def sock_sendall(self, _sock, _data):
            return None

    monkeypatch.setattr("minds.agents.anton_agent.anton.backends.docker.asyncio.get_running_loop", lambda: _Loop())

    async def _read_result(**_kwargs):
        raise asyncio.TimeoutError("timeout")
        yield  # pragma: no cover

    monkeypatch.setattr(rt, "_read_result", _read_result)

    rt.client = SimpleNamespace(containers=SimpleNamespace(get=lambda _name: SimpleNamespace(restart=lambda: None)))
    items = [i async for i in rt.execute_streaming("print(1)", description="d", estimated_seconds=1)]
    assert any(hasattr(x, "error") and "Runtime restarted" in (x.error or "") for x in items)


@pytest.mark.asyncio
async def test_docker_read_result_parses_progress_and_json(monkeypatch):
    from minds.agents.anton_agent.anton.backends.docker import (
        _PROGRESS_MARKER,
        _RESULT_END,
        _RESULT_START,
        DockerScratchpadRuntime,
    )

    rt = DockerScratchpadRuntime(name="x", _extra_env={"ANTON_MINDS_CONVERSATION_ID": "c1"})
    rt.sock = SimpleNamespace(_sock=object())

    lines = [
        f"{_PROGRESS_MARKER} doing\n",
        f"{_RESULT_START}\n",
        '{"stdout":"ok","stderr":"","logs":"","error":null}\n',
        f"{_RESULT_END}\n",
    ]
    payload = "".join(lines).encode()
    chunks = [payload]

    class Loop:
        def __init__(self):
            self.i = 0

        async def sock_recv(self, _sock, _n):
            if self.i >= len(chunks):
                return b""
            b = chunks[self.i]
            self.i += 1
            return b

    loop = Loop()
    monkeypatch.setattr("minds.agents.anton_agent.anton.backends.docker.asyncio.get_running_loop", lambda: loop)

    out = [x async for x in rt._read_result(total_timeout=5, inactivity_timeout=5)]
    assert out[0] == "doing"
    assert out[1]["stdout"] == "ok"


@pytest.mark.asyncio
async def test_docker_read_result_handles_garbage_and_malformed(monkeypatch):
    from minds.agents.anton_agent.anton.backends.docker import _RESULT_END, _RESULT_START, DockerScratchpadRuntime

    rt = DockerScratchpadRuntime(name="x", _extra_env={"ANTON_MINDS_CONVERSATION_ID": "c1"})
    rt.sock = SimpleNamespace(_sock=object())

    payload = (
        f"{_RESULT_START}\n"
        + 'garbage-prefix {"stdout":"ok","stderr":"","logs":"","error":null}\n'
        + f"{_RESULT_END}\n"
    ).encode()
    payload2 = (f"{_RESULT_START}\n{_RESULT_END}\n").encode()

    chunks = [payload, payload2]

    class Loop:
        def __init__(self):
            self.i = 0

        async def sock_recv(self, _sock, _n):
            if self.i >= len(chunks):
                return b""
            b = chunks[self.i]
            self.i += 1
            return b

    loop = Loop()
    monkeypatch.setattr("minds.agents.anton_agent.anton.backends.docker.asyncio.get_running_loop", lambda: loop)

    out1 = [x async for x in rt._read_result(total_timeout=5, inactivity_timeout=5)]
    assert out1[0]["stdout"] == "ok"

    rt2 = DockerScratchpadRuntime(name="y", _extra_env={"ANTON_MINDS_CONVERSATION_ID": "c1"})
    rt2.sock = SimpleNamespace(_sock=object())
    loop2 = Loop()
    loop2.i = 1
    monkeypatch.setattr("minds.agents.anton_agent.anton.backends.docker.asyncio.get_running_loop", lambda: loop2)
    out2 = [x async for x in rt2._read_result(total_timeout=5, inactivity_timeout=5)]
    assert "Malformed result" in out2[0]["error"]


@pytest.mark.asyncio
async def test_docker_start_wires_exec_socket_and_copies_boot(monkeypatch):
    from minds.agents.anton_agent.anton.backends.docker import DockerScratchpadRuntime

    class _UnderlyingSock:
        def __init__(self):
            self.setblocking_calls: list[bool] = []

        def setblocking(self, flag: bool):
            self.setblocking_calls.append(flag)

    underlying = _UnderlyingSock()
    sock_wrapper = SimpleNamespace(_sock=underlying)

    container = SimpleNamespace(id="cid", status="running", start=lambda: None, reload=lambda: None)

    api = SimpleNamespace(
        exec_create=lambda _cid, cmd, stdin, tty: "execid",
        exec_start=lambda _exec_id, socket=True: sock_wrapper,
    )
    client = SimpleNamespace(api=api, containers=SimpleNamespace(get=lambda _name: container))

    rt = DockerScratchpadRuntime(name="x", client=client, _extra_env={"ANTON_MINDS_CONVERSATION_ID": "c1"})
    copied: list[str] = []
    monkeypatch.setattr(rt, "_copy_files_to_container", lambda _c: copied.append("ok"))

    await rt.start()
    assert rt.sock is sock_wrapper
    assert copied == ["ok"]
    assert underlying.setblocking_calls == [False]


@pytest.mark.asyncio
async def test_docker_execute_streaming_when_not_running_yields_error_cell():
    from minds.agents.anton_agent.anton.backends.base import Cell
    from minds.agents.anton_agent.anton.backends.docker import DockerScratchpadRuntime

    rt = DockerScratchpadRuntime(name="x", _extra_env={"ANTON_MINDS_CONVERSATION_ID": "c1"})
    rt.sock = None
    items = [i async for i in rt.execute_streaming("print(1)", description="d")]
    cells = [x for x in items if isinstance(x, Cell)]
    assert cells and "not running" in (cells[0].error or "").lower()


@pytest.mark.asyncio
async def test_docker_read_result_inactivity_timeout_message(monkeypatch):
    from minds.agents.anton_agent.anton.backends.docker import DockerScratchpadRuntime

    rt = DockerScratchpadRuntime(name="x", _extra_env={"ANTON_MINDS_CONVERSATION_ID": "c1"})
    rt.sock = SimpleNamespace(_sock=object())

    class Loop:
        async def sock_recv(self, _sock, _n):
            return b""  # pragma: no cover

    monkeypatch.setattr("minds.agents.anton_agent.anton.backends.docker.asyncio.get_running_loop", lambda: Loop())

    async def _wait_for(_coro, timeout):
        raise asyncio.TimeoutError()

    monkeypatch.setattr("minds.agents.anton_agent.anton.backends.docker.asyncio.wait_for", _wait_for)

    with pytest.raises(asyncio.TimeoutError, match="inactivity"):
        _ = [x async for x in rt._read_result(total_timeout=10, inactivity_timeout=1)]


@pytest.mark.asyncio
async def test_docker_get_or_run_container_starts_if_stopped(monkeypatch):
    from minds.agents.anton_agent.anton.backends.docker import DockerScratchpadRuntime

    started: list[bool] = []

    def _start():
        started.append(True)

    container = SimpleNamespace(status="exited", start=_start, reload=lambda: None)
    client = SimpleNamespace(containers=SimpleNamespace(get=lambda _name: container))
    rt = DockerScratchpadRuntime(name="x", client=client, _extra_env={"ANTON_MINDS_CONVERSATION_ID": "c1"})
    got = rt._get_or_run_container()
    assert got is container
    assert started == [True]


@pytest.mark.asyncio
async def test_docker_get_or_run_container_not_found_runs_container(monkeypatch):
    from minds.agents.anton_agent.anton.backends.docker import DockerScratchpadRuntime, NotFound

    ran: list[str] = []

    def _get(_name):
        raise NotFound("missing")

    def _run(*_a, **_k):
        ran.append("ok")
        return SimpleNamespace(status="running", reload=lambda: None)

    client = SimpleNamespace(containers=SimpleNamespace(get=_get, run=_run))
    rt = DockerScratchpadRuntime(name="x", client=client, _extra_env={"X": "1"})
    _ = rt._get_or_run_container()
    assert ran == ["ok"]


@pytest.mark.asyncio
async def test_docker_get_or_run_container_apierror_409_race_recovers(monkeypatch):
    from minds.agents.anton_agent.anton.backends.docker import APIError, DockerScratchpadRuntime, NotFound

    # Force NotFound -> run() -> APIError(409) -> get() and start if not running.
    calls: list[str] = []
    state = {"ran": False}

    class Container:
        def __init__(self):
            self.status = "exited"

        def start(self):
            calls.append("start")
            self.status = "running"

        def reload(self):
            calls.append("reload")

    container = Container()

    def _get(_name):
        if not state["ran"]:
            raise NotFound("missing")
        return container

    class _Resp:
        status_code = 409

    err = APIError("conflict", response=_Resp())

    def _run(*_a, **_k):
        state["ran"] = True
        raise err

    client = SimpleNamespace(containers=SimpleNamespace(get=_get, run=_run))
    rt = DockerScratchpadRuntime(name="x", client=client, _extra_env={"X": "1"})
    got = rt._get_or_run_container()
    assert got is container
    assert "start" in calls


@pytest.mark.asyncio
async def test_docker_reset_closes_sock_clears_cells_and_restarts(monkeypatch):
    from minds.agents.anton_agent.anton.backends.docker import DockerScratchpadRuntime

    class _Sock:
        def __init__(self):
            self.closed = False
            self._sock = SimpleNamespace(setblocking=lambda _flag: None)

        def close(self):
            self.closed = True
            raise RuntimeError("close fails but is suppressed")

    class Container:
        def __init__(self):
            self.id = "cid"
            self.status = "running"
            self.restart_calls = 0
            self.reload_calls = 0

        def restart(self):
            self.restart_calls += 1

        def reload(self):
            self.reload_calls += 1

    container = Container()

    api = SimpleNamespace(
        exec_create=lambda _cid, cmd, stdin, tty: "execid",
        exec_start=lambda _exec_id, socket=True: SimpleNamespace(_sock=SimpleNamespace(setblocking=lambda _f: None)),
    )
    client = SimpleNamespace(api=api, containers=SimpleNamespace(get=lambda _name: container))

    rt = DockerScratchpadRuntime(name="x", client=client, _extra_env={"X": "1"})
    rt.sock = _Sock()
    rt.cells = [SimpleNamespace()]

    # Make to_thread synchronous for test determinism (but still awaitable).
    async def _to_thread(fn, *a):
        return fn(*a)

    monkeypatch.setattr("minds.agents.anton_agent.anton.backends.docker.asyncio.to_thread", _to_thread)

    copied: list[str] = []
    monkeypatch.setattr(rt, "_copy_files_to_container", lambda _c: copied.append("ok"))

    await rt.reset()
    assert rt.sock is not None
    assert len(rt.cells) == 1
    assert rt.cells[0].error == "Cancelled by user."
    assert copied == ["ok"]
    assert container.restart_calls == 1
    assert container.reload_calls == 2


@pytest.mark.asyncio
async def test_docker_close_cancel_install_packages(monkeypatch):
    from minds.agents.anton_agent.anton.backends.docker import DockerScratchpadRuntime

    class Container:
        def __init__(self):
            self.stop_calls = 0
            self.remove_calls = 0
            self.exec_calls: list[list[str]] = []
            self.restart_calls = 0
            self.reload_calls = 0
            self.id = "cid"
            self.status = "running"

        def stop(self):
            self.stop_calls += 1

        def remove(self):
            self.remove_calls += 1

        def restart(self):
            self.restart_calls += 1

        def reload(self):
            self.reload_calls += 1

        def exec_run(self, cmd):
            self.exec_calls.append(list(cmd))

    container = Container()
    api = SimpleNamespace(
        exec_create=lambda _cid, cmd, stdin, tty: "execid",
        exec_start=lambda _exec_id, socket=True: SimpleNamespace(_sock=SimpleNamespace(setblocking=lambda _f: None)),
    )
    client = SimpleNamespace(api=api, containers=SimpleNamespace(get=lambda _name: container))
    rt = DockerScratchpadRuntime(name="x", client=client, _extra_env={"X": "1"})

    await rt.close(cleanup=False)
    assert container.stop_calls == 1
    assert container.remove_calls == 0

    await rt.close(cleanup=True)
    assert container.stop_calls == 2
    assert container.remove_calls == 1

    # cancel() restarts the container and reconnects the socket.
    monkeypatch.setattr(rt, "_copy_files_to_container", lambda _c: None)
    await rt.cancel()
    assert container.restart_calls == 1
    assert container.reload_calls >= 1
    assert rt.sock is not None

    msg = await rt.install_packages(["requests", "numpy"])
    assert msg == "Packages requests, numpy installed."
    assert container.exec_calls == [["pip", "install", "requests", "numpy"]]


@pytest.mark.asyncio
async def test_docker_execute_streaming_auto_installed_and_missing_result_data(monkeypatch):
    from minds.agents.anton_agent.anton.backends.base import Cell
    from minds.agents.anton_agent.anton.backends.docker import DockerScratchpadRuntime

    rt = DockerScratchpadRuntime(name="x", _extra_env={"X": "1"})
    rt.sock = SimpleNamespace(_sock=object())

    class _Loop:
        async def sock_sendall(self, _sock, _data):
            return None

    monkeypatch.setattr("minds.agents.anton_agent.anton.backends.docker.asyncio.get_running_loop", lambda: _Loop())

    async def _read_result_auto(**_kwargs):
        yield {"stdout": "out", "stderr": "", "error": None, "logs": "", "auto_installed": ["NumPy", "PANDAS"]}

    monkeypatch.setattr(rt, "_read_result", _read_result_auto)
    items = [i async for i in rt.execute_streaming("x=1", estimated_seconds=1)]
    cell = next(x for x in items if isinstance(x, Cell))
    assert cell.stdout == "out"
    assert rt._installed_packages == {"numpy", "pandas"}

    async def _read_result_no_final(**_kwargs):
        yield "p1"
        return

    monkeypatch.setattr(rt, "_read_result", _read_result_no_final)
    items2 = [i async for i in rt.execute_streaming("x=2", estimated_seconds=1)]
    cell2 = next(x for x in items2 if isinstance(x, Cell))
    assert "Process exited unexpectedly" in (cell2.error or "")


@pytest.mark.asyncio
async def test_docker_read_result_stray_end_marker_and_decode_error_and_exit(monkeypatch):
    import json as _json

    from minds.agents.anton_agent.anton.backends.docker import (
        _RESULT_END,
        _RESULT_START,
        DockerScratchpadRuntime,
    )

    rt = DockerScratchpadRuntime(name="x", _extra_env={"X": "1"})
    rt.sock = SimpleNamespace(_sock=object())

    # 1) Stray _RESULT_END before start should be ignored; then valid JSON should parse.
    payload1 = (f"{_RESULT_END}\n{_RESULT_START}\n" + _json.dumps({"stdout": "ok"}) + f"\n{_RESULT_END}\n").encode()

    class Loop1:
        def __init__(self):
            self.done = False

        async def sock_recv(self, _sock, _n):
            if self.done:
                return b""
            self.done = True
            return payload1

    monkeypatch.setattr("minds.agents.anton_agent.anton.backends.docker.asyncio.get_running_loop", lambda: Loop1())
    out = [x async for x in rt._read_result(total_timeout=5, inactivity_timeout=5)]
    assert out[0]["stdout"] == "ok"

    # 2) JSON decode failure should yield the descriptive error dict.
    rt2 = DockerScratchpadRuntime(name="y", _extra_env={"X": "1"})
    rt2.sock = SimpleNamespace(_sock=object())
    payload2 = (f"{_RESULT_START}\nnot json at all\n{_RESULT_END}\n").encode()

    class Loop2:
        def __init__(self):
            self.done = False

        async def sock_recv(self, _sock, _n):
            if self.done:
                return b""
            self.done = True
            return payload2

    monkeypatch.setattr("minds.agents.anton_agent.anton.backends.docker.asyncio.get_running_loop", lambda: Loop2())
    out2 = [x async for x in rt2._read_result(total_timeout=5, inactivity_timeout=5)]
    assert "Failed to decode Docker result JSON" in out2[0]["error"]

    # 3) Immediate EOF yields "Process exited unexpectedly."
    rt3 = DockerScratchpadRuntime(name="z", _extra_env={"X": "1"})
    rt3.sock = SimpleNamespace(_sock=object())

    class Loop3:
        async def sock_recv(self, _sock, _n):
            return b""  # immediate EOF

    monkeypatch.setattr("minds.agents.anton_agent.anton.backends.docker.asyncio.get_running_loop", lambda: Loop3())

    async def _wait_for(_coro, timeout):
        return b""

    monkeypatch.setattr("minds.agents.anton_agent.anton.backends.docker.asyncio.wait_for", _wait_for)
    out3 = [x async for x in rt3._read_result(total_timeout=5, inactivity_timeout=5)]
    assert out3[0]["error"] == "Process exited unexpectedly."


@pytest.mark.asyncio
async def test_docker_read_result_total_timeout_message(monkeypatch):
    # Covers the "timed out after Xs total" message in the wait_for timeout handler.
    import time

    from minds.agents.anton_agent.anton.backends.docker import DockerScratchpadRuntime

    rt = DockerScratchpadRuntime(name="x", _extra_env={"X": "1"})
    rt.sock = SimpleNamespace(_sock=object())

    class Loop:
        async def sock_recv(self, _sock, _n):
            return b""  # pragma: no cover

    monkeypatch.setattr("minds.agents.anton_agent.anton.backends.docker.asyncio.get_running_loop", lambda: Loop())

    async def _wait_for(_coro, timeout):
        raise asyncio.TimeoutError()

    monkeypatch.setattr("minds.agents.anton_agent.anton.backends.docker.asyncio.wait_for", _wait_for)

    # Force elapsed >= total_timeout - 0.5 inside the exception handler.
    t = {"v": 0.0}

    def _mono():
        t["v"] += 10.0
        return t["v"]

    monkeypatch.setattr(time, "monotonic", _mono)

    with pytest.raises(asyncio.TimeoutError, match="timed out after"):
        _ = [x async for x in rt._read_result(total_timeout=1, inactivity_timeout=1)]
