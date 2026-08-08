import subprocess
import sys
import textwrap
from pathlib import Path

ANTON_ROOT = Path(__file__).resolve().parents[2]  # .../anton


def test_sqlite_path_does_not_import_dynamo_or_boto3(tmp_path):
    code = textwrap.dedent(
        f"""
        import sys
        from anton_state import open_store, StateSchema, Attr
        s = open_store(StateSchema(pk=Attr(name="pk")), state=None, local_path=r"{tmp_path}/x.db")
        assert "anton_state.dynamo_driver" not in sys.modules, "dynamo eagerly imported"
        assert "boto3" not in sys.modules, "boto3 eagerly imported"
        print("OK")
        """
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ANTON_ROOT)
    )
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout
