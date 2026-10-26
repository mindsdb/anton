from __future__ import annotations

import json
import subprocess
import sys

import pytest

# The top-level packages the MCP SDK installs.
_MCP_SDK_PACKAGES = ("mcp", "mcp_types")


def _sdk_modules_after_importing(*, module: str) -> list[str]:
    """The MCP SDK modules a fresh interpreter holds after importing `module`.
    A fresh one, because this test process imported the SDK already."""
    probe = (
        "import importlib, json, sys\n"
        f"importlib.import_module({module!r})\n"
        f"print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in {_MCP_SDK_PACKAGES!r})))\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


@pytest.mark.parametrize("module", ["anton.core.mcp.names", "anton.core.mcp.access"])
def test_importing_names_or_access_does_not_import_the_mcp_sdk(module):
    """A caller that only splits a stored tool row's name imports `names`,
    and one that asks whether a tool returns records imports `access`.
    cowork-server's figure sources import both, and must not load the MCP
    SDK to do it. `registry` does load it, which shows the probe sees the
    SDK when it is there."""
    assert _sdk_modules_after_importing(module="anton.core.mcp.registry")
    assert _sdk_modules_after_importing(module=module) == []
