"""The shared auth rules module that other services carry byte for byte.

auth and the call gateway keep exact copies of
``anton/core/tools/connector_auth_rules.py`` and pin its SHA-256, so these
tests pin what makes a copy safe to carry: its hash, and that it loads with
the standard library alone.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

_MODULE = Path(__file__).resolve().parents[1] / "anton" / "core" / "tools" / "connector_auth_rules.py"
_MODULE_SHA256 = "e38ba5cb3ee2e81f3d11b6fdbdd01acd46ada355bd910e90b114a48841399d21"
#: Copies to update, and the hash their tests pin, whenever the module changes.
_COPIES = (
    "auth: auth/datavault/connector_auth_rules.py",
    "mindshub_inference: the custom API gateway's copy",
)
#: Loads the module from its file in a fresh interpreter and prints what it imported.
_LOAD_ALONE = (
    "import importlib.util, json, sys\n"
    "before = set(sys.modules)\n"
    "spec = importlib.util.spec_from_file_location('connector_auth_rules', sys.argv[1])\n"
    "module = importlib.util.module_from_spec(spec)\n"
    "spec.loader.exec_module(module)\n"
    "print(json.dumps(sorted(set(sys.modules) - before)))\n"
)


def test_the_module_hash_is_pinned_so_its_copies_are_resynced():
    digest = hashlib.sha256(_MODULE.read_bytes()).hexdigest()

    assert digest == _MODULE_SHA256, (
        "connector_auth_rules.py changed. Copy it byte for byte to: "
        + "; ".join(_COPIES)
        + f". Then set the pinned SHA-256 here and in each copy's test to {digest}."
    )


def test_the_module_loads_with_the_standard_library_alone():
    result = subprocess.run(
        [sys.executable, "-I", "-c", _LOAD_ALONE, str(_MODULE)], capture_output=True, text=True, check=True
    )

    loaded = json.loads(result.stdout)
    top_level = {name.split(".")[0] for name in loaded}
    assert top_level <= set(sys.stdlib_module_names) | {"connector_auth_rules"}
