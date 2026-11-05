"""Execute the shipped installers' release sections without changing user state."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TAG = "v9.8.7"
TAG_OBJECT = "f" * 40
COMMIT = "1" * 40
REPOSITORY = "https://github.com/mindsdb/anton.git"
CONTRACT = "fixture-dependency==1.2.3\n"
INSTALLERS = [
    pytest.param("shell", marks=pytest.mark.skipif(os.name == "nt" or shutil.which("sh") is None, reason="requires POSIX sh")),
    pytest.param("powershell", marks=pytest.mark.skipif(shutil.which("pwsh") is None, reason="requires PowerShell")),
]


def release_sections(source):
    """Keep the real control flow; omit prerequisites, profile edits and app startup."""
    headings = list(re.finditer(r"^# ── \d+\. ([^\n]+)$", source, re.MULTILINE))
    sections = []
    for title in ("Resolve latest release tag", "Install anton via uv tool"):
        matches = [index for index, heading in enumerate(headings) if heading[1].startswith(title)]
        assert len(matches) == 1, f"Expected one installer section: {title}"
        index = matches[0]
        end = headings[index + 1].start() if index + 1 < len(headings) else len(source)
        sections.append(source[headings[index].end():end])
    return "\n".join(sections)


SHELL_COMMAND = r'''
import json
import os
from pathlib import Path
import sys

name, args = Path(sys.argv[0]).name, sys.argv[1:]
record = {"command": name, "args": args}
if name == "uv" and "--constraints" in args:
    contract = Path(args[args.index("--constraints") + 1])
    record.update(contract=contract.read_text(), contract_path=str(contract))
with open(os.environ["ANTON_TEST_LOG"], "a") as output:
    output.write(json.dumps(record) + "\n")
if name == "curl":
    url = next(value for value in args if value.startswith("https://"))
    if url == "https://api.github.com/repos/mindsdb/anton/releases/latest":
        print(json.dumps({"tag_name": os.environ["ANTON_TEST_TAG"]}))
    elif url.startswith("https://raw.githubusercontent.com/mindsdb/anton/"):
        if os.environ["ANTON_TEST_EXPORT"] == "missing":
            raise SystemExit(22)
        content = "" if os.environ["ANTON_TEST_EXPORT"] == "empty" else os.environ["ANTON_TEST_CONTRACT"]
        Path(args[args.index("-o") + 1]).write_text(content)
    else:
        raise SystemExit("Unexpected download: " + url)
elif name == "git":
    print(os.environ["ANTON_TEST_REFS"])
elif name != "uv":
    raise SystemExit("Unexpected command: " + name)
'''


POWERSHELL_COMMANDS = r'''
$ErrorActionPreference = "Stop"
function Write-TestRecord($record) {
    Add-Content -LiteralPath $env:ANTON_TEST_LOG -Value ($record | ConvertTo-Json -Compress) -Encoding utf8
}
function Confirm-Step { return $true }
function Invoke-RestMethod {
    param([string]$Uri)
    Write-TestRecord @{ command = "release"; args = @($Uri) }
    if ($Uri -ne "https://api.github.com/repos/mindsdb/anton/releases/latest") { throw "Unexpected release endpoint" }
    return [pscustomobject]@{ tag_name = $env:ANTON_TEST_TAG }
}
function git {
    Write-TestRecord @{ command = "git"; args = @($args) }
    $global:LASTEXITCODE = 0
    return $env:ANTON_TEST_REFS -split "`n"
}
function Invoke-WebRequest {
    param([Parameter(Position=0)][string]$Uri, [switch]$UseBasicParsing, [string]$OutFile)
    Write-TestRecord @{ command = "contract"; args = @($Uri, $OutFile) }
    if ($env:ANTON_TEST_EXPORT -eq "missing") { throw "HTTP 404: no release contract" }
    $content = if ($env:ANTON_TEST_EXPORT -eq "empty") { "" } else { $env:ANTON_TEST_CONTRACT }
    [System.IO.File]::WriteAllText($OutFile, $content)
}
function uv {
    $record = @{ command = "uv"; args = @($args) }
    $index = [Array]::IndexOf($args, "--constraints")
    if ($index -ge 0) {
        $record.contract_path = $args[$index + 1]
        $record.contract = [System.IO.File]::ReadAllText($record.contract_path)
    }
    Write-TestRecord $record
    $global:LASTEXITCODE = 0
}
'''


def run_installer(tmp_path, kind, *, refs, tag=TAG, export="present"):
    # Spaces exercise quoting of both executable and temporary contract paths.
    fixture = tmp_path / "release fixture"
    fixture.mkdir()
    log = fixture / "calls.jsonl"
    environment = {
        **os.environ,
        "ANTON_TEST_LOG": str(log),
        "ANTON_TEST_TAG": tag,
        "ANTON_TEST_REFS": refs,
        "ANTON_TEST_EXPORT": export,
        "ANTON_TEST_CONTRACT": CONTRACT,
        "TMPDIR": str(fixture),
        "TMP": str(fixture),
        "TEMP": str(fixture),
    }
    if kind == "shell":
        commands = fixture / "bin"
        commands.mkdir()
        for name in ("curl", "git", "uv"):
            command = commands / name
            command.write_text(f"#!{sys.executable}\n" + SHELL_COMMAND)
            command.chmod(0o755)
        environment["PATH"] = str(commands) + os.pathsep + os.environ["PATH"]
        script = fixture / "install.sh"
        script.write_text("set -e\ninfo() { :; }\nwarn() { :; }\nerror() { echo \"$1\" >&2; }\nconfirm() { return 0; }\n" + release_sections((ROOT / "install.sh").read_text(encoding="utf-8")), encoding="utf-8")
        command = [shutil.which("sh"), str(script)]
    else:
        script = fixture / "install.ps1"
        script.write_text(POWERSHELL_COMMANDS + release_sections((ROOT / "install.ps1").read_text(encoding="utf-8")), encoding="utf-8")
        command = [shutil.which("pwsh"), "-NoProfile", "-NonInteractive", "-File", str(script)]
    result = subprocess.run(command, cwd=fixture, env=environment, capture_output=True, text=True, timeout=15)
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8-sig").splitlines()] if log.exists() else []
    return result, calls


@pytest.mark.parametrize("kind", INSTALLERS)
@pytest.mark.parametrize("tag_kind", ["annotated", "annotated-reversed", "lightweight"])
def test_code_and_contract_use_the_same_peeled_release_commit(tmp_path, kind, tag_kind):
    refs = [f"{TAG_OBJECT}\trefs/tags/{TAG}", f"{COMMIT}\trefs/tags/{TAG}^{{}}"]
    if tag_kind == "annotated-reversed":
        refs.reverse()
    elif tag_kind == "lightweight":
        refs = [f"{COMMIT}\trefs/tags/{TAG}"]
    result, calls = run_installer(tmp_path, kind, refs="\n".join(refs))
    assert result.returncode == 0, result.stdout + result.stderr
    git = [call for call in calls if call["command"] == "git"]
    assert [call["args"] for call in git] == [["ls-remote", REPOSITORY, f"refs/tags/{TAG}", f"refs/tags/{TAG}^{{}}"]]
    installs = [call for call in calls if call["command"] == "uv"]
    assert len(installs) == 1
    install = installs[0]
    assert "contract_path" in install, "uv must receive the downloaded release contract"
    assert install["args"] == ["tool", "install", f"git+{REPOSITORY}@{COMMIT}", "--force", "--constraints", install["contract_path"]]
    assert install["contract"] == CONTRACT
    contract_url = f"https://raw.githubusercontent.com/mindsdb/anton/{COMMIT}/anton/release-requirements.txt"
    downloads = [call for call in calls if contract_url in call["args"]]
    assert len(downloads) == 1
    assert install["contract_path"] in downloads[0]["args"]
    assert calls.index(downloads[0]) < calls.index(install)
    assert not Path(install["contract_path"]).exists(), "The temporary contract must be cleaned up"


@pytest.mark.parametrize("kind", INSTALLERS)
@pytest.mark.parametrize("export", ["missing", "empty"])
def test_missing_or_empty_export_refuses_install(tmp_path, kind, export):
    result, calls = run_installer(tmp_path, kind, refs=f"{COMMIT}\trefs/tags/{TAG}", export=export)
    assert result.returncode != 0, "An unavailable release contract must stop installation"
    assert not any(call["command"] == "uv" for call in calls)


@pytest.mark.parametrize("kind", INSTALLERS)
@pytest.mark.parametrize("failure", ["missing-tag", "invalid-commit"])
def test_unresolved_release_never_falls_back_to_a_branch(tmp_path, kind, failure):
    result, calls = run_installer(
        tmp_path, kind,
        refs="invalid\trefs/tags/" + TAG if failure == "invalid-commit" else "",
        tag="" if failure == "missing-tag" else TAG,
    )
    assert result.returncode != 0, "An unresolved release must stop installation"
    assert not any(call["command"] == "uv" for call in calls)
