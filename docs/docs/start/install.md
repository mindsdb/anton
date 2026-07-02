---
title: Installation
description: Every way to install Anton — install script, desktop app, PATH setup, firewall notes, and troubleshooting.
---

# Installation

Anton is an open-source AI coworker that can execute tasks, connect to tools
and data, remember lessons, and improve its workflows over time. If you just
want to get going, the [Quickstart](/start/quickstart) covers the happy path
in five minutes. This page covers the details: what the installer actually
does, the desktop apps, PATH setup, Windows firewall, and troubleshooting.

## Install script (macOS / Linux)

```bash
curl -sSf https://raw.githubusercontent.com/mindsdb/anton/main/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
```

The script is plain POSIX shell, needs no sudo, and is idempotent — safe to
re-run. It does the following:

1. Checks that `git` and `curl` are available (and tells you how to install them if not).
2. Resolves the latest Anton release tag from GitHub.
3. Finds `uv`, or offers to install it to `~/.local/bin` via the official Astral installer.
4. Runs `uv tool install` to put Anton in an isolated virtual environment under `~/.local/share/uv/tools/anton`. Python 3.11+ is downloaded automatically if not present.
5. Adds `~/.local/bin` to your `PATH` in your shell config (`.zshrc`, `.bashrc`/`.bash_profile`, fish config, or `.profile`).
6. Runs a scratchpad health check — verifies `uv` can create a working Python virtual environment, since that is what Anton's scratchpad uses at runtime.

Pass `--force` to skip all confirmation prompts. When piped from `curl` the
script is non-interactive and proceeds without prompting.

## Install script (Windows PowerShell)

```powershell
irm https://raw.githubusercontent.com/mindsdb/anton/main/install.ps1 | iex
```

The Windows installer additionally:

- Checks your PowerShell execution policy and offers to set it to `RemoteSigned` for the current user if it is too restrictive.
- Checks for `git` (install with `winget install Git.Git` if missing).
- Offers to add Windows Firewall rules so the scratchpad's Python can reach the internet (one UAC prompt).
- Adds `~\.local\bin` to your user `PATH`.

## Desktop app

Anton is also available as a desktop application wrapping the same engine:

- **macOS**: [anton-latest.pkg](https://downloads.mindshub.ai/anton/mac/anton-latest.pkg)
- **Windows**: [anton-latest.exe](https://downloads.mindshub.ai/anton/windows/anton-latest.exe)

See [Desktop app](/use/desktop) for more.

## Windows scratchpad firewall

Windows Firewall blocks new executables by default, which can make the
scratchpad's network calls time out. The installer offers to add the rules for
you; if you skipped that step, run this in an elevated PowerShell:

```powershell
netsh advfirewall firewall add rule name="Anton Scratchpad" dir=out action=allow program="$env:USERPROFILE\.anton\scratchpad-venv\Scripts\python.exe"
```

Each scratchpad gets its own virtual environment under
`~\.anton\scratchpad-venvs\`, so if a new scratchpad's internet calls time
out, re-run the install script or add a rule pointing at that scratchpad's
`Scripts\python.exe`.

## Verify the install

Open a new terminal (so the `PATH` change takes effect) and run:

```bash
anton version
```

You should see something like `Anton v2.26.4.30.0`. Then just run `anton` to
start.

## Troubleshooting

**`anton: command not found`** — your shell hasn't picked up
`~/.local/bin` yet. Open a new terminal, or run
`export PATH="$HOME/.local/bin:$PATH"` (macOS/Linux).

**Missing dependencies** — if Anton starts but detects missing Python
packages, it lists them and offers to install them with `uv` on the spot,
then restarts itself. If `uv` isn't found, it prints the exact `pip install`
or reinstall command to run.

**Broken scratchpad venv (macOS)** — if the installer's health check warns
about a broken Python binary, a Homebrew Python upgrade likely left stale
symlinks. Fix with `brew reinstall python`, or install a managed Python with
`uv python install 3.12`.

## Upgrading and uninstalling

Anton auto-updates at startup (see [Updating Anton](/start/updating)). You can
also manage it directly through `uv`:

```bash
uv tool upgrade anton      # upgrade manually
uv tool uninstall anton    # uninstall
```

Uninstalling removes the program but leaves your configuration and memory in
`~/.anton/` (and any per-project `.anton/` folders). Delete those directories
manually if you want a clean slate — note that `~/.anton/.env` contains your
API keys.
