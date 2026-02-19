#!/bin/sh
# Anton install script — curl -sSf https://raw.githubusercontent.com/mindsdb/anton/main/install.sh | sh
# Pure POSIX sh, no sudo, idempotent.

set -e

CYAN='\033[36m'
GREEN='\033[32m'
RED='\033[31m'
BOLD='\033[1m'
RESET='\033[0m'

ANTON_HOME="$HOME/.anton"
VENV_DIR="$ANTON_HOME/venv"
LOCAL_BIN="$HOME/.local/bin"
REPO_URL="git+https://github.com/mindsdb/anton.git"

info()  { printf "%b\n" "$1"; }
error() { printf "${RED}error:${RESET} %s\n" "$1" >&2; }

# ── 1. Branded logo ────────────────────────────────────────────────
info ""
info "${CYAN} ▄▀█ █▄ █ ▀█▀ █▀█ █▄ █${RESET}"
info "${CYAN} █▀█ █ ▀█  █  █▄█ █ ▀█${RESET}"
info "${CYAN} autonomous coworker${RESET}"
info ""

# ── 2. Check for / install uv ──────────────────────────────────────
USE_UV=1

if command -v uv >/dev/null 2>&1; then
    info "  Found uv: $(command -v uv)"
elif [ -f "$HOME/.local/bin/uv" ]; then
    export PATH="$LOCAL_BIN:$PATH"
    info "  Found uv: $HOME/.local/bin/uv"
else
    info "  Installing uv..."
    if curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; then
        # Source uv's env setup if available
        if [ -f "$HOME/.local/bin/env" ]; then
            . "$HOME/.local/bin/env"
        else
            export PATH="$LOCAL_BIN:$PATH"
        fi
        info "  Installed uv: $(command -v uv)"
    else
        info "  Could not install uv — falling back to python3 + pip"
        USE_UV=0
    fi
fi

# Verify python3 exists if falling back
if [ "$USE_UV" -eq 0 ]; then
    if ! command -v python3 >/dev/null 2>&1; then
        error "python3 not found. Please install Python 3 and try again."
        exit 1
    fi
fi

# ── 3. Create venv and install anton ───────────────────────────────
info "  Creating venv at ${ANTON_HOME}/venv..."

if [ "$USE_UV" -eq 1 ]; then
    uv venv "$VENV_DIR" --quiet 2>/dev/null || uv venv "$VENV_DIR"
    info "  Installing anton..."
    uv pip install --python "$VENV_DIR/bin/python" "$REPO_URL" --quiet 2>/dev/null \
        || uv pip install --python "$VENV_DIR/bin/python" "$REPO_URL"
else
    python3 -m venv "$VENV_DIR"
    info "  Installing anton..."
    "$VENV_DIR/bin/pip" install --upgrade pip --quiet
    "$VENV_DIR/bin/pip" install "$REPO_URL" --quiet
fi

# Verify the binary was created
if [ ! -f "$VENV_DIR/bin/anton" ]; then
    error "Installation finished but anton binary not found at $VENV_DIR/bin/anton"
    exit 1
fi

info "  Installed anton into ${VENV_DIR}"

# ── 4. Add anton to PATH ──────────────────────────────────────────
add_to_path() {
    mkdir -p "$LOCAL_BIN"
    ln -sf "$VENV_DIR/bin/anton" "$LOCAL_BIN/anton"
    info "  Linked ${LOCAL_BIN}/anton"

    # Check if ~/.local/bin is already in PATH
    case ":$PATH:" in
        *":$LOCAL_BIN:"*) return ;;
    esac

    # Detect shell config file
    SHELL_NAME="$(basename "$SHELL" 2>/dev/null || echo "sh")"
    case "$SHELL_NAME" in
        zsh)  SHELL_RC="$HOME/.zshrc" ;;
        bash)
            if [ -f "$HOME/.bash_profile" ]; then
                SHELL_RC="$HOME/.bash_profile"
            else
                SHELL_RC="$HOME/.bashrc"
            fi
            ;;
        *)    SHELL_RC="$HOME/.profile" ;;
    esac

    LINE='export PATH="$HOME/.local/bin:$PATH"'

    # Only append if not already present
    if [ -f "$SHELL_RC" ] && grep -qF '.local/bin' "$SHELL_RC" 2>/dev/null; then
        return
    fi

    printf '\n# Added by anton installer\n%s\n' "$LINE" >> "$SHELL_RC"
    info "  Updated ${SHELL_RC}"
}

# If stdin is a terminal, prompt; otherwise default to yes (piped install)
if [ -t 0 ]; then
    printf "  Add anton to your PATH? [Y/n] "
    read -r REPLY
    case "$REPLY" in
        [nN]*) ADD_PATH=0 ;;
        *)     ADD_PATH=1 ;;
    esac
else
    ADD_PATH=1
fi

if [ "$ADD_PATH" -eq 1 ]; then
    add_to_path
else
    info ""
    info "  To use anton, add this to your PATH:"
    info "    export PATH=\"${VENV_DIR}/bin:\$PATH\""
fi

# ── 5. Success message ────────────────────────────────────────────
info ""
info "${GREEN}  ✓ anton installed successfully!${RESET}"
info ""
info "  Get started:"
info "    anton                     ${CYAN}# Dashboard${RESET}"
info "    anton run \"fix the tests\" ${CYAN}# Run a task${RESET}"
info ""
info "  Config: ~/.anton/.env"
info ""
