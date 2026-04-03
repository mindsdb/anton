#!/bin/sh
# Anton install script
# curl -sSf https://raw.githubusercontent.com/mindsdb/anton/main/install.sh | sh && export PATH="$HOME/.local/bin:$PATH" && anton
# Pure POSIX sh, no sudo, idempotent.
# Pass --force to skip confirmation prompts.

set -e

CYAN='\033[36m'
GREEN='\033[32m'
RED='\033[31m'
YELLOW='\033[33m'
BOLD='\033[1m'
RESET='\033[0m'

LOCAL_BIN="$HOME/.local/bin"

FORCE=0
for arg in "$@"; do
    case "$arg" in
        --force|-f) FORCE=1 ;;
    esac
done

info()  { printf "%b\n" "$1"; }
warn()  { printf "${YELLOW}warning:${RESET} %s\n" "$1"; }
error() { printf "${RED}error:${RESET} %s\n" "$1" >&2; }

# Prompt for confirmation. Auto-yes when --force or non-interactive (piped).
confirm() {
    if [ "$FORCE" -eq 1 ]; then
        return 0
    fi
    if [ ! -t 0 ]; then
        # Non-interactive (piped install) — proceed without prompting
        return 0
    fi
    printf "%b" "  $1 [Y/n] "
    read -r REPLY
    case "$REPLY" in
        [nN]*) return 1 ;;
        *)     return 0 ;;
    esac
}

# ── 1. Branded logo ────────────────────────────────────────────────
info ""
info "${CYAN} ▄▀█ █▄ █ ▀█▀ █▀█ █▄ █${RESET}"
info "${CYAN} █▀█ █ ▀█  █  █▄█ █ ▀█${RESET}"
info "${CYAN} autonomous coworker${RESET}"
info ""

# ── 2. Check prerequisites ──────────────────────────────────────────
if ! command -v git >/dev/null 2>&1; then
    error "git is required but not found."
    info "  Install it with your package manager:"
    info "    macOS:  xcode-select --install"
    info "    Ubuntu: sudo apt install git"
    info "    Fedora: sudo dnf install git"
    exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
    error "curl is required but not found."
    info "  Install it with your package manager:"
    info "    Ubuntu: sudo apt install curl"
    info "    Fedora: sudo dnf install curl"
    exit 1
fi

# ── 3. Resolve latest release tag ─────────────────────────────────
info "  Fetching latest release..."
LATEST_TAG=$(curl -sSf "https://api.github.com/repos/mindsdb/anton/releases/latest" \
    | grep '"tag_name"' | head -1 \
    | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/')
if [ -z "$LATEST_TAG" ]; then
    warn "Could not determine latest release tag. Falling back to main branch."
    REPO_URL="git+https://github.com/mindsdb/anton.git"
else
    REPO_URL="git+https://github.com/mindsdb/anton.git@${LATEST_TAG}"
fi

# ── 4. Find or install uv ──────────────────────────────────────────
NEED_UV=0

if command -v uv >/dev/null 2>&1; then
    info "  Found uv: $(command -v uv)"
elif [ -f "$HOME/.local/bin/uv" ]; then
    export PATH="$LOCAL_BIN:$PATH"
    info "  Found uv: $HOME/.local/bin/uv"
elif [ -f "$HOME/.cargo/bin/uv" ]; then
    export PATH="$HOME/.cargo/bin:$PATH"
    info "  Found uv: $HOME/.cargo/bin/uv"
else
    NEED_UV=1
fi

if [ "$NEED_UV" -eq 1 ]; then
    warn "uv is not installed. It is required to manage anton's isolated environment."
    info "  uv will be installed to ~/.local/bin via https://astral.sh/uv/install.sh"
    if confirm "Install uv?"; then
        info "  Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
        # Source uv's env setup if available
        if [ -f "$HOME/.local/bin/env" ]; then
            . "$HOME/.local/bin/env"
        else
            export PATH="$LOCAL_BIN:$PATH"
        fi
        info "  Installed uv: $(command -v uv)"
    else
        error "uv is required. Install it manually: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
fi

# ── 5. Install anton via uv tool ───────────────────────────────────
info ""
info "  This will install:"
info "    - ${BOLD}anton${RESET} (from ${REPO_URL})"
info "    - Into an isolated virtual environment managed by uv"
info "    - Python 3.11+ will be downloaded automatically if not present"
info ""

if confirm "Proceed with installation?"; then
    info "  Installing anton into an isolated venv..."
    uv tool install "$REPO_URL" --force
    info "  Installed anton"
else
    info "  Installation cancelled."
    exit 0
fi

# ── 6. Verify the venv was created ─────────────────────────────────
UV_TOOL_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/uv/tools/anton"
if [ -d "$UV_TOOL_DIR" ]; then
    info "  Venv: ${UV_TOOL_DIR}"
else
    warn "Could not verify anton's virtual environment."
    info "  Expected at: ${UV_TOOL_DIR}"
    info "  Anton may still work — uv manages the environment internally."
fi

# ── 7. Ensure ~/.local/bin is in PATH ──────────────────────────────
ensure_path() {
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
        fish) SHELL_RC="$HOME/.config/fish/config.fish" ;;
        *)    SHELL_RC="$HOME/.profile" ;;
    esac

    # Only append if not already present
    if [ -f "$SHELL_RC" ] && grep -qF '.local/bin' "$SHELL_RC" 2>/dev/null; then
        return
    fi

    if [ "$SHELL_NAME" = "fish" ]; then
        mkdir -p "$(dirname "$SHELL_RC")"
        printf '\n# Added by anton installer\nfish_add_path %s\n' "$LOCAL_BIN" >> "$SHELL_RC"
    else
        printf '\n# Added by anton installer\nexport PATH="$HOME/.local/bin:$PATH"\n' >> "$SHELL_RC"
    fi
    info "  Updated ${SHELL_RC}"
}

ensure_path

# ── 8. Scratchpad health check ────────────────────────────────────
# Verify that uv can create a venv — this is what the scratchpad uses at runtime.
# Catches broken Python symlinks, missing venv module, etc. before the user hits it.
info ""
info "  Running scratchpad health check..."
HEALTH_DIR=$(mktemp -d "${TMPDIR:-/tmp}/anton_healthcheck_XXXXXX")
if uv venv "$HEALTH_DIR/venv" --system-site-packages --seed --quiet 2>/dev/null; then
    HEALTH_PYTHON="$HEALTH_DIR/venv/bin/python"
    if [ -f "$HEALTH_PYTHON" ] && "$HEALTH_PYTHON" -c "print('ok')" >/dev/null 2>&1; then
        info "  ${GREEN}✓${RESET} Scratchpad environment OK"
    else
        warn "uv created a venv but the Python binary is broken."
        info "  This usually means a Homebrew Python upgrade left stale symlinks."
        info "  Fix with: ${BOLD}brew reinstall python${RESET}"
    fi
else
    warn "uv venv creation failed. The scratchpad may not work."
    info "  Try: ${BOLD}uv python install 3.12${RESET}"
    info "  Or:  ${BOLD}brew reinstall python${RESET}"
fi
rm -rf "$HEALTH_DIR" 2>/dev/null

# ── 9. Success message ──────────────────────────────────────────────
info ""
info "${GREEN}  ✓ anton installed successfully!${RESET}"
info ""
info "  Upgrade:    uv tool upgrade anton"
info "  Uninstall:  uv tool uninstall anton"
info ""
info "  Config: ~/.anton/.env"
info ""
