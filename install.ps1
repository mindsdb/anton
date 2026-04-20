#Requires -Version 5.1
<#
.SYNOPSIS
    Install Anton — autonomous AI coworker.
.DESCRIPTION
    Downloads and installs Anton via uv tool install into an isolated virtual environment.
    Run: irm https://raw.githubusercontent.com/mindsdb/anton/main/install.ps1 | iex
.PARAMETER Force
    Skip all confirmation prompts.
#>
param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Confirm-Step {
    param([string]$Message)
    if ($Force) { return $true }
    $reply = Read-Host "  $Message [Y/n]"
    return ($reply -eq '' -or $reply -match '^[yY]')
}

# ── 1. Branded logo ────────────────────────────────────────────────
Write-Host ""
Write-Host "  ▄▀█ █▄ █ ▀█▀ █▀█ █▄ █" -ForegroundColor Cyan
Write-Host "  █▀█ █ ▀█  █  █▄█ █ ▀█" -ForegroundColor Cyan
Write-Host "  autonomous coworker" -ForegroundColor Cyan
Write-Host ""

# ── 2. Check execution policy ─────────────────────────────────────────
$policy = Get-ExecutionPolicy -Scope CurrentUser
if ($policy -eq "Restricted" -or $policy -eq "Undefined") {
    $globalPolicy = Get-ExecutionPolicy -Scope LocalMachine
    if ($globalPolicy -eq "Restricted" -or $globalPolicy -eq "Undefined") {
        Write-Host "  PowerShell execution policy is too restrictive ($policy)." -ForegroundColor Yellow
        Write-Host "  Anton needs 'RemoteSigned' to install and run scripts."
        Write-Host ""
        if (Confirm-Step "Set execution policy to RemoteSigned for current user?") {
            try {
                Set-ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
                Write-Host "  Execution policy updated to RemoteSigned" -ForegroundColor Green
            }
            catch {
                Write-Host "error: Could not set execution policy automatically." -ForegroundColor Red
                Write-Host ""
                Write-Host "  Please run this command manually, then re-run the installer:" -ForegroundColor Yellow
                Write-Host ""
                Write-Host "    Set-ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Cyan
                Write-Host ""
                exit 1
            }
        }
        else {
            Write-Host ""
            Write-Host "  To install Anton, you need to run this command first:" -ForegroundColor Yellow
            Write-Host ""
            Write-Host "    Set-ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Cyan
            Write-Host ""
            exit 1
        }
    }
}

# ── 3. Check prerequisites ──────────────────────────────────────────
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "error: git is required but not found." -ForegroundColor Red
    Write-Host "  Install it with:  winget install Git.Git"
    Write-Host "  Or download from: https://git-scm.com/downloads/win"
    exit 1
}

# ── 4. Find or install uv ──────────────────────────────────────────
$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
$needUv = $false

if (-not $uvCmd) {
    $localUv = Join-Path $HOME ".local\bin\uv.exe"
    $cargoUv = Join-Path $HOME ".cargo\bin\uv.exe"

    if (Test-Path $localUv) {
        $uvBinDir = Join-Path $HOME ".local\bin"
        $env:PATH = "$uvBinDir;$env:PATH"
        Write-Host "  Found uv: $localUv"
    }
    elseif (Test-Path $cargoUv) {
        $cargoBinDir = Join-Path $HOME ".cargo\bin"
        $env:PATH = "$cargoBinDir;$env:PATH"
        Write-Host "  Found uv: $cargoUv"
    }
    else {
        $needUv = $true
    }
}
else {
    Write-Host "  Found uv: $($uvCmd.Source)"
}

if ($needUv) {
    Write-Host "warning: uv is not installed. It is required to manage anton's isolated environment." -ForegroundColor Yellow
    Write-Host "  uv will be installed to ~\.local\bin via https://astral.sh/uv/install.ps1"
    if (Confirm-Step "Install uv?") {
        Write-Host "  Installing uv..."
        & ([scriptblock]::Create((Invoke-RestMethod https://astral.sh/uv/install.ps1)))
        # Refresh PATH to pick up uv
        $uvBinDir = Join-Path $HOME ".local\bin"
        $env:PATH = "$uvBinDir;$env:PATH"
        Write-Host "  Installed uv"
    }
    else {
        Write-Host "error: uv is required. Install it manually: https://docs.astral.sh/uv/getting-started/installation/" -ForegroundColor Red
        exit 1
    }
}

# Verify uv is available
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "error: uv installation failed. Please install uv manually:" -ForegroundColor Red
    Write-Host "  https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
}

# ── 5. Resolve latest release tag ─────────────────────────────────
Write-Host "  Fetching latest release..."
try {
    $release = Invoke-RestMethod "https://api.github.com/repos/mindsdb/anton/releases/latest"
    $latestTag = $release.tag_name
    $repoUrl = "git+https://github.com/mindsdb/anton.git@$latestTag"
}
catch {
    Write-Host "warning: Could not determine latest release tag. Falling back to main branch." -ForegroundColor Yellow
    $repoUrl = "git+https://github.com/mindsdb/anton.git"
    $latestTag = ""
}

# ── 6. Install anton via uv tool ───────────────────────────────────
Write-Host ""
Write-Host "  This will install:"
Write-Host "    - anton (from $repoUrl)"
Write-Host "    - Into an isolated virtual environment managed by uv"
Write-Host "    - Python 3.11+ will be downloaded automatically if not present"
Write-Host ""

if (Confirm-Step "Proceed with installation?") {
    Write-Host "  Installing anton into an isolated venv..."
    uv tool install $repoUrl --force
    Write-Host "  Installed anton"
}
else {
    Write-Host "  Installation cancelled."
    exit 0
}

# ── 7. Verify the venv was created ─────────────────────────────────
$uvToolDir = Join-Path $HOME ".local\share\uv\tools\anton"
if (Test-Path $uvToolDir) {
    Write-Host "  Venv: $uvToolDir"
}
else {
    Write-Host "warning: Could not verify anton's virtual environment." -ForegroundColor Yellow
    Write-Host "  Expected at: $uvToolDir"
    Write-Host "  Anton may still work — uv manages the environment internally."
}

# ── 8. Configure firewall for scratchpad internet access ──────────
#    Anton's scratchpads run Python in per-scratchpad venvs under
#    ~/.anton/scratchpad-venvs/<name>/Scripts/python.exe.
#    Windows Firewall blocks new executables by default, so we add a
#    wildcard-style rule covering the entire scratchpad-venvs directory.
$scratchpadVenvsDir = Join-Path $HOME ".anton\scratchpad-venvs"
Write-Host ""
Write-Host "  Anton's scratchpads need internet access (for web scraping, APIs, etc.)."
Write-Host "  This adds a Windows Firewall rule allowing Python executables under:"
Write-Host "    $scratchpadVenvsDir"
Write-Host ""

if (Confirm-Step "Allow scratchpad internet access? (requires admin)") {
    # Create the venvs dir so the path exists
    if (-not (Test-Path $scratchpadVenvsDir)) {
        New-Item -ItemType Directory -Path $scratchpadVenvsDir -Force | Out-Null
    }

    # Build a single script with ALL netsh commands so we only trigger
    # one UAC elevation prompt instead of one per rule.
    $fwCommands = @()

    # Remove old single-venv rule if it exists (from previous installs)
    $fwCommands += "netsh advfirewall firewall delete rule name=`"Anton Scratchpad`" 2>`$null"

    # Add rules for any existing scratchpad venvs
    $added = 0
    if (Test-Path $scratchpadVenvsDir) {
        Get-ChildItem -Path $scratchpadVenvsDir -Directory | ForEach-Object {
            $pyExe = Join-Path $_.FullName "Scripts\python.exe"
            if (Test-Path $pyExe) {
                $fwCommands += "netsh advfirewall firewall add rule name=`"Anton Scratchpad - $($_.Name)`" dir=out action=allow program=`"$pyExe`""
                $added++
            }
        }
    }

    # Also add a rule for the uv tool python (used to create venvs)
    $uvToolPython = Join-Path $HOME ".local\share\uv\tools\anton\Scripts\python.exe"
    if (Test-Path $uvToolPython) {
        $fwCommands += "netsh advfirewall firewall add rule name=`"Anton Tool Python`" dir=out action=allow program=`"$uvToolPython`""
    }

    try {
        # Run all firewall commands in a single elevated PowerShell — one UAC prompt
        $script = $fwCommands -join "; "
        Start-Process -FilePath "powershell" `
            -ArgumentList "-NoProfile -ExecutionPolicy Bypass -Command `"$script`"" `
            -Verb RunAs -Wait -WindowStyle Hidden

        Write-Host "  Firewall rules added ($added scratchpad venvs)" -ForegroundColor Green
        Write-Host "  Note: New scratchpads will create venvs automatically." -ForegroundColor Yellow
        Write-Host "  If a new scratchpad's internet calls time out, re-run this script or add a rule manually:"
        Write-Host "    netsh advfirewall firewall add rule name=`"Anton Scratchpad`" dir=out action=allow program=`"$scratchpadVenvsDir\<name>\Scripts\python.exe`""
    }
    catch {
        Write-Host "  Could not add firewall rules (admin declined or unavailable)." -ForegroundColor Yellow
        Write-Host "  You can add them manually later for each scratchpad:"
        Write-Host "    netsh advfirewall firewall add rule name=`"Anton Scratchpad`" dir=out action=allow program=`"$scratchpadVenvsDir\<name>\Scripts\python.exe`""
    }
}
else {
    Write-Host "  Skipped. If scratchpad internet calls time out, run this in an admin PowerShell:"
    Write-Host "    netsh advfirewall firewall add rule name=`"Anton Scratchpad`" dir=out action=allow program=`"$scratchpadVenvsDir\<name>\Scripts\python.exe`""
}

# ── 9. Ensure ~/.local/bin is in user PATH ──────────────────────────
$uvBinDir = Join-Path $HOME ".local\bin"
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")

if ($userPath -and ($userPath -split ";" | ForEach-Object { $_.TrimEnd('\') }) -contains $uvBinDir.TrimEnd('\')) {
    # Already in PATH
}
else {
    if ($userPath) {
        $newPath = "$uvBinDir;$userPath"
    }
    else {
        $newPath = $uvBinDir
    }
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    $env:PATH = "$uvBinDir;$env:PATH"
    Write-Host "  Added $uvBinDir to user PATH"
}

# ── 10. Success message ──────────────────────────────────────────────
Write-Host ""
Write-Host "  ✓ anton installed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "  Open a new terminal, then:"
Write-Host ""
Write-Host "    anton                                          " -NoNewline
Write-Host "# Dashboard" -ForegroundColor Cyan
Write-Host "    anton run `"analyze last month's sales data`"    " -NoNewline
Write-Host "# Give Anton a task" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Upgrade:    uv tool upgrade anton"
Write-Host "  Uninstall:  uv tool uninstall anton"
Write-Host ""
Write-Host "  Config: ~\.anton\.env"
Write-Host ""
