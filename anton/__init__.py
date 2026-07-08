# Version is derived from the installed package metadata (hatch-vcs sets it from
# the git tag at build/install time — see pyproject `[tool.hatch.version]`).
# Never hardcode it here: a manually-maintained constant drifted from the release
# tags (stuck at 2.26.6.30.1 across several releases), which made the CLI
# self-updater compare a stale local version against the real release tag and
# "update" on every launch forever (ENG-655).
#
# Try the current distribution name first, then the legacy "anton" name (some
# installs predate the anton -> anton-agent rename); fall back to a dev version
# only when no metadata exists (source checkout). The whole block is defensively
# wrapped: this runs on EVERY `import anton` — including the desktop app's
# cowork-server — so it must never raise, or it would brick all anton imports.
try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    __version__ = "0.0.0.dev0"
    for _dist in ("anton-agent", "anton"):
        try:
            __version__ = _pkg_version(_dist)
            break
        except PackageNotFoundError:
            continue
except Exception:  # pragma: no cover - metadata machinery should never fail import
    __version__ = "0.0.0.dev0"
