# Version is derived from the installed package metadata (hatch-vcs sets it from
# the git tag at build/install time — see pyproject `[tool.hatch.version]`).
# Never hardcode it here: a manually-maintained constant drifted from the release
# tags (stuck at 2.26.6.30.1 across several releases), which made the CLI
# self-updater compare a stale local version against the real release tag and
# "update" on every launch forever (ENG-655).
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("anton-agent")
except PackageNotFoundError:  # source checkout without installed metadata
    __version__ = "0.0.0.dev0"
