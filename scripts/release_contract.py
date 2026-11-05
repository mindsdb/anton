"""Export the reviewed runtime lock without resolving or upgrading packages."""

import argparse
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "anton" / "release-requirements.txt"


def check_distributions(directory: Path, expected: str) -> None:
    wheels = list(directory.glob("*.whl"))
    sdists = list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Expected exactly one wheel and one sdist")
    member = CONTRACT.relative_to(ROOT).as_posix()
    with zipfile.ZipFile(wheels[0]) as wheel:
        if wheel.namelist().count(member) != 1 or wheel.read(member).decode() != expected:
            raise ValueError("Wheel runtime contract differs from the reviewed export")
    with tarfile.open(sdists[0]) as sdist:
        members = [item for item in sdist.getmembers() if item.name.endswith("/" + member)]
        if len(members) != 1 or not members[0].isfile():
            raise ValueError("Sdist must contain one regular runtime contract file")
        stream = sdist.extractfile(members[0])
        if stream is None or stream.read().decode() != expected:
            raise ValueError("Sdist runtime contract differs from the reviewed export")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="reject a stale export")
    parser.add_argument("--dist-dir", type=Path, help="also verify the built wheel and sdist")
    args = parser.parse_args()
    result = subprocess.run(
        ["uv", "export", "--locked", "--no-dev", "--no-emit-project", "--no-header", "--no-annotate"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if result.returncode:
        print(result.stderr, end="", file=sys.stderr)
        return result.returncode
    if args.check:
        if not CONTRACT.is_file() or CONTRACT.read_text() != result.stdout:
            print("Runtime export is stale; run python scripts/release_contract.py and review the diff.", file=sys.stderr)
            return 1
    else:
        CONTRACT.write_text(result.stdout)
    if args.dist_dir:
        try:
            check_distributions(args.dist_dir, result.stdout)
        except (ValueError, KeyError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
