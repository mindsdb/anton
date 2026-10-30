"""Build the offline place index used to put source place names on maps.

  python scripts/build_places.py DUMP_DIR

DUMP_DIR holds GeoNames dumps (https://download.geonames.org/export/dump/, CC BY 4.0):
cities5000.txt, admin1CodesASCII.txt and countryInfo.txt. Writes
anton/core/tools/generate_artifact/data/places.tsv.gz.
"""
import gzip
import re
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "anton/core/tools/generate_artifact/data/places.tsv.gz"
# Latin letters only: other scripts would grow the file for names source data rarely uses
_LATIN = re.compile(r"^[\sA-Za-zÀ-ɏḀ-ỿ'’.\-()]+$")


def _alternates(name, raw):
    seen = {name.casefold()}
    out = []
    for alt in raw.split(","):
        alt = alt.strip()
        # codes such as airport ids are all capitals; they would match unrelated values
        if 2 <= len(alt) <= 60 and _LATIN.match(alt) and alt != alt.upper() and alt.casefold() not in seen:
            seen.add(alt.casefold())
            out.append(alt)
    return out


def main(dump):
    dump = Path(dump)
    lines = []
    for row in (dump / "countryInfo.txt").read_text(encoding="utf-8").splitlines():
        if row and not row.startswith("#"):
            f = row.split("\t")
            lines.append("\t".join(["C", f[0], f[1], f[4]]))
    for row in (dump / "admin1CodesASCII.txt").read_text(encoding="utf-8").splitlines():
        code, name, ascii_name, _ = row.split("\t")
        lines.append("\t".join(["A", code, name, ascii_name]))
    for row in (dump / "cities5000.txt").read_text(encoding="utf-8").splitlines():
        f = row.split("\t")
        lines.append("\t".join(["P", f[1], f[4], f[5], f[8], f[10], f[14],
                                "|".join(_alternates(f[1], f[2] + "," + f[3]))]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT, "wt", encoding="utf-8", compresslevel=9) as out:
        out.write("\n".join(lines) + "\n")
    print(OUT, OUT.stat().st_size)


if __name__ == "__main__":
    main(sys.argv[1])
