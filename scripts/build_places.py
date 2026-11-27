"""Build the offline place index used to put source place names on maps.

  python scripts/build_places.py DUMP_DIR
  python scripts/build_places.py --region-codes ISO_3166_2_JSON CLDR_EN_XML

DUMP_DIR holds GeoNames dumps (https://download.geonames.org/export/dump/, CC BY 4.0):
cities5000.txt, admin1CodesASCII.txt and countryInfo.txt. Writes
anton/core/tools/generate_artifact/data/places.tsv.gz.

--region-codes joins GeoNames admin1 entries in places.tsv.gz with official ISO 3166-2
(Debian iso-codes iso_3166-2.json) and Unicode CLDR English subdivision names (en.xml)
to produce anton/core/tools/generate_artifact/data/region_codes.tsv for all countries.
"""
from collections import defaultdict
import gzip
import json
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "anton/core/tools/generate_artifact/data"
OUT = DATA_DIR / "places.tsv.gz"
REGION_CODES_OUT = DATA_DIR / "region_codes.tsv"
# Latin letters only: other scripts would grow the file for names source data rarely uses
_LATIN = re.compile(r"^[\sA-Za-zÀ-ɏḀ-ỿ'’.\-()]+$")
_GENERIC_ADMIN_WORDS = re.compile(
    r"\b(?:province|state|region|territory|district|department|oblast|krai|republic|municipality|"
    r"county|canton|prefecture|governorate|emirate|parish|division|autonomous|special|administrative|"
    r"capital|federal|metropolitan|city|island|islands|voivodeship|commune|community|regione|"
    r"provincia|estado|departamento|distrito|regiao|land|freie|hansestadt|freistaat|"
    r"sheng|shi|zizhiqu|sar|lan|principado|comunidad|foral|of|the|de|da|do|del|la|las|el|y|et)\b",
    re.I,
)


def _norm(value: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(value)).casefold() if c.isalnum())


def _strip_generic(value: str) -> str:
    decomposed = "".join(c for c in unicodedata.normalize("NFKD", value) if not unicodedata.combining(c))
    cleaned = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", decomposed)
    return _norm(_GENERIC_ADMIN_WORDS.sub(" ", cleaned))


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


def _match_country(entries, geo_rows, *, allow_reg_eq: bool) -> dict[str, str]:
    by_reg = {reg: gcode for gcode, reg, _, _ in geo_rows}
    by_exact: dict[str, list[str]] = defaultdict(list)
    by_stripped: dict[str, list[str]] = defaultdict(list)
    for gcode, _, name, aname in geo_rows:
        for nm in (name, aname):
            by_exact[_norm(nm)].append(gcode)
            stripped = _strip_generic(nm)
            if stripped:
                by_stripped[stripped].append(gcode)
    out: dict[str, str] = {}
    for sub, iname, en_name in entries:
        variants: list[str] = []
        for raw in (en_name, iname):
            if raw:
                variants.append(raw)
                for part in re.split(r"\s*/\s*|;", raw):
                    if part.strip():
                        variants.append(part.strip())
        found = by_reg.get(sub) if allow_reg_eq else None
        if not found:
            for variant in variants:
                hits = set(by_exact.get(_norm(variant), ()))
                if len(hits) == 1:
                    found = next(iter(hits))
                    break
        if not found:
            for variant in variants:
                stripped = _strip_generic(variant)
                if len(stripped) >= 3:
                    hits = set(by_stripped.get(stripped, ()))
                    if len(hits) == 1:
                        found = next(iter(hits))
                        break
        if found and found not in out:
            out[found] = sub
    return out


def build_region_codes(iso_json_path: Path, cldr_xml_path: Path) -> None:
    iso = json.loads(iso_json_path.read_text(encoding="utf-8"))["3166-2"]
    cldr_root = ET.parse(cldr_xml_path).getroot()
    cldr_en = {
        el.attrib["type"].upper(): el.text.strip()
        for el in cldr_root.findall(".//subdivision")
        if el.attrib.get("type") and el.text
    }
    admin1: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    with gzip.open(OUT, "rt", encoding="utf-8") as rows:
        for row in rows:
            kind, *f = row.rstrip("\n").split("\t")
            if kind == "A":
                cc, reg = f[0].split(".", 1)
                admin1[cc].append((f[0], reg, f[1], f[2]))

    iso_top: dict[str, list[tuple[str, str, str | None]]] = defaultdict(list)
    iso_child: dict[str, list[tuple[str, str, str | None]]] = defaultdict(list)
    for entry in iso:
        cc, sub = entry["code"].split("-", 1)
        if re.fullmatch(r"[A-Z]{2,3}", sub):
            rec = (sub, entry["name"], cldr_en.get(cc + sub))
            (iso_child if entry.get("parent") else iso_top)[cc].append(rec)

    matched: dict[str, str] = {}
    for cc, geo_rows in sorted(admin1.items()):
        country_map = _match_country(iso_top.get(cc, []), geo_rows, allow_reg_eq=True)
        if len(country_map) * 4 < len(geo_rows) and iso_child.get(cc):
            child_map = _match_country(iso_child[cc], geo_rows, allow_reg_eq=False)
            if len(child_map) > len(country_map):
                country_map = child_map
        matched.update(country_map)

    lines = [f"{gcode}\t{sub}" for gcode, sub in sorted(matched.items())]
    REGION_CODES_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    countries = {gcode.split(".", 1)[0] for gcode in matched}
    print(REGION_CODES_OUT, len(lines), "subdivisions across", len(countries), "countries")


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
    if len(sys.argv) == 4 and sys.argv[1] == "--region-codes":
        build_region_codes(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        main(sys.argv[1])
