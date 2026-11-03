"""Conservative request classification and static Earth-geography evidence.

This checks recognizable implementation patterns, not execution, data provenance,
or coastline accuracy. Browser checks still establish whether a page loads.
Unrecognized geographic renderers get an advisory rather than a false refusal.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Literal


@dataclass(frozen=True)
class GeographyRequirement:
    kind: Literal["map", "globe", "map_and_globe"]
    required: bool = True


@dataclass
class GeographyCheck:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


_NON_GEOGRAPHIC = re.compile(
    r"\b(?:road|mind|heat|site|concept|network|process|story|seating)\s*[- ]?maps?\b"
    r"|\b(?:floor|seating)\s+plans?\b|\b\w+\.map\s*\("
    r"|\bsnow\s+globes?\b"
    r"|\b(?:fictional|fantasy|schematic)\b(?:\s+\w+){0,4}\s+(?:maps?|globes?)\b"
    r"|\bmaps?\s+(?:of\s+)?(?:a\s+|our\s+|the\s+)?(?:fictional|fantasy|business\s+process)\b"
    r"|\b(?:maps?|globes?)\b[^.;\n]{0,80}\b(?:explicitly\s+requested\s+)?schematic\b"
    r"|\b(?:do\s+not|don't|without|no|not|instead\s+of)\s+"
    r"(?:(?:show|include|add|draw|render|use|create|build|make|a|an|any|the|flat|world|earth|geographic|interactive)\s+)*"
    r"(?:maps?|globes?)\b",
    re.I,
)
_EARTH_CONTEXT = re.compile(
    r"\b(?:earth|world|geograph\w*|coastlines?|continents?|countries|country|cities|city|"
    r"latitude|longitude|lat\s*[/,]\s*lon|africa|antarctica|asia|europe|oceania|"
    r"north\s+america|south\s+america|united\s+states|united\s+kingdom|"
    r"canada|mexico|brazil|france|germany|italy|spain|india|china|japan|australia)\b",
    re.I,
)


def classify_request(request: str) -> GeographyRequirement | None:
    """Require clear Earth requests; leave an unspecified map as advisory.

    This intentionally does not infer a map merely from country-valued data.
    It is not a general natural-language or place-name parser.
    """
    text = _NON_GEOGRAPHIC.sub(" ", request)
    has_map = bool(re.search(r"\bmaps?\b", text, re.I))
    has_globe = bool(re.search(r"\bglobes?\b", text, re.I))
    if not (has_map or has_globe):
        return None
    kind = "map_and_globe" if has_map and has_globe else "map" if has_map else "globe"
    return GeographyRequirement(kind, required=has_globe or bool(_EARTH_CONTEXT.search(text)))


class _Scripts(HTMLParser):
    """Collect executable inline scripts; imports and document prose are not code."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.parts: list[str] = []
        self.json_data: dict[str, str] = {}
        self._json_id: str | None = None
        self._script = False
        self._inert = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"template", "noscript"}:
            self._inert += 1
        if tag == "script":
            values = dict(attrs)
            script_type = (values.get("type") or "").strip().lower()
            if not self._inert and script_type == "application/json" and values.get("id"):
                self._json_id = values["id"]
                self.json_data[self._json_id] = ""
            self._script = not self._inert and "src" not in values and script_type in {
                "", "module", "text/javascript", "application/javascript",
            }

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._json_id = None
            self._script = False
            self.parts.append("\n")
        if tag in {"template", "noscript"}:
            self._inert = max(0, self._inert - 1)

    def handle_data(self, data: str) -> None:
        if self._script:
            self.parts.append(data)
        elif self._json_id is not None:
            self.json_data[self._json_id] += data


_JS_TEXT = re.compile(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"|`(?:\\.|[^`\\])*`|//[^\n]*|/\*.*?\*/", re.S)
_SIMPLE_LITERAL = re.compile(r"[\w .:/@+?=&%#_-]*\Z")


def _code(source: str) -> str:
    """Remove comments and opaque string contents without eating URLs.

    Keep simple option values and asset URLs. A string containing a whole code
    recipe cannot provide call-site evidence; template interpolation is left to
    the browser rather than parsed here.
    """
    def replace(match: re.Match[str]) -> str:
        value = match.group()
        if value.startswith(("//", "/*")):
            return " "
        literal = value[1:-1]
        if literal.lower().startswith("data:image/"):
            return '"embedded-image"'
        return '"' + literal + '"' if _SIMPLE_LITERAL.fullmatch(literal) else '""'

    code = _JS_TEXT.sub(replace, source)
    return re.sub(r'"(\w+)"\s*:', r'\1:', code)


def _has(pattern: str, code: str) -> bool:
    return bool(re.search(pattern, code))


def _component_kinds(code: str) -> set[str]:
    if not _has(r"\bCoworkPage\s*\(", code):
        return set()
    kinds = set(re.findall(r'\btype\s*:\s*"(map|globe)"', code))
    # presentation.js drawChart automatically registers these world aliases.
    if _has(r'\btype\s*:\s*"chart"', code) and _has(
        r'\bmap\s*:\s*"(?:world|world-map|cowork-world)"', code
    ) and (_has(r'\bgeo\s*:', code) or "map" in kinds):
        kinds.add("map")
    return kinds


def _contains_geometry(value: object) -> bool:
    if isinstance(value, dict):
        if value.get("type") in ("Polygon", "MultiPolygon") and value.get("coordinates"):
            return True
        if value.get("type") == "Topology" and value.get("objects") and value.get("arcs"):
            return True
        return any(_contains_geometry(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_geometry(item) for item in value)
    return False


def _consumed_json_geometry(scripts: _Scripts, code: str) -> bool:
    # A JSON script is data, never JavaScript evidence. Require a real consumer
    # as well as parsed geometry, including observed_data.bind's standard path.
    for script_id, source in scripts.json_data.items():
        if not _has(
            r'\bJSON\.parse\s*\(\s*document\.getElementById\s*\(\s*"'
            + re.escape(script_id) + r'"\s*\)\.textContent', code
        ):
            continue
        try:
            if _contains_geometry(json.loads(source)):
                return True
        except (ValueError, RecursionError):
            continue
    return False


def _ordinary_kinds(code: str, *, supplied_data: bool = False) -> tuple[set[str], set[str]]:
    kinds: set[str] = set()
    uncertain: set[str] = set()
    # Data may be loaded by the page or supplied as actual GeoJSON. Recognizing
    # its shape here does not establish its authenticity or geometric accuracy.
    data = supplied_data or _has(r"\bd3\.json\s*\(", code) or (
        _has(r"\bfetch\s*\(", code) and _has(r"\.json\s*\(", code)
    ) or (
        _has(r'\btype\s*:\s*"(?:Polygon|MultiPolygon|Topology)"', code)
        and _has(r"\b(?:coordinates|objects)\s*:\s*(?!\[\s*\]|\{\s*\})(?:\[|\{|[A-Za-z_$])", code)
    )
    parsed_json = _has(
        r'\bJSON\.parse\s*\(\s*(?!document\s*\.\s*getElementById\b)(?:[A-Za-z_$]|")', code
    )
    named_d3 = _has(r'\bimport\s*\{[^}]+\}\s*from\s*"[^"\n]*\bd3(?:@|/|")', code)
    d3_path = _has(r"\b(?:d3\.)?geoPath\s*\(", code)
    projection = _has(
        r"\b(?:d3\.)?geo(?:NaturalEarth1|Mercator|Equirectangular|Albers|AlbersUsa|EqualEarth|"
        r"AzimuthalEqualArea|AzimuthalEquidistant|ConicConformal|ConicEqualArea|"
        r"ConicEquidistant|Gnomonic|Stereographic|TransverseMercator)\s*\(", code
    )
    orthographic = _has(r"\b(?:d3\.)?geoOrthographic\s*\(", code)
    draw_path = _has(r'\.(?:attr|setAttribute)\s*\(\s*"d"\s*,\s*\(?\s*[A-Za-z_$]', code) or (
        _has(r"\.context\s*\(", code) and _has(r"\.(?:fill|stroke)\s*\(", code)
    )
    if (data or parsed_json or named_d3) and d3_path and draw_path:
        evidence = kinds if data and _has(r"\bd3\.geoPath\s*\(", code) else uncertain
        if projection or orthographic:
            evidence.add("map")
        else:
            # A custom projection still has geographic data and a drawing
            # chain. Leave its correctness to the browser/visual check.
            uncertain.add("map")
        if orthographic:
            evidence.add("globe")
    if _has(r"\becharts\.registerMap\s*\(", code) and _has(r"\.setOption\s*\(", code):
        if _has(r'\btype\s*:\s*"map"', code) or _has(r"\bgeo\s*:", code):
            registered = set(re.findall(r'\becharts\.registerMap\s*\(\s*"([^"\n]+)"', code))
            rendered = set(re.findall(r'\bmap\s*:\s*"([^"\n]+)"', code))
            if registered & rendered and data:
                kinds.add("map")
            elif (registered & rendered and parsed_json) or _has(
                r"\becharts\.registerMap\s*\(\s*[A-Za-z_$]|\bmap\s*:\s*[A-Za-z_$]", code
            ):
                uncertain.add("map")
    globe = _has(r"\b(?:new\s+)?Globe\s*\(", code)
    land = _has(r"\.polygonsData\s*\(\s*(?!\[\s*\]|null\b|undefined\b)[\w$]", code)
    image = _has(
        r'(?i)\.globeImageUrl\s*\(\s*"[^"\n]*(?:earth|world|blue[-_]?marble|daymap)[^"\n]*"\s*\)', code
    )
    if globe and (data and land or image):
        kinds.add("globe")
    elif globe and land and parsed_json:
        uncertain.add("globe")
    return kinds, uncertain


def check_geography(html: str, requirement: GeographyRequirement | None) -> GeographyCheck:
    """Report missing requested geography, without certifying runtime behavior."""
    result = GeographyCheck()
    if requirement is None:
        return result

    # The component runtime bundles every renderer even on a text-only page.
    # Only its generated program establishes which sections the page requests.
    from anton.core.tools.generate_artifact.component_page import existing_program

    program = existing_program(html)
    uncertain: set[str] = set()
    if program is not None:
        code = _code(program)
        kinds = _component_kinds(code)
        if _has(r"\bCoworkPage\s*\(", code):
            if _has(r"\btype\s*:\s*[A-Za-z_$]", code):
                uncertain.update({"map", "globe"})
            elif _has(r"\bgeo\s*:", code) and _has(r"\bmap\s*:\s*[A-Za-z_$]", code):
                uncertain.add("map")
    else:
        scripts = _Scripts()
        scripts.feed(html)
        code = _code("\n".join(scripts.parts))
        kinds, uncertain = _ordinary_kinds(code, supplied_data=_consumed_json_geometry(scripts, code))
    expected = {"map", "globe"} if requirement.kind == "map_and_globe" else {requirement.kind}
    missing = expected - kinds
    if not missing:
        return result

    # Known alternative geospatial APIs are evidence of custom work, not proof
    # of real coastlines. Leave their correctness to browser/visual inspection.
    custom = bool(missing & uncertain) or _has(
        r"\b(?:new\s+)?(?:Cesium\.Viewer|deck\.GeoJsonLayer|deck\.GlobeView|"
        r"maplibregl\.Map|mapboxgl\.Map|L\.map|L\.geoJSON|ol\.Map)\s*\(", code
    ) or (
        _has(r"\b(?:new\s+)?Globe\s*\(", code)
        and _has(r'\.globeImageUrl\s*\(\s*(?:"[^"\n]+"|[A-Za-z_$][\w$]*)\s*\)', code)
        and not _has(r"\.globeImageUrl\s*\(\s*(?:null|undefined)\s*\)", code)
    )
    label = " and ".join(sorted(missing))
    if custom:
        result.warnings.append(
            f"Geography check: custom {label} implementation needs browser/visual verification; "
            "static evidence cannot establish authentic Earth geometry. Preserve the requested geography."
        )
    else:
        message = (
            f"Geography check: no recognized {label} implementation with geographic data was found. "
            "Load the shared world atlas in the browser and draw its country geometry, or use authentic "
            "supplied GeoJSON with a geographic renderer. Preserve the requested map/globe and interactions "
            "while repairing it; do not replace it with invented outlines, decorative shapes, or a blank sphere."
        )
        if requirement.required:
            result.errors.append(message)
        else:
            result.warnings.append(message)
    return result
