"""Hand-built minimal OOXML packages for the `.pptx`/`.docx` lint tests.

Built part by part (no python-pptx/python-docx, neither is a dependency)
so each test can knock out exactly one part or relationship and assert the
lint names it. `minimal_pptx_parts()` is the smallest deck PowerPoint's
schema requires: presentation, one slide master, one layout, one theme,
one slide.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

_CT = "http://schemas.openxmlformats.org/package/2006/content-types"
_PR = "http://schemas.openxmlformats.org/package/2006/relationships"
_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PML_CT = "application/vnd.openxmlformats-officedocument.presentationml"

_DECL = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'


def _rels(*rels: tuple[str, str, str]) -> str:
    body = "".join(
        f'<Relationship Id="{rid}" Type="{_REL}/{kind}" Target="{target}"/>'
        for rid, kind, target in rels
    )
    return f'{_DECL}<Relationships xmlns="{_PR}">{body}</Relationships>'


_SP_TREE = (
    "<p:cSld><p:spTree>"
    '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
    "<p:grpSpPr/>"
    "</p:spTree></p:cSld>"
)

_THEME = (
    f'{_DECL}<a:theme xmlns:a="{_A}" name="Minimal"><a:themeElements>'
    '<a:clrScheme name="Minimal">'
    '<a:dk1><a:srgbClr val="000000"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>'
    '<a:dk2><a:srgbClr val="1F497D"/></a:dk2><a:lt2><a:srgbClr val="EEECE1"/></a:lt2>'
    '<a:accent1><a:srgbClr val="4F81BD"/></a:accent1><a:accent2><a:srgbClr val="C0504D"/></a:accent2>'
    '<a:accent3><a:srgbClr val="9BBB59"/></a:accent3><a:accent4><a:srgbClr val="8064A2"/></a:accent4>'
    '<a:accent5><a:srgbClr val="4BACC6"/></a:accent5><a:accent6><a:srgbClr val="F79646"/></a:accent6>'
    '<a:hlink><a:srgbClr val="0000FF"/></a:hlink><a:folHlink><a:srgbClr val="800080"/></a:folHlink>'
    "</a:clrScheme>"
    '<a:fontScheme name="Minimal">'
    '<a:majorFont><a:latin typeface="Calibri"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont>'
    '<a:minorFont><a:latin typeface="Calibri"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont>'
    "</a:fontScheme>"
    '<a:fmtScheme name="Minimal">'
    "<a:fillStyleLst>"
    + '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>' * 3
    + "</a:fillStyleLst><a:lnStyleLst>"
    + '<a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>' * 3
    + "</a:lnStyleLst><a:effectStyleLst>"
    + "<a:effectStyle><a:effectLst/></a:effectStyle>" * 3
    + "</a:effectStyleLst><a:bgFillStyleLst>"
    + '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>' * 3
    + "</a:bgFillStyleLst></a:fmtScheme>"
    "</a:themeElements></a:theme>"
)


def minimal_pptx_parts() -> dict[str, str]:
    """name -> XML text for a one-slide deck. Mutate a copy, then `write_package`."""
    ns = f'xmlns:a="{_A}" xmlns:r="{_R}" xmlns:p="{_P}"'
    return {
        "[Content_Types].xml": (
            f'{_DECL}<Types xmlns="{_CT}">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            f'<Override PartName="/ppt/presentation.xml" ContentType="{_PML_CT}.presentation.main+xml"/>'
            f'<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="{_PML_CT}.slideMaster+xml"/>'
            f'<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="{_PML_CT}.slideLayout+xml"/>'
            f'<Override PartName="/ppt/slides/slide1.xml" ContentType="{_PML_CT}.slide+xml"/>'
            '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>'
            "</Types>"
        ),
        "_rels/.rels": _rels(("rId1", "officeDocument", "ppt/presentation.xml")),
        "ppt/presentation.xml": (
            f"{_DECL}<p:presentation {ns}>"
            '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>'
            '<p:sldIdLst><p:sldId id="256" r:id="rId2"/></p:sldIdLst>'
            '<p:sldSz cx="9144000" cy="6858000"/><p:notesSz cx="6858000" cy="9144000"/>'
            "</p:presentation>"
        ),
        "ppt/_rels/presentation.xml.rels": _rels(
            ("rId1", "slideMaster", "slideMasters/slideMaster1.xml"),
            ("rId2", "slide", "slides/slide1.xml"),
            ("rId3", "theme", "theme/theme1.xml"),
        ),
        "ppt/slideMasters/slideMaster1.xml": (
            f"{_DECL}<p:sldMaster {ns}>{_SP_TREE}"
            '<p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" '
            'accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" '
            'hlink="hlink" folHlink="folHlink"/>'
            '<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>'
            "</p:sldMaster>"
        ),
        "ppt/slideMasters/_rels/slideMaster1.xml.rels": _rels(
            ("rId1", "slideLayout", "../slideLayouts/slideLayout1.xml"),
            ("rId2", "theme", "../theme/theme1.xml"),
        ),
        "ppt/slideLayouts/slideLayout1.xml": (
            f'{_DECL}<p:sldLayout {ns} type="blank">{_SP_TREE}'
            "<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>"
        ),
        "ppt/slideLayouts/_rels/slideLayout1.xml.rels": _rels(
            ("rId1", "slideMaster", "../slideMasters/slideMaster1.xml"),
        ),
        "ppt/slides/slide1.xml": (
            f"{_DECL}<p:sld {ns}>{_SP_TREE}"
            "<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"
        ),
        "ppt/slides/_rels/slide1.xml.rels": _rels(
            ("rId1", "slideLayout", "../slideLayouts/slideLayout1.xml"),
        ),
        "ppt/theme/theme1.xml": _THEME,
    }


def minimal_docx_parts() -> dict[str, str]:
    return {
        "[Content_Types].xml": (
            f'{_DECL}<Types xmlns="{_CT}">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>"
        ),
        "_rels/.rels": _rels(("rId1", "officeDocument", "word/document.xml")),
        "word/document.xml": (
            f'{_DECL}<w:document xmlns:w="{_W}"><w:body>'
            "<w:p><w:r><w:t>Hello</w:t></w:r></w:p>"
            "</w:body></w:document>"
        ),
    }


def write_package(path: Path, parts: dict[str, str | bytes]) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in parts.items():
            zf.writestr(name, data)
    return path


def write_minimal_pptx(path: Path) -> Path:
    return write_package(path, minimal_pptx_parts())


def write_minimal_docx(path: Path) -> Path:
    return write_package(path, minimal_docx_parts())
