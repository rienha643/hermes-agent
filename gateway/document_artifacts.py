"""Helpers for publishing document artifacts to the HermesWork Documents tree.

This module centralizes two behaviors that used to be spread across the
delegate, gateway delivery, and response-dispatch code paths:

1. Canonicalize all document outputs under ``HermesWork/Documents`` before
   they are surfaced to the user.
2. Queue the NAS documents sync hook for the published artifact.

It also performs a best-effort docx modernization pass for minimal OOXML
packages so Word opens them with the expected settings/theme/font table/
numbering parts present.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

from hermes_constants import get_hermes_work_dir
from gateway.project_registry import next_versioned_child_path, resolve_project_artifact_dir
import nas_sync_hooks

logger = logging.getLogger(__name__)


def queue_nas_sync_hook(*args, **kwargs):
    return nas_sync_hooks.queue_nas_sync_hook(*args, **kwargs)

_DOCUMENT_ARTIFACT_EXTENSIONS = {
    ".docx",
    ".doc",
    ".odt",
    ".rtf",
    ".pdf",
    ".md",
    ".txt",
    ".pptx",
    ".ppt",
    ".xlsx",
    ".xls",
}

_DOCX_CONTENT_TYPES = {
    "settings": "application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml",
    "fontTable": "application/vnd.openxmlformats-officedocument.wordprocessingml.fontTable+xml",
    "numbering": "application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml",
    "theme": "application/vnd.openxmlformats-officedocument.theme+xml",
}

_DOCX_REL_TYPES = {
    "settings": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings",
    "fontTable": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/fontTable",
    "numbering": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering",
    "theme": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme",
}

_DOCX_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_REL_NAMESPACE = "http://schemas.openxmlformats.org/package/2006/relationships"
_CT_NAMESPACE = "http://schemas.openxmlformats.org/package/2006/content-types"
_DRAWINGML_NAMESPACE = "http://schemas.openxmlformats.org/drawingml/2006/main"

_STORY_SCOPE_ALLOWLIST = {
    "lore",
    "setting",
    "story",
    "worldbuilding",
}
_INTERNAL_SCOPE_NAMES = {
    "ai_agent",
    "artist",
    "balance",
    "coder",
    "comfy",
    "cron-fast",
    "designer",
    "forge",
    "hermes-agent",
    "pm",
    "qa",
    "root",
    "scenario",
    "speedy",
}
_STORY_SCOPE_EMPTY_NAMES = {
    "",
    ".",
    "ai_agent",
    "archive",
    "archives",
    "document",
    "documents",
    "game",
    "games",
    "image",
    "images",
    "misc",
    "root",
    "story",
    "stories",
    *_INTERNAL_SCOPE_NAMES,
}
_STORY_HINT_TOKENS = {
    "chronicle",
    "lore",
    "lyra",
    "tyr",
    "worldbuilding",
    "세계관",
    "설정집",
    "시나리오",
    "연대기",
}
_SCOPE_TOKEN_SPLIT_RE = re.compile(r"[^0-9A-Za-z가-힣]+")

ET.register_namespace("w", _DOCX_NAMESPACE)
ET.register_namespace("r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships")
ET.register_namespace("a", _DRAWINGML_NAMESPACE)


def _is_under_root(path: Path, root: Path) -> bool:
    try:
        return path == root or root in path.parents
    except Exception:
        return False


def _absolute_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else candidate.absolute()


def _is_under_root_consistent(path: str | Path, root: str | Path) -> bool:
    logical_path = _absolute_path(path)
    logical_root = _absolute_path(root)
    if _is_under_root(logical_path, logical_root):
        return True
    try:
        return _is_under_root(logical_path.resolve(strict=False), logical_root.resolve(strict=False))
    except Exception:
        return False


def _relative_to_root_consistent(path: str | Path, root: str | Path) -> Path:
    logical_path = _absolute_path(path)
    logical_root = _absolute_path(root)
    try:
        return logical_path.relative_to(logical_root)
    except ValueError:
        return logical_path.resolve(strict=False).relative_to(logical_root.resolve(strict=False))


def _paths_equivalent(left: str | Path, right: str | Path) -> bool:
    left_path = _absolute_path(left)
    right_path = _absolute_path(right)
    if left_path == right_path:
        return True
    try:
        return left_path.resolve(strict=False) == right_path.resolve(strict=False)
    except Exception:
        return False


def _scope_tokens(*values: str) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        if not value:
            continue
        lowered = value.casefold()
        tokens.add(lowered)
        for piece in _SCOPE_TOKEN_SPLIT_RE.split(
            lowered.replace("-", " ").replace("_", " ").replace("/", " ").replace("\\", " ")
        ):
            piece = piece.strip("._ ")
            if piece:
                tokens.add(piece)
    return tokens


def _candidate_scope_name(folder_name: Optional[str], *, source: Path) -> str:
    return (folder_name or source.parent.name or "").strip()


def _is_internal_scope_name(name: str) -> bool:
    return name.casefold() in _INTERNAL_SCOPE_NAMES


def _looks_like_story_artifact(source: Path, story_root: Path, *, folder_name: Optional[str] = None) -> bool:
    if _is_under_root(source, story_root):
        return True
    tokens = _scope_tokens(
        folder_name or "",
        source.name,
        source.stem,
        source.parent.name,
        *source.parts[-4:],
    )
    return bool(tokens & _STORY_HINT_TOKENS)


def _normalized_scope_name(folder_name: Optional[str], *, source: Path) -> str:
    candidate = _candidate_scope_name(folder_name, source=source)
    if not candidate:
        return "misc"

    if _is_internal_scope_name(candidate):
        return "misc"
    if candidate.casefold() == "ai_agent":
        return "misc"
    return candidate


def _normalized_story_scope(source: Path, story_root: Path, *, folder_name: Optional[str] = None) -> str:
    if _is_under_root(source, story_root):
        try:
            relative_parent = source.parent.relative_to(story_root)
        except ValueError:
            relative_parent = Path()
        if not relative_parent.parts:
            return ""

        first = relative_parent.parts[0].strip()
        if first.casefold() in _STORY_SCOPE_EMPTY_NAMES:
            return ""
        return first

    candidate = _candidate_scope_name(folder_name, source=source)
    if not candidate:
        return ""
    if candidate.casefold() in _STORY_SCOPE_EMPTY_NAMES:
        return ""
    if candidate.casefold() in _STORY_SCOPE_ALLOWLIST:
        return candidate
    return ""


_MINIMAL_SETTINGS_XML = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="{_DOCX_NAMESPACE}">
  <w:zoom w:percent="100"/>
  <w:defaultTabStop w:val="720"/>
  <w:characterSpacingControl w:val="doNotCompress"/>
  <w:compat>
    <w:compatSetting w:name="compatibilityMode" w:uri="http://schemas.microsoft.com/office/word" w:val="15"/>
    <w:compatSetting w:name="overrideTableStyleFontSizeAndJustification" w:uri="http://schemas.microsoft.com/office/word" w:val="1"/>
    <w:compatSetting w:name="enableOpenTypeFeatures" w:uri="http://schemas.microsoft.com/office/word" w:val="1"/>
    <w:compatSetting w:name="doNotExpandShiftReturn" w:uri="http://schemas.microsoft.com/office/word" w:val="1"/>
  </w:compat>
  <w:themeFontLang w:val="en-US" w:eastAsia="en-US" w:bidi="ar-SA"/>
</w:settings>
"""

_MINIMAL_FONT_TABLE_XML = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:fonts xmlns:w="{_DOCX_NAMESPACE}">
  <w:font w:name="Aptos">
    <w:panose1 w:val="020B0604030504040204"/>
    <w:charset w:val="00"/>
    <w:family w:val="swiss"/>
    <w:pitch w:val="variable"/>
  </w:font>
  <w:font w:name="Aptos Display">
    <w:panose1 w:val="020B0604030504040204"/>
    <w:charset w:val="00"/>
    <w:family w:val="swiss"/>
    <w:pitch w:val="variable"/>
  </w:font>
</w:fonts>
"""

_MINIMAL_NUMBERING_XML = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:numbering xmlns:w="{_DOCX_NAMESPACE}">
  <w:abstractNum w:abstractNumId="0">
    <w:nsid w:val="00000000"/>
    <w:multiLevelType w:val="hybridMultilevel"/>
    <w:lvl w:ilvl="0">
      <w:start w:val="1"/>
      <w:numFmt w:val="decimal"/>
      <w:lvlText w:val="%1."/>
      <w:lvlJc w:val="left"/>
      <w:pPr>
        <w:ind w:left="720" w:hanging="360"/>
      </w:pPr>
    </w:lvl>
  </w:abstractNum>
  <w:num w:numId="1">
    <w:abstractNumId w:val="0"/>
  </w:num>
</w:numbering>
"""

_MINIMAL_THEME_XML = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="{_DRAWINGML_NAMESPACE}" name="Hermes Theme">
  <a:themeElements>
    <a:clrScheme name="Hermes">
      <a:dk1><a:sysClr val="windowText" lastClr="000000"/></a:dk1>
      <a:lt1><a:sysClr val="window" lastClr="FFFFFF"/></a:lt1>
      <a:dk2><a:srgbClr val="1F1F1F"/></a:dk2>
      <a:lt2><a:srgbClr val="F3F3F3"/></a:lt2>
      <a:accent1><a:srgbClr val="4472C4"/></a:accent1>
      <a:accent2><a:srgbClr val="ED7D31"/></a:accent2>
      <a:accent3><a:srgbClr val="A5A5A5"/></a:accent3>
      <a:accent4><a:srgbClr val="FFC000"/></a:accent4>
      <a:accent5><a:srgbClr val="5B9BD5"/></a:accent5>
      <a:accent6><a:srgbClr val="70AD47"/></a:accent6>
      <a:hlink><a:srgbClr val="0563C1"/></a:hlink>
      <a:folHlink><a:srgbClr val="954F72"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="Hermes">
      <a:majorFont>
        <a:latin typeface="Aptos Display"/>
        <a:ea typeface=""/>
        <a:cs typeface=""/>
      </a:majorFont>
      <a:minorFont>
        <a:latin typeface="Aptos"/>
        <a:ea typeface=""/>
        <a:cs typeface=""/>
      </a:minorFont>
    </a:fontScheme>
    <a:fmtScheme name="Hermes">
      <a:fillStyleLst>
        <a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
        <a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
        <a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
      </a:fillStyleLst>
      <a:lnStyleLst>
        <a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/></a:ln>
        <a:ln w="25400"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/></a:ln>
        <a:ln w="38100"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/></a:ln>
      </a:lnStyleLst>
      <a:effectStyleLst>
        <a:effectStyle><a:effectLst/></a:effectStyle>
        <a:effectStyle><a:effectLst/></a:effectStyle>
        <a:effectStyle><a:effectLst/></a:effectStyle>
      </a:effectStyleLst>
      <a:bgFillStyleLst>
        <a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
        <a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
        <a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
      </a:bgFillStyleLst>
    </a:fmtScheme>
  </a:themeElements>
  <a:objectDefaults/>
  <a:extraClrSchemeLst/>
</a:theme>
"""


_DOCUMENT_ARTIFACT_DELIVERY = "Slack 첨부"
_DOCUMENT_ARTIFACT_NAS_STATE_GENERATED = "hook state 생성"
_DOCUMENT_ARTIFACT_NAS_STATE_NONE = "hook state 없음"
_DOCUMENT_ARTIFACT_NAS_STATE_UNKNOWN = "확인 불가"


def is_document_artifact_path(path: str | Path) -> bool:
    """Return True when *path* looks like a document artifact we should publish."""
    suffix = Path(str(path)).suffix.lower()
    return suffix in _DOCUMENT_ARTIFACT_EXTENSIONS


def _display_document_artifact_path(path: str | Path) -> str:
    """Prefer a HermesWork-relative path when the artifact lives under it."""
    logical_path = _absolute_path(path)
    work_root = _absolute_path(get_hermes_work_dir())
    try:
        relative = _relative_to_root_consistent(logical_path, work_root)
    except ValueError:
        try:
            fallback = logical_path.resolve(strict=False)
            idx = next(i for i, part in enumerate(fallback.parts) if part.casefold() == "hermeswork")
            relative = Path(*fallback.parts[idx + 1 :])
        except StopIteration:
            return str(logical_path)
    if not relative.parts:
        return "HermesWork"
    return f"HermesWork/{relative.as_posix()}"


def _document_artifact_category_scope(path: Path) -> tuple[Optional[str], Optional[str]]:
    logical_path = _absolute_path(path)
    story_root = _absolute_path(get_hermes_work_dir("Story"))
    documents_root = _absolute_path(get_hermes_work_dir("Documents"))

    if _is_under_root_consistent(logical_path, story_root):
        try:
            relative_parent = _relative_to_root_consistent(logical_path.parent, story_root)
        except ValueError:
            relative_parent = Path()
        if not relative_parent.parts:
            return "story", ""
        first = relative_parent.parts[0].strip()
        if first.casefold() in _STORY_SCOPE_EMPTY_NAMES:
            return "story", ""
        return "story", first

    if _is_under_root_consistent(logical_path, documents_root):
        try:
            relative_parent = _relative_to_root_consistent(logical_path.parent, documents_root)
        except ValueError:
            relative_parent = Path()
        scope = "" if str(relative_parent) == "." else relative_parent.as_posix()
        return "documents", scope

    return None, None


def infer_document_artifact_nas_state(path: str | Path) -> str:
    """Best-effort NAS state label for a published HermesWork document artifact."""
    artifact_path = Path(path)
    category, scope = _document_artifact_category_scope(artifact_path)
    if not category:
        return _DOCUMENT_ARTIFACT_NAS_STATE_NONE

    try:
        key = nas_sync_hooks._artifact_hook_key(category, scope or "", artifact_path)
        state_path = nas_sync_hooks._resolve_nas_hook_state_dir() / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.json"
        return (
            _DOCUMENT_ARTIFACT_NAS_STATE_GENERATED
            if state_path.exists()
            else _DOCUMENT_ARTIFACT_NAS_STATE_UNKNOWN
        )
    except Exception:
        return _DOCUMENT_ARTIFACT_NAS_STATE_UNKNOWN


def describe_document_artifact(
    path: str | Path,
    *,
    nas_state: str = _DOCUMENT_ARTIFACT_NAS_STATE_GENERATED,
    delivery: str = _DOCUMENT_ARTIFACT_DELIVERY,
) -> dict[str, str]:
    """Return the standard document artifact UX fields for a published file."""
    artifact_path = Path(path)
    suffix = artifact_path.suffix.lstrip(".").upper() or "FILE"
    return {
        "format": suffix,
        "path": _display_document_artifact_path(artifact_path),
        "delivery": delivery,
        "nas_state": nas_state,
    }


def format_document_artifact_lines(
    path: str | Path,
    *,
    nas_state: str = _DOCUMENT_ARTIFACT_NAS_STATE_GENERATED,
    delivery: str = _DOCUMENT_ARTIFACT_DELIVERY,
) -> list[str]:
    """Return the canonical bullet lines for a document artifact report."""
    info = describe_document_artifact(path, nas_state=nas_state, delivery=delivery)
    return [
        f"형식: {info['format']}",
        f"저장 위치: `{info['path']}`",
        f"전달 방식: {info['delivery']}",
        f"NAS 상태: {info['nas_state']}",
    ]


def format_document_artifact_block(
    path: str | Path,
    *,
    nas_state: str = _DOCUMENT_ARTIFACT_NAS_STATE_GENERATED,
    delivery: str = _DOCUMENT_ARTIFACT_DELIVERY,
) -> str:
    """Return the canonical completion block for document artifacts."""
    return "- 산출물\n  - " + "\n  - ".join(
        format_document_artifact_lines(path, nas_state=nas_state, delivery=delivery)
    )


def _ensure_xml_override(root: ET.Element, part_name: str, content_type: str) -> bool:
    for override in root.findall(f"{{{_CT_NAMESPACE}}}Override"):
        if override.get("PartName") == part_name:
            return False
    ET.SubElement(root, f"{{{_CT_NAMESPACE}}}Override", {"PartName": part_name, "ContentType": content_type})
    return True


def _ensure_relationship(root: ET.Element, rel_type: str, target: str) -> bool:
    existing = [rel.get("Type") for rel in root.findall(f"{{{_REL_NAMESPACE}}}Relationship")]
    if rel_type in existing:
        return False
    next_id = 1
    for rel in root.findall(f"{{{_REL_NAMESPACE}}}Relationship"):
        rid = rel.get("Id", "")
        if rid.startswith("rId"):
            try:
                next_id = max(next_id, int(rid[3:]) + 1)
            except ValueError:
                continue
    ET.SubElement(
        root,
        f"{{{_REL_NAMESPACE}}}Relationship",
        {
            "Id": f"rId{next_id}",
            "Type": rel_type,
            "Target": target,
        },
    )
    return True


def _canonical_story_relative_path(path: Path, story_root: Path) -> Path:
    relative = path.relative_to(story_root)
    parts = list(relative.parts)
    if len(parts) <= 1:
        return relative

    idx = 0
    while idx < len(parts) - 1 and parts[idx].casefold() in _STORY_SCOPE_EMPTY_NAMES:
        idx += 1
    if idx == 0:
        return relative
    return Path(*parts[idx:])


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cleanup_story_duplicate_tree(story_root: Path) -> None:
    if not story_root.exists():
        return

    story_root = story_root.resolve(strict=False)
    moved = 0
    deduped = 0
    conflicts = 0

    for source_file in sorted((p for p in story_root.rglob("*") if p.is_file()), key=lambda p: len(p.parts), reverse=True):
        try:
            relative = source_file.relative_to(story_root)
        except ValueError:
            continue
        if not relative.parts:
            continue
        target_relative = _canonical_story_relative_path(source_file, story_root)
        target = story_root / target_relative
        if target == source_file:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        source_hash = _sha256_path(source_file)
        if target.exists():
            if target.is_file() and _sha256_path(target) == source_hash:
                source_file.unlink()
                deduped += 1
                continue
            conflict = target.with_name(f"{target.stem}__{source_hash[:8]}{target.suffix}")
            suffix = 1
            while conflict.exists():
                conflict = target.with_name(f"{target.stem}__{source_hash[:8]}_{suffix}{target.suffix}")
                suffix += 1
            shutil.move(str(source_file), conflict)
            conflicts += 1
            continue
        shutil.move(str(source_file), target)
        moved += 1

    for direct_child in sorted(
        (story_root / name for name in _STORY_SCOPE_EMPTY_NAMES if name),
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        if not direct_child.exists() or not direct_child.is_dir():
            continue
        for directory in sorted((p for p in direct_child.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            try:
                directory.rmdir()
            except OSError:
                pass
        try:
            direct_child.rmdir()
        except OSError:
            pass

    if moved or deduped or conflicts:
        logger.info(
            "Story cleanup repaired duplicate tree under %s: moved=%s deduped=%s conflicts=%s",
            story_root,
            moved,
            deduped,
            conflicts,
        )


def _modernize_docx_package(docx_path: Path) -> bool:
    """Add common Word parts to minimal OOXML docx packages in place.

    Returns True when the archive was rewritten.
    """
    try:
        with zipfile.ZipFile(docx_path, "r") as archive:
            names = set(archive.namelist())
            if "word/document.xml" not in names:
                return False
            payloads = {name: archive.read(name) for name in names}
    except Exception:
        logger.debug("DOCX modernization skipped: could not read %s", docx_path, exc_info=True)
        return False

    changed = False

    if "word/settings.xml" not in payloads:
        payloads["word/settings.xml"] = _MINIMAL_SETTINGS_XML.encode("utf-8")
        changed = True
    if "word/fontTable.xml" not in payloads:
        payloads["word/fontTable.xml"] = _MINIMAL_FONT_TABLE_XML.encode("utf-8")
        changed = True
    if "word/numbering.xml" not in payloads:
        payloads["word/numbering.xml"] = _MINIMAL_NUMBERING_XML.encode("utf-8")
        changed = True
    if "word/theme/theme1.xml" not in payloads:
        payloads["word/theme/theme1.xml"] = _MINIMAL_THEME_XML.encode("utf-8")
        changed = True

    ct_name = "[Content_Types].xml"
    try:
        ct_root = ET.fromstring(payloads[ct_name])
    except Exception:
        ct_root = ET.fromstring(
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="{_CT_NAMESPACE}"/>
"""
        )
        changed = True
    changed |= _ensure_xml_override(ct_root, "/word/settings.xml", _DOCX_CONTENT_TYPES["settings"])
    changed |= _ensure_xml_override(ct_root, "/word/fontTable.xml", _DOCX_CONTENT_TYPES["fontTable"])
    changed |= _ensure_xml_override(ct_root, "/word/numbering.xml", _DOCX_CONTENT_TYPES["numbering"])
    changed |= _ensure_xml_override(ct_root, "/word/theme/theme1.xml", _DOCX_CONTENT_TYPES["theme"])
    payloads[ct_name] = ET.tostring(ct_root, encoding="utf-8", xml_declaration=True)

    rels_name = "word/_rels/document.xml.rels"
    if rels_name in payloads:
        try:
            rel_root = ET.fromstring(payloads[rels_name])
        except Exception:
            rel_root = ET.Element(f"{{{_REL_NAMESPACE}}}Relationships")
            changed = True
    else:
        rel_root = ET.Element(f"{{{_REL_NAMESPACE}}}Relationships")
        changed = True
    changed |= _ensure_relationship(rel_root, _DOCX_REL_TYPES["settings"], "settings.xml")
    changed |= _ensure_relationship(rel_root, _DOCX_REL_TYPES["fontTable"], "fontTable.xml")
    changed |= _ensure_relationship(rel_root, _DOCX_REL_TYPES["numbering"], "numbering.xml")
    changed |= _ensure_relationship(rel_root, _DOCX_REL_TYPES["theme"], "theme/theme1.xml")
    payloads[rels_name] = ET.tostring(rel_root, encoding="utf-8", xml_declaration=True)

    if not changed:
        return False

    fd, tmp_name = tempfile.mkstemp(dir=str(docx_path.parent), suffix=".tmp", prefix=docx_path.name + ".")
    tmp_path = Path(tmp_name)
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as out_zip:
            for name, data in payloads.items():
                out_zip.writestr(name, data)
        tmp_path.replace(docx_path)
        return True
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def publish_document_artifact(source: Path, *, folder_name: Optional[str] = None) -> Path:
    """Publish *source* into the canonical HermesWork documents or story tree.

    General planning / QA / report documents continue to land under
    ``HermesWork/Documents``.  Worldbuilding / lore / setting deliverables are
    routed to ``HermesWork/Story``.  If an artifact already lives under the
    canonical tree, it is left in place and only the NAS hook is queued.

    DOCX files receive a best-effort modernization pass before the hook is
    queued so the published artifact carries standard settings/theme/font table
    and numbering parts.
    """
    if not isinstance(source, Path):
        source = Path(source)
    source = _absolute_path(source)
    if not source.exists() or not source.is_file() or not is_document_artifact_path(source):
        return source

    try:
        project_name = _normalized_scope_name(folder_name, source=source)

        story_root = get_hermes_work_dir("Story")
        resolved_story_root = story_root.resolve(strict=False)
        if _looks_like_story_artifact(source, resolved_story_root, folder_name=folder_name):
            scope = _normalized_story_scope(source, resolved_story_root, folder_name=folder_name)
            if _is_under_root_consistent(source, story_root):
                canonical_relative = _relative_to_root_consistent(source, story_root)
                canonical_relative = _canonical_story_relative_path(story_root / canonical_relative, story_root)
                published_path = story_root / canonical_relative
                source_root = story_root
                _cleanup_story_duplicate_tree(story_root)
            else:
                _, published_dir = resolve_project_artifact_dir("Story", project_name)
                published_dir.mkdir(parents=True, exist_ok=True)
                published_path = next_versioned_child_path(published_dir, source.name)
                if not _paths_equivalent(published_path, source):
                    try:
                        shutil.copy2(source, published_path)
                    except PermissionError as exc:
                        logger.warning(
                            "Story artifact metadata copy failed; falling back to content copy: %s",
                            exc,
                        )
                        shutil.copyfile(source, published_path)
                source_root = published_dir
                scope = published_path.parent.name

            if published_path.suffix.lower() == ".docx" and published_path.exists():
                _modernize_docx_package(published_path)

            queue_nas_sync_hook(
                category="story",
                scope=scope,
                artifact_path=published_path,
                source_root=source_root,
            )
            return published_path

        documents_root = get_hermes_work_dir("Documents")
        if _is_under_root_consistent(source, documents_root):
            published_path = source
            relative_parent = _relative_to_root_consistent(published_path.parent, documents_root)
            scope = "" if str(relative_parent) == "." else str(relative_parent).replace("/", os.sep)
            source_root = published_path.parent
        else:
            _, published_dir = resolve_project_artifact_dir("Documents", project_name)
            published_dir.mkdir(parents=True, exist_ok=True)
            published_path = next_versioned_child_path(published_dir, source.name)
            if not _paths_equivalent(published_path, source):
                try:
                    shutil.copy2(source, published_path)
                except PermissionError as exc:
                    logger.warning(
                        "Document artifact metadata copy failed; falling back to content copy: %s",
                        exc,
                    )
                    shutil.copyfile(source, published_path)
            scope = published_path.parent.name
            source_root = published_dir

        if published_path.suffix.lower() == ".docx" and published_path.exists():
            _modernize_docx_package(published_path)

        queue_nas_sync_hook(
            category="documents",
            scope=scope,
            artifact_path=published_path,
            source_root=source_root,
        )
        return published_path
    except Exception:
        logger.debug("Document publish/NAS hook failed for %s", source, exc_info=True)
        return source
