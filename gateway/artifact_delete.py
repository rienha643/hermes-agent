from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
import shutil
from pathlib import Path
from typing import Any

LOCAL_HERMESWORK_ROOT_ENV = "HERMESWORK_ROOT"
NAS_ROOT_ENV = "HERMESWORK_NAS_ROOT"
DEFAULT_LOCAL_HERMESWORK_ROOT = Path("/mnt/c/Users/AI_Agent/HermesWork")
DEFAULT_NAS_ROOT = r"\\hyungwoo\\Hermes"
CATEGORY_DIR_NAMES = {
    "documents": "Documents",
    "story": "Story",
    "image": "Image",
    "games": "Games",
}
PROTECTED_NAMES = {
    "SOUL.md",
    "MEMORY.md",
    "USER.md",
    "config.yaml",
    "auth.json",
    "nas_credentials.json",
    "project_registry.json",
    ".gitkeep",
    "ai-image-gallery",
}
APPROVAL_PAPERWORK = {
    "approval_purpose": "Hermes 아티팩트 삭제",
    "approval_work": "로컬 및 NAS 산출물 삭제 계획 확인",
    "requires_approval": True,
}


@dataclass(frozen=True, slots=True)
class DeleteRequest:
    path: Path
    category: str | None = None
    mirror: bool | None = None


@dataclass(frozen=True, slots=True)
class DeletePlan:
    category: str
    local_path: str
    nas_path: str
    will_delete_local: bool
    will_delete_nas: bool
    approval_purpose: str
    approval_work: str
    requires_approval: bool
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.approval_purpose.strip():
            raise ValueError("approval_purpose must not be blank")
        if not self.approval_work.strip():
            raise ValueError("approval_work must not be blank")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["warnings"] = list(self.warnings)
        payload["delete_mode"] = _delete_mode_for(self)
        payload["deletion_executed"] = False
        payload["local_delete_executed"] = False
        payload["local_delete_verified"] = False
        payload["nas_deletion_executed"] = False
        payload["nas_delete_pending"] = not _is_blocked_plan(self)
        payload["delete_status"] = _delete_status_for(self)
        payload["local_delete_planned"] = self.will_delete_local
        payload["nas_delete_planned"] = self.will_delete_nas
        payload["blocked_reasons"] = list(self.warnings) if _is_blocked_plan(self) else []
        payload["approval_metadata"] = _approval_metadata_for(self)
        payload["approved"] = False
        payload["user_message"] = _format_user_message(self)
        return payload


class ArtifactDeleteOrchestrator:
    def __init__(self, *, local_root: str | Path | None = None, nas_root: str | None = None) -> None:
        self.local_root = Path(
            local_root
            or os.environ.get(LOCAL_HERMESWORK_ROOT_ENV)
            or DEFAULT_LOCAL_HERMESWORK_ROOT
        ).expanduser()
        self.nas_root = (nas_root or os.environ.get(NAS_ROOT_ENV) or DEFAULT_NAS_ROOT).strip()

    def normalize_request(
        self,
        path: str | Path,
        *,
        category: str | None = None,
        mirror: bool | None = None,
    ) -> DeleteRequest:
        return DeleteRequest(path=Path(path).expanduser(), category=_normalize_category(category), mirror=mirror)

    def build_plan(
        self,
        path: str | Path,
        *,
        category: str | None = None,
        mirror: bool | None = None,
    ) -> DeletePlan | None:
        request = self.normalize_request(path, category=category, mirror=mirror)
        resolved = request.path.resolve(strict=False)
        inferred = self._infer_category(resolved, request.category)
        if inferred is None:
            return None
        if self._is_protected_path(resolved):
            return self._blocked_plan(resolved, inferred)
        local_path, nas_path = self._build_paths(resolved, inferred)
        return DeletePlan(
            category=inferred,
            local_path=str(local_path),
            nas_path=nas_path,
            will_delete_local=True,
            will_delete_nas=True,
            approval_purpose=APPROVAL_PAPERWORK["approval_purpose"],
            approval_work=APPROVAL_PAPERWORK["approval_work"],
            requires_approval=APPROVAL_PAPERWORK["requires_approval"],
            warnings=self._warnings_for(resolved),
        )

    def _infer_category(self, path: Path, requested_category: str | None) -> str | None:
        category = _normalize_category(requested_category)
        if category is not None:
            category_dir = self.local_root / CATEGORY_DIR_NAMES[category]
            try:
                path.relative_to(category_dir)
            except ValueError:
                return None
            return category

        for candidate, dirname in CATEGORY_DIR_NAMES.items():
            category_dir = self.local_root / dirname
            try:
                path.relative_to(category_dir)
            except ValueError:
                continue
            return candidate
        return None

    def _build_paths(self, path: Path, category: str) -> tuple[Path, str]:
        category_dir = self.local_root / CATEGORY_DIR_NAMES[category]
        relative = path.relative_to(category_dir)
        nas_path = self._nas_join(category, relative)
        return path, nas_path

    def _nas_join(self, category: str, relative: Path) -> str:
        base = self.nas_root.rstrip("\\/")
        parts = [base, CATEGORY_DIR_NAMES[category]]
        if relative.parts:
            parts.extend(relative.parts)
        return "\\".join(parts)

    def _blocked_plan(self, path: Path, category: str) -> DeletePlan:
        warning = f"protected asset blocked: {path.name}"
        local_path, nas_path = self._build_paths(path, category)
        return DeletePlan(
            category=category,
            local_path=str(local_path),
            nas_path=nas_path,
            will_delete_local=False,
            will_delete_nas=False,
            approval_purpose=APPROVAL_PAPERWORK["approval_purpose"],
            approval_work=APPROVAL_PAPERWORK["approval_work"],
            requires_approval=False,
            warnings=(warning,),
        )

    def _warnings_for(self, path: Path) -> tuple[str, ...]:
        warnings: list[str] = []
        if not path.exists():
            warnings.append(f"missing path: {path}")
        return tuple(warnings)

    def _is_protected_path(self, path: Path) -> bool:
        parts = {part.casefold() for part in path.parts}
        if any(name.casefold() in parts for name in PROTECTED_NAMES):
            return True
        if path.name.casefold() in {name.casefold() for name in PROTECTED_NAMES}:
            return True
        return any(part.casefold() in {"ai-image-gallery"} for part in path.parts)


def _normalize_category(category: str | None) -> str | None:
    if category is None:
        return None
    normalized = category.strip().lower()
    return normalized if normalized in CATEGORY_DIR_NAMES else None


def _is_blocked_plan(plan: DeletePlan) -> bool:
    return not plan.will_delete_local and not plan.will_delete_nas


def _delete_mode_for(plan: DeletePlan) -> str:
    return "blocked" if _is_blocked_plan(plan) else "dry-run"


def _delete_status_for(plan: DeletePlan) -> str:
    if _is_blocked_plan(plan):
        return "blocked_protected_asset"
    return "dry_run"


def _approval_metadata_for(plan: DeletePlan) -> dict[str, Any]:
    return {
        "purpose": plan.approval_purpose,
        "work": plan.approval_work,
        "required": plan.requires_approval,
        "status": "not_required" if not plan.requires_approval else "ready_for_approval",
    }


def _format_bool(value: bool) -> str:
    return "예" if value else "아니오"


def _format_warning_lines(plan: DeletePlan) -> list[str]:
    if not plan.warnings:
        return []
    label = "차단 사유" if _is_blocked_plan(plan) else "경고 사유"
    return [f"- {label}: {', '.join(plan.warnings)}"]


def _format_user_message(plan: DeletePlan) -> str:
    category_label = CATEGORY_DIR_NAMES.get(plan.category, plan.category)
    lines = ["삭제 계획을 생성했습니다."]
    if _is_blocked_plan(plan):
        lines[0] = "운영 자산으로 판단되어 삭제 계획을 생성하지 않았습니다."
    lines.extend(
        [
            f"- 삭제 모드: {_delete_mode_for(plan)}",
            f"- 실제 삭제 실행: {_format_bool(False)}",
            f"- 분류: {category_label}",
            f"- 로컬 경로: {plan.local_path}",
            f"- NAS 경로: {plan.nas_path}",
            f"- 로컬 삭제 예정: {_format_bool(plan.will_delete_local)}",
            f"- NAS 삭제 예정: {_format_bool(plan.will_delete_nas)}",
            f"- 승인 필요: {_format_bool(plan.requires_approval)}",
            f"- 목적: {plan.approval_purpose}",
            f"- 작업: {plan.approval_work}",
        ]
    )
    lines.extend(_format_warning_lines(plan))
    lines.append("\n아직 실제 삭제는 수행하지 않았습니다.")
    return "\n".join(lines)


def _format_execution_user_message(plan: DeletePlan, *, local_delete_verified: bool) -> str:
    lines = ["로컬 삭제를 실행했습니다." if local_delete_verified else "로컬 삭제 실행 후 검증에 실패했습니다."]
    lines.extend(
        [
            f"- 승인 상태: {_format_bool(True)}",
            f"- 로컬 삭제 실행: {_format_bool(True)}",
            f"- 로컬 삭제 검증: {'성공' if local_delete_verified else '실패'}",
            f"- NAS 삭제 실행: {_format_bool(False)}",
            f"- NAS 삭제 상태: 대기",
            f"- 분류: {CATEGORY_DIR_NAMES.get(plan.category, plan.category)}",
            f"- 로컬 경로: {plan.local_path}",
            f"- NAS 경로: {plan.nas_path}",
        ]
    )
    return "\n".join(lines)


def _delete_local_target(target: Path) -> None:
    if target.is_dir() and not target.is_symlink():
        shutil.rmtree(target)
        return
    target.unlink()


delete_orchestrator = ArtifactDeleteOrchestrator()


def execute_approved_local_delete(
    path: str | Path,
    *,
    category: str | None = None,
    mirror: bool | None = None,
    approved: bool | None = None,
    local_root: str | Path | None = None,
    nas_root: str | None = None,
) -> dict[str, Any] | None:
    plan_dict = build_delete_dry_run(
        path,
        category=category,
        mirror=mirror,
        local_root=local_root,
        nas_root=nas_root,
    )
    if plan_dict is None:
        return None

    plan_dict["approved"] = approved is True
    if approved is not True:
        plan_dict["local_delete_status"] = "approval_required"
        return plan_dict
    if not plan_dict.get("will_delete_local", False):
        plan_dict["local_delete_status"] = plan_dict.get("delete_status", "blocked_protected_asset")
        return plan_dict

    target = Path(plan_dict["local_path"])
    _delete_local_target(target)
    local_delete_verified = not target.exists()
    plan_dict.update(
        {
            "deletion_executed": True,
            "local_delete_executed": True,
            "local_delete_verified": local_delete_verified,
            "nas_deletion_executed": False,
            "nas_delete_pending": True,
            "delete_mode": "actual_delete",
            "delete_status": "deleted_locally_verified" if local_delete_verified else "deleted_locally_unverified",
            "local_delete_status": "deleted_locally_verified" if local_delete_verified else "deleted_locally_unverified",
            "local_path_exists_after": target.exists(),
            "user_message": _format_execution_user_message(
                DeletePlan(
                    category=plan_dict["category"],
                    local_path=plan_dict["local_path"],
                    nas_path=plan_dict["nas_path"],
                    will_delete_local=plan_dict["will_delete_local"],
                    will_delete_nas=plan_dict["will_delete_nas"],
                    approval_purpose=plan_dict["approval_purpose"],
                    approval_work=plan_dict["approval_work"],
                    requires_approval=plan_dict["requires_approval"],
                    warnings=tuple(plan_dict.get("warnings", [])),
                ),
                local_delete_verified=local_delete_verified,
            ),
        }
    )
    return plan_dict


def build_delete_dry_run(
    path: str | Path,
    *,
    category: str | None = None,
    mirror: bool | None = None,
    local_root: str | Path | None = None,
    nas_root: str | None = None,
) -> dict[str, Any] | None:
    plan = ArtifactDeleteOrchestrator(local_root=local_root, nas_root=nas_root).build_plan(
        path,
        category=category,
        mirror=mirror,
    )
    return None if plan is None else plan.to_dict()
