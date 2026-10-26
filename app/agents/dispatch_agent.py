"""整改派单 Agent（Agent 4）。

派单硬规则：**证据类型由问题类型+等级决定，整改人不能自选删除必交项。**
缺项在证据验证第一道校验直接打回。
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from app.core.config import settings
from app.core.schemas import (
    ClauseMapResult,
    DispatchOrder,
    EvidenceType,
    GradeResult,
    IssueDraft,
    RiskLevel,
    utcnow,
)

# 问题类型 → 必交证据清单（对应设计稿第二节表）
REQUIRED_EVIDENCE: dict[str, list[EvidenceType]] = {
    "温度记录缺失/断链": [EvidenceType.DOCUMENT, EvidenceType.PHOTO, EvidenceType.DOCUMENT],
    "清洁不彻底/积水": [EvidenceType.PHOTO, EvidenceType.DOCUMENT],
    "人员卫生/着装": [EvidenceType.PHOTO, EvidenceType.DOCUMENT],
    "异物风险（工具/包装破损）": [EvidenceType.PHOTO, EvidenceType.DOCUMENT],
    "标签标识错误": [EvidenceType.PHOTO, EvidenceType.DOCUMENT],
    "虫害设施异常": [EvidenceType.PHOTO, EvidenceType.DOCUMENT],
    "设备参数漂移": [EvidenceType.DOCUMENT, EvidenceType.PHOTO],
    "记录缺失/笔误": [EvidenceType.DOCUMENT],
    "设施轻微缺陷": [EvidenceType.PHOTO],
    "供应商到货偏差": [EvidenceType.PHOTO, EvidenceType.DOCUMENT],
    "过敏原交叉污染": [EvidenceType.PHOTO, EvidenceType.DOCUMENT, EvidenceType.SYSTEM_RECORD],
}

# 问题类型 → 交叉校验数据源
CROSS_CHECK_SOURCES: dict[str, list[str]] = {
    "温度记录缺失/断链": ["冷库日志", "设备运行记录"],
    "清洁不彻底/积水": ["环境采样结果"],
    "人员卫生/着装": ["监控抽查记录"],
    "异物风险（工具/包装破损）": ["金属探测记录", "X光记录"],
    "标签标识错误": ["放行记录"],
    "虫害设施异常": ["服务商报告"],
    "设备参数漂移": ["设备日志", "维护记录"],
    "记录缺失/笔误": ["原始记录追溯链"],
}

# A 级追加必交项：临时遏制证明 + 复检数据
_A_EXTRA = [EvidenceType.PHOTO, EvidenceType.DOCUMENT]

# 等级 → 默认责任部门
_DEFAULT_DEPT = {
    RiskLevel.A: "生产部/质量部",
    RiskLevel.B: "生产部",
    RiskLevel.C: "当班班组",
}


def required_evidence_types(
    category: str, level: RiskLevel, initial_category: str = ""
) -> list[EvidenceType]:
    """证据类型由问题类型 + 等级决定。缺任意必交项 → 校验1直接打回。"""
    cat = category or initial_category
    types = list(REQUIRED_EVIDENCE.get(cat, [EvidenceType.PHOTO, EvidenceType.DOCUMENT]))
    if level == RiskLevel.A:
        for t in _A_EXTRA:
            if t not in types:
                types.append(t)
    return types


class DispatchAgent:
    def build_order(
        self,
        draft: IssueDraft,
        grade: GradeResult,
        clause_map: Optional[ClauseMapResult] = None,
        owner: str = "",
        department: str = "",
        now: Optional[datetime] = None,
        deadline_override: Optional[datetime] = None,
    ) -> DispatchOrder:
        now = now or utcnow()
        hours = settings.clock.fix_deadline_hours[grade.level.value]
        deadline = deadline_override or (now + timedelta(hours=hours))

        category = ""
        if clause_map and clause_map.candidates:
            category = clause_map.candidates[0].hazard_type  # 占位，下面用条款类别覆盖
        # 优先用映射到的条款类别
        clause_category = ""
        if clause_map and clause_map.selected_clause_id:
            from app.rag.knowledge import ClauseLibrary

            c = ClauseLibrary.load().get(clause_map.selected_clause_id)
            if c:
                clause_category = c.question_category

        requirement = self._requirement_text(draft, grade, clause_category)
        return DispatchOrder(
            issue_id=draft.issue_id,
            owner=owner or self._default_owner(draft),
            department=department or _DEFAULT_DEPT[grade.level],
            requirement=requirement,
            level=grade.level,
            dispatched_at=now,
            deadline=deadline,
            required_evidence=required_evidence_types(
                clause_category, grade.level, draft.initial_category
            ),
            recheck_method="现场复查" if grade.level == RiskLevel.A else "按比例抽验",
            clause_id=(clause_map.selected_clause_id if clause_map else "") or "",
        )

    @staticmethod
    def _default_owner(draft: IssueDraft) -> str:
        dept = {"前处理间": "前处理班组长", "加工间": "加工班组长", "灌装间": "灌装班组长",
                "冷库": "仓储班组长", "发酵间": "发酵班组长", "烘焙间": "烘焙班组长"}
        return dept.get(draft.zone, "当班班组长")

    @staticmethod
    def _requirement_text(draft: IssueDraft, grade: GradeResult, clause_category: str) -> str:
        base = (
            f"针对【{draft.zone or draft.workstation or '现场'}】发现的"
            f"「{draft.phenomenon or draft.raw_text}」（问题类型：{clause_category or draft.initial_category or '通用'}），"
            f"依据条款 {grade.issue_id and ''}"
        )
        parts = [base.strip()]
        if draft.involved_batch:
            parts.append(f"涉及批次：{draft.involved_batch}")
        if draft.involved_equipment:
            parts.append(f"涉及设备：{draft.involved_equipment}")
        if grade.level == RiskLevel.A:
            parts.append("A级：24小时内完成临时遏制并提交证据，72小时内提交根因与CAPA草案")
        elif grade.level == RiskLevel.B:
            parts.append("B级：3个工作日内完成整改并提交证据，期间加严抽检")
        else:
            parts.append("C级：7个工作日内完成整改并提交证据")
        parts.append("证据须为本次整改后拍摄/导出，不得复用历史材料")
        return "；".join(parts)
