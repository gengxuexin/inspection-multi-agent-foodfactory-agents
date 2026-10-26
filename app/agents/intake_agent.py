"""巡检录入 Agent（Agent 1）。

把语音/照片/文字的零散巡检记录转成结构化问题单草稿。
抽取置信度低的字段标黄要求巡检员确认——**不自动提交**。
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from app.core.schemas import EvidenceType, IssueDraft, utcnow

_ZONE_RE = re.compile(r"(前处理间|加工间|灌装间|内包材库|外包材库|冷库|发酵间|烘焙间|罐装区|金检间|仓库|更衣间|洗手间)")
_PROCESS_RE = re.compile(r"(清洗|切割|腌制|发酵|烘焙|灌装|封口|金检|冷却|入库|转运)")
_BATCH_RE = re.compile(r"(?:批次|批号)\s*[:：]?\s*([A-Za-z0-9\-_]+)")
_EQUIP_RE = re.compile(r"(温度计|pH计|发酵罐|烤箱|灌装机|金检机|冷库|输送带|搅拌罐|[A-Za-z]+设备)")

# 问题类别 → 建议证据类型（与证据验证 Agent 的必交清单模板保持一致）
_CATEGORY_EVIDENCE: dict[str, list[EvidenceType]] = {
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
}

_TEXT_KEYWORDS = {
    "温度记录缺失/断链": ["温度", "冷库", "断链"],
    "清洁不彻底/积水": ["积水", "清洁", "排水"],
    "人员卫生/着装": ["发网", "口罩", "手套", "洗手"],
    "异物风险（工具/包装破损）": ["破损", "异物", "碎片"],
    "标签标识错误": ["标签", "标识"],
    "虫害设施异常": ["捕鼠", "粘鼠板", "风幕"],
    "设备参数漂移": ["漂移", "维护", "校准"],
    "记录缺失/笔误": ["记录", "台账", "笔误"],
    "设施轻微缺陷": ["照明", "门窗"],
    "供应商到货偏差": ["供应商", "到货"],
}


class IntakeAgent:
    """结构化抽取 + 低置信字段标记。"""

    def draft_issue(
        self,
        issue_id: str,
        raw_text: str,
        reporter: str = "巡检员",
        photos: Optional[list[str]] = None,
        discovered_at: Optional[datetime] = None,
        known_zone: str = "",
    ) -> IssueDraft:
        text = raw_text.strip()
        zone = known_zone or (m.group(0) if (m := _ZONE_RE.search(text)) else "")
        process = m.group(0) if (m := _PROCESS_RE.search(text)) else ""
        batch = m.group(1) if (m := _BATCH_RE.search(text)) else ""
        equip = m.group(0) if (m := _EQUIP_RE.search(text)) else ""

        category = ""
        for cat, kws in _TEXT_KEYWORDS.items():
            if any(k in text for k in kws):
                category = cat
                break

        low_conf: list[str] = []
        if not zone:
            low_conf.append("zone")
        if not process:
            low_conf.append("process")
        if not category:
            low_conf.append("initial_category")

        phenomenon = re.sub(r"^(今天|上午|下午|现场)?(巡检)?(发现|发现存在|存在|有)\s*", "", text)

        return IssueDraft(
            issue_id=issue_id,
            raw_text=text,
            reporter=reporter,
            discovered_at=discovered_at or utcnow(),
            zone=zone,
            workstation="",
            process=process,
            phenomenon=phenomenon or text,
            involved_batch=batch,
            involved_equipment=equip,
            initial_category=category,
            suggested_evidence_types=_CATEGORY_EVIDENCE.get(category, [EvidenceType.PHOTO]),
            low_confidence_fields=low_conf,
            photos=photos or [],
        )
