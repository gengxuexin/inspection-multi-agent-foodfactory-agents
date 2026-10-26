"""风险分级 Agent（Agent 3）：可执行决策表。

原则（设计稿第二节）：
- **规则先行**，所有规则并行评估，命中结果**取最高等级**，同等级全部记录理由；
- **LLM 只在规则全部未命中时给建议**，且**不能单独定 A**；
- A级 / CCP / 证据不足 / 规则冲突 → 强制人工确认；
- 人工降级必须留痕，原等级保留供复盘。

每条规则实现 `matches()` 与 `render_reason()`，质量部可逐条评审。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from app.core.config import settings
from app.core.llm import suggest_level
from app.core.schemas import GradeResult, RiskLevel, RuleHit

LEVEL_RANK = {RiskLevel.A: 3, RiskLevel.B: 2, RiskLevel.C: 1}


def _has(text: str, words: list[str]) -> bool:
    return any(w in text for w in words)


@dataclass
class Rule:
    id: str
    level: RiskLevel
    need_human: bool
    reason_template: str
    matcher: Callable[[dict[str, Any]], bool]
    reason_builder: Optional[Callable[[dict[str, Any]], str]] = None
    name: str = ""

    def matches(self, ctx: dict[str, Any]) -> bool:
        try:
            return bool(self.matcher(ctx))
        except Exception:
            return False

    def render_reason(self, ctx: dict[str, Any]) -> str:
        if self.reason_builder:
            return self.reason_builder(ctx)
        return self.reason_template


def _t(ctx: dict[str, Any]) -> str:
    return ctx.get("raw_text", "")


# --------------------------------------------------------------------------
# A 级规则：食品安全 / CCP / 批量风险
# --------------------------------------------------------------------------
def _r_ccp_deviation(ctx: dict[str, Any]) -> bool:
    return bool(ctx.get("clause_is_ccp")) and bool(ctx.get("monitoring_out_of_cl"))


def _r_allergen(ctx: dict[str, Any]) -> bool:
    return _has(_t(ctx), ["过敏原", "交叉接触", "交叉污染", "麸质", "未申报"])


def _r_pathogen(ctx: dict[str, Any]) -> bool:
    return _has(_t(ctx), ["致病菌", "阳性", "沙门氏菌", "金黄色葡萄球菌", "检出"])


def _r_foreign(ctx: dict[str, Any]) -> bool:
    return _has(_t(ctx), ["破损", "碎片"]) and _has(_t(ctx), ["刀具", "工具", "玻璃", "包装", "异物"])


def _r_coldchain(ctx: dict[str, Any]) -> bool:
    return _has(_t(ctx), ["断链", "冷链"]) or (
        _has(_t(ctx), ["温度"]) and _has(_t(ctx), ["超", "超出", "超标"])
    )


def _r_clean_fail(ctx: dict[str, Any]) -> bool:
    return _has(_t(ctx), ["消毒", "CIP", "清洁剂浓度"]) and _has(_t(ctx), ["失效", "不足", "残留"])


def _r_systemic(ctx: dict[str, Any]) -> bool:
    scope = ctx.get("recent_same_issue_count", 0) or 0
    batches = ctx.get("affected_batches", 0) or 0
    return scope >= 3 or batches >= 2 or _has(_t(ctx), ["整批"])


def _r_released(ctx: dict[str, Any]) -> bool:
    return _has(_t(ctx), ["已售", "已发货", "已放行"])


# --------------------------------------------------------------------------
# B 级规则：过程失控 / 扩散风险
# --------------------------------------------------------------------------
def _r_record_missing(ctx: dict[str, Any]) -> bool:
    return _has(_t(ctx), ["记录缺失", "记录不全", "漏填", "未记录", "记录未填"]) or (
        _has(_t(ctx), ["记录"]) and _has(_t(ctx), ["缺失", "不全"])
    )


def _r_equip_drift(ctx: dict[str, Any]) -> bool:
    return _has(_t(ctx), ["参数漂移", "维护逾期", "校准过期", "校准超期"])


def _r_hygiene(ctx: dict[str, Any]) -> bool:
    return _has(_t(ctx), ["积水", "排水口", "台面残留", "垃圾桶", "不洁", "地面积水"])


def _r_personnel(ctx: dict[str, Any]) -> bool:
    return _has(_t(ctx), ["未戴", "发网", "口罩", "手套破损", "洗手", "着装"])


def _r_label(ctx: dict[str, Any]) -> bool:
    return _has(_t(ctx), ["标签", "标识", "批次号"]) and _has(_t(ctx), ["错误", "不符", "不清"])


def _r_pest(ctx: dict[str, Any]) -> bool:
    return _has(_t(ctx), ["捕鼠", "粘鼠板", "风幕", "虫害设施"]) and not _has(_t(ctx), ["发现虫害"])


def _r_recurrence(ctx: dict[str, Any]) -> bool:
    return int(ctx.get("same_issue_in_30d", 0) or 0) >= 1


def _r_supplier(ctx: dict[str, Any]) -> bool:
    return _has(_t(ctx), ["供应商", "到货"]) and _has(_t(ctx), ["变形", "不符", "轻微"])


# --------------------------------------------------------------------------
# C 级规则：孤立 / 可立即纠正
# --------------------------------------------------------------------------
def _r_single_pack(ctx: dict[str, Any]) -> bool:
    return _has(_t(ctx), ["单件", "单个"]) and _has(_t(ctx), ["包装", "变形", "污渍"])


def _r_tool_place(ctx: dict[str, Any]) -> bool:
    return _has(_t(ctx), ["摆放", "定置", "堆放"])


def _r_typo(ctx: dict[str, Any]) -> bool:
    return _has(_t(ctx), ["笔误", "写错", "填错"])


def _r_facility(ctx: dict[str, Any]) -> bool:
    return _has(_t(ctx), ["照明", "灯", "门窗", "地面破损"])


def _ev(ctx: dict[str, Any], key: str, default: str = "相关项") -> str:
    return str(ctx.get(key) or default)


RULES: list[Rule] = [
    # A 级
    Rule("A01", RiskLevel.A, True,
         "{clause_id}为CCP，监控值{value}超出关键限值{CL}，已发生关键限值偏离。",
         _r_ccp_deviation,
         lambda c: f"{_ev(c,'clause_id')}为CCP，监控值{_ev(c,'monitoring_value')}超出关键限值"
                   f"{_ev(c,'critical_limit','CL')}，已发生关键限值偏离。", "CCP关键限值偏离"),
    Rule("A02", RiskLevel.A, True,
         "问题涉及过敏原{hazard}，存在未申报或交叉污染风险，影响消费者安全。",
         _r_allergen, lambda c: f"问题涉及过敏原（{_ev(c,'hazard_word','过敏原')}），"
                                "存在未申报或交叉污染风险，影响消费者安全。", "过敏原风险"),
    Rule("A03", RiskLevel.A, True,
         "检出{pathogen}，产品/环境存在致病菌污染，需立即隔离并评估。",
         _r_pathogen, lambda c: f"检出{_ev(c,'pathogen','致病菌')}，产品/环境存在致病菌污染，"
                                "需立即隔离并评估。", "致病菌检出"),
    Rule("A04", RiskLevel.A, True,
         "{foreign_object}破损，碎片可能进入产品流，构成物理危害。",
         _r_foreign, lambda c: f"{_ev(c,'foreign_object','工具/包装')}破损，碎片可能进入产品流，"
                                "构成物理危害。", "异物进入产品风险"),
    Rule("A05", RiskLevel.A, True,
         "温度{temp}超出允许范围{range}，冷链断链，产品腐败/微生物风险显著上升。",
         _r_coldchain, lambda c: f"{_ev(c,'temp_clause','温度')}超出允许范围{_ev(c,'critical_limit','CL')}，"
                                  "冷链断链，产品腐败/微生物风险显著上升。", "冷链断链"),
    Rule("A06", RiskLevel.A, True,
         "{cleaning_step}失效，接触面可能残留{residue}，存在生物/化学污染风险。",
         _r_clean_fail, lambda c: f"{_ev(c,'cleaning_step','清洁消毒')}失效，接触面可能残留"
                                  f"{_ev(c,'residue','微生物/化学残留')}，存在生物/化学污染风险。",
         "清洁消毒失效"),
    Rule("A07", RiskLevel.A, True,
         "同类问题在{scope}出现{n}例/涉及{m}批次，属系统性偏差，影响范围超出单件。",
         _r_systemic, lambda c: f"同类问题在{_ev(c,'zone','该区域')}出现"
                                 f"{_ev(c,'recent_same_issue_count','3')}例/涉及"
                                 f"{_ev(c,'affected_batches','2')}批次，属系统性偏差，"
                                 "影响范围超出单件。", "批量/系统性偏差"),
    Rule("A08", RiskLevel.A, True,
         "问题涉及已放行批次{batch}，需启动追溯与召回评估。",
         _r_released, lambda c: f"问题涉及已放行/已售批次{_ev(c,'involved_batch','相关')}，"
                                 "需启动追溯与召回评估。", "已放行批次"),
    # B 级
    Rule("B01", RiskLevel.B, True,
         "{record}缺失，暂无法确认过程受控，需补证并加严抽检。",
         _r_record_missing, lambda c: f"{_ev(c,'record','相关')}记录缺失，暂无法确认过程受控，"
                                       "需补证并加严抽检。", "记录缺失"),
    Rule("B02", RiskLevel.B, False,
         "{equipment}参数偏离设定值但未超CL/维护逾期，存在失控趋势。",
         _r_equip_drift, lambda c: f"{_ev(c,'involved_equipment','设备')}参数偏离设定值但未超CL"
                                   "/维护逾期，存在失控趋势。", "设备参数漂移"),
    Rule("B03", RiskLevel.B, False,
         "{area}存在{issue}，违反SSOP卫生要求，可能滋生微生物。",
         _r_hygiene, lambda c: f"{_ev(c,'zone','该区域')}存在{_ev(c,'phenomenon','卫生偏差')}，"
                               "违反SSOP卫生要求，可能滋生微生物。", "一般卫生偏差"),
    Rule("B04", RiskLevel.B, False,
         "{person}未遵守{hygiene_rule}，存在人为污染风险。",
         _r_personnel, lambda c: f"作业人员未遵守{_ev(c,'hygiene_rule','人员卫生规范')}，"
                                 "存在人为污染风险。", "人员卫生/着装"),
    Rule("B05", RiskLevel.B, False,
         "{label_error}导致标识不清，但批次可隔离追溯，未流入下道。",
         _r_label, lambda c: f"{_ev(c,'label_error','标签/标识错误')}导致标识不清，"
                             "但批次可隔离追溯，未流入下道。", "标识/批次记录错误"),
    Rule("B06", RiskLevel.B, False,
         "{pest_device}异常，防控屏障失效，需修复并加严监测。",
         _r_pest, lambda c: f"{_ev(c,'pest_device','虫害设施')}异常，防控屏障失效，"
                            "需修复并加严监测。", "虫害设施异常"),
    Rule("B07", RiskLevel.B, False,
         "{issue}在30天内第2次出现，呈复发趋势，需根因分析。",
         _r_recurrence, lambda c: f"{_ev(c,'phenomenon','该问题')}在30天内再次出现，"
                                  "呈复发趋势，需根因分析。", "30天内复发"),
    Rule("B08", RiskLevel.B, False,
         "{supplier_issue}，属供应商到货偏差，需反馈并加严验收。",
         _r_supplier, lambda c: f"{_ev(c,'supplier_issue','到货偏差')}，属供应商到货偏差，"
                                "需反馈并加严验收。", "供应商到货偏差"),
    # C 级
    Rule("C01", RiskLevel.C, False,
         "{packaging_issue}为孤立单件问题，可立即纠正，无产品安全影响。",
         _r_single_pack, lambda c: f"{_ev(c,'packaging_issue','单件包装问题')}为孤立单件问题，"
                                   "可立即纠正，无产品安全影响。", "单件包装轻微问题"),
    Rule("C02", RiskLevel.C, False,
         "{tool_issue}属现场定置问题，可立即整改。",
         _r_tool_place, lambda c: f"{_ev(c,'tool_issue','工具/容器摆放')}属现场定置问题，"
                                  "可立即整改。", "工具定置不规范"),
    Rule("C03", RiskLevel.C, False,
         "{record_typo}为笔误，可当场更正，追溯链完整。",
         _r_typo, lambda c: f"{_ev(c,'record_typo','记录填写错误')}为笔误，可当场更正，追溯链完整。",
         "记录笔误"),
    Rule("C04", RiskLevel.C, False,
         "{facility_issue}为设施轻微缺陷，未构成卫生危害。",
         _r_facility, lambda c: f"{_ev(c,'facility_issue','照明/门窗/地面')}为设施轻微缺陷，"
                                 "未构成卫生危害。", "设施轻微缺陷"),
]


class RiskGradeAgent:
    """规则并行评估 → 取最高等级 → 分流人工确认。"""

    def __init__(self, rules: Optional[list[Rule]] = None) -> None:
        self.rules = rules or RULES
        self.cfg = settings.grade

    def grade_issue(
        self,
        issue_id: str,
        raw_text: str,
        clause_id: str = "",
        clause_text: str = "",
        clause_is_ccp: bool = False,
        critical_limit: str = "",
        context: Optional[dict[str, Any]] = None,
    ) -> GradeResult:
        ctx: dict[str, Any] = {
            "raw_text": raw_text,
            "clause_id": clause_id,
            "clause_text": clause_text,
            "clause_is_ccp": clause_is_ccp,
            "critical_limit": critical_limit,
        }
        ctx.update(context or {})

        # 1) 并行评估所有规则
        hits: list[RuleHit] = []
        for rule in self.rules:
            if rule.matches(ctx):
                hits.append(
                    RuleHit(
                        rule_id=rule.id,
                        level=rule.level,
                        reason=rule.render_reason(ctx),
                        need_human=rule.need_human,
                    )
                )

        evidence_gaps = self._evidence_gaps(ctx)

        # 2) 规则全部未命中 → LLM 建议分支
        if not hits:
            lv, reason, conf = suggest_level(raw_text, clause_text, clause_id)
            need_human = True if lv == "A" else conf < self.cfg.llm_confidence_threshold
            return GradeResult(
                issue_id=issue_id,
                level=RiskLevel(lv),
                source="llm_suggestion",
                hits=[],
                llm_reason=reason,
                llm_confidence=conf,
                llm_clause_ref=clause_id,
                need_human=need_human,
                need_human_reason=(
                    "LLM 建议 A 级，模型不得单独定 A，强制人工确认"
                    if lv == "A"
                    else ("模型建议置信度不足，需人工确认" if need_human else "")
                ),
                evidence_gaps=evidence_gaps,
                clause_is_ccp=clause_is_ccp,
                agent_version=self.cfg.agent_version,
            )

        # 3) 取最高等级
        final_level = max(hits, key=lambda h: LEVEL_RANK[h.level]).level
        top_rule = max(hits, key=lambda h: LEVEL_RANK[h.level]).rule_id

        # 4) 人工确认条件
        reasons: list[str] = []
        if final_level == RiskLevel.A:
            reasons.append("A级必须人工确认")
        if clause_is_ccp:
            reasons.append("CCP相关强制人工确认")
        if any(h.need_human for h in hits):
            reasons.append("命中规则要求人工确认")
        if self._has_conflicting_levels(hits):
            reasons.append("命中不同等级规则，取最高等级后需人工复核")
        if evidence_gaps:
            reasons.append("证据不足，待补证后重新评估")
        if bool(ctx.get("no_reliable_clause")):
            reasons.append("条款映射为 no_reliable_match，不自动定级")

        # 证据不足时默认不低于 B（宁可过度升级，不可漏判）
        if evidence_gaps and final_level == RiskLevel.C:
            final_level = RiskLevel.B
            reasons.append("证据不足，默认不低于 B 级")

        return GradeResult(
            issue_id=issue_id,
            level=final_level,
            source="rule",
            top_rule=top_rule,
            hits=hits,
            need_human=bool(reasons),
            need_human_reason="；".join(reasons),
            evidence_gaps=evidence_gaps,
            clause_is_ccp=clause_is_ccp,
            agent_version=self.cfg.agent_version,
        )

    @staticmethod
    def _has_conflicting_levels(hits: list[RuleHit]) -> bool:
        return len({h.level for h in hits}) > 1

    @staticmethod
    def _evidence_gaps(ctx: dict[str, Any]) -> list[str]:
        """条件证据不足检测：规则条件所需字段缺失时记录，保证可追溯。"""
        text: str = ctx.get("raw_text", "")
        gaps: list[str] = []
        if ("温度" in text or "pH" in text) and not re.search(r"-?\d+(\.\d+)?\s*(℃|度|pH|C)", text):
            gaps.append("缺少具体监控数值")
        if _has(text, ["温度"]) and "已接触产品" not in text and "冷链" not in text:
            gaps.append("未说明是否已接触产品")
        return gaps

    # -- 人工确认 / 降级留痕 --------------------------------------------
    @staticmethod
    def confirm(
        result: GradeResult,
        confirmed_by: str,
        final_level: Optional[RiskLevel] = None,
        downgrade_reason: str = "",
    ) -> GradeResult:
        """人工确认。A级降级必须填写理由，并保留 original_level 供复盘。"""
        from app.core.schemas import utcnow

        new_level = final_level or result.level
        original = result.level
        if original == RiskLevel.A and new_level != RiskLevel.A and not downgrade_reason:
            raise ValueError("A级降级必须填写降级理由并留痕")

        updated = result.model_copy(deep=True)
        updated.original_level = original if new_level != original else None
        updated.level = new_level
        updated.source = "human"
        updated.confirmed_by = confirmed_by
        updated.confirmed_at = utcnow()
        updated.need_human = False
        updated.need_human_reason = f"已由 {confirmed_by} 确认等级 {new_level.value}"
        if new_level != original:
            updated.downgrade_reason = downgrade_reason or "等级调整"
        return updated
