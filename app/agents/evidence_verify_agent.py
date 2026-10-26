"""证据验证 Agent（Agent 6）：五道闸门，防"口头答应、表上假改"。

判定语义（设计稿第一节）：
    通过 → 进入下一道；打回 → 终止，无效整改+1，倒计时继续；存疑 → 继续，最终汇总转人工。

分流（设计稿第四节）：
    任一打回 → reject；A级/CCP → human_onsite_review（硬规则，自动结果只作参考）；
    B/C 有存疑 → human_review；否则 auto_pass。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from app.core.config import settings
from app.core.llm import vlm_verify
from app.core.schemas import (
    CheckResult,
    CheckVerdict,
    DispatchOrder,
    EvidenceItem,
    EvidenceSubmission,
    EvidenceType,
    GradeResult,
    RiskLevel,
    VerifyResult,
    VerifyRoute,
)
from app.rag.fingerprint import doc_text_similarity, phash_similarity


@dataclass
class HistoryRecord:
    """历史证据库条目。"""

    history_id: str
    issue_id: str
    sha256: str
    phash: str
    text: str = ""
    submitted_at: Optional[datetime] = None


class HistoryStore:
    """历史证据库（生产环境替换为 PostgreSQL/对象存储索引）。"""

    def __init__(self) -> None:
        self.records: list[HistoryRecord] = []

    def add(self, rec: HistoryRecord) -> None:
        self.records.append(rec)

    def find_matches(
        self, sha256: str = "", phash: str = "", text: str = ""
    ) -> list[tuple[HistoryRecord, str, float]]:
        """返回 [(历史记录, 命中规则ID, 相似度)]。"""
        out: list[tuple[HistoryRecord, str, float]] = []
        for rec in self.records:
            if sha256 and rec.sha256 and rec.sha256 == sha256:
                out.append((rec, "E301", 1.0))
                continue
            if phash and rec.phash:
                sim = phash_similarity(phash, rec.phash)
                if sim >= settings.evidence.image_uncertain_low:
                    rule = ("E302" if sim >= settings.evidence.image_reject_threshold
                            else "E303")
                    out.append((rec, rule, round(sim, 4)))
                    continue
            if text and rec.text:
                sim = doc_text_similarity(text, rec.text)
                if sim >= 0.9:
                    out.append((rec, "E305", round(sim, 4)))
        return out


class EvidenceVerifyAgent:
    """五道校验顺序执行。任一打回即终止。"""

    def __init__(self, history: Optional[HistoryStore] = None) -> None:
        self.history = history or HistoryStore()
        self.cfg = settings.evidence

    # ------------------------------------------------------------------
    def verify_evidence(
        self,
        submission: EvidenceSubmission,
        order: DispatchOrder,
        grade: GradeResult,
        clause_is_ccp: bool = False,
        invalid_fix_count: int = 0,
        linked_records: Optional[dict[str, Any]] = None,
        requirement: str = "",
    ) -> VerifyResult:
        checks: list[CheckResult] = []
        dispatched_at = order.dispatched_at
        provider = "rules_only"

        # 校验1：清单完整性 ------------------------------------------------
        c1 = self.check_completeness(submission, order.required_evidence)
        checks.append(c1)
        if c1.verdict == CheckVerdict.REJECT:
            return self._finalize(submission, checks, VerifyRoute.REJECT, invalid_fix_count, "rules_only")

        # 校验2：元数据（防旧图/旧文件）------------------------------------
        for ev in submission.evidences:
            c = self.check_metadata(ev, dispatched_at, submission.submitted_at)
            checks.append(c)
            if c.verdict == CheckVerdict.REJECT:
                return self._finalize(submission, checks, VerifyRoute.REJECT, invalid_fix_count, "rules_only")

        # 校验3：历史复用检测 ----------------------------------------------
        for ev in submission.evidences:
            c = self.check_reuse(ev)
            checks.append(c)
            if c.verdict == CheckVerdict.REJECT:
                return self._finalize(submission, checks, VerifyRoute.REJECT, invalid_fix_count, "rules_only")

        # 校验4：语义核验（VLM / 文档字段抽取）-----------------------------
        for ev in submission.evidences:
            if ev.type in (EvidenceType.PHOTO, EvidenceType.VIDEO):
                j, provider = vlm_verify(ev.file_path, requirement or order.requirement)
                c = self._vlm_to_check(ev, j)
            else:
                c = self.doc_field_check(ev, requirement or order.requirement)
            checks.append(c)
            if c.verdict == CheckVerdict.REJECT:
                return self._finalize(submission, checks, VerifyRoute.REJECT, invalid_fix_count, provider)

        # 校验5：交叉校验 ---------------------------------------------------
        c5 = self.cross_check(submission, linked_records or {})
        checks.append(c5)
        if c5.verdict == CheckVerdict.REJECT:
            return self._finalize(submission, checks, VerifyRoute.REJECT, invalid_fix_count, provider)

        # 分流 -------------------------------------------------------------
        route, reason = self.route_after_validation(grade, clause_is_ccp, checks)
        return self._finalize(submission, checks, route, invalid_fix_count, provider, reason)

    # ------------------------------------------------------------------
    # 校验1：清单完整性（E101-E104）
    # ------------------------------------------------------------------
    def check_completeness(
        self, submission: EvidenceSubmission, required: list[EvidenceType]
    ) -> CheckResult:
        if not submission.evidences:
            return CheckResult(
                check="清单完整性", rule_id="E104", verdict=CheckVerdict.REJECT,
                reason="未提交任何整改证据。",
            )
        submitted = [e.type for e in submission.evidences]
        missing = [t.value for t in required if t not in submitted]
        if missing:
            return CheckResult(
                check="清单完整性", rule_id="E101", verdict=CheckVerdict.REJECT,
                reason=f"缺少必交证据：{'、'.join(missing)}，整改未完成。",
                detail={"required": [t.value for t in required], "missing": missing},
            )
        for ev in submission.evidences:
            if not ev.file_path:
                return CheckResult(
                    check="清单完整性", rule_id="E103", verdict=CheckVerdict.REJECT,
                    reason=f"证据文件 {ev.file_name or ev.evidence_id} 无法读取，请重新提交。",
                    evidence_id=ev.evidence_id,
                )
        return CheckResult(
            check="清单完整性", rule_id="E102", verdict=CheckVerdict.PASS,
            reason="必交证据齐全。",
            detail={"required": [t.value for t in required]},
        )

    # ------------------------------------------------------------------
    # 校验2：元数据（E201-E208）
    # ------------------------------------------------------------------
    def check_metadata(
        self, ev: EvidenceItem, dispatched_at: datetime, submitted_at: datetime
    ) -> CheckResult:
        base = dict(check="元数据校验", evidence_id=ev.evidence_id)

        if ev.type in (EvidenceType.PHOTO, EvidenceType.VIDEO):
            if ev.captured_at is None:
                return CheckResult(
                    **base, rule_id="E203", verdict=CheckVerdict.UNCERTAIN,
                    reason="照片缺少拍摄时间元数据，无法确认是否为整改后拍摄。",
                )
            if ev.captured_at < dispatched_at:
                return CheckResult(
                    **base, rule_id="E201", verdict=CheckVerdict.REJECT,
                    reason=(f"照片拍摄时间{ev.captured_at.isoformat()}早于派单时间"
                            f"{dispatched_at.isoformat()}，疑似旧图。"),
                )
            if ev.captured_at > submitted_at:
                return CheckResult(
                    **base, rule_id="E202", verdict=CheckVerdict.UNCERTAIN,
                    reason=(f"照片时间{ev.captured_at.isoformat()}晚于提交时间"
                            f"{submitted_at.isoformat()}，时间异常，需人工确认。"),
                )
            if not ev.location:
                return CheckResult(
                    **base, rule_id="E204", verdict=CheckVerdict.UNCERTAIN,
                    reason="照片缺少定位信息，无法确认是否在厂区拍摄，需人工确认。",
                )
            return CheckResult(**base, rule_id="E208", verdict=CheckVerdict.PASS,
                               reason="拍摄时间在派单后、提交前，定位已提供。")

        if ev.type == EvidenceType.DOCUMENT:
            doc_date = ev.doc_fields.get("date")
            if doc_date:
                try:
                    d = datetime.fromisoformat(str(doc_date))
                    if d < dispatched_at:
                        return CheckResult(
                            **base, rule_id="E205", verdict=CheckVerdict.REJECT,
                            reason=f"文档{ev.file_name}内日期{str(doc_date)[:10]}早于派单时间，疑似复用旧记录。",
                        )
                except ValueError:
                    pass
            return CheckResult(**base, rule_id="E208", verdict=CheckVerdict.PASS,
                               reason="文档日期校验通过。")

        ts = ev.doc_fields.get("timestamp")
        if ts:
            try:
                d = datetime.fromisoformat(str(ts))
                if d < dispatched_at:
                    return CheckResult(
                        **base, rule_id="E206", verdict=CheckVerdict.REJECT,
                        reason=f"系统记录时间戳{str(ts)[:19]}早于派单时间，非本次整改记录。",
                    )
            except ValueError:
                pass
        return CheckResult(**base, rule_id="E208", verdict=CheckVerdict.PASS,
                           reason="系统记录时间戳校验通过。")

    # ------------------------------------------------------------------
    # 校验3：历史复用（E301-E305）
    # ------------------------------------------------------------------
    def check_reuse(self, ev: EvidenceItem) -> CheckResult:
        base = dict(check="历史复用检测", evidence_id=ev.evidence_id)
        if not ev.sha256 and not ev.phash and not ev.doc_fields:
            return CheckResult(**base, rule_id="E304", verdict=CheckVerdict.PASS,
                               reason="无可比对指纹，跳过复用检测（记录为首次证据）。")
        matches = self.history.find_matches(sha256=ev.sha256, phash=ev.phash,
                                            text=str(ev.doc_fields.get("text", "")))
        if not matches:
            return CheckResult(**base, rule_id="E304", verdict=CheckVerdict.PASS,
                               reason="与历史证据无相似记录。")
        for rec, rule_id, sim in matches:
            if rule_id == "E301":
                return CheckResult(**base, rule_id=rule_id, verdict=CheckVerdict.REJECT,
                                   reason=f"证据与历史记录{rec.history_id}完全一致，属重复提交。",
                                   detail={"history_id": rec.history_id, "similarity": sim})
            if rule_id in ("E302", "E305"):
                return CheckResult(**base, rule_id=rule_id, verdict=CheckVerdict.REJECT,
                                   reason=(f"证据与历史记录{rec.history_id}高度相似（相似度{sim}），疑似复用。"),
                                   detail={"history_id": rec.history_id, "similarity": sim})
            if rule_id == "E303":
                return CheckResult(**base, rule_id=rule_id, verdict=CheckVerdict.UNCERTAIN,
                                   reason=(f"证据与历史记录{rec.history_id}相似度{sim}，"
                                           "需人工确认是否为同一场景新拍。"),
                                   detail={"history_id": rec.history_id, "similarity": sim})
        return CheckResult(**base, rule_id="E304", verdict=CheckVerdict.PASS, reason="复用检测通过。")

    # ------------------------------------------------------------------
    # 校验4：语义核验（E401-E406）
    # ------------------------------------------------------------------
    @staticmethod
    def _vlm_to_check(ev: EvidenceItem, j: Any) -> CheckResult:
        base = dict(check="语义核验", evidence_id=ev.evidence_id)
        detail = {"vlm": j.model_dump() if hasattr(j, "model_dump") else dict(j)}
        if j.verdict == "mismatch" and j.confidence >= 0.8:
            return CheckResult(**base, rule_id="E401", verdict=CheckVerdict.REJECT,
                               reason=f"照片内容与整改要求不符：{j.reason}。", detail=detail)
        if j.verdict == "match" and j.confidence >= settings.evidence.vlm_pass_threshold:
            return CheckResult(**base, rule_id="E402", verdict=CheckVerdict.PASS,
                               reason=f"内容一致：{j.reason}", detail=detail)
        return CheckResult(**base, rule_id="E403", verdict=CheckVerdict.UNCERTAIN,
                           reason=f"照片内容无法可靠判断是否完成整改：{j.reason}", detail=detail)

    def doc_field_check(self, ev: EvidenceItem, requirement: str) -> CheckResult:
        """文档类证据：LLM 字段抽取比对（此处为规则化实现）。"""
        base = dict(check="语义核验", evidence_id=ev.evidence_id)
        fields = ev.doc_fields or {}
        if not fields:
            return CheckResult(**base, rule_id="E404", verdict=CheckVerdict.UNCERTAIN,
                               reason=f"文档{ev.file_name}未能抽取关键字段，需人工核对。")
        expected = str(fields.get("expected", ""))
        actual = str(fields.get("actual", ""))
        if expected and actual:
            if actual.strip() == expected.strip():
                return CheckResult(**base, rule_id="E405", verdict=CheckVerdict.PASS,
                                   reason=f"文档关键字段与整改要求一致（{actual}）。")
            return CheckResult(**base, rule_id="E406", verdict=CheckVerdict.REJECT,
                               reason=f"文档字段值{actual}，与整改要求{expected}不符。",
                               detail={"expected": expected, "actual": actual})
        return CheckResult(**base, rule_id="E405", verdict=CheckVerdict.PASS,
                           reason="文档关键字段校验通过。")

    # ------------------------------------------------------------------
    # 校验5：交叉校验（E501-E504）
    # ------------------------------------------------------------------
    def cross_check(self, submission: EvidenceSubmission, linked: dict[str, Any]) -> CheckResult:
        base = dict(check="交叉校验")
        if not linked:
            return CheckResult(**base, rule_id="E502", verdict=CheckVerdict.UNCERTAIN,
                               reason="缺少关联系统记录，无法交叉验证整改结果。")
        if linked.get("retest_failed"):
            return CheckResult(**base, rule_id="E504", verdict=CheckVerdict.REJECT,
                               reason=f"复检结果{linked.get('retest_result','不合格')}，整改未达效果。")
        claim = str(linked.get("evidence_claim", ""))
        actual = str(linked.get("record_value", ""))
        if claim and actual and claim != actual:
            return CheckResult(**base, rule_id="E501", verdict=CheckVerdict.UNCERTAIN,
                               reason=f"证据显示{claim}，但系统记录显示{actual}，数据矛盾。")
        return CheckResult(**base, rule_id="E503", verdict=CheckVerdict.PASS,
                           reason="关联系统记录与证据一致。")

    # ------------------------------------------------------------------
    # 分流
    # ------------------------------------------------------------------
    @staticmethod
    def route_after_validation(
        grade: GradeResult, clause_is_ccp: bool, checks: list[CheckResult]
    ) -> tuple[VerifyRoute, str]:
        if any(c.verdict == CheckVerdict.REJECT for c in checks):
            return VerifyRoute.REJECT, "存在明确不合格项，直接打回"
        # A级 / CCP：无论自动结果如何，强制人工现场复核（硬规则）
        if grade.level == RiskLevel.A or clause_is_ccp:
            return VerifyRoute.HUMAN_ONSITE, "A级/CCP相关问题强制人工现场复核"
        uncertain = [c for c in checks if c.verdict == CheckVerdict.UNCERTAIN]
        if uncertain:
            ids = "、".join(sorted({c.rule_id for c in uncertain}))
            return VerifyRoute.HUMAN_REVIEW, f"存在存疑校验（{ids}），转远程人工复核"
        return VerifyRoute.AUTO_PASS, "全部校验通过，进入复查抽验"

    def _finalize(
        self,
        submission: EvidenceSubmission,
        checks: list[CheckResult],
        route: VerifyRoute,
        invalid_fix_count: int,
        provider: str,
        reason: str = "",
    ) -> VerifyResult:
        # 打回后果：无效整改+1，**倒计时不重置**
        count = invalid_fix_count + (1 if route == VerifyRoute.REJECT else 0)
        return VerifyResult(
            issue_id=submission.issue_id,
            route=route,
            checks=checks,
            invalid_fix_count=count,
            deadline_kept=True,
            need_human_reason=reason,
            vlm_used=provider,
            agent_version=self.cfg.agent_version,
        )

    # ------------------------------------------------------------------
    # 打回后果与时钟规则（第六节）
    # ------------------------------------------------------------------
    @staticmethod
    def reject_consequences(invalid_fix_count: int, level: RiskLevel) -> list[str]:
        """连续打回升级规则。"""
        actions = [f"无效整改次数累计至 {invalid_fix_count}，纳入责任部门月度统计"]
        if invalid_fix_count >= settings.evidence.consecutive_reject_notify:
            actions.append("通知厂长/质量总监")
        elif invalid_fix_count >= settings.evidence.consecutive_reject_escalate:
            actions.append("自动升级一级响应")
        if level == RiskLevel.A:
            actions.append("A级任何一次打回，强制质量经理知悉")
        return actions
