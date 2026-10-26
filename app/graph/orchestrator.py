"""问题单状态机（LangGraph 编排）。

架构不是"多个 Agent 自由对话协商"，而是：
    Orchestrator + 共享状态 + 条件路由 + 硬闸门 + Human-in-the-loop

硬闸门（Agent 无权绕过）：
1. 条款映射低置信度 → 强制人工选条款，**不进入自动分级**；
2. 分级命中 A级/CCP → 强制人工确认分级；
3. 证据验证打回 → 回到待整改，**不重置倒计时**，无效整改次数+1；
4. A级/CCP 证据 → 无论自动结果如何转人工现场复核；
5. 复查不通过 → 打回重开；30天内同类复发 → 升级 CAPA；
6. 未通过验证+复查（A级需现场复核签字）→ 系统不允许关闭。
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from typing import Any, Optional

from app.agents.clause_map_agent import ClauseMapAgent
from app.agents.dispatch_agent import DispatchAgent
from app.agents.evidence_verify_agent import EvidenceVerifyAgent, HistoryStore, HistoryRecord
from app.agents.intake_agent import IntakeAgent
from app.agents.risk_grade_agent import RiskGradeAgent
from app.core.clock import DepartmentStats, EscalationClock, HumanTaskQueue
from app.core.config import settings
from app.core.repository import Repository, get_repo
from app.core.schemas import (
    AuditRecord,
    ClauseMapResult,
    EvidenceSubmission,
    GradeResult,
    IssueState,
    IssueStateModel,
    MapMode,
    RecheckResult,
    RiskLevel,
    VerifyRoute,
    utcnow,
)
from app.rag.knowledge import ClauseLibrary
from app.rag.retriever import ClauseRetriever

# 合法状态转换表——非法转换直接抛错（设计目标：非法状态转换 0 次）
ALLOWED_TRANSITIONS: dict[IssueState, set[IssueState]] = {
    IssueState.DRAFT: {IssueState.PENDING_MAPPING},
    IssueState.PENDING_MAPPING: {IssueState.PENDING_GRADE, IssueState.UNCERTAIN_REVIEW},
    # 人工在存疑复核中解决事项后回到待分级（选条款 / 确认分级）
    IssueState.PENDING_GRADE: {IssueState.PENDING_FIX, IssueState.UNCERTAIN_REVIEW,
                               IssueState.PENDING_GRADE},
    IssueState.PENDING_FIX: {IssueState.PENDING_VERIFY, IssueState.EXTENSION_PENDING},
    IssueState.EXTENSION_PENDING: {IssueState.PENDING_FIX},
    IssueState.PENDING_VERIFY: {
        IssueState.PENDING_RECHECK,
        IssueState.REJECTED_REOPEN,
        IssueState.UNCERTAIN_REVIEW,
    },
    IssueState.UNCERTAIN_REVIEW: {
        IssueState.PENDING_RECHECK,
        IssueState.REJECTED_REOPEN,
        IssueState.PENDING_FIX,
        IssueState.PENDING_GRADE,   # 人工解决存疑事项（选条款/确认分级）后回到待分级
    },
    IssueState.REJECTED_REOPEN: {IssueState.PENDING_FIX, IssueState.PENDING_VERIFY},
    IssueState.PENDING_RECHECK: {IssueState.CLOSED, IssueState.REJECTED_REOPEN, IssueState.CAPA},
    IssueState.CAPA: {IssueState.CLOSED},
    # 已关闭后 30 天内同类复发 → 升级 CAPA（可重新打开为 CAPA 状态）
    IssueState.CLOSED: {IssueState.CAPA},
}


def _digest(obj: Any) -> str:
    s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


class TransitionError(RuntimeError):
    """非法状态转换。"""


class InspectionOrchestrator:
    """编排器：持有 Agent 实例，维护共享状态，执行条件路由与硬闸门。"""

    def __init__(
        self,
        repo: Optional[Repository] = None,
        library: Optional[ClauseLibrary] = None,
    ) -> None:
        self.repo = repo or get_repo()
        self.library = library or ClauseLibrary.load()
        self.retriever = ClauseRetriever(self.library)
        self.intake = IntakeAgent()
        self.mapper = ClauseMapAgent(self.retriever)
        self.grader = RiskGradeAgent()
        self.dispatcher = DispatchAgent()
        self.history = self._load_history()
        self.verifier = EvidenceVerifyAgent(self.history)
        self.clock = EscalationClock(self.repo)
        self.queue = HumanTaskQueue(self.repo)

    def _load_history(self) -> HistoryStore:
        h = HistoryStore()
        for r in self.repo.load_evidence_history():
            h.add(HistoryRecord(
                history_id=r["history_id"], issue_id=r["issue_id"],
                sha256=r.get("sha256") or "", phash=r.get("phash") or "",
                text=r.get("text") or "",
                submitted_at=datetime.fromisoformat(r["submitted_at"]) if r.get("submitted_at") else None,
            ))
        return h

    # ------------------------------------------------------------------
    # 审计
    # ------------------------------------------------------------------
    def _audit(
        self,
        issue: IssueStateModel,
        actor: str,
        actor_type: str,
        action: str,
        from_state: str = "",
        to_state: str = "",
        output: Any = None,
        confidence: Optional[float] = None,
        before: Optional[dict] = None,
        after: Optional[dict] = None,
        source_refs: Optional[list[str]] = None,
        agent_version: str = "",
    ) -> None:
        self.repo.append_audit(AuditRecord(
            issue_id=issue.issue_id,
            seq=self.repo.next_seq(issue.issue_id),
            actor=actor,
            actor_type=actor_type,  # type: ignore[arg-type]
            action=action,
            from_state=from_state,
            to_state=to_state,
            input_digest=_digest(issue.model_dump(mode="json", exclude={"verify"}))[:16],
            output_digest=_digest(output)[:16] if output is not None else "",
            confidence=confidence,
            before_value=before or {},
            after_value=after or {},
            source_refs=source_refs or [],
            agent_version=agent_version,
            version_no=issue.version,
        ))

    def _transition(self, issue: IssueStateModel, to_state: IssueState, actor: str) -> None:
        """受治理的状态转换：非法转换抛错，每次转换留审计。"""
        if to_state == issue.state:
            return
        if to_state not in ALLOWED_TRANSITIONS.get(issue.state, set()):
            raise TransitionError(
                f"[{issue.issue_id}] 非法状态转换：{issue.state.value} → {to_state.value}"
            )
        frm = issue.state
        issue.state = to_state
        issue.version += 1
        self._audit(issue, actor, "system" if actor.startswith("clock") else "human",
                    "transition", frm.value, to_state.value)

    def _persist(self, issue: IssueStateModel) -> IssueStateModel:
        self.repo.save_issue(issue)
        return issue

    # ==================================================================
    # 节点 1：巡检录入
    # ==================================================================
    def create_issue(
        self, issue_id: str, raw_text: str, reporter: str = "巡检员",
        photos: Optional[list[str]] = None, known_zone: str = "",
        idempotency_key: str = "",
    ) -> IssueStateModel:
        if idempotency_key:
            first = self.repo.claim_idempotency_key(idempotency_key, issue_id)
            if not first:
                existing = self.repo.load_issue(issue_id)
                if existing:
                    self._audit(existing, "system", "system", "idempotent_replay",
                                after={"idempotency_key": idempotency_key})
                    return existing
                raise TransitionError(f"幂等键 {idempotency_key} 已被占用")

        draft = self.intake.draft_issue(issue_id, raw_text, reporter, photos, known_zone=known_zone)
        issue = IssueStateModel(issue_id=issue_id, draft=draft, state=IssueState.DRAFT)
        issue.idempotency_keys = [idempotency_key] if idempotency_key else []
        self._audit(issue, "intake-agent", "agent", "draft_issue",
                    output=draft.model_dump(mode="json"), after={"low_confidence_fields": draft.low_confidence_fields},
                    agent_version="intake-v0.3")
        if draft.low_confidence_fields:
            self.queue.create(issue_id, "confirm_intake" if False else "select_clause",
                              IssueState.DRAFT, {"low_confidence_fields": draft.low_confidence_fields},
                              assignee_role="巡检员", due_hours=8)
        return self._persist(issue)

    # ==================================================================
    # 节点 2：条款映射（低置信度 → 人工选，不进入自动分级）
    # ==================================================================
    def run_clause_map(self, issue_id: str) -> IssueStateModel:
        issue = self.repo.load_issue(issue_id)
        assert issue and issue.draft
        # 主链路：草稿 → 待映射
        if issue.state == IssueState.DRAFT:
            self._transition(issue, IssueState.PENDING_MAPPING, "clause-map-agent")
        result = self.mapper.map_clause(issue_id, issue.draft.raw_text)
        issue.clause_map = result
        issue.original_top1 = result.candidates[0].clause_id if result.candidates else None
        self._audit(issue, "clause-map-agent", "agent", "map_clause", output=result.model_dump(mode="json"),
                    confidence=result.candidates[0].confidence if result.candidates else None,
                    after={"mode": result.mode.value, "needs_human": result.needs_human,
                           "relaxed": result.relaxed_filter_layers},
                    source_refs=[c.clause_id for c in result.candidates],
                    agent_version=result.agent_version)

        if result.mode == MapMode.NO_RELIABLE_MATCH:
            # 硬闸门1：不进入自动分级
            self._transition(issue, IssueState.UNCERTAIN_REVIEW, "clause-map-agent")
            self.queue.create(issue_id, "select_clause", IssueState.UNCERTAIN_REVIEW,
                              {"reason": result.needs_human_reason,
                               "candidates": [c.model_dump(mode="json") for c in result.candidates]},
                              assignee_role="质量工程师")
        else:
            self._transition(issue, IssueState.PENDING_GRADE, "clause-map-agent")
            if result.mode == MapMode.HUMAN_SELECT:
                self.queue.create(issue_id, "select_clause", IssueState.PENDING_GRADE,
                                  {"reason": result.needs_human_reason,
                                   "candidates": [c.model_dump(mode="json") for c in result.candidates]},
                                  assignee_role="质量工程师")
        return self._persist(issue)

    def confirm_clause(self, issue_id: str, clause_id: str, confirmed_by: str) -> IssueStateModel:
        """人工确认条款。结果回流样本库（不在线自动训练）。"""
        issue = self.repo.load_issue(issue_id)
        assert issue and issue.clause_map
        sample = self.mapper.build_sample(issue.clause_map, issue.draft.raw_text if issue.draft else "",
                                          clause_id, confirmed_by)
        self.repo.add_mapping_sample(sample)
        issue.selected_clause_id = clause_id
        cm = issue.clause_map
        cm.selected_clause_id = clause_id
        cm.needs_human = False
        cm.needs_human_reason = f"已由 {confirmed_by} 确认条款 {clause_id}"
        self._audit(issue, confirmed_by, "human", "confirm_clause",
                    before={"selected_clause_id": None}, after={"selected_clause_id": clause_id},
                    output=sample, agent_version=cm.agent_version)
        # 条款确认后回到"待分级"；若已流转到下游（如 prefill 后直接派单）则不强制回退
        if issue.state in (IssueState.PENDING_MAPPING, IssueState.UNCERTAIN_REVIEW):
            self._transition(issue, IssueState.PENDING_GRADE, confirmed_by)
        return self._persist(issue)

    # ==================================================================
    # 节点 3：风险分级（A级/CCP → 强制人工确认）
    # ==================================================================
    def run_grade(self, issue_id: str, context: Optional[dict[str, Any]] = None) -> IssueStateModel:
        issue = self.repo.load_issue(issue_id)
        assert issue and issue.draft
        # 硬闸门1：no_reliable_match 且未人工确认 → 不得自动定级
        if issue.clause_map and issue.clause_map.mode == MapMode.NO_RELIABLE_MATCH \
                and not issue.selected_clause_id:
            raise TransitionError("条款映射为 no_reliable_match，必须先人工选条款才能分级")

        clause_id = issue.selected_clause_id or (issue.clause_map.candidates[0].clause_id
                                                 if issue.clause_map and issue.clause_map.candidates else "")
        clause = self.library.get(clause_id) if clause_id else None
        ctx = dict(context or {})
        if clause:
            ctx.setdefault("clause_id", clause.clause_id)
            ctx.setdefault("phenomenon", issue.draft.phenomenon)
            ctx.setdefault("zone", issue.draft.zone)
            ctx.setdefault("involved_batch", issue.draft.involved_batch)
            ctx.setdefault("involved_equipment", issue.draft.involved_equipment)
            ctx.setdefault("critical_limit", clause.critical_limit)
            # CCP 监控偏离：条款原文含 CL 且描述提及超限/超出
            if clause.is_ccp:
                ctx["monitoring_out_of_cl"] = any(
                    k in issue.draft.raw_text for k in ("超出", "超限", "超标", "超", "断链"))

        result = self.grader.grade_issue(
            issue_id=issue_id,
            raw_text=issue.draft.raw_text,
            clause_id=clause.clause_id if clause else clause_id,
            clause_text=clause.text if clause else "",
            clause_is_ccp=bool(clause and clause.is_ccp),
            critical_limit=clause.critical_limit if clause else "",
            context=ctx,
        )
        issue.grade = result
        self._audit(issue, "risk-grade-agent", "agent", "grade_issue", output=result.model_dump(mode="json"),
                    confidence=result.llm_confidence or None,
                    after={"level": result.level.value, "source": result.source,
                           "hits": [h.rule_id for h in result.hits], "need_human": result.need_human},
                    source_refs=[result.llm_clause_ref] if result.llm_clause_ref else [],
                    agent_version=result.agent_version)

        # 硬闸门2：A级/CCP/低置信 → 强制人工确认
        if result.need_human:
            self._transition(issue, IssueState.UNCERTAIN_REVIEW, "risk-grade-agent")
            self.queue.create(issue_id, "confirm_grade", IssueState.UNCERTAIN_REVIEW,
                              {"level": result.level.value, "reason": result.need_human_reason,
                               "hits": [h.model_dump() for h in result.hits],
                               "llm_reason": result.llm_reason,
                               "clause_ref": result.llm_clause_ref,
                               "evidence_gaps": result.evidence_gaps},
                              assignee_role="质量授权人" if result.level == RiskLevel.A else "质量工程师")
        else:
            self._transition(issue, IssueState.PENDING_FIX, "risk-grade-agent")
            self._dispatch(issue)
        return self._persist(issue)

    def confirm_grade(
        self, issue_id: str, confirmed_by: str,
        final_level: Optional[RiskLevel] = None, downgrade_reason: str = "",
    ) -> IssueStateModel:
        issue = self.repo.load_issue(issue_id)
        assert issue and issue.grade
        before = issue.grade.model_dump(mode="json")
        updated = self.grader.confirm(issue.grade, confirmed_by, final_level, downgrade_reason)
        issue.grade = updated
        self._audit(issue, confirmed_by, "human", "confirm_grade", before=before,
                    after=updated.model_dump(mode="json"),
                    agent_version=updated.agent_version)
        self._transition(issue, IssueState.PENDING_FIX, confirmed_by)
        self._dispatch(issue)
        return self._persist(issue)

    # ==================================================================
    # 节点 4：派单
    # ==================================================================
    def _dispatch(self, issue: IssueStateModel) -> None:
        assert issue.draft and issue.grade
        issue.order = self.dispatcher.build_order(issue.draft, issue.grade, issue.clause_map)
        issue.deadline = issue.order.deadline
        self._audit(issue, "dispatch-agent", "agent", "dispatch_order",
                    output=issue.order.model_dump(mode="json"),
                    after={"deadline": issue.order.deadline.isoformat(),
                           "required_evidence": [t.value for t in issue.order.required_evidence]},
                    agent_version="dispatch-v0.3")

    # ==================================================================
    # 节点 6：证据验证（打回不重置倒计时）
    # ==================================================================
    def submit_evidence(self, issue_id: str, submission: EvidenceSubmission) -> IssueStateModel:
        issue = self.repo.load_issue(issue_id)
        assert issue and issue.order
        issue.submission = submission
        issue.state = IssueState.PENDING_VERIFY if issue.state != IssueState.PENDING_VERIFY else issue.state
        self._audit(issue, submission.submitter or "整改人", "human", "submit_evidence",
                    output={"count": len(submission.evidences)},
                    after={"types": [e.type.value for e in submission.evidences]})
        return self._persist(issue)

    def run_verify(
        self, issue_id: str, linked_records: Optional[dict[str, Any]] = None
    ) -> IssueStateModel:
        issue = self.repo.load_issue(issue_id)
        assert issue and issue.order and issue.grade and issue.submission
        clause = self.library.get(issue.selected_clause_id) if issue.selected_clause_id else None
        result = self.verifier.verify_evidence(
            submission=issue.submission,
            order=issue.order,
            grade=issue.grade,
            clause_is_ccp=bool(clause and clause.is_ccp),
            invalid_fix_count=issue.invalid_fix_count,
            linked_records=linked_records,
        )
        issue.verify = result
        before_invalid = issue.invalid_fix_count
        issue.invalid_fix_count = result.invalid_fix_count
        self._audit(issue, "evidence-verify-agent", "agent", "verify_evidence",
                    output=result.model_dump(mode="json"),
                    after={"route": result.route.value,
                           "invalid_fix_count": result.invalid_fix_count,
                           "deadline_kept": result.deadline_kept,
                           "vlm_used": result.vlm_used},
                    agent_version=result.agent_version)

        if result.route == VerifyRoute.REJECT:
            # 硬闸门3：打回 → 回到待整改，**不重置倒计时**
            issue.reject_count += 1
            self._transition(issue, IssueState.REJECTED_REOPEN, "evidence-verify-agent")
            # deadline 保持不变
            issue.deadline = issue.order.deadline
            for act in self.verifier.reject_consequences(
                result.invalid_fix_count, issue.grade.level
            ):
                self._audit(issue, "evidence-verify-agent", "agent", "reject_consequence",
                            after={"action": act, "invalid_fix_count": result.invalid_fix_count})
            self._transition(issue, IssueState.PENDING_FIX, "evidence-verify-agent")
        elif result.route == VerifyRoute.HUMAN_ONSITE:
            # 硬闸门4：A级/CCP 强制现场复核
            self._transition(issue, IssueState.UNCERTAIN_REVIEW, "evidence-verify-agent")
            self.queue.create(issue_id, "onsite_review", IssueState.UNCERTAIN_REVIEW,
                              {"reason": result.need_human_reason,
                               "checks": [c.model_dump(mode="json") for c in result.checks]},
                              assignee_role="质量授权人")
        elif result.route == VerifyRoute.HUMAN_REVIEW:
            self._transition(issue, IssueState.UNCERTAIN_REVIEW, "evidence-verify-agent")
            self.queue.create(issue_id, "remote_review", IssueState.UNCERTAIN_REVIEW,
                              {"reason": result.need_human_reason,
                               "checks": [c.model_dump(mode="json") for c in result.checks]},
                              assignee_role="质量工程师")
        else:
            self._transition(issue, IssueState.PENDING_RECHECK, "evidence-verify-agent")
        return self._persist(issue)

    def review_evidence(self, issue_id: str, passed: bool, reviewer: str, note: str = "") -> IssueStateModel:
        """人工复核证据（远程或现场）。不通过等同无效整改。"""
        issue = self.repo.load_issue(issue_id)
        assert issue and issue.verify
        if not passed:
            issue.reject_count += 1
            issue.invalid_fix_count += 1
            issue.deadline = issue.order.deadline if issue.order else issue.deadline  # 不重置
            self._audit(issue, reviewer, "human", "review_evidence_reject",
                        after={"invalid_fix_count": issue.invalid_fix_count, "note": note})
            self._transition(issue, IssueState.REJECTED_REOPEN, reviewer)
            self._transition(issue, IssueState.PENDING_FIX, reviewer)
        else:
            self._audit(issue, reviewer, "human", "review_evidence_pass", after={"note": note})
            self._transition(issue, IssueState.PENDING_RECHECK, reviewer)
        return self._persist(issue)

    # ==================================================================
    # 节点 7：复查抽验 + CAPA
    # ==================================================================
    def should_recheck(self, issue: IssueStateModel, rng_value: Optional[float] = None) -> bool:
        """按比例抽验（不预先通知现场）。"""
        level = issue.grade.level.value if issue.grade else "C"
        ratio = settings.recheck.sampling[level]
        # 可注入随机值，便于测试与重放
        import random

        return (rng_value if rng_value is not None else random.random()) < ratio

    def run_recheck(
        self, issue_id: str, passed: bool, reviewer: str,
        sampled: Optional[bool] = None, note: str = "", rng_value: Optional[float] = None,
    ) -> IssueStateModel:
        issue = self.repo.load_issue(issue_id)
        assert issue and issue.grade
        do = self.should_recheck(issue, rng_value) if sampled is None else sampled
        issue.recheck = RecheckResult(
            issue_id=issue_id, sampled=do, method="现场复查（不预先通知）" if do else "免复查",
            passed=passed, reviewer=reviewer, reviewed_at=utcnow(), note=note,
        )
        self._audit(issue, reviewer, "human", "recheck",
                    output=issue.recheck.model_dump(mode="json"),
                    after={"sampled": do, "passed": passed})
        if passed:
            self._transition(issue, IssueState.CLOSED, reviewer)
            issue.closed_at = utcnow()
            self._close_issue(issue)
        else:
            # 复查不通过等同无效整改
            issue.reject_count += 1
            issue.invalid_fix_count += 1
            issue.deadline = issue.order.deadline if issue.order else issue.deadline
            self._transition(issue, IssueState.REJECTED_REOPEN, reviewer)
            self._transition(issue, IssueState.PENDING_FIX, reviewer)
        return self._persist(issue)

    def _close_issue(self, issue: IssueStateModel) -> None:
        """关闭：证据写入历史库（供后续复用检测）+ 复发检查 → CAPA。"""
        if issue.submission:
            for ev in issue.submission.evidences:
                hid = f"H-{hashlib.md5(f'{issue.issue_id}:{ev.evidence_id}'.encode()).hexdigest()[:10]}"
                self.repo.add_evidence_history(hid, issue.issue_id, ev.sha256, ev.phash,
                                               str(ev.doc_fields.get("text", "")),
                                               issue.submission.submitted_at.isoformat())
                self.history.add(HistoryRecord(history_id=hid, issue_id=issue.issue_id,
                                               sha256=ev.sha256, phash=ev.phash,
                                               submitted_at=issue.submission.submitted_at))
        self._audit(issue, issue.recheck.reviewer if issue.recheck else "system", "system",
                    "close_issue", to_state=IssueState.CLOSED.value)
        self._check_recurrence(issue)

    def _check_recurrence(self, issue: IssueStateModel) -> None:
        """30天内同区域+工序+条款同类复发 → 升级 CAPA。"""
        if not issue.draft or not issue.selected_clause_id:
            return
        window = timedelta(days=settings.recheck.recurrence_window_days)
        cutoff = utcnow() - window
        same = [
            i for i in self.repo.list_issues(limit=1000)
            if i.issue_id != issue.issue_id
            and i.draft and i.selected_clause_id == issue.selected_clause_id
            and i.draft.zone == issue.draft.zone and i.draft.process == issue.draft.process
            and i.closed_at and i.closed_at >= cutoff
        ]
        if same:
            capa_id = f"CAPA-{issue.issue_id}"
            issue.capa_id = capa_id
            # 升级 CAPA：已关闭 → CAPA（CAPA 闭环后由授权人关闭）
            if issue.state == IssueState.CLOSED:
                self._transition(issue, IssueState.CAPA, "system")
            self._audit(issue, "system", "system", "escalate_capa",
                        after={"capa_id": capa_id, "recurrences": len(same),
                               "window_days": settings.recheck.recurrence_window_days,
                               "note": "触发 HACCP 计划/SSOP 评审提醒"})

    # ==================================================================
    # 延期审批
    # ==================================================================
    def request_extension(self, issue_id: str, hours: int, requested_by: str) -> IssueStateModel:
        issue = self.repo.load_issue(issue_id)
        assert issue and issue.grade and issue.order
        allowed, msg = self.clock.allow_extension(issue.grade.level)
        self._audit(issue, requested_by, "human", "request_extension",
                    after={"hours": hours, "allowed": allowed, "policy": msg})
        if not allowed:
            issue.state = IssueState.PENDING_FIX  # 保持原状态，记录拒绝
            return self._persist(issue)
        self._transition(issue, IssueState.EXTENSION_PENDING, requested_by)
        self.queue.create(issue_id, "extension_approval", IssueState.EXTENSION_PENDING,
                          {"hours": hours, "policy": msg}, assignee_role="质量授权人")
        return self._persist(issue)

    def approve_extension(self, issue_id: str, approved: bool, approver: str) -> IssueStateModel:
        issue = self.repo.load_issue(issue_id)
        assert issue and issue.order
        self._audit(issue, approver, "human", "extension_decision", after={"approved": approved})
        self._transition(issue, IssueState.PENDING_FIX, approver)
        if approved:
            issue.order.deadline = issue.order.deadline + timedelta(hours=24)
            issue.deadline = issue.order.deadline
        return self._persist(issue)

    # ==================================================================
    # 时钟 tick
    # ==================================================================
    def tick(self, issue_id: str, now: Optional[datetime] = None) -> IssueStateModel:
        issue = self.repo.load_issue(issue_id)
        assert issue
        actions = self.clock.tick(issue, now)
        for a in actions:
            self._audit(issue, "clock", "system", "escalate", after=a)
        return self._persist(issue)

    # ==================================================================
    # 统计
    # ==================================================================
    def stats(self) -> dict[str, Any]:
        issues = self.repo.list_issues(limit=1000)
        total = len(issues)
        auto = sum(1 for i in issues
                   if i.state == IssueState.CLOSED and not (i.grade and i.grade.need_human))
        return {
            "total": total,
            "by_state": self.repo.count_by_state(),
            "auto_flow_rate": round(auto / total, 3) if total else 0.0,
            "invalid_fix_total": sum(i.invalid_fix_count for i in issues),
            "reject_total": sum(i.reject_count for i in issues),
            "by_department": DepartmentStats.summarize(issues),
            "pending_human_tasks": len(self.queue.pending()),
        }


_default_orchestrator: Optional[InspectionOrchestrator] = None


def get_default_orchestrator() -> InspectionOrchestrator:
    global _default_orchestrator
    if _default_orchestrator is None:
        _default_orchestrator = InspectionOrchestrator()
    return _default_orchestrator


def build_graph_description() -> dict[str, Any]:
    """导出状态机图描述（与设计稿图1 一致），供前端/文档渲染。"""
    return {
        "states": [s.value for s in IssueState],
        "main_flow": [s.value for s in
                      (IssueState.DRAFT, IssueState.PENDING_MAPPING, IssueState.PENDING_GRADE,
                       IssueState.PENDING_FIX, IssueState.PENDING_VERIFY,
                       IssueState.PENDING_RECHECK, IssueState.CLOSED)],
        "transitions": {k.value: sorted(x.value for x in v) for k, v in ALLOWED_TRANSITIONS.items()},
        "hard_gates": [
            "低置信度条款映射不得自动定级，必须转人工选条款",
            "A级问题和CCP相关问题不得仅凭照片自动关闭，必须人工现场复核",
            "打回整改不重置倒计时；无效整改次数单独累计",
            "所有LLM输出带来源引用或明确标注模型建议，人工修改留痕",
            "未通过验证+复查（A级需现场复核签字）不允许关闭",
        ],
    }
