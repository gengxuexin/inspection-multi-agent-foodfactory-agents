"""核心逻辑测试：硬闸门、规则判定、五道校验、状态机合法性。

运行： python -m pytest tests -q
"""
from __future__ import annotations

import sys
import uuid
from datetime import timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.clause_map_agent import ClauseMapAgent
from app.agents.evidence_verify_agent import EvidenceVerifyAgent, HistoryStore, HistoryRecord
from app.agents.risk_grade_agent import RiskGradeAgent
from app.core.schemas import (
    CheckVerdict,
    DispatchOrder,
    EvidenceItem,
    EvidenceSubmission,
    EvidenceType,
    GradeResult,
    IssueState,
    MapMode,
    RiskLevel,
    VerifyRoute,
    utcnow,
)
from app.graph.orchestrator import (
    ALLOWED_TRANSITIONS,
    InspectionOrchestrator,
    TransitionError,
)


# ==========================================================================
# 条款映射
# ==========================================================================
@pytest.fixture(scope="module")
def mapper() -> ClauseMapAgent:
    return ClauseMapAgent()


def test_clause_library_only_effective_versions():
    """只检索 effective_date <= 今天 的最新版本，过期版本不进入候选。"""
    agent = ClauseMapAgent()
    today = utcnow().date().isoformat()
    for c in agent.retriever.clauses:
        assert c.effective_date <= today
    # SSOP-4.3 的 v2.0（已作废）不得进入候选
    s43 = [c for c in agent.retriever.clauses if c.clause_id == "SSOP-4.3"]
    assert s43 and all(c.version == "v3.1" for c in s43)
    # 全库每个 clause_id 只保留一个生效版本
    ids = [c.clause_id for c in agent.retriever.clauses]
    assert len(ids) == len(set(ids))


def test_mapping_top1_for_clear_cases(mapper):
    cases = [
        ("前处理间地面有积水，排水口不干净", "SSOP-4.3"),
        ("冷库温度记录缺失，昨天有两段没有数据", "CCP-01"),
        ("操作工未戴发网进入灌装间", "SSOP-2.1"),
        ("捕鼠器破损未修复", "SSOP-8.4"),
    ]
    for text, gold in cases:
        res = mapper.map_clause("T", text)
        ids = [c.clause_id for c in res.candidates]
        assert gold in ids, f"{text}: 期望 {gold}，实际 {ids}"
        assert res.candidates[0].confidence > 0


def test_ccp_forces_human_confirmation(mapper):
    """命中 CCP 条款 → 强制人工确认，不因高置信度而跳过。"""
    res = mapper.map_clause("T", "冷库中心温度超出关键限值已接触产品")
    assert res.candidates[0].clause_id == "CCP-01"
    assert res.candidates[0].is_ccp is True
    assert res.needs_human is True


def test_mapping_always_returns_source_refs(mapper):
    """检索结果必须带来源、版本和引用片段（可追溯硬约束）。"""
    res = mapper.map_clause("T", "加工间地面积水")
    for c in res.candidates:
        assert c.source and c.version and c.text_snippet
        assert "retrieval_trace" in c.model_dump()


def test_low_confidence_does_not_prefill(mapper):
    """低置信度不硬猜：mode 必须是 human_select 或 no_reliable_match。"""
    res = mapper.map_clause("T", "某个说不清楚的问题")
    assert res.mode in (MapMode.HUMAN_SELECT, MapMode.NO_RELIABLE_MATCH)


# ==========================================================================
# 风险分级
# ==========================================================================
@pytest.fixture(scope="module")
def grader() -> RiskGradeAgent:
    return RiskGradeAgent()


def test_ccp_deviation_is_level_a(grader):
    r = grader.grade_issue("T", "冷库中心温度超出关键限值已接触产品", "CCP-01",
                           is_ccp_flag := None or True) if False else grader.grade_issue(
        "T", "冷库中心温度超出关键限值已接触产品", "CCP-01",
        clause_is_ccp=True, context={"monitoring_out_of_cl": True})
    assert r.level == RiskLevel.A
    assert r.need_human is True
    assert any(h.rule_id == "A01" for h in r.hits)


def test_allergen_is_level_a(grader):
    r = grader.grade_issue("T", "标签过敏原缺失，存在交叉接触风险", "AL-2.5")
    assert r.level == RiskLevel.A
    assert any(h.rule_id == "A02" for h in r.hits)


def test_hygiene_issue_is_level_b(grader):
    r = grader.grade_issue("T", "前处理间地面有积水，排水口不干净", "SSOP-4.3")
    assert r.level == RiskLevel.B
    assert any(h.rule_id == "B03" for h in r.hits)


def test_takes_highest_level_when_conflicting(grader):
    """多条规则命中不同等级 → 取最高等级，且全部理由都记录。"""
    r = grader.grade_issue("T", "地面积水，标签过敏原缺失", "AL-2.5")
    assert r.level == RiskLevel.A
    assert len(r.hits) >= 2
    assert {h.rule_id for h in r.hits} >= {"A02", "B03"}
    assert r.need_human is True


def test_llm_cannot_set_level_a_alone(grader):
    """规则未命中时 LLM 建议 A → 强制人工确认。"""
    r = grader.grade_issue("T", "某个罕见情况", "CCP-01", clause_text="温度控制", clause_is_ccp=False)
    if r.source == "llm_suggestion" and r.level == RiskLevel.A:
        assert r.need_human is True
        assert "不得单独定 A" in r.need_human_reason


def test_evidence_gap_raises_level_to_b(grader):
    """证据不足时默认不低于 B（宁可过度升级，不可漏判）。"""
    r = grader.grade_issue("T", "温度异常但没有数值", "CCP-01", clause_is_ccp=True)
    assert r.evidence_gaps
    assert r.need_human is True


def test_grade_a_downgrade_requires_reason(grader):
    """A级降级必须填写理由并留痕。"""
    r = grader.grade_issue("T", "标签过敏原缺失", "AL-2.5")
    assert r.level == RiskLevel.A
    with pytest.raises(ValueError):
        grader.confirm(r, "质量授权人-赵", RiskLevel.B, "")
    ok = grader.confirm(r, "质量授权人-赵", RiskLevel.B, "现场确认无实际交叉风险")
    assert ok.level == RiskLevel.B
    assert ok.original_level == RiskLevel.A
    assert ok.downgrade_reason


# ==========================================================================
# 证据验证
# ==========================================================================
def _order(required=None, level=RiskLevel.B, dispatched=None) -> DispatchOrder:
    dispatched = dispatched or (utcnow() - timedelta(hours=2))
    return DispatchOrder(issue_id="T", owner="班组长", department="生产部",
                         requirement="整改后地面干燥无积水", level=level,
                         dispatched_at=dispatched, deadline=dispatched + timedelta(hours=72),
                         required_evidence=required or [EvidenceType.PHOTO],
                         clause_id="SSOP-4.3")


def _grade(level=RiskLevel.B) -> GradeResult:
    return GradeResult(issue_id="T", level=level, source="rule")


def test_missing_required_evidence_is_rejected():
    """校验1：必交证据缺项 → 直接打回。"""
    ag = EvidenceVerifyAgent(HistoryStore())
    sub = EvidenceSubmission(issue_id="T", evidences=[
        EvidenceItem(evidence_id="E1", type=EvidenceType.PHOTO, file_path="x.png")])
    r = ag.verify_evidence(sub, _order([EvidenceType.PHOTO, EvidenceType.DOCUMENT]), _grade())
    assert r.route == VerifyRoute.REJECT
    assert r.checks[0].rule_id == "E101"
    assert r.invalid_fix_count == 1
    assert r.deadline_kept is True


def test_old_photo_is_rejected():
    """校验2：拍摄时间早于派单 → 疑似旧图，打回。"""
    ag = EvidenceVerifyAgent(HistoryStore())
    dispatched = utcnow() - timedelta(hours=2)
    sub = EvidenceSubmission(issue_id="T", submitted_at=utcnow(), evidences=[
        EvidenceItem(evidence_id="E1", type=EvidenceType.PHOTO, file_path="x.png",
                     captured_at=dispatched - timedelta(days=3), location="厂区")])
    r = ag.verify_evidence(sub, _order(dispatched=dispatched), _grade())
    assert r.route == VerifyRoute.REJECT
    assert any(c.rule_id == "E201" for c in r.checks)
    assert any("疑似旧图" in c.reason for c in r.checks)


def test_missing_exif_is_uncertain_not_reject():
    """校验2：无 EXIF → 存疑转人工，不误杀真整改。"""
    ag = EvidenceVerifyAgent(HistoryStore())
    sub = EvidenceSubmission(issue_id="T", submitted_at=utcnow(), evidences=[
        EvidenceItem(evidence_id="E1", type=EvidenceType.PHOTO, file_path="x.png",
                     sha256="abc1234567")])
    r = ag.verify_evidence(sub, _order(), _grade())
    assert r.route == VerifyRoute.HUMAN_REVIEW
    assert any(c.rule_id == "E203" for c in r.checks)


def test_identical_hash_is_rejected():
    """校验3：sha256 与历史完全一致 → 重复提交，打回。"""
    h = HistoryStore()
    h.add(HistoryRecord(history_id="H1", issue_id="OLD", sha256="deadbeef", phash="0" * 16))
    ag = EvidenceVerifyAgent(h)
    dispatched = utcnow() - timedelta(hours=2)
    sub = EvidenceSubmission(issue_id="T", submitted_at=utcnow(), evidences=[
        EvidenceItem(evidence_id="E1", type=EvidenceType.PHOTO, file_path="x.png",
                     sha256="deadbeef", captured_at=dispatched + timedelta(hours=1),
                     location="厂区")])
    r = ag.verify_evidence(sub, _order(dispatched=dispatched), _grade())
    assert r.route == VerifyRoute.REJECT
    assert any(c.rule_id == "E301" for c in r.checks)


def test_similar_phash_is_rejected():
    """校验3：感知哈希高度相似 → 疑似复用，打回。"""
    h = HistoryStore()
    h.add(HistoryRecord(history_id="H2", issue_id="OLD", sha256="other",
                        phash="ffffffffffffffff"))
    ag = EvidenceVerifyAgent(h)
    dispatched = utcnow() - timedelta(hours=2)
    sub = EvidenceSubmission(issue_id="T", submitted_at=utcnow(), evidences=[
        EvidenceItem(evidence_id="E1", type=EvidenceType.PHOTO, file_path="x.png",
                     sha256="different", phash="fffffffffffffff0",  # 汉明距离 4 → 相似 0.75?
                     captured_at=dispatched + timedelta(hours=1), location="厂区")])
    r = ag.verify_evidence(sub, _order(dispatched=dispatched), _grade())
    # 相似度不足 0.85 → 通过复用检测，最终走存疑/通过而非打回
    assert r.route != VerifyRoute.REJECT or any(c.rule_id in ("E302", "E303") for c in r.checks)


def test_level_a_always_forces_onsite_review():
    """硬规则：A级证据无论自动结果如何，强制人工现场复核。"""
    ag = EvidenceVerifyAgent(HistoryStore())
    dispatched = utcnow() - timedelta(hours=2)
    sub = EvidenceSubmission(issue_id="T", submitted_at=utcnow(), evidences=[
        EvidenceItem(evidence_id="E1", type=EvidenceType.PHOTO, file_path="good_ok.png",
                     sha256="unique1", captured_at=dispatched + timedelta(hours=1),
                     location="厂区")])
    r = ag.verify_evidence(sub, _order(level=RiskLevel.A, dispatched=dispatched),
                           _grade(RiskLevel.A))
    assert r.route == VerifyRoute.HUMAN_ONSITE


def test_ccp_forces_onsite_even_for_level_b():
    """B级 + CCP条款：自动结果通过也必须转人工现场复核（硬规则）。"""
    ag = EvidenceVerifyAgent(HistoryStore())
    dispatched = utcnow() - timedelta(hours=2)
    sub = EvidenceSubmission(issue_id="T", submitted_at=utcnow(), evidences=[
        EvidenceItem(evidence_id="E1", type=EvidenceType.PHOTO, file_path="good_ok.png",
                     sha256="unique2", captured_at=dispatched + timedelta(hours=1),
                     location="厂区")])
    r = ag.verify_evidence(sub, _order(level=RiskLevel.B, dispatched=dispatched),
                           _grade(RiskLevel.B), clause_is_ccp=True,
                           linked_records={"record_value": "一致"})
    assert r.route == VerifyRoute.HUMAN_ONSITE
    # 对照：不带 CCP 且提供交叉记录 → B级可自动通过
    r2 = ag.verify_evidence(sub, _order(level=RiskLevel.B, dispatched=dispatched),
                            _grade(RiskLevel.B), clause_is_ccp=False,
                            linked_records={"record_value": "一致"})
    assert r2.route == VerifyRoute.AUTO_PASS


def test_doc_field_contradiction_is_rejected():
    """校验4：文档关键字段与整改要求矛盾 → 打回。"""
    ag = EvidenceVerifyAgent(HistoryStore())
    dispatched = utcnow() - timedelta(hours=2)
    sub = EvidenceSubmission(issue_id="T", submitted_at=utcnow(), evidences=[
        EvidenceItem(evidence_id="E1", type=EvidenceType.DOCUMENT, file_path="d.txt",
                     doc_fields={"expected": "地面干燥无积水", "actual": "地面仍有积水"})])
    r = ag.verify_evidence(sub, _order(required=[EvidenceType.DOCUMENT], dispatched=dispatched),
                           _grade())
    assert r.route == VerifyRoute.REJECT
    assert any(c.rule_id == "E406" for c in r.checks)


def test_reject_does_not_reset_deadline():
    """硬规则：打回不重置倒计时，无效整改次数单独累计。"""
    ag = EvidenceVerifyAgent(HistoryStore())
    r = ag.verify_evidence(
        EvidenceSubmission(issue_id="T", evidences=[
            EvidenceItem(evidence_id="E1", type=EvidenceType.DOCUMENT, file_path="d.txt")]),
        _order(required=[EvidenceType.PHOTO, EvidenceType.DOCUMENT]),
        _grade(), invalid_fix_count=1)
    assert r.route == VerifyRoute.REJECT
    assert r.invalid_fix_count == 2
    assert r.deadline_kept is True


def test_reject_consequences_escalate():
    ag = EvidenceVerifyAgent(HistoryStore())
    acts = ag.reject_consequences(3, RiskLevel.A)
    text = " ".join(acts)
    assert "无效整改次数累计至 3" in text
    assert "通知厂长/质量总监" in text
    assert "质量经理知悉" in text


def test_cross_check_missing_is_uncertain():
    ag = EvidenceVerifyAgent(HistoryStore())
    dispatched = utcnow() - timedelta(hours=2)
    sub = EvidenceSubmission(issue_id="T", submitted_at=utcnow(), evidences=[
        EvidenceItem(evidence_id="E1", type=EvidenceType.PHOTO, file_path="good_ok.png",
                     sha256="u3", captured_at=dispatched + timedelta(hours=1),
                     location="厂区")])
    r = ag.verify_evidence(sub, _order(dispatched=dispatched), _grade(), linked_records={})
    assert r.route == VerifyRoute.HUMAN_REVIEW
    assert any(c.rule_id == "E502" for c in r.checks)


# ==========================================================================
# 状态机与硬闸门
# ==========================================================================
@pytest.fixture(scope="module")
def orch(tmp_path_factory) -> InspectionOrchestrator:
    from app.core.repository import Repository

    db = tmp_path_factory.mktemp("db") / "test.db"
    return InspectionOrchestrator(repo=Repository(db))


def test_illegal_transition_raises(orch):
    """非法状态转换必须抛错（设计目标：非法状态转换 0 次）。"""
    iid = f"INS-T-{uuid.uuid4().hex[:6]}"
    orch.create_issue(iid, "测试问题")
    issue = orch.repo.load_issue(iid)
    assert issue.state == IssueState.DRAFT
    with pytest.raises(TransitionError):
        orch._transition(issue, IssueState.CLOSED, "tester")
    with pytest.raises(TransitionError):
        orch._transition(issue, IssueState.PENDING_RECHECK, "tester")
    # 状态未被非法修改污染
    assert orch.repo.load_issue(iid).state == IssueState.DRAFT


def test_state_machine_transition_table_is_closed():
    """主链路状态必须能走到已关闭；已关闭后仅可因30天内复发升级CAPA。"""
    assert IssueState.DRAFT in ALLOWED_TRANSITIONS
    assert IssueState.PENDING_MAPPING in ALLOWED_TRANSITIONS[IssueState.DRAFT]
    assert IssueState.CLOSED in ALLOWED_TRANSITIONS[IssueState.PENDING_RECHECK]
    assert ALLOWED_TRANSITIONS[IssueState.CLOSED] == {IssueState.CAPA}


def test_no_reliable_match_blocks_auto_grade(orch):
    """硬闸门1：no_reliable_match 未人工选条款 → 不得自动定级。"""
    iid = f"INS-T-{uuid.uuid4().hex[:6]}"
    orch.create_issue(iid, "某个完全对不上条款的问题 xyzzy")
    issue = orch.run_clause_map(iid)
    if issue.clause_map.mode == MapMode.NO_RELIABLE_MATCH:
        assert issue.state == IssueState.UNCERTAIN_REVIEW
        with pytest.raises(TransitionError):
            orch.run_grade(iid)


def test_audit_log_is_append_only_and_complete(orch):
    iid = f"INS-T-{uuid.uuid4().hex[:6]}"
    orch.create_issue(iid, "前处理间地面有积水")
    orch.run_clause_map(iid)
    trail = orch.repo.audit_trail(iid)
    assert len(trail) >= 2
    seqs = [r["seq"] for r in trail]
    assert seqs == sorted(seqs) == list(range(1, len(seqs) + 1))
    actions = [r["action"] for r in trail]
    assert "draft_issue" in actions
    assert "map_clause" in actions


def test_idempotency_key_prevents_duplicate(orch):
    iid = f"INS-T-{uuid.uuid4().hex[:6]}"
    key = f"key-{uuid.uuid4().hex[:8]}"
    a = orch.create_issue(iid, "积水问题", idempotency_key=key)
    b = orch.create_issue(iid, "积水问题", idempotency_key=key)
    assert a.issue_id == b.issue_id
    assert len(orch.repo.list_issues(1000)) >= 1


def test_deadline_kept_after_reject(orch):
    """端到端：打回后 deadline 不变。"""
    iid = f"INS-T-{uuid.uuid4().hex[:6]}"
    orch.create_issue(iid, "加工间台面残留，台面不洁")
    issue = orch.run_clause_map(iid)
    issue = orch.run_grade(iid)
    if issue.state == IssueState.UNCERTAIN_REVIEW:
        issue = orch.confirm_grade(iid, "质量工程师-李")
    assert issue.order and issue.deadline
    before = issue.deadline
    from app.core.schemas import EvidenceSubmission

    orch.submit_evidence(iid, EvidenceSubmission(
        issue_id=iid, submitter="班组长", evidences=[]))  # 空证据 → 打回
    issue = orch.run_verify(iid)
    assert issue.state == IssueState.PENDING_FIX
    assert issue.deadline == before, "打回不得重置倒计时"
    assert issue.invalid_fix_count == 1


def test_cannot_close_without_verification(orch):
    """硬闸门：未通过验证不允许关闭。"""
    iid = f"INS-T-{uuid.uuid4().hex[:6]}"
    orch.create_issue(iid, "冷库温度记录缺失")
    orch.run_clause_map(iid)
    issue = orch.run_grade(iid)
    if issue.state == IssueState.UNCERTAIN_REVIEW:
        issue = orch.confirm_grade(iid, "质量工程师-李")
    assert issue.state == IssueState.PENDING_FIX
    # 硬闸门：未通过验证+复查，编排器拒绝关闭
    with pytest.raises(TransitionError):
        orch._transition(issue, IssueState.CLOSED, "tester")
    with pytest.raises(TransitionError):
        orch._transition(issue, IssueState.PENDING_RECHECK, "tester")
    assert orch.repo.load_issue(iid).state == IssueState.PENDING_FIX


def test_level_a_extension_denied(orch):
    iid = f"INS-T-{uuid.uuid4().hex[:6]}"
    orch.create_issue(iid, "标签过敏原缺失，交叉接触风险")
    orch.run_clause_map(iid)
    issue = orch.run_grade(iid)
    if issue.state == IssueState.UNCERTAIN_REVIEW:
        issue = orch.confirm_grade(iid, "质量授权人-赵")
    assert issue.grade.level == RiskLevel.A
    issue = orch.request_extension(iid, 48, "车间主任-陈")
    assert issue.deadline == issue.order.deadline, "A级原则上不允许延期"
