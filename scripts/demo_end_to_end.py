"""端到端演示：跑通三个 Agent 的完整闭环。

演示四个场景，覆盖设计稿中的核心闸门：
  场景1：普通卫生问题（地面积水）→ 条款映射预填 → B级自动派单 → 证据通过 → 复查关闭
  场景2：A级过敏原问题 → 条款映射人工选 → A级强制人工确认 → 证据强制现场复核 → 签字关闭
  场景3：假整改（旧图复用）→ 五道校验打回 → 倒计时不重置 → 无效整改+1 → 重新提交通过
  场景4：低置信度条款映射 → 显式弃答 → 人工选条款后才允许分级

运行： python scripts/demo_end_to_end.py
"""
from __future__ import annotations

import sys
import uuid
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.schemas import (  # noqa: E402
    EvidenceItem,
    EvidenceSubmission,
    EvidenceType,
    IssueState,
    MapMode,
    RiskLevel,
    utcnow,
)
from app.graph.orchestrator import InspectionOrchestrator  # noqa: E402
from app.rag.fingerprint import sha256_file  # noqa: E402
from app.core.config import UPLOAD_DIR  # noqa: E402

UPLOAD = UPLOAD_DIR / "demo"
UPLOAD.mkdir(parents=True, exist_ok=True)


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def dump_issue(issue, show_audit: bool = False, orch=None) -> None:
    print(f"\n  状态：{issue.state.value}    版本：v{issue.version}")
    if issue.clause_map:
        cm = issue.clause_map
        print(f"  条款映射：mode={cm.mode.value}  needs_human={cm.needs_human}")
        for c in cm.candidates:
            ccp = " [CCP]" if c.is_ccp else ""
            print(f"      - {c.clause_id}{ccp} ({c.source} {c.version}) conf={c.confidence}  {c.reason}")
        print(f"      分流原因：{cm.needs_human_reason}")
    if issue.grade:
        g = issue.grade
        print(f"  风险分级：{g.level.value} 级   来源={g.source}   需人工={g.need_human}")
        for h in g.hits:
            print(f"      - [{h.rule_id}] {h.reason}")
        if g.llm_reason:
            print(f"      - LLM建议：{g.llm_reason}")
        if g.need_human_reason:
            print(f"      人工确认原因：{g.need_human_reason}")
        if g.confirmed_by:
            print(f"      已由 {g.confirmed_by} 确认（原级别={g.original_level or g.level.value}）")
    if issue.order:
        o = issue.order
        print(f"  派单：{o.owner}/{o.department}  时限={o.deadline.isoformat()}")
        print(f"      必交证据：{[t.value for t in o.required_evidence]}")
    if issue.verify:
        v = issue.verify
        print(f"  证据验证：route={v.route.value}  无效整改次数={v.invalid_fix_count}  "
              f"倒计时保持={v.deadline_kept}  VLM={v.vlm_used}")
        for c in v.checks:
            mark = {"通过": "✓", "打回": "✗", "存疑": "?"}[c.verdict.value]
            print(f"      [{mark}] {c.check:<12} {c.rule_id:<5} {c.reason}")
        if v.need_human_reason:
            print(f"      分流原因：{v.need_human_reason}")
    if issue.recheck:
        print(f"  复查：抽验={issue.recheck.sampled} 通过={issue.recheck.passed} "
              f"方式={issue.recheck.method}")
    if issue.escalations:
        for e in issue.escalations:
            print(f"  升级：L{e['level']} → {e['to']}  {e['note']}")
    if issue.capa_id:
        print(f"  CAPA：{issue.capa_id}")
    if show_audit and orch:
        print("\n  审计轨迹：")
        for r in orch.repo.audit_trail(issue.issue_id):
            print(f"      #{r['seq']:<2} {r['actor']:<22} {r['action']:<22} "
                  f"{r['from_state']}→{r['to_state']}")


def _mk_file(name: str, payload) -> str:
    p = UPLOAD / name
    if isinstance(payload, str):
        p.write_text(payload, encoding="utf-8")
    else:
        p.write_bytes(payload)
    return str(p)


def _drive_to_fix(orch, iid: str, tag: str = "") -> IssueStateModel:
    """把问题单推进到"待整改"，遇到人工闸门就按指定角色确认。

    演示脚本用：真实业务里这些确认由人在前端完成。
    """
    issue = orch.repo.load_issue(iid)
    for _ in range(8):
        if issue.state == IssueState.PENDING_FIX:
            return issue
        if issue.state == IssueState.DRAFT:
            issue = orch.run_clause_map(iid)
        elif issue.state == IssueState.UNCERTAIN_REVIEW:
            if issue.clause_map and not issue.selected_clause_id:
                cid = next((c.clause_id for c in issue.clause_map.candidates), "SSOP-4.3")
                issue = orch.confirm_clause(iid, cid, "质量工程师-李")
                issue = orch.run_grade(iid)
            elif issue.grade and issue.grade.need_human:
                issue = orch.confirm_grade(iid, "质量授权人-赵")
            else:
                break
        elif issue.state == IssueState.PENDING_GRADE:
            issue = orch.run_grade(iid)
        else:
            break
        issue = orch.repo.load_issue(iid)
    return issue


# ==========================================================================
def scenario_1(orch: InspectionOrchestrator) -> None:
    hr("场景1：B级卫生问题 —— 自动预填条款 → 自动分级 → 自动派单 → 自动通过 → 复查关闭")
    iid = f"INS-DEMO-{uuid.uuid4().hex[:6]}"
    issue = orch.create_issue(iid, "前处理间地面有积水，排水口不干净", reporter="张巡检")
    print(f"  录入：{issue.draft.phenomenon}")
    print(f"      区域={issue.draft.zone} 工序={issue.draft.process} "
          f"类别={issue.draft.initial_category}")
    print(f"      建议证据={[t.value for t in issue.draft.suggested_evidence_types]}")

    issue = orch.run_clause_map(iid)
    issue = orch.run_grade(iid)
    dump_issue(issue, show_audit=True, orch=orch)

    # 人工确认条款（人工点：条款确认）。prefill 模式允许人工改，这里演示改选。
    if issue.clause_map.mode != MapMode.PREFILL or not issue.clause_map.selected_clause_id:
        clause = issue.clause_map.selected_clause_id or issue.clause_map.candidates[0].clause_id
        issue = orch.confirm_clause(iid, clause, "质量工程师-李")
    else:
        print("  （prefill 模式已自动预填条款 SSOP-4.3，巡检员可直接修改）")

    dispatched = issue.order.dispatched_at
    img = _mk_file("s1_fixed_ok.png", b"S1-GENUINE-CLEAN-FLOOR" * 8)
    doc = _mk_file("s1_record.txt", "清洁消毒记录：2025-xx 已完成")
    sub = EvidenceSubmission(
        issue_id=iid, submitted_at=dispatched + timedelta(hours=3), submitter="班组长-王",
        evidences=[
            EvidenceItem(evidence_id="EV-1", type=EvidenceType.PHOTO, file_path=img,
                         file_name="s1_fixed_ok.png", sha256=sha256_file(img),
                         captured_at=dispatched + timedelta(hours=2), location="厂区-前处理间"),
            EvidenceItem(evidence_id="EV-2", type=EvidenceType.DOCUMENT, file_path=doc,
                         file_name="s1_record.txt",
                         doc_fields={"date": (dispatched + timedelta(hours=2)).isoformat(),
                                     "expected": "地面干燥无积水", "actual": "地面干燥无积水",
                                     "text": "清洁消毒记录"}),
        ],
    )
    orch.submit_evidence(iid, sub)
    issue = orch.run_verify(iid, linked_records={"record_value": "地面干燥无积水",
                                                 "evidence_claim": "地面干燥无积水"})
    print("\n  证据提交后：")
    dump_issue(issue)

    if issue.state == IssueState.PENDING_RECHECK:
        issue = orch.run_recheck(iid, passed=True, reviewer="质量授权人-赵", sampled=True,
                                 note="现场复查通过", rng_value=0.1)
    print("\n  关闭后：")
    dump_issue(issue)


# ==========================================================================
def scenario_2(orch: InspectionOrchestrator) -> None:
    hr("场景2：A级过敏原问题 —— 强制人工确认分级 → A级证据强制现场复核 → 签字关闭")
    iid = f"INS-DEMO-{uuid.uuid4().hex[:6]}"
    orch.create_issue(iid, "灌装间标签过敏原信息缺失，换线后未做交叉清洗", reporter="张巡检")
    issue = orch.run_clause_map(iid)
    dump_issue(issue, orch=orch)

    clause = next((c.clause_id for c in issue.clause_map.candidates
                   if c.hazard_type == "过敏原"), "AL-2.5")
    if not issue.clause_map.selected_clause_id:
        issue = orch.confirm_clause(iid, clause, "质量工程师-李")
    issue = orch.run_grade(iid)
    dump_issue(issue, orch=orch)

    if issue.state == IssueState.UNCERTAIN_REVIEW and issue.grade.need_human:
        issue = orch.confirm_grade(iid, "质量授权人-赵")
    print("\n  人工确认分级后：")
    dump_issue(issue)

    dispatched = issue.order.dispatched_at
    img = _mk_file("s2_containment_ok.png", b"S2-CONTAINMENT-OK" * 8)
    doc1 = _mk_file("s2_label.txt", "标签更正后照片说明")
    doc2 = _mk_file("s2_rca.txt", "根因分析与CAPA草案")
    sub = EvidenceSubmission(
        issue_id=iid, submitted_at=dispatched + timedelta(hours=5), submitter="车间主任-陈",
        evidences=[
            EvidenceItem(evidence_id="EV-1", type=EvidenceType.PHOTO, file_path=img,
                         file_name="s2_containment_ok.png", sha256=sha256_file(img),
                         captured_at=dispatched + timedelta(hours=4), location="厂区-灌装间"),
            EvidenceItem(evidence_id="EV-2", type=EvidenceType.DOCUMENT, file_path=doc1,
                         file_name="s2_label.txt",
                         doc_fields={"date": (dispatched + timedelta(hours=4)).isoformat(),
                                     "expected": "标签含过敏原标示", "actual": "标签含过敏原标示",
                                     "text": "标签更正"}),
            EvidenceItem(evidence_id="EV-3", type=EvidenceType.DOCUMENT, file_path=doc2,
                         file_name="s2_rca.txt",
                         doc_fields={"date": (dispatched + timedelta(hours=4)).isoformat(),
                                     "expected": "根因与CAPA", "actual": "根因与CAPA",
                                     "text": "RCA"}),
            # A级必交项追加：交叉校验用的系统记录
            EvidenceItem(evidence_id="EV-4", type=EvidenceType.SYSTEM_RECORD,
                         file_path=_mk_file("s2_mes.log", "MES放行记录：批次已隔离"),
                         file_name="s2_mes.log",
                         doc_fields={"timestamp": (dispatched + timedelta(hours=4)).isoformat(),
                                     "record_value": "批次已隔离", "text": "MES放行记录"}),
        ],
    )
    orch.submit_evidence(iid, sub)
    issue = orch.run_verify(iid, linked_records={"record_value": "标签已更正",
                                                 "evidence_claim": "标签已更正"})
    print("\n  A级证据验证结果（注意：无论自动结果如何都转现场复核）：")
    dump_issue(issue)

    if issue.state == IssueState.UNCERTAIN_REVIEW:
        issue = orch.review_evidence(iid, passed=True, reviewer="质量授权人-赵",
                                      note="现场复核签字通过")
        issue = orch.run_recheck(iid, passed=True, reviewer="质量授权人-赵", sampled=True,
                                 note="A级100%现场复查通过", rng_value=0.0)
    print("\n  关闭后：")
    dump_issue(issue)


# ==========================================================================
def scenario_3(orch: InspectionOrchestrator) -> None:
    hr("场景3：假整改（旧图复用）—— 五道校验拦截 → 打回 → 倒计时不重置 → 重新提交通过")
    iid = f"INS-DEMO-{uuid.uuid4().hex[:6]}"
    orch.create_issue(iid, "加工间台面残留，台面不洁", reporter="李巡检")
    issue = _drive_to_fix(orch, iid)
    assert issue.order and issue.grade, f"未推进到待整改（{issue.state.value}）"
    print(f"  分级：{issue.grade.level.value} 级   截止时间：{issue.deadline.isoformat()}")

    # 先用一张合规照片关闭另一个问题单，制造"历史证据"
    seed = f"INS-DEMO-SEED-{uuid.uuid4().hex[:6]}"
    orch.create_issue(seed, "前处理间地面清洁记录", reporter="李巡检")
    s = _drive_to_fix(orch, seed, tag="seed")
    assert s.order is not None, f"种子单未派单（{s.state.value}）"
    hist_img = _mk_file("s3_history_ok.png", b"S3-HISTORY-CLEAN" * 8)
    hist_dispatched = s.order.dispatched_at
    orch.submit_evidence(seed, EvidenceSubmission(
        issue_id=seed, submitted_at=hist_dispatched + timedelta(hours=2), submitter="班组长-王",
        evidences=[
            EvidenceItem(evidence_id="EV-1", type=EvidenceType.PHOTO, file_path=hist_img,
                         file_name="s3_history_ok.png", sha256=sha256_file(hist_img),
                         captured_at=hist_dispatched + timedelta(hours=1),
                         location="厂区-加工间"),
            EvidenceItem(evidence_id="EV-2", type=EvidenceType.DOCUMENT,
                         file_path=_mk_file("s3_seed_record.txt", "历史清洁记录"),
                         file_name="s3_seed_record.txt",
                         doc_fields={"date": (hist_dispatched + timedelta(hours=1)).isoformat(),
                                     "expected": "台面清洁", "actual": "台面清洁",
                                     "text": "清洁记录"}),
        ]))
    s = orch.run_verify(seed, linked_records={"record_value": "台面清洁", "evidence_claim": "台面清洁"})
    if s.state == IssueState.PENDING_RECHECK:
        s = orch.run_recheck(seed, passed=True, reviewer="质量工程师-李", sampled=True,
                             note="历史问题单复查通过", rng_value=0.0)
    assert s.state in (IssueState.CLOSED, IssueState.CAPA), \
        f"历史种子单应关闭或升级CAPA，实际 {s.state.value}"
    if s.capa_id:
        print(f"  历史问题单 {seed} 关闭后触发 CAPA：{s.capa_id}"
              f"（30天内同区域同条款复发，触发 HACCP 计划/SSOP 评审提醒）")
    else:
        print(f"  历史问题单 {seed} 已关闭（{s.state.value}）")
    print(f"  历史证据库条目数：{len(orch.history.records)}")

    # 场景3：拿同一张照片重复提交
    dispatched = issue.order.dispatched_at
    deadline_before = issue.deadline
    reuse_doc = _mk_file("s3_reuse_record.txt", "本次清洁记录")
    orch.submit_evidence(iid, EvidenceSubmission(
        issue_id=iid, submitted_at=dispatched + timedelta(hours=3), submitter="班组长-王",
        evidences=[
            EvidenceItem(evidence_id="EV-1", type=EvidenceType.PHOTO, file_path=hist_img,
                         file_name="s3_history_ok.png", sha256=sha256_file(hist_img),
                         captured_at=dispatched + timedelta(hours=2), location="厂区-加工间"),
            EvidenceItem(evidence_id="EV-2", type=EvidenceType.DOCUMENT, file_path=reuse_doc,
                         file_name="s3_reuse_record.txt",
                         doc_fields={"date": (dispatched + timedelta(hours=2)).isoformat(),
                                     "expected": "台面清洁", "actual": "台面清洁",
                                     "text": "本次清洁记录"}),
        ]))
    issue = orch.run_verify(iid, linked_records={"record_value": "台面清洁",
                                                 "evidence_claim": "台面清洁"})
    print("\n  旧图复用被打回：")
    dump_issue(issue)
    print(f"\n  截止时间是否被重置：{'是（错误！）' if issue.deadline != deadline_before else '否（正确，保持不变）'}"
          f"  {deadline_before.isoformat()} → {issue.deadline.isoformat() if issue.deadline else 'None'}")

    # 重新提交真实证据
    new_img = _mk_file("s3_new_ok.png", b"S3-NEW-CLEAN-FRESH" * 8)
    new_doc = _mk_file("s3_new_record.txt", "本次整改清洁记录")
    orch.submit_evidence(iid, EvidenceSubmission(
        issue_id=iid, submitted_at=dispatched + timedelta(hours=5), submitter="班组长-王",
        evidences=[
            EvidenceItem(evidence_id="EV-3", type=EvidenceType.PHOTO, file_path=new_img,
                         file_name="s3_new_ok.png", sha256=sha256_file(new_img),
                         captured_at=dispatched + timedelta(hours=4), location="厂区-加工间"),
            EvidenceItem(evidence_id="EV-4", type=EvidenceType.DOCUMENT, file_path=new_doc,
                         file_name="s3_new_record.txt",
                         doc_fields={"date": (dispatched + timedelta(hours=4)).isoformat(),
                                     "expected": "台面清洁", "actual": "台面清洁",
                                     "text": "本次整改清洁记录"}),
        ]))
    issue = orch.run_verify(iid, linked_records={"record_value": "台面清洁",
                                                 "evidence_claim": "台面清洁"})
    if issue.state == IssueState.PENDING_RECHECK:
        issue = orch.run_recheck(iid, passed=True, reviewer="质量工程师-李", sampled=True,
                                 note="复查通过", rng_value=0.2)
    print("\n  重新提交后：")
    dump_issue(issue)


# ==========================================================================
def scenario_4(orch: InspectionOrchestrator) -> None:
    hr("场景4：条款库无对应项 → 显式弃答 → 必须人工选条款才能分级（低置信度不硬猜）")
    iid = f"INS-DEMO-{uuid.uuid4().hex[:6]}"
    orch.create_issue(iid, "厂区绿化带浇灌系统滴灌头大量漏水导致积水", reporter="王巡检")
    issue = orch.run_clause_map(iid)
    print(f"  映射结果 mode={issue.clause_map.mode.value}")
    print(f"  分流原因：{issue.clause_map.needs_human_reason}")
    print(f"  当前状态：{issue.state.value}（未进入自动分级）")
    print("  硬闸门验证 —— 尝试在未人工选条款前直接分级：")
    try:
        orch.run_grade(iid)
        print("      ！异常：不应允许自动分级")
    except Exception as e:
        print(f"      ✓ 被拦截：{e}")

    print("\n  人工从候选中选择条款并继续：")
    if issue.clause_map.candidates:
        chosen = issue.clause_map.candidates[0].clause_id
    else:
        chosen = "SSOP-11.2"
    issue = orch.confirm_clause(iid, chosen, "质量工程师-李")
    issue = orch.run_grade(iid)
    dump_issue(issue)
    print(f"\n  人工选条款结果已回流映射样本库（共 "
          f"{len(orch.repo.mapping_samples(100))} 条），用于离线调阈值，不在线训练。")


def main() -> None:
    orch = InspectionOrchestrator()
    print("\n巡检整改多Agent闭环系统 —— 端到端演示")
    print(f"条款库规模：{len(orch.library.clauses)} 条（生效版本 {len(orch.library.latest_by_id())} 条）")
    problems = orch.library.validate()
    print(f"条款库元数据校验：{'通过' if not problems else problems}")

    scenario_1(orch)
    scenario_2(orch)
    scenario_3(orch)
    scenario_4(orch)

    hr("系统看板")
    import json

    print(json.dumps(orch.stats(), ensure_ascii=False, indent=2))
    print(f"\n待人工处理任务：{len(orch.queue.pending())} 项")
    for t in orch.queue.pending()[:8]:
        print(f"    [{t['kind']}] {t['issue_id']} → {t['assignee_role']}  "
              f"截止 {t['due_at']}")


if __name__ == "__main__":
    main()
