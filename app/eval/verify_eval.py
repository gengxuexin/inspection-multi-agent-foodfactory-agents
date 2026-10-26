"""证据验证离线评测：构造四类样本，评估打回准确率与误杀率。

四类样本（设计稿第七节，每类 ≥30 条）：
    真整改 / 假整改（内容不符）/ 旧证据复用 / 证据不全

构造方式：
- 真整改：合规照片（文件名含 _ok）+ 正确元数据 + 齐全证据 → 期望自动通过
- 假整改：照片内容不符（_bad）→ 期望打回
- 旧证据复用：sha256/phash 与历史库完全一致或高度相似 → 期望打回
- 证据不全：缺少必交类型 / 文件损坏 → 期望打回
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

from app.agents.evidence_verify_agent import (
    EvidenceVerifyAgent,
    HistoryRecord,
    HistoryStore,
)
from app.core.config import UPLOAD_DIR
from app.core.schemas import (
    DispatchOrder,
    EvidenceItem,
    EvidenceSubmission,
    EvidenceType,
    GradeResult,
    RiskLevel,
    utcnow,
)
from app.eval.evaluate import build_verify_cases, summarize_verify
from app.rag.fingerprint import sha256_file


def _make_image(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha256_file(path)


def _order(level: RiskLevel, required: list[EvidenceType], dispatched_at) -> DispatchOrder:
    return DispatchOrder(
        issue_id="INS-EVAL-EV",
        owner="班组长",
        department="生产部",
        requirement="整改后照片应显示地面干燥无积水",
        level=level,
        dispatched_at=dispatched_at,
        deadline=dispatched_at + timedelta(hours=72),
        required_evidence=required,
        clause_id="SSOP-4.3",
    )


def _grade(level: RiskLevel) -> GradeResult:
    return GradeResult(issue_id="INS-EVAL-EV", level=level, source="rule", need_human=False)


def run_verify_eval(n_each: int = 30, work_dir: Optional[Path] = None) -> dict[str, Any]:
    work_dir = work_dir or (UPLOAD_DIR / "eval")
    work_dir.mkdir(parents=True, exist_ok=True)
    dispatched = utcnow() - timedelta(hours=2)

    required = [EvidenceType.PHOTO, EvidenceType.DOCUMENT]
    order = _order(RiskLevel.B, required, dispatched)
    grade = _grade(RiskLevel.B)

    # 预置历史证据库（用于复用检测）
    history = HistoryStore()
    old_img = work_dir / "old_reused.png"
    old_sha = _make_image(old_img, b"FAKE-PNG-CONTENT-REUSED-IMAGE" * 8)
    old_phash = "ffffffffffffffff"
    history.add(HistoryRecord(history_id="H-OLD-001", issue_id="INS-OLD-001",
                              sha256=old_sha, phash=old_phash,
                              submitted_at=dispatched - timedelta(days=10)))

    agent = EvidenceVerifyAgent(history)
    results: list[dict[str, Any]] = []

    for i, case in enumerate(build_verify_cases(n_each)):
        kind = case.kind
        evs: list[EvidenceItem] = []
        location = "厂区-前处理间"
        captured = dispatched + timedelta(hours=1)

        if kind == "incomplete":
            # 只交照片，缺 document → 校验1打回
            img = work_dir / f"incomplete_{i}.png"
            s = _make_image(img, f"INCOMPLETE-{i}".encode() * 8)
            evs = [EvidenceItem(evidence_id=f"EV-{i}", type=EvidenceType.PHOTO,
                                file_path=str(img), sha256=s, captured_at=captured,
                                location=location)]
        elif kind == "reused":
            # 直接复用历史指纹
            img = work_dir / f"reused_{i}.png"
            _make_image(img, f"REUSED-{i}".encode() * 8)
            evs = [
                EvidenceItem(evidence_id=f"EV-{i}a", type=EvidenceType.PHOTO,
                             file_path=str(img), sha256=old_sha, phash=old_phash,
                             captured_at=captured, location=location),
                EvidenceItem(evidence_id=f"EV-{i}b", type=EvidenceType.DOCUMENT,
                             file_path=str(work_dir / f"reused_{i}.txt"),
                             doc_fields={"date": (dispatched + timedelta(hours=1)).isoformat(),
                                         "text": "清洁记录-已清洁"}),
            ]
        elif kind == "fake":
            img = work_dir / f"fake_bad_{i}.png"
            s = _make_image(img, f"FAKE-{i}".encode() * 8)
            evs = [
                EvidenceItem(evidence_id=f"EV-{i}a", type=EvidenceType.PHOTO,
                             file_path=str(img), sha256=s, captured_at=captured,
                             location=location),
                EvidenceItem(evidence_id=f"EV-{i}b", type=EvidenceType.DOCUMENT,
                             file_path=str(work_dir / f"fake_{i}.txt"),
                             doc_fields={"date": (dispatched + timedelta(hours=1)).isoformat(),
                                         "expected": "地面干燥无积水", "actual": "地面干燥无积水",
                                         "text": "清洁记录"}),
            ]
        else:  # genuine
            img = work_dir / f"genuine_ok_{i}.png"
            s = _make_image(img, f"GENUINE-{i}".encode() * 8)
            evs = [
                EvidenceItem(evidence_id=f"EV-{i}a", type=EvidenceType.PHOTO,
                             file_path=str(img), sha256=s, captured_at=captured,
                             location=location),
                EvidenceItem(evidence_id=f"EV-{i}b", type=EvidenceType.DOCUMENT,
                             file_path=str(work_dir / f"genuine_{i}.txt"),
                             doc_fields={"date": (dispatched + timedelta(hours=1)).isoformat(),
                                         "expected": "地面干燥无积水", "actual": "地面干燥无积水",
                                         "text": "清洁记录"}),
            ]

        sub = EvidenceSubmission(issue_id="INS-EVAL-EV", submitted_at=captured + timedelta(minutes=10),
                                 submitter="班组长", evidences=evs)
        # 真整改提供一致的关联系统记录（MES/环境采样已接入的情形）
        linked = {"record_value": "台面清洁", "evidence_claim": "台面清洁"} \
            if kind == "genuine" else {}
        res = agent.verify_evidence(sub, order, grade, clause_is_ccp=False,
                                    linked_records=linked)
        results.append({
            "kind": kind,
            "route": res.route.value,
            "reject_expected": case.reject_expected,
            "checks": [(c.rule_id, c.verdict.value) for c in res.checks],
            "first_verdict": res.checks[0].verdict.value if res.checks else None,
        })

    summary = summarize_verify(results)
    # 分样本类型明细
    by_kind: dict[str, dict[str, int]] = {}
    for r in results:
        k = r["kind"]
        by_kind.setdefault(k, {"total": 0, "rejected": 0})
        by_kind[k]["total"] += 1
        by_kind[k]["rejected"] += int(r["route"] == "reject")
    summary["by_kind"] = by_kind
    return summary
