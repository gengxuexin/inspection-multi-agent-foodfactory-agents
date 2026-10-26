"""升级时钟与人工任务队列（Agent 5）。

硬规则：
- 打回**不重置**倒计时，无效整改次数单独累计；
- 超时按级自动上推，不依赖人工催办；
- 人工任务超时未处理继续升级；
- A级原则上不允许延期（需质量总监特批并留痕）。
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from typing import Any, Optional

from app.core.config import settings
from app.core.repository import Repository, get_repo
from app.core.schemas import (
    HumanTask,
    IssueState,
    IssueStateModel,
    RiskLevel,
    utcnow,
)

_ESCALATE_TITLE = {
    "班组长": "整改确认超时",
    "车间主任": "整改逾期未闭环",
    "质量经理": "A/B级严重逾期",
    "厂长/质量总监": "重大食品安全风险逾期",
    "质量总监": "重大食品安全风险逾期",
}


class EscalationClock:
    """A/B/C 三级响应时钟。"""

    def __init__(self, repo: Optional[Repository] = None) -> None:
        self.repo = repo or get_repo()
        self.cfg = settings.clock

    def compute_deadline(self, level: RiskLevel, dispatched_at: datetime) -> datetime:
        return dispatched_at + timedelta(hours=self.cfg.fix_deadline_hours[level.value])

    def first_response_deadline(self, level: RiskLevel, dispatched_at: datetime) -> datetime:
        return dispatched_at + timedelta(hours=self.cfg.first_response_hours[level.value])

    def tick(self, issue: IssueStateModel, now: Optional[datetime] = None) -> list[dict[str, Any]]:
        """检查超时并生成升级记录。幂等：同一层级不重复升级。"""
        now = now or utcnow()
        if issue.deadline is None or issue.state in (IssueState.CLOSED, IssueState.CAPA):
            return []
        if now <= issue.deadline:
            return []
        level = issue.grade.level if issue.grade else RiskLevel.C
        path = self.cfg.escalation_path[level.value]

        # 已升级到第几层
        done = {e["level"] for e in issue.escalations}
        actions: list[dict[str, Any]] = []
        for idx, target in enumerate(path, start=1):
            if idx in done:
                continue
            # 已超时的时长决定上推几层
            overdue_h = (now - issue.deadline).total_seconds() / 3600
            if overdue_h < (idx - 1) * 24:
                break
            action = {
                "level": idx,
                "to": target,
                "title": _ESCALATE_TITLE.get(target, "整改升级"),
                "at": now.isoformat(),
                "issue_id": issue.issue_id,
                "issue_level": level.value,
                "invalid_fix_count": issue.invalid_fix_count,
                "note": "打回不重置倒计时" if issue.invalid_fix_count else "",
            }
            actions.append(action)
            issue.escalations.append(action)
        return actions

    def allow_extension(self, level: RiskLevel) -> tuple[bool, str]:
        """延期审批策略。A级原则上不允许。"""
        if level == RiskLevel.A:
            return False, "A级问题原则上不允许延期；如需延期须质量总监特批并留痕"
        return True, "需质量授权人批准并留痕"


class HumanTaskQueue:
    """统一人工待办队列。超时未处理继续上推。"""

    def __init__(self, repo: Optional[Repository] = None) -> None:
        self.repo = repo or get_repo()

    @staticmethod
    def _task_id(issue_id: str, kind: str) -> str:
        return f"T-{hashlib.md5(f'{issue_id}:{kind}'.encode()).hexdigest()[:10]}"

    def create(
        self,
        issue_id: str,
        kind: str,
        state: IssueState,
        payload: dict[str, Any],
        assignee_role: str = "",
        due_hours: Optional[int] = None,
    ) -> HumanTask:
        due_hours = due_hours or settings.clock.human_task_timeout_hours
        task = HumanTask(
            task_id=self._task_id(issue_id, kind),
            issue_id=issue_id,
            kind=kind,  # type: ignore[arg-type]
            state=state,
            payload=payload,
            assignee_role=assignee_role,
            due_at=utcnow() + timedelta(hours=due_hours),
        )
        self.repo.upsert_human_task(json.loads(task.model_dump_json()))
        return task

    def complete(self, task_id: str, done_by: str) -> bool:
        return self.repo.close_human_task(task_id, done_by)

    def pending(self) -> list[dict[str, Any]]:
        return self.repo.pending_human_tasks()

    def escalate_overdue(self, now: Optional[datetime] = None) -> list[dict[str, Any]]:
        """人工任务超时 → 上推一级。"""
        now = now or utcnow()
        out: list[dict[str, Any]] = []
        for t in self.pending():
            if not t.get("due_at"):
                continue
            due = datetime.fromisoformat(t["due_at"])
            if now <= due or t.get("escalated_to"):
                continue
            t["escalated_to"] = [t.get("assignee_role") or "班组长", "车间主任"]
            t["due_at"] = (now + timedelta(hours=24)).isoformat()
            self.repo.upsert_human_task(t)
            out.append({"task_id": t["task_id"], "escalated_to": t["escalated_to"]})
        return out


class DepartmentStats:
    """无效整改次数纳入责任部门月度统计（第六节打回后果）。"""

    @staticmethod
    def summarize(issues: list[IssueStateModel]) -> dict[str, dict[str, Any]]:
        stats: dict[str, dict[str, Any]] = {}
        for i in issues:
            dept = i.order.department if i.order else "未派单"
            row = stats.setdefault(dept, {"total": 0, "invalid_fix": 0, "reject": 0, "open": 0})
            row["total"] += 1
            row["invalid_fix"] += i.invalid_fix_count
            row["reject"] += i.reject_count
            if i.state not in (IssueState.CLOSED,):
                row["open"] += 1
        for row in stats.values():
            row["invalid_fix_rate"] = round(
                row["invalid_fix"] / row["total"], 3) if row["total"] else 0.0
        return stats
