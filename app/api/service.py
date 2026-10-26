"""FastAPI 服务层。

对外接口按角色划分：巡检员 / 质量工程师 / 质量授权人 / 管理员。
所有写操作均产生审计记录；证据提交接口会计算指纹并做五道校验。
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.core.clock import HumanTaskQueue
from app.core.repository import get_repo
from app.core.schemas import (
    EvidenceItem,
    EvidenceSubmission,
    EvidenceType,
    RiskLevel,
    utcnow,
)
from app.graph.langgraph_flow import run_graph
from app.graph.orchestrator import (
    TransitionError,
    build_graph_description,
    get_default_orchestrator,
)

app = FastAPI(
    title="巡检整改多Agent闭环系统",
    description="条款映射 / 风险分级 / 证据验证 —— LLM做抽取检索与语义核验，规则做分级与闸门，人工保留A级确认与现场复核权。",
    version="0.3.0",
)


def get_orch():
    return get_default_orchestrator()


# --------------------------------------------------------------------------
# 请求/响应模型
# --------------------------------------------------------------------------
class CreateIssueReq(BaseModel):
    issue_id: Optional[str] = None
    raw_text: str
    reporter: str = "巡检员"
    photos: list[str] = Field(default_factory=list)
    known_zone: str = ""
    idempotency_key: str = ""


class EvidenceItemReq(BaseModel):
    evidence_id: Optional[str] = None
    type: EvidenceType
    file_path: str
    file_name: str = ""
    captured_at: Optional[datetime] = None
    sha256: str = ""
    phash: str = ""
    location: str = ""
    doc_fields: dict[str, Any] = Field(default_factory=dict)
    note: str = ""


class SubmitEvidenceReq(BaseModel):
    submitter: str = ""
    note: str = ""
    evidences: list[EvidenceItemReq]


class ConfirmClauseReq(BaseModel):
    clause_id: str
    confirmed_by: str


class ConfirmGradeReq(BaseModel):
    confirmed_by: str
    final_level: Optional[RiskLevel] = None
    downgrade_reason: str = ""


class ReviewReq(BaseModel):
    passed: bool
    reviewer: str
    note: str = ""


class RecheckReq(BaseModel):
    passed: bool
    reviewer: str
    sampled: Optional[bool] = None
    note: str = ""
    rng_value: Optional[float] = None


class ExtensionReq(BaseModel):
    hours: int
    requested_by: str


class VerifyReq(BaseModel):
    linked_records: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# 巡检员接口
# --------------------------------------------------------------------------
@app.post("/api/issues")
def create_issue(req: CreateIssueReq, orch=Depends(get_orch)) -> dict[str, Any]:
    issue_id = req.issue_id or f"INS-{utcnow().strftime('%Y%m%d-%H%M%S')}"
    try:
        issue = orch.create_issue(issue_id, req.raw_text, req.reporter, req.photos,
                                   req.known_zone, req.idempotency_key)
    except TransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return issue.model_dump(mode="json")


@app.get("/api/issues")
def list_issues(limit: int = 100) -> dict[str, Any]:
    issues = get_repo().list_issues(limit)
    return {"items": [i.model_dump(mode="json") for i in issues], "count": len(issues)}


@app.get("/api/issues/{issue_id}")
def get_issue(issue_id: str) -> dict[str, Any]:
    issue = get_repo().load_issue(issue_id)
    if not issue:
        raise HTTPException(status_code=404, detail="问题单不存在")
    return issue.model_dump(mode="json")


@app.get("/api/issues/{issue_id}/audit")
def audit_trail(issue_id: str) -> dict[str, Any]:
    return {"issue_id": issue_id, "records": get_repo().audit_trail(issue_id)}


# --------------------------------------------------------------------------
# 编排流程
# --------------------------------------------------------------------------
@app.post("/api/issues/{issue_id}/advance")
def advance(issue_id: str, orch=Depends(get_orch)) -> dict[str, Any]:
    """驱动状态机到下一个需要人工或外部输入的节点。"""
    try:
        return run_graph(issue_id, orch)
    except TransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/issues/{issue_id}/verify")
def run_verify(issue_id: str, req: VerifyReq, orch=Depends(get_orch)) -> dict[str, Any]:
    try:
        issue = orch.run_verify(issue_id, req.linked_records)
    except TransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return issue.model_dump(mode="json")


@app.post("/api/issues/{issue_id}/tick")
def tick(issue_id: str, orch=Depends(get_orch)) -> dict[str, Any]:
    return orch.tick(issue_id).model_dump(mode="json")


# --------------------------------------------------------------------------
# 人工确认接口
# --------------------------------------------------------------------------
@app.post("/api/issues/{issue_id}/clause/confirm")
def confirm_clause(issue_id: str, req: ConfirmClauseReq, orch=Depends(get_orch)) -> dict[str, Any]:
    try:
        return orch.confirm_clause(issue_id, req.clause_id, req.confirmed_by).model_dump(mode="json")
    except TransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/api/issues/{issue_id}/grade/confirm")
def confirm_grade(issue_id: str, req: ConfirmGradeReq, orch=Depends(get_orch)) -> dict[str, Any]:
    try:
        return orch.confirm_grade(issue_id, req.confirmed_by, req.final_level,
                                   req.downgrade_reason).model_dump(mode="json")
    except TransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/issues/{issue_id}/evidence")
def submit_evidence(issue_id: str, req: SubmitEvidenceReq, orch=Depends(get_orch)) -> dict[str, Any]:
    items = []
    for i, ev in enumerate(req.evidences):
        ev = ev.model_copy()
        if not ev.evidence_id:
            ev.evidence_id = f"EV-{issue_id}-{i + 1:02d}"
        items.append(EvidenceItem(**ev.model_dump()))
    sub = EvidenceSubmission(issue_id=issue_id, submitter=req.submitter,
                             evidences=items, note=req.note)
    return orch.submit_evidence(issue_id, sub).model_dump(mode="json")


@app.post("/api/issues/{issue_id}/evidence/review")
def review_evidence(issue_id: str, req: ReviewReq, orch=Depends(get_orch)) -> dict[str, Any]:
    return orch.review_evidence(issue_id, req.passed, req.reviewer, req.note).model_dump(mode="json")


@app.post("/api/issues/{issue_id}/recheck")
def recheck(issue_id: str, req: RecheckReq, orch=Depends(get_orch)) -> dict[str, Any]:
    return orch.run_recheck(issue_id, req.passed, req.reviewer, req.sampled,
                            req.note, req.rng_value).model_dump(mode="json")


@app.post("/api/issues/{issue_id}/extension")
def request_extension(issue_id: str, req: ExtensionReq, orch=Depends(get_orch)) -> dict[str, Any]:
    return orch.request_extension(issue_id, req.hours, req.requested_by).model_dump(mode="json")


@app.post("/api/issues/{issue_id}/extension/decision")
def extension_decision(issue_id: str, req: ReviewReq, orch=Depends(get_orch)) -> dict[str, Any]:
    return orch.approve_extension(issue_id, req.passed, req.reviewer).model_dump(mode="json")


# --------------------------------------------------------------------------
# 人工任务队列 / 看板 / 状态机
# --------------------------------------------------------------------------
@app.get("/api/tasks")
def pending_tasks() -> dict[str, Any]:
    return {"items": HumanTaskQueue().pending()}


@app.post("/api/tasks/{task_id}/complete")
def complete_task(task_id: str, body: dict[str, Any]) -> dict[str, Any]:
    ok = HumanTaskQueue().complete(task_id, body.get("done_by", "system"))
    if not ok:
        raise HTTPException(status_code=404, detail="任务不存在或已完成")
    return {"task_id": task_id, "done": True}


@app.post("/api/tasks/escalate")
def escalate_tasks() -> dict[str, Any]:
    return {"escalated": HumanTaskQueue().escalate_overdue()}


@app.get("/api/stats")
def stats(orch=Depends(get_orch)) -> dict[str, Any]:
    return orch.stats()


@app.get("/api/graph")
def graph() -> dict[str, Any]:
    return build_graph_description()


@app.get("/api/knowledge/validate")
def validate_knowledge() -> dict[str, Any]:
    lib = get_default_orchestrator().library
    problems = lib.validate()
    return {"clause_count": len(lib.clauses), "effective_count": len(lib.latest_by_id()),
            "problems": problems}


@app.get("/api/samples/mapping")
def mapping_samples(limit: int = 200) -> dict[str, Any]:
    """人工确认回流的映射样本库（用于离线调阈值，不在线训练）。"""
    return {"items": get_repo().mapping_samples(limit)}


def main() -> None:
    import uvicorn

    uvicorn.run("app.api.service:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
