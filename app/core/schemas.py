"""统一问题单 Schema（Pydantic）。

这是《最小系统规格说明》第二节"核心数据模型"的落地：
所有 Agent 的输出必须先通过这里的结构化约束，才允许写入共享状态机。
Agent 之间不直接对话，只通过这些模型读写状态。
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


# --------------------------------------------------------------------------
# 枚举
# --------------------------------------------------------------------------
class IssueState(str, Enum):
    """问题单状态机主链路 + 旁路。"""

    DRAFT = "草稿"
    PENDING_MAPPING = "待映射"
    PENDING_GRADE = "待分级确认"
    PENDING_FIX = "待整改"
    PENDING_VERIFY = "待验证"
    PENDING_RECHECK = "待复查"
    CLOSED = "已关闭"
    # 旁路
    REJECTED_REOPEN = "打回重开"
    UNCERTAIN_REVIEW = "存疑复核"
    CAPA = "升级CAPA"
    EXTENSION_PENDING = "延期审批中"


MAIN_FLOW = [
    IssueState.DRAFT,
    IssueState.PENDING_MAPPING,
    IssueState.PENDING_GRADE,
    IssueState.PENDING_FIX,
    IssueState.PENDING_VERIFY,
    IssueState.PENDING_RECHECK,
    IssueState.CLOSED,
]


class MapMode(str, Enum):
    PREFILL = "prefill"
    HUMAN_SELECT = "human_select"
    NO_RELIABLE_MATCH = "no_reliable_match"


class RiskLevel(str, Enum):
    A = "A"
    B = "B"
    C = "C"


class EvidenceType(str, Enum):
    PHOTO = "photo"
    VIDEO = "video"
    DOCUMENT = "document"
    SYSTEM_RECORD = "system_record"


class CheckVerdict(str, Enum):
    PASS = "通过"
    REJECT = "打回"
    UNCERTAIN = "存疑"


class VerifyRoute(str, Enum):
    REJECT = "reject"
    HUMAN_ONSITE = "human_onsite_review"
    HUMAN_REVIEW = "human_review"
    AUTO_PASS = "auto_pass"


# --------------------------------------------------------------------------
# Agent 1：巡检录入
# --------------------------------------------------------------------------
class QueryElements(BaseModel):
    """条款映射的查询要素抽取结果。抽取失败字段留空，不阻断流程。"""

    phenomenon: str = ""
    object: str = ""
    zone: str = ""
    process: str = ""
    deviation_type: str = ""
    hazard_guess: str = ""
    keywords: list[str] = Field(default_factory=list)
    extraction_ok: bool = True


class IssueDraft(BaseModel):
    """巡检录入 Agent 输出的结构化问题单草稿。"""

    issue_id: str
    raw_text: str
    discovered_at: datetime = Field(default_factory=utcnow)
    reporter: str = "巡检员"
    zone: str = ""
    workstation: str = ""
    process: str = ""
    phenomenon: str = ""
    involved_batch: str = ""
    involved_equipment: str = ""
    initial_category: str = ""
    suggested_evidence_types: list[EvidenceType] = Field(default_factory=list)
    low_confidence_fields: list[str] = Field(default_factory=list)
    photos: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Agent 2：条款映射
# --------------------------------------------------------------------------
class ClauseCandidate(BaseModel):
    clause_id: str
    source: str
    version: str
    effective_date: str = ""
    text_snippet: str
    confidence: float
    reason: str = ""
    zone: list[str] = Field(default_factory=list)
    process: list[str] = Field(default_factory=list)
    is_ccp: bool = False
    hazard_type: str = ""
    retrieval_trace: dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def _clip(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class ClauseMapResult(BaseModel):
    """条款映射 Agent 输出 Schema（对应设计稿第 4 节）。"""

    issue_id: str
    mode: MapMode
    selected_clause_id: Optional[str] = None
    candidates: list[ClauseCandidate] = Field(default_factory=list)
    needs_human: bool = True
    needs_human_reason: str = ""
    relaxed_filter_layers: list[str] = Field(default_factory=list)
    query_elements: Optional[QueryElements] = None
    retrieval_trace: dict[str, Any] = Field(default_factory=dict)
    retrieved_at: datetime = Field(default_factory=utcnow)
    agent_version: str = "clause-map-v0.3"


# --------------------------------------------------------------------------
# Agent 3：风险分级
# --------------------------------------------------------------------------
class RuleHit(BaseModel):
    rule_id: str
    level: RiskLevel
    reason: str
    need_human: bool = False


class GradeResult(BaseModel):
    issue_id: str
    level: RiskLevel
    source: Literal["rule", "llm_suggestion", "human"] = "rule"
    top_rule: Optional[str] = None
    hits: list[RuleHit] = Field(default_factory=list)
    llm_reason: str = ""
    llm_confidence: float = 0.0
    llm_clause_ref: str = ""
    need_human: bool = False
    need_human_reason: str = ""
    evidence_gaps: list[str] = Field(default_factory=list)
    clause_is_ccp: bool = False
    confirmed_by: Optional[str] = None
    confirmed_at: Optional[datetime] = None
    original_level: Optional[RiskLevel] = None
    downgrade_reason: str = ""
    agent_version: str = "risk-grade-v0.3"


# --------------------------------------------------------------------------
# Agent 4/5：派单与跟踪
# --------------------------------------------------------------------------
class DispatchOrder(BaseModel):
    issue_id: str
    owner: str
    department: str
    requirement: str
    level: RiskLevel
    dispatched_at: datetime = Field(default_factory=utcnow)
    deadline: datetime
    required_evidence: list[EvidenceType] = Field(default_factory=list)
    recheck_method: str = "现场复查"
    clause_id: str = ""


class EvidenceItem(BaseModel):
    evidence_id: str
    type: EvidenceType
    file_path: str
    file_name: str = ""
    captured_at: Optional[datetime] = None
    sha256: str = ""
    phash: str = ""
    location: str = ""
    doc_fields: dict[str, Any] = Field(default_factory=dict)
    note: str = ""


class EvidenceSubmission(BaseModel):
    issue_id: str
    submitted_at: datetime = Field(default_factory=utcnow)
    submitter: str = ""
    evidences: list[EvidenceItem] = Field(default_factory=list)
    note: str = ""


class CheckResult(BaseModel):
    """单道校验的判定结果。"""

    check: str
    rule_id: str
    verdict: CheckVerdict
    reason: str = ""
    evidence_id: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)


class VlmJudgement(BaseModel):
    """VLM 输出必须结构化，不允许自由文本。"""

    verdict: Literal["match", "mismatch", "uncertain"]
    confidence: float = 0.0
    reason: str = ""
    evidence_refs: list[str] = Field(default_factory=list)


class VerifyResult(BaseModel):
    issue_id: str
    route: VerifyRoute
    checks: list[CheckResult] = Field(default_factory=list)
    invalid_fix_count: int = 0
    deadline_kept: bool = True
    need_human_reason: str = ""
    vlm_used: str = "rules_only"  # rules_only | vlm | vlm_unavailable
    agent_version: str = "evidence-verify-v0.3"


# --------------------------------------------------------------------------
# Agent 7：复查与 CAPA
# --------------------------------------------------------------------------
class RecheckResult(BaseModel):
    issue_id: str
    sampled: bool
    method: str = ""
    passed: Optional[bool] = None
    reviewer: str = ""
    reviewed_at: Optional[datetime] = None
    note: str = ""


# --------------------------------------------------------------------------
# 共享状态
# --------------------------------------------------------------------------
class HumanTask(BaseModel):
    task_id: str
    issue_id: str
    kind: Literal[
        "select_clause", "confirm_grade", "onsite_review", "remote_review", "extension_approval"
    ]
    state: IssueState
    payload: dict[str, Any] = Field(default_factory=dict)
    assignee_role: str = ""
    created_at: datetime = Field(default_factory=utcnow)
    due_at: Optional[datetime] = None
    escalated_to: list[str] = Field(default_factory=list)
    done: bool = False
    done_by: str = ""
    done_at: Optional[datetime] = None


class AuditRecord(BaseModel):
    """审计日志只可追加，不可编辑。"""

    issue_id: str
    seq: int
    actor: str
    actor_type: Literal["agent", "human", "system"] = "system"
    action: str
    from_state: str = ""
    to_state: str = ""
    input_digest: str = ""
    output_digest: str = ""
    confidence: Optional[float] = None
    before_value: dict[str, Any] = Field(default_factory=dict)
    after_value: dict[str, Any] = Field(default_factory=dict)
    source_refs: list[str] = Field(default_factory=list)
    agent_version: str = ""
    version_no: int = 1
    created_at: datetime = Field(default_factory=utcnow)


class IssueStateModel(BaseModel):
    """问题单共享状态。Agent 只读写这个对象，不互相调用。"""

    issue_id: str
    version: int = 1
    state: IssueState = IssueState.DRAFT
    idempotency_keys: list[str] = Field(default_factory=list)

    draft: Optional[IssueDraft] = None
    clause_map: Optional[ClauseMapResult] = None
    selected_clause_id: Optional[str] = None
    original_top1: Optional[str] = None
    grade: Optional[GradeResult] = None
    order: Optional[DispatchOrder] = None
    submission: Optional[EvidenceSubmission] = None
    verify: Optional[VerifyResult] = None
    recheck: Optional[RecheckResult] = None

    reject_count: int = 0
    invalid_fix_count: int = 0
    deadline: Optional[datetime] = None
    escalations: list[dict[str, Any]] = Field(default_factory=list)
    capa_id: Optional[str] = None
    closed_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=utcnow)

    def key(self) -> str:
        return f"{self.issue_id}:{self.state.value}:{self.version}"
