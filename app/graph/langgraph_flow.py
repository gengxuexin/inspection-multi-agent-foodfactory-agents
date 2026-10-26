"""LangGraph 版状态机。

编排逻辑与 `orchestrator.py` 共用同一批 Agent 与同一套硬闸门，
差别只在于由 LangGraph 的条件边驱动跳转。LangGraph 未安装时自动回退到
`InspectionOrchestrator` 的显式流程，两条路径行为一致。
"""
from __future__ import annotations

from typing import Any, Optional, TypedDict

from app.core.schemas import EvidenceSubmission, IssueState, IssueStateModel, RiskLevel
from app.graph.orchestrator import InspectionOrchestrator, get_default_orchestrator


class FlowState(TypedDict, total=False):
    """LangGraph 共享状态。

    注意：每个节点只返回自己改动的 key，否则 LangGraph 会因同一步写入同一
    channel 两次而报 InvalidUpdateError。
    """

    issue_id: str
    next_node: str
    halt_reason: str


def LANGGRAPH_AVAILABLE() -> bool:
    try:
        import langgraph.graph  # noqa: F401

        return True
    except ImportError:
        return False


def build_issue_graph(orch: Optional[InspectionOrchestrator] = None) -> Any:
    """构建 LangGraph 状态机。

    节点对应设计稿主流程；条件边承载"低置信度转人工""A级强制确认"
    "证据打回不重置倒计时"等硬闸门。
    """
    from langgraph.graph import END, START, StateGraph

    orch = orch or get_default_orchestrator()

    def _load(state: FlowState) -> dict[str, Any]:
        return {}

    def _map(state: FlowState) -> dict[str, Any]:
        orch.run_clause_map(state["issue_id"])
        i = orch.repo.load_issue(state["issue_id"])
        if i.state == IssueState.UNCERTAIN_REVIEW:
            return {"next_node": "human_clause",
                    "halt_reason": i.clause_map.needs_human_reason if i.clause_map else "低置信度条款映射"}
        return {"next_node": "grade"}

    def _grade(state: FlowState) -> dict[str, Any]:
        i = orch.repo.load_issue(state["issue_id"])
        if i.state == IssueState.UNCERTAIN_REVIEW and i.clause_map and not i.selected_clause_id:
            return {"next_node": "human_clause", "halt_reason": "需人工选条款后才能分级"}
        if i.state == IssueState.PENDING_GRADE:
            orch.run_grade(state["issue_id"])
            i = orch.repo.load_issue(state["issue_id"])
        if i.state == IssueState.UNCERTAIN_REVIEW and i.grade and i.grade.need_human:
            return {"next_node": "human_grade", "halt_reason": i.grade.need_human_reason}
        return {"next_node": "await_fix"}

    def _verify(state: FlowState) -> dict[str, Any]:
        i = orch.repo.load_issue(state["issue_id"])
        if i.state == IssueState.PENDING_VERIFY and i.submission:
            orch.run_verify(state["issue_id"])
            i = orch.repo.load_issue(state["issue_id"])
        if i.state == IssueState.PENDING_FIX:
            # 区分"首次等待整改提交"与"证据被打回"
            if i.verify and i.verify.route.value == "reject":
                return {"next_node": "await_fix", "halt_reason": "证据打回，倒计时继续，不重置"}
            return {"next_node": "await_fix", "halt_reason": "等待整改人提交证据"}
        if i.state == IssueState.UNCERTAIN_REVIEW:
            return {"next_node": "human_review", "halt_reason": "存疑或A级/CCP强制人工复核"}
        if i.state == IssueState.PENDING_RECHECK:
            return {"next_node": "recheck"}
        return {"next_node": "closed"}

    def _halt(state: FlowState) -> dict[str, Any]:
        return {}

    g = StateGraph(FlowState)
    g.add_node("load", _load)
    g.add_node("map", _map)
    g.add_node("grade", _grade)
    g.add_node("verify", _verify)
    for n in ("human_clause", "human_grade", "await_fix", "human_review", "recheck", "closed"):
        g.add_node(n, _halt)

    g.add_edge(START, "load")
    g.add_edge("load", "map")
    g.add_edge("map", "grade")
    g.add_edge("grade", "verify")
    g.add_conditional_edges("map", lambda s: s["next_node"],
                             {"human_clause": "human_clause", "grade": "grade"})
    g.add_conditional_edges("grade", lambda s: s["next_node"],
                             {"human_clause": "human_clause", "human_grade": "human_grade",
                              "await_fix": "await_fix"})
    g.add_conditional_edges("verify", lambda s: s["next_node"],
                             {"await_fix": "await_fix", "human_review": "human_review",
                              "recheck": "recheck", "closed": "closed"})
    for n in ("human_clause", "human_grade", "await_fix", "human_review", "recheck"):
        g.add_edge(n, END)
    g.add_edge("closed", END)
    return g.compile()


def run_graph(issue_id: str, orch: Optional[InspectionOrchestrator] = None,
              thread_id: Optional[str] = None) -> dict[str, Any]:
    """驱动状态机到下一个需要人工或外部输入的节点（checkpoint 支持恢复）。"""
    orch = orch or get_default_orchestrator()
    if not LANGGRAPH_AVAILABLE():
        return _fallback_run(issue_id, orch)
    app = build_issue_graph(orch)
    cfg = {"configurable": {"thread_id": thread_id or issue_id}}
    return app.invoke({"issue_id": issue_id}, cfg)  # type: ignore[return-value]


def _fallback_run(issue_id: str, orch: InspectionOrchestrator) -> dict[str, Any]:
    """LangGraph 不可用时的等效显式流转（行为与图一致）。"""
    i = orch.repo.load_issue(issue_id)
    if not i:
        raise ValueError(f"问题单 {issue_id} 不存在")
    if i.state == IssueState.DRAFT:
        i = orch.run_clause_map(issue_id)
    if i.state == IssueState.UNCERTAIN_REVIEW:
        return {"issue_id": issue_id, "next_node": "human_clause",
                "halt_reason": i.clause_map.needs_human_reason if i.clause_map else "需人工选条款"}
    if i.state == IssueState.PENDING_GRADE:
        i = orch.run_grade(issue_id)
        if i.state == IssueState.UNCERTAIN_REVIEW:
            return {"issue_id": issue_id, "next_node": "human_grade",
                    "halt_reason": i.grade.need_human_reason if i.grade else "需人工确认分级"}
    if i.state == IssueState.PENDING_FIX:
        return {"issue_id": issue_id, "next_node": "await_fix",
                "halt_reason": "等待整改人提交证据"}
    if i.state == IssueState.PENDING_VERIFY:
        i = orch.run_verify(issue_id)
        if i.state == IssueState.PENDING_FIX:
            reason = ("证据打回，倒计时继续，不重置"
                      if i.verify and i.verify.route.value == "reject"
                      else "等待整改人提交证据")
            return {"issue_id": issue_id, "next_node": "await_fix", "halt_reason": reason}
        if i.state == IssueState.UNCERTAIN_REVIEW:
            return {"issue_id": issue_id, "next_node": "human_review",
                    "halt_reason": i.verify.need_human_reason if i.verify else "存疑转人工"}
        return {"issue_id": issue_id, "next_node": "recheck"}
    if i.state == IssueState.PENDING_RECHECK:
        return {"issue_id": issue_id, "next_node": "recheck"}
    return {"issue_id": issue_id, "next_node": "closed", "state": i.state.value}
