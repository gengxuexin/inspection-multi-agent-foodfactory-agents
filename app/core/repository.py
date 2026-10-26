"""仓储层：问题单、审计日志、历史证据库、样本库。

生产环境把 SQLite 换成 PostgreSQL + Redis 即可，接口保持不变。
审计日志**只可追加，不可编辑**（Repository 只暴露 append，无 update/delete）。
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

from app.core.config import DB_PATH
from app.core.schemas import AuditRecord, IssueStateModel, utcnow

_LOCK = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS issues (
    issue_id TEXT PRIMARY KEY,
    version   INTEGER NOT NULL,
    state     TEXT NOT NULL,
    deadline  TEXT,
    payload   TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    issue_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    actor TEXT NOT NULL,
    actor_type TEXT NOT NULL,
    action TEXT NOT NULL,
    from_state TEXT,
    to_state TEXT,
    input_digest TEXT,
    output_digest TEXT,
    confidence REAL,
    before_value TEXT,
    after_value TEXT,
    source_refs TEXT,
    agent_version TEXT,
    version_no INTEGER,
    created_at TEXT NOT NULL,
    UNIQUE(issue_id, seq)
);
CREATE TABLE IF NOT EXISTS evidence_history (
    history_id TEXT PRIMARY KEY,
    issue_id TEXT NOT NULL,
    sha256 TEXT,
    phash TEXT,
    text TEXT,
    submitted_at TEXT
);
CREATE TABLE IF NOT EXISTS mapping_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    issue_id TEXT,
    sample TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS human_tasks (
    task_id TEXT PRIMARY KEY,
    issue_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    state TEXT NOT NULL,
    payload TEXT NOT NULL,
    assignee_role TEXT,
    due_at TEXT,
    done INTEGER DEFAULT 0,
    done_by TEXT,
    done_at TEXT,
    escalated_to TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS issue_keys (
    key TEXT PRIMARY KEY,
    issue_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class Repository:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path or DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=15)
        c.row_factory = sqlite3.Row
        return c

    def _init(self) -> None:
        with _LOCK, self._conn() as c:
            c.executescript(_SCHEMA)

    # -- 幂等键 ---------------------------------------------------------
    def claim_idempotency_key(self, key: str, issue_id: str) -> bool:
        """返回 True 表示首次占用；False 表示重复提交。"""
        with _LOCK, self._conn() as c:
            try:
                c.execute(
                    "INSERT INTO issue_keys(key, issue_id, created_at) VALUES(?,?,?)",
                    (key, issue_id, utcnow().isoformat()),
                )
                return True
            except sqlite3.IntegrityError:
                return False

    # -- 问题单 ---------------------------------------------------------
    def save_issue(self, issue: IssueStateModel) -> None:
        payload = issue.model_dump_json()
        with _LOCK, self._conn() as c:
            c.execute(
                """INSERT INTO issues(issue_id,version,state,deadline,payload,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?)
                   ON CONFLICT(issue_id) DO UPDATE SET
                     version=excluded.version, state=excluded.state,
                     deadline=excluded.deadline, payload=excluded.payload,
                     updated_at=excluded.updated_at""",
                (
                    issue.issue_id, issue.version, issue.state.value,
                    issue.deadline.isoformat() if issue.deadline else None,
                    payload, issue.created_at.isoformat(), utcnow().isoformat(),
                ),
            )

    def load_issue(self, issue_id: str) -> Optional[IssueStateModel]:
        with self._conn() as c:
            row = c.execute("SELECT payload FROM issues WHERE issue_id=?", (issue_id,)).fetchone()
        if not row:
            return None
        return IssueStateModel.model_validate_json(row["payload"])

    def list_issues(self, limit: int = 100) -> list[IssueStateModel]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT payload FROM issues ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [IssueStateModel.model_validate_json(r["payload"]) for r in rows]

    def count_by_state(self) -> dict[str, int]:
        with self._conn() as c:
            rows = c.execute("SELECT state, COUNT(*) n FROM issues GROUP BY state").fetchall()
        return {r["state"]: r["n"] for r in rows}

    # -- 审计日志（只追加） ----------------------------------------------
    def append_audit(self, rec: AuditRecord) -> None:
        with _LOCK, self._conn() as c:
            c.execute(
                """INSERT INTO audit_log(issue_id,seq,actor,actor_type,action,from_state,to_state,
                   input_digest,output_digest,confidence,before_value,after_value,source_refs,
                   agent_version,version_no,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    rec.issue_id, rec.seq, rec.actor, rec.actor_type, rec.action,
                    rec.from_state, rec.to_state, rec.input_digest, rec.output_digest,
                    rec.confidence,
                    json.dumps(rec.before_value, ensure_ascii=False),
                    json.dumps(rec.after_value, ensure_ascii=False),
                    json.dumps(rec.source_refs, ensure_ascii=False),
                    rec.agent_version, rec.version_no, rec.created_at.isoformat(),
                ),
            )

    def next_seq(self, issue_id: str) -> int:
        with self._conn() as c:
            row = c.execute(
                "SELECT COALESCE(MAX(seq),0) s FROM audit_log WHERE issue_id=?", (issue_id,)
            ).fetchone()
        return int(row["s"]) + 1

    def audit_trail(self, issue_id: str) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM audit_log WHERE issue_id=? ORDER BY seq", (issue_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # -- 历史证据库 -----------------------------------------------------
    def add_evidence_history(self, history_id: str, issue_id: str, sha256: str,
                             phash: str, text: str = "", submitted_at: str = "") -> None:
        with _LOCK, self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO evidence_history VALUES(?,?,?,?,?,?)",
                (history_id, issue_id, sha256, phash, text, submitted_at),
            )

    def load_evidence_history(self) -> list[dict[str, Any]]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM evidence_history").fetchall()]

    # -- 映射样本库（人工确认回流） --------------------------------------
    def add_mapping_sample(self, sample: dict[str, Any]) -> None:
        with _LOCK, self._conn() as c:
            c.execute(
                "INSERT INTO mapping_samples(issue_id,sample,created_at) VALUES(?,?,?)",
                (sample.get("issue_id", ""), json.dumps(sample, ensure_ascii=False),
                 utcnow().isoformat()),
            )

    def mapping_samples(self, limit: int = 1000) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT sample FROM mapping_samples ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [json.loads(r["sample"]) for r in rows]

    # -- 人工任务队列 ---------------------------------------------------
    def upsert_human_task(self, task: dict[str, Any]) -> None:
        with _LOCK, self._conn() as c:
            c.execute(
                """INSERT INTO human_tasks(task_id,issue_id,kind,state,payload,assignee_role,
                   due_at,done,done_by,done_at,escalated_to,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(task_id) DO UPDATE SET
                     payload=excluded.payload, assignee_role=excluded.assignee_role,
                     due_at=excluded.due_at, done=excluded.done, done_by=excluded.done_by,
                     done_at=excluded.done_at, escalated_to=excluded.escalated_to""",
                (
                    task["task_id"], task["issue_id"], task["kind"], task["state"],
                    json.dumps(task.get("payload", {}), ensure_ascii=False),
                    task.get("assignee_role", ""), task.get("due_at"),
                    1 if task.get("done") else 0, task.get("done_by", ""), task.get("done_at"),
                    json.dumps(task.get("escalated_to", []), ensure_ascii=False),
                    task.get("created_at") or utcnow().isoformat(),
                ),
            )

    def pending_human_tasks(self) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM human_tasks WHERE done=0 ORDER BY created_at"  # noqa: E712
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d["payload"] or "{}")
            d["escalated_to"] = json.loads(d["escalated_to"] or "[]")
            out.append(d)
        return out

    def close_human_task(self, task_id: str, done_by: str) -> bool:
        with _LOCK, self._conn() as c:
            cur = c.execute(
                "UPDATE human_tasks SET done=1, done_by=?, done_at=? WHERE task_id=? AND done=0",
                (done_by, utcnow().isoformat(), task_id),
            )
            return cur.rowcount > 0


_repo: Optional[Repository] = None


def get_repo() -> Repository:
    global _repo
    if _repo is None:
        _repo = Repository()
    return _repo
