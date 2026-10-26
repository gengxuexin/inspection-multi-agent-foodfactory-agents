# 巡检整改多Agent闭环系统

> 条款映射 / 风险分级 / 证据验证 —— LLM 负责抽取、检索与语义核验，**规则负责分级与闸门**，**人工保留 A 级确认和现场复核权**。

针对食品工厂巡检的三个真实痛点构建的整改闭环系统：

| 痛点 | 对应 Agent | 关键设计 |
|---|---|---|
| 问题与 HACCP 条款对不上、查表慢 | 条款映射 Agent | 元数据硬过滤 + 向量/BM25 双路召回 + RRF + rerank，置信度分三档，**低置信度不硬猜** |
| 人工分级容易错 | 风险分级 Agent | 20 条可执行规则决策表并行评估取最高等级，**LLM 不能单独定 A 级** |
| 口头答应、表上假改 | 证据验证 Agent | 五道闸门：清单完整性 → 元数据防旧图 → 哈希/相似度复用检测 → VLM 语义核验 → 交叉校验 |

架构不是"多个 Agent 自由对话协商"，而是 **Orchestrator + 共享状态 + 条件路由 + 硬闸门 + Human-in-the-loop**。

---

## 快速开始

```bash
pip install -r requirements.txt

# 1) 端到端演示（4 个场景，无需任何 API key）
python scripts/demo_end_to_end.py

# 2) 离线评测（三套评测 + 阈值网格搜索）
python scripts/run_eval.py

# 3) 单元测试（31 个用例，覆盖全部硬闸门）
python -m pytest tests -q

# 4) 启动 API 服务
python -m app.api.service          # http://127.0.0.1:8000/docs
```

默认使用 **mock LLM provider**（本地启发式实现），全流程离线可跑。
接入真实 LLM/VLM：

```bash
export FQA_LLM_PROVIDER=openai
export FQA_LLM_API_KEY=sk-xxx
export FQA_LLM_MODEL=gpt-4o-mini
```

LLM 不可用时**自动降级为规则分支 + 人工复核，绝不自动放行**。

---

## 目录结构

```
食品项目/
├── app/
│   ├── core/
│   │   ├── config.py            # 全部可调参数（阈值、时钟、相似度阈值）
│   │   ├── schemas.py           # 统一问题单 Schema（Pydantic）—— Agent 契约
│   │   ├── llm.py               # LLM/VLM 适配层（mock / OpenAI 兼容）
│   │   ├── repository.py        # SQLite 仓储 + 只追加审计日志
│   │   └── clock.py             # A/B/C 三级响应时钟 + 人工任务队列
│   ├── rag/
│   │   ├── knowledge.py         # 条款级知识库（HACCP/SSOP/GMP/法规）
│   │   ├── retriever.py         # 硬过滤→双路召回→RRF→rerank→置信度
│   │   └── fingerprint.py       # sha256 + 感知哈希(pHash) + 文本相似度
│   ├── agents/
│   │   ├── intake_agent.py           # Agent 1 巡检录入
│   │   ├── clause_map_agent.py       # Agent 2 条款映射（RAG）
│   │   ├── risk_grade_agent.py       # Agent 3 风险分级（规则决策表）
│   │   ├── dispatch_agent.py         # Agent 4 整改派单
│   │   └── evidence_verify_agent.py  # Agent 6 证据验证（五道闸门）
│   ├── graph/
│   │   ├── orchestrator.py      # 状态机 + 硬闸门 + 审计（核心）
│   │   └── langgraph_flow.py    # LangGraph 版条件边（可选依赖）
│   ├── eval/                    # 三套离线评测
│   └── api/service.py           # FastAPI 服务层
├── scripts/                     # 演示 / 评测 / 知识库构建
├── tests/                       # 31 个用例
├── data/                        # SQLite + 上传证据（运行时生成）
└── knowledge/                   # 条款库 JSON（企业自有条款放这里）
```

---

## 核心设计：三条不可绕过的硬约束

### 1. 低置信度条款映射不得自动定级

置信度由**多信号合成**（rerank 归一化分数 + 双路一致性 + 元数据匹配），
不是单看向量分数——单分数无法区分"检索偏了"和"条款库缺项"。

| 置信度 | 分流 | 行为 |
|---|---|---|
| ≥ 0.75 | `prefill` | 预填 Top-1，巡检员可改 |
| 0.50 – 0.75 | `human_select` | 展示 Top-3，**必须人工选，不预填** |
| < 0.50 | `no_reliable_match` | 显式弃答，转人工，可标记"新增条款需求" |

命中 `is_ccp=True` 的条款**不额外加分，但强制走人工确认**。
`no_reliable_match` 状态下调用分级会直接抛 `TransitionError`。

### 2. A 级 / CCP 问题不得仅凭照片自动关闭

规则表 20 条（A01–A08 / B01–B08 / C01–C04）并行评估，**取命中规则中的最高等级**，
同等级多条命中时**全部记录理由**。

规则全部未命中 → 进入 LLM 建议分支，但：
- LLM 建议 A → **强制人工确认**（模型不得单独定 A）
- LLM 建议 B/C 且置信度 < 0.7 → 人工确认
- 证据不足（缺监控数值等）→ **默认不低于 B**，标记"待补证"

人工降级必须填理由，原等级保留在 `original_level` 供复盘，代码层强制校验。

证据验证分流时，A 级/CCP **无论自动结果如何都转人工现场复核**——
自动通过率统计中不包含这部分（`tests/test_agents.py::test_level_a_always_forces_onsite_review`）。

### 3. 打回整改不重置倒计时

```
打回 → invalid_fix_count += 1，deadline 保持不变 → 超时照样升级
连续打回 2 次 → 自动升级一级响应
连续打回 3 次 → 通知厂长/质量总监
A级任何一次打回 → 强制质量经理知悉
```

无效整改次数进入责任部门月度统计（`DepartmentStats`）。
延期与打回独立：延期需质量授权人批准并留痕，**打回不能作为延期理由**；A级原则上不允许延期。

---

## 证据验证：五道闸门

判定语义：`通过` 进下一道；`打回` 立即终止；`存疑` 继续但最终汇总转人工。

| # | 校验 | 规则 | 关键拦截 |
|---|---|---|---|
| 1 | 清单完整性 | E101–E104 | 必交证据缺项 → 打回（证据类型由**问题类型+等级**决定，整改人不能自删） |
| 2 | 元数据 | E201–E208 | EXIF 早于派单 → **疑似旧图打回**；无 EXIF → 存疑（不误杀） |
| 3 | 历史复用 | E301–E305 | sha256 完全一致 → 打回；pHash 相似度 ≥0.95 → 打回；0.85–0.95 → 存疑 |
| 4 | 语义核验 | E401–E406 | VLM 输出结构化 `match/mismatch/uncertain`，**仅对明确不符自动打回**；文档字段矛盾 → 打回 |
| 5 | 交叉校验 | E501–E504 | 与 MES/设备日志矛盾 → 存疑；复检不合格 → 打回 |

分流结果：

| 分流 | 后续动作 | 处理人 |
|---|---|---|
| `reject` | 无效整改+1，倒计时继续 | 系统自动 |
| `human_onsite_review` | 现场确认签字后才可进入复查 | 质量授权人 |
| `human_review` | 远程复核证据+关联记录 | 质量工程师 |
| `auto_pass` | 进入复查抽验 | 系统自动 |

**防假改的四条硬规则**：证据类型不可自选、证据必须新鲜（派单后）、历史复用检测、打回不重置倒计时。

---

## 状态机与审计

```
草稿 → 待映射 → 待分级确认 → 待整改 → 待验证 → 待复查 → 已关闭
旁路：打回重开 / 存疑复核 / 升级CAPA / 延期审批中
```

`ALLOWED_TRANSITIONS` 是显式的合法转换表，`_transition()` 对非法转换**直接抛错**，
不静默通过（`tests/test_agents.py::test_illegal_transition_raises`）。

每次状态转换记录：操作人/Agent、时间、输入输出摘要、置信度、人工修改前后值、
引用来源、Agent 版本、问题单版本号。审计表**只提供 append 接口，无 update/delete**。

- **幂等**：状态转换带幂等键，重复提交返回既有记录并留痕
- **重试**：Agent 调用失败按策略重试，LLM 不合规降级规则分支
- **重放**：全链路可按 issue_id 重放

---

## 离线评测结果

```bash
python scripts/run_eval.py
```

### 条款映射（27 条人工标注样本）

| 指标 | 数值 |
|---|---|
| Top-1 准确率 | 85.19% |
| Top-3 命中率 | 96.30% |
| 预填率 | 37.04% |
| 无可靠匹配（弃答）率 | 18.52% |
| **错误预填率** | **3.70%** ← 优先压低 |

阈值网格搜索覆盖 high ∈ {0.70,0.75,0.80,0.85} × mid ∈ {0.45,0.50,0.55,0.60}，
推荐 **high=0.70 / mid=0.45**（错误预填率不变的前提下预填率从 37% 提到 52%）。

### 风险分级（22 条样本）

| 指标 | 数值 |
|---|---|
| A 级召回率 | 100% |
| A 级精确率 | 100% |
| 人工确认负担 | 50% |
| LLM 分支占比 | 4.55% |

### 证据验证（4 类 × 30 = 120 条）

| 指标 | 数值 |
|---|---|
| 打回精确率 / 召回率 | 1.0 / 1.0 |
| **真整改误杀率** | **0%**（目标 < 5%） |
| 自动通过率 | 25% |

报告写入 `data/eval_report.json`。

> **关于这些数字的诚实说明**：评测集样本量小，且规则与样本同源，
> 反映的是**逻辑正确性与闸门有效性**，不是工厂真实场景的泛化性能。
> 分级 100% 一致率尤其需要警惕——真实场景需要质量专家重新标注 100–200 条历史问题，
> 并用**独立标注**的样本复核。真实 VLM 的误判率也必须在真实照片库上重新测量。
> 简历中请按此口径表述，不要写成"已在生产环境验证"。

---

## 接入生产还需要做什么

当前实现可完整运行，但以下部分是**可运行的简化版**，生产化需替换：

| 模块 | 当前实现 | 生产替换 |
|---|---|---|
| 向量检索 | 可复现 hashing 投影（离线可跑） | ChromaDB / FAISS + 真实 embedding |
| Rerank | 词面重叠 + 元数据交叉特征 | cross-encoder（如 bge-reranker） |
| VLM | mock 启发式（读文件名标注） | 真实视觉模型 + 真实照片库评测 |
| 存储 | SQLite | PostgreSQL + Redis（接口已隔离在 `repository.py`） |
| 鉴权 | 无 | JWT/OIDC，按角色（巡检员/质量工程师/质量授权人） |
| 通知 | 仅落库 | 企业微信/钉钉待办与告警推送 |
| 条款库 | 内置 15 条示例 | 导入企业 HACCP/SSOP/GMP 全文，元数据由质量部校验 |

替换要点：只需覆盖 `retriever.py` 的 `vector_search` / `rerank`、
`llm.py` 的 `_chat` / `vlm_verify`，其余逻辑（过滤、融合、置信度、闸门、状态机）完全不变。

### 条款库导入

```bash
python scripts/build_knowledge_base.py --export    # 导出模板
# 编辑 knowledge/clauses.json 后校验
python scripts/build_knowledge_base.py --validate
python scripts/build_knowledge_base.py --stats
```

**元数据 `zone` / `process` / `is_ccp` / `hazard_type` 必须由质量部人工校验**——
它们是硬过滤的正确性前提，不能全靠模型抽取。

---

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/issues` | 创建问题单（支持幂等键） |
| GET | `/api/issues` / `/api/issues/{id}` | 查询问题单 |
| GET | `/api/issues/{id}/audit` | 审计轨迹 |
| POST | `/api/issues/{id}/advance` | 驱动状态机到下一个人工节点 |
| POST | `/api/issues/{id}/clause/confirm` | 人工确认条款（回流样本库） |
| POST | `/api/issues/{id}/grade/confirm` | 人工确认分级（A级降级需理由） |
| POST | `/api/issues/{id}/evidence` | 提交整改证据 |
| POST | `/api/issues/{id}/verify` | 执行五道校验 |
| POST | `/api/issues/{id}/evidence/review` | 人工远程/现场复核 |
| POST | `/api/issues/{id}/recheck` | 复查抽验并关闭 |
| POST | `/api/issues/{id}/extension` | 申请延期 |
| GET | `/api/tasks` | 人工任务队列 |
| GET | `/api/stats` | 看板（自动流转率、无效整改、部门统计） |
| GET | `/api/graph` | 状态机与硬闸门描述 |
| GET | `/api/knowledge/validate` | 条款库元数据校验 |

---

## 核心原则（写在代码里，不只是文档里）

1. **LLM 不做食品安全判定**。它只做抽取、检索理解、语义核验和理由生成。
2. **规则先行**。确定性规则分支直接短路，不调用 LLM，控制成本与延迟。
3. **低置信度弃答优于错误预填**。预填错条款会误导后续分级，比不预填更糟。
4. **误杀比漏放更伤信任**。真整改被打回会摧毁现场对系统的信任，所以存疑一律转人工。
5. **人工修改全部留痕**。谁改的、改前改后、为什么改，都可追溯。
6. **LLM 不可用不自动放行**，降级为规则分支 + 人工复核。
