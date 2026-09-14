# 测试说明

## 快速运行

```bash
pytest                 # 跑全部测试（默认跳过真实模型用例）
pytest -v              # 显示每个用例名
pytest tests/test_agent_graph.py -k 槽位   # 只跑槽位填充相关用例
pytest -m real_model   # 显式启用真实模型用例（需 API Key 与网络）
```

## 目录结构

| 文件 | 覆盖范围 |
|---|---|
| `conftest.py` | 共享 fixtures：模型桩、图实例、路径注入 |
| `test_tools.py` | 两个业务工具的契约：参数校验、输出格式、检索正确性 |
| `test_agent_graph.py` | 状态机：路由层单测 + **黄金用例** + 多轮上下文 |
| `test_api.py` | HTTP 路由、入参校验、**SSE 事件协议契约**、前端契约 |

## 两套模型，两种证明力

这是本测试体系最重要的设计，**不要把两者混为一谈**：

| | mock 模型（默认） | 真实 `qwen-plus`（`-m real_model`） |
|---|---|---|
| 联网 | 否 | 是 |
| 消耗额度 | 否 | 是 |
| 进 CI | ✅ | ❌（默认跳过） |
| 确定性 | 完全确定 | 受模型随机性影响 |
| **能证明** | 图结构、条件边路由、工具节点执行、状态归并、槽位判定逻辑、多轮上下文 | **模型是否真的遵守 Prompt 约束** |
| **不能证明** | 真实模型的行为是否符合预期 | — |

> **为什么必须保留真实模型测试**：mock 模型复刻的是「规则」，
> 而真实模型测试验证的是「规则是否被遵守」。
> 项目中曾出现过真实模型把「诉求已齐备」判为「还需追问订单号」、
> 导致**不触发工单**的问题 —— 这类问题只有真实模型测试能发现。

### 真实模型测试需要 API Key

```bash
# .env 中配置 DASHSCOPE_API_KEY 后：
pytest -m real_model -v
```

未配置真实 Key 时，脚本会使用占位 Key，真实模型用例会失败（这是预期行为，
因为它们确实需要真实凭据）。

## 黄金用例（Golden Cases）

把已验证的业务行为固化为断言，防止后续改 Prompt / 换模型 / 调检索时**悄悄改坏**。

### 核心约束

| 约束 | 用例 |
|---|---|
| 查询类意图必须检索知识库 | `TestGoldenKnowledgeQuery::test_triggers_knowledge_base`（5 种问法） |
| **缺参绝不调用工具** | `TestGoldenSlotFilling::test_missing_slots_never_calls_tool`（4 种问法） |
| 槽位齐备必须提单且分类正确 | `TestGoldenSlotFilling::test_complete_slots_submits_ticket`（3 类诉求） |
| 任何调用不得携带占位符参数 | `test_forbidden_placeholder_values_rejected` |
| 工具结果必须回边由模型转写 | `test_follows_tool_call_with_natural_language` |
| 多轮补槽后应自动提单 | `TestMultiTurnSlotFilling::test_second_turn_submits_after_supplying_user_id` |
| 会话线程互相隔离 | `test_threads_are_isolated` |

### SSE 协议契约

`test_api.py::TestSSEProtocol` 守住对外协议，避免后端改动破坏前端：

* 事件顺序（`tool_start → tool_end → token → done`）
* 每种事件必带的字段（`tool_start` 有 `tool`+`args`，`tool_end` 有 `tool`+`output`）
* 每帧均为 `data: {...}\n\n` 格式
* 上游异常必须转换为 `error` 事件，而不是静默断流

## 编写新用例的约定

1. **需要用图时优先用 `fresh_graph_factory`** —— 每个用例拿全新实例，避免检查点状态串味。
2. **不要在测试里发真实网络请求**（`real_model` 标记的用例除外）。
3. **断言业务行为，而非实现细节** —— 例如断言「是否调用了 `submit_ticket`」，
   而不是断言「回复里第 20 个字是什么」。
4. **新增 Prompt 约束时，同步补一条黄金用例** —— 否则该约束无人守护。

## 与 `src/api/smoke_sse.py` 的分工

`smoke_sse.py` 是**冒烟脚本**，用于启动服务后**人眼观察**流式输出，不做断言、不构成测试覆盖。
严格校验请用 `pytest`。两者共享同一批测试场景，但职责不同：

| | `smoke_sse.py` | `pytest` |
|---|---|---|
| 目的 | 目视确认服务可用 | 自动判定行为正确 |
| 断言 | 无 | 有 |
| 需要启动服务 | 是 | 否（进程内） |
| 进 CI | 否 | 是 |
