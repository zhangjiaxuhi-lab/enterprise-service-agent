"""
LangGraph 状态机测试（架构层 + 黄金用例）。

测试分层
--------
1. **路由层单测**：用极简模型桩验证 ``should_continue`` 与图拓扑，
   与业务规则解耦。
2. **黄金用例（golden cases）**：把真实业务场景固化为断言。
   这是本文件的核心价值 —— 任何 Prompt / 模型 / 检索改动后，
   跑一次即可知道有没有改坏既有行为。

关于 mock 模型能证明什么、不能证明什么
--------------------------------------
``_MockToolCallingModel`` 复刻了与 System Prompt 相同的槽位规则，因此它能验证：

* 图结构、条件边路由、工具节点执行、状态归并（**架构正确性**）；
* 槽位填充的**判定逻辑**（缺参不调用工具）；
* 多轮上下文是否被正确保留。

它**不能**验证真实 ``qwen-plus`` 是否遵守 Prompt —— 那属于
``@pytest.mark.real_model`` 测试的职责（需要真实 API Key）。
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent.customer_agent import (
    END,
    NODE_AGENT,
    NODE_TOOLS,
    AgentState,
    should_continue,
)

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _run(graph: Any, message: str, thread_id: str = "t") -> tuple[list[str], list[dict], list[Any]]:
    """
    执行一轮对话，返回节点流转轨迹、工具调用记录与最终消息。

    Args:
        graph: 已编译的图。
        message: 用户输入。
        thread_id: 会话线程 ID。

    Returns:
        tuple: (节点名序列, 工具调用列表, 最终消息列表)
    """
    config = {"configurable": {"thread_id": thread_id}}
    nodes: list[str] = []
    calls: list[dict] = []

    for chunk in graph.stream(
        {"messages": [HumanMessage(content=message)]},
        config=config,
        stream_mode="updates",
    ):
        for node, update in chunk.items():
            if node.startswith("__"):
                continue
            nodes.append(node)
            for item in (update or {}).get("messages", []) or []:
                for call in getattr(item, "tool_calls", None) or []:
                    calls.append({"name": call.get("name"), "args": call.get("args", {})})

    final = list(graph.get_state(config).values.get("messages", []))
    return nodes, calls, final


def _tool_text(messages: list[Any], tool_name: str) -> str:
    """
    取出指定工具最后一次返回的**原始文本**。

    适用于 ``query_knowledge_base`` 这类返回纯文本（非 JSON）的工具。

    Args:
        messages: 消息历史。
        tool_name: 工具名。

    Returns:
        str: 工具返回的原始内容；未找到返回空串。
    """
    for item in reversed(messages):
        if isinstance(item, ToolMessage) and getattr(item, "name", None) == tool_name:
            content = item.content
            return content if isinstance(content, str) else str(content)
    return ""


def _tool_payload(messages: list[Any], tool_name: str) -> dict:
    """
    从消息历史中取出指定工具最后一次返回的 **JSON** 负载。

    仅适用于返回 JSON 字符串的工具（如 ``submit_ticket``）；
    对纯文本工具返回空字典，此时请改用 :func:`_tool_text`。

    Args:
        messages: 消息历史。
        tool_name: 工具名。

    Returns:
        dict: 解析后的负载；非 JSON 或解析失败返回空字典。
    """
    raw = _tool_text(messages, tool_name)
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


# ---------------------------------------------------------------------------
# 1) 路由层单测
# ---------------------------------------------------------------------------


class TestRouting:
    """``should_continue`` 条件边行为。"""

    def test_no_tool_calls_ends(self) -> None:
        """无 tool_calls 时应结束，而不是进入工具节点。"""
        state: AgentState = {"messages": [AIMessage(content="你好")]}
        assert should_continue(state) == END

    def test_with_tool_calls_goes_to_tools(self) -> None:
        """存在 tool_calls 时应流转至 tools 节点。"""
        message = AIMessage(
            content="",
            tool_calls=[
                {"name": "submit_ticket", "args": {}, "id": "c1", "type": "tool_call"}
            ],
        )
        assert should_continue({"messages": [message]}) == NODE_TOOLS

    def test_empty_tool_calls_ends(self) -> None:
        """``tool_calls=[]``（而非 None）同样应结束。"""
        assert should_continue({"messages": [AIMessage(content="x", tool_calls=[])]}) == END


class TestGraphTopology:
    """图拓扑契约：节点与边的存在性。"""

    def test_nodes_exist(self, mock_graph: Any) -> None:
        """必需节点存在，且恰好是 agent 与 tools。"""
        nodes = set(mock_graph.get_graph().nodes)
        assert {NODE_AGENT, NODE_TOOLS} <= nodes

    def test_tools_loops_back_to_agent(self, mock_graph: Any) -> None:
        """关键契约：tools 必须回边到 agent（而非直接 END）。

        这条边是「工具结果由模型转写成自然语言」的实现基础，
        若被误改为 tools → END，用户会直接看到 JSON 报文。
        """
        edges = {(e.source, e.target) for e in mock_graph.get_graph().edges}
        assert (NODE_TOOLS, NODE_AGENT) in edges, "tools → agent 强制回边缺失"
        assert (NODE_TOOLS, END) not in edges, "tools 不应直接结束"

    def test_conditional_edge_from_agent(self, mock_graph: Any) -> None:
        """agent 出边为条件边，且能到达 tools 与 END。"""
        cond = {
            (e.source, e.target)
            for e in mock_graph.get_graph().edges
            if e.conditional
        }
        assert (NODE_AGENT, NODE_TOOLS) in cond
        assert (NODE_AGENT, END) in cond


class TestRoutingThroughGraph:
    """通过真实图执行验证路由。"""

    def test_no_tool_path_skips_tools_node(self, empty_graph: Any) -> None:
        """模型不请求工具时，日志中不应出现 tools 节点。"""
        nodes, calls, _ = _run(empty_graph, "随便聊聊")
        assert nodes == [NODE_AGENT], f"期望仅 agent 节点，实际 {nodes}"
        assert calls == []


# ---------------------------------------------------------------------------
# 2) 黄金用例 —— 知识库问答
# ---------------------------------------------------------------------------

KNOWLEDGE_QUERIES: list[str] = [
    "你好，系统登录提示 403 权限不足怎么排查解决？",
    "403 权限不足怎么解决？",
    "遇到 401 未授权报错怎么办？",
    "密码连续输错账号被锁定了，多久能解锁？",
    "你们的退款政策是什么？",
]


class TestGoldenKnowledgeQuery:
    """查询类意图：应检索知识库，且不得凭空作答。"""

    @pytest.mark.parametrize("question", KNOWLEDGE_QUERIES)
    def test_triggers_knowledge_base(self, fresh_graph_factory: Any, question: str) -> None:
        """每个问法都应触发 query_knowledge_base 检索。"""
        nodes, calls, _ = _run(fresh_graph_factory(), question)
        assert NODE_TOOLS in nodes, f"未检索知识库，节点轨迹={nodes}"
        assert [c["name"] for c in calls] == ["query_knowledge_base"], f"实际调用={calls}"

    def test_tool_result_is_grounded_in_documents(self, fresh_graph_factory: Any) -> None:
        """检索结果必须命中真实知识库文档，并标注来源。"""
        _, _, messages = _run(fresh_graph_factory(), "403 权限不足怎么解决？")
        text = _tool_text(messages, "query_knowledge_base")
        assert text, "工具未返回内容"
        assert "命中片段数：0" not in text, "知识库零命中"
        assert "来源：" in text and ".md" in text, f"缺少来源标注：{text[:200]!r}"

    def test_follows_tool_call_with_natural_language(self, fresh_graph_factory: Any) -> None:
        """工具执行后必须回到 agent 生成自然语言总结（回边生效）。"""
        nodes, _, messages = _run(fresh_graph_factory(), "403 权限不足怎么解决？")
        assert nodes == [NODE_AGENT, NODE_TOOLS, NODE_AGENT], f"节点轨迹异常：{nodes}"
        last = messages[-1]
        assert isinstance(last, AIMessage)
        assert last.content.strip(), "最终答复为空"
        assert not last.tool_calls, "最终答复不应再请求工具"


# ---------------------------------------------------------------------------
# 3) 黄金用例 —— 工单槽位填充（核心业务约束）
# ---------------------------------------------------------------------------

# (场景描述, 用户输入) —— 全部缺少 user_id，必须反问而非调用工具
MISSING_SLOT_CASES: list[tuple[str, str]] = [
    ("完全缺参", "我买错套餐了，帮我提个退费申请。"),
    ("只有诉求无账号", "我要申请退款，昨天买错了套餐。"),
    ("只有账号无诉求", "账号是 U-987654。"),
    ("只有诉求和账号在问句里但无原因", "帮我提交一个报障工单可以吗？"),
]

# (场景描述, 用户输入, 期望 issue_type) —— 三槽齐备，必须提交工单
COMPLETE_SLOT_CASES: list[tuple[str, str, str]] = [
    (
        "退款 + 误购",
        "账号是 U-987654，我要申请退款，昨天购买的年费企业版误选了双份，申请退订一份。",
        "refund",
    ),
    (
        "退款 + 买错套餐",
        "账号 U-100238 买错套餐了，需要退款，请帮我处理。",
        "refund",
    ),
    (
        "投诉",
        "我是 U-555111，对这次的服务非常不满，要投诉你们的技术支持。",
        "complaint",
    ),
]

# 判断描述是否合格的 API 契约
_DESCRIPTION_MIN_LEN: int = 10


class TestGoldenSlotFilling:
    """办理类意图的槽位填充规则 —— 本项目最关键的业务约束。"""

    @pytest.mark.parametrize(
        "label,question", MISSING_SLOT_CASES, ids=[c[0] for c in MISSING_SLOT_CASES]
    )
    def test_missing_slots_never_calls_tool(
        self, fresh_graph_factory: Any, label: str, question: str
    ) -> None:
        """缺参时**绝不能**调用工具，只能反问。

        这是防「占位符/臆造参数」的核心防线。若此断言失败，
        说明模型开始用 unknown / N/A 之类的假参数硬凑工单，必须立即修复。
        """
        nodes, calls, messages = _run(fresh_graph_factory(), question)
        assert calls == [], f"[{label}] 缺参却调用了工具：{calls}"
        assert NODE_TOOLS not in nodes, f"[{label}] 不应进入工具节点，轨迹={nodes}"
        last = messages[-1]
        assert isinstance(last, AIMessage) and last.content.strip(), (
            f"[{label}] 应以自然语言反问，实际为空"
        )

    @pytest.mark.parametrize(
        "label,question,expected_type",
        COMPLETE_SLOT_CASES,
        ids=[c[0] for c in COMPLETE_SLOT_CASES],
    )
    def test_complete_slots_submits_ticket(
        self, fresh_graph_factory: Any, label: str, question: str, expected_type: str
    ) -> None:
        """三槽齐备时必须提交工单，且 issue_type 分类正确。"""
        nodes, calls, messages = _run(fresh_graph_factory(), question)
        assert [c["name"] for c in calls] == ["submit_ticket"], f"[{label}] 调用={calls}"
        assert nodes == [NODE_AGENT, NODE_TOOLS, NODE_AGENT], f"[{label}] 轨迹={nodes}"

        args = calls[0]["args"]
        # 三个必需参数齐备且非占位值
        for slot in ("user_id", "issue_type", "description"):
            assert args.get(slot), f"[{label}] 缺少 {slot}"
        assert args["issue_type"] == expected_type, (
            f"[{label}] issue_type 期望 {expected_type}，实际 {args['issue_type']}"
        )
        assert args["issue_type"] != "unknown", "不得使用占位符 issue_type"
        assert len(args["description"]) >= _DESCRIPTION_MIN_LEN, (
            f"[{label}] description 过短：{args['description']!r}"
        )

        # 工单确实创建成功，且流水号格式正确
        payload = _tool_payload(messages, "submit_ticket")
        assert payload.get("success") is True, f"[{label}] 工单未创建：{payload}"
        assert payload.get("ticket_id", "").startswith("TK-"), (
            f"[{label}] 流水号格式异常：{payload.get('ticket_id')}"
        )

    def test_forbidden_placeholder_values_rejected(self, fresh_graph_factory: Any) -> None:
        """显式断言：任何工具调用都不得携带占位符参数（跨全部问法）。"""
        forbidden = {"unknown", "n/a", "na", "待补充", "无", "", "none", "null"}
        for _, question in MISSING_SLOT_CASES:
            _, calls, _ = _run(fresh_graph_factory(), question)
            for call in calls:
                for key, value in call["args"].items():
                    assert str(value).strip().lower() not in forbidden, (
                        f"出现占位符参数 {key}={value!r}（问题：{question}）"
                    )


# ---------------------------------------------------------------------------
# 4) 多轮槽位填充（上下文保持）
# ---------------------------------------------------------------------------


class TestMultiTurnSlotFilling:
    """同一 thread_id 下跨轮补槽。"""

    def test_second_turn_submits_after_supplying_user_id(
        self, fresh_graph_factory: Any
    ) -> None:
        """第 1 轮缺账号 → 反问；第 2 轮补账号 → 立即提单。"""
        graph = fresh_graph_factory()

        _, calls1, _ = _run(graph, "我买错套餐了，帮我提个退费申请。", thread_id="mt-1")
        assert calls1 == [], "第 1 轮缺账号却调用了工具"

        _, calls2, messages2 = _run(graph, "账号是 U-555111。", thread_id="mt-1")
        assert [c["name"] for c in calls2] == ["submit_ticket"], (
            f"第 2 轮补全槽位后未提单：{calls2}"
        )
        args = calls2[0]["args"]
        assert args["user_id"] == "U-555111"
        # 描述应跨轮累积，不能只剩「账号是 U-555111」
        assert "退费" in args["description"] or "买错" in args["description"], (
            f"描述丢失了第 1 轮诉求：{args['description']!r}"
        )

    def test_threads_are_isolated(self, fresh_graph_factory: Any) -> None:
        """不同 thread_id 之间上下文必须隔离。"""
        graph = fresh_graph_factory()
        _run(graph, "账号是 U-987654，我要申请退款，昨天买错了套餐。", thread_id="iso-a")
        # 另一个线程只问业务知识，不应受到 A 线程工单上下文影响
        _, calls_b, _ = _run(graph, "退款需要多久到账？", thread_id="iso-b")
        assert all(c["name"] != "submit_ticket" for c in calls_b), (
            f"线程隔离失效，B 线程误触发提单：{calls_b}"
        )


# ---------------------------------------------------------------------------
# 5) 真实模型测试（默认跳过，需 API Key 与网络）
# ---------------------------------------------------------------------------


@pytest.mark.real_model
class TestRealModelGolden:
    """用真实 qwen-plus 验证 Prompt 约束是否被遵守。

    这组测试是 mock 测试无法替代的：mock 复刻的是「规则」，而这里验证的是
    「模型是否真的按规则行事」。默认跳过，按需运行：

        pytest -m real_model
    """

    @staticmethod
    def _build():
        from langgraph.checkpoint.memory import MemorySaver

        from src.agent.customer_agent import build_graph, build_model

        return build_graph(model=build_model(mock=False), checkpointer=MemorySaver())

    def test_knowledge_query_calls_kb(self) -> None:
        """真实模型应检索知识库。"""
        _, calls, _ = _run(self._build(), "403 权限不足怎么解决？")
        assert [c["name"] for c in calls] == ["query_knowledge_base"]

    def test_missing_slot_asks_instead_of_calling_tool(self) -> None:
        """真实模型在缺参时必须反问，不得调用工具（最关键的人工验证项）。"""
        _, calls, _ = _run(self._build(), "我买错套餐了，帮我提个退费申请。")
        assert calls == [], f"真实模型缺参仍调用工具：{calls}"

    def test_complete_slot_submits_ticket(self) -> None:
        """真实模型在槽位齐备时必须提单（曾因 Prompt 过严而不触发，回归保护）。"""
        _, calls, messages = _run(
            self._build(),
            "账号是 U-987654，我要申请退款，昨天购买的年费企业版误选了双份，申请退订一份。",
        )
        assert [c["name"] for c in calls] == ["submit_ticket"], f"调用={calls}"
        payload = _tool_payload(messages, "submit_ticket")
        assert payload.get("success") is True
