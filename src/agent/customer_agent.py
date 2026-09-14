"""
企业级智能客服与工单协同中枢 —— LangGraph 状态机编排（第二阶段）。

本模块基于 LangGraph ``StateGraph`` 构建「分析 → 工具执行 → 总结汇报」的
客服智能体闭环，节点编排如下::

        START
          │
          ▼
      ┌────────┐  有 tool_calls
      │ agent  │ ───────────────► ┌────────┐
      │ (LLM)  │ ◄─────────────── │ tools  │
      └────────┘   强制回边闭环    │(ToolNode)│
          │ 无 tool_calls          └────────┘
          ▼
         END

设计要点：
    * ``agent`` 节点调用已 ``bind_tools`` 的模型，负责意图识别与工具决策；
    * ``tools`` 节点由 ``ToolNode`` 执行第一阶段的两个真实 Python 工具；
    * 工具执行后**强制回边**至 ``agent``，由模型把工具返回的 JSON/文本
      转写成面向客户的得体自然语言（而非直接把 JSON 抛给用户）。

关于槽位填充（Slot Filling）：
    工单提报属「办理类意图」，模型必须先集齐 ``user_id``、``issue_type``、
    ``description`` 三个必需参数。缺失时必须自然语言反问，**严禁**用
    占位符或臆造值调用 ``submit_ticket``。该约束通过 System Prompt 强约束，
    并在 mock 模式下同样遵守，以保证测试结果可信。

运行方式（详见文件末尾 ``__main__`` 说明）::

    python -m src.agent.customer_agent              # 真实调用 DashScope
    $env:CUSTOMER_AGENT_MOCK="1"; python -m src.agent.customer_agent   # 离线验证
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# 路径与 .env 加载
# ---------------------------------------------------------------------------

# 项目根目录：本文件位于 <root>/src/agent/customer_agent.py
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# 自动加载根目录 .env（override=False：不覆盖已存在的真实环境变量）
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=False)

# 允许以 `python src/agent/customer_agent.py` 直接运行时仍能 import src.tools
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.tools.customer_service_tools import (  # noqa: E402
    query_knowledge_base,
    submit_ticket,
)

# ---------------------------------------------------------------------------
# 配置常量
# ---------------------------------------------------------------------------

DASHSCOPE_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME: str = "qwen-plus"
TEMPERATURE: float = 0.0

# 本阶段绑定的工具集合
TOOLS: list[BaseTool] = [query_knowledge_base, submit_ticket]

# 节点名称常量，避免拼写漂移
NODE_AGENT: str = "agent"
NODE_TOOLS: str = "tools"


# ---------------------------------------------------------------------------
# 核心业务 Prompt（AI PM 业务规则）
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = """你是「企业级智能售后支持专家」，服务于企业级 SaaS 客户的账号、\
权限、退款与工单事务。

# 一、身份与语气
- 你是专业、克制、有同理心的售后支持专家，代表企业面向客户沟通。
- 语气要求：先共情、后事实；不使用夸张营销话术，不承诺职责范围外的结果。
- 回答使用简体中文，结构清晰（必要时分点），涉及金额、时限、错误码时必须准确。

# 二、查询类意图（FAQ / 排障 / 政策咨询）
- 当客户询问错误排查（如 401 未授权、403 权限不足、密码锁定）、
  退款政策、工单流转规则等知识性问题时，**必须先调用 `query_knowledge_base`**
  检索企业知识库。
- **严格基于检索结果作答，严禁臆造知识库中未提及的规定、时限、金额或流程。**
- 若检索结果为空或与问题无关，应如实告知「知识库中暂无相关依据」，
  并建议转人工或补充信息，**不得编造答案**。
- 引用政策时，尽量说明依据来源（如「依据账号排障指南」）与关键数值。

# 三、办理类意图（工单提报 —— 槽位填充规则，最高优先级）
- 当客户表达提交退款、报障、投诉等办理需求时，必须集齐以下三个必需参数：
    1. `user_id`：客户账号唯一标识（形如 U-987654）；
    2. `issue_type`：问题类型，只能取以下枚举值之一 ——
       account（账号与登录）、refund（退款与结算）、billing（账单与发票）、
       technical（技术故障）、feature（功能咨询）、complaint（投诉与建议）；
    3. `description`：问题详细描述，需包含基本诉求与原因。
- **【参数合格判定 —— 宽松标准，务必遵守】**
  - 判定参数是否合格时，**不得要求客户必须提供精准订单号**。
  - 只要满足以下两点，即视为参数合格，应**立即调用 `submit_ticket`**，不得继续追问：
      1. 客户已提供账号 ID（`user_id`）；
      2. `description` 中具备**基本诉求 + 退款/报障原因**
         （例如「买错套餐，申请退费」「年费误选双份，申请退订一份」）。
  - 订单号、支付时间、金额等属于**锦上添花**的信息，缺失不影响提交：
    应先按现有信息提交工单，再由工单受理方在跟进时补充，**绝不因此驳回或反复追问**。
- **【硬性禁止】仅在 `user_id` 或「诉求/原因」确实缺失时，才允许不调用工具。**
  - 禁止使用 `"unknown"`、`"N/A"`、`"待补充"`、`""` 等占位符或虚构值凑齐参数；
  - 禁止自行猜测 `user_id` 或编造问题类型；
  - 此时必须**以亲切自然的语气反问客户**，一次性、清晰地列出所有缺失项，
    并说明为什么需要该信息（便于定位账号与快速处理）。
- **仅当三个参数全部明确且有效时**，才允许调用 `submit_ticket`。
- 工具返回后，用自然语言向客户汇报：工单号、当前状态、受理部门、
  预计响应时效与下一步动作；不要把原始 JSON 直接粘贴给客户。
  - 若客户未提供订单号，可在汇报时顺带说明「后续受理同事会与您核对订单信息」。

# 四、对话策略
- 一次回复只做一件事：要么检索/办理，要么追问缺失信息，避免同时追问过多无关信息。
- 若客户未提供账号，优先追问 `user_id`；随后确认问题类型；
  最后引导客户补充具体描述（订单号、时间、现象、诉求）。
- 不索取明文密码；提醒客户敏感信息需脱敏。
"""

# 工单提报所需的三个必需槽位（供 mock 模式与自检共用）
REQUIRED_TICKET_SLOTS: tuple[str, ...] = ("user_id", "issue_type", "description")

# 判定「办理类意图」的关键词
_TICKET_INTENT_KEYWORDS: tuple[str, ...] = (
    "工单", "提个", "提交", "报障", "报修", "投诉", "退费", "退款", "退订",
    "申请退", "开票", "帮我提", "帮我办",
)


# ---------------------------------------------------------------------------
# 状态定义
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """状态机状态。

    ``add_messages`` 是 LangGraph 提供的 reducer：节点只需返回**增量**消息，
    框架会按消息 ID 自动去重并追加到历史中，从而天然支持多轮工具循环。
    """

    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# 模型与图构建
# ---------------------------------------------------------------------------


def build_model(*, mock: bool = False) -> Any:
    """
    构建对话模型。

    真实模式使用 ``ChatOpenAI`` 对接 DashScope 的 OpenAI 兼容模式
    （``qwen-plus``，``temperature=0`` 保证输出稳定可复现）。

    Args:
        mock: 为 True 时返回离线确定性模型，用于无 API Key 环境下验证
            图结构、路由与工具调用链路。

    Returns:
        Any: 已调用 ``bind_tools`` 绑定工具的模型（或 mock 模型）。

    Raises:
        RuntimeError: 真实模式下未配置 ``DASHSCOPE_API_KEY``。
    """
    if mock:
        # 延迟到此处构造，避免真实运行时引入测试替身
        model = _MockToolCallingModel()
        return model

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key or api_key.strip() in {"", "your_key_here"}:
        raise RuntimeError(
            "未检测到有效的 DASHSCOPE_API_KEY。\n"
            f"请在 {_PROJECT_ROOT / '.env'} 中配置 DASHSCOPE_API_KEY=<你的密钥>；\n"
            "或使用离线模式验证图结构与路由：\n"
            '  PowerShell:  $env:CUSTOMER_AGENT_MOCK="1"; python -m src.agent.customer_agent'
        )

    model = ChatOpenAI(
        model=MODEL_NAME,
        base_url=DASHSCOPE_BASE_URL,
        api_key=api_key,
        temperature=TEMPERATURE,
        # 开启流式：第三阶段 SSE 接口依赖它逐 token 产出，
        # 否则 LangGraph 的 stream_mode="messages" 只会拿到整段内容。
        streaming=True,
    )
    return model.bind_tools(TOOLS)


def agent_node(state: AgentState, *, model: Any) -> dict[str, list[BaseMessage]]:
    """
    节点 1（``agent``）：调用模型分析意图，产出回复或工具调用请求。

    Args:
        state: 当前状态（含完整消息历史）。
        model: 已绑定工具的模型；通过闭包注入，避免把模型塞进状态。

    Returns:
        dict: 仅包含**新增**的 AI 消息，交由 ``add_messages`` 归并。
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    response = model.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """
    条件边：判断是否需要进入工具节点。

    Args:
        state: 当前状态。

    Returns:
        str: ``"tools"`` 表示最后一条 AI 消息含工具调用；``END`` 表示可以结束。
    """
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None)
    if tool_calls:
        return NODE_TOOLS
    return END


def build_graph(
    *,
    model: Any | None = None,
    mock: bool = False,
    checkpointer: Any | None = None,
) -> Any:
    """
    构建并编译客服智能体状态图。

    图结构::

        START → agent ──(有 tool_calls)──► tools ──┐
                  ▲                                 │
                  └───────────(强制回边闭环)────────┘
                  │
                  └──(无 tool_calls)──► END

    Args:
        model: 可选的已绑定模型；为 None 时按 ``mock`` 参数自行构建。
        mock: 是否构建离线 mock 模型（仅当 ``model`` 为 None 时生效）。
        checkpointer: 可选持久化检查点（如 ``MemorySaver()``），
            用于保留多轮会话状态。

    Returns:
        Any: 已编译的 LangGraph 应用，支持 ``.stream(...)`` / ``.invoke(...)``。
    """
    if model is None:
        model = build_model(mock=mock)

    def _agent(state: AgentState) -> dict[str, list[BaseMessage]]:
        """绑定模型的 agent 节点（闭包注入模型依赖）。"""
        return agent_node(state, model=model)

    builder = StateGraph(AgentState)
    builder.add_node(NODE_AGENT, _agent)
    # 节点 2：ToolNode 负责真实执行 Python 工具（含异常包装与 ToolMessage 回填）
    builder.add_node(NODE_TOOLS, ToolNode(TOOLS))

    builder.add_edge(START, NODE_AGENT)
    # 条件边：由 should_continue 决定去 tools 还是结束
    builder.add_conditional_edges(
        NODE_AGENT,
        should_continue,
        {NODE_TOOLS: NODE_TOOLS, END: END},
    )
    # 闭环回路：工具执行完毕强制回到 agent，由模型组织自然语言汇报
    builder.add_edge(NODE_TOOLS, NODE_AGENT)

    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# 节点日志渲染（流式打印状态机流转）
# ---------------------------------------------------------------------------


def _normalize_content(content: Any) -> str:
    """
    将消息内容归一化为纯文本。

    兼容两种形态：纯字符串；以及 LangChain 新式的内容块列表
    （``[{"type": "text", "text": ...}, ...]``）。

    Args:
        content: 原始 ``message.content``。

    Returns:
        str: 可打印文本。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or block))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _describe_tool_call(call: dict[str, Any]) -> str:
    """
    将单次工具调用渲染为人类可读的一行摘要。

    Args:
        call: LangChain 工具调用字典（含 ``name`` / ``args``）。

    Returns:
        str: 形如 ``submit_ticket(user_id="U-1", issue_type="refund", ...)``。
    """
    name = call.get("name", "?")
    args = call.get("args", {}) or {}
    rendered = ", ".join(f"{key}={value!r}" for key, value in args.items())
    return f"{name}({rendered})"


def _render_node_output(node: str, update: dict[str, Any]) -> None:
    """
    打印单个节点的流转与输出内容。

    Args:
        node: 节点名称。
        update: 该节点返回的状态增量。
    """
    messages: Sequence[Any] = update.get("messages", []) or []

    for message in messages:
        if isinstance(message, AIMessage):
            text = _normalize_content(message.content).strip()
            tool_calls = getattr(message, "tool_calls", None) or []
            if text:
                print(f"    [LLM 回复] {text}")
            for call in tool_calls:
                print(f"    [LLM 决策] 请求调用工具 → {_describe_tool_call(call)}")
            if not text and not tool_calls:
                print("    [LLM 回复] (空内容)")
        elif isinstance(message, ToolMessage):
            name = getattr(message, "name", None) or "tool"
            raw = _normalize_content(message.content)
            preview = raw if len(raw) <= 400 else raw[:400] + " …(已截断)"
            print(f"    [工具执行] {name} 返回：")
            for line in preview.splitlines():
                print(f"        {line}")
            # 工单类结果额外高亮流水号，便于核对业务闭环
            if name == "submit_ticket":
                try:
                    payload = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    payload = {}
                if payload.get("success"):
                    print(
                        f"    [业务闭环] 工单号 {payload.get('ticket_id')} "
                        f"| 状态 {payload.get('status')} "
                        f"| 受理 {payload.get('routing', {}).get('assigned_group')}"
                    )
                else:
                    print(f"    [业务拦截] 工单未创建：{payload.get('error', raw[:120])}")
        else:
            print(f"    [消息] {type(message).__name__}: {_normalize_content(message.content)[:200]}")


def run_scenario(
    app: Any,
    user_input: str,
    *,
    thread_id: str,
    verbose_steps: bool = True,
) -> list[AnyMessage]:
    """
    流式执行单个测试场景，实时打印状态机流转日志。

    Args:
        app: 已编译的图。
        user_input: 用户输入文本。
        thread_id: 会话线程 ID，配合 checkpointer 隔离多轮上下文。
        verbose_steps: 是否打印 ``updates`` 级别的节点流转日志。

    Returns:
        list[AnyMessage]: 运行结束后的完整消息历史。
    """
    config = {"configurable": {"thread_id": thread_id}}
    step = 0
    for chunk in app.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="updates",
    ):
        # stream_mode="updates" 的 chunk 形如 {"节点名": 状态增量}
        for node, update in chunk.items():
            if node.startswith("__"):
                # 跳过 __start__ / __end__ 等框架内部伪节点
                continue
            step += 1
            if verbose_steps:
                print(f"  ── 步骤 {step} | 节点：{node} ──")
                _render_node_output(node, update or {})

    snapshot = app.get_state(config)
    return list(snapshot.values.get("messages", []))


# ---------------------------------------------------------------------------
# 离线 mock 模型（无 API Key 时验证图结构 / 路由 / 工具链路）
# ---------------------------------------------------------------------------


def _extract_user_id(text: str) -> str | None:
    """
    从用户输入中提取账号标识。

    Args:
        text: 用户输入。

    Returns:
        str | None: 形如 ``U-987654`` 的账号；未匹配返回 None。
    """
    match = re.search(r"\bU-\d{4,}\b", text, flags=re.IGNORECASE)
    return match.group(0).upper() if match else None


def _classify_issue_type(text: str) -> str:
    """
    依据关键词粗略判定问题类型（仅 mock 模式使用）。

    Args:
        text: 用户输入。

    Returns:
        str: ``issue_type`` 枚举值之一。
    """
    if any(word in text for word in ("退费", "退款", "退订")):
        return "refund"
    if any(word in text for word in ("投诉", "不满")):
        return "complaint"
    if any(word in text for word in ("发票", "账单", "开票")):
        return "billing"
    if any(word in text for word in ("登录", "账号", "密码", "权限")):
        return "account"
    if any(word in text for word in ("报障", "故障", "报错", "异常")):
        return "technical"
    return "feature"


class _MockToolCallingModel:
    """
    离线确定性模型替身，用于在无 API Key 时验证状态机行为。

    它**不**做任何真实推理，仅按确定性规则模拟 ``qwen-plus`` 的两种输出：
        1. 需要检索/办理时，产出带 ``tool_calls`` 的 ``AIMessage``；
        2. 槽位不齐时，产出纯自然语言反问（不调用工具）。

    该替身严格遵守与 System Prompt 相同的槽位规则，因此可用于验证
    「缺参不调用工具」这一关键业务约束。
    """

    def __init__(self) -> None:
        # bind_tools 在真实链路中由基类提供；此处声明实例属性以保证接口一致
        self._bound_tools = TOOLS

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "_MockToolCallingModel":
        """模拟 ``bind_tools``，返回自身以保持链式调用兼容。"""
        self._bound_tools = list(tools)
        return self

    def invoke(self, messages: Sequence[Any], **kwargs: Any) -> AIMessage:
        """
        依据消息历史产出确定性的下一跳动作。

        Args:
            messages: 含 SystemMessage 的完整消息序列。

        Returns:
            AIMessage: 含 tool_calls 的决策，或不含 tool_calls 的自然语言回复。
        """
        last = messages[-1]

        # 情况 A：刚收到工具结果 → 转写为面向客户的自然语言汇报
        if isinstance(last, ToolMessage):
            return AIMessage(content=self._summarize_tool_result(last))

        # 收集历史用户原话：最近一轮用于提取参数，累计文本用于意图判定。
        # 真实模型能看到完整历史；mock 同样跨轮累积，才能正确模拟
        # 「第 1 轮反问 → 第 2 轮仅补账号 → 立即提交」的槽位填充过程。
        human_texts = [
            _normalize_content(m.content)
            for m in messages
            if isinstance(m, HumanMessage)
        ]
        user_text = human_texts[-1] if human_texts else ""
        cumulative_text = "\n".join(human_texts)

        # 情况 B：知识类问题 → 调用知识库检索
        if any(k in user_text for k in ("403", "401", "权限", "登录", "密码", "锁定", "退款政策", "政策")):
            if not any(k in user_text for k in ("帮我提", "申请退", "提交", "报障", "投诉")):
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "query_knowledge_base",
                            "args": {"query": user_text},
                            "id": "call_mock_kb_1",
                            "type": "tool_call",
                        }
                    ],
                )

        # 情况 C：办理类意图 → 槽位填充校验（跨轮累积判定，模拟槽位填充）
        if any(k in cumulative_text for k in _TICKET_INTENT_KEYWORDS):
            user_id = _extract_user_id(cumulative_text)
            missing: list[str] = []
            if not user_id:
                missing.append("user_id（您的账号 ID，形如 U-987654）")
            if not self._has_meaningful_description(cumulative_text):
                missing.append("description（购买时间与具体诉求/原因）")

            if missing:
                # 关键约束：缺参时绝不调用工具，改为亲切反问
                joined = "；".join(missing)
                return AIMessage(
                    content=(
                        "非常理解您希望尽快处理这件事，为了准确为您提交工单，"
                        f"还需要您补充以下信息：{joined}。\n"
                        "补充后我会立即为您提交并反馈工单号；"
                        "这些信息仅用于定位账号与加速处理，不会对外披露。"
                    )
                )

            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_ticket",
                        "args": {
                            "user_id": user_id,
                            "issue_type": _classify_issue_type(cumulative_text),
                            # 描述取跨轮累积文本，避免仅用「账号是 U-xxx」这类
                            # 补充语句导致工单描述丢失原始诉求与原因。
                            "description": cumulative_text,
                        },
                        "id": "call_mock_ticket_1",
                        "type": "tool_call",
                    }
                ],
            )

        # 兜底：礼貌澄清
        return AIMessage(
            content=(
                "感谢您的咨询。为更准确地协助您，请补充说明具体问题"
                "（例如错误码、账号 ID 或您希望办理的事项），我会立刻为您跟进。"
            )
        )

    @staticmethod
    def _has_meaningful_description(text: str) -> bool:
        """
        判断用户描述是否足够构成工单描述。

        判定为两步，避免只覆盖某一类场景：

        1. 长度达标（过滤「太短」这类无信息量的输入）；
        2. 能识别出**诉求** —— 或命中通用办理类意图关键词（投诉/报障/退款…），
           或命中描述性细节词（订单号/时间/规格等）。

        早期实现只用了「购买/误选/订单」这类**退款专有**细节词，导致
        「对服务不满，要投诉」这种描述完整但非退款的诉求被误判为缺失，
        进而错误地触发反问而**不提交工单**。此处改为覆盖全部工单类型。

        Args:
            text: 用户输入（多轮场景下为累积文本）。

        Returns:
            bool: 描述足以构成工单时返回 True。
        """
        if len(text) < 12:
            return False

        # 诉求信号一：通用办理类意图关键词，覆盖退款/投诉/报障/账单等全部枚举
        if any(keyword in text for keyword in _TICKET_INTENT_KEYWORDS):
            return True

        # 诉求信号二：描述性细节词（用于「没写明办理动词、但有具体线索」的输入）
        detail_markers = (
            "购买", "买错", "误选", "订单", "年费", "套餐", "双份", "扣费",
            "付款", "支付", "发票", "账单", "登录", "密码", "权限", "故障",
        )
        return any(marker in text for marker in detail_markers)

    @staticmethod
    def _summarize_tool_result(message: ToolMessage) -> str:
        """
        将工具返回结果转写为面向客户的自然语言。

        Args:
            message: 工具返回的 ToolMessage。

        Returns:
            str: 自然语言汇报文本。
        """
        raw = _normalize_content(message.content)
        name = getattr(message, "name", "")

        if name == "submit_ticket":
            try:
                payload = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                payload = {}
            if payload.get("success"):
                routing = payload.get("routing", {})
                return (
                    f"已为您成功提交工单，工单号 **{payload.get('ticket_id')}**，"
                    f"当前状态「{payload.get('status')}」，已流转至{routing.get('assigned_group')}，"
                    f"{routing.get('sla')}。请留意后续联系，如需加急可回复工单号。"
                )
            return (
                f"很抱歉，工单未能创建：{payload.get('error', '参数校验未通过')}。"
                "请补充必要信息后我再为您提交。"
            )

        if name == "query_knowledge_base":
            # 真实模型会基于检索结果重写为精炼答复；mock 仅做可读性裁剪，
            # 提取命中最高的片段要点，避免把整段知识库原样抛给客户。
            first_block = raw.split("--- 片段 1", 1)
            body = first_block[1] if len(first_block) > 1 else raw
            source = "知识库文档"
            source_match = re.search(r"来源：([\w.\-]+)", body)
            if source_match:
                source = source_match.group(1)
            # 去掉分隔符与空行，取前若干条要点
            lines = [
                line.strip()
                for line in body.splitlines()
                if line.strip() and not line.strip().startswith("---")
            ]
            bullets = lines[1:8] if len(lines) > 1 else lines
            detail = "\n".join(f"- {bullet}" for bullet in bullets)
            return (
                f"已为您检索到相关处理依据（来源：{source}）：\n{detail}\n"
                "以上建议依据内部知识库文档整理；如需我协助提报工单，请告知您的账号 ID。"
            )

        return f"工具 {name} 已返回结果：{raw[:300]}"


# ---------------------------------------------------------------------------
# 测试入口
# ---------------------------------------------------------------------------

TEST_SCENARIOS: list[tuple[str, str]] = [
    (
        "场景 1 · 知识库问答（预期：调用 query_knowledge_base 并基于文档作答）",
        "你好，系统登录提示 403 权限不足怎么排查解决？",
    ),
    (
        "场景 2 · 槽位缺失反问（预期：不调用任何工具，自然语言反问补充 user_id 与 description）",
        "我买错套餐了，帮我提个退费申请。",
    ),
    (
        "场景 3 · 参数齐备执行（预期：调用 submit_ticket 完成业务闭环）",
        "账号是 U-987654，我要申请退款，昨天购买的年费企业版误选了双份，申请退订一份。",
    ),
]


def main() -> int:
    """
    依次运行三个测试场景并流式打印状态机流转日志。

    Returns:
        int: 进程退出码，0 表示全部场景执行完毕。
    """
    use_mock = os.getenv("CUSTOMER_AGENT_MOCK", "").strip().lower() in {"1", "true", "yes"}
    mode = "离线 MOCK（不调用真实模型）" if use_mock else f"真实 DashScope · {MODEL_NAME}"

    print("=" * 78)
    print("企业级智能客服与工单协同中枢 · LangGraph 状态机测试")
    print(f"运行模式：{mode}")
    print(f"温度：{TEMPERATURE} | 绑定工具：{[t.name for t in TOOLS]}")
    print("=" * 78)

    try:
        model = build_model(mock=use_mock)
    except RuntimeError as error:
        print(f"\n[启动失败] {error}\n")
        return 2

    # MemorySaver 让每个场景拥有独立线程的多轮上下文
    from langgraph.checkpoint.memory import MemorySaver

    app = build_graph(model=model, checkpointer=MemorySaver())

    for index, (title, question) in enumerate(TEST_SCENARIOS, start=1):
        print(f"\n{'─' * 78}")
        print(f"【{title}】")
        print(f"用户：{question}")
        print(f"{'─' * 78}")
        try:
            messages = run_scenario(
                app, question, thread_id=f"scenario-{index}", verbose_steps=True
            )
        except Exception as error:  # noqa: BLE001 - 测试入口需完整暴露失败原因
            print(f"  [场景异常] {type(error).__name__}: {error}")
            continue

        final_text = ""
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                candidate = _normalize_content(message.content).strip()
                if candidate and not getattr(message, "tool_calls", None):
                    final_text = candidate
                    break
        print(f"\n  ▶ 最终答复：{final_text or '(无最终自然语言答复)'}")

    print(f"\n{'=' * 78}")
    print("全部场景执行完毕。")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
