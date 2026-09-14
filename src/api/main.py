"""
企业级智能客服与工单协同中枢 —— FastAPI + SSE 流式 API 服务（第三阶段）。

本模块把第二阶段编译好的 LangGraph 状态机包装为高性能流式 HTTP 接口，
对外暴露 ``POST /api/chat/stream``，以 Server-Sent Events 协议逐事件推送
推理过程与工具调用生命周期。

事件协议（每条均为 ``data: {json}\\n\\n``）::

    {"type": "token",      "content": "..."}          # 模型增量文本
    {"type": "tool_start", "tool": "工具名", "args": {…}}   # 调用工具前
    {"type": "tool_end",   "tool": "工具名", "output": "…"} # 工具执行完
    {"type": "error",      "message": "..."}          # 异常兜底
    {"type": "done",       "thread_id": "..."}        # 正常结束

流式数据来源说明（LangGraph 多模式流）：
    * ``stream_mode="messages"`` —— 产出 ``(message_chunk, metadata)``，
      用于逐 token 推送。仅转发**面向用户的最终答复**（``agent`` 节点），
      工具原始输出不逐字推送，避免知识库全文灌给前端。
    * ``stream_mode="updates"``  —— 产出 ``{节点名: 状态增量}``，
      用于识别工具调用的起止（``tool_start`` / ``tool_end``）。

运行::

    python -m src.api.main                # 或 uvicorn src.api.main:app --port 8000
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 路径与 .env 加载（与 agent 模块保持一致）
# ---------------------------------------------------------------------------

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=False)

from src.agent.customer_agent import (  # noqa: E402
    NODE_AGENT,
    TOOLS,
    AgentState,
    build_graph,
    build_model,
)

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

API_HOST: str = "0.0.0.0"
API_PORT: int = 8000

# SSE 响应头：禁用各类缓冲，保证事件实时到达客户端
SSE_HEADERS: dict[str, str] = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    # 关闭 Nginx 等反向代理的响应缓冲
    "X-Accel-Buffering": "no",
}


# ---------------------------------------------------------------------------
# 全局应用状态（图的单例持有者）
# ---------------------------------------------------------------------------


class _AppState:
    """
    持有编译后的图实例与持久化检查点。

    图在 FastAPI ``lifespan`` 启动阶段构建一次并复用，避免每个请求
    重复编译图（编译本身有开销，且能保证 MemorySaver 会话不丢失）。
    """

    def __init__(self) -> None:
        self.graph: Any | None = None
        self.checkpointer: Any | None = None
        self.init_error: str | None = None

    @property
    def ready(self) -> bool:
        """图是否已成功构建。"""
        return self.graph is not None


state = _AppState()


def _use_mock() -> bool:
    """是否使用离线 mock 模型（由环境变量控制，便于无 Key 验证）。"""
    return os.getenv("CUSTOMER_AGENT_MOCK", "").strip().lower() in {"1", "true", "yes"}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    应用生命周期：启动时构建状态机，关闭时释放引用。

    构建失败（如缺少 API Key）不会阻止服务启动，而是记录错误并在
    请求时以 ``error`` 事件优雅告知调用方，便于前端拿到明确提示。
    """
    try:
        from langgraph.checkpoint.memory import MemorySaver

        state.checkpointer = MemorySaver()
        model = build_model(mock=_use_mock())
        state.graph = build_graph(model=model, checkpointer=state.checkpointer)
        mode = "mock" if _use_mock() else "dashscope"
        print(f"[启动] 客服智能体图构建成功（模型模式：{mode}）")
    except Exception as error:  # noqa: BLE001 - 启动阶段需捕获全部异常
        state.graph = None
        state.init_error = f"{type(error).__name__}: {error}"
        print(f"[启动] 图构建失败：{state.init_error}")

    try:
        yield
    finally:
        state.graph = None
        state.checkpointer = None


app = FastAPI(
    title="企业级智能客服与工单协同中枢",
    description="基于 LangGraph + FastAPI 的流式客服 API（SSE）",
    version="0.3.0",
    lifespan=lifespan,
)

# 全源 CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # 与 allow_origins=["*"] 搭配时必须为 False
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# ---------------------------------------------------------------------------
# 请求 / 响应模型
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """``POST /api/chat/stream`` 请求体。"""

    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="用户消息内容",
        examples=["系统登录提示 403 权限不足怎么排查解决？"],
    )
    thread_id: Optional[str] = Field(
        default=None,
        description="会话线程 ID；不传则自动生成，用于串联多轮上下文",
        examples=["U-987654-session-1"],
    )


class ChatSyncResponse(BaseModel):
    """``POST /api/chat``（非流式）响应体，便于脚本化调试。"""

    thread_id: str
    reply: str


# ---------------------------------------------------------------------------
# SSE 事件编码
# ---------------------------------------------------------------------------


def sse_event(payload: dict[str, Any]) -> str:
    """
    将字典编码为一条 SSE 事件。

    Args:
        payload: 事件负载，需含 ``type`` 字段。

    Returns:
        str: 形如 ``data: {...}\\n\\n`` 的 SSE 报文。
    """
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _normalize_content(content: Any) -> str:
    """
    归一化消息内容为纯文本（兼容字符串与内容块列表两种形态）。

    Args:
        content: ``message.content``。

    Returns:
        str: 可推送文本。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
        return "".join(parts)
    return str(content)


# ---------------------------------------------------------------------------
# 核心：SSE 事件生成器
# ---------------------------------------------------------------------------


async def event_stream(
    graph: Any,
    message: str,
    thread_id: str,
) -> AsyncIterator[str]:
    """
    驱动状态机并把流转过程转换为 SSE 事件流。

    事件顺序典型为::

        token… → tool_start → tool_end → token… → done

    实现要点：
        * ``stream_mode=["messages", "updates"]`` 一次遍历同时拿到
          token 增量与节点级状态增量，无需重复执行图；
        * token 仅在 ``agent`` 节点产出时推送，工具原始输出不逐字下发；
        * 若某轮未产生 token 流（例如模型一次性返回、或非流式模型），
          则在 ``updates`` 阶段补推整段内容，避免答复丢失；
        * 以消息 ``id`` 去重，防止补推与已流式内容重复。

    Args:
        graph: 已编译的 LangGraph 应用。
        message: 用户消息。
        thread_id: 会话线程 ID。

    Yields:
        str: SSE 报文片段。
    """
    config = {"configurable": {"thread_id": thread_id}}
    # 已推送过内容的 AI 消息 id，用于去重补推
    streamed_ai_ids: set[str] = set()

    try:
        async for chunk in graph.astream(
            {"messages": [HumanMessage(content=message)]},
            config=config,
            stream_mode=["messages", "updates"],
        ):
            # 多模式流输出统一为 (mode, payload) 二元组
            if not (isinstance(chunk, tuple) and len(chunk) == 2):
                continue
            mode, payload = chunk

            # ---------------- 模式一：逐 token 增量 ----------------
            if mode == "messages":
                chunk_message, metadata = payload
                node = (metadata or {}).get("langgraph_node")
                # 只转发最终答复节点的文本；ToolMessage 等不入 token 流
                if node != NODE_AGENT or not isinstance(chunk_message, AIMessage):
                    continue
                text = _normalize_content(chunk_message.content)
                if not text:
                    continue
                message_id = getattr(chunk_message, "id", None)
                if message_id:
                    streamed_ai_ids.add(message_id)
                yield sse_event({"type": "token", "content": text})
                continue

            # ---------------- 模式二：节点级状态增量 ----------------
            if mode != "updates" or not isinstance(payload, dict):
                continue

            for node_name, update in payload.items():
                if node_name.startswith("__") or not isinstance(update, dict):
                    continue
                for item in update.get("messages", []) or []:
                    # 工具执行结果 → tool_end
                    if isinstance(item, ToolMessage):
                        yield sse_event(
                            {
                                "type": "tool_end",
                                "tool": getattr(item, "name", None) or "tool",
                                "output": _normalize_content(item.content),
                            }
                        )
                        continue

                    if not isinstance(item, AIMessage):
                        continue

                    # 工具调用请求 → tool_start（可能一轮多个）
                    for call in getattr(item, "tool_calls", None) or []:
                        yield sse_event(
                            {
                                "type": "tool_start",
                                "tool": call.get("name", "unknown"),
                                "args": call.get("args", {}) or {},
                            }
                        )

                    # 补推：该轮没有 token 流时，整段下发，保证不丢内容
                    message_id = getattr(item, "id", None)
                    text = _normalize_content(item.content)
                    if text and message_id not in streamed_ai_ids:
                        streamed_ai_ids.add(message_id or "")
                        yield sse_event({"type": "token", "content": text})

        yield sse_event({"type": "done", "thread_id": thread_id})

    except Exception as error:  # noqa: BLE001 - 流式中任何异常都要以 error 事件收尾
        yield sse_event(
            {
                "type": "error",
                "message": f"{type(error).__name__}: {error}",
                "thread_id": thread_id,
            }
        )


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------


@app.get("/health", summary="健康检查")
async def health() -> dict[str, Any]:
    """
    返回服务与状态机就绪情况，便于容器探针与联调自检。

    Returns:
        dict: 包含 ``status``、``graph_ready``、``tools``、``init_error``。
    """
    return {
        "status": "ok" if state.ready else "degraded",
        "graph_ready": state.ready,
        "model_mode": "mock" if _use_mock() else "dashscope",
        "tools": [tool.name for tool in TOOLS],
        "init_error": state.init_error,
    }


@app.post("/api/chat/stream", summary="流式对话（SSE）")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    以 Server-Sent Events 流式返回客服智能体的推理与工具调用过程。

    Args:
        request: 含 ``message`` 与可选 ``thread_id``。

    Returns:
        StreamingResponse: ``text/event-stream`` 事件流。
    """
    if not request.message.strip():
        raise HTTPException(status_code=422, detail="message 不能为空")

    thread_id = (request.thread_id or "").strip() or f"session-{uuid.uuid4().hex[:12]}"

    # 图未就绪时，以单条 error 事件返回（HTTP 仍为 200，便于前端统一处理）
    if not state.ready:
        async def _degraded() -> AsyncIterator[str]:
            yield sse_event(
                {
                    "type": "error",
                    "message": state.init_error or "智能体图尚未就绪，请检查服务启动日志。",
                    "thread_id": thread_id,
                }
            )
            yield sse_event({"type": "done", "thread_id": thread_id})

        return StreamingResponse(
            _degraded(), media_type="text/event-stream", headers=SSE_HEADERS
        )

    return StreamingResponse(
        event_stream(state.graph, request.message, thread_id),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@app.post("/api/chat", response_model=ChatSyncResponse, summary="非流式对话")
async def chat_sync(request: ChatRequest) -> ChatSyncResponse:
    """
    非流式对话接口，便于脚本化回归测试。

    Args:
        request: 含 ``message`` 与可选 ``thread_id``。

    Returns:
        ChatSyncResponse: 最终自然语言答复。
    """
    if not state.ready:
        raise HTTPException(
            status_code=503,
            detail=state.init_error or "智能体图尚未就绪",
        )

    thread_id = (request.thread_id or "").strip() or f"session-{uuid.uuid4().hex[:12]}"
    config = {"configurable": {"thread_id": thread_id}}
    result = await state.graph.ainvoke(
        {"messages": [HumanMessage(content=request.message)]}, config=config
    )

    reply = ""
    for item in reversed(result.get("messages", [])):
        if isinstance(item, AIMessage) and not getattr(item, "tool_calls", None):
            candidate = _normalize_content(item.content).strip()
            if candidate:
                reply = candidate
                break
    return ChatSyncResponse(thread_id=thread_id, reply=reply)


@app.get("/api/chat/history/{thread_id}", summary="查询会话历史")
async def chat_history(thread_id: str) -> dict[str, Any]:
    """
    读取指定线程的消息历史，用于验证多轮上下文是否被正确保留。

    Args:
        thread_id: 会话线程 ID。

    Returns:
        dict: 含角色与内容的精简消息列表。
    """
    if not state.ready:
        raise HTTPException(
            status_code=503, detail=state.init_error or "智能体图尚未就绪"
        )

    config = {"configurable": {"thread_id": thread_id}}
    snapshot = state.graph.get_state(config)
    raw_messages: list[BaseMessage] = list(snapshot.values.get("messages", []))

    def _role(item: BaseMessage) -> str:
        if isinstance(item, HumanMessage):
            return "user"
        if isinstance(item, ToolMessage):
            return "tool"
        if isinstance(item, AIMessage):
            return "assistant"
        return "system"

    return {
        "thread_id": thread_id,
        "count": len(raw_messages),
        "messages": [
            {
                "role": _role(item),
                "content": _normalize_content(item.content),
                "tool_calls": [
                    {"name": call.get("name"), "args": call.get("args")}
                    for call in (getattr(item, "tool_calls", None) or [])
                ],
            }
            for item in raw_messages
        ],
    }


def main() -> int:
    """
    以 uvicorn 启动服务（端口 8000）。

    Returns:
        int: 进程退出码。
    """
    import uvicorn

    print(f"启动服务：http://127.0.0.1:{API_PORT}  (文档：/docs)")
    uvicorn.run(
        "src.api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
