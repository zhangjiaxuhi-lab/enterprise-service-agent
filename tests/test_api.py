"""
API 层测试（``src/api/main.py``）。

两层验证策略
------------
* **路由与页面**：用 ``TestClient`` 在进程内验证 HTTP 语义（状态码、媒体类型、404）。
* **SSE 流**：直接驱动 ``event_stream`` 异步生成器，避免 TestClient 对
  流式响应体的缓冲，从而能逐帧校验事件协议。

SSE 事件协议是本项目的对外契约，因此单独设一组测试守住它。
"""

from __future__ import annotations

import json
from typing import Any

import anyio
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage

from src.api.main import INDEX_FILE, STATIC_DIR, app, event_stream

# ---------------------------------------------------------------------------
# HTTP 路由
# ---------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Any:
    """
    进程内 HTTP 客户端（mock 模型模式，不联网）。

    Args:
        monkeypatch: pytest 环境变量补丁。

    Yields:
        TestClient: 已进入 lifespan 的客户端。
    """
    monkeypatch.setenv("CUSTOMER_AGENT_MOCK", "1")
    with TestClient(app) as test_client:
        yield test_client


class TestStaticAndRoutes:
    """静态页面与探针路由。"""

    def test_index_serves_workbench(self, client: Any) -> None:
        """根路径返回工作台页面。"""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        body = response.text
        for marker in ["智能客服中枢", "qwen-plus", "tool_start", "ReadableStream"]:
            assert marker in body, f"页面缺少标记：{marker}"

    def test_static_mount_serves_index(self, client: Any) -> None:
        """静态目录挂载生效。"""
        assert STATIC_DIR.is_dir() and INDEX_FILE.is_file()
        response = client.get("/static/index.html")
        assert response.status_code == 200

    def test_static_missing_file_is_404(self, client: Any) -> None:
        """静态目录下不存在的文件返回 404。"""
        assert client.get("/static/definitely-missing.html").status_code == 404

    def test_health_reports_ready(self, client: Any) -> None:
        """健康探针应报告图已就绪、工具列表与运行模式。"""
        payload = client.get("/health").json()
        assert payload["graph_ready"] is True
        assert payload["model_mode"] == "mock"
        assert set(payload["tools"]) == {"query_knowledge_base", "submit_ticket"}
        assert payload["init_error"] is None

    def test_vue_workbench_not_shadowing_api(self, client: Any) -> None:
        """确认静态挂载未遮蔽 API 路由（根路径与 /api 共存）。"""
        assert client.get("/health").status_code == 200
        assert client.post("/api/chat", json={"message": "你好"}).status_code == 200


# ---------------------------------------------------------------------------
# 请求校验
# ---------------------------------------------------------------------------


class TestRequestValidation:
    """入参校验。"""

    def test_empty_message_rejected(self, client: Any) -> None:
        """空 message 应返回 422。"""
        assert client.post("/api/chat/stream", json={"message": ""}).status_code == 422

    def test_missing_message_rejected(self, client: Any) -> None:
        """缺少 message 字段应返回 422。"""
        assert client.post("/api/chat/stream", json={}).status_code == 422

    def test_overlong_message_rejected(self, client: Any) -> None:
        """超过 4000 字符应返回 422。"""
        response = client.post("/api/chat/stream", json={"message": "啊" * 4001})
        assert response.status_code == 422

    def test_thread_id_is_optional(self, client: Any) -> None:
        """thread_id 可选，不传时应自动生成。"""
        payload = client.post("/api/chat", json={"message": "你好"}).json()
        assert payload["thread_id"].startswith("session-")


# ---------------------------------------------------------------------------
# SSE 事件协议
# ---------------------------------------------------------------------------


def _collect_events(graph: Any, message: str, thread_id: str = "api-test") -> list[dict]:
    """
    同步收集一次 SSE 流的全部事件。

    Args:
        graph: 已编译的图。
        message: 用户消息。
        thread_id: 会话线程 ID。

    Returns:
        list[dict]: 解析后的事件列表。
    """
    events: list[dict] = []

    async def _run() -> None:
        async for frame in event_stream(graph, message, thread_id):
            assert frame.startswith("data: "), f"SSE 帧格式错误：{frame[:40]!r}"
            assert frame.endswith("\n\n"), "SSE 帧未以空行结尾"
            events.append(json.loads(frame[len("data: "):].strip()))

    anyio.run(_run)
    return events


def _kinds(events: list[dict]) -> list[str]:
    """提取事件类型序列。"""
    return [e.get("type") for e in events]


class TestSSEProtocol:
    """SSE 事件协议契约。"""

    @pytest.fixture
    def graph(self, mock_graph: Any) -> Any:
        """使用 mock 模型的图。"""
        return mock_graph

    def test_knowledge_flow_event_order(self, graph: Any) -> None:
        """知识库问答：tool_start → tool_end → token → done。"""
        events = _collect_events(graph, "403 权限不足怎么解决？")
        kinds = _kinds(events)
        assert kinds[0] == "tool_start"
        assert "tool_end" in kinds
        assert "token" in kinds
        assert kinds[-1] == "done", f"最后一个事件应为 done，实际 {kinds[-1]}"

    def test_missing_slot_has_no_tool_events(self, graph: Any) -> None:
        """缺参反问：不得出现任何工具事件（对外契约级约束）。"""
        events = _collect_events(graph, "我买错套餐了，帮我提个退费申请。")
        kinds = _kinds(events)
        assert "tool_start" not in kinds, f"缺参却推送了工具事件：{kinds}"
        assert "tool_end" not in kinds
        assert "token" in kinds
        assert kinds[-1] == "done"

    def test_ticket_flow_events_carry_required_fields(self, graph: Any) -> None:
        """工单流程：tool_start 带 tool+args；tool_end 带 tool+output。"""
        events = _collect_events(
            graph,
            "账号是 U-987654，我要申请退款，昨天购买的年费企业版误选了双份，申请退订一份。",
        )
        starts = [e for e in events if e["type"] == "tool_start"]
        ends = [e for e in events if e["type"] == "tool_end"]
        assert starts and ends

        assert starts[0]["tool"] == "submit_ticket"
        assert isinstance(starts[0]["args"], dict)
        assert starts[0]["args"]["user_id"] == "U-987654"

        assert ends[0]["tool"] == "submit_ticket"
        output = json.loads(ends[0]["output"])
        assert output["success"] is True

    def test_done_event_carries_thread_id(self, graph: Any) -> None:
        """done 事件应回传 thread_id，便于前端续接会话。"""
        events = _collect_events(graph, "你好", thread_id="tid-xyz")
        done = [e for e in events if e["type"] == "done"]
        assert done and done[0]["thread_id"] == "tid-xyz"

    def test_every_event_has_type_field(self, graph: Any) -> None:
        """所有事件都必须带 type 字段（前端据此分派）。"""
        events = _collect_events(graph, "403 权限不足怎么解决？")
        assert events
        for event in events:
            assert "type" in event, f"事件缺少 type：{event}"

    def test_token_events_are_non_empty(self, graph: Any) -> None:
        """token 事件不应携带空内容（否则前端会渲染空白）。"""
        events = _collect_events(graph, "403 权限不足怎么解决？")
        tokens = [e for e in events if e["type"] == "token"]
        assert tokens, "未收到任何 token 事件"
        for token in tokens:
            assert token["content"], "存在空 content 的 token 事件"

    def test_stream_error_is_caught(self) -> None:
        """上游异常时必须推送 error 事件，而不是让流静默断开。"""

        class _BoomGraph:
            async def astream(self, *args: Any, **kwargs: Any):
                raise RuntimeError("模拟上游超时")
                yield  # pragma: no cover - 仅为保持异步生成器语义

        events = _collect_events(_BoomGraph(), "你好")
        kinds = _kinds(events)
        assert "error" in kinds, f"异常未转换为 error 事件：{kinds}"
        payload = next(e for e in events if e["type"] == "error")
        assert "模拟上游超时" in payload["message"]


# ---------------------------------------------------------------------------
# 非流式接口与会话历史
# ---------------------------------------------------------------------------


class TestSyncAndHistory:
    """``/api/chat`` 与 ``/api/chat/history``。"""

    def test_sync_returns_reply(self, client: Any) -> None:
        """非流式接口返回最终自然语言答复。"""
        payload = client.post(
            "/api/chat", json={"message": "403 权限不足怎么解决？", "thread_id": "sync-1"}
        ).json()
        assert payload["thread_id"] == "sync-1"
        assert payload["reply"].strip(), "答复为空"

    def test_history_returns_roles_in_order(self, client: Any) -> None:
        """会话历史应包含 user / assistant（及工具消息）。"""
        client.post(
            "/api/chat",
            json={
                "message": "账号是 U-987654，我要申请退款，昨天买错了套餐。",
                "thread_id": "hist-1",
            },
        )
        payload = client.get("/api/chat/history/hist-1").json()
        assert payload["count"] > 0
        roles = [m["role"] for m in payload["messages"]]
        assert roles[0] == "user", f"首条应为 user，实际 {roles}"
        assert "assistant" in roles

    def test_history_of_unknown_thread_is_empty(self, client: Any) -> None:
        """未知 thread_id 应返回空历史，而不是报错。"""
        payload = client.get("/api/chat/history/never-existed").json()
        assert payload["count"] == 0
        assert payload["messages"] == []


# ---------------------------------------------------------------------------
# 前端契约
# ---------------------------------------------------------------------------


class TestFrontendContract:
    """前端依赖的接口契约（防止后端改动悄悄破坏页面）。"""

    def test_page_fetches_documented_endpoints(self) -> None:
        """页面引用的接口路径必须真实存在。"""
        html = INDEX_FILE.read_text(encoding="utf-8")
        for endpoint in ["/api/chat/stream", "/health"]:
            assert endpoint in html, f"页面未引用 {endpoint}"

    def test_page_handles_all_sse_event_types(self) -> None:
        """页面必须处理全部五类 SSE 事件。"""
        html = INDEX_FILE.read_text(encoding="utf-8")
        for event_type in ["token", "tool_start", "tool_end", "error", "done"]:
            assert f'"{event_type}"' in html, f"页面未处理事件类型：{event_type}"
