"""
可观测性测试（``src/observability/``）。

覆盖三块：

1. **请求上下文** —— contextvars 的绑定/重置与并发隔离；
2. **结构化日志** —— JSON / 控制台格式、上下文注入、附加字段；
3. **链路追踪** —— 开关判定、变量兼容、trace 元数据构造。
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.observability.context import (
    bind_request_context,
    get_context,
    get_request_id,
    get_thread_id,
    new_request_id,
    reset_request_context,
)
from src.observability.logging import (
    ContextFormatter,
    get_logger,
    resolve_format,
    resolve_level,
    setup_logging,
)
from src.observability.tracing import (
    API_KEY_ENV,
    API_KEY_ENV_LEGACY,
    DEFAULT_PROJECT,
    PROJECT_ENV,
    TRACING_ENV,
    TRACING_ENV_LEGACY,
    configure_tracing,
    is_tracing_enabled,
    trace_config,
)

# ---------------------------------------------------------------------------
# 1) 请求上下文
# ---------------------------------------------------------------------------


class TestRequestContext:
    """contextvars 上下文行为。"""

    def test_defaults_when_unbound(self) -> None:
        """未绑定时返回占位值，而不是抛错。"""
        assert get_request_id() == "-"
        assert get_thread_id() == "-"
        assert get_context().extra == {}

    def test_new_request_id_is_unique(self) -> None:
        """生成的请求 ID 应唯一且带可识别前缀。"""
        ids = {new_request_id() for _ in range(50)}
        assert len(ids) == 50
        assert all(i.startswith("req-") for i in ids)

    def test_bind_and_reset(self) -> None:
        """绑定后可读取，重置后恢复原值。"""
        tokens = bind_request_context(request_id="req-abc", thread_id="U-1")
        try:
            assert get_request_id() == "req-abc"
            assert get_thread_id() == "U-1"
        finally:
            reset_request_context(tokens)
        assert get_request_id() == "-"
        assert get_thread_id() == "-"

    def test_auto_generated_request_id(self) -> None:
        """request_id 为 None 时自动生成。"""
        tokens = bind_request_context()
        try:
            assert get_request_id().startswith("req-")
        finally:
            reset_request_context(tokens)

    def test_thread_id_preserved_when_omitted(self) -> None:
        """省略 thread_id 时不应清空既有值（支持嵌套绑定）。"""
        outer = bind_request_context(request_id="req-1", thread_id="U-9")
        inner = bind_request_context(request_id="req-2")
        try:
            assert get_request_id() == "req-2"
            assert get_thread_id() == "U-9", "嵌套绑定不应清空 thread_id"
        finally:
            reset_request_context(inner)
            reset_request_context(outer)

    def test_extra_fields_merge(self) -> None:
        """附加字段应累加而非覆盖。"""
        a = bind_request_context(user_id="U-1")
        b = bind_request_context(issue_type="refund")
        try:
            extra = get_context().extra
            assert extra == {"user_id": "U-1", "issue_type": "refund"}
        finally:
            reset_request_context(b)
            reset_request_context(a)

    def test_reset_is_idempotent(self) -> None:
        """重复重置不应抛错。"""
        tokens = bind_request_context(request_id="req-x")
        reset_request_context(tokens)
        reset_request_context(tokens)  # 第二次应被安全忽略

    @pytest.mark.anyio
    async def test_context_isolated_between_tasks(self) -> None:
        """并发任务之间的上下文必须隔离（否则日志会串号）。"""
        import anyio

        results: dict[str, str] = {}

        async def worker(name: str) -> None:
            tokens = bind_request_context(request_id=f"req-{name}")
            try:
                await anyio.sleep(0.01)  # 制造交错
                results[name] = get_request_id()
            finally:
                reset_request_context(tokens)

        async with anyio.create_task_group() as tg:
            for name in ("a", "b", "c"):
                tg.start_soon(worker, name)

        assert results == {"a": "req-a", "b": "req-b", "c": "req-c"}


# ---------------------------------------------------------------------------
# 2) 结构化日志
# ---------------------------------------------------------------------------


class TestLoggingConfiguration:
    """日志级别与格式解析。"""

    def test_default_format_is_console(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """默认使用控制台格式。"""
        monkeypatch.delenv("LOG_FORMAT", raising=False)
        assert resolve_format() == "console"

    def test_json_format_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """可切换为 JSON。"""
        monkeypatch.setenv("LOG_FORMAT", "json")
        assert resolve_format() == "json"

    def test_invalid_format_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """非法格式回退为 console，而不是崩溃。"""
        monkeypatch.setenv("LOG_FORMAT", "xml")
        assert resolve_format() == "console"

    def test_level_resolution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """级别字符串应被正确解析。"""
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        assert resolve_level() == logging.DEBUG

    def test_invalid_level_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """非法级别回退为 INFO。"""
        monkeypatch.setenv("LOG_LEVEL", "VERBOSE")
        assert resolve_level() == logging.INFO

    def test_setup_logging_is_idempotent(self) -> None:
        """重复配置不应叠加 handler。"""
        setup_logging(force=True)
        logger = setup_logging()
        only = logging.getLogger("esa")
        assert len(only.handlers) == 1, "handler 被重复添加"
        assert logger is only
        assert only.propagate is False, "不应向 root 传播以免重复输出"


class TestLogFormatting:
    """两种格式的输出内容。"""

    @staticmethod
    def _record(message: str = "测试消息", **extra: Any) -> logging.LogRecord:
        """构造一条日志记录。"""
        record = logging.LogRecord(
            name="esa.test", level=logging.INFO, pathname=__file__,
            lineno=1, msg=message, args=(), exc_info=None,
        )
        for key, value in extra.items():
            setattr(record, key, value)
        return record

    def test_json_contains_context_and_extras(self) -> None:
        """JSON 格式应含基础字段、上下文与附加字段，且可被解析。"""
        tokens = bind_request_context(request_id="req-j", thread_id="U-7", user_id="U-7")
        try:
            output = ContextFormatter("json").format(
                self._record("工单已提交", tool="submit_ticket", duration_ms=12.5)
            )
        finally:
            reset_request_context(tokens)

        payload = json.loads(output)  # 必须是合法 JSON
        assert payload["level"] == "INFO"
        assert payload["message"] == "工单已提交"
        assert payload["request_id"] == "req-j"
        assert payload["thread_id"] == "U-7"
        assert payload["user_id"] == "U-7"
        assert payload["tool"] == "submit_ticket"
        assert payload["duration_ms"] == 12.5

    def test_json_output_is_single_line(self) -> None:
        """JSON 必须单行输出，否则采集器无法按行解析。"""
        output = ContextFormatter("json").format(self._record("多行\n消息"))
        assert "\n" not in output

    def test_json_preserves_chinese(self) -> None:
        """中文不应被转义为 \\uXXXX，便于直接检索。"""
        output = ContextFormatter("json").format(self._record("中文内容"))
        assert "中文内容" in output

    def test_console_contains_context_and_extras(self) -> None:
        """控制台格式应含上下文与附加字段。"""
        tokens = bind_request_context(request_id="req-c", thread_id="U-3")
        try:
            output = ContextFormatter("console").format(
                self._record("处理中", status=200)
            )
        finally:
            reset_request_context(tokens)

        assert "req-c" in output
        assert "U-3" in output
        assert "处理中" in output
        assert "status=200" in output

    def test_logger_records_are_formatted(self, capsys: pytest.CaptureFixture) -> None:
        """端到端：通过 logger 输出时上下文自动注入。"""
        setup_logging(level=logging.INFO, fmt="json", force=True)
        logger = get_logger("test")
        tokens = bind_request_context(request_id="req-e2e")
        try:
            logger.info("端到端日志", extra={"key": "value"})
        finally:
            reset_request_context(tokens)

        captured = capsys.readouterr()
        assert "req-e2e" in captured.err, "日志应输出到 stderr"
        assert captured.out == "", "日志不应污染 stdout"
        payload = json.loads(captured.err.strip().splitlines()[-1])
        assert payload["request_id"] == "req-e2e"
        assert payload["key"] == "value"

    def test_logger_name_namespace(self) -> None:
        """子日志器应位于 esa 命名空间下。"""
        assert get_logger("api").name == "esa.api"
        assert get_logger().name == "esa"


# ---------------------------------------------------------------------------
# 3) 链路追踪
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_tracing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """清理全部追踪相关环境变量，保证用例不受外部影响。"""
    for name in (
        TRACING_ENV, TRACING_ENV_LEGACY, API_KEY_ENV, API_KEY_ENV_LEGACY,
        PROJECT_ENV, "LANGCHAIN_PROJECT",
    ):
        monkeypatch.delenv(name, raising=False)


class TestTracingConfiguration:
    """追踪开关与变量兼容。"""

    def test_disabled_by_default(self, clean_tracing_env: None) -> None:
        """默认不启用（避免误上传内容）。"""
        assert is_tracing_enabled() is False
        status = configure_tracing()
        assert status["enabled"] is False
        assert status["reason"]

    def test_enabled_with_flag_and_key(self, clean_tracing_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """同时具备开关与 Key 时才启用。"""
        monkeypatch.setenv(TRACING_ENV, "true")
        monkeypatch.setenv(API_KEY_ENV, "lsv2-test-key")
        assert is_tracing_enabled() is True

        status = configure_tracing()
        assert status["enabled"] is True
        assert status["project"] == DEFAULT_PROJECT

    def test_flag_without_key_is_disabled(self, clean_tracing_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """只开开关但没 Key 时不应启用（否则会产生大量上传失败）。"""
        monkeypatch.setenv(TRACING_ENV, "true")
        assert is_tracing_enabled() is False
        assert "未配置" in configure_tracing()["reason"]

    def test_explicit_disable_wins(self, clean_tracing_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """显式关闭优先于存在 Key。"""
        monkeypatch.setenv(TRACING_ENV, "false")
        monkeypatch.setenv(API_KEY_ENV, "lsv2-test-key")
        assert is_tracing_enabled() is False

    def test_legacy_variables_supported(self, clean_tracing_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """兼容旧变量名 LANGCHAIN_*。"""
        monkeypatch.setenv(TRACING_ENV_LEGACY, "true")
        monkeypatch.setenv(API_KEY_ENV_LEGACY, "lsv2-legacy-key")
        assert is_tracing_enabled() is True

    def test_placeholder_key_not_treated_as_real(self, clean_tracing_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """占位 Key 不应被当作有效凭据。"""
        monkeypatch.setenv(TRACING_ENV, "true")
        monkeypatch.setenv(API_KEY_ENV, "your_key_here")
        assert is_tracing_enabled() is False

    def test_custom_project_respected(self, clean_tracing_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """自定义项目名应被采用。"""
        monkeypatch.setenv(TRACING_ENV, "true")
        monkeypatch.setenv(API_KEY_ENV, "lsv2-test-key")
        monkeypatch.setenv(PROJECT_ENV, "my-project")
        assert configure_tracing()["project"] == "my-project"


class TestTraceConfig:
    """trace 元数据构造。"""

    def test_metadata_and_run_name(self) -> None:
        """应携带 thread_id / request_id 与运行名。"""
        config = trace_config(thread_id="U-1", request_id="req-1", run_name="chat_stream")
        assert config["metadata"] == {"thread_id": "U-1", "request_id": "req-1"}
        assert config["run_name"] == "chat_stream"

    def test_empty_when_no_args(self) -> None:
        """无参数时返回空字典，便于安全展开合并。"""
        assert trace_config() == {}

    def test_partial_args(self) -> None:
        """仅提供部分参数时只包含对应字段。"""
        config = trace_config(thread_id="U-1")
        assert config["metadata"] == {"thread_id": "U-1"}
        assert "run_name" not in config


# ---------------------------------------------------------------------------
# 4) 中间件与 /health 集成
# ---------------------------------------------------------------------------


class TestMiddlewareIntegration:
    """请求 ID 透传与健康探针。"""

    @pytest.fixture
    def client(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        """进程内客户端（mock 模型 + 内存检查点）。"""
        monkeypatch.setenv("CUSTOMER_AGENT_MOCK", "1")
        monkeypatch.setenv("CHECKPOINT_BACKEND", "memory")
        from src.api.main import app

        with TestClient(app) as test_client:
            yield test_client

    def test_response_carries_request_id(self, client: Any) -> None:
        """响应头应回传 X-Request-ID，便于前端上报与日志检索。"""
        response = client.get("/health")
        assert response.headers.get("X-Request-ID", "").startswith("req-")

    def test_inbound_request_id_is_reused(self, client: Any) -> None:
        """调用方传入的 X-Request-ID 应被复用，实现跨服务串联。"""
        response = client.get("/health", headers={"X-Request-ID": "req-from-client"})
        assert response.headers["X-Request-ID"] == "req-from-client"

    def test_each_request_gets_distinct_id(self, client: Any) -> None:
        """不同请求应获得不同 ID。"""
        first = client.get("/health").headers["X-Request-ID"]
        second = client.get("/health").headers["X-Request-ID"]
        assert first != second

    def test_health_exposes_tracing_status(self, client: Any) -> None:
        """健康探针应报告追踪状态（含关闭原因）。"""
        payload = client.get("/health").json()
        assert "tracing" in payload
        assert payload["tracing"]["enabled"] is False
        assert payload["tracing"]["reason"]


class TestTraceMetadataReachesGraph:
    """确认 trace 元数据真的被传给了 LangGraph，而不只是「构造出来了」。"""

    @pytest.mark.anyio
    async def test_event_stream_passes_metadata(self) -> None:
        """``event_stream`` 应把 request_id / thread_id 放入 config 传给图。"""
        from src.api.main import event_stream

        captured: dict[str, Any] = {}

        class _SpyGraph:
            async def astream(self, inputs: Any, config: Any = None, **kwargs: Any):
                captured["config"] = config
                return
                yield  # pragma: no cover - 保持异步生成器语义

        tokens = bind_request_context(request_id="req-trace-1")
        try:
            async for _ in event_stream(_SpyGraph(), "你好", "thread-abc"):
                pass
        finally:
            reset_request_context(tokens)

        config = captured.get("config") or {}
        assert config.get("configurable", {}).get("thread_id") == "thread-abc"
        metadata = config.get("metadata", {})
        assert metadata.get("thread_id") == "thread-abc"
        assert metadata.get("request_id") == "req-trace-1", (
            "request_id 未进入 trace 元数据，LangSmith 中将无法按请求检索"
        )
        assert config.get("run_name") == "chat_stream"

    @pytest.mark.anyio
    async def test_metadata_absent_when_request_id_unbound(self) -> None:
        """未绑定上下文时仍应正常传入 thread_id（不得因缺 request_id 报错）。"""
        from src.api.main import event_stream

        captured: dict[str, Any] = {}

        class _SpyGraph:
            async def astream(self, inputs: Any, config: Any = None, **kwargs: Any):
                captured["config"] = config
                return
                yield  # pragma: no cover

        async for _ in event_stream(_SpyGraph(), "你好", "thread-xyz"):
            pass

        metadata = (captured.get("config") or {}).get("metadata", {})
        assert metadata.get("thread_id") == "thread-xyz"
        # request_id 此时为占位值 "-"，仍应存在但不影响运行
        assert "request_id" in metadata
