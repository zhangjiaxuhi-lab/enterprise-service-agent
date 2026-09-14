"""
可观测性支撑层：结构化日志、请求上下文与链路追踪。

包含三个模块：

* ``context``  —— 基于 ``contextvars`` 的请求上下文（request_id / thread_id）；
* ``logging``  —— 结构化日志配置（JSON / 控制台双格式，自动注入上下文）；
* ``tracing``  —— LangSmith / LangChain 链路追踪的可选接入。

设计原则
--------
1. **默认开箱可用，且不产生噪音**：日志默认输出到 stderr，追踪默认关闭
   （仅在显式配置了 API Key 时启用）。
2. **零额外依赖**：日志基于标准库 ``logging``，追踪复用已安装的 langsmith。
3. **上下文自动透传**：``request_id`` / ``thread_id`` 通过 ``contextvars``
   传递，无需在每处调用点手动传参；异步任务与 SSE 生成器内同样可见。
"""

from __future__ import annotations

from src.observability.context import (
    bind_request_context,
    get_request_id,
    get_thread_id,
    new_request_id,
    reset_request_context,
)
from src.observability.logging import get_logger, setup_logging
from src.observability.tracing import (
    configure_tracing,
    is_tracing_enabled,
    trace_config,
)

__all__ = [
    "bind_request_context",
    "get_request_id",
    "get_thread_id",
    "new_request_id",
    "reset_request_context",
    "get_logger",
    "setup_logging",
    "configure_tracing",
    "is_tracing_enabled",
    "trace_config",
]
