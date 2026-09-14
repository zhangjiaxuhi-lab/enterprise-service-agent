"""
请求上下文（基于 ``contextvars``）。

作用
----
让 ``request_id`` 与 ``thread_id`` **自动**出现在该请求产生的每一条日志中，
无需逐层传参。``contextvars`` 在 asyncio 下按任务隔离，因此并发请求之间
不会串号；SSE 生成器作为同一任务的分支，同样能读到上下文。

用法::

    token = bind_request_context(request_id="abc123", thread_id="U-1")
    try:
        logger.info("处理中")          # 自动带上 request_id=abc123
    finally:
        reset_request_context(token)
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any

#: 哨兵：区分「未传参」与「显式传 None」
_UNSET: Any = object()

# ---------------------------------------------------------------------------
# 上下文变量
# ---------------------------------------------------------------------------

_request_id: ContextVar[str] = ContextVar("request_id", default="-")
_thread_id: ContextVar[str] = ContextVar("thread_id", default="-")
#: 附加字段（如 user_id、issue_type），用于在日志中补充业务维度
_extra_fields: ContextVar["dict[str, str] | None"] = ContextVar(
    "extra_fields", default=None
)


def new_request_id() -> str:
    """
    生成一个新的请求 ID。

    Returns:
        str: 形如 ``req-3f9a2c1b8d47`` 的短 ID（便于在日志中肉眼比对）。
    """
    return f"req-{uuid.uuid4().hex[:12]}"


@dataclass(frozen=True)
class ContextSnapshot:
    """上下文快照，便于在日志或响应头中一次性取出。"""

    request_id: str
    thread_id: str
    extra: dict[str, str]


def get_request_id() -> str:
    """
    读取当前请求 ID。

    Returns:
        str: 请求 ID；未绑定时返回 ``"-"``。
    """
    return _request_id.get()


def get_thread_id() -> str:
    """
    读取当前会话线程 ID。

    Returns:
        str: 线程 ID；未绑定时返回 ``"-"``。
    """
    return _thread_id.get()


def get_context() -> ContextSnapshot:
    """
    读取完整上下文快照。

    Returns:
        ContextSnapshot: 当前 request_id、thread_id 与附加字段。
    """
    return ContextSnapshot(
        request_id=_request_id.get(),
        thread_id=_thread_id.get(),
        extra=dict(_extra_fields.get() or {}),
    )


def bind_request_context(
    request_id: Any = _UNSET,
    thread_id: str | None = None,
    **extra: str,
) -> list[Token]:
    """
    绑定请求上下文。

    ``request_id`` 的三种取值语义：

    * **不传**（默认）：自动生成一个新的请求 ID —— 这是服务端入口的常见用法；
    * 传入字符串：使用指定 ID（如复用调用方传来的 ``X-Request-ID``）；
    * 显式传 ``None``：**不改动**当前 ID —— 用于嵌套绑定，避免覆盖外层请求 ID。

    Args:
        request_id: 请求 ID，语义见上。
        thread_id: 会话线程 ID；为 None 时保持不变（便于嵌套绑定）。
        **extra: 附加业务字段（如 ``user_id="U-1"``）。

    Returns:
        list[Token]: 用于恢复的 token 列表，交给 :func:`reset_request_context`。
    """
    tokens: list[Token] = []
    if request_id is _UNSET:
        tokens.append(_request_id.set(new_request_id()))
    elif request_id is not None:
        tokens.append(_request_id.set(str(request_id)))

    if thread_id is not None:
        tokens.append(_thread_id.set(thread_id))
    if extra:
        merged = dict(_extra_fields.get() or {})
        merged.update({k: str(v) for k, v in extra.items()})
        tokens.append(_extra_fields.set(merged))
    return tokens


def reset_request_context(tokens: list[Token]) -> None:
    """
    恢复上下文（按绑定的逆序回滚）。

    幂等：重复调用同一批 token 是安全的。标准库在 token 已被消费时会抛
    ``RuntimeError``（而非 ``ValueError``），两者都需捕获——否则在
    ``finally`` 中重复重置会把正常流程带崩。

    Args:
        tokens: :func:`bind_request_context` 返回的 token 列表。
    """
    for token in reversed(tokens):
        try:
            token.var.reset(token)
        except (ValueError, RuntimeError):
            # token 已被重置或跨上下文使用；忽略即可，不影响主流程
            pass
