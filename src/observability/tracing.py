"""
链路追踪接入（LangSmith / LangChain）。

定位
----
结构化日志能回答「发生了什么」，但回答不了「**模型那一步为什么这么决策**」——
例如某次对话为何选了 ``refund`` 而不是 ``complaint``、耗时花在哪一步、
这一轮烧了多少 token。这需要**链路追踪**。

设计
----
* **可选接入**：仅在显式配置了 API Key 时才启用；未配置时完全不产生开销，
  也不会因为缺少凭据而报错。CI 与本地默认不启用。
* **零额外依赖**：复用 ``langsmith``（已作为 langchain 的依赖安装）。
* **元数据可检索**：把 ``request_id`` / ``thread_id`` 作为 trace 的
  metadata，便于在 LangSmith 里按会话或请求精确定位。

启用方式::

    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=lsv2_...
    LANGSMITH_PROJECT=enterprise-service-agent

兼容旧变量名 ``LANGCHAIN_TRACING_V2`` / ``LANGCHAIN_API_KEY`` /
``LANGCHAIN_PROJECT`` —— 两者同时存在时以 ``LANGSMITH_*`` 为准。

隐私提醒
--------
开启追踪后，**对话内容、工具入参与返回结果都会上传到 LangSmith**。
生产环境启用前请确认符合你所在组织的合规要求；如需脱敏，
应自定义 ``langsmith.Client(hide_inputs=...)`` 或在应用层先行脱敏。
"""

from __future__ import annotations

import os
from typing import Any

# ---------------------------------------------------------------------------
# 配置项（新旧变量名）
# ---------------------------------------------------------------------------

#: 是否启用追踪
TRACING_ENV: str = "LANGSMITH_TRACING"
TRACING_ENV_LEGACY: str = "LANGCHAIN_TRACING_V2"

#: API Key
API_KEY_ENV: str = "LANGSMITH_API_KEY"
API_KEY_ENV_LEGACY: str = "LANGCHAIN_API_KEY"

#: 项目名（在 LangSmith 中的分组）
PROJECT_ENV: str = "LANGSMITH_PROJECT"
PROJECT_ENV_LEGACY: str = "LANGCHAIN_PROJECT"

DEFAULT_PROJECT: str = "enterprise-service-agent"

#: 判定为「真」的取值
_TRUTHY: frozenset[str] = frozenset({"1", "true", "yes", "on"})

#: 判定为「假」的取值
_FALSY: frozenset[str] = frozenset({"0", "false", "no", "off"})


def _env(*names: str) -> str:
    """
    按顺序读取第一个非空环境变量。

    Args:
        *names: 候选变量名（优先级从高到低）。

    Returns:
        str: 变量值；均未设置时返回空串。
    """
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _has_api_key() -> bool:
    """
    是否已配置 LangSmith API Key。

    Returns:
        bool: 存在非占位 Key 时返回 True。
    """
    key = _env(API_KEY_ENV, API_KEY_ENV_LEGACY)
    return bool(key) and key not in {"your_key_here", "test-placeholder-not-a-real-key"}


def _tracing_flag() -> str:
    """
    读取追踪开关的原始取值（不区分新旧变量名）。

    Returns:
        str: 小写取值；未设置时返回空串。
    """
    return _env(TRACING_ENV, TRACING_ENV_LEGACY).lower()


def is_tracing_enabled() -> bool:
    """
    判断追踪当前是否处于启用状态。

    判定要求：开关未被显式关闭 **且** 配置了 API Key。
    仅开启开关而无 Key 时视为未启用 —— 否则每次请求都会产生上传失败日志，
    比不开追踪更糟。未显式设置开关时，只要配置了 Key 即可启用
    （与 langsmith 的默认行为一致）。

    Returns:
        bool: 启用返回 True。
    """
    if _tracing_flag() in _FALSY:
        return False
    return _has_api_key()


def configure_tracing() -> dict[str, Any]:
    """
    依据环境变量配置追踪，并返回状态摘要（供启动日志与 ``/health`` 使用）。

    幂等：重复调用不产生副作用。若环境已显式设置 langsmith 变量，
    则尊重既有取值，只补齐缺失项。

    Returns:
        dict: 含 ``enabled``、``project``、``has_api_key``、``reason``。
    """
    enabled = is_tracing_enabled()

    if not enabled:
        flag = _tracing_flag()
        # 显式写入 false，避免 langsmith 因检测到 Key 而自行启用
        os.environ.setdefault(TRACING_ENV, "false")
        if flag in _TRUTHY and not _has_api_key():
            reason = "已开启开关但未配置 LANGSMITH_API_KEY，追踪未启用"
        elif flag in _FALSY:
            reason = "已显式关闭"
        else:
            reason = "未启用（默认关闭，未配置 LANGSMITH_API_KEY）"
        return {
            "enabled": False,
            "project": None,
            "has_api_key": _has_api_key(),
            "reason": reason,
        }

    project = _env(PROJECT_ENV, PROJECT_ENV_LEGACY) or DEFAULT_PROJECT
    # 补齐变量，确保 langsmith 按预期工作（不覆盖用户已显式设置的值）
    os.environ.setdefault(TRACING_ENV, "true")
    os.environ.setdefault(TRACING_ENV_LEGACY, "true")
    os.environ.setdefault(PROJECT_ENV, project)
    os.environ.setdefault(PROJECT_ENV_LEGACY, project)

    return {
        "enabled": True,
        "project": project,
        "has_api_key": True,
        "reason": "已启用",
    }


def trace_config(
    *,
    thread_id: str | None = None,
    request_id: str | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """
    构造附带追踪元数据的 ``RunnableConfig`` 片段。

    这些元数据会出现在 LangSmith 的 trace 上，便于按会话或请求检索。
    追踪未启用时返回的字典依然可用（只携带 ``configurable``），
    因此调用方无需分支判断。

    Args:
        thread_id: 会话线程 ID。
        request_id: 请求 ID。
        run_name: 本次运行的显示名（如 ``"chat_stream"``）。

    Returns:
        dict: 可直接与 ``{"configurable": {...}}`` 合并的配置片段。
    """
    metadata: dict[str, Any] = {}
    if thread_id:
        metadata["thread_id"] = thread_id
    if request_id:
        metadata["request_id"] = request_id

    config: dict[str, Any] = {}
    if metadata:
        config["metadata"] = metadata
    if run_name:
        config["run_name"] = run_name
    return config


__all__ = [
    "API_KEY_ENV",
    "DEFAULT_PROJECT",
    "PROJECT_ENV",
    "TRACING_ENV",
    "configure_tracing",
    "is_tracing_enabled",
    "trace_config",
]
