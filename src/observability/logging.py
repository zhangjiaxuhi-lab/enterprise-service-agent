"""
结构化日志配置。

为什么不用 ``print``
--------------------
原实现中服务端、检查点与 agent 库代码都直接用 ``print``，带来三个问题：

1. **无级别** —— 开发调试信息与生产错误混在一起，无法按级别过滤；
2. **无结构** —— 无法被日志采集系统（ELK / Loki / CloudWatch）解析成字段；
3. **污染 stdout** —— SSE 是流式响应，任何写到 stdout 的输出都可能干扰
   观测，日志应统一走 **stderr**。

格式
----
通过 ``LOG_FORMAT`` 选择：

* ``console``（默认）：人类可读单行，便于本地开发；
* ``json``：每行一个 JSON 对象，便于采集与检索。

两种格式都会自动注入请求上下文（``request_id`` / ``thread_id``）。
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

from src.observability.context import get_context

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 日志级别（DEBUG / INFO / WARNING / ERROR）
LEVEL_ENV: str = "LOG_LEVEL"
#: 日志格式（console / json）
FORMAT_ENV: str = "LOG_FORMAT"

DEFAULT_LEVEL: str = "INFO"
DEFAULT_FORMAT: str = "console"

VALID_FORMATS: frozenset[str] = frozenset({"console", "json"})

#: 本项目的日志器命名空间
ROOT_LOGGER_NAME: str = "esa"

#: LogRecord 的内置属性，序列化时需排除，以免把整个 record 写进日志
_RESERVED: frozenset[str] = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()
) | {"message", "asctime", "taskName"}


def resolve_level() -> int:
    """
    解析日志级别。

    Returns:
        int: 标准库日志级别；取值非法时回退 INFO。
    """
    raw = os.getenv(LEVEL_ENV, DEFAULT_LEVEL).strip().upper()
    level = logging.getLevelName(raw)
    return level if isinstance(level, int) else logging.INFO


def resolve_format() -> str:
    """
    解析日志格式。

    Returns:
        str: ``console`` 或 ``json``；取值非法时回退 console。
    """
    raw = os.getenv(FORMAT_ENV, DEFAULT_FORMAT).strip().lower()
    return raw if raw in VALID_FORMATS else DEFAULT_FORMAT


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


class ContextFormatter(logging.Formatter):
    """
    注入请求上下文并支持 JSON / 控制台两种输出的 Formatter。

    附加字段（通过 ``logger.info(..., extra={"key": value})`` 传入）会被
    一并带出，便于记录 user_id、tool、duration_ms 等业务维度。
    """

    def __init__(self, fmt: str = "json") -> None:
        super().__init__()
        self.fmt = fmt

    def format(self, record: logging.LogRecord) -> str:
        """
        将日志记录格式化为一行文本。

        Args:
            record: 标准库日志记录。

        Returns:
            str: console 或 JSON 格式的单行日志。
        """
        context = get_context()
        # 合并：上下文附加字段 + 调用点通过 extra= 传入的字段
        extras: dict[str, Any] = {**context.extra, **self._declared_extras(record)}

        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"

        if self.fmt == "json":
            payload: dict[str, Any] = {
                "timestamp": timestamp,
                "level": record.levelname,
                "logger": record.name,
                "message": message,
                "request_id": context.request_id,
                "thread_id": context.thread_id,
            }
            payload.update(extras)
            # ensure_ascii=False：中文按原样输出，便于直接阅读与检索
            return json.dumps(payload, ensure_ascii=False, default=str)

        # console：固定宽度便于肉眼对齐
        suffix = " ".join(
            f'{key}="{value}"' if " " in str(value) else f"{key}={value}"
            for key, value in extras.items()
        )
        line = (
            f"{timestamp} {record.levelname:<8} "
            f"[{context.request_id}] [{context.thread_id}] "
            f"{record.name}: {message}"
        )
        return f"{line} {suffix}".rstrip()

    @staticmethod
    def _declared_extras(record: logging.LogRecord) -> dict[str, Any]:
        """
        提取调用点通过 ``extra=`` 传入的非标准字段。

        Args:
            record: 标准库日志记录。

        Returns:
            dict[str, Any]: 额外字段；无则返回空字典。
        """
        return {
            key: value
            for key, value in vars(record).items()
            if key not in _RESERVED and not key.startswith("_")
        }


# ---------------------------------------------------------------------------
# 安装
# ---------------------------------------------------------------------------

_configured: bool = False


def setup_logging(
    level: int | str | None = None,
    fmt: str | None = None,
    *,
    force: bool = False,
) -> logging.Logger:
    """
    配置项目日志器（幂等，可重复调用）。

    输出到 **stderr**：stdout 保留给正常输出（如 CLI 演示的打印内容），
    避免日志与业务输出相互干扰。

    Args:
        level: 级别，接受字符串或标准库常量；默认读 ``LOG_LEVEL``。
        fmt: ``console`` / ``json``；默认读 ``LOG_FORMAT``。
        force: 已配置时是否重新配置（测试用）。

    Returns:
        logging.Logger: 项目根日志器（``esa``）。
    """
    global _configured

    logger = logging.getLogger(ROOT_LOGGER_NAME)
    if _configured and not force:
        return logger

    resolved_level = level if isinstance(level, int) else resolve_level()
    if isinstance(level, str):
        parsed = logging.getLevelName(level.strip().upper())
        resolved_level = parsed if isinstance(parsed, int) else logging.INFO

    resolved_fmt = fmt if fmt in VALID_FORMATS else resolve_format()

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(ContextFormatter(resolved_fmt))

    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(resolved_level)
    # 不向 root 传播：避免被第三方库（uvicorn 等）的 handler 重复输出
    logger.propagate = False

    _configured = True
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """
    获取项目命名空间下的日志器。

    首次调用会自动完成配置，因此业务代码无需关心初始化顺序。

    Args:
        name: 子模块名（如 ``"api"`` / ``"agent"``）；为空则返回根日志器。

    Returns:
        logging.Logger: 形如 ``esa.api`` 的日志器。
    """
    setup_logging()
    return logging.getLogger(ROOT_LOGGER_NAME if not name else f"{ROOT_LOGGER_NAME}.{name}")


__all__ = [
    "FORMAT_ENV",
    "LEVEL_ENV",
    "VALID_FORMATS",
    "ContextFormatter",
    "get_logger",
    "resolve_format",
    "resolve_level",
    "setup_logging",
]
