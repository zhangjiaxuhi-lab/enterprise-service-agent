"""
检查点（checkpoint）后端工厂。

为什么需要它
------------
LangGraph 的检查点负责保存会话状态（消息历史），是「多轮对话」的存储层。
原先使用 ``MemorySaver``（纯内存），带来两个生产级问题：

1. **进程重启即丢上下文** —— 用户正在进行的多轮工单流程被打断；
2. **多 worker 部署直接失效** —— 同一会话的下一轮请求若落到另一个进程，
   该进程没有这段历史，槽位填充逻辑会退化为「首次对话」（重新反问、丢失诉求）。

本模块把检查点后端做成**可配置**的，默认使用 SQLite 落盘，从而支持
单机持久化与多进程共享同一份状态。

后端选择
--------
通过环境变量 ``CHECKPOINT_BACKEND`` 控制：

* ``sqlite``（默认）：落盘到 ``data/runtime/checkpoints.db``，WAL 模式，
  支持多进程安全读写；
* ``memory``：纯内存，仅用于测试与一次性演示。

失败降级
--------
SQLite 初始化失败（如目录只读、磁盘满）时**自动降级为 memory**，
并把原因记录在 :attr:`CheckpointerFactory.degraded_reason`，
由 ``/health`` 暴露出来。理由：一个只读文件系统的部署仍应能提供
（无持久化的）对话服务，而不是直接启动失败；但降级必须**可见**，
不能静默。

同步 / 异步的注意事项（重要）
-----------------------------
``AsyncSqliteSaver`` **只支持异步接口**。因此：

* 异步调用方（FastAPI `astream` / `ainvoke`）必须用 ``await graph.aget_state(...)``，
  直接调用 ``get_state`` 会抛 ``InvalidStateError``；
* 需要同步访问的场景（CLI 脚本、部分测试）请使用 ``backend="memory"``，
  或使用 :func:`build_sync_checkpointer` 返回的同步 saver。
"""

from __future__ import annotations

import os
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

# 项目根目录：本文件位于 <root>/src/agent/checkpointer.py
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

#: 检查点后端：``sqlite``（默认）或 ``memory``
BACKEND_ENV: str = "CHECKPOINT_BACKEND"

#: SQLite 数据库文件路径（相对路径基于项目根目录）
DB_PATH_ENV: str = "CHECKPOINT_DB_PATH"

#: 默认数据库路径
DEFAULT_DB_RELATIVE: str = "data/runtime/checkpoints.db"

VALID_BACKENDS: frozenset[str] = frozenset({"sqlite", "memory"})


def resolve_backend() -> str:
    """
    解析当前应使用的检查点后端。

    Returns:
        str: ``"sqlite"`` 或 ``"memory"``；取值非法时回退为 ``sqlite``。
    """
    raw = os.getenv(BACKEND_ENV, "sqlite").strip().lower()
    return raw if raw in VALID_BACKENDS else "sqlite"


def resolve_db_path() -> Path:
    """
    解析 SQLite 数据库文件路径。

    相对路径一律基于**项目根目录**解析（而非当前工作目录），
    避免因启动目录不同而在多处生成数据库文件。

    Returns:
        Path: 绝对路径。
    """
    raw = os.getenv(DB_PATH_ENV, "").strip()
    if not raw:
        return _PROJECT_ROOT / DEFAULT_DB_RELATIVE
    path = Path(raw)
    return path if path.is_absolute() else (_PROJECT_ROOT / path)


# ---------------------------------------------------------------------------
# 同步检查点（供 CLI 脚本与同步测试使用）
# ---------------------------------------------------------------------------


def build_sync_checkpointer(
    backend: str | None = None,
    db_path: Path | None = None,
) -> Any:
    """
    构建**同步**检查点，用于非异步场景（CLI 脚本、同步测试）。

    Args:
        backend: 覆盖后端选择；默认读取环境变量。
        db_path: 覆盖数据库路径；默认读取环境变量。

    Returns:
        Any: ``SqliteSaver`` 或 ``MemorySaver`` 实例。

    Raises:
        RuntimeError: 选择 sqlite 但依赖缺失或初始化失败。
    """
    backend = (backend or resolve_backend()).lower()
    if backend == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()

    import sqlite3

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as error:  # pragma: no cover - 依赖缺失
        raise RuntimeError(
            "缺少 langgraph-checkpoint-sqlite，无法使用 sqlite 检查点。\n"
            "安装：pip install 'langgraph-checkpoint-sqlite>=2.0,<3.0'"
        ) from error

    path = db_path or resolve_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    # check_same_thread=False：允许在 LangGraph 的执行线程中复用连接
    conn = sqlite3.connect(str(path), check_same_thread=False)
    # WAL 模式：允许「多读一写」并发，是多进程共享同一数据库的关键
    conn.execute("PRAGMA journal_mode=WAL")
    saver = SqliteSaver(conn)
    saver.setup()
    # 暴露底层连接，便于调用方显式释放。
    # 若不关闭，Windows 上文件句柄会保持占用，导致临时目录无法删除。
    # SqliteSaver 自身没有 close()，因此这里附加引用（不改动其行为）。
    saver._esa_conn = conn  # noqa: SLF001
    return saver


def close_sync_checkpointer(saver: Any) -> None:
    """
    关闭同步检查点持有的 SQLite 连接（若存在）。

    主要供测试与短生命周期脚本使用：不关闭会在 Windows 上锁住数据库文件，
    使临时目录无法清理。对 ``MemorySaver`` 等无连接的实现是安全的空操作。

    Args:
        saver: :func:`build_sync_checkpointer` 的返回值。
    """
    conn = getattr(saver, "_esa_conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:  # noqa: BLE001 - 关闭失败不应影响调用方
            pass


# ---------------------------------------------------------------------------
# 异步检查点（供 FastAPI 使用）
# ---------------------------------------------------------------------------


class CheckpointerFactory:
    """
    异步检查点的生命周期管理器。

    用法::

        factory = CheckpointerFactory()
        async with factory.open_async() as checkpointer:
            graph = build_graph(model=..., checkpointer=checkpointer)

    ``AsyncSqliteSaver.from_conn_string`` 是异步上下文管理器，其连接必须在
    应用关闭时释放，因此这里用 ``AsyncExitStack`` 托管生命周期。
    """

    def __init__(self) -> None:
        self.backend: str = resolve_backend()
        self.db_path: Path = resolve_db_path()
        self.checkpointer: Any | None = None
        #: 非空表示发生了降级（sqlite 初始化失败 -> memory），用于 /health 暴露
        self.degraded_reason: str | None = None

    @property
    def effective_backend(self) -> str:
        """实际生效的后端（考虑降级后的结果）。"""
        if self.checkpointer is None:
            return self.backend
        return "memory" if self.degraded_reason and self.backend != "memory" else self.backend

    @asynccontextmanager
    async def open_async(self):
        """
        打开检查点并托管其生命周期。

        Yields:
            Any: 可传给 ``build_graph(checkpointer=...)`` 的检查点实例。

        注意：降级判定**只覆盖初始化阶段**。``yield`` 之后调用方代码抛出的异常
        属于业务异常，不应被误判为「sqlite 不可用」而触发降级。
        """
        stack = AsyncExitStack()
        checkpointer: Any | None = None

        try:
            # ---------------- 初始化阶段（失败则降级） ----------------
            if self.backend == "memory":
                from langgraph.checkpoint.memory import MemorySaver

                checkpointer = MemorySaver()
            else:
                try:
                    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

                    # 目录创建是阻塞 IO，放到线程执行，避免阻塞事件循环
                    import anyio

                    await anyio.to_thread.run_sync(
                        lambda: self.db_path.parent.mkdir(parents=True, exist_ok=True)
                    )

                    # from_conn_string 是异步上下文管理器，交由 ExitStack 托管
                    saver = await stack.enter_async_context(
                        AsyncSqliteSaver.from_conn_string(str(self.db_path))
                    )
                    await saver.setup()
                    checkpointer = saver
                except Exception as error:  # noqa: BLE001 - 降级需捕获全部异常
                    from langgraph.checkpoint.memory import MemorySaver

                    self.degraded_reason = f"{type(error).__name__}: {error}"
                    print(
                        "[检查点] sqlite 初始化失败，已降级为内存模式"
                        f"（重启将丢失上下文）：{self.degraded_reason}",
                        file=sys.stderr,
                    )
                    checkpointer = MemorySaver()

            self.checkpointer = checkpointer
            # ---------------- 使用阶段（异常直接向上传播） ----------------
            yield checkpointer
        finally:
            await stack.aclose()
            self.checkpointer = None


__all__ = [
    "BACKEND_ENV",
    "close_sync_checkpointer",
    "DB_PATH_ENV",
    "DEFAULT_DB_RELATIVE",
    "VALID_BACKENDS",
    "CheckpointerFactory",
    "build_sync_checkpointer",
    "resolve_backend",
    "resolve_db_path",
]
