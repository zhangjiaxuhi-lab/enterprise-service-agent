"""
检查点后端测试（``src/agent/checkpointer.py``）。

本模块验证第 2 项改进的核心承诺：**会话状态可持久化、可跨进程共享**。
这是原先 ``MemorySaver`` 无法提供的能力，也是多 worker 部署的前提。

测试分三组：

1. **配置解析**：环境变量驱动的后端与路径选择；
2. **同步后端**：``build_sync_checkpointer``（供 CLI / 同步场景）；
3. **异步持久化**：跨「进程」读回上下文 —— 模拟重启与多 worker。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from src.agent.checkpointer import (
    BACKEND_ENV,
    close_sync_checkpointer,
    DB_PATH_ENV,
    CheckpointerFactory,
    build_sync_checkpointer,
    resolve_backend,
    resolve_db_path,
)
from src.agent.customer_agent import build_graph, build_model

# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _mock_graph(checkpointer: Any) -> Any:
    """用 mock 模型与指定检查点构建一个图。"""
    return build_graph(model=build_model(mock=True), checkpointer=checkpointer)


def _drain_calls(messages: list[Any]) -> list[tuple[str, dict]]:
    """从消息历史中提取全部工具调用。"""
    calls: list[tuple[str, dict]] = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            calls.append((call.get("name", ""), call.get("args", {}) or {}))
    return calls


# ---------------------------------------------------------------------------
# 1) 配置解析
# ---------------------------------------------------------------------------


class TestConfiguration:
    """环境变量驱动的配置解析。"""

    def test_default_backend_is_sqlite(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """未配置时应默认使用 sqlite（持久化是默认行为）。"""
        monkeypatch.delenv(BACKEND_ENV, raising=False)
        assert resolve_backend() == "sqlite"

    def test_backend_can_be_overridden(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """可显式切换到 memory。"""
        monkeypatch.setenv(BACKEND_ENV, "memory")
        assert resolve_backend() == "memory"

    def test_invalid_backend_falls_back_to_sqlite(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """非法取值应回退为 sqlite，而不是崩溃。"""
        monkeypatch.setenv(BACKEND_ENV, "not-a-backend")
        assert resolve_backend() == "sqlite"

    def test_backend_is_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """取值大小写不敏感。"""
        monkeypatch.setenv(BACKEND_ENV, "MEMORY")
        assert resolve_backend() == "memory"

    def test_default_db_path_is_under_project(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """默认数据库路径应落在项目内的 data/runtime/ 下。"""
        monkeypatch.delenv(DB_PATH_ENV, raising=False)
        path = resolve_db_path()
        assert path.is_absolute()
        assert path.as_posix().endswith("data/runtime/checkpoints.db")

    def test_relative_db_path_resolves_against_project_root(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """相对路径应基于项目根目录，而非当前工作目录。"""
        monkeypatch.setenv(DB_PATH_ENV, "tmp/custom.db")
        path = resolve_db_path()
        assert path.is_absolute()
        assert path.as_posix().endswith("tmp/custom.db")

    def test_absolute_db_path_is_respected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_dir: Path
    ) -> None:
        """绝对路径应原样使用。"""
        monkeypatch.setenv(DB_PATH_ENV, str(tmp_dir / "x.db"))
        assert resolve_db_path() == tmp_dir / "x.db"


# ---------------------------------------------------------------------------
# 2) 同步后端
# ---------------------------------------------------------------------------


class TestSyncCheckpointer:
    """``build_sync_checkpointer`` 行为。"""

    def test_memory_backend(self) -> None:
        """memory 后端返回可用的内存检查点。"""
        saver = build_sync_checkpointer(backend="memory")
        try:
            assert saver is not None
        finally:
            close_sync_checkpointer(saver)

    def test_sqlite_backend_creates_file(self, tmp_dir: Path) -> None:
        """sqlite 后端应实际落盘并建表。"""
        db = tmp_dir / "sync.db"
        saver = build_sync_checkpointer(backend="sqlite", db_path=db)
        try:
            assert saver is not None
            assert db.is_file(), "数据库文件未创建"
        finally:
            # 必须关闭连接，否则 Windows 上文件句柄被占用，临时目录无法清理
            close_sync_checkpointer(saver)

    def test_sqlite_creates_parent_dirs(self, tmp_dir: Path) -> None:
        """父目录不存在时应自动创建。"""
        db = tmp_dir / "nested" / "deep" / "sync.db"
        saver = build_sync_checkpointer(backend="sqlite", db_path=db)
        try:
            assert db.is_file()
        finally:
            close_sync_checkpointer(saver)

    def test_wal_mode_enabled(self, tmp_dir: Path) -> None:
        """应启用 WAL，这是多进程并发读写的关键。"""
        import sqlite3

        db = tmp_dir / "wal.db"
        saver = build_sync_checkpointer(backend="sqlite", db_path=db)
        try:
            conn = sqlite3.connect(str(db))
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            conn.close()
            assert mode.lower() == "wal", f"journal_mode={mode}，WAL 未启用"
        finally:
            close_sync_checkpointer(saver)


# ---------------------------------------------------------------------------
# 3) 异步持久化（核心：模拟重启与多 worker）
# ---------------------------------------------------------------------------


class TestAsyncPersistence:
    """``CheckpointerFactory`` 的持久化能力。"""

    @pytest.mark.anyio
    async def test_sqlite_persists_across_sessions(self, tmp_dir: Path) -> None:
        """核心用例：写入后用**全新**检查点读回同一会话历史。

        模拟进程重启 —— 这正是 ``MemorySaver`` 做不到的事。
        """
        db = tmp_dir / "persist.db"
        config = {"configurable": {"thread_id": "persist-1"}}

        # --- 会话 1：写入 ---
        factory1 = CheckpointerFactory()
        factory1.backend = "sqlite"
        factory1.db_path = db
        async with factory1.open_async() as checkpointer:
            graph = _mock_graph(checkpointer)
            await graph.ainvoke(
                {"messages": [HumanMessage(content="403 权限不足怎么解决？")]},
                config=config,
            )
            before = await graph.aget_state(config)
            assert len(before.values.get("messages", [])) > 0

        # --- 会话 2：全新工厂 / 全新图，读回 ---
        factory2 = CheckpointerFactory()
        factory2.backend = "sqlite"
        factory2.db_path = db
        async with factory2.open_async() as checkpointer:
            graph = _mock_graph(checkpointer)
            after = await graph.aget_state(config)
            assert len(after.values.get("messages", [])) == len(
                before.values.get("messages", [])
            ), "重启后上下文丢失"

    @pytest.mark.anyio
    async def test_multi_turn_slot_filling_survives_restart(self, tmp_dir: Path) -> None:
        """模拟多 worker：进程 A 反问、进程 B 续话仍能正确补槽并提单。

        这是本项改进的直接价值 —— 原 ``MemorySaver`` 下，请求落到另一个
        worker 会丢失历史，导致重复反问、无法完成工单。
        """
        db = tmp_dir / "multiworker.db"
        config = {"configurable": {"thread_id": "mw-1"}}

        # --- worker A：缺账号，应反问 ---
        factory_a = CheckpointerFactory()
        factory_a.backend = "sqlite"
        factory_a.db_path = db
        async with factory_a.open_async() as checkpointer:
            graph = _mock_graph(checkpointer)
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content="我买错套餐了，帮我提个退费申请。")]},
                config=config,
            )
            assert _drain_calls(result["messages"]) == [], "第 1 轮不应调用工具"

        # --- worker B：只补账号，应基于共享历史直接提单 ---
        factory_b = CheckpointerFactory()
        factory_b.backend = "sqlite"
        factory_b.db_path = db
        async with factory_b.open_async() as checkpointer:
            graph = _mock_graph(checkpointer)
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content="账号是 U-555111。")]}, config=config
            )
            calls = _drain_calls(result["messages"])
            assert [name for name, _ in calls] == ["submit_ticket"], (
                f"worker B 未基于共享历史提单：{calls}"
            )
            args = calls[0][1]
            assert args["user_id"] == "U-555111"
            # 描述应保留 worker A 阶段的诉求
            assert "退费" in args["description"] or "买错" in args["description"]

    @pytest.mark.anyio
    async def test_ticket_id_survives_restart(self, tmp_dir: Path) -> None:
        """工单流水号应可从持久化历史中读回（审计可用性）。"""
        db = tmp_dir / "audit.db"
        config = {"configurable": {"thread_id": "audit-1"}}

        factory = CheckpointerFactory()
        factory.backend = "sqlite"
        factory.db_path = db
        async with factory.open_async() as checkpointer:
            graph = _mock_graph(checkpointer)
            await graph.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content="账号是 U-987654，我要申请退款，昨天买错了套餐。"
                        )
                    ]
                },
                config=config,
            )

        # 重启后读回，确认工单号仍在历史中
        factory2 = CheckpointerFactory()
        factory2.backend = "sqlite"
        factory2.db_path = db
        async with factory2.open_async() as checkpointer:
            graph = _mock_graph(checkpointer)
            snapshot = await graph.aget_state(config)
            blob = " ".join(
                str(getattr(m, "content", "")) for m in snapshot.values.get("messages", [])
            )
            assert "TK-" in blob, "重启后无法从历史中读回工单号"

    @pytest.mark.anyio
    async def test_threads_isolated_in_sqlite(self, tmp_dir: Path) -> None:
        """SQLite 后端下不同 thread_id 仍应相互隔离。"""
        db = tmp_dir / "iso.db"
        factory = CheckpointerFactory()
        factory.backend = "sqlite"
        factory.db_path = db
        async with factory.open_async() as checkpointer:
            graph = _mock_graph(checkpointer)
            await graph.ainvoke(
                {"messages": [HumanMessage(content="账号是 U-987654，我要申请退款。")]},
                config={"configurable": {"thread_id": "iso-a"}},
            )
            snap_b = await graph.aget_state({"configurable": {"thread_id": "iso-b"}})
            assert snap_b.values.get("messages", []) == [], "线程未隔离"

    @pytest.mark.anyio
    async def test_memory_backend_does_not_persist(self, tmp_dir: Path) -> None:
        """对照组：memory 后端确实不持久化（证明上述用例真的有区分度）。"""
        config = {"configurable": {"thread_id": "mem-1"}}

        factory1 = CheckpointerFactory()
        factory1.backend = "memory"
        async with factory1.open_async() as checkpointer:
            graph = _mock_graph(checkpointer)
            await graph.ainvoke(
                {"messages": [HumanMessage(content="你好")]}, config=config
            )

        factory2 = CheckpointerFactory()
        factory2.backend = "memory"
        async with factory2.open_async() as checkpointer:
            graph = _mock_graph(checkpointer)
            snapshot = await graph.aget_state(config)
            assert snapshot.values.get("messages", []) == [], (
                "memory 后端不应跨实例保留状态"
            )


# ---------------------------------------------------------------------------
# 4) 降级行为
# ---------------------------------------------------------------------------


class TestDegradation:
    """sqlite 不可用时应降级为内存，且降级原因可见。"""

    @pytest.mark.anyio
    async def test_unwritable_path_degrades_to_memory(self, tmp_dir: Path) -> None:
        """数据库路径不可用时降级为 memory，并记录原因。"""
        # 用一个「父路径是文件」的非法路径，稳定触发初始化失败
        blocker = tmp_dir / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")

        factory = CheckpointerFactory()
        factory.backend = "sqlite"
        factory.db_path = blocker / "sub" / "db.sqlite"

        async with factory.open_async() as checkpointer:
            assert checkpointer is not None
            assert factory.degraded_reason is not None, "降级原因应被记录"
            assert factory.effective_backend == "memory"

    @pytest.mark.anyio
    async def test_body_exception_does_not_trigger_degradation(
        self, tmp_dir: Path
    ) -> None:
        """使用阶段的业务异常不得被误判为降级。

        这是实现上的一个易错点：若把 ``yield`` 也包进 try/except，
        调用方代码抛错会被误当成「sqlite 不可用」而静默切到内存。
        """
        db = tmp_dir / "ok.db"
        factory = CheckpointerFactory()
        factory.backend = "sqlite"
        factory.db_path = db

        with pytest.raises(ValueError, match="业务异常"):
            async with factory.open_async():
                raise ValueError("业务异常")

        assert factory.degraded_reason is None, "业务异常不应触发降级"
