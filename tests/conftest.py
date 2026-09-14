"""
pytest 配置与共享 fixtures。

设计要点
--------
* **默认离线**：全部测试默认使用确定性 mock 模型，不消耗 API 额度、不依赖网络，
  因此可以直接进 CI。
* **真实模型单列**：打上 ``@pytest.mark.real_model`` 的测试才会真正调用
  DashScope，默认跳过（``-m real_model`` 显式启用）。理由见 tests/README.md。
* **可注入依赖**：图与模型通过 fixture 提供，测试不依赖真实 API Key。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# 路径注入
# ---------------------------------------------------------------------------

# 项目根目录加入 sys.path，使 `import src.*` 在测试中可用
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 注意：这里**刻意不设置** DASHSCOPE_API_KEY 占位值。
#
# 原因：`src/agent/customer_agent.py` 在导入时执行
# `load_dotenv(override=False)` —— 若此处抢先写入占位 Key，
# .env 中的真实 Key 将永远无法被加载（override=False 不覆盖已存在变量），
# 导致 `-m real_model` 用例全部 401 失败。
#
# 默认测试全部使用注入的 mock 模型，不需要任何 Key；
# 只有 real_model 用例需要真实 Key，而它由 .env 提供。


# ---------------------------------------------------------------------------
# 模型 fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_model():
    """
    确定性 mock 模型（不联网、不耗额度）。

    Returns:
        _MockToolCallingModel: 遵守与 System Prompt 相同的槽位规则。
    """
    from src.agent.customer_agent import build_model

    return build_model(mock=True)


@pytest.fixture
def empty_model():
    """
    极简模型桩：始终返回无 tool_calls 的固定文本。

    用于隔离测试「图的路由行为」本身，避免受 mock 模型业务规则影响。

    Returns:
        _EmptyModel: 每次 invoke 都返回同一句话。
    """

    class _EmptyModel:
        def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
            """保持链式接口兼容。"""
            return self

        def invoke(self, messages, **kwargs):  # noqa: ANN001, ANN003
            """返回不含工具调用的固定回复。"""
            from langchain_core.messages import AIMessage

            return AIMessage(content="这是固定回复。")

    return _EmptyModel()


# ---------------------------------------------------------------------------
# 图 fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_graph(mock_model):
    """
    使用 mock 模型的已编译图（带内存检查点，支持多轮）。

    Returns:
        CompiledStateGraph: 可直接 stream / ainvoke 的应用。
    """
    from langgraph.checkpoint.memory import MemorySaver

    from src.agent.customer_agent import build_graph

    return build_graph(model=mock_model, checkpointer=MemorySaver())


@pytest.fixture
def empty_graph(empty_model):
    """
    使用极简模型桩的已编译图，用于路由层单测。

    Returns:
        CompiledStateGraph: 始终不触发工具的图。
    """
    from langgraph.checkpoint.memory import MemorySaver

    from src.agent.customer_agent import build_graph

    return build_graph(model=empty_model, checkpointer=MemorySaver())


@pytest.fixture
def fresh_graph_factory():
    """
    返回一个「按需新建图」的工厂，保证用例间状态完全隔离。

    Returns:
        Callable[[], CompiledStateGraph]: 每次调用产出全新的图实例。
    """
    from langgraph.checkpoint.memory import MemorySaver

    from src.agent.customer_agent import build_graph, build_model

    def _make():
        return build_graph(model=build_model(mock=True), checkpointer=MemorySaver())

    return _make
