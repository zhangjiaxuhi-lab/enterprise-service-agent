"""
SSE 流式接口终端**冒烟**脚本（第三阶段配套）。

定位说明：本脚本用于人眼观察流式输出是否正常，**不做断言、不构成测试覆盖**。
严格的行为校验（含 SSE 事件协议契约）由 `tests/test_api.py` 中的 pytest 用例负责。
运行完整测试请用：``pytest``；本脚本仅用于快速目视确认服务可用。

在**服务已启动**的前提下，用标准库（无需 requests/httpx）逐事件读取
``POST /api/chat/stream`` 的 Server-Sent Events 流，并按事件类型着色打印，
便于快速确认「推理 token / 工具调用 / 异常 / 结束」四类事件的推送顺序。

用法::

    # 1) 先启动服务（另开一个终端）
    python -m src.api.main

    # 2) 再运行本脚本
    python src/api/smoke_sse.py                       # 跑三个内置场景
    python src/api/smoke_sse.py "你们的退款政策是什么？"   # 自定义单条消息
    python src/api/smoke_sse.py --url http://127.0.0.1:8000 --thread t-1 "你好"

依赖：仅使用 Python 标准库，无需额外安装。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

DEFAULT_URL: str = "http://127.0.0.1:8000"

# 内置三个验证场景（与状态机测试场景一致）
DEFAULT_SCENARIOS: list[tuple[str, str]] = [
    ("场景1 知识库问答", "你好，系统登录提示 403 权限不足怎么排查解决？"),
    ("场景2 槽位缺失反问", "我买错套餐了，帮我提个退费申请。"),
    (
        "场景3 参数齐备提交",
        "账号是 U-987654，我要申请退款，昨天购买的年费企业版误选了双份，申请退订一份。",
    ),
]

# 事件类型 -> 显示前缀
_PREFIX: dict[str, str] = {
    "token": "  [token]     ",
    "tool_start": "  [tool_start]",
    "tool_end": "  [tool_end]  ",
    "error": "  [error]     ",
    "done": "  [done]      ",
}


def stream_once(base_url: str, message: str, thread_id: str, *, quiet: bool = False) -> list[dict]:
    """
    向流式接口发送一条消息并逐事件读取。

    Args:
        base_url: 服务基地址，如 ``http://127.0.0.1:8000``。
        message: 用户消息。
        thread_id: 会话线程 ID。
        quiet: 为 True 时只收集事件、不打印。

    Returns:
        list[dict]: 收到的事件列表。

    Raises:
        SystemExit: 服务不可达或返回非 200。
    """
    payload = json.dumps({"message": message, "thread_id": thread_id}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat/stream",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )

    events: list[dict] = []
    try:
        # urlopen 返回的是流式响应对象，可逐字节读取，不会被整体缓冲
        with urllib.request.urlopen(request, timeout=120) as response:
            if not quiet:
                content_type = response.headers.get("Content-Type", "")
                print(f"  HTTP {response.status} | Content-Type: {content_type}")
                # 校验服务端声明的媒体类型
                if "text/event-stream" not in content_type:
                    print(f"  ⚠ 期望 text/event-stream，实际为 {content_type}")

            buffer = b""
            while True:
                chunk = response.read(1)
                if not chunk:
                    break
                buffer += chunk
                # SSE 以空行分隔事件
                if not buffer.endswith(b"\n\n"):
                    continue
                raw = buffer.decode("utf-8").strip()
                buffer = b""
                if not raw.startswith("data: "):
                    continue
                event = json.loads(raw[len("data: "):])
                events.append(event)
                if not quiet:
                    _print_event(event)
    except urllib.error.HTTPError as error:
        print(f"  ✗ HTTP {error.code}: {error.read().decode('utf-8', 'replace')[:200]}")
        raise SystemExit(1)
    except urllib.error.URLError as error:
        print(f"  ✗ 无法连接 {base_url}：{error.reason}")
        print("    请先启动服务：python -m src.api.main")
        raise SystemExit(1)

    return events


def _print_event(event: dict) -> None:
    """
    按事件类型格式化打印单条 SSE 事件。

    Args:
        event: 已解析的事件字典。
    """
    kind = event.get("type", "unknown")
    prefix = _PREFIX.get(kind, f"  [{kind}]")

    if kind == "token":
        # token 增量较多，单行显示并转义换行，便于观察流式切片
        print(f"{prefix} {event.get('content', '')!r}")
    elif kind == "tool_start":
        args = json.dumps(event.get("args", {}), ensure_ascii=False)
        print(f"{prefix} {event.get('tool')}  args={args}")
    elif kind == "tool_end":
        output = event.get("output", "")
        print(f"{prefix} {event.get('tool')}  output({len(output)} 字符) {output[:80]!r}")
    elif kind == "error":
        print(f"{prefix} {event.get('message')}")
    elif kind == "done":
        print(f"{prefix} thread_id={event.get('thread_id')}")
    else:
        print(f"{prefix} {json.dumps(event, ensure_ascii=False)[:160]}")


def _summarize(events: list[dict]) -> str:
    """
    汇总事件序列，便于快速判定行为是否符合预期。

    Args:
        events: 事件列表。

    Returns:
        str: 形如 ``tool_start → tool_end → token → done`` 的序列摘要。
    """
    return " → ".join(event.get("type", "?") for event in events)


def main() -> int:
    """
    命令行入口。

    Returns:
        int: 退出码，0 表示全部通过。
    """
    parser = argparse.ArgumentParser(description="SSE 流式接口终端验证脚本")
    parser.add_argument("message", nargs="?", help="自定义用户消息（省略则跑内置三场景）")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"服务地址，默认 {DEFAULT_URL}")
    parser.add_argument("--thread", default="", help="会话线程 ID（默认按场景自动生成）")
    args = parser.parse_args()

    # 先探活，尽早暴露「服务未启动 / 图未就绪」问题
    print("=" * 74)
    try:
        with urllib.request.urlopen(f"{args.url.rstrip('/')}/health", timeout=10) as response:
            health = json.loads(response.read().decode("utf-8"))
        print(f"健康检查：status={health.get('status')} | graph_ready={health.get('graph_ready')} "
              f"| model_mode={health.get('model_mode')}")
        if not health.get("graph_ready"):
            print(f"⚠ 图未就绪：{health.get('init_error')}")
    except Exception as error:  # noqa: BLE001 - 探活失败即提示启动方式
        print(f"✗ 健康检查失败：{type(error).__name__}: {error}")
        print("  请先启动服务：python -m src.api.main")
        return 1
    print("=" * 74)

    scenarios = (
        [("自定义消息", args.message)] if args.message else DEFAULT_SCENARIOS
    )

    all_events: list[dict] = []
    for index, (label, message) in enumerate(scenarios, start=1):
        thread_id = args.thread or f"sse-test-{index}"
        print(f"\n【{label}】thread_id={thread_id}")
        print(f"用户：{message}")
        print("-" * 74)
        events = stream_once(args.url, message, thread_id)
        all_events.extend(events)
        print(f"  事件序列：{_summarize(events)}")

        # 仅做可视化提示，不做断言。
        # 严格的行为校验在 tests/ 下由 pytest 完成（见 tests/README.md），
        # 本脚本的定位是「人眼观察流式输出」，不是测试。
        kinds = {event.get("type") for event in events}
        if "done" not in kinds:
            print("  ⚠ 未收到 done 结束事件")
        if label == "场景2 槽位缺失反问":
            print(
                "  ℹ 未触发工具" if "tool_start" not in kinds
                else "  ⚠ 槽位缺失却触发了工具调用"
            )
        elif label == "场景3 参数齐备提交":
            hit = any(
                e.get("type") == "tool_start" and e.get("tool") == "submit_ticket"
                for e in events
            )
            print("  ℹ 已调用 submit_ticket" if hit else "  ⚠ 未调用 submit_ticket")

    print("\n" + "=" * 74)
    print(f"完成，共接收 {len(all_events)} 条事件：{_summarize(all_events)}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
