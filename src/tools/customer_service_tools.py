"""
企业级智能客服与工单协同中枢 —— 基础业务工具集。

本模块提供第一阶段的两个标准 LangChain 工具：

1. ``query_knowledge_base`` —— 基于关键词的基础知识库检索（读取 data/docs/ 下的 Markdown 文档）。
2. ``submit_ticket``         —— 模拟企业微信 / OA 提交工单，返回结构化 JSON 流水号。

设计说明：
    * 检索采用「分词打分 + 段落切分」的轻量方案，不依赖向量库或外部服务，
      便于第一阶段快速验证；后续阶段可平滑替换为向量检索实现。
    * 所有工具均使用 ``@tool`` 装饰器，函数签名与 docstring 会被 LangGraph /
      LLM 用于工具调用参数推断，因此参数名与描述需保持准确。
"""

from __future__ import annotations

import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# 常量与路径配置
# ---------------------------------------------------------------------------

# 项目根目录：本文件位于 <root>/src/tools/customer_service_tools.py
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# 知识库文档目录
KNOWLEDGE_BASE_DIR: Path = _PROJECT_ROOT / "data" / "docs"

# 支持的文档后缀
_SUPPORTED_SUFFIXES: tuple[str, ...] = (".md", ".markdown", ".txt")

# 返回给 LLM 的最大字符数，防止上下文溢出
_MAX_SNIPPET_CHARS: int = 1500

# 单个检索片段的长度阈值（字符）：累计超过即独立成段，避免片段过大导致匹配被稀释
_CHUNK_TARGET_CHARS: int = 300

# 单个关键词命中时的权重
_WEIGHT_TITLE: int = 5      # 标题行命中
_WEIGHT_KEYWORD: int = 3    # 正文命中
_WEIGHT_FILENAME: int = 2   # 文件名命中（如 account / refund）

# 检索时忽略的停用词（中英文常见虚词）
_STOPWORDS: frozenset[str] = frozenset(
    {
        "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一",
        "如何", "怎么", "怎样", "什么", "为什么", "请问", "一下", "可以", "需要",
        "the", "a", "an", "is", "are", "of", "to", "and", "or", "for", "how",
        "what", "why", "can", "i", "my", "please",
    }
)

# 工单流水号前缀与年份来源
_TICKET_PREFIX: str = "TK"

# 合法的问题类型枚举（与 data/docs/refund_policy.md 保持一致）
VALID_ISSUE_TYPES: frozenset[str] = frozenset(
    {
        "account",    # 账号与登录
        "refund",     # 退款与结算
        "billing",    # 账单与发票
        "technical",  # 技术故障
        "feature",    # 功能咨询
        "complaint",  # 投诉与建议
    }
)

# 问题类型 -> 默认处理组（模拟路由表）
_ISSUE_ROUTING: dict[str, str] = {
    "account": "一线技术支持组",
    "refund": "财务结算组",
    "billing": "财务结算组",
    "technical": "二线技术支持组",
    "feature": "客户成功组",
    "complaint": "客户成功负责人",
}


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------


def _load_documents() -> list[dict[str, Any]]:
    """
    加载知识库目录下的全部文本文档。

    Returns:
        list[dict]: 每项包含 ``name``（文件名）、``path``（路径）、``content``（全文）。
        若目录不存在或无匹配文件，返回空列表。
    """
    if not KNOWLEDGE_BASE_DIR.is_dir():
        return []

    documents: list[dict[str, Any]] = []
    for file_path in sorted(KNOWLEDGE_BASE_DIR.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in _SUPPORTED_SUFFIXES:
            continue
        try:
            content = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            # 单个文件读取失败不应中断整体检索
            continue
        documents.append(
            {"name": file_path.name, "path": str(file_path), "content": content}
        )
    return documents


def _tokenize(text: str) -> list[str]:
    """
    轻量分词：抽取英文/数字词组与中文字符(含 2-gram)。

    中英文混排文档不做严格分词，采用「英文单词 + 中文单字 + 中文双字组合」
    的方式生成候选词，足以支撑关键词打分。

    Args:
        text: 待分词的文本。

    Returns:
        list[str]: 去重后的候选词列表（保留原始大小写，另由打分函数统一小写）。
    """
    lowered = text.lower()
    tokens: list[str] = []

    # 英文单词与数字
    tokens.extend(re.findall(r"[a-z0-9_]+", lowered))

    # 中文单字
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", lowered)
    tokens.extend(chinese_chars)

    # 中文双字组合（提升「退款条件」这类词的区分度）
    for match in re.findall(r"[\u4e00-\u9fff]{2,}", lowered):
        tokens.extend(match[i : i + 2] for i in range(len(match) - 1))

    # 去重并剔除停用词与单词
    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        if len(token) < 2 or token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result


def _split_into_chunks(content: str) -> list[str]:
    """
    将 Markdown 文档切分为携带标题上下文的语义片段。

    切分策略：
        1. 以标题行（``#``~``######``）为界切分为「章节」；
        2. 章节过长时按空行拆分为段落缓冲，累积到阈值即输出；
        3. 每个片段前置「文档标题 + 当前章节标题」，避免出现脱离上下文的
           孤立片段（例如仅有「### 2.5 客服处理规则」正文而不知属于 401 章节）。

    标题上下文的保留对第一阶段的关键词检索至关重要：用户查询往往只含
    错误码或业务名词，标题能显著提升片段的区分度。

    Args:
        content: 文档全文。

    Returns:
        list[str]: 片段文本列表；每个片段均以标题上下文开头。
    """
    chunks: list[str] = []
    doc_title = ""
    current_section = ""
    buffer: list[str] = []

    def flush() -> None:
        """
        输出当前缓冲的段落，并前置标题上下文。

        片段格式为「文档标题 / 章节标题 / 正文」三段式。章节标题既作为
        上下文，也作为正文首行参与打分，使「标准解决步骤」等小节标题获得与
        「常见错误码总览」表格行同等的命中机会，避免概览类片段仅因标题未
        计入正文而排在操作步骤之前。
        """
        if not buffer:
            return
        header_parts = [part for part in (doc_title, current_section) if part]
        header = "\n".join(header_parts)
        body = "\n\n".join(buffer)
        chunks.append(f"{header}\n\n{body}" if header else body)
        buffer.clear()

    for line in content.splitlines():
        stripped = line.strip()
        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)

        if heading:
            # 标题行是章节边界：先落盘上一章节的缓冲，再切换当前章节标题
            flush()
            level, title = len(heading.group(1)), heading.group(2).strip()
            if level == 1:
                doc_title = f"# {title}"
                current_section = ""
            else:
                current_section = f"{'#' * level} {title}"
            continue

        if not stripped:
            # 空行仅作为段落边界，不立即落盘：需累积到阈值，
            # 否则片段过碎（每段仅一行），匹配得分难以体现区分度。
            if sum(len(item) for item in buffer) >= _CHUNK_TARGET_CHARS:
                flush()
            continue

        buffer.append(stripped)
        if sum(len(item) for item in buffer) >= _CHUNK_TARGET_CHARS:
            flush()

    flush()
    return chunks


def _score_chunk(chunk: str, query_tokens: list[str], filename: str) -> int:
    """
    对单个片段计算关键词匹配得分。

    Args:
        chunk: 待打分的文本片段。
        query_tokens: 查询分词结果。
        filename: 来源文件名，用于文件名命中加权。

    Returns:
        int: 匹配得分，0 表示无命中。
    """
    lowered_chunk = chunk.lower()
    lowered_name = filename.lower()
    score = 0

    for token in query_tokens:
        hits = lowered_chunk.count(token)
        if hits:
            score += _WEIGHT_KEYWORD * hits
            # 标题行（以 # 开头）命中额外加权
            if re.search(rf"^#+\s*.*{re.escape(token)}", lowered_chunk, re.MULTILINE):
                score += _WEIGHT_TITLE
        if token in lowered_name:
            score += _WEIGHT_FILENAME
    return score


def _format_hits(query: str, hits: list[tuple[int, str, str]]) -> str:
    """
    将检索结果格式化为便于 LLM 阅读的文本。

    Args:
        query: 原始查询。
        hits: ``(得分, 文件名, 片段)`` 三元组列表，已按得分降序排列。

    Returns:
        str: 结构化检索结果文本。
    """
    lines: list[str] = [f"【知识库检索结果】查询：{query}", f"命中片段数：{len(hits)}", ""]
    total_chars = 0
    for index, (score, name, chunk) in enumerate(hits, start=1):
        block = f"--- 片段 {index} | 来源：{name} | 匹配度：{score} ---\n{chunk}"
        if total_chars + len(block) > _MAX_SNIPPET_CHARS:
            # 截断至上限，保证返回长度可控
            remaining = _MAX_SNIPPET_CHARS - total_chars
            if remaining > 0:
                lines.append(block[:remaining] + "\n...(内容已截断)")
            break
        lines.append(block)
        total_chars += len(block)
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# 工具 1：知识库检索
# ---------------------------------------------------------------------------


@tool
def query_knowledge_base(query: str) -> str:
    """检索企业内部知识库，返回与问题最相关的政策或排障步骤文本。

    知识库来源为项目 ``data/docs/`` 目录下的 Markdown 文档，当前包含：
    ``account_faq.md``（企业 SaaS 账号排障指南：401 未授权、403 权限不足、
    密码连续输错锁定）与 ``refund_policy.md``（企业退款政策与工单流转规范）。

    适用场景：客户询问账号登录异常、权限申请、密码锁定、退款条件、
    工单流转规则等问题时，应先调用本工具获取权威政策原文，再组织答复，
    不得凭记忆编造政策条款。

    Args:
        query: 用户问题的自然语言描述，建议包含关键实体，例如
            "401 未授权怎么解决"、"密码输错锁定多久解锁"、"退款条件是什么"。

    Returns:
        str: 匹配到的知识库片段文本（含来源文件名与匹配度）。
            若未命中或知识库为空，返回说明性提示文本，便于模型转人工或改写查询。
    """
    if not query or not query.strip():
        return "【知识库检索失败】查询内容为空，请提供具体问题后再试。"

    documents = _load_documents()
    if not documents:
        return (
            f"【知识库为空】未在目录 {KNOWLEDGE_BASE_DIR} 下找到任何文档，"
            "请确认知识库文件已正确放置。"
        )

    query_tokens = _tokenize(query)
    if not query_tokens:
        return (
            f"【未命中】查询「{query}」未提取到有效关键词，"
            "建议补充具体错误码或业务名词（如 401、403、退款、锁定）。"
        )

    scored_hits: list[tuple[int, str, str]] = []
    for document in documents:
        for chunk in _split_into_chunks(document["content"]):
            score = _score_chunk(chunk, query_tokens, document["name"])
            if score > 0:
                scored_hits.append((score, document["name"], chunk))

    if not scored_hits:
        return (
            f"【未命中】知识库中未找到与「{query}」相关的内容。"
            "建议：1) 换用更具体的关键词重试；2) 若确属新问题，请创建工单转人工处理。"
        )

    # 得分降序，最多返回 3 个片段
    scored_hits.sort(key=lambda item: item[0], reverse=True)
    return _format_hits(query, scored_hits[:3])


# ---------------------------------------------------------------------------
# 工具 2：提交工单
# ---------------------------------------------------------------------------


def _generate_ticket_id() -> str:
    """
    生成形如 ``TK-2026-XXXX`` 的工单流水号。

    Returns:
        str: 工单号，年份取当前系统年份，后缀为 4 位随机数字。
    """
    year = datetime.now().year
    suffix = f"{random.randint(0, 9999):04d}"
    return f"{_TICKET_PREFIX}-{year}-{suffix}"


@tool
def submit_ticket(user_id: str, issue_type: str, description: str) -> str:
    """提交企业工单到企业微信 / OA 协同系统，并返回结构化的提交结果。

    本工具为第一阶段模拟实现：不产生真实网络请求，仅生成工单流水号、
    完成基础参数校验并返回结构化 JSON，用于打通状态机与工具调用的链路。

    调用前请确保三个参数齐备，尤其是 ``user_id`` —— 若用户尚未提供账号，
    应先向用户追问，不得臆造或留空。

    Args:
        user_id: 用户账号唯一标识，用于定位客户主体，例如 ``"U-100238"``。
        issue_type: 问题类型，必须为以下枚举值之一：
            ``account``（账号与登录）、``refund``（退款与结算）、
            ``billing``（账单与发票）、``technical``（技术故障）、
            ``feature``（功能咨询）、``complaint``（投诉与建议）。
        description: 问题详细描述，应包含订单 / 资源标识、发生时间、
            现象与诉求、已尝试动作、影响范围等可核验信息。

    Returns:
        str: 结构化 JSON 字符串，字段包括 ``success``、``ticket_id``、
            ``status``、``routing``、``sla``、``submitted_at`` 等；
            参数校验失败时返回 ``success=false`` 及 ``error`` 说明。
    """
    # ---- 参数校验 ----
    errors: list[str] = []

    cleaned_user_id = (user_id or "").strip()
    cleaned_issue_type = (issue_type or "").strip().lower()
    cleaned_description = (description or "").strip()

    if not cleaned_user_id:
        errors.append("user_id 不能为空")
    if not cleaned_issue_type:
        errors.append("issue_type 不能为空")
    elif cleaned_issue_type not in VALID_ISSUE_TYPES:
        errors.append(
            f"issue_type「{issue_type}」无效，可选值：{', '.join(sorted(VALID_ISSUE_TYPES))}"
        )
    if not cleaned_description:
        errors.append("description 不能为空")
    elif len(cleaned_description) < 10:
        errors.append("description 过短（少于 10 个字符），请补充订单号、时间与具体诉求")

    if errors:
        return json.dumps(
            {
                "success": False,
                "error": "工单参数校验失败",
                "details": errors,
                "hint": "请补齐必需参数（user_id、issue_type、description）后重新提交。",
            },
            ensure_ascii=False,
            indent=2,
        )

    # ---- 生成工单 ----
    ticket_id = _generate_ticket_id()
    submitted_at = datetime.now().astimezone().isoformat(timespec="seconds")
    routing = _ISSUE_ROUTING.get(cleaned_issue_type, "一线技术支持组")

    # SLA 依据 data/docs/refund_policy.md 第 4.1 节
    sla_map = {
        "account": "首次响应 15 分钟",
        "refund": "首次响应 4 小时",
        "billing": "首次响应 8 小时",
        "technical": "首次响应 30 分钟",
        "feature": "首次响应 8 小时",
        "complaint": "首次响应 2 小时",
    }

    payload = {
        "success": True,
        "ticket_id": ticket_id,
        "status": "待受理",
        "message": f"工单已成功提交至企业协同系统，流水号 {ticket_id}，请留意处理进度。",
        "channel": "企业微信 / OA 协同中台",
        "ticket": {
            "user_id": cleaned_user_id,
            "issue_type": cleaned_issue_type,
            "description": cleaned_description,
        },
        "routing": {
            "assigned_group": routing,
            "priority": "P2",
            "sla": sla_map.get(cleaned_issue_type, "首次响应 4 小时"),
        },
        "submitted_at": submitted_at,
        "next_action": "客服将在 SLA 时限内联系您；如需加急请回复工单号。",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 工具导出清单（供后续 LangGraph 状态机统一绑定）
# ---------------------------------------------------------------------------

CUSTOMER_SERVICE_TOOLS = [query_knowledge_base, submit_ticket]

__all__ = [
    "query_knowledge_base",
    "submit_ticket",
    "CUSTOMER_SERVICE_TOOLS",
    "KNOWLEDGE_BASE_DIR",
    "VALID_ISSUE_TYPES",
]
