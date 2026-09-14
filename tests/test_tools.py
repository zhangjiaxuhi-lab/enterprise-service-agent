"""
业务工具测试（``src/tools/customer_service_tools.py``）。

覆盖两个工具的**契约**：参数校验、输出格式、检索正确性、以及
「缺参不得静默成功」这一安全底线。
"""

from __future__ import annotations

import json
import re

import pytest

from src.tools.customer_service_tools import (
    KNOWLEDGE_BASE_DIR,
    VALID_ISSUE_TYPES,
    query_knowledge_base,
    submit_ticket,
)

# 工单号契约：TK-YYYY-XXXX
TICKET_ID_PATTERN = re.compile(r"^TK-\d{4}-\d{4}$")


# ---------------------------------------------------------------------------
# query_knowledge_base
# ---------------------------------------------------------------------------


class TestQueryKnowledgeBase:
    """知识库检索工具。"""

    def test_knowledge_base_dir_exists(self) -> None:
        """知识库目录必须存在且含 Markdown 文档。"""
        assert KNOWLEDGE_BASE_DIR.is_dir(), f"知识库目录缺失：{KNOWLEDGE_BASE_DIR}"
        docs = list(KNOWLEDGE_BASE_DIR.glob("*.md"))
        assert docs, "知识库中没有任何 .md 文档"

    @pytest.mark.parametrize(
        "question,expected_source",
        [
            ("403 权限不足怎么解决？", "account_faq"),
            ("401 未授权怎么处理？", "account_faq"),
            ("密码连续输错账号锁定多久解锁？", "account_faq"),
            ("退款条件是什么？", "refund_policy"),
            ("工单需要哪些参数？", "refund_policy"),
        ],
    )
    def test_retrieves_expected_document(self, question: str, expected_source: str) -> None:
        """高频问题应命中预期文档（防止检索逻辑被改坏）。"""
        result = query_knowledge_base.invoke(question)
        assert "命中片段数：0" not in result, f"「{question}」零命中"
        assert expected_source in result, (
            f"「{question}」未命中 {expected_source}，实际返回：{result[:200]}"
        )

    def test_output_contains_source_and_score(self) -> None:
        """输出应包含来源文件名与匹配度，便于追溯依据。"""
        result = query_knowledge_base.invoke("403 权限不足")
        assert "来源：" in result
        assert "匹配度：" in result
        assert ".md" in result

    def test_unmatched_query_returns_graceful_hint(self) -> None:
        """无关问题应返回可操作的兜底提示，而不是抛异常或瞎编。"""
        result = query_knowledge_base.invoke("量子引力波咖啡机的冲泡温度是多少")
        assert "未命中" in result
        assert "工单" in result or "关键词" in result, "兜底提示应给出下一步建议"

    def test_empty_query_is_rejected(self) -> None:
        """空查询应被明确拒绝。"""
        assert "失败" in query_knowledge_base.invoke("") or "为空" in query_knowledge_base.invoke("")

    def test_stopword_only_query_is_handled(self) -> None:
        """仅含停用词/单字的查询不应崩溃。"""
        result = query_knowledge_base.invoke("的了吗呢")
        assert isinstance(result, str) and result

    def test_output_length_is_bounded(self) -> None:
        """输出长度应受控，避免把整份文档灌进模型上下文。"""
        result = query_knowledge_base.invoke("退款 政策 条件 工单 账号 权限")
        assert len(result) <= 2000, f"返回过长：{len(result)} 字符"


# ---------------------------------------------------------------------------
# submit_ticket
# ---------------------------------------------------------------------------


class TestSubmitTicket:
    """工单提交工具。"""

    def test_success_payload_contract(self) -> None:
        """成功时返回结构化 JSON，且字段齐备。"""
        raw = submit_ticket.invoke(
            {
                "user_id": "U-100238",
                "issue_type": "refund",
                "description": "订单号 SO-20260115-0087 已支付年费未开通，申请全额退款。",
            }
        )
        payload = json.loads(raw)
        assert payload["success"] is True
        assert TICKET_ID_PATTERN.match(payload["ticket_id"]), (
            f"流水号格式不符：{payload['ticket_id']}"
        )
        assert payload["status"] == "待受理"
        assert payload["ticket"]["user_id"] == "U-100238"
        assert payload["ticket"]["issue_type"] == "refund"
        assert payload["routing"]["assigned_group"], "缺少受理组"
        assert payload["submitted_at"], "缺少提交时间"

    @pytest.mark.parametrize("issue_type", sorted(VALID_ISSUE_TYPES))
    def test_all_valid_issue_types_accepted(self, issue_type: str) -> None:
        """全部合法 issue_type 都应受理并路由到对应处理组。"""
        payload = json.loads(
            submit_ticket.invoke(
                {
                    "user_id": "U-1",
                    "issue_type": issue_type,
                    "description": "这是一个足够长的测试描述，用于验证路由逻辑。",
                }
            )
        )
        assert payload["success"] is True
        assert payload["routing"]["assigned_group"], f"{issue_type} 未分配处理组"

    def test_issue_type_is_case_insensitive(self) -> None:
        """大小写不应影响枚举匹配。"""
        payload = json.loads(
            submit_ticket.invoke(
                {
                    "user_id": "U-1",
                    "issue_type": "REFUND",
                    "description": "大小写测试描述，长度足够触发校验通过。",
                }
            )
        )
        assert payload["success"] is True
        assert payload["ticket"]["issue_type"] == "refund"

    def test_invalid_issue_type_rejected(self) -> None:
        """非法枚举必须被拒绝，并提示可选值。"""
        payload = json.loads(
            submit_ticket.invoke(
                {
                    "user_id": "U-1",
                    "issue_type": "foo",
                    "description": "非法类型的测试描述，长度足够。",
                }
            )
        )
        assert payload["success"] is False
        assert any("issue_type" in d for d in payload["details"])

    @pytest.mark.parametrize(
        "user_id,issue_type,description,expected_field",
        [
            ("", "refund", "描述足够长的测试内容用于校验。", "user_id"),
            ("U-1", "", "描述足够长的测试内容用于校验。", "issue_type"),
            ("U-1", "refund", "", "description"),
            ("U-1", "refund", "太短", "description"),
        ],
    )
    def test_missing_or_invalid_params_rejected(
        self, user_id: str, issue_type: str, description: str, expected_field: str
    ) -> None:
        """缺参或描述过短必须返回 ``success=false``（绝不静默成功）。"""
        payload = json.loads(
            submit_ticket.invoke(
                {
                    "user_id": user_id,
                    "issue_type": issue_type,
                    "description": description,
                }
            )
        )
        assert payload["success"] is False, f"缺 {expected_field} 却成功了：{payload}"
        assert any(expected_field in d for d in payload["details"]), (
            f"未指出缺失字段 {expected_field}：{payload['details']}"
        )

    def test_whitespace_only_values_rejected(self) -> None:
        """纯空白字符应视同为空。"""
        payload = json.loads(
            submit_ticket.invoke(
                {
                    "user_id": "   ",
                    "issue_type": "refund",
                    "description": "       ",
                }
            )
        )
        assert payload["success"] is False

    def test_ticket_ids_are_generated_per_call(self) -> None:
        """流水号应随机生成；多次调用不应恒定返回同一编号。"""
        ids = set()
        for _ in range(20):
            payload = json.loads(
                submit_ticket.invoke(
                    {
                        "user_id": "U-1",
                        "issue_type": "account",
                        "description": "批量流水号测试描述，长度足够通过校验。",
                    }
                )
            )
            ids.add(payload["ticket_id"])
        assert len(ids) > 1, "流水号未随机化，20 次调用返回了同一个编号"
