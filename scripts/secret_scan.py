#!/usr/bin/env python3
"""
提交前密钥自检（pre-commit secret scanner）。

用途
----
作为 git ``pre-commit`` 钩子的执行体，扫描**本次提交暂存的内容**，
一旦发现疑似 API Key / 令牌 / 私钥，立即中止提交并给出精确位置。

设计原则
--------
1. **只看暂存内容**（``git diff --cached``），而非工作区文件。
   这保证「检查的内容」与「将要提交的内容」完全一致。
2. **永不回显完整密钥**。命中时只打印脱敏形式（前 4 位 + 长度 + 后 2 位），
   避免把密钥二次写入终端日志、CI 日志或聊天记录。
3. **零第三方依赖**，仅用标准库，确保任何环境都能跑。
4. **低误报**。通过「占位符白名单 + 熵值判定」区分真实密钥与文档示例，
   避免把 ``your_key_here`` / ``sk-xxxx`` 这类模板误判为泄漏。
5. **可审计可绕过**。支持全仓历史扫描；也允许 `git commit --no-verify` 强制
   跳过（git 原生行为），但会明确提示风险。

扫描模式
--------
* ``--staged``（默认）：检查暂存区，供 pre-commit 钩子调用。
* ``--all``          ：检查整个仓库的**跟踪文件**当前内容。
* ``--history``      ：检查全部历史提交，用于核查「密钥是否曾经提交过」。
* ``--range A..B``   ：检查指定提交区间的变更。

退出码
------
0 = 未发现疑似密钥（允许提交）
1 = 发现疑似密钥（阻止提交）
2 = 执行环境异常（如不在 git 仓库中）
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# 强特征规则：一旦命中即判定为密钥（不参与熵值判定）
# ---------------------------------------------------------------------------

# 说明：每条规则给出「人类可读描述」、正则与一个布尔标记。
# 正则只匹配密钥本身，因此命中片段可直接用于脱敏输出，不会带出整行上下文。
#
# 第三个字段 filter_placeholders 的意义：
#   True  —— 命中后仍需通过 is_placeholder 判定。用于「形状相似」的规则，
#            例如 sk- 开头的文档示例 `sk-xxxxxxxx` 与真实密钥前缀一致，
#            必须靠多样性/重复段判定区分，否则文档会大量误报。
#   False —— 形状本身即为确凿证据，不做占位符过滤。典型是 PEM 私钥头
#            `-----BEGIN ... PRIVATE KEY-----`：它含有 `-----` 这种长重复段，
#            若交给占位符规则会被误判为示例而**漏报真实私钥**，必须豁免。
STRONG_PATTERNS: list[tuple[str, re.Pattern[str], bool]] = [
    ("OpenAI / DashScope / 通义千问 密钥 (sk-)",
     re.compile(r"sk-[A-Za-z0-9_\-]{16,}"), True),
    ("Anthropic 密钥 (sk-ant-)",
     re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,}"), True),
    ("Google API 密钥 (AIza)", re.compile(r"AIza[0-9A-Za-z_\-]{35}"), True),
    ("AWS Access Key ID (AKIA)", re.compile(r"(?:AKIA|ASIA)[0-9A-Z]{16}"), True),
    ("GitHub 令牌 (ghp_/gho_/ghu_/ghs_/ghr_)",
     re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"), True),
    ("GitHub 细粒度令牌 (github_pat_)",
     re.compile(r"github_pat_[A-Za-z0-9_]{22,}"), True),
    ("Slack 令牌 (xox)", re.compile(r"xox[aboprs]-[A-Za-z0-9\-]{10,}"), True),
    # 私钥：形状即证据，豁免占位符过滤（见上方说明）
    ("私钥文件内容 (PRIVATE KEY)",
     re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"), False),
    ("数据库 URI 内嵌密码",
     re.compile(r"(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://[^:@\s/]+:[^@\s/]{8,}@"),
     True),
]

# 弱特征规则：形如 token = "xxxx"。需再通过「熵值 + 白名单」判定，
# 否则 README 里的示例赋值会大量误报。
WEAK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"""(?ix)                       # 忽略大小写、允许换行内空格
        \b(?:api[_\-]?key|apikey|access[_\-]?token|auth[_\-]?token
           |secret[_\-]?key|client[_\-]?secret|private[_\-]?key
           |dashscope[_\-]?api[_\-]?key|openai[_\-]?api[_\-]?key)\b
        \s*[:=]\s*
        ["']?([A-Za-z0-9_\-+/=]{16,})["']?
        """
    ),
]

# 占位符 / 文档示例关键词：出现在候选中即视为非密钥
PLACEHOLDER_MARKERS: tuple[str, ...] = (
    "your", "yourkey", "yourapikey", "placeholder", "example", "sample", "dummy",
    "fake", "test", "demo", "todo", "changeme", "change_me", "redacted",
    "xxxx", "yyy", "zzz", "abcabc", "12345678", "0000", "none", "null", "undefined",
)

# 熵值阈值（比特/字符）。真实随机密钥通常 > 3.5；自然语言/重复串更低。
ENTROPY_THRESHOLD: float = 3.2
MIN_CANDIDATE_LEN: int = 16

# 默认放行的文件名（仅含占位符的模板/文档）
#
# 注意：这里**刻意不放行 *.md / docs/**。虽然文档里常有示例，但真实事故中
# 「把密钥贴进 README」极为常见，若整类放行等于开了后门。文档示例交由
# 「占位符白名单 + 熵值判定」识别，因此 your_key_here、sk-xxxx 不会误报。
ALLOWLIST_NAMES: frozenset[str] = frozenset(
    {
        ".env.example", ".env.sample", ".env.template", ".env.dist",
        "secret_scan.py",            # 本文件自身含正则与关键词字面量
        ".secretsignore",
    }
)

# 默认放行的路径（正则）：仅限不应参与扫描的区域
ALLOWLIST_PATH_PARTS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(^|/)\.git/"),
    re.compile(r"(^|/)__pycache__/"),
    re.compile(r"(^|/)node_modules/"),
)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def shannon_entropy(text: str) -> float:
    """
    计算字符串的 Shannon 熵（比特/字符），用于区分随机密钥与自然语言。

    Args:
        text: 待计算文本。

    Returns:
        float: 熵值；空串返回 0.0。
    """
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum(
        (count / length) * math.log2(count / length) for count in counts.values()
    )


def mask(secret: str) -> str:
    """
    将密钥脱敏为可安全打印的形式。

    只保留前 4 位与后 2 位，中间以固定长度掩码替代，
    并标注真实长度——既便于定位误报，又不泄露可用信息。

    Args:
        secret: 原始密钥片段。

    Returns:
        str: 形如 ``sk-w***…***Q (len=115)`` 的脱敏文本。
    """
    if len(secret) <= 8:
        return f"{'*' * len(secret)} (len={len(secret)})"
    return f"{secret[:4]}***…***{secret[-2:]} (len={len(secret)})"


def is_placeholder(candidate: str) -> bool:
    """
    判断候选串是否为占位符或文档示例。

    采用四项启发式判定，任一命中即视为占位符：

    1. **关键词命中**：含 ``your``、``example``、``changeme``、``xxxx`` 等标记；
    2. **字符多样性过低**：唯一字符少于 6 个；
    3. **存在长重复段**：连续 4 个及以上相同字符（如 ``sk-xxxxxxxx…``）。
       注意必须用 ``finditer`` 取 ``group(0)`` 得到完整重复段——
       ``findall`` 在含捕获组时只返回组内容，会漏判。
    4. **单一字符占比过高**：某字符占比超过 40%（如 ``AKIAAAAAAAAAAAAAAAAA``）。

    Args:
        candidate: 待判定的候选密钥串。

    Returns:
        bool: True 表示应放行（非真实密钥）。
    """
    lowered = candidate.lower()
    if any(marker in lowered for marker in PLACEHOLDER_MARKERS):
        return True
    if len(set(candidate)) < 6:
        return True
    # 长重复段：取完整匹配（group(0)）而非捕获组
    if any(len(m.group(0)) >= 4 for m in re.finditer(r"(.)\1{3,}", candidate)):
        return True
    return max(Counter(candidate).values()) / len(candidate) > 0.4


def load_secretsignore() -> list[re.Pattern[str]]:
    """
    读取仓库根目录的 ``.secretsignore``，返回用户自定义放行规则。

    每行一条正则（``#`` 开头为注释，空行忽略），匹配**文件路径**。
    这是误报的正式出口：把确实含示例的文件列入，而不是放宽全局规则。

    Returns:
        list[re.Pattern[str]]: 编译后的正则列表；文件不存在或格式错误时返回空列表。
    """
    ignore_file = Path(".secretsignore")
    if not ignore_file.is_file():
        return []

    patterns: list[re.Pattern[str]] = []
    try:
        for line in ignore_file.read_text(encoding="utf-8").splitlines():
            rule = line.strip()
            if not rule or rule.startswith("#"):
                continue
            try:
                patterns.append(re.compile(rule))
            except re.error:
                print(f"⚠ .secretsignore 中无效正则，已忽略：{rule}", file=sys.stderr)
    except OSError:
        return []
    return patterns


# 用户自定义放行规则（模块加载时读取一次）
USER_IGNORE_PATTERNS: list[re.Pattern[str]] = load_secretsignore()


def is_allowlisted_path(path: str) -> bool:
    """
    判断文件路径是否在放行名单中。

    优先级：内置白名单 > 路径正则 > 用户 ``.secretsignore`` 自定义规则。

    Args:
        path: 仓库内相对路径。

    Returns:
        bool: True 表示跳过扫描。
    """
    if Path(path).name in ALLOWLIST_NAMES:
        return True
    if any(pattern.search(path) for pattern in ALLOWLIST_PATH_PARTS):
        return True
    return any(pattern.search(path) for pattern in USER_IGNORE_PATTERNS)


def looks_like_secret(candidate: str) -> bool:
    """
    对弱特征候选做熵值与长度判定。

    Args:
        candidate: 候选串。

    Returns:
        bool: True 表示疑似真实密钥。
    """
    if len(candidate) < MIN_CANDIDATE_LEN:
        return False
    if is_placeholder(candidate):
        return False
    return shannon_entropy(candidate) >= ENTROPY_THRESHOLD


# ---------------------------------------------------------------------------
# 扫描目标：行级定位
# ---------------------------------------------------------------------------


class Finding:
    """一条命中记录。"""

    __slots__ = ("path", "line", "rule", "masked")

    def __init__(self, path: str, line: int, rule: str, masked: str) -> None:
        self.path = path
        self.line = line
        self.rule = rule
        self.masked = masked

    def render(self) -> str:
        """渲染为单行报告文本。"""
        location = f"{self.path}:{self.line}" if self.line > 0 else self.path
        return f"  [{self.rule}] {location}  →  {self.masked}"


def scan_text(path: str, text: str) -> list[Finding]:
    """
    扫描一段文本（保留行号）。

    Args:
        path: 用于报告的文件路径。
        text: 文本内容。

    Returns:
        list[Finding]: 命中列表。
    """
    findings: list[Finding] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        # 记录本行已被强特征覆盖的字符区间，避免同一密钥被弱特征重复报告
        claimed: list[tuple[int, int]] = []

        # ---- 强特征 ----
        # 是否做占位符过滤由规则自身声明（见 STRONG_PATTERNS 第三个字段）：
        # sk- 等「形状相似」规则需过滤以抑制文档误报；
        # PEM 私钥头等「形状即证据」规则必须豁免，否则真实私钥会漏报。
        for rule, pattern, filter_placeholders in STRONG_PATTERNS:
            for match in pattern.finditer(line):
                claimed.append(match.span())
                if filter_placeholders and is_placeholder(match.group(0)):
                    continue
                findings.append(Finding(path, lineno, rule, mask(match.group(0))))

        # ---- 弱特征（需熵值判定，且跳过已被强特征命中的区间）----
        for pattern in WEAK_PATTERNS:
            for match in pattern.finditer(line):
                start, end = match.span(1)
                if any(start < c_end and end > c_start for c_start, c_end in claimed):
                    continue
                candidate = match.group(1)
                if looks_like_secret(candidate):
                    claimed.append((start, end))
                    findings.append(
                        Finding(path, lineno, "疑似密钥赋值（高熵）", mask(candidate))
                    )
    return findings


def parse_staged_diff(raw: str) -> dict[str, tuple[str, int]]:
    """
    解析 ``git diff --cached`` 输出，提取**新增行**及其在目标文件中的行号。

    Args:
        raw: diff 文本。

    Returns:
        dict[str, tuple[str, int]]: 路径 -> (新增内容, 首个新增行的行号)。
            行号用于把命中位置换算成真实文件行号。
    """
    result: dict[str, tuple[str, int]] = {}
    current_path: str | None = None
    added_lines: list[str] = []
    new_start: int = 0
    hunk_start: int = 0

    def flush() -> None:
        if current_path and added_lines:
            result[current_path] = ("\n".join(added_lines), new_start or hunk_start)

    for line in raw.splitlines():
        if line.startswith("+++ b/"):
            flush()
            current_path = line[len("+++ b/"):]
            added_lines = []
            new_start = 0
            continue
        if line.startswith("+++ /dev/null"):
            # 文件被删除，无需扫描
            flush()
            current_path = None
            added_lines = []
            continue
        if line.startswith("@@"):
            match = re.search(r"\+(\d+)", line)
            hunk_start = int(match.group(1)) if match else 0
            if not new_start:
                new_start = hunk_start
            continue
        if line.startswith("+") and not line.startswith("+++"):
            added_lines.append(line[1:])

    flush()
    return result


# ---------------------------------------------------------------------------
# 扫描模式
# ---------------------------------------------------------------------------


def git(*args: str) -> tuple[int, str]:
    """
    执行 git 命令。

    Args:
        *args: git 子命令及参数。

    Returns:
        tuple[int, str]: (退出码, 标准输出)。
    """
    proc = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode, proc.stdout


def scan_staged() -> list[Finding]:
    """
    扫描暂存区内容（pre-commit 钩子的主路径）。

    Returns:
        list[Finding]: 命中列表。
    """
    code, raw = git("diff", "--cached", "--unified=0", "--no-color",
                    "--no-ext-diff", "--text", "HEAD")
    if code != 0:
        # 首次提交（尚无 HEAD）时改用空树对比
        code, raw = git("diff", "--cached", "--unified=0", "--no-color",
                        "--no-ext-diff", "--text", "4b825dc642cb6eb9a060e54bf8d69288fbee4904")
        if code != 0:
            return []

    findings: list[Finding] = []
    for path, (text, start_line) in parse_staged_diff(raw).items():
        if is_allowlisted_path(path):
            continue
        # 后处理：把局部行号换算为真实文件行号
        for finding in scan_text(path, text):
            finding.line = start_line + finding.line - 1
            findings.append(finding)
    return findings


def scan_tracked() -> list[Finding]:
    """
    扫描所有被跟踪文件的当前内容（``--all``）。

    Returns:
        list[Finding]: 命中列表。
    """
    code, files = git("ls-files")
    if code != 0:
        return []

    findings: list[Finding] = []
    for path in files.splitlines():
        path = path.strip()
        if not path or is_allowlisted_path(path):
            continue
        try:
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        findings.extend(scan_text(path, text))
    return findings


def scan_history() -> list[Finding]:
    """
    扫描全部历史提交的变更内容（``--history``）。

    用于核查「密钥是否曾经被提交过」——这是判断是否需要轮换密钥的关键依据。
    按「提交 + 文件」维度统计，同一密钥在多个提交中重复出现会分别列出，
    因为它们的处置方式不同（越早的提交越需要重写历史）。

    Returns:
        list[Finding]: 命中列表，``path`` 形如 ``文件 (commit ab12cd34)``。
    """
    code, commits = git("rev-list", "--all")
    if code != 0:
        return []

    findings: list[Finding] = []
    for commit in commits.split():
        code, raw = git("show", "--unified=0", "--no-color", "--no-ext-diff",
                        "--text", "--format=", commit)
        if code != 0:
            continue
        # 复用 diff 解析：逐个文件提取新增内容，保持文件维度的定位能力
        for path, (text, start_line) in parse_staged_diff(raw).items():
            if is_allowlisted_path(path):
                continue
            for finding in scan_text(path, text):
                finding.line = start_line + finding.line - 1
                finding.path = f"{path} (commit {commit[:8]})"
                findings.append(finding)
    return findings


# ---------------------------------------------------------------------------
# 输出与主流程
# ---------------------------------------------------------------------------


def report(findings: list[Finding], mode: str) -> int:
    """
    打印扫描结果并返回退出码。

    Args:
        findings: 命中列表。
        mode: 模式描述，用于报告标题。

    Returns:
        int: 0（通过）或 1（发现疑似密钥）。
    """
    print("=" * 74)
    if not findings:
        print(f"🔒 密钥自检通过：{mode}未发现疑似密钥。")
        print("=" * 74)
        return 0

    print(f"🚫 密钥自检拦截：{mode}发现 {len(findings)} 处疑似密钥/令牌！")
    print("=" * 74)
    for finding in findings:
        print(finding.render())
    print()
    print("已阻止本次操作。请按以下顺序处置：")
    print("  1. 若确认是真实密钥：立即到对应平台**吊销并重建**该密钥，")
    print("     然后从文件中移除，改用环境变量（.env，且确保已被 .gitignore 忽略）。")
    print("     注意：密钥一旦提交，仅删除文件无效，必须轮换。")
    print("  2. 若确认是误报（文档示例/占位符）：将其改为明显的占位形式，")
    print("     如 your_key_here / sk-xxxxxxxx，或把该文件加入 .secretsignore。")
    print("  3. 确认无误、坚持提交：git commit --no-verify（风险自负）。")
    print("=" * 74)
    return 1


def main(argv: list[str] | None = None) -> int:
    """
    命令行入口。

    Args:
        argv: 参数列表（默认取 sys.argv）。

    Returns:
        int: 退出码。
    """
    parser = argparse.ArgumentParser(
        description="提交前密钥自检：扫描 sk- 等密钥/令牌模式并阻止提交。"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--staged", action="store_true",
                       help="扫描暂存区（默认，供 pre-commit 钩子使用）")
    group.add_argument("--all", action="store_true",
                       help="扫描所有跟踪文件的当前内容")
    group.add_argument("--history", action="store_true",
                       help="扫描全部历史提交（用于核查是否曾经泄漏）")
    args = parser.parse_args(argv)

    # 确认处于 git 仓库中
    code, _ = git("rev-parse", "--is-inside-work-tree")
    if code != 0:
        print("⚠ 当前目录不是 git 仓库，跳过密钥自检。", file=sys.stderr)
        return 0  # 不阻塞非 git 场景

    if args.all:
        return report(scan_tracked(), "全量跟踪文件 ")
    if args.history:
        return report(scan_history(), "全量历史提交 ")
    return report(scan_staged(), "暂存区 ")


if __name__ == "__main__":
    raise SystemExit(main())
