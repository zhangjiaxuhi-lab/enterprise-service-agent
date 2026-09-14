"""
安装提交前密钥自检钩子 (pre-commit)。

作用
----
把 ``scripts/secret_scan.py`` 注册为 git 的 ``pre-commit`` 钩子，
使每次 ``git commit`` 都自动扫描暂存区，发现疑似密钥即阻止提交。

用法
----
    python scripts/install_hooks.py            # 安装 / 覆盖
    python scripts/install_hooks.py --uninstall  # 卸载（含清理旧版 .sample）

实现说明
--------
钩子按平台生成两种形态，均为**自包含**、不依赖额外工具：

* **Windows**：``.git/hooks/pre-commit.bat``（批处理）。
  这是刻意的设计选择。Python 版钩子靠 ``#!/usr/bin/env python3`` 启动，而
  ``env`` 属于 MSYS2 组件，在某些受管控的 Windows 环境中会因
  ``CreateFileMapping ... Win32 error 5`` 直接崩溃，导致**钩子静默不执行**——
  比不装钩子更危险。批处理由 cmd.exe 原生执行，路径最稳。
* **POSIX（Linux / macOS）**：``.git/hooks/pre-commit``（sh 脚本），
  依次尝试 python3 / python / py。

Git 查找钩子的顺序为 pre-commit → pre-commit.exe → .bat → .cmd → .sh，
因此 Windows 上只放 ``.bat``、POSIX 上只放无后缀脚本，两者互不干扰。
"""

from __future__ import annotations

import argparse
import os
import stat
import subprocess
import sys
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
SCANNER_REL: str = "scripts/secret_scan.py"
HOOK_MARKER: str = "DSH secret-scan pre-commit hook"

IS_WINDOWS: bool = os.name == "nt"

# ---------------------------------------------------------------------------
# Windows：批处理钩子
# ---------------------------------------------------------------------------

HOOK_BAT: str = f"""@echo off
REM >>> {HOOK_MARKER} >>>
REM 本文件由 scripts/install_hooks.py 自动生成，请勿手工修改。
REM 手工修改会在下次安装时被覆盖；如需调整规则，请改 scripts/secret_scan.py。
setlocal

REM 定位仓库根目录（本文件位于 <root>\\.git\\hooks\\）
set "REPO_ROOT=%~dp0..\\.."
pushd "%REPO_ROOT%"
if errorlevel 1 (
    echo [pre-commit] 无法进入仓库根目录，跳过密钥自检。 1>&2
    exit /b 0
)

set "SCANNER={SCANNER_REL.replace('/', chr(92))}"
if not exist "%SCANNER%" (
    echo [pre-commit] 未找到扫描器 %SCANNER%，跳过密钥自检。 1>&2
    echo [pre-commit] 如需恢复：python scripts\\install_hooks.py 1>&2
    popd
    exit /b 0
)

REM 依次尝试 py 启动器与 python，取第一个可用的
where py >nul 2>nul && (set "PYRUN=py -3") || (set "PYRUN=python")

%PYRUN% "%SCANNER%" --staged
set "RC=%ERRORLEVEL%"
popd

REM 仅透传 0（通过）与 1（发现密钥）；其他情况视为环境异常，不阻塞提交
if "%RC%"=="0" exit /b 0
if "%RC%"=="1" exit /b 1
exit /b 0
REM <<< {HOOK_MARKER} <<<
"""

# ---------------------------------------------------------------------------
# POSIX：sh 钩子
# ---------------------------------------------------------------------------

HOOK_SH: str = f"""#!/bin/sh
# >>> {HOOK_MARKER} >>>
# 本文件由 scripts/install_hooks.py 自动生成，请勿手工修改。
# 手工修改会在下次安装时被覆盖；如需调整规则，请改 scripts/secret_scan.py。

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT" || exit 0

SCANNER="{SCANNER_REL}"
if [ ! -f "$SCANNER" ]; then
    echo "[pre-commit] 未找到扫描器 $SCANNER，跳过密钥自检。" >&2
    echo "[pre-commit] 如需恢复：python scripts/install_hooks.py" >&2
    exit 0
fi

# 依次尝试可用的 Python 解释器
if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
elif command -v py >/dev/null 2>&1; then
    PY=py
else
    echo "[pre-commit] 未找到 Python 解释器，跳过密钥自检。" >&2
    exit 0
fi

"$PY" "$SCANNER" --staged
RC=$?

# 仅透传 0（通过）与 1（发现密钥）
if [ "$RC" = "1" ]; then exit 1; fi
exit 0
# <<< {HOOK_MARKER} <<<
"""


def git(*args: str) -> tuple[int, str]:
    """
    执行 git 命令。

    Args:
        *args: git 子命令。

    Returns:
        tuple[int, str]: (退出码, 标准输出)。
    """
    proc = subprocess.run(
        ["git", *args], capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    return proc.returncode, proc.stdout.strip()


def hooks_dir() -> Path | None:
    """
    解析实际生效的 hooks 目录（尊重 ``core.hooksPath`` 配置）。

    Returns:
        Path | None: 目录路径；非 git 仓库时返回 None。
    """
    code, out = git("rev-parse", "--git-dir")
    if code != 0:
        return None

    code, custom = git("config", "core.hooksPath")
    if code == 0 and custom:
        path = Path(custom)
        return path if path.is_absolute() else (REPO_ROOT / path)

    git_dir = Path(out)
    if not git_dir.is_absolute():
        git_dir = (REPO_ROOT / git_dir).resolve()
    return git_dir / "hooks"


def install(force: bool = True) -> int:
    """
    安装 pre-commit 钩子（按当前平台选择形态）。

    Args:
        force: 已存在非本工具生成的钩子时是否覆盖（会先备份）。

    Returns:
        int: 进程退出码。
    """
    target_dir = hooks_dir()
    if target_dir is None:
        print("✗ 当前目录不是 git 仓库，无法安装钩子。", file=sys.stderr)
        return 2

    target_dir.mkdir(parents=True, exist_ok=True)
    scanner = REPO_ROOT / SCANNER_REL
    if not scanner.is_file():
        print(f"✗ 未找到扫描器 {SCANNER_REL}，请勿删除该文件。", file=sys.stderr)
        return 2

    # 目标文件名：Windows 用 .bat（cmd.exe 原生执行），POSIX 用无后缀脚本
    hook_path = target_dir / ("pre-commit.bat" if IS_WINDOWS else "pre-commit")
    body = HOOK_BAT if IS_WINDOWS else HOOK_SH

    # 已存在他人编写的同名钩子时先备份，避免破坏用户配置
    if hook_path.is_file():
        existing = hook_path.read_text(encoding="utf-8", errors="ignore")
        if HOOK_MARKER not in existing:
            if not force:
                print(f"✗ 已存在自定义钩子：{hook_path}（未覆盖）", file=sys.stderr)
                return 1
            backup = hook_path.with_name(hook_path.name + ".bak")
            backup.write_text(existing, encoding="utf-8")
            print(f"⚠ 检测到已存在的钩子，已备份至：{backup.name}")

    # 清理旧版无后缀钩子：早期实现依赖 shebang（#!/usr/bin/env python3），
    # 在受管控的 Windows 环境中会因 MSYS2 env.exe 失败而**静默不执行**，
    # 必须移除以免与 .bat 钩子混淆、造成"以为有防护其实没有"的假象。
    if IS_WINDOWS:
        legacy = target_dir / "pre-commit"
        if legacy.is_file():
            content = legacy.read_text(encoding="utf-8", errors="ignore")
            if HOOK_MARKER in content or content.lstrip().startswith("#!"):
                legacy.unlink()
                print(f"· 已移除失效的旧版钩子：{legacy.name}")

    # 批处理必须使用 CRLF 换行，否则 cmd.exe 在部分环境下解析异常
    if IS_WINDOWS:
        hook_path.write_bytes(body.replace("\n", "\r\n").encode("utf-8"))
    else:
        hook_path.write_text(body, encoding="utf-8")
        try:
            hook_path.chmod(
                hook_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
            )
        except OSError:
            pass

    print(f"✅ 已安装 pre-commit 钩子：{hook_path}")
    print(f"   形态：{'Windows 批处理 (.bat)' if IS_WINDOWS else 'POSIX sh 脚本'}")
    print(f"   扫描器：{SCANNER_REL}")
    print("   生效范围：每次 git commit 的暂存内容")
    print("   临时跳过：git commit --no-verify")
    print("   验证：python scripts/secret_scan.py --staged")
    return 0


def uninstall() -> int:
    """
    卸载钩子并恢复备份（如存在）。

    Returns:
        int: 进程退出码。
    """
    target_dir = hooks_dir()
    if target_dir is None:
        print("✗ 当前目录不是 git 仓库。", file=sys.stderr)
        return 2

    removed = False
    # 两种形态都清理，避免跨平台切换时残留
    for name in ("pre-commit", "pre-commit.bat"):
        hook_path = target_dir / name
        if not hook_path.is_file():
            continue
        content = hook_path.read_text(encoding="utf-8", errors="ignore")
        if HOOK_MARKER not in content:
            print(f"✗ {name} 非本工具安装，已保留。", file=sys.stderr)
            continue
        hook_path.unlink()
        print(f"✅ 已卸载钩子：{hook_path}")
        removed = True

        backup = hook_path.with_name(hook_path.name + ".bak")
        if backup.is_file():
            backup.rename(hook_path)
            print(f"   已恢复原有钩子（来自 {backup.name}）")

    if not removed:
        print("· 未发现由本工具安装的钩子，无需卸载。")
    return 0


def main(argv: list[str] | None = None) -> int:
    """
    命令行入口。

    Args:
        argv: 参数列表。

    Returns:
        int: 进程退出码。
    """
    parser = argparse.ArgumentParser(description="安装/卸载提交前密钥自检钩子")
    parser.add_argument("--uninstall", action="store_true", help="卸载钩子")
    parser.add_argument("--no-force", action="store_true",
                        help="已存在自定义钩子时不覆盖")
    args = parser.parse_args(argv)

    if args.uninstall:
        return uninstall()
    return install(force=not args.no_force)


if __name__ == "__main__":
    raise SystemExit(main())
