"""终端输出相关工具。"""

from __future__ import annotations

import os
import sys


def configure_console_output() -> None:
    """在 Windows 终端下尽量统一为 UTF-8 输出。

    该函数用于缓解 PowerShell / cmd 中中文日志乱码问题。
    对不支持 `reconfigure()` 的流会自动跳过，不影响主流程。
    """
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                continue
