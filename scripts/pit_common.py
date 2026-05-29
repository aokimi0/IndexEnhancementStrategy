"""点位重建公共工具：baostock 超时重试、代码转换、日志。

设计目标：
- 所有 baostock 调用都包一层硬超时（线程 + future.result(timeout)），单次卡死不拖垮全局。
- 代码格式互转：项目 ts_code(600000.SH) <-> baostock(sh.600000)。
- 统一进度日志写入 logs/pit_rebuild.log。
"""

from __future__ import annotations

import concurrent.futures
import datetime as _dt
import threading
from pathlib import Path
from typing import Any, Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "logs" / "pit_rebuild.log"

_BS_LOCK = threading.Lock()
_BS_MODULE: Any = None
_BS_LOGGED_IN = False
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def log(message: str) -> None:
    """打印并追加写入日志。"""
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def to_ts_code(symbol: str) -> str:
    """6 位代码或带后缀代码 -> ts_code(600000.SH)。"""
    if "." in symbol:
        if symbol.startswith(("sh.", "sz.")):
            prefix, code = symbol.split(".")
            return f"{code}.{prefix.upper()}"
        return symbol
    if symbol.startswith(("600", "601", "603", "605", "688", "689")):
        return f"{symbol}.SH"
    if symbol.startswith(("000", "001", "002", "003", "30")):
        return f"{symbol}.SZ"
    return symbol


def to_baostock_code(ts_code: str) -> str:
    """ts_code(600000.SH) -> baostock(sh.600000)。"""
    normalized = to_ts_code(ts_code)
    symbol, market = normalized.split(".")
    prefix = "sh" if market.upper() == "SH" else "sz"
    return f"{prefix}.{symbol}"


def get_baostock() -> Any:
    """惰性登录 baostock（带锁，进程内单例）。"""
    global _BS_MODULE, _BS_LOGGED_IN
    with _BS_LOCK:
        if _BS_MODULE is None:
            import baostock as bs

            _BS_MODULE = bs
        if not _BS_LOGGED_IN:
            res = _BS_MODULE.login()
            if getattr(res, "error_code", "0") != "0":
                raise RuntimeError(f"baostock 登录失败: {res.error_msg}")
            _BS_LOGGED_IN = True
    return _BS_MODULE


def call_with_timeout(func: Callable[[], Any], timeout: float, label: str) -> Any:
    """在单线程执行 func，超过 timeout 秒抛 TimeoutError（不阻塞主流程）。"""
    future = _EXECUTOR.submit(func)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError as exc:
        raise TimeoutError(f"{label} 超时 ({timeout}s)") from exc


def bs_result_to_frame(result: Any) -> pd.DataFrame:
    """baostock ResultData -> DataFrame。"""
    if getattr(result, "error_code", "0") != "0":
        raise RuntimeError(f"baostock 查询失败: {result.error_msg}")
    rows: list[list[str]] = []
    while result.next():
        rows.append(result.get_row_data())
    if not rows:
        return pd.DataFrame(columns=getattr(result, "fields", []))
    return pd.DataFrame(rows, columns=result.fields)


def bs_query_with_retry(
    query_builder: Callable[[Any], Any],
    label: str,
    timeout: float = 25.0,
    retries: int = 3,
    sleep_seconds: float = 2.0,
) -> pd.DataFrame:
    """带硬超时与重试的 baostock 查询。

    Args:
        query_builder: 接收 bs 模块、返回 ResultData 的函数。
        label: 用于日志的标签。
        timeout: 单次硬超时秒数。
        retries: 最大尝试次数。
        sleep_seconds: 重试间隔。
    """
    import time

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            bs = get_baostock()
            result = call_with_timeout(lambda: query_builder(bs), timeout=timeout, label=label)
            return bs_result_to_frame(result)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            log(f"  [retry {attempt}/{retries}] {label} 失败: {exc}")
            if attempt < retries:
                time.sleep(sleep_seconds)
    raise RuntimeError(f"{label} 最终失败: {last_error}")
