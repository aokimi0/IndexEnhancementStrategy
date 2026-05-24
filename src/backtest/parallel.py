"""基于 joblib 的并行实验框架。

本模块提供三个工具：

1. :func:`run_in_parallel`：把若干无副作用的可调用对象并行执行。
2. :func:`cached_to_disk`：把函数返回值用 ``joblib.dump`` 缓存到本地，命中时
   直接 ``joblib.load`` 取回。
3. :func:`parallel_backtest`：把多份因子面板/变体并行跑回测，便于做参数
   扫描或鲁棒性测试。
"""

from __future__ import annotations

import functools
import hashlib
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from joblib import Parallel, delayed

from src.backtest.engine import BaselineBacktestEngine, BaselineBacktestResult

logger = logging.getLogger(__name__)


def run_in_parallel(
    jobs: Iterable[Callable[[], Any]],
    n_jobs: int = -1,
    backend: str = "loky",
    verbose: int = 0,
) -> list[Any]:
    """并行执行一组无参可调用对象。

    Args:
        jobs: 任务列表，每个元素需是无参可调用对象，例如
            ``functools.partial(fn, **kwargs)``。
        n_jobs: 并发数。``-1`` 表示使用全部物理核心；``1`` 退化为串行。
        backend: joblib 后端，常用 ``"loky"``（默认，跨进程）、``"threading"``
            （线程）、``"multiprocessing"``。
        verbose: joblib 的日志详细程度。

    Returns:
        list[Any]: 与 ``jobs`` 顺序一致的结果列表。
    """
    job_list = list(jobs)
    if not job_list:
        return []
    return Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(job)() for job in job_list
    )


def _default_key_fn(*args: object, **kwargs: object) -> str:
    """根据位置/关键字参数生成默认缓存键。

    Args:
        *args: 位置参数。
        **kwargs: 关键字参数。

    Returns:
        str: 16 位十六进制摘要，可作为文件名前缀。
    """
    payload = repr(args) + "|" + repr(sorted(kwargs.items()))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def cached_to_disk(
    cache_dir: str | Path,
    key_fn: Callable[..., str] | None = None,
    compress: int = 3,
) -> Callable[[Callable], Callable]:
    """把函数返回值持久化到磁盘的装饰器工厂。

    Args:
        cache_dir: 缓存根目录，会自动创建。
        key_fn: 自定义键函数，签名与被装饰函数一致，返回字符串。为空时
            根据参数生成 SHA1 摘要。
        compress: ``joblib.dump`` 的压缩等级，0 表示不压缩。

    Returns:
        Callable: 装饰器，可作用于任意返回值可被 ``joblib`` 序列化的函数。
    """
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    def _decorator(fn: Callable) -> Callable:
        resolved_key_fn = key_fn or _default_key_fn

        @functools.wraps(fn)
        def _wrapper(*args: object, **kwargs: object) -> Any:
            key = resolved_key_fn(*args, **kwargs)
            cache_path = cache_root / f"{fn.__name__}_{key}.joblib"
            if cache_path.exists():
                logger.debug("缓存命中: %s", cache_path)
                return joblib.load(cache_path)
            result = fn(*args, **kwargs)
            joblib.dump(result, cache_path, compress=compress)
            return result

        return _wrapper

    return _decorator


def _run_single_variant(
    variant_name: str,
    panel: pd.DataFrame,
    engine_factory: Callable[[], BaselineBacktestEngine],
) -> tuple[str, BaselineBacktestResult]:
    """在子进程中跑单个变体回测。

    Args:
        variant_name: 变体名称，用作返回字典的 key。
        panel: 当前变体使用的因子面板。
        engine_factory: 引擎工厂函数。每个子进程内部各调用一次以避免共享
            可变状态。

    Returns:
        tuple: ``(variant_name, BaselineBacktestResult)``。
    """
    engine = engine_factory()
    result = engine.run(panel)
    return variant_name, result


def parallel_backtest(
    panels: dict[str, pd.DataFrame],
    engine_factory: Callable[[], BaselineBacktestEngine],
    n_jobs: int = 4,
    backend: str = "loky",
    verbose: int = 0,
) -> dict[str, BaselineBacktestResult]:
    """并行跑多份变体面板的回测。

    每个 ``(variant_name, panel)`` 对在独立子进程中调用 ``engine_factory()``
    构造引擎并执行 :meth:`BaselineBacktestEngine.run`，结果按变体名返回。

    Args:
        panels: ``{variant_name: factor_panel}`` 映射。
        engine_factory: 无参可调用对象，返回一个新的 ``BaselineBacktestEngine``
            实例。建议用 ``functools.partial`` 预绑定不同的回测参数。
        n_jobs: joblib 并发数。
        backend: joblib 后端。
        verbose: joblib 日志详细程度。

    Returns:
        dict[str, BaselineBacktestResult]: 变体名到回测结果的映射，键集合
        与 ``panels`` 一致。
    """
    items = list(panels.items())
    if not items:
        return {}
    results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(_run_single_variant)(name, panel, engine_factory)
        for name, panel in items
    )
    return dict(results)


__all__ = [
    "cached_to_disk",
    "parallel_backtest",
    "run_in_parallel",
]
