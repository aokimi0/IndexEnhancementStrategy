"""回测引擎底层数值核函数。

本模块提供两个核心算子的高性能实现：

1. ``compute_nav_loop``：按日累乘组合净值与基准净值，输出收益序列。
2. ``compute_turnovers``：按权重向量计算单边换手率。

默认走 Numba JIT 编译路径（``@njit(cache=True)``，编译产物缓存到
``__pycache__``），当 Numba 不可用或显式禁用时，自动回落到纯 NumPy 向量化
实现，保证接口与数值一致。
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


try:
    from numba import njit as _njit

    HAS_NUMBA = True
except ImportError:  # pragma: no cover - 运行环境无 numba 时降级
    HAS_NUMBA = False
    logger.warning(
        "未检测到 numba 包，回测核函数将回落到纯 numpy 实现，性能可能下降。"
    )

    def _njit(*args: object, **kwargs: object) -> Callable:
        """numba.njit 的空操作替身。

        Args:
            *args: 与 ``numba.njit`` 兼容的位置参数。
            **kwargs: 与 ``numba.njit`` 兼容的关键字参数。

        Returns:
            Callable: 原函数本身或一个透传装饰器。
        """
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _decorator(fn: Callable) -> Callable:
            return fn

        return _decorator


@_njit(cache=True)
def _compute_nav_loop_numba(
    daily_returns_matrix: np.ndarray,
    weights_matrix: np.ndarray,
    benchmark_returns: np.ndarray,
    rebalance_mask: np.ndarray,
    turnovers: np.ndarray,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Numba JIT 版 NAV 累乘核（内部实现，请通过 ``compute_nav_loop`` 调用）。"""
    n_days, n_stocks = daily_returns_matrix.shape
    portfolio_nav_arr = np.empty(n_days, dtype=np.float64)
    benchmark_nav_arr = np.empty(n_days, dtype=np.float64)
    portfolio_return_arr = np.empty(n_days, dtype=np.float64)
    benchmark_return_arr = np.empty(n_days, dtype=np.float64)
    excess_return_arr = np.empty(n_days, dtype=np.float64)
    transaction_cost_arr = np.zeros(n_days, dtype=np.float64)

    portfolio_nav = 1.0
    benchmark_nav = 1.0
    cost_factor = fee_rate + slippage_rate

    for i in range(n_days):
        gross_portfolio_return = 0.0
        for j in range(n_stocks):
            gross_portfolio_return += weights_matrix[i, j] * daily_returns_matrix[i, j]

        benchmark_return = benchmark_returns[i]
        start_nav = portfolio_nav
        portfolio_nav *= 1.0 + gross_portfolio_return
        benchmark_nav *= 1.0 + benchmark_return

        tx_cost = 0.0
        if rebalance_mask[i]:
            tx_cost = 2.0 * turnovers[i] * cost_factor
            adjusted = portfolio_nav * (1.0 - tx_cost)
            if adjusted < 0.0:
                adjusted = 0.0
            portfolio_nav = adjusted

        if start_nav != 0.0:
            port_return = portfolio_nav / start_nav - 1.0
        else:
            port_return = 0.0

        portfolio_nav_arr[i] = portfolio_nav
        benchmark_nav_arr[i] = benchmark_nav
        portfolio_return_arr[i] = port_return
        benchmark_return_arr[i] = benchmark_return
        excess_return_arr[i] = port_return - benchmark_return
        transaction_cost_arr[i] = tx_cost

    return (
        portfolio_nav_arr,
        benchmark_nav_arr,
        portfolio_return_arr,
        benchmark_return_arr,
        excess_return_arr,
        transaction_cost_arr,
    )


def _compute_nav_loop_numpy(
    daily_returns_matrix: np.ndarray,
    weights_matrix: np.ndarray,
    benchmark_returns: np.ndarray,
    rebalance_mask: np.ndarray,
    turnovers: np.ndarray,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """纯 NumPy 向量化版 NAV 累乘核（与 numba 版数值等价）。

    Args:
        daily_returns_matrix: 形状 ``(n_days, n_stocks)`` 的日收益矩阵。
        weights_matrix: 形状 ``(n_days, n_stocks)`` 的权重矩阵，第 ``i`` 行
            为计算第 ``i`` 日组合收益时使用的开盘前持仓权重。
        benchmark_returns: 形状 ``(n_days,)`` 的基准日收益。
        rebalance_mask: 形状 ``(n_days,)`` 的布尔数组，标记调仓日。
        turnovers: 形状 ``(n_days,)`` 的单边换手率，非调仓日填 0。
        fee_rate: 单边手续费率。
        slippage_rate: 单边滑点率。

    Returns:
        tuple: 依次为 ``portfolio_nav_arr``、``benchmark_nav_arr``、
        ``portfolio_return_arr``、``benchmark_return_arr``、
        ``excess_return_arr``、``transaction_cost_arr``。
    """
    n_days = daily_returns_matrix.shape[0]
    gross_returns = np.sum(weights_matrix * daily_returns_matrix, axis=1)
    cost_factor = fee_rate + slippage_rate
    transaction_costs = np.where(
        rebalance_mask, 2.0 * turnovers * cost_factor, 0.0
    )
    cost_multiplier = np.where(
        rebalance_mask, np.maximum(1.0 - transaction_costs, 0.0), 1.0
    )
    growth_factor = (1.0 + gross_returns) * cost_multiplier
    portfolio_nav = np.cumprod(growth_factor)
    benchmark_nav = np.cumprod(1.0 + benchmark_returns)

    prev_nav = np.empty(n_days, dtype=np.float64)
    prev_nav[0] = 1.0
    if n_days > 1:
        prev_nav[1:] = portfolio_nav[:-1]
    portfolio_returns = np.where(
        prev_nav != 0.0, portfolio_nav / prev_nav - 1.0, 0.0
    )
    excess_returns = portfolio_returns - benchmark_returns

    return (
        portfolio_nav,
        benchmark_nav,
        portfolio_returns,
        benchmark_returns.astype(np.float64, copy=True),
        excess_returns,
        transaction_costs,
    )


def compute_nav_loop(
    daily_returns_matrix: np.ndarray,
    weights_matrix: np.ndarray,
    benchmark_returns: np.ndarray,
    rebalance_mask: np.ndarray,
    turnovers: np.ndarray,
    fee_rate: float,
    slippage_rate: float,
    use_numba: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按日累乘组合净值与基准净值。

    日内执行顺序与 ``BaselineBacktestEngine._build_nav`` 的原始 Python 循环完全
    一致：

    1. 用 ``weights_matrix[i]`` 与 ``daily_returns_matrix[i]`` 的内积得到
       当日毛收益；
    2. 用毛收益更新组合 NAV、用基准日收益更新基准 NAV；
    3. 若 ``rebalance_mask[i]`` 为真，按 ``2 * turnovers[i] * (fee_rate +
       slippage_rate)`` 扣除调仓交易成本，并把 NAV 在 0 处截断；
    4. 净收益 = 当日 NAV / 上日 NAV − 1，超额收益 = 净收益 − 基准收益。

    Args:
        daily_returns_matrix: 形状 ``(n_days, n_stocks)`` 的日收益矩阵，缺失
            值需提前填 0。
        weights_matrix: 形状 ``(n_days, n_stocks)`` 的权重矩阵。
        benchmark_returns: 形状 ``(n_days,)`` 的基准日收益。
        rebalance_mask: 形状 ``(n_days,)`` 的布尔数组，标记调仓日。
        turnovers: 形状 ``(n_days,)`` 的单边换手率。
        fee_rate: 单边手续费率。
        slippage_rate: 单边滑点率。
        use_numba: 是否启用 Numba JIT 路径。当模块加载阶段未检测到 numba
            或调用方显式置为 False 时回退到纯 numpy 实现。

    Returns:
        tuple: 依次为 ``portfolio_nav_arr``、``benchmark_nav_arr``、
        ``portfolio_return_arr``、``benchmark_return_arr``、
        ``excess_return_arr``、``transaction_cost_arr``，均为
        ``float64`` 一维数组。

    Raises:
        ValueError: 当输入数组维度不一致时抛出。
    """
    n_days, n_stocks = daily_returns_matrix.shape
    if weights_matrix.shape != (n_days, n_stocks):
        raise ValueError(
            f"weights_matrix shape {weights_matrix.shape} 与 daily_returns_matrix "
            f"shape {daily_returns_matrix.shape} 不匹配。"
        )
    if benchmark_returns.shape != (n_days,):
        raise ValueError(
            f"benchmark_returns shape {benchmark_returns.shape} 与 n_days={n_days} 不匹配。"
        )
    if rebalance_mask.shape != (n_days,):
        raise ValueError(
            f"rebalance_mask shape {rebalance_mask.shape} 与 n_days={n_days} 不匹配。"
        )
    if turnovers.shape != (n_days,):
        raise ValueError(
            f"turnovers shape {turnovers.shape} 与 n_days={n_days} 不匹配。"
        )

    daily_returns_matrix = np.ascontiguousarray(daily_returns_matrix, dtype=np.float64)
    weights_matrix = np.ascontiguousarray(weights_matrix, dtype=np.float64)
    benchmark_returns = np.ascontiguousarray(benchmark_returns, dtype=np.float64)
    rebalance_mask = np.ascontiguousarray(rebalance_mask, dtype=np.bool_)
    turnovers = np.ascontiguousarray(turnovers, dtype=np.float64)

    if use_numba and HAS_NUMBA:
        return _compute_nav_loop_numba(
            daily_returns_matrix,
            weights_matrix,
            benchmark_returns,
            rebalance_mask,
            turnovers,
            float(fee_rate),
            float(slippage_rate),
        )
    return _compute_nav_loop_numpy(
        daily_returns_matrix,
        weights_matrix,
        benchmark_returns,
        rebalance_mask,
        turnovers,
        float(fee_rate),
        float(slippage_rate),
    )


@_njit(cache=True)
def _compute_turnovers_numba(
    weights_t: np.ndarray,
    weights_prev: np.ndarray,
) -> float:
    """Numba JIT 版单边换手率（内部实现）。"""
    n = weights_t.shape[0]
    total = 0.0
    for j in range(n):
        diff = weights_t[j] - weights_prev[j]
        if diff < 0.0:
            diff = -diff
        total += diff
    return 0.5 * total


def _compute_turnovers_numpy(
    weights_t: np.ndarray,
    weights_prev: np.ndarray,
) -> float:
    """纯 NumPy 版单边换手率。"""
    return float(0.5 * np.sum(np.abs(weights_t - weights_prev)))


def compute_turnovers(
    weights_t: np.ndarray,
    weights_prev: np.ndarray,
    use_numba: bool = True,
) -> float:
    """按权重向量计算单边换手率。

    实现公式为 ``0.5 * Σ|w_t − w_prev|``，两个向量需提前对齐同一只股票
    的下标位置。缺失股票位置应填 0。

    Args:
        weights_t: 当期权重向量，形状 ``(n_stocks,)``。
        weights_prev: 上期权重向量，形状 ``(n_stocks,)``。
        use_numba: 是否启用 Numba JIT 路径。

    Returns:
        float: 单边换手率。

    Raises:
        ValueError: 当两个向量形状不一致时抛出。
    """
    if weights_t.shape != weights_prev.shape:
        raise ValueError(
            f"weights_t shape {weights_t.shape} 与 weights_prev shape "
            f"{weights_prev.shape} 不一致。"
        )

    weights_t = np.ascontiguousarray(weights_t, dtype=np.float64)
    weights_prev = np.ascontiguousarray(weights_prev, dtype=np.float64)

    if use_numba and HAS_NUMBA:
        return float(_compute_turnovers_numba(weights_t, weights_prev))
    return _compute_turnovers_numpy(weights_t, weights_prev)


__all__ = [
    "HAS_NUMBA",
    "compute_nav_loop",
    "compute_turnovers",
]
