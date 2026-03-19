"""截面预处理函数。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_by_mad(
    frame: pd.DataFrame,
    columns: list[str],
    group_col: str = "trade_date",
    n_mad: float = 3.0,
) -> pd.DataFrame:
    """按截面使用 MAD 去极值。

    Args:
        frame: 输入数据表。
        columns: 需要处理的列名。
        group_col: 分组列，默认按交易日截面处理。
        n_mad: 中位数绝对偏差倍数。

    Returns:
        pd.DataFrame: 处理后的数据表副本。
    """
    result = frame.copy()
    for column in columns:
        grouped = result.groupby(group_col)[column]
        median = grouped.transform("median")
        mad = grouped.transform(lambda x: np.median(np.abs(x - np.median(x))))
        lower = median - n_mad * mad
        upper = median + n_mad * mad
        result[column] = result[column].clip(lower=lower, upper=upper)
    return result


def zscore_by_group(
    frame: pd.DataFrame,
    columns: list[str],
    group_col: str | list[str] = "trade_date",
) -> pd.DataFrame:
    """按截面做 Z-Score 标准化。

    Args:
        frame: 输入数据表。
        columns: 需要处理的列名。
        group_col: 分组列，默认按交易日处理。支持传入列表实现多级分组（如交易日+行业）。

    Returns:
        pd.DataFrame: 标准化后的数据表副本。
    """
    result = frame.copy()
    for column in columns:
        grouped = result.groupby(group_col)[column]
        mean = grouped.transform("mean")
        std = grouped.transform("std")
        std = std.replace(0, np.nan)
        result[column] = (result[column] - mean) / std
    return result
