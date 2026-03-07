"""回测指标计算。"""

from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def compute_performance_metrics(nav_frame: pd.DataFrame) -> pd.DataFrame:
    """计算基线回测指标。

    Args:
        nav_frame: 至少包含 `portfolio_nav`、`benchmark_nav` 和对应日收益列的数据表。

    Returns:
        pd.DataFrame: 单行指标表。
    """
    frame = nav_frame.copy()
    frame = frame.sort_values("trade_date").reset_index(drop=True)

    portfolio_returns = frame["portfolio_return"].fillna(0.0)
    benchmark_returns = frame["benchmark_return"].fillna(0.0)
    excess_returns = portfolio_returns - benchmark_returns

    annual_return = _annualized_return(frame["portfolio_nav"])
    benchmark_annual_return = _annualized_return(frame["benchmark_nav"])
    annual_excess_return = annual_return - benchmark_annual_return
    annual_volatility = portfolio_returns.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe_ratio = (
        annual_return / annual_volatility if annual_volatility and not np.isnan(annual_volatility) else np.nan
    )
    tracking_error = excess_returns.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)
    information_ratio = (
        annual_excess_return / tracking_error
        if tracking_error and not np.isnan(tracking_error)
        else np.nan
    )
    max_drawdown = _max_drawdown(frame["portfolio_nav"])

    return pd.DataFrame(
        [
            {
                "annual_return": annual_return,
                "benchmark_annual_return": benchmark_annual_return,
                "annual_excess_return": annual_excess_return,
                "annual_volatility": annual_volatility,
                "sharpe_ratio": sharpe_ratio,
                "tracking_error": tracking_error,
                "information_ratio": information_ratio,
                "max_drawdown": max_drawdown,
            }
        ]
    )


def _annualized_return(nav_series: pd.Series) -> float:
    """计算年化收益率。"""
    clean_nav = nav_series.dropna()
    if clean_nav.empty or len(clean_nav) < 2:
        return float("nan")
    total_return = clean_nav.iloc[-1] / clean_nav.iloc[0] - 1.0
    periods = len(clean_nav)
    return (1.0 + total_return) ** (TRADING_DAYS_PER_YEAR / periods) - 1.0


def _max_drawdown(nav_series: pd.Series) -> float:
    """计算最大回撤。"""
    clean_nav = nav_series.dropna()
    if clean_nav.empty:
        return float("nan")
    rolling_max = clean_nav.cummax()
    drawdown = clean_nav / rolling_max - 1.0
    return float(drawdown.min())
