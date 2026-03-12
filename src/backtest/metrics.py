"""回测指标计算。"""

from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def compute_performance_metrics(
    nav_frame: pd.DataFrame,
    positions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """计算基线回测指标。

    Args:
        nav_frame: 至少包含 `portfolio_nav`、`benchmark_nav` 和对应日收益列的数据表。
        positions: 调仓持仓表，用于汇总换手与约束诊断。

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
    benchmark_max_drawdown = _max_drawdown(frame["benchmark_nav"])
    excess_nav = (
        frame["excess_nav"]
        if "excess_nav" in frame.columns
        else frame["portfolio_nav"] / frame["benchmark_nav"].replace(0, np.nan)
    )
    excess_max_drawdown = _max_drawdown(excess_nav)
    monthly_win_rate = _monthly_win_rate(frame)
    upside_capture = _capture_ratio(
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        direction="up",
    )
    downside_capture = _capture_ratio(
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        direction="down",
    )

    annual_turnover = np.nan
    avg_ex_ante_tracking_error = np.nan
    max_ex_ante_tracking_error = np.nan
    avg_max_industry_deviation = np.nan
    if positions is not None and not positions.empty:
        turnover_frame = positions.groupby("rebalance_date", as_index=False).agg(
            turnover=("turnover", "first"),
            ex_ante_tracking_error=("ex_ante_tracking_error", "first"),
            max_industry_deviation=("max_industry_deviation", "first"),
        )
        annual_turnover = float(turnover_frame["turnover"].fillna(0.0).mean() * 12.0)
        avg_ex_ante_tracking_error = float(
            turnover_frame["ex_ante_tracking_error"].dropna().mean()
        )
        max_ex_ante_tracking_error = float(
            turnover_frame["ex_ante_tracking_error"].dropna().max()
        )
        avg_max_industry_deviation = float(
            turnover_frame["max_industry_deviation"].dropna().mean()
        )

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
                "benchmark_max_drawdown": benchmark_max_drawdown,
                "excess_max_drawdown": excess_max_drawdown,
                "monthly_win_rate": monthly_win_rate,
                "upside_capture": upside_capture,
                "downside_capture": downside_capture,
                "annual_turnover": annual_turnover,
                "avg_ex_ante_tracking_error": avg_ex_ante_tracking_error,
                "max_ex_ante_tracking_error": max_ex_ante_tracking_error,
                "avg_max_industry_deviation": avg_max_industry_deviation,
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


def _monthly_win_rate(nav_frame: pd.DataFrame) -> float:
    """计算月度跑赢基准的胜率。"""
    frame = nav_frame.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    monthly = (
        frame.assign(month=frame["trade_date"].dt.to_period("M"))
        .groupby("month", as_index=False)
        .agg(
            portfolio_return=("portfolio_return", lambda x: (1.0 + x).prod() - 1.0),
            benchmark_return=("benchmark_return", lambda x: (1.0 + x).prod() - 1.0),
        )
    )
    if monthly.empty:
        return float("nan")
    return float((monthly["portfolio_return"] > monthly["benchmark_return"]).mean())


def _capture_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    direction: str,
) -> float:
    """计算上涨或下跌市场捕获比率。"""
    if direction == "up":
        mask = benchmark_returns > 0
    else:
        mask = benchmark_returns < 0
    if not mask.any():
        return float("nan")

    portfolio_subset = portfolio_returns[mask]
    benchmark_subset = benchmark_returns[mask]
    benchmark_total_return = (1.0 + benchmark_subset).prod() - 1.0
    if np.isclose(benchmark_total_return, 0.0):
        return float("nan")
    portfolio_total_return = (1.0 + portfolio_subset).prod() - 1.0
    return float(portfolio_total_return / benchmark_total_return)
