"""多因子基线回测引擎。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.metrics import compute_performance_metrics


@dataclass
class BaselineBacktestResult:
    """基线回测结果。"""

    nav_frame: pd.DataFrame
    positions: pd.DataFrame
    metrics: pd.DataFrame


class BaselineBacktestEngine:
    """最小多因子基线回测引擎。

    当前版本采用：

    - 月度调仓
    - 截面因子等权打分
    - 选前 N 只股票
    - 等权持有
    """

    def __init__(
        self,
        top_n: int = 5,
        rebalance_frequency: str = "M",
        factor_columns: list[str] | None = None,
    ) -> None:
        """初始化回测参数。

        Args:
            top_n: 每次调仓持有股票数。
            rebalance_frequency: 调仓频率，默认月度。
            factor_columns: 使用的因子列列表。
        """
        self.top_n = top_n
        self.rebalance_frequency = rebalance_frequency
        self.factor_columns = factor_columns or ["ret_20", "ret_60", "volatility_20"]

    def run(self, factor_panel: pd.DataFrame) -> BaselineBacktestResult:
        """执行基线回测。

        Args:
            factor_panel: 因子面板数据。

        Returns:
            BaselineBacktestResult: 回测结果。
        """
        frame = factor_panel.copy()
        frame["trade_date"] = pd.to_datetime(
            frame["trade_date"].astype(str),
            format="%Y%m%d",
            errors="coerce",
        )
        frame = frame.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        frame["daily_return"] = pd.to_numeric(frame["daily_return"], errors="coerce")
        if "score" not in frame.columns:
            frame["score"] = self._build_score(frame)

        rebalance_dates = self._get_rebalance_dates(frame)
        positions = self._build_positions(frame, rebalance_dates)
        nav_frame = self._build_nav(frame, positions)
        metrics = compute_performance_metrics(nav_frame)
        return BaselineBacktestResult(
            nav_frame=nav_frame,
            positions=positions,
            metrics=metrics,
        )

    def _build_score(self, frame: pd.DataFrame) -> pd.Series:
        """构建截面综合得分。"""
        score_columns: list[pd.Series] = []
        for column in self.factor_columns:
            if column not in frame.columns:
                continue
            series = pd.to_numeric(frame[column], errors="coerce")
            if column == "volatility_20":
                score_columns.append(-series)
            else:
                score_columns.append(series)
        if not score_columns:
            return pd.Series(np.nan, index=frame.index)
        return pd.concat(score_columns, axis=1).mean(axis=1, skipna=True)

    def _get_rebalance_dates(self, frame: pd.DataFrame) -> list[pd.Timestamp]:
        """获取调仓日列表。"""
        trade_dates = (
            frame[["trade_date"]]
            .drop_duplicates()
            .sort_values("trade_date")
            .assign(rebalance_month=lambda df: df["trade_date"].dt.to_period(self.rebalance_frequency))
        )
        rebalance_dates = (
            trade_dates.groupby("rebalance_month")["trade_date"].max().tolist()
        )
        return rebalance_dates

    def _build_positions(
        self,
        frame: pd.DataFrame,
        rebalance_dates: list[pd.Timestamp],
    ) -> pd.DataFrame:
        """按调仓日生成持仓表。"""
        positions: list[pd.DataFrame] = []
        for rebalance_date in rebalance_dates:
            snapshot = frame[frame["trade_date"] == rebalance_date].copy()
            snapshot = snapshot.dropna(subset=["score"])
            if snapshot.empty:
                continue
            selected = snapshot.nlargest(self.top_n, "score").copy()
            selected["weight"] = 1.0 / len(selected)
            selected["rebalance_date"] = rebalance_date
            positions.append(selected[["rebalance_date", "ts_code", "weight", "score"]])
        if not positions:
            return pd.DataFrame(columns=["rebalance_date", "ts_code", "weight", "score"])
        return pd.concat(positions, ignore_index=True)

    def _build_nav(
        self,
        frame: pd.DataFrame,
        positions: pd.DataFrame,
    ) -> pd.DataFrame:
        """根据持仓表生成组合净值。"""
        trade_dates = sorted(frame["trade_date"].drop_duplicates().tolist())
        benchmark = (
            frame[["trade_date", "benchmark_future_return_20d"]]
            .drop_duplicates(subset=["trade_date"])
            .sort_values("trade_date")
        )
        benchmark["benchmark_return"] = (
            benchmark["trade_date"].map(
                frame.groupby("trade_date")["daily_return"].mean()
            ).fillna(0.0)
        )

        nav_rows: list[dict[str, float | pd.Timestamp]] = []
        current_weights: dict[str, float] = {}
        rebalance_map = {
            rebalance_date: group[["ts_code", "weight"]]
            for rebalance_date, group in positions.groupby("rebalance_date")
        }

        portfolio_nav = 1.0
        benchmark_nav = 1.0

        for trade_date in trade_dates:
            if trade_date in rebalance_map:
                current_weights = dict(
                    zip(
                        rebalance_map[trade_date]["ts_code"],
                        rebalance_map[trade_date]["weight"],
                        strict=False,
                    )
                )

            day_frame = frame[frame["trade_date"] == trade_date]
            daily_return_map = (
                day_frame.set_index("ts_code")["daily_return"].fillna(0.0).to_dict()
            )
            portfolio_return = sum(
                weight * daily_return_map.get(ts_code, 0.0)
                for ts_code, weight in current_weights.items()
            )
            benchmark_return = (
                day_frame["daily_return"].mean() if not day_frame.empty else 0.0
            )
            benchmark_return = 0.0 if pd.isna(benchmark_return) else benchmark_return

            portfolio_nav *= 1.0 + portfolio_return
            benchmark_nav *= 1.0 + benchmark_return
            nav_rows.append(
                {
                    "trade_date": trade_date,
                    "portfolio_return": portfolio_return,
                    "benchmark_return": benchmark_return,
                    "portfolio_nav": portfolio_nav,
                    "benchmark_nav": benchmark_nav,
                }
            )

        return pd.DataFrame(nav_rows)
