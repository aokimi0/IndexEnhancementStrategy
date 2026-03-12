"""多因子基线回测引擎。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.metrics import compute_performance_metrics
from src.factors.engine import FactorEngine
from src.portfolio import ConstrainedPortfolioOptimizer, OptimizationConfig


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
        use_optimizer: bool = False,
        optimization_config: OptimizationConfig | None = None,
        score_column: str = "score",
        fee_rate: float = 0.001,
        slippage_rate: float = 0.001,
    ) -> None:
        """初始化回测参数。

        Args:
            top_n: 每次调仓持有股票数。
            rebalance_frequency: 调仓频率，默认月度。
            factor_columns: 使用的因子列列表。
            use_optimizer: 是否启用约束型组合优化。
            optimization_config: 组合优化参数。
            score_column: 用于选股或优化的信号列。
            fee_rate: 单边手续费率。
            slippage_rate: 单边滑点率。
        """
        self.top_n = top_n
        self.rebalance_frequency = rebalance_frequency
        self.factor_columns = factor_columns or FactorEngine.default_factor_columns()
        self.use_optimizer = use_optimizer
        self.score_column = score_column
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.optimizer = (
            ConstrainedPortfolioOptimizer(config=optimization_config)
            if use_optimizer
            else None
        )

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
        if self.score_column not in frame.columns:
            frame[self.score_column] = self._build_score(frame)

        rebalance_dates = self._get_rebalance_dates(frame)
        positions = self._build_positions(frame, rebalance_dates)
        nav_frame = self._build_nav(frame, positions)
        metrics = compute_performance_metrics(nav_frame, positions=positions)
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
        previous_weights: dict[str, float] = {}
        for rebalance_date in rebalance_dates:
            snapshot = frame[frame["trade_date"] == rebalance_date].copy()
            snapshot = snapshot.dropna(subset=[self.score_column])
            if snapshot.empty:
                continue

            if self.optimizer is None:
                selected = self._build_equal_weight_positions(
                    snapshot=snapshot,
                    rebalance_date=rebalance_date,
                    previous_weights=previous_weights,
                )
            else:
                selected = self._build_optimized_positions(
                    frame=frame,
                    snapshot=snapshot,
                    rebalance_date=rebalance_date,
                    previous_weights=previous_weights,
                )
            if selected.empty:
                continue
            previous_weights = dict(zip(selected["ts_code"], selected["weight"], strict=False))
            positions.append(selected)
        if not positions:
            return pd.DataFrame(
                columns=[
                    "rebalance_date",
                    "ts_code",
                    "weight",
                    self.score_column,
                    "benchmark_weight",
                    "industry_name",
                    "turnover",
                    "ex_ante_tracking_error",
                    "max_industry_deviation",
                    "optimization_status",
                ]
            )
        return pd.concat(positions, ignore_index=True)

    def _build_equal_weight_positions(
        self,
        snapshot: pd.DataFrame,
        rebalance_date: pd.Timestamp,
        previous_weights: dict[str, float],
    ) -> pd.DataFrame:
        """构建前 N 等权持仓。"""
        selected = snapshot.nlargest(self.top_n, self.score_column).copy()
        selected["weight"] = 1.0 / len(selected)
        current_weights = dict(zip(selected["ts_code"], selected["weight"], strict=False))
        selected["rebalance_date"] = rebalance_date
        selected["turnover"] = self._compute_turnover(previous_weights, current_weights)
        selected["ex_ante_tracking_error"] = np.nan
        selected["max_industry_deviation"] = np.nan
        selected["optimization_status"] = "equal_weight"
        columns = [
            "rebalance_date",
            "ts_code",
            "weight",
            self.score_column,
            "benchmark_weight",
            "industry_name",
            "turnover",
            "ex_ante_tracking_error",
            "max_industry_deviation",
            "optimization_status",
        ]
        for column in ("benchmark_weight", "industry_name"):
            if column not in selected.columns:
                selected[column] = np.nan
        return selected[columns]

    def _build_optimized_positions(
        self,
        frame: pd.DataFrame,
        snapshot: pd.DataFrame,
        rebalance_date: pd.Timestamp,
        previous_weights: dict[str, float],
    ) -> pd.DataFrame:
        """构建带约束的优化持仓。"""
        if self.optimizer is None:
            return pd.DataFrame()
        history = frame[frame["trade_date"] < rebalance_date][
            ["trade_date", "ts_code", "daily_return"]
        ].copy()
        optimization_result = self.optimizer.optimize(
            snapshot=snapshot,
            return_history=history,
            previous_weights=previous_weights,
            score_column=self.score_column,
        )
        selected = optimization_result.weights.copy()
        selected = selected[selected["weight"] > 1e-6].copy()
        selected["rebalance_date"] = rebalance_date
        selected = selected.merge(
            snapshot[["ts_code", self.score_column]].drop_duplicates(subset=["ts_code"]),
            on="ts_code",
            how="left",
        )
        selected["turnover"] = optimization_result.diagnostics["turnover"]
        selected["ex_ante_tracking_error"] = optimization_result.diagnostics[
            "ex_ante_tracking_error"
        ]
        selected["max_industry_deviation"] = optimization_result.diagnostics[
            "max_industry_deviation"
        ]
        selected["optimization_status"] = optimization_result.status
        return selected[
            [
                "rebalance_date",
                "ts_code",
                "weight",
                self.score_column,
                "benchmark_weight",
                "industry_name",
                "turnover",
                "ex_ante_tracking_error",
                "max_industry_deviation",
                "optimization_status",
            ]
        ]

    def _build_nav(
        self,
        frame: pd.DataFrame,
        positions: pd.DataFrame,
    ) -> pd.DataFrame:
        """根据持仓表生成组合净值。"""
        trade_dates = sorted(frame["trade_date"].drop_duplicates().tolist())
        benchmark = (
            frame[["trade_date", "benchmark_daily_return"]]
            .drop_duplicates(subset=["trade_date"])
            .sort_values("trade_date")
        )
        if "benchmark_daily_return" in benchmark.columns:
            benchmark["benchmark_return"] = pd.to_numeric(
                benchmark["benchmark_daily_return"],
                errors="coerce",
            ).fillna(0.0)
        else:
            benchmark["benchmark_return"] = (
                benchmark["trade_date"].map(
                    frame.groupby("trade_date")["daily_return"].mean()
                ).fillna(0.0)
            )

        nav_rows: list[dict[str, float | pd.Timestamp]] = []
        current_weights: dict[str, float] = {}
        rebalance_map = {
            rebalance_date: group
            for rebalance_date, group in positions.groupby("rebalance_date")
        }

        portfolio_nav = 1.0
        benchmark_nav = 1.0

        for trade_date in trade_dates:
            day_frame = frame[frame["trade_date"] == trade_date]
            daily_return_map = (
                day_frame.set_index("ts_code")["daily_return"].fillna(0.0).to_dict()
            )
            gross_portfolio_return = sum(
                weight * daily_return_map.get(ts_code, 0.0)
                for ts_code, weight in current_weights.items()
            )
            benchmark_return = (
                benchmark.loc[benchmark["trade_date"] == trade_date, "benchmark_return"].iloc[0]
                if trade_date in set(benchmark["trade_date"])
                else 0.0
            )
            benchmark_return = 0.0 if pd.isna(benchmark_return) else benchmark_return

            start_nav = portfolio_nav
            portfolio_nav *= 1.0 + gross_portfolio_return
            benchmark_nav *= 1.0 + benchmark_return
            transaction_cost = 0.0
            if trade_date in rebalance_map:
                rebalance_group = rebalance_map[trade_date]
                turnover = float(rebalance_group["turnover"].iloc[0])
                transaction_cost = 2.0 * turnover * (self.fee_rate + self.slippage_rate)
                portfolio_nav *= max(1.0 - transaction_cost, 0.0)
                current_weights = dict(
                    zip(
                        rebalance_group["ts_code"],
                        rebalance_group["weight"],
                        strict=False,
                    )
                )
            portfolio_return = portfolio_nav / start_nav - 1.0 if start_nav != 0 else 0.0
            excess_return = portfolio_return - benchmark_return
            excess_nav = portfolio_nav / benchmark_nav if benchmark_nav != 0 else np.nan
            nav_rows.append(
                {
                    "trade_date": trade_date,
                    "portfolio_return": portfolio_return,
                    "gross_portfolio_return": gross_portfolio_return,
                    "benchmark_return": benchmark_return,
                    "excess_return": excess_return,
                    "transaction_cost": transaction_cost,
                    "portfolio_nav": portfolio_nav,
                    "benchmark_nav": benchmark_nav,
                    "excess_nav": excess_nav,
                }
            )

        return pd.DataFrame(nav_rows)

    @staticmethod
    def _compute_turnover(
        previous_weights: dict[str, float],
        current_weights: dict[str, float],
    ) -> float:
        """计算单边换手率。"""
        all_codes = set(previous_weights).union(current_weights)
        return float(
            0.5
            * sum(
                abs(current_weights.get(code, 0.0) - previous_weights.get(code, 0.0))
                for code in all_codes
            )
        )
