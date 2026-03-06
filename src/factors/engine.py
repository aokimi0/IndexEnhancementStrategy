"""因子计算引擎。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.preprocess import winsorize_by_mad, zscore_by_group


class FactorEngine:
    """计算研究阶段使用的基础因子。"""

    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 60,
        volatility_window: int = 20,
        turnover_window: int = 20,
    ) -> None:
        """初始化因子窗口参数。

        Args:
            short_window: 短期动量窗口。
            long_window: 长期动量窗口。
            volatility_window: 波动率窗口。
            turnover_window: 换手率窗口。
        """
        self.short_window = short_window
        self.long_window = long_window
        self.volatility_window = volatility_window
        self.turnover_window = turnover_window

    def compute_factors(self, panel: pd.DataFrame) -> pd.DataFrame:
        """计算核心因子。

        Args:
            panel: 研究面板，至少包含行情与部分基本面字段。

        Returns:
            pd.DataFrame: 附带因子列的数据表。
        """
        frame = panel.copy()
        frame = frame.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        grouped = frame.groupby("ts_code", group_keys=False)

        frame["daily_return"] = grouped["close"].pct_change()
        frame["ret_20"] = grouped["close"].pct_change(self.short_window)
        frame["ret_60"] = grouped["close"].pct_change(self.long_window)
        frame["volatility_20"] = (
            grouped["daily_return"]
            .rolling(self.volatility_window)
            .std()
            .reset_index(level=0, drop=True)
            * np.sqrt(self.volatility_window)
        )
        frame["turnover_20"] = (
            grouped["turnover_rate"]
            .rolling(self.turnover_window)
            .mean()
            .reset_index(level=0, drop=True)
        )

        frame["ep_ttm"] = np.where(frame["pe_ttm"] > 0, 1.0 / frame["pe_ttm"], np.nan)
        frame["bp"] = np.where(frame["pb"] > 0, 1.0 / frame["pb"], np.nan)

        if "roe" not in frame.columns:
            frame["roe"] = np.nan
        if "grossprofitmargin" not in frame.columns:
            frame["grossprofitmargin"] = np.nan

        return frame

    def build_excess_return_label(
        self,
        factor_panel: pd.DataFrame,
        benchmark: pd.DataFrame,
        horizon: int = 20,
    ) -> pd.DataFrame:
        """构造未来超额收益标签。

        Args:
            factor_panel: 已计算因子的面板数据。
            benchmark: 基准指数日线数据。
            horizon: 未来收益窗口，默认 20 个交易日。

        Returns:
            pd.DataFrame: 附带超额收益标签的数据表。
        """
        frame = factor_panel.copy()
        frame = frame.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        grouped = frame.groupby("ts_code", group_keys=False)
        frame[f"future_return_{horizon}d"] = (
            grouped["close"].shift(-horizon) / frame["close"] - 1.0
        )

        benchmark_frame = benchmark.copy()
        benchmark_frame = benchmark_frame.sort_values("trade_date").reset_index(drop=True)
        benchmark_frame[f"benchmark_future_return_{horizon}d"] = (
            benchmark_frame["close"].shift(-horizon) / benchmark_frame["close"] - 1.0
        )
        benchmark_frame = benchmark_frame[
            ["trade_date", f"benchmark_future_return_{horizon}d"]
        ]

        frame = frame.merge(benchmark_frame, on="trade_date", how="left")
        frame[f"label_excess_return_{horizon}d"] = (
            frame[f"future_return_{horizon}d"]
            - frame[f"benchmark_future_return_{horizon}d"]
        )
        return frame

    def prepare_model_panel(
        self,
        panel: pd.DataFrame,
        factor_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """生成供模型训练使用的标准化因子面板。

        Args:
            panel: 原始因子面板。
            factor_columns: 因子列列表。为空时使用默认核心因子。

        Returns:
            pd.DataFrame: 预处理后的面板数据。
        """
        selected_columns = factor_columns or self.default_factor_columns()
        frame = winsorize_by_mad(panel, columns=selected_columns)
        frame = zscore_by_group(frame, columns=selected_columns)
        return frame

    @staticmethod
    def default_factor_columns() -> list[str]:
        """返回默认核心因子列。

        Returns:
            list[str]: 默认因子列名列表。
        """
        return [
            "ep_ttm",
            "bp",
            "roe",
            "grossprofitmargin",
            "ret_20",
            "ret_60",
            "volatility_20",
            "turnover_20",
        ]
