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
        for column in (
            "turnover_rate",
            "pe_ttm",
            "pb",
            "roe",
            "grossprofitmargin",
            "netprofitmargin",
            "yoynetprofit",
            "assetturnover",
            "cfotoor",
            "equitymultiplier",
        ):
            if column not in frame.columns:
                frame[column] = np.nan
        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "vol",
            "amount",
            "turnover_rate",
            "pe_ttm",
            "pb",
            "roe",
            "grossprofitmargin",
            "netprofitmargin",
            "yoynetprofit",
            "assetturnover",
            "cfotoor",
            "equitymultiplier",
        ]
        for column in numeric_columns:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
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
        if "netprofitmargin" not in frame.columns:
            frame["netprofitmargin"] = np.nan
        if "yoynetprofit" not in frame.columns:
            frame["yoynetprofit"] = np.nan
        if "assetturnover" not in frame.columns:
            frame["assetturnover"] = np.nan
        if "cfotoor" not in frame.columns:
            frame["cfotoor"] = np.nan
        if "equitymultiplier" not in frame.columns:
            frame["equitymultiplier"] = np.nan

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
        frame["trade_date"] = frame["trade_date"].astype(str)
        grouped = frame.groupby("ts_code", group_keys=False)
        frame[f"future_return_{horizon}d"] = (
            grouped["close"].shift(-horizon) / frame["close"] - 1.0
        )

        benchmark_frame = benchmark.copy()
        benchmark_frame = benchmark_frame.sort_values("trade_date").reset_index(drop=True)
        benchmark_frame["trade_date"] = benchmark_frame["trade_date"].astype(str)
        benchmark_frame["benchmark_daily_return"] = benchmark_frame["close"].pct_change()
        benchmark_frame[f"benchmark_future_return_{horizon}d"] = (
            benchmark_frame["close"].shift(-horizon) / benchmark_frame["close"] - 1.0
        )
        benchmark_frame = benchmark_frame[
            [
                "trade_date",
                "benchmark_daily_return",
                f"benchmark_future_return_{horizon}d",
            ]
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
        neutralize_by_industry: bool = True,
    ) -> pd.DataFrame:
        """生成供模型训练使用的标准化因子面板。

        Args:
            panel: 原始因子面板。
            factor_columns: 因子列列表。为空时使用默认核心因子。
            neutralize_by_industry: 是否进行行业中性化（若面板中存在 industry_name 列）。

        Returns:
            pd.DataFrame: 预处理后的面板数据。
        """
        selected_columns = factor_columns or self.default_factor_columns()
        frame = winsorize_by_mad(panel, columns=selected_columns)
        
        group_cols = ["trade_date"]
        if neutralize_by_industry and "industry_name" in frame.columns:
            group_cols.append("industry_name")
            
        frame = zscore_by_group(frame, columns=selected_columns, group_col=group_cols)
        return frame

    @staticmethod
    def feature_groups() -> dict[str, list[str]]:
        """返回按论文语义划分的因子分组。

        Returns:
            dict[str, list[str]]: 分组名到因子列列表的映射。
        """
        return {
            "value": ["ep_ttm", "bp"],
            "quality": [
                "roe",
                "grossprofitmargin",
                "netprofitmargin",
                "yoynetprofit",
                "assetturnover",
                "cfotoor",
            ],
            "technical": ["ret_20", "ret_60", "volatility_20"],
            "liquidity": ["turnover_20"],
            "leverage": ["equitymultiplier"],
            "external": ["northbound_net_inflow", "m2_yoy", "cn_spread_10y_2y"],
        }

    @classmethod
    def resolve_feature_columns(
        cls,
        feature_groups: list[str] | None = None,
        extra_columns: list[str] | None = None,
        available_columns: list[str] | None = None,
    ) -> list[str]:
        """根据分组解析并过滤最终特征列。

        Args:
            feature_groups: 需要启用的因子分组。为空时使用默认核心分组。
            extra_columns: 额外追加的特征列。
            available_columns: 当前面板中实际存在的列，用于过滤不可用特征。

        Returns:
            list[str]: 去重后且按顺序排列的特征列列表。

        Raises:
            ValueError: 当传入未知分组时抛出异常。
        """
        selected_groups = feature_groups or ["value", "quality", "technical", "liquidity"]
        group_mapping = cls.feature_groups()
        resolved_columns: list[str] = []
        seen_columns: set[str] = set()

        for group_name in selected_groups:
            if group_name not in group_mapping:
                raise ValueError(f"未知因子分组: {group_name}")
            for column in group_mapping[group_name]:
                if column not in seen_columns:
                    resolved_columns.append(column)
                    seen_columns.add(column)

        for column in extra_columns or []:
            if column not in seen_columns:
                resolved_columns.append(column)
                seen_columns.add(column)

        if available_columns is None:
            return resolved_columns
        available_set = set(available_columns)
        return [column for column in resolved_columns if column in available_set]

    @staticmethod
    def default_factor_columns() -> list[str]:
        """返回默认核心因子列。

        Returns:
            list[str]: 默认因子列名列表。
        """
        return FactorEngine.resolve_feature_columns()
