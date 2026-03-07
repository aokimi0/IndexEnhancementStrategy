"""研究数据读取与拼接。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from src.config import ProjectConfig
from src.data.akshare_client import AkshareClient
from src.data.tushare_client import TushareClient


@dataclass
class DataBundle:
    """研究阶段使用的数据集合。

    Attributes:
        universe: 股票池成分。
        daily: 个股日线行情。
        daily_basic: 个股日频基本面指标。
        financial_indicators: 财务指标数据。
        benchmark: 基准指数日线行情。
        northbound: 北向资金流数据。
        research_panel: 研究面板数据。
    """

    universe: pd.DataFrame
    daily: pd.DataFrame
    daily_basic: pd.DataFrame
    financial_indicators: pd.DataFrame
    benchmark: pd.DataFrame
    northbound: pd.DataFrame
    research_panel: pd.DataFrame


class DataService:
    """数据读取服务。

    当前版本优先解决“能稳定拉取和拼接数据”的问题，不在这里引入复杂缓存和调度逻辑。
    """

    def __init__(
        self,
        client: TushareClient | AkshareClient,
        config: ProjectConfig,
    ) -> None:
        """初始化数据服务。

        Args:
            client: 数据客户端。
            config: 项目配置。
        """
        self.client = client
        self.config = config
        self.failure_log_path = self.config.logs_dir / "akshare_failures.log"

    def get_index_components(
        self,
        index_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """获取指定指数在区间内的成分权重。

        Args:
            index_code: 指数代码。
            start_date: 开始日期，格式为 `YYYYMMDD`。
            end_date: 结束日期，格式为 `YYYYMMDD`。

        Returns:
            pd.DataFrame: 指数成分权重表。
        """
        cache_path = self._build_cache_path(
            dataset_name="index_components",
            object_name=index_code,
            start_date=start_date,
            end_date=end_date,
        )
        frame = self._load_or_cache(
            cache_path=cache_path,
            loader=lambda: self.client.index_weight(
                index_code=index_code,
                start_date=start_date,
                end_date=end_date,
            ),
        )
        if frame.empty:
            return frame
        # Akshare 当前返回的是“最新成分快照”，这里统一映射到研究区间末日。
        if frame["trade_date"].nunique() == 1:
            frame = frame.copy()
            frame["trade_date"] = end_date
        return frame.sort_values(["trade_date", "con_code"]).reset_index(drop=True)

    def get_research_universe(
        self,
        index_code: str,
        start_date: str,
        end_date: str,
        universe_limit: int | None = None,
    ) -> list[str]:
        """获取研究股票池。

        当前版本取区间内曾经进入指数的全部成分股并去重，适合作为第一版稳定研究池。

        Args:
            index_code: 指数代码。
            start_date: 开始日期。
            end_date: 结束日期。

        Returns:
            list[str]: 股票代码列表。
        """
        components = self.get_index_components(
            index_code=index_code,
            start_date=start_date,
            end_date=end_date,
        )
        if components.empty:
            return []
        codes = sorted(components["con_code"].dropna().unique().tolist())
        if universe_limit and universe_limit > 0:
            return codes[:universe_limit]
        return codes

    def load_stock_daily(
        self,
        ts_codes: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """批量加载个股日线数据。"""
        fields = "ts_code,trade_date,open,high,low,close,pre_close,vol,amount"
        frame = self._concat_by_code(
            dataset_name="daily",
            ts_codes=ts_codes,
            start_date=start_date,
            end_date=end_date,
            loader=lambda code: self.client.daily(
                ts_code=code,
                start_date=start_date,
                end_date=end_date,
                fields=fields,
            ),
        )
        if not frame.empty:
            return frame
        return pd.DataFrame(
            columns=[
                "ts_code",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "pre_close",
                "vol",
                "amount",
                "turnover_rate",
            ]
        )

    def load_stock_daily_basic(
        self,
        ts_codes: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """批量加载个股日频基本面指标。"""
        fields = (
            "ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,"
            "pe_ttm,pb,total_mv,float_mv,ps_ttm,dv_ttm"
        )
        frame = self._concat_by_code(
            dataset_name="daily_basic",
            ts_codes=ts_codes,
            start_date=start_date,
            end_date=end_date,
            loader=lambda code: self.client.daily_basic(
                ts_code=code,
                start_date=start_date,
                end_date=end_date,
                fields=fields,
            ),
        )
        if not frame.empty:
            return frame
        return pd.DataFrame(
            columns=[
                "ts_code",
                "trade_date",
                "turnover_rate",
                "turnover_rate_f",
                "volume_ratio",
                "pe_ttm",
                "pb",
                "total_mv",
                "float_mv",
                "ps_ttm",
                "dv_ttm",
            ]
        )

    def load_hs300_benchmark(
        self,
        start_date: str,
        end_date: str,
        index_code: str = "000300.SH",
    ) -> pd.DataFrame:
        """加载沪深300基准日线数据。"""
        fields = "ts_code,trade_date,open,high,low,close,pre_close,vol,amount"
        cache_path = self._build_cache_path(
            dataset_name="benchmark",
            object_name=index_code,
            start_date=start_date,
            end_date=end_date,
        )
        frame = self._load_or_cache(
            cache_path=cache_path,
            loader=lambda: self.client.index_daily(
                ts_code=index_code,
                start_date=start_date,
                end_date=end_date,
                fields=fields,
            ),
        )
        if frame.empty:
            return frame
        return frame.sort_values("trade_date").reset_index(drop=True)

    def load_financial_indicators(
        self,
        ts_codes: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """批量加载财务指标。

        Args:
            ts_codes: 股票代码列表。
            start_date: 开始日期。
            end_date: 结束日期。

        Returns:
            pd.DataFrame: 财务指标表。
        """
        fields = "ts_code,ann_date,end_date,roe,grossprofit_margin"
        frame = self._concat_by_code(
            dataset_name="financial_indicators",
            ts_codes=ts_codes,
            start_date=start_date,
            end_date=end_date,
            loader=lambda code: self.client.fina_indicator(
                ts_code=code,
                start_date=start_date,
                end_date=end_date,
                fields=fields,
            ),
        )
        if frame.empty:
            return pd.DataFrame(
                columns=["ts_code", "ann_date", "end_date", "roe", "grossprofit_margin"]
            )
        return frame.sort_values(["ts_code", "ann_date"]).reset_index(drop=True)

    def load_northbound_flow(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """加载北向资金流数据。"""
        fields = "trade_date,north_money,sh_amount,sz_amount"
        cache_path = self._build_cache_path(
            dataset_name="northbound_flow",
            object_name="northbound",
            start_date=start_date,
            end_date=end_date,
        )
        frame = self._load_or_cache(
            cache_path=cache_path,
            loader=lambda: self.client.moneyflow_hsgt(
                start_date=start_date,
                end_date=end_date,
                fields=fields,
            ),
        )
        if frame.empty:
            return frame
        return frame.sort_values("trade_date").reset_index(drop=True)

    def load_stock_minute(
        self,
        ts_codes: list[str],
        start_datetime: str,
        end_datetime: str,
        period: str = "5",
    ) -> pd.DataFrame:
        """批量加载个股分钟级行情。"""
        frame = self._concat_by_code(
            dataset_name=f"minute_{period}",
            ts_codes=ts_codes,
            start_date=self._compact_datetime(start_datetime),
            end_date=self._compact_datetime(end_datetime),
            loader=lambda code: self.client.minute(
                ts_code=code,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                period=period,
            ),
        )
        if not frame.empty:
            return frame
        return pd.DataFrame(
            columns=[
                "ts_code",
                "trade_datetime",
                "open",
                "high",
                "low",
                "close",
                "vol",
                "amount",
            ]
        )

    def load_benchmark_minute(
        self,
        start_datetime: str,
        end_datetime: str,
        index_code: str = "000300.SH",
        period: str = "5",
    ) -> pd.DataFrame:
        """加载基准指数分钟级行情。"""
        cache_path = self._build_cache_path(
            dataset_name=f"benchmark_minute_{period}",
            object_name=index_code,
            start_date=self._compact_datetime(start_datetime),
            end_date=self._compact_datetime(end_datetime),
        )
        try:
            frame = self._load_or_cache(
                cache_path=cache_path,
                loader=lambda: self.client.index_minute(
                    ts_code=index_code,
                    start_datetime=start_datetime,
                    end_datetime=end_datetime,
                    period=period,
                ),
            )
        except Exception as exc:
            self._log_failure(
                dataset_name=f"benchmark_minute_{period}",
                object_name=index_code,
                error=str(exc),
            )
            frame = pd.DataFrame()
        if frame.empty:
            return pd.DataFrame(
                columns=[
                    "ts_code",
                    "trade_datetime",
                    "open",
                    "high",
                    "low",
                    "close",
                    "vol",
                    "amount",
                ]
            )
        return frame.sort_values("trade_datetime").reset_index(drop=True)

    def build_research_panel(
        self,
        start_date: str,
        end_date: str,
        index_code: str = "000300.SH",
        universe_limit: int | None = None,
    ) -> DataBundle:
        """构建研究面板。

        Args:
            start_date: 开始日期。
            end_date: 结束日期。
            index_code: 指数代码。

        Returns:
            DataBundle: 研究数据集合。
        """
        universe = self.get_index_components(
            index_code=index_code,
            start_date=start_date,
            end_date=end_date,
        )
        ts_codes = self.get_research_universe(
            index_code=index_code,
            start_date=start_date,
            end_date=end_date,
            universe_limit=universe_limit,
        )
        daily = self.load_stock_daily(ts_codes, start_date, end_date)
        daily_basic = self.load_stock_daily_basic(ts_codes, start_date, end_date)
        financial_indicators = self.load_financial_indicators(
            ts_codes=ts_codes,
            start_date=start_date,
            end_date=end_date,
        )
        benchmark = self.load_hs300_benchmark(
            start_date=start_date,
            end_date=end_date,
            index_code=index_code,
        )
        northbound = self.load_northbound_flow(start_date=start_date, end_date=end_date)

        if daily.empty:
            empty_panel = daily.copy()
            return DataBundle(
                universe=universe,
                daily=daily,
                daily_basic=daily_basic,
                financial_indicators=financial_indicators,
                benchmark=benchmark,
                northbound=northbound,
                research_panel=empty_panel,
            )

        panel = daily.merge(
            daily_basic,
            on=["ts_code", "trade_date"],
            how="left",
            suffixes=("", "_basic"),
        ) if not daily_basic.empty else daily.copy()
        panel = self._merge_financial_indicators(panel, financial_indicators)
        if not northbound.empty:
            panel = panel.merge(northbound, on="trade_date", how="left")
        if "ts_code" not in panel.columns and "ts_code_x" in panel.columns:
            panel["ts_code"] = panel["ts_code_x"]
        if "trade_date" not in panel.columns and "trade_date_x" in panel.columns:
            panel["trade_date"] = panel["trade_date_x"]
        panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

        return DataBundle(
            universe=universe,
            daily=daily,
            daily_basic=daily_basic,
            financial_indicators=financial_indicators,
            benchmark=benchmark,
            northbound=northbound,
            research_panel=panel,
        )

    def save_frame(self, frame: pd.DataFrame, relative_path: str) -> Path:
        """保存数据表到项目数据目录。

        Args:
            frame: 待保存的数据表。
            relative_path: 相对 `data/` 的保存路径。

        Returns:
            Path: 保存后的文件路径。
        """
        target_path = self.config.data_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target_path, index=False, encoding="utf-8-sig")
        return target_path

    def _concat_by_code(
        self,
        dataset_name: str,
        ts_codes: list[str],
        start_date: str,
        end_date: str,
        loader,
    ) -> pd.DataFrame:
        """按股票代码批量拼接数据。

        Args:
            ts_codes: 股票代码列表。
            loader: 单只股票加载函数。

        Returns:
            pd.DataFrame: 拼接后的数据表。
        """
        frames: list[pd.DataFrame] = []
        for ts_code in ts_codes:
            cache_path = self._build_cache_path(
                dataset_name=dataset_name,
                object_name=ts_code,
                start_date=start_date,
                end_date=end_date,
            )
            try:
                frame = self._load_or_cache(
                    cache_path=cache_path,
                    loader=lambda code=ts_code: loader(code),
                )
                if not frame.empty:
                    frames.append(frame)
            except Exception as exc:
                self._log_failure(
                    dataset_name=dataset_name,
                    object_name=ts_code,
                    error=str(exc),
                )
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _merge_financial_indicators(
        self,
        daily_panel: pd.DataFrame,
        financial_indicators: pd.DataFrame,
    ) -> pd.DataFrame:
        """将最近一期财务指标向前对齐到交易日。

        Args:
            daily_panel: 日频研究面板。
            financial_indicators: 财务指标表。

        Returns:
            pd.DataFrame: 融合财务指标后的面板数据。
        """
        if daily_panel.empty or financial_indicators.empty:
            return daily_panel

        result_frames: list[pd.DataFrame] = []
        financial_frame = financial_indicators.rename(
            columns={"grossprofit_margin": "grossprofitmargin"}
        ).copy()
        financial_frame["ann_date"] = pd.to_datetime(financial_frame["ann_date"])

        for ts_code, stock_frame in daily_panel.groupby("ts_code", group_keys=False):
            stock_daily = stock_frame.sort_values("trade_date").copy()
            stock_daily["trade_date"] = pd.to_datetime(stock_daily["trade_date"])
            stock_financial = financial_frame[financial_frame["ts_code"] == ts_code]
            if stock_financial.empty:
                stock_daily["roe"] = pd.NA
                stock_daily["grossprofitmargin"] = pd.NA
                result_frames.append(stock_daily)
                continue

            merged = pd.merge_asof(
                stock_daily,
                stock_financial.sort_values("ann_date"),
                left_on="trade_date",
                right_on="ann_date",
                by="ts_code",
                direction="backward",
            )
            result_frames.append(merged)

        result = pd.concat(result_frames, ignore_index=True)
        result["trade_date"] = result["trade_date"].dt.strftime("%Y%m%d")
        if "ann_date" in result.columns:
            result["ann_date"] = result["ann_date"].dt.strftime("%Y%m%d")
        return result.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    def _build_cache_path(
        self,
        dataset_name: str,
        object_name: str,
        start_date: str,
        end_date: str,
    ) -> Path:
        """构建缓存文件路径。"""
        safe_object_name = object_name.replace(".", "_")
        return (
            self.config.cache_dir
            / dataset_name
            / f"{safe_object_name}_{start_date}_{end_date}.csv"
        )

    @staticmethod
    def _compact_datetime(datetime_string: str) -> str:
        """将时间字符串压缩为适合文件名的形式。"""
        return (
            datetime_string.replace("-", "")
            .replace(":", "")
            .replace(" ", "_")
        )

    def _load_or_cache(
        self,
        cache_path: Path,
        loader: Callable[[], pd.DataFrame],
    ) -> pd.DataFrame:
        """优先读取缓存，否则执行加载并写入缓存。"""
        if cache_path.exists():
            return pd.read_csv(cache_path)
        frame = loader()
        if frame is None:
            return pd.DataFrame()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache_path, index=False, encoding="utf-8-sig")
        return frame

    def _log_failure(
        self,
        dataset_name: str,
        object_name: str,
        error: str,
    ) -> None:
        """记录失败下载信息。"""
        self.failure_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.failure_log_path.open("a", encoding="utf-8") as file:
            file.write(f"{dataset_name},{object_name},{error}\n")
