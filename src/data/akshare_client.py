"""Akshare 数据客户端。"""

from __future__ import annotations

from datetime import date
import time
from typing import Any

import numpy as np
import pandas as pd


class AkshareClient:
    """Akshare 轻量客户端。

    该客户端作为 Tushare 权限不足时的兜底实现，优先保证研究链路可运行。
    当前版本重点覆盖：

    - 沪深300成分
    - 个股日线
    - 沪深300指数日线
    - 个股分钟级行情
    - 指数分钟级行情

    估值、财务和北向资金数据暂时返回空表，由后续阶段再逐步补齐。
    """

    def __init__(self) -> None:
        """初始化客户端。"""
        self._ak = None

    @property
    def ak(self) -> Any:
        """延迟初始化 Akshare 模块。"""
        if self._ak is None:
            try:
                import akshare as ak
            except ImportError as exc:
                raise ImportError("未安装 akshare，请先安装后再运行兜底数据流。") from exc
            self._ak = ak
        return self._ak

    def index_weight(
        self,
        index_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """获取指数成分表。

        Args:
            index_code: 指数代码，如 `000300.SH`。
            start_date: 开始日期。
            end_date: 结束日期。

        Returns:
            pd.DataFrame: 与 `index_weight` 接口兼容的简化结果。
        """
        symbol = self._normalize_symbol(index_code)
        frame = self.ak.index_stock_cons_csindex(symbol=symbol)
        if frame.empty:
            return pd.DataFrame(columns=["trade_date", "index_code", "con_code"])

        date_col = frame.columns[0]
        index_code_col = frame.columns[1]
        con_code_col = frame.columns[4]

        result = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(frame[date_col]).dt.strftime("%Y%m%d"),
                "index_code": frame[index_code_col].astype(str).map(
                    lambda x: f"{x}.SH" if x.startswith("000") else x
                ),
                "con_code": frame[con_code_col].astype(str).map(self._to_ts_code),
            }
        )
        if result["trade_date"].nunique() == 1:
            result["trade_date"] = end_date
        else:
            result = result[
                (result["trade_date"] >= start_date) & (result["trade_date"] <= end_date)
            ]
        return result.reset_index(drop=True)

    def daily(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        fields: str | None = None,
    ) -> pd.DataFrame:
        """获取个股日线行情。"""
        del fields
        symbol = self._to_akshare_market_symbol(ts_code)
        frame = self._with_retry(
            lambda: self.ak.stock_zh_a_hist_tx(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust="",
            )
        )
        if frame.empty:
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
                ]
            )

        date_col = frame.columns[0]
        open_col = frame.columns[1]
        close_col = frame.columns[2]
        high_col = frame.columns[3]
        low_col = frame.columns[4]
        vol_col = frame.columns[5]

        result = pd.DataFrame(
            {
                "ts_code": ts_code,
                "trade_date": pd.to_datetime(frame[date_col]).dt.strftime("%Y%m%d"),
                "open": frame[open_col],
                "close": frame[close_col],
                "high": frame[high_col],
                "low": frame[low_col],
                "vol": frame[vol_col],
                "amount": np.nan,
                "turnover_rate": np.nan,
            }
        )
        result = result.sort_values("trade_date").reset_index(drop=True)
        result["pre_close"] = result["close"].shift(1)
        return result[
            [
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
        ]

    def daily_basic(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        fields: str | None = None,
    ) -> pd.DataFrame:
        """返回空的日频基本面表。"""
        del ts_code, start_date, end_date, fields
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

    def index_daily(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        fields: str | None = None,
    ) -> pd.DataFrame:
        """获取指数日线行情。"""
        del fields
        symbol = self._normalize_symbol(ts_code)
        frame = self._with_retry(
            lambda: self.ak.stock_zh_index_hist_csindex(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
            )
        )
        if frame.empty:
            return pd.DataFrame(
                columns=["ts_code", "trade_date", "open", "high", "low", "close", "pre_close"]
            )

        date_col = frame.columns[0]
        open_col = frame.columns[6]
        high_col = frame.columns[7]
        low_col = frame.columns[8]
        close_col = frame.columns[9]
        vol_col = frame.columns[12]
        amount_col = frame.columns[13]

        result = pd.DataFrame(
            {
                "ts_code": ts_code,
                "trade_date": pd.to_datetime(frame[date_col]).dt.strftime("%Y%m%d"),
                "open": frame[open_col],
                "high": frame[high_col],
                "low": frame[low_col],
                "close": frame[close_col],
                "vol": frame[vol_col],
                "amount": frame[amount_col],
            }
        )
        result = result.sort_values("trade_date").reset_index(drop=True)
        result["pre_close"] = result["close"].shift(1)
        return result

    def minute(
        self,
        ts_code: str,
        start_datetime: str,
        end_datetime: str,
        period: str = "5",
    ) -> pd.DataFrame:
        """获取个股分钟级行情。"""
        symbol = self._to_akshare_market_symbol(ts_code)
        frame = self._with_retry(
            lambda: self.ak.stock_zh_a_minute(
                symbol=symbol,
                period=period,
                adjust="",
            )
        )
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

        columns = frame.columns.tolist()
        result = pd.DataFrame(
            {
                "ts_code": ts_code,
                "trade_datetime": pd.to_datetime(frame[columns[0]], errors="coerce"),
                "open": pd.to_numeric(frame[columns[1]], errors="coerce"),
                "high": pd.to_numeric(frame[columns[2]], errors="coerce"),
                "low": pd.to_numeric(frame[columns[3]], errors="coerce"),
                "close": pd.to_numeric(frame[columns[4]], errors="coerce"),
                "vol": pd.to_numeric(frame[columns[5]], errors="coerce"),
                "amount": pd.to_numeric(frame[columns[6]], errors="coerce"),
            }
        )
        start_ts = pd.to_datetime(start_datetime)
        end_ts = pd.to_datetime(end_datetime)
        result = result[
            (result["trade_datetime"] >= start_ts)
            & (result["trade_datetime"] <= end_ts)
        ]
        return result.sort_values("trade_datetime").reset_index(drop=True)

    def index_minute(
        self,
        ts_code: str,
        start_datetime: str,
        end_datetime: str,
        period: str = "5",
    ) -> pd.DataFrame:
        """获取指数分钟级行情。

        当前使用 `510300` ETF 作为沪深300分钟级近似代理，以提高接口稳定性。
        """
        symbol = "sh510300" if ts_code == "000300.SH" else self._to_akshare_market_symbol(ts_code)
        frame = self._with_retry(
            lambda: self.ak.stock_zh_a_minute(
                symbol=symbol,
                period=period,
                adjust="",
            )
        )
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

        columns = frame.columns.tolist()
        result = pd.DataFrame(
            {
                "ts_code": ts_code,
                "trade_datetime": pd.to_datetime(frame[columns[0]], errors="coerce"),
                "open": pd.to_numeric(frame[columns[1]], errors="coerce"),
                "high": pd.to_numeric(frame[columns[2]], errors="coerce"),
                "low": pd.to_numeric(frame[columns[3]], errors="coerce"),
                "close": pd.to_numeric(frame[columns[4]], errors="coerce"),
                "vol": pd.to_numeric(frame[columns[5]], errors="coerce"),
                "amount": pd.to_numeric(frame[columns[6]], errors="coerce"),
            }
        )
        start_ts = pd.to_datetime(start_datetime)
        end_ts = pd.to_datetime(end_datetime)
        result = result[
            (result["trade_datetime"] >= start_ts)
            & (result["trade_datetime"] <= end_ts)
        ]
        return result.sort_values("trade_datetime").reset_index(drop=True)

    def moneyflow_hsgt(
        self,
        start_date: str,
        end_date: str,
        fields: str | None = None,
    ) -> pd.DataFrame:
        """返回空的北向资金流表。"""
        del start_date, end_date, fields
        return pd.DataFrame(columns=["trade_date", "north_money", "sh_amount", "sz_amount"])

    def fina_indicator(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        fields: str | None = None,
    ) -> pd.DataFrame:
        """返回空的财务指标表。"""
        del ts_code, start_date, end_date, fields
        return pd.DataFrame(columns=["ts_code", "ann_date", "end_date", "roe", "grossprofit_margin"])

    @staticmethod
    def _normalize_symbol(ts_code: str) -> str:
        """将 Tushare 风格代码转为 Akshare 所需格式。"""
        return ts_code.split(".")[0]

    @staticmethod
    def _to_akshare_market_symbol(ts_code: str) -> str:
        """将 Tushare 风格代码转为 Akshare 腾讯接口格式。"""
        normalized_code = AkshareClient._to_ts_code(ts_code)
        symbol, market = normalized_code.split(".")
        prefix = "sh" if market.upper() == "SH" else "sz"
        return f"{prefix}{symbol}"

    @staticmethod
    def _to_ts_code(symbol: str) -> str:
        """将 6 位股票代码转为 Tushare 风格代码。"""
        if "." in symbol:
            return symbol
        if symbol.startswith(("600", "601", "603", "605", "688", "689")):
            return f"{symbol}.SH"
        if symbol.startswith(("000", "001", "002", "003")):
            return f"{symbol}.SZ"
        if symbol.startswith("30"):
            return f"{symbol}.SZ"
        return symbol

    @staticmethod
    def _with_retry(func, retries: int = 3, sleep_seconds: float = 1.0) -> pd.DataFrame:
        """执行带重试的数据请求。

        Args:
            func: 数据请求函数。
            retries: 最大重试次数。
            sleep_seconds: 重试间隔秒数。

        Returns:
            pd.DataFrame: 请求结果。
        """
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                return func()
            except Exception as exc:  # pragma: no cover - 第三方接口波动
                last_error = exc
                if attempt == retries - 1:
                    raise
                time.sleep(sleep_seconds)
        if last_error is not None:
            raise last_error
        return pd.DataFrame()
