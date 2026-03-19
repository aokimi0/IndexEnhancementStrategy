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
    对于 Akshare 覆盖较弱的日频估值和财务指标，内部会自动尝试使用
    BaoStock 做补充，以减少研究阶段的数据缺口。
    当前版本重点覆盖：

    - 沪深300成分
    - 个股日线
    - 沪深300指数日线
    - 个股分钟级行情
    - 指数分钟级行情

    北向资金数据暂时仍返回空表。
    """

    def __init__(self) -> None:
        """初始化客户端。"""
        self._ak = None
        self._bs = None
        self._bs_logged_in = False

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

    @property
    def bs(self) -> Any:
        """延迟初始化 BaoStock 模块并完成登录。"""
        if self._bs is None:
            try:
                import baostock as bs
            except ImportError as exc:
                raise ImportError(
                    "未安装 baostock，请先在 conda 的 index-enhancement 环境中安装。"
                ) from exc
            self._bs = bs
        if not self._bs_logged_in:
            login_result = self._bs.login()
            if getattr(login_result, "error_code", "0") != "0":
                raise RuntimeError(f"baostock 登录失败: {login_result.error_msg}")
            self._bs_logged_in = True
        return self._bs

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
        weight_frame = self.ak.index_stock_cons_weight_csindex(symbol=symbol)
        if weight_frame.empty:
            frame = self.ak.index_stock_cons_csindex(symbol=symbol)
            if frame.empty:
                return pd.DataFrame(columns=["trade_date", "index_code", "con_code", "weight"])

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
                    "weight": np.nan,
                }
            )
        else:
            date_col = weight_frame.columns[0]
            index_code_col = weight_frame.columns[1]
            con_code_col = weight_frame.columns[4]
            weight_col = weight_frame.columns[-1]
            result = pd.DataFrame(
                {
                    "trade_date": pd.to_datetime(weight_frame[date_col]).dt.strftime("%Y%m%d"),
                    "index_code": weight_frame[index_code_col].astype(str).map(
                        lambda x: f"{x}.SH" if x.startswith("000") else x
                    ),
                    "con_code": weight_frame[con_code_col].astype(str).map(self._to_ts_code),
                    "weight": pd.to_numeric(weight_frame[weight_col], errors="coerce"),
                }
            )
        if result["trade_date"].nunique() == 1:
            result["trade_date"] = end_date
        else:
            result = result[
                (result["trade_date"] >= start_date) & (result["trade_date"] <= end_date)
            ]
        return result.reset_index(drop=True)

    def stock_industry(
        self,
        ts_code: str,
    ) -> pd.DataFrame:
        """获取个股所属行业。"""
        symbol = self._normalize_symbol(ts_code)
        frame = self._with_retry(lambda: self.ak.stock_individual_info_em(symbol=symbol))
        if frame.empty:
            return pd.DataFrame(columns=["ts_code", "industry_name"])

        item_col = frame.columns[0]
        value_col = frame.columns[1]
        item_series = frame[item_col].astype(str)
        industry_rows = frame[item_series.str.contains("行业", na=False)]
        industry_name = (
            industry_rows.iloc[0][value_col]
            if not industry_rows.empty
            else "未知行业"
        )
        return pd.DataFrame([{"ts_code": self._to_ts_code(ts_code), "industry_name": industry_name}])

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
        """基于 BaoStock 获取日频估值与换手率指标。"""
        del fields
        frame = self._query_baostock_history_metrics(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
        )
        if frame.empty:
            return self._empty_daily_basic_frame()

        result = pd.DataFrame(
            {
                "ts_code": ts_code,
                "trade_date": pd.to_datetime(frame["date"]).dt.strftime("%Y%m%d"),
                "turnover_rate": pd.to_numeric(frame["turn"], errors="coerce"),
                "turnover_rate_f": np.nan,
                "volume_ratio": np.nan,
                "pe_ttm": pd.to_numeric(frame["peTTM"], errors="coerce"),
                "pb": pd.to_numeric(frame["pbMRQ"], errors="coerce"),
                "total_mv": np.nan,
                "float_mv": np.nan,
                "ps_ttm": pd.to_numeric(frame["psTTM"], errors="coerce"),
                "dv_ttm": np.nan,
            }
        )
        return result.sort_values("trade_date").reset_index(drop=True)

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
        """获取北向资金历史流向。"""
        del fields
        sh_frame = self._query_hsgt_hist(symbol="\u6caa\u80a1\u901a")
        sz_frame = self._query_hsgt_hist(symbol="\u6df1\u80a1\u901a")
        if sh_frame.empty and sz_frame.empty:
            return pd.DataFrame(
                columns=["trade_date", "north_money", "sh_amount", "sz_amount"]
            )

        result = pd.DataFrame()
        if not sh_frame.empty:
            result["trade_date"] = pd.to_datetime(sh_frame["date"]).dt.strftime("%Y%m%d")
            result["sh_amount"] = pd.to_numeric(sh_frame["net_amount"], errors="coerce")
        if not sz_frame.empty:
            sz_result = pd.DataFrame(
                {
                    "trade_date": pd.to_datetime(sz_frame["date"]).dt.strftime("%Y%m%d"),
                    "sz_amount": pd.to_numeric(sz_frame["net_amount"], errors="coerce"),
                }
            )
            result = (
                result.merge(sz_result, on="trade_date", how="outer")
                if not result.empty
                else sz_result
            )
        if "sh_amount" not in result.columns:
            result["sh_amount"] = np.nan
        if "sz_amount" not in result.columns:
            result["sz_amount"] = np.nan
        result["north_money"] = result["sh_amount"].fillna(0.0) + result["sz_amount"].fillna(0.0)
        result = result[
            (result["trade_date"] >= start_date) & (result["trade_date"] <= end_date)
        ]
        return result.sort_values("trade_date").reset_index(drop=True)

    def macro_m2_yoy(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """获取 M2 同比时间序列。"""
        frame = self._with_retry(lambda: self.ak.macro_china_m2_yearly())
        if frame.empty:
            return pd.DataFrame(columns=["trade_date", "m2_yoy"])

        date_col = frame.columns[1]
        value_col = frame.columns[2]
        result = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(frame[date_col], errors="coerce"),
                "m2_yoy": pd.to_numeric(frame[value_col], errors="coerce"),
            }
        )
        result = result.dropna(subset=["trade_date", "m2_yoy"]).copy()
        result = result[
            (result["trade_date"] >= pd.to_datetime(start_date))
            & (result["trade_date"] <= pd.to_datetime(end_date))
        ]
        result["trade_date"] = result["trade_date"].dt.strftime("%Y%m%d")
        return result.sort_values("trade_date").reset_index(drop=True)

    def macro_interest_rate_spread(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """获取中美主要国债收益率及利差。
        使用 bond_zh_us_rate 接口获取中国国债10年-2年利差等信息。
        """
        frame = self._with_retry(lambda: self.ak.bond_zh_us_rate())
        if frame.empty:
            return pd.DataFrame(columns=["trade_date", "cn_spread_10y_2y"])

        # 列名中可能包含特殊字符，使用位置索引或者容错获取
        date_col = frame.columns[0]
        # 正常为 '中国国债收益率10年-2年'
        cn_spread_col = "中国国债收益率10年-2年"
        if cn_spread_col not in frame.columns:
            # 取第5列为 fallback (通常第5列或第6列为利差)
            for col in frame.columns:
                if "10年-2年" in col and "中国" in col:
                    cn_spread_col = col
                    break
        
        if cn_spread_col not in frame.columns:
            return pd.DataFrame(columns=["trade_date", "cn_spread_10y_2y"])

        result = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(frame[date_col], errors="coerce"),
                "cn_spread_10y_2y": pd.to_numeric(frame[cn_spread_col], errors="coerce"),
            }
        )
        result = result.dropna(subset=["trade_date", "cn_spread_10y_2y"]).copy()
        result = result[
            (result["trade_date"] >= pd.to_datetime(start_date))
            & (result["trade_date"] <= pd.to_datetime(end_date))
        ]
        result["trade_date"] = result["trade_date"].dt.strftime("%Y%m%d")
        return result.sort_values("trade_date").reset_index(drop=True)

    def fina_indicator(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        fields: str | None = None,
    ) -> pd.DataFrame:
        """基于 BaoStock 获取季度财务指标。"""
        del fields
        report_frame = self._query_baostock_financial_data(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
        )
        if report_frame.empty:
            return self._empty_fina_indicator_frame()

        report_frame["pubDate"] = pd.to_datetime(report_frame["pubDate"], errors="coerce")
        report_frame["statDate"] = pd.to_datetime(report_frame["statDate"], errors="coerce")
        report_frame = report_frame.dropna(subset=["pubDate", "statDate"])
        report_frame = report_frame[report_frame["pubDate"] <= pd.to_datetime(end_date)]
        if report_frame.empty:
            return self._empty_fina_indicator_frame()

        result = pd.DataFrame(
            {
                "ts_code": ts_code,
                "ann_date": report_frame["pubDate"].dt.strftime("%Y%m%d"),
                "end_date": report_frame["statDate"].dt.strftime("%Y%m%d"),
                "roe": pd.to_numeric(report_frame["roeAvg"], errors="coerce"),
                "grossprofit_margin": pd.to_numeric(
                    report_frame["gpMargin"], errors="coerce"
                ),
                "netprofit_margin": pd.to_numeric(
                    report_frame["npMargin"], errors="coerce"
                ),
                "yoy_net_profit": pd.to_numeric(report_frame["YOYNI"], errors="coerce"),
                "asset_turnover": pd.to_numeric(
                    report_frame["AssetTurnRatio"], errors="coerce"
                ).combine_first(
                    pd.to_numeric(report_frame["dupontAssetTurn"], errors="coerce")
                ),
                "cfo_to_or": pd.to_numeric(report_frame["CFOToOR"], errors="coerce"),
                "equity_multiplier": pd.to_numeric(
                    report_frame["dupontAssetStoEquity"], errors="coerce"
                ),
            }
        )
        result = result.drop_duplicates(subset=["ann_date", "end_date"]).copy()
        return result.sort_values(["ann_date", "end_date"]).reset_index(drop=True)

    def _query_baostock_history_metrics(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """查询单只股票的日频估值与换手率指标。"""
        rs = self.bs.query_history_k_data_plus(
            self._to_baostock_code(ts_code),
            "date,code,turn,peTTM,pbMRQ,psTTM",
            start_date=self._to_baostock_date(start_date),
            end_date=self._to_baostock_date(end_date),
            frequency="d",
            adjustflag="3",
        )
        return self._baostock_result_to_frame(rs)

    def _query_hsgt_hist(self, symbol: str) -> pd.DataFrame:
        """查询单个互联互通通道的历史净买额。"""
        frame = self._with_retry(lambda: self.ak.stock_hsgt_hist_em(symbol=symbol))
        if frame.empty:
            return pd.DataFrame(columns=["date", "net_amount"])
        columns = frame.columns.tolist()
        return pd.DataFrame(
            {
                "date": pd.to_datetime(frame[columns[0]], errors="coerce"),
                "net_amount": pd.to_numeric(frame[columns[1]], errors="coerce"),
            }
        ).dropna(subset=["date"])

    def _query_baostock_financial_data(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """查询单只股票区间内可用的季度财务指标。"""
        frames: list[pd.DataFrame] = []
        for year, quarter in self._iter_year_quarters(start_date, end_date):
            quarter_frame = self._query_baostock_quarter_frame(
                ts_code=ts_code,
                year=year,
                quarter=quarter,
            )
            if not quarter_frame.empty:
                frames.append(quarter_frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _query_baostock_quarter_frame(
        self,
        ts_code: str,
        year: int,
        quarter: int,
    ) -> pd.DataFrame:
        """查询单个季度的多张财务表并按公告日合并。"""
        code = self._to_baostock_code(ts_code)
        base_keys = ["code", "pubDate", "statDate"]
        query_functions = [
            self.bs.query_profit_data,
            self.bs.query_growth_data,
            self.bs.query_operation_data,
            self.bs.query_cash_flow_data,
            self.bs.query_dupont_data,
        ]

        merged: pd.DataFrame | None = None
        for query_func in query_functions:
            frame = self._baostock_result_to_frame(
                query_func(code=code, year=year, quarter=quarter)
            )
            if frame.empty:
                continue
            if merged is None:
                merged = frame
                continue
            merged = merged.merge(frame, on=base_keys, how="outer")

        if merged is None:
            return pd.DataFrame()
        return merged

    @staticmethod
    def _baostock_result_to_frame(result: Any) -> pd.DataFrame:
        """将 BaoStock 查询结果转换为 DataFrame。"""
        if getattr(result, "error_code", "0") != "0":
            raise RuntimeError(f"baostock 查询失败: {result.error_msg}")

        rows: list[list[str]] = []
        while result.next():
            rows.append(result.get_row_data())
        if not rows:
            return pd.DataFrame(columns=getattr(result, "fields", []))
        return pd.DataFrame(rows, columns=result.fields)

    @staticmethod
    def _iter_year_quarters(
        start_date: str,
        end_date: str,
    ) -> list[tuple[int, int]]:
        """根据日期区间生成需要查询的年报季报范围。"""
        start_year = pd.to_datetime(start_date).year - 1
        end_year = pd.to_datetime(end_date).year
        return [(year, quarter) for year in range(start_year, end_year + 1) for quarter in range(1, 5)]

    @staticmethod
    def _to_baostock_code(ts_code: str) -> str:
        """将 Tushare 风格代码转为 BaoStock 风格代码。"""
        normalized_code = AkshareClient._to_ts_code(ts_code)
        symbol, market = normalized_code.split(".")
        prefix = "sh" if market.upper() == "SH" else "sz"
        return f"{prefix}.{symbol}"

    @staticmethod
    def _to_baostock_date(date_string: str) -> str:
        """将 `YYYYMMDD` 转为 BaoStock 需要的 `YYYY-MM-DD`。"""
        return pd.to_datetime(date_string).strftime("%Y-%m-%d")

    @staticmethod
    def _empty_daily_basic_frame() -> pd.DataFrame:
        """构建空的日频基本面表。"""
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

    @staticmethod
    def _empty_fina_indicator_frame() -> pd.DataFrame:
        """构建空的财务指标表。"""
        return pd.DataFrame(
            columns=[
                "ts_code",
                "ann_date",
                "end_date",
                "roe",
                "grossprofit_margin",
                "netprofit_margin",
                "yoy_net_profit",
                "asset_turnover",
                "cfo_to_or",
                "equity_multiplier",
            ]
        )

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
