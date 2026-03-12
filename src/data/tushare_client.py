"""Tushare 客户端封装。"""

from __future__ import annotations

from typing import Any

import pandas as pd


class TushareClient:
    """对 Tushare Pro 接口的轻量封装。

    该类只保留当前项目最需要的少量接口，避免把数据层写成散乱的脚本。
    """

    def __init__(self, token: str) -> None:
        """初始化客户端。

        Args:
            token: Tushare Pro 令牌。
        """
        if not token:
            raise ValueError("缺少 Tushare Token，请先设置环境变量 TUSHARE_TOKEN。")
        self._token = token
        self._pro = None

    @property
    def pro(self) -> Any:
        """延迟初始化 Tushare Pro 客户端。

        Returns:
            Any: `tushare.pro_api()` 返回的客户端对象。
        """
        if self._pro is None:
            try:
                import tushare as ts
            except ImportError as exc:
                raise ImportError(
                    "未安装 tushare，请先安装后再运行数据管道。"
                ) from exc
            ts.set_token(self._token)
            self._pro = ts.pro_api()
        return self._pro

    def query(self, api_name: str, **kwargs: Any) -> pd.DataFrame:
        """调用指定接口。

        Args:
            api_name: Tushare Pro 接口名。
            **kwargs: 接口参数。

        Returns:
            pd.DataFrame: 查询结果。
        """
        method = getattr(self.pro, api_name)
        frame = method(**kwargs)
        return frame if frame is not None else pd.DataFrame()

    def daily(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        fields: str | None = None,
    ) -> pd.DataFrame:
        """获取个股日线行情。"""
        return self.query(
            "daily",
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields=fields,
        )

    def daily_basic(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        fields: str | None = None,
    ) -> pd.DataFrame:
        """获取个股日频基本面指标。"""
        return self.query(
            "daily_basic",
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields=fields,
        )

    def index_daily(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        fields: str | None = None,
    ) -> pd.DataFrame:
        """获取指数日线行情。"""
        return self.query(
            "index_daily",
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields=fields,
        )

    def index_weight(
        self,
        index_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """获取指数成分权重。"""
        return self.query(
            "index_weight",
            index_code=index_code,
            start_date=start_date,
            end_date=end_date,
        )

    def fina_indicator(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        fields: str | None = None,
    ) -> pd.DataFrame:
        """获取财务指标数据。"""
        return self.query(
            "fina_indicator",
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields=fields,
        )

    def moneyflow_hsgt(
        self,
        start_date: str,
        end_date: str,
        fields: str | None = None,
    ) -> pd.DataFrame:
        """获取沪深港通资金流向。"""
        return self.query(
            "moneyflow_hsgt",
            start_date=start_date,
            end_date=end_date,
            fields=fields,
        )

    def macro_m2_yoy(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """返回空的 M2 同比表。"""
        del start_date, end_date
        return pd.DataFrame(columns=["trade_date", "m2_yoy"])