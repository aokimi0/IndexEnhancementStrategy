"""从离线 BaoStock CSV 包读取估值/财务/行情，替代在线 BaoStock 请求。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.akshare_client import AkshareClient

REQUIRED_FILES = (
    "constituents.csv",
    "daily_valuation.csv",
    "quarterly_financials.csv",
)
OPTIONAL_FILES = ("daily_kline.csv",)


def _normalize_date_series(series: pd.Series) -> pd.Series:
    """将日期列统一为 ``YYYYMMDD`` 字符串。"""
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y%m%d")


class LocalBaoStockCsvStore:
    """加载并索引离线 CSV 包，按 ``ts_code`` 提供与 ``AkshareClient`` 兼容的切片。"""

    def __init__(self, input_dir: Path) -> None:
        self.input_dir = Path(input_dir)
        self._constituents: pd.DataFrame | None = None
        self._daily_valuation: pd.DataFrame | None = None
        self._quarterly_financials: pd.DataFrame | None = None
        self._daily_kline: pd.DataFrame | None = None
        self._code6_to_ts: dict[str, str] = {}
        self._ts_to_code6: dict[str, str] = {}

    def load_all(self) -> None:
        """读取目录下全部 CSV 并建立代码映射。"""
        missing = [name for name in REQUIRED_FILES if not (self.input_dir / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"缺少必需文件 {missing}，请将 zip 解压到 {self.input_dir}"
            )

        constituents = pd.read_csv(self.input_dir / "constituents.csv")
        self._constituents = self._normalize_constituents(constituents)
        self._code6_to_ts = dict(
            zip(self._constituents["code6"], self._constituents["ts_code"])
        )
        self._ts_to_code6 = {v: k for k, v in self._code6_to_ts.items()}

        valuation = pd.read_csv(self.input_dir / "daily_valuation.csv")
        self._daily_valuation = self._normalize_daily_valuation(valuation)

        financials = pd.read_csv(self.input_dir / "quarterly_financials.csv")
        self._quarterly_financials = self._normalize_quarterly_financials(financials)

        kline_path = self.input_dir / "daily_kline.csv"
        if kline_path.exists():
            self._daily_kline = self._normalize_daily_kline(pd.read_csv(kline_path))

    @property
    def ts_codes(self) -> list[str]:
        """成分股 ``ts_code`` 列表。"""
        if self._constituents is None:
            raise RuntimeError("请先调用 load_all()")
        return sorted(self._constituents["ts_code"].dropna().unique().tolist())

    def has_daily_kline(self) -> bool:
        """是否提供了离线日频行情。"""
        return self._daily_kline is not None and not self._daily_kline.empty

    def get_daily(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """返回与 ``AkshareClient.daily`` 同 schema 的日频行情。"""
        if not self.has_daily_kline():
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
        assert self._daily_kline is not None
        frame = self._slice_by_code(self._daily_kline, ts_code, "trade_date", start_date, end_date)
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
                    "turnover_rate",
                ]
            )
        result = frame.copy()
        result["ts_code"] = ts_code
        if "pre_close" not in result.columns or result["pre_close"].isna().all():
            result["pre_close"] = result["close"].shift(1)
        return result.sort_values("trade_date").reset_index(drop=True)

    def get_daily_basic(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """返回与 ``AkshareClient.daily_basic`` 同 schema 的日频估值表。"""
        assert self._daily_valuation is not None
        frame = self._slice_by_code(
            self._daily_valuation, ts_code, "trade_date", start_date, end_date
        )
        if frame.empty:
            return AkshareClient._empty_daily_basic_frame()

        result = pd.DataFrame(
            {
                "ts_code": ts_code,
                "trade_date": frame["trade_date"],
                "turnover_rate": pd.to_numeric(frame["turnover_rate"], errors="coerce"),
                "turnover_rate_f": np.nan,
                "volume_ratio": np.nan,
                "pe_ttm": pd.to_numeric(frame["pe_ttm"], errors="coerce"),
                "pb": pd.to_numeric(frame["pb"], errors="coerce"),
                "total_mv": np.nan,
                "float_mv": np.nan,
                "ps_ttm": pd.to_numeric(frame["ps_ttm"], errors="coerce"),
                "dv_ttm": np.nan,
            }
        )
        return result.sort_values("trade_date").reset_index(drop=True)

    def get_fina_indicator(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """返回与 ``AkshareClient.fina_indicator`` 同 schema 的季度财务表。"""
        assert self._quarterly_financials is not None
        code6 = self._ts_to_code6.get(ts_code)
        if code6 is None:
            return AkshareClient._empty_fina_indicator_frame()

        frame = self._quarterly_financials[
            self._quarterly_financials["code6"].astype(str) == str(code6)
        ].copy()
        if frame.empty:
            return AkshareClient._empty_fina_indicator_frame()

        frame = frame[
            (frame["ann_date"] >= start_date) & (frame["ann_date"] <= end_date)
        ]
        if frame.empty:
            return AkshareClient._empty_fina_indicator_frame()

        result = pd.DataFrame(
            {
                "ts_code": ts_code,
                "ann_date": frame["ann_date"],
                "end_date": frame["end_date"],
                "roe": pd.to_numeric(frame["roe"], errors="coerce"),
                "grossprofit_margin": pd.to_numeric(
                    frame["grossprofit_margin"], errors="coerce"
                ),
                "netprofit_margin": pd.to_numeric(
                    frame["netprofit_margin"], errors="coerce"
                ),
                "yoy_net_profit": pd.to_numeric(frame["yoy_net_profit"], errors="coerce"),
                "asset_turnover": pd.to_numeric(frame["asset_turnover"], errors="coerce"),
                "cfo_to_or": pd.to_numeric(frame["cfo_to_or"], errors="coerce"),
                "equity_multiplier": pd.to_numeric(
                    frame["equity_multiplier"], errors="coerce"
                ),
            }
        )
        return result.drop_duplicates(subset=["ann_date", "end_date"]).sort_values(
            ["ann_date", "end_date"]
        ).reset_index(drop=True)

    def validate(self) -> dict[str, object]:
        """返回数据完整性摘要，便于导入前自检。"""
        if self._constituents is None:
            self.load_all()

        assert self._daily_valuation is not None
        assert self._quarterly_financials is not None

        val = self._daily_valuation
        fin = self._quarterly_financials
        summary: dict[str, object] = {
            "input_dir": str(self.input_dir),
            "constituents": len(self.ts_codes),
            "daily_valuation_rows": len(val),
            "daily_valuation_stocks": int(val["code6"].nunique()),
            "pe_ttm_non_null_ratio": float(val["pe_ttm"].notna().mean()),
            "pb_non_null_ratio": float(val["pb"].notna().mean()),
            "quarterly_rows": len(fin),
            "quarterly_stocks": int(fin["code6"].nunique()),
            "roe_non_null_ratio": float(fin["roe"].notna().mean()),
            "grossprofit_margin_non_null_ratio": float(
                fin["grossprofit_margin"].notna().mean()
            ),
            "has_daily_kline": self.has_daily_kline(),
        }
        if self.has_daily_kline():
            assert self._daily_kline is not None
            summary["daily_kline_rows"] = len(self._daily_kline)
            summary["daily_kline_stocks"] = int(self._daily_kline["code6"].nunique())
        return summary

    def _slice_by_code(
        self,
        frame: pd.DataFrame,
        ts_code: str,
        date_col: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        code6 = self._ts_to_code6.get(ts_code)
        if code6 is None:
            return frame.iloc[0:0].copy()
        subset = frame[frame["code6"].astype(str) == str(code6)].copy()
        if subset.empty:
            return subset
        return subset[
            (subset[date_col] >= start_date) & (subset[date_col] <= end_date)
        ]

    def _normalize_constituents(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        if "code6" not in result.columns and "code" in result.columns:
            result["code6"] = result["code"].astype(str).str.replace(r"^(sh|sz)\.", "", regex=True)
        result["code6"] = result["code6"].astype(str).str.zfill(6)
        result["ts_code"] = result["code6"].map(AkshareClient._to_ts_code)
        return result.dropna(subset=["ts_code"]).drop_duplicates(subset=["ts_code"])

    def _normalize_daily_valuation(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result["code6"] = self._resolve_code6(result)
        result["trade_date"] = _normalize_date_series(result["date"])
        result["turnover_rate"] = pd.to_numeric(
            result.get("turn", result.get("turnover_rate")), errors="coerce"
        )
        result["pe_ttm"] = pd.to_numeric(
            result.get("peTTM", result.get("pe_ttm")), errors="coerce"
        )
        result["pb"] = pd.to_numeric(
            result.get("pbMRQ", result.get("pb")), errors="coerce"
        )
        result["ps_ttm"] = pd.to_numeric(
            result.get("psTTM", result.get("ps_ttm")), errors="coerce"
        )
        return result.dropna(subset=["code6", "trade_date"])

    def _normalize_quarterly_financials(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result["code6"] = self._resolve_code6(result)
        result["ann_date"] = _normalize_date_series(result.get("pubDate", result.get("ann_date")))
        result["end_date"] = _normalize_date_series(result.get("statDate", result.get("end_date")))
        mapping = {
            "roe": ["roeAvg", "roe"],
            "grossprofit_margin": ["gpMargin", "grossprofit_margin"],
            "netprofit_margin": ["npMargin", "netprofit_margin"],
            "yoy_net_profit": ["YOYNI", "yoy_net_profit"],
            "asset_turnover": ["AssetTurnRatio", "dupontAssetTurn", "asset_turnover"],
            "cfo_to_or": ["CFOToOR", "cfo_to_or"],
            "equity_multiplier": ["dupontAssetStoEquity", "equity_multiplier"],
        }
        for target, candidates in mapping.items():
            for candidate in candidates:
                if candidate in result.columns:
                    result[target] = pd.to_numeric(result[candidate], errors="coerce")
                    break
            if target not in result.columns:
                result[target] = np.nan
        if "asset_turnover" in result.columns and "dupontAssetTurn" in result.columns:
            alt = pd.to_numeric(result["dupontAssetTurn"], errors="coerce")
            result["asset_turnover"] = result["asset_turnover"].combine_first(alt)
        return result.dropna(subset=["code6", "ann_date", "end_date"])

    def _normalize_daily_kline(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result["code6"] = self._resolve_code6(result)
        result["trade_date"] = _normalize_date_series(result["date"])
        rename_map = {
            "preclose": "pre_close",
            "pre_close": "pre_close",
            "volume": "vol",
            "vol": "vol",
        }
        result = result.rename(columns=rename_map)
        for col in ("open", "high", "low", "close", "pre_close", "vol", "amount"):
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors="coerce")
        if "turn" in result.columns:
            result["turnover_rate"] = pd.to_numeric(result["turn"], errors="coerce")
        elif "turnover_rate" not in result.columns:
            result["turnover_rate"] = np.nan
        return result.dropna(subset=["code6", "trade_date"])

    @staticmethod
    def _resolve_code6(frame: pd.DataFrame) -> pd.Series:
        if "code6" in frame.columns:
            return frame["code6"].astype(str).str.zfill(6)
        if "code" in frame.columns:
            return (
                frame["code"]
                .astype(str)
                .str.replace(r"^(sh|sz)\.", "", regex=True)
                .str.zfill(6)
            )
        raise ValueError("CSV 缺少 code 或 code6 列")


class HybridAkshareClient(AkshareClient):
    """估值/财务（及可选行情）走离线 CSV，宏观/基准/北向等仍走 akshare。"""

    def __init__(self, local_csv_dir: Path) -> None:
        super().__init__()
        self._store = LocalBaoStockCsvStore(local_csv_dir)
        self._store.load_all()

    @property
    def local_store(self) -> LocalBaoStockCsvStore:
        """暴露底层 CSV 存储，供导入脚本写入缓存。"""
        return self._store

    def daily(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        fields: str | None = None,
    ) -> pd.DataFrame:
        if self._store.has_daily_kline():
            return self._store.get_daily(ts_code, start_date, end_date)
        return super().daily(ts_code, start_date, end_date, fields)

    def daily_basic(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        fields: str | None = None,
    ) -> pd.DataFrame:
        del fields
        return self._store.get_daily_basic(ts_code, start_date, end_date)

    def fina_indicator(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        fields: str | None = None,
    ) -> pd.DataFrame:
        del fields
        return self._store.get_fina_indicator(ts_code, start_date, end_date)
