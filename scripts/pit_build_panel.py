"""Phase 1.3 + 1.4：构建并集因子面板并做点位后处理。

流程：
  1) 以点位并集(639)为股票池，复用 DataService 的缓存读取与拼接逻辑构建原始面板；
  2) FactorEngine 计算因子 + 超额标签（在过滤前，保证滚动窗口/前瞻收益正确）；
  3) 补 industry_name（沿用已有 299 映射，缺失填“未知行业”）；
  4) 点位过滤：仅保留 ts_code 在该 trade_date 所属月份为沪深300成分的行；
  5) benchmark_weight：每个 trade_date 内按 float_mv 代理(close*vol/turn)归一化，
     缺失用当日中位数兜底并记录占比；
  6) 截面标准化(行业中性) -> 存基线面板；外部增强版由 augment 脚本另产。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ProjectConfig  # noqa: E402
from src.data import AkshareClient, DataService  # noqa: E402
from src.factors import FactorEngine  # noqa: E402
from scripts.pit_common import log  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MEMBERSHIP = ROOT / "data" / "processed" / "hs300_pit_membership.csv"
EXISTING_PANEL = ROOT / "data" / "processed" / "hs300_factor_panel_constrained_fast_extended_2015_2024.csv"
OUT_BASE = "processed/hs300_factor_panel_pit_2015_2024.csv"
START, END = "20150101", "20241231"


class PitDataService(DataService):
    """覆盖股票池与行业加载，使用点位并集、跳过逐股行业请求。"""

    def __init__(self, client, config, union: list[str]) -> None:
        super().__init__(client=client, config=config)
        self._union = union

    def get_index_components(self, index_code, start_date, end_date):  # type: ignore[override]
        return pd.DataFrame({"trade_date": END, "index_code": index_code, "con_code": self._union})

    def get_research_universe(self, index_code, start_date, end_date, universe_limit=None):  # type: ignore[override]
        return list(self._union)

    def load_stock_industry(self, ts_codes):  # type: ignore[override]
        return pd.DataFrame(columns=["ts_code", "industry_name"])

    @staticmethod
    def _coerce_trade_date_str(frame: pd.DataFrame) -> pd.DataFrame:
        if not frame.empty and "trade_date" in frame.columns:
            frame = frame.copy()
            frame["trade_date"] = frame["trade_date"].astype(str)
        return frame

    def load_northbound_flow(self, start_date, end_date):  # type: ignore[override]
        return self._coerce_trade_date_str(super().load_northbound_flow(start_date, end_date))

    def load_macro_m2_yoy(self, start_date, end_date):  # type: ignore[override]
        return self._coerce_trade_date_str(super().load_macro_m2_yoy(start_date, end_date))

    def load_macro_interest_rate_spread(self, start_date, end_date):  # type: ignore[override]
        return self._coerce_trade_date_str(super().load_macro_interest_rate_spread(start_date, end_date))


def main() -> None:
    log("===== Phase 1.3/1.4 面板构建 + 点位后处理 开始 =====")
    config = ProjectConfig.from_root()
    config.ensure_directories()

    membership = pd.read_csv(MEMBERSHIP)
    membership["mkey"] = membership["month"].str.replace("-", "", regex=False)  # YYYYMM
    member_keys = set(zip(membership["ts_code"], membership["mkey"]))
    union = sorted(membership["ts_code"].unique().tolist())
    log(f"点位并集 universe: {len(union)} 只, 月份: {membership['month'].nunique()}")

    service = PitDataService(client=AkshareClient(), config=config, union=union)
    log("读取缓存并拼接原始面板（daily/daily_basic/financial/benchmark/macro）……")
    bundle = service.build_research_panel(start_date=START, end_date=END, index_code="000300.SH")
    raw = bundle.research_panel
    log(f"原始面板: {raw.shape}, 股票 {raw['ts_code'].nunique()}")

    engine = FactorEngine()
    log("计算因子……")
    factors = engine.compute_factors(raw)
    log("生成超额标签……")
    labeled = engine.build_excess_return_label(factor_panel=factors, benchmark=bundle.benchmark)

    # 行业映射
    ind_map = (
        pd.read_csv(EXISTING_PANEL, usecols=["ts_code", "industry_name"])
        .drop_duplicates("ts_code")
        .set_index("ts_code")["industry_name"]
        .to_dict()
    )
    labeled["industry_name"] = labeled["ts_code"].map(ind_map).fillna("未知行业")
    n_unknown_stocks = sum(1 for c in union if c not in ind_map)
    log(f"行业未知股票数: {n_unknown_stocks}/{len(union)}")

    # 点位过滤
    labeled["trade_date"] = labeled["trade_date"].astype(str)
    labeled["mkey"] = labeled["trade_date"].str[:6]
    before = len(labeled)
    mask = [(c, m) in member_keys for c, m in zip(labeled["ts_code"], labeled["mkey"])]
    pit = labeled[pd.Series(mask, index=labeled.index)].copy()
    log(f"点位过滤: {before} -> {len(pit)} 行 ({len(pit)/before:.1%})")
    log(f"过滤后股票数: {pit['ts_code'].nunique()}, 每月均成分数≈ {len(pit)/pit['trade_date'].str[:6].nunique()/21:.0f}")

    # benchmark_weight：float_mv 代理 = close*vol/turn，按交易日归一化
    close = pd.to_numeric(pit["close"], errors="coerce")
    vol = pd.to_numeric(pit["vol"], errors="coerce")
    turn = pd.to_numeric(pit["turnover_rate"], errors="coerce")
    proxy = close * vol / turn.where(turn > 0)
    proxy = proxy.where(np.isfinite(proxy) & (proxy > 0))
    pit["_proxy"] = proxy.values
    n_missing = int(pit["_proxy"].isna().sum())
    log(f"float_mv 代理缺失(成分-日): {n_missing}/{len(pit)} = {n_missing/len(pit):.2%}")

    # 当日缺失用当日中位数兜底，再归一化
    med = pit.groupby("trade_date")["_proxy"].transform("median")
    pit["_proxy"] = pit["_proxy"].fillna(med)
    # 若整日全缺失（中位数仍 NaN），用 1 兜底（等权）
    pit["_proxy"] = pit["_proxy"].fillna(1.0)
    date_sum = pit.groupby("trade_date")["_proxy"].transform("sum")
    pit["benchmark_weight"] = pit["_proxy"] / date_sum
    pit = pit.drop(columns=["_proxy", "mkey"])

    # 标准化（行业中性）
    log("截面标准化（winsorize + 行业中性 zscore）……")
    model_panel = engine.prepare_model_panel(pit)

    out_path = service.save_frame(model_panel, OUT_BASE)
    log(f"基线点位面板已写出: {out_path}  shape={model_panel.shape}")
    log(f"列: {list(model_panel.columns)}")

    # 记录关键统计
    stats = {
        "union": len(union),
        "rows": len(model_panel),
        "stocks": model_panel["ts_code"].nunique(),
        "months": membership["month"].nunique(),
        "industry_unknown_stocks": n_unknown_stocks,
        "floatmv_missing_frac": round(n_missing / len(pit), 4),
    }
    log(f"统计: {stats}")
    pd.DataFrame([stats]).to_csv(ROOT / "logs" / "pit_panel_stats.csv", index=False)


if __name__ == "__main__":
    main()
