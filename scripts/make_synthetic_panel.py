"""生成一个小规模合成因子面板用于 Stacking demo。

为了避免在 demo 中触发网络请求，本脚本完全离线地构造一份满足
:mod:`src.pipelines.run_stacking_experiment` 所需列结构的因子面板：

    * 30 只虚拟股票，覆盖 2 个交易年（约 500 个交易日）；
    * 默认核心因子分组（value/quality/technical/liquidity）共 12 列；
    * ``label_excess_return_20d`` 为部分因子的线性组合 + 噪声；
    * 同时填充 ``daily_return`` 与 ``benchmark_daily_return`` 供
      :class:`src.backtest.BaselineBacktestEngine` 使用。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ProjectConfig
from src.factors import FactorEngine


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="生成 Stacking demo 用合成面板")
    parser.add_argument(
        "--output",
        default="processed/synthetic_panel_stacking_demo.csv",
        help="位于 data/ 目录下的输出相对路径",
    )
    parser.add_argument("--n-stocks", type=int, default=30, help="虚拟股票数量")
    parser.add_argument("--n-days", type=int, default=500, help="交易日数量")
    parser.add_argument(
        "--start-date",
        default="20230101",
        help="起始日期（仅用于生成 trade_date 数字串）",
    )
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    return parser.parse_args()


def main() -> None:
    """构造并保存合成面板。"""
    args = parse_args()
    config = ProjectConfig.from_root()
    config.ensure_directories()

    rng = np.random.default_rng(args.random_state)
    feature_columns = FactorEngine.resolve_feature_columns()
    n_features = len(feature_columns)
    n_stocks = args.n_stocks
    n_days = args.n_days
    ts_codes = [f"S{idx:03d}.SH" for idx in range(n_stocks)]
    trade_dates = pd.bdate_range(start=args.start_date, periods=n_days).strftime(
        "%Y%m%d"
    )

    panel_rows: list[dict[str, float | str]] = []
    factor_storage: dict[str, np.ndarray] = {}
    for ts_code in ts_codes:
        x = rng.standard_normal((n_days, n_features)).astype(np.float32)
        factor_storage[ts_code] = x

    true_coef = rng.standard_normal(n_features).astype(np.float32) * 0.04
    benchmark_drift = rng.normal(loc=0.0003, scale=0.012, size=n_days).astype(
        np.float32
    )

    daily_returns: dict[str, np.ndarray] = {}
    for ts_code, x in factor_storage.items():
        noise = rng.normal(scale=0.018, size=n_days).astype(np.float32)
        daily = (x @ true_coef * 0.3 + benchmark_drift + noise).astype(np.float32)
        daily_returns[ts_code] = daily

    benchmark_return = np.mean(list(daily_returns.values()), axis=0)

    for d_idx, date in enumerate(trade_dates):
        for ts_code in ts_codes:
            feature_vec = factor_storage[ts_code][d_idx]
            future_window = daily_returns[ts_code][d_idx + 1 : d_idx + 21]
            bench_future = benchmark_return[d_idx + 1 : d_idx + 21]
            if len(future_window) < 20:
                label = float("nan")
            else:
                portfolio_future = float(np.prod(1.0 + future_window) - 1.0)
                bench_future_total = float(np.prod(1.0 + bench_future) - 1.0)
                label = portfolio_future - bench_future_total
            row: dict[str, float | str] = {
                "trade_date": str(date),
                "ts_code": ts_code,
                "daily_return": float(daily_returns[ts_code][d_idx]),
                "benchmark_daily_return": float(benchmark_return[d_idx]),
                "label_excess_return_20d": label,
            }
            for f_idx, fname in enumerate(feature_columns):
                row[fname] = float(feature_vec[f_idx])
            panel_rows.append(row)

    panel = pd.DataFrame(panel_rows)
    output_path: Path = config.data_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"合成面板已保存：{output_path}")
    print(
        f"shape={panel.shape}, columns={panel.columns.tolist()}, n_features={n_features}"
    )


if __name__ == "__main__":
    main()
