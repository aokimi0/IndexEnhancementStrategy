"""运行多因子基线回测。"""

from __future__ import annotations

import argparse

import pandas as pd

from src.backtest import BaselineBacktestEngine
from src.config import ProjectConfig


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="运行最小多因子基线回测")
    parser.add_argument(
        "--input",
        default="processed/hs300_factor_panel_sample.csv",
        help="位于 data/ 目录下的因子面板相对路径",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="每次调仓持有股票数量",
    )
    parser.add_argument(
        "--nav-output",
        default="processed/baseline_nav_sample.csv",
        help="净值输出相对路径",
    )
    parser.add_argument(
        "--positions-output",
        default="processed/baseline_positions_sample.csv",
        help="持仓输出相对路径",
    )
    parser.add_argument(
        "--metrics-output",
        default="processed/baseline_metrics_sample.csv",
        help="指标输出相对路径",
    )
    return parser.parse_args()


def main() -> None:
    """执行基线回测流程。"""
    args = parse_args()
    config = ProjectConfig.from_root()
    config.ensure_directories()

    panel_path = config.data_dir / args.input
    factor_panel = pd.read_csv(panel_path)

    engine = BaselineBacktestEngine(top_n=args.top_n)
    result = engine.run(factor_panel)

    nav_path = config.data_dir / args.nav_output
    positions_path = config.data_dir / args.positions_output
    metrics_path = config.data_dir / args.metrics_output

    nav_path.parent.mkdir(parents=True, exist_ok=True)
    result.nav_frame.to_csv(nav_path, index=False, encoding="utf-8-sig")
    result.positions.to_csv(positions_path, index=False, encoding="utf-8-sig")
    result.metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    print(f"净值结果已生成：{nav_path}")
    print(f"持仓结果已生成：{positions_path}")
    print(f"指标结果已生成：{metrics_path}")


if __name__ == "__main__":
    main()
