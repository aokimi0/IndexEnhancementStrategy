"""比较基线与 LightGBM 策略结果。"""

from __future__ import annotations

import argparse

import pandas as pd

from src.config import ProjectConfig


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="比较策略指标")
    parser.add_argument(
        "--baseline-metrics",
        default="processed/baseline_metrics_extended_2023_2024.csv",
        help="基线指标文件相对路径",
    )
    parser.add_argument(
        "--ml-metrics",
        default="processed/lightgbm_metrics_extended_2023_2024.csv",
        help="LightGBM 指标文件相对路径",
    )
    parser.add_argument(
        "--output",
        default="processed/strategy_comparison_extended_2023_2024.csv",
        help="对比结果输出相对路径",
    )
    return parser.parse_args()


def main() -> None:
    """执行策略指标对比。"""
    args = parse_args()
    config = ProjectConfig.from_root()
    config.ensure_directories()

    baseline = pd.read_csv(config.data_dir / args.baseline_metrics)
    baseline.insert(0, "strategy", "baseline")
    ml = pd.read_csv(config.data_dir / args.ml_metrics)
    ml.insert(0, "strategy", "lightgbm")

    comparison = pd.concat([baseline, ml], ignore_index=True)
    output_path = config.data_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"策略对比结果已生成：{output_path}")


if __name__ == "__main__":
    main()
