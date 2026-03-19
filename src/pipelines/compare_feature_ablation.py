"""汇总并对比特征消融实验结果。"""

from __future__ import annotations

import argparse

import pandas as pd

from src.config import ProjectConfig
from src.utils.console import configure_console_output


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="汇总并对比特征消融实验结果")
    parser.add_argument(
        "--optimizer-summary",
        default="processed/lightgbm_feature_ablation_summary_2015_2024.csv",
        help="启用优化器版本的汇总表路径（相对 data/）",
    )
    parser.add_argument(
        "--noopt-summary",
        default="processed/lightgbm_feature_ablation_summary_2015_2024_noopt.csv",
        help="不启用优化器版本的汇总表路径（相对 data/）",
    )
    parser.add_argument(
        "--output",
        default="processed/lightgbm_feature_ablation_comparison_2015_2024.csv",
        help="合并后的对比结果输出路径（相对 data/）",
    )
    return parser.parse_args()


def main() -> None:
    """执行对比表生成。"""
    configure_console_output()
    args = parse_args()
    config = ProjectConfig.from_root()
    config.ensure_directories()

    opt_path = config.data_dir / args.optimizer_summary
    noopt_path = config.data_dir / args.noopt_summary

    opt = pd.read_csv(opt_path)
    opt.insert(0, "portfolio_construction", "constrained_optimizer")

    noopt = pd.read_csv(noopt_path)
    noopt.insert(0, "portfolio_construction", "rank_equal_weight")

    comparison = pd.concat([opt, noopt], ignore_index=True)
    output_path = config.data_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"特征消融对比表已生成：{output_path}")


if __name__ == "__main__":
    main()

