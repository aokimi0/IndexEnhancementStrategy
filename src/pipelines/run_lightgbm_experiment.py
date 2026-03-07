"""运行 LightGBM 增强实验。"""

from __future__ import annotations

import argparse

import pandas as pd

from src.backtest import BaselineBacktestEngine
from src.config import ProjectConfig
from src.models import LightgbmAlphaModel


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="运行 LightGBM 增强实验")
    parser.add_argument(
        "--input",
        default="processed/hs300_factor_panel_extended_2023_2024.csv",
        help="位于 data/ 目录下的因子面板相对路径",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="每次调仓持有股票数",
    )
    parser.add_argument(
        "--prediction-output",
        default="processed/lightgbm_predictions_extended_2023_2024.csv",
        help="预测结果输出相对路径",
    )
    parser.add_argument(
        "--importance-output",
        default="processed/lightgbm_feature_importance_extended_2023_2024.csv",
        help="特征重要性输出相对路径",
    )
    parser.add_argument(
        "--nav-output",
        default="processed/lightgbm_nav_extended_2023_2024.csv",
        help="净值输出相对路径",
    )
    parser.add_argument(
        "--positions-output",
        default="processed/lightgbm_positions_extended_2023_2024.csv",
        help="持仓输出相对路径",
    )
    parser.add_argument(
        "--metrics-output",
        default="processed/lightgbm_metrics_extended_2023_2024.csv",
        help="指标输出相对路径",
    )
    return parser.parse_args()


def main() -> None:
    """执行 LightGBM 增强实验。"""
    args = parse_args()
    config = ProjectConfig.from_root()
    config.ensure_directories()

    panel_path = config.data_dir / args.input
    factor_panel = pd.read_csv(panel_path)
    factor_panel["trade_date"] = factor_panel["trade_date"].astype(str)
    feature_columns = ["ret_20", "ret_60", "volatility_20"]

    model = LightgbmAlphaModel(
        feature_columns=feature_columns,
        train_months=12,
        min_train_rows=1500,
    )
    prediction_result = model.fit_predict(factor_panel)

    merged_panel = factor_panel.merge(
        prediction_result.prediction_frame[["trade_date", "ts_code", "ml_score"]],
        on=["trade_date", "ts_code"],
        how="left",
    )
    merged_panel["score"] = merged_panel["ml_score"]

    backtest_engine = BaselineBacktestEngine(top_n=args.top_n)
    backtest_result = backtest_engine.run(merged_panel)

    prediction_path = config.data_dir / args.prediction_output
    importance_path = config.data_dir / args.importance_output
    nav_path = config.data_dir / args.nav_output
    positions_path = config.data_dir / args.positions_output
    metrics_path = config.data_dir / args.metrics_output

    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_result.prediction_frame.to_csv(
        prediction_path, index=False, encoding="utf-8-sig"
    )
    prediction_result.feature_importance.to_csv(
        importance_path, index=False, encoding="utf-8-sig"
    )
    backtest_result.nav_frame.to_csv(nav_path, index=False, encoding="utf-8-sig")
    backtest_result.positions.to_csv(positions_path, index=False, encoding="utf-8-sig")
    backtest_result.metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    print(f"预测结果已生成：{prediction_path}")
    print(f"特征重要性已生成：{importance_path}")
    print(f"净值结果已生成：{nav_path}")
    print(f"持仓结果已生成：{positions_path}")
    print(f"指标结果已生成：{metrics_path}")


if __name__ == "__main__":
    main()
