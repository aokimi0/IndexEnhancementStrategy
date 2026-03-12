"""运行 LightGBM 增强实验。"""

from __future__ import annotations

import argparse

import pandas as pd

from src.backtest import BaselineBacktestEngine
from src.backtest.metrics import compute_performance_metrics
from src.config import ProjectConfig
from src.factors import FactorEngine
from src.models import LightgbmAlphaModel
from src.portfolio import OptimizationConfig
from src.utils.console import configure_console_output


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
    parser.add_argument("--train-months", type=int, default=12, help="滚动训练窗口月数")
    parser.add_argument("--min-train-rows", type=int, default=1500, help="最小训练样本数")
    parser.add_argument("--freeze-train-end-date", help="冻结训练模式下的训练集截止日，格式 YYYYMMDD")
    parser.add_argument("--test-start-date", help="测试区间开始日，格式 YYYYMMDD")
    parser.add_argument("--test-end-date", help="测试区间结束日，格式 YYYYMMDD")
    parser.add_argument(
        "--use-external-features",
        action="store_true",
        help="是否将北向资金和 M2 同比加入特征集",
    )
    parser.add_argument(
        "--use-optimizer",
        action="store_true",
        help="是否启用带约束的组合优化",
    )
    parser.add_argument("--max-tracking-error", type=float, default=0.08, help="年化跟踪误差上限")
    parser.add_argument("--max-industry-deviation", type=float, default=0.02, help="行业相对基准偏离上限")
    parser.add_argument("--max-weight", type=float, default=0.05, help="单只个股权重上限")
    parser.add_argument("--max-turnover", type=float, default=0.20, help="月度单边换手率上限")
    parser.add_argument("--fee-rate", type=float, default=0.001, help="单边手续费率")
    parser.add_argument("--slippage-rate", type=float, default=0.001, help="单边滑点率")
    return parser.parse_args()


def main() -> None:
    """执行 LightGBM 增强实验。"""
    configure_console_output()
    args = parse_args()
    config = ProjectConfig.from_root()
    config.ensure_directories()

    panel_path = config.data_dir / args.input
    factor_panel = pd.read_csv(panel_path)
    factor_panel["trade_date"] = factor_panel["trade_date"].astype(str)
    feature_columns = FactorEngine.default_factor_columns()
    if args.use_external_features:
        feature_columns = feature_columns + ["northbound_net_inflow", "m2_yoy"]
    feature_columns = [column for column in feature_columns if column in factor_panel.columns]

    model = LightgbmAlphaModel(
        feature_columns=feature_columns,
        train_months=args.train_months,
        min_train_rows=args.min_train_rows,
    )
    if args.freeze_train_end_date:
        prediction_result = model.fit_predict_frozen(
            panel=factor_panel,
            train_end_date=args.freeze_train_end_date,
            test_start_date=args.test_start_date or args.freeze_train_end_date,
            test_end_date=args.test_end_date,
        )
    else:
        prediction_result = model.fit_predict(factor_panel)

    merged_panel = factor_panel.merge(
        prediction_result.prediction_frame[["trade_date", "ts_code", "ml_score"]],
        on=["trade_date", "ts_code"],
        how="left",
    )
    merged_panel["score"] = merged_panel["ml_score"]

    optimization_config = OptimizationConfig(
        max_tracking_error=args.max_tracking_error,
        max_industry_deviation=args.max_industry_deviation,
        max_weight=args.max_weight,
        max_turnover=args.max_turnover,
    )
    backtest_engine = BaselineBacktestEngine(
        top_n=args.top_n,
        use_optimizer=args.use_optimizer,
        optimization_config=optimization_config,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    backtest_result = backtest_engine.run(merged_panel)

    if args.test_start_date:
        backtest_result = _slice_backtest_result(
            result=backtest_result,
            test_start_date=args.test_start_date,
            test_end_date=args.test_end_date,
        )

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


def _slice_backtest_result(
    result,
    test_start_date: str,
    test_end_date: str | None,
):
    """截取测试区间并重算指标。"""
    nav_frame = result.nav_frame.copy()
    nav_frame["trade_date"] = pd.to_datetime(nav_frame["trade_date"])
    start_ts = pd.to_datetime(test_start_date, format="%Y%m%d")
    end_ts = (
        pd.to_datetime(test_end_date, format="%Y%m%d")
        if test_end_date
        else nav_frame["trade_date"].max()
    )
    nav_frame = nav_frame[
        (nav_frame["trade_date"] >= start_ts) & (nav_frame["trade_date"] <= end_ts)
    ].copy()
    if not nav_frame.empty:
        nav_frame["portfolio_nav"] = nav_frame["portfolio_nav"] / nav_frame["portfolio_nav"].iloc[0]
        nav_frame["benchmark_nav"] = nav_frame["benchmark_nav"] / nav_frame["benchmark_nav"].iloc[0]
        nav_frame["excess_nav"] = nav_frame["portfolio_nav"] / nav_frame["benchmark_nav"]
        nav_frame["trade_date"] = nav_frame["trade_date"].dt.strftime("%Y-%m-%d")

    positions = result.positions.copy()
    if not positions.empty and "rebalance_date" in positions.columns:
        positions["rebalance_date"] = pd.to_datetime(positions["rebalance_date"])
        positions = positions[
            (positions["rebalance_date"] >= start_ts) & (positions["rebalance_date"] <= end_ts)
        ].copy()
        positions["rebalance_date"] = positions["rebalance_date"].dt.strftime("%Y-%m-%d")

    metrics = compute_performance_metrics(nav_frame, positions=positions)
    result.nav_frame = nav_frame
    result.positions = positions
    result.metrics = metrics
    return result


if __name__ == "__main__":
    main()
