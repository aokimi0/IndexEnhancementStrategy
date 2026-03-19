"""运行 LightGBM 增强实验。"""

from __future__ import annotations

import argparse

import pandas as pd

from src.backtest import BaselineBacktestEngine
from src.backtest.engine import BaselineBacktestResult
from src.backtest.metrics import compute_performance_metrics
from src.config import ProjectConfig
from src.factors import FactorEngine
from src.models import LightgbmAlphaModel, LightgbmPredictionResult
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
        "--feature-groups",
        help="按分组选择特征，使用逗号分隔，可选 value,quality,technical,liquidity,leverage,external",
    )
    parser.add_argument(
        "--feature-columns",
        help="显式指定特征列，使用逗号分隔，优先级高于 feature-groups",
    )
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
    feature_columns = resolve_feature_columns(
        factor_panel=factor_panel,
        feature_groups_arg=args.feature_groups,
        feature_columns_arg=args.feature_columns,
        use_external_features=args.use_external_features,
    )
    prediction_result, backtest_result = run_lightgbm_pipeline(
        factor_panel=factor_panel,
        feature_columns=feature_columns,
        top_n=args.top_n,
        train_months=args.train_months,
        min_train_rows=args.min_train_rows,
        freeze_train_end_date=args.freeze_train_end_date,
        test_start_date=args.test_start_date,
        test_end_date=args.test_end_date,
        use_optimizer=args.use_optimizer,
        optimization_config=OptimizationConfig(
            max_tracking_error=args.max_tracking_error,
            max_industry_deviation=args.max_industry_deviation,
            max_weight=args.max_weight,
            max_turnover=args.max_turnover,
        ),
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
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


def resolve_feature_columns(
    factor_panel: pd.DataFrame,
    feature_groups_arg: str | None = None,
    feature_columns_arg: str | None = None,
    use_external_features: bool = False,
) -> list[str]:
    """解析当前实验应使用的特征列。

    Args:
        factor_panel: 原始因子面板。
        feature_groups_arg: 命令行传入的分组列表。
        feature_columns_arg: 命令行传入的显式特征列。
        use_external_features: 是否在默认核心因子上追加外部特征。

    Returns:
        list[str]: 过滤不可用列后的特征列列表。

    Raises:
        ValueError: 当没有可用特征时抛出异常。
    """
    explicit_columns = _parse_csv_argument(feature_columns_arg)
    available_columns = factor_panel.columns.tolist()
    if explicit_columns:
        feature_columns = [
            column for column in explicit_columns if column in set(available_columns)
        ]
    else:
        feature_groups = _parse_csv_argument(feature_groups_arg)
        extra_columns = (
            FactorEngine.feature_groups()["external"] if use_external_features else None
        )
        feature_columns = FactorEngine.resolve_feature_columns(
            feature_groups=feature_groups or None,
            extra_columns=extra_columns,
            available_columns=available_columns,
        )
    if not feature_columns:
        raise ValueError("当前面板中没有可用的模型特征，请检查输入面板和特征参数。")
    return feature_columns


def run_lightgbm_pipeline(
    factor_panel: pd.DataFrame,
    feature_columns: list[str],
    top_n: int,
    train_months: int,
    min_train_rows: int,
    freeze_train_end_date: str | None = None,
    test_start_date: str | None = None,
    test_end_date: str | None = None,
    use_optimizer: bool = False,
    optimization_config: OptimizationConfig | None = None,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.001,
) -> tuple[LightgbmPredictionResult, BaselineBacktestResult]:
    """运行单组特征的 LightGBM 训练与回测。

    Args:
        factor_panel: 因子面板。
        feature_columns: 本次实验使用的特征列。
        top_n: 每次调仓持有股票数。
        train_months: 滚动训练窗口月数。
        min_train_rows: 最小训练样本数。
        freeze_train_end_date: 冻结训练模式下的训练集截止日。
        test_start_date: 测试区间开始日。
        test_end_date: 测试区间结束日。
        use_optimizer: 是否启用带约束的组合优化。
        optimization_config: 组合优化参数。
        fee_rate: 单边手续费率。
        slippage_rate: 单边滑点率。

    Returns:
        tuple[LightgbmPredictionResult, BaselineBacktestResult]:
            预测结果和回测结果。
    """
    model = LightgbmAlphaModel(
        feature_columns=feature_columns,
        train_months=train_months,
        min_train_rows=min_train_rows,
    )
    if freeze_train_end_date:
        prediction_result = model.fit_predict_frozen(
            panel=factor_panel,
            train_end_date=freeze_train_end_date,
            test_start_date=test_start_date or freeze_train_end_date,
            test_end_date=test_end_date,
        )
    else:
        prediction_result = model.fit_predict(factor_panel)

    merged_panel = factor_panel.merge(
        prediction_result.prediction_frame[["trade_date", "ts_code", "ml_score"]],
        on=["trade_date", "ts_code"],
        how="left",
    )
    merged_panel["score"] = merged_panel["ml_score"]

    backtest_engine = BaselineBacktestEngine(
        top_n=top_n,
        use_optimizer=use_optimizer,
        optimization_config=optimization_config,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )
    backtest_result = backtest_engine.run(merged_panel)
    if test_start_date:
        backtest_result = _slice_backtest_result(
            result=backtest_result,
            test_start_date=test_start_date,
            test_end_date=test_end_date,
        )
    return prediction_result, backtest_result


def _parse_csv_argument(value: str | None) -> list[str]:
    """解析逗号分隔参数。"""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


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
