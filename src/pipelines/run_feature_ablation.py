"""运行 LightGBM 特征分组消融实验。"""

from __future__ import annotations

import argparse

import pandas as pd

from src.config import ProjectConfig
from src.factors import FactorEngine
from src.portfolio import OptimizationConfig
from src.pipelines.run_lightgbm_experiment import run_lightgbm_pipeline
from src.utils.console import configure_console_output


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="运行 LightGBM 特征分组消融实验")
    parser.add_argument(
        "--input",
        required=True,
        help="位于 data/ 目录下的因子面板相对路径",
    )
    parser.add_argument(
        "--output-dir",
        default="processed/feature_ablation",
        help="位于 data/ 目录下的消融实验输出目录",
    )
    parser.add_argument(
        "--comparison-output",
        default="processed/lightgbm_feature_ablation_summary.csv",
        help="位于 data/ 目录下的汇总对比表输出路径",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="自定义方案，格式为 方案名=分组1+分组2，例如 full=value+quality+technical+liquidity",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="每次调仓持有股票数",
    )
    parser.add_argument("--train-months", type=int, default=12, help="滚动训练窗口月数")
    parser.add_argument("--min-train-rows", type=int, default=1500, help="最小训练样本数")
    parser.add_argument("--freeze-train-end-date", help="冻结训练模式下的训练集截止日，格式 YYYYMMDD")
    parser.add_argument("--test-start-date", help="测试区间开始日，格式 YYYYMMDD")
    parser.add_argument("--test-end-date", help="测试区间结束日，格式 YYYYMMDD")
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
    """执行特征分组消融实验。"""
    configure_console_output()
    args = parse_args()
    config = ProjectConfig.from_root()
    config.ensure_directories()

    panel_path = config.data_dir / args.input
    factor_panel = pd.read_csv(panel_path)
    factor_panel["trade_date"] = factor_panel["trade_date"].astype(str)
    print(f"[Load] 已加载因子面板: {panel_path}", flush=True)
    print(f"[Load] 面板规模: {len(factor_panel)} 行", flush=True)

    output_dir = config.data_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = config.data_dir / args.comparison_output
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    optimization_config = OptimizationConfig(
        max_tracking_error=args.max_tracking_error,
        max_industry_deviation=args.max_industry_deviation,
        max_weight=args.max_weight,
        max_turnover=args.max_turnover,
    )

    variants = build_variants(
        custom_variants=args.variant,
        available_columns=factor_panel.columns.tolist(),
    )
    print(f"[Plan] 共需执行 {len(variants)} 个消融方案", flush=True)
    comparison_rows: list[pd.DataFrame] = []

    for idx, (variant_name, feature_groups) in enumerate(variants, start=1):
        feature_columns = FactorEngine.resolve_feature_columns(
            feature_groups=feature_groups,
            available_columns=factor_panel.columns.tolist(),
        )
        if not feature_columns:
            print(f"[Skip] {variant_name} 缺少可用特征，已跳过", flush=True)
            continue
        print(
            f"[Run] ({idx}/{len(variants)}) {variant_name}: 分组={','.join(feature_groups)} | "
            f"特征数={len(feature_columns)}",
            flush=True,
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
            optimization_config=optimization_config,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
        )

        save_variant_outputs(
            output_dir=output_dir,
            variant_name=variant_name,
            prediction_frame=prediction_result.prediction_frame,
            importance_frame=prediction_result.feature_importance,
            nav_frame=backtest_result.nav_frame,
            positions_frame=backtest_result.positions,
            metrics_frame=backtest_result.metrics,
        )

        metrics_frame = backtest_result.metrics.copy()
        metrics_frame.insert(0, "variant", variant_name)
        metrics_frame.insert(1, "feature_groups", ",".join(feature_groups))
        metrics_frame.insert(2, "feature_columns", ",".join(feature_columns))
        metrics_frame.insert(3, "feature_count", len(feature_columns))
        comparison_rows.append(metrics_frame)
        print(
            f"[Done] {variant_name} 完成，使用 {len(feature_columns)} 个特征: "
            f"{', '.join(feature_columns)}",
            flush=True,
        )
        _save_incremental_comparison(
            comparison_rows=comparison_rows,
            comparison_path=comparison_path,
        )

    comparison = (
        pd.concat(comparison_rows, ignore_index=True)
        if comparison_rows
        else pd.DataFrame(
            columns=[
                "variant",
                "feature_groups",
                "feature_columns",
                "feature_count",
            ]
        )
    )
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    print(f"特征消融汇总结果已生成：{comparison_path}", flush=True)


def build_variants(
    custom_variants: list[str],
    available_columns: list[str],
) -> list[tuple[str, list[str]]]:
    """构建待执行的消融方案。

    Args:
        custom_variants: 用户传入的自定义方案。
        available_columns: 面板实际可用列。

    Returns:
        list[tuple[str, list[str]]]: 方案名与分组列表。
    """
    if custom_variants:
        return [parse_variant_definition(item) for item in custom_variants]

    feature_groups = FactorEngine.feature_groups()
    variants: list[tuple[str, list[str]]] = [
        ("fundamental_only", ["value", "quality"]),
        ("technical_only", ["technical", "liquidity"]),
        ("full_factor", ["value", "quality", "technical", "liquidity"]),
    ]
    if all(column in set(available_columns) for column in feature_groups["external"]):
        variants.append(
            (
                "full_with_external",
                ["value", "quality", "technical", "liquidity", "external"],
            )
        )
    return variants


def parse_variant_definition(definition: str) -> tuple[str, list[str]]:
    """解析单条自定义消融方案定义。

    Args:
        definition: 形如 `full=value+quality+technical` 的字符串。

    Returns:
        tuple[str, list[str]]: 方案名与分组列表。

    Raises:
        ValueError: 当方案定义格式不合法时抛出异常。
    """
    if "=" not in definition:
        raise ValueError(
            "自定义方案格式错误，应为 方案名=分组1+分组2，例如 full=value+quality"
        )
    variant_name, group_text = definition.split("=", maxsplit=1)
    feature_groups = [item.strip() for item in group_text.split("+") if item.strip()]
    if not variant_name.strip() or not feature_groups:
        raise ValueError(
            "自定义方案格式错误，应为 方案名=分组1+分组2，例如 full=value+quality"
        )
    return variant_name.strip(), feature_groups


def save_variant_outputs(
    output_dir,
    variant_name: str,
    prediction_frame: pd.DataFrame,
    importance_frame: pd.DataFrame,
    nav_frame: pd.DataFrame,
    positions_frame: pd.DataFrame,
    metrics_frame: pd.DataFrame,
) -> None:
    """保存单个方案的全部实验产物。

    Args:
        output_dir: 消融实验输出目录。
        variant_name: 当前方案名。
        prediction_frame: 预测结果表。
        importance_frame: 特征重要性表。
        nav_frame: 净值表。
        positions_frame: 持仓表。
        metrics_frame: 指标表。
    """
    prediction_frame.to_csv(
        output_dir / f"predictions_{variant_name}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    importance_frame.to_csv(
        output_dir / f"importance_{variant_name}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    nav_frame.to_csv(
        output_dir / f"nav_{variant_name}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    positions_frame.to_csv(
        output_dir / f"positions_{variant_name}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics_frame.to_csv(
        output_dir / f"metrics_{variant_name}.csv",
        index=False,
        encoding="utf-8-sig",
    )


def _save_incremental_comparison(
    comparison_rows: list[pd.DataFrame],
    comparison_path,
) -> None:
    """在长任务运行中间阶段增量保存对比表。

    Args:
        comparison_rows: 已累计的指标表列表。
        comparison_path: 对比表输出路径。
    """
    if not comparison_rows:
        return
    comparison = pd.concat(comparison_rows, ignore_index=True)
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
