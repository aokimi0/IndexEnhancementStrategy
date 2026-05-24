"""运行 Stacking 集成实验：5 个 variant 对比 + 单 L1 消融。

支持以下 variant：

    * ``lgbm`` —— 基线 :class:`LightgbmAlphaModel`
    * ``xgb``  —— :class:`XgboostAlphaModel`
    * ``ridge`` —— :class:`RidgeAlphaModel`
    * ``gru``  —— :class:`GRUAlphaModel`
    * ``stacking`` —— :class:`StackingAlphaModel` 整合上述 4 个 L1

每个 variant 共享同一份 :class:`BaselineBacktestEngine` 配置；消融实验逐个剔除 L1 重新跑
Stacking，观测 IR、年化超额、夏普等指标的变化。
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from typing import Any

import pandas as pd

from src.backtest import BaselineBacktestEngine
from src.backtest.engine import BaselineBacktestResult
from src.config import ProjectConfig
from src.factors import FactorEngine
from src.models import (
    GRUAlphaModel,
    LightgbmAlphaModel,
    RidgeAlphaModel,
    StackingAlphaModel,
    XgboostAlphaModel,
)
from src.portfolio import OptimizationConfig
from src.utils.console import configure_console_output


_VARIANT_KEYS: tuple[str, ...] = ("lgbm", "xgb", "gru", "ridge", "stacking")
_L1_KEYS: tuple[str, ...] = ("lgbm", "xgb", "gru", "ridge")
_KEY_METRICS: tuple[str, ...] = (
    "annual_excess_return",
    "sharpe_ratio",
    "information_ratio",
    "max_drawdown",
    "annual_turnover",
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        argparse.Namespace: 包含输入路径、variant 列表、回测配置等的参数对象。
    """
    parser = argparse.ArgumentParser(description="运行 Stacking 集成实验")
    parser.add_argument(
        "--input",
        default="processed/hs300_factor_panel_extended_2023_2024.csv",
        help="位于 data/ 目录下的因子面板相对路径",
    )
    parser.add_argument(
        "--output-metrics",
        default="processed/stacking_metrics_compare.csv",
        help="variant 指标对比表输出相对路径",
    )
    parser.add_argument(
        "--output-ablation",
        default="processed/stacking_ablation.csv",
        help="消融实验输出相对路径",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(_VARIANT_KEYS),
        choices=_VARIANT_KEYS,
        help="需要运行的 variant，至少一个",
    )
    parser.add_argument("--top-n", type=int, default=20, help="每次调仓持有股票数")
    parser.add_argument("--train-months", type=int, default=12, help="滚动训练窗口月数")
    parser.add_argument("--min-train-rows", type=int, default=1500, help="最小训练样本数")
    parser.add_argument(
        "--meta-model",
        default="linear",
        choices=("linear", "logistic_avg", "mlp"),
        help="Stacking 元学习器类型",
    )
    parser.add_argument("--n-jobs", type=int, default=4, help="Stacking OOF 并行任务数")
    parser.add_argument(
        "--feature-groups",
        help="按分组选择特征，逗号分隔，可选 value,quality,technical,liquidity,leverage,external",
    )
    parser.add_argument(
        "--feature-columns",
        help="显式指定特征列，逗号分隔；优先级高于 feature-groups",
    )
    parser.add_argument(
        "--use-external-features",
        action="store_true",
        help="是否在默认特征上追加 external 分组",
    )
    parser.add_argument(
        "--use-optimizer",
        action="store_true",
        help="是否启用带约束的组合优化",
    )
    parser.add_argument("--max-tracking-error", type=float, default=0.08)
    parser.add_argument("--max-industry-deviation", type=float, default=0.02)
    parser.add_argument("--max-weight", type=float, default=0.05)
    parser.add_argument("--max-turnover", type=float, default=0.20)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--slippage-rate", type=float, default=0.001)
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="跳过 Stacking 消融实验；当 stacking 不在 variant 列表中时自动跳过",
    )
    return parser.parse_args()


def main() -> None:
    """执行 5-variant 对比 + 消融实验。"""
    configure_console_output()
    args = parse_args()
    config = ProjectConfig.from_root()
    config.ensure_directories()

    factor_panel = pd.read_csv(config.data_dir / args.input)
    factor_panel["trade_date"] = factor_panel["trade_date"].astype(str)
    feature_columns = _resolve_feature_columns(factor_panel, args)
    print(f"[Stacking] 使用特征列 ({len(feature_columns)})：{feature_columns}")

    optimization_config = OptimizationConfig(
        max_tracking_error=args.max_tracking_error,
        max_industry_deviation=args.max_industry_deviation,
        max_weight=args.max_weight,
        max_turnover=args.max_turnover,
    )
    variant_rows: list[dict[str, Any]] = []
    for variant in args.variants:
        print(f"\n[Variant] 开始运行 {variant}")
        start_ts = time.perf_counter()
        model_factory = _build_model_factory(
            variant=variant,
            feature_columns=feature_columns,
            args=args,
        )
        metrics = _run_variant(
            factor_panel=factor_panel,
            model_factory=model_factory,
            top_n=args.top_n,
            use_optimizer=args.use_optimizer,
            optimization_config=optimization_config,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
        )
        elapsed = time.perf_counter() - start_ts
        print(f"[Variant] {variant} 完成，用时 {elapsed:.1f}s，指标 {metrics}")
        variant_rows.append({"variant": variant, "elapsed_seconds": elapsed, **metrics})
    metrics_frame = pd.DataFrame(variant_rows)
    print("\n=== Variant 指标对比 ===")
    print(metrics_frame.to_string(index=False))
    metrics_path = config.data_dir / args.output_metrics
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"\n指标对比已保存：{metrics_path}")

    if "stacking" in args.variants and not args.skip_ablation:
        ablation_frame = _run_stacking_ablation(
            factor_panel=factor_panel,
            feature_columns=feature_columns,
            args=args,
            optimization_config=optimization_config,
        )
        print("\n=== Stacking 单 L1 消融 ===")
        print(ablation_frame.to_string(index=False))
        ablation_path = config.data_dir / args.output_ablation
        ablation_frame.to_csv(ablation_path, index=False, encoding="utf-8-sig")
        print(f"\n消融实验已保存：{ablation_path}")


def _resolve_feature_columns(
    factor_panel: pd.DataFrame,
    args: argparse.Namespace,
) -> list[str]:
    """根据命令行参数解析当前实验使用的特征列。

    Args:
        factor_panel: 已读取的因子面板。
        args: 命令行参数对象。

    Returns:
        list[str]: 过滤掉不可用列后的特征列列表。

    Raises:
        ValueError: 当解析后没有任何可用特征列时抛出。
    """
    explicit = _parse_csv_arg(args.feature_columns)
    available_columns = factor_panel.columns.tolist()
    if explicit:
        feature_columns = [c for c in explicit if c in set(available_columns)]
    else:
        groups = _parse_csv_arg(args.feature_groups)
        extra = (
            FactorEngine.feature_groups()["external"]
            if args.use_external_features
            else None
        )
        feature_columns = FactorEngine.resolve_feature_columns(
            feature_groups=groups or None,
            extra_columns=extra,
            available_columns=available_columns,
        )
    if not feature_columns:
        raise ValueError("当前面板中没有可用的模型特征，请检查输入面板和特征参数。")
    return feature_columns


def _parse_csv_arg(value: str | None) -> list[str]:
    """解析逗号分隔的命令行参数。"""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _build_model_factory(
    variant: str,
    feature_columns: list[str],
    args: argparse.Namespace,
) -> Callable[[], Any]:
    """根据 variant 构造一个无参工厂函数。

    Args:
        variant: variant 名称，必须是 ``_VARIANT_KEYS`` 之一。
        feature_columns: 特征列。
        args: 命令行参数。

    Returns:
        Callable[[], Any]: 调用即返回一个全新的模型实例。

    Raises:
        ValueError: 当传入未知 variant 时抛出。
    """
    train_months = args.train_months
    min_train_rows = args.min_train_rows

    if variant == "lgbm":
        return lambda: LightgbmAlphaModel(
            feature_columns=feature_columns,
            train_months=train_months,
            min_train_rows=min_train_rows,
        )
    if variant == "xgb":
        return lambda: XgboostAlphaModel(
            feature_columns=feature_columns,
            train_months=train_months,
            min_train_rows=min_train_rows,
        )
    if variant == "ridge":
        return lambda: RidgeAlphaModel(
            feature_columns=feature_columns,
            train_months=train_months,
            min_train_rows=min_train_rows,
        )
    if variant == "gru":
        return lambda: GRUAlphaModel(
            feature_columns=feature_columns,
            train_months=train_months,
            min_train_rows=min_train_rows,
        )
    if variant == "stacking":
        return lambda: _build_stacking(
            feature_columns=feature_columns,
            args=args,
            keep_l1=list(_L1_KEYS),
        )
    raise ValueError(f"未知 variant: {variant}")


def _build_stacking(
    feature_columns: list[str],
    args: argparse.Namespace,
    keep_l1: list[str],
) -> StackingAlphaModel:
    """构造 Stacking 模型实例。

    Args:
        feature_columns: 特征列。
        args: 命令行参数。
        keep_l1: 需要保留的 L1 关键字列表。

    Returns:
        StackingAlphaModel: 已配置好 L1 列表与元模型的 Stacking 实例。
    """
    train_months = args.train_months
    min_train_rows = args.min_train_rows
    l1_models: list[Any] = []
    for key in keep_l1:
        if key == "lgbm":
            l1_models.append(
                LightgbmAlphaModel(
                    feature_columns=feature_columns,
                    train_months=train_months,
                    min_train_rows=min_train_rows,
                )
            )
        elif key == "xgb":
            l1_models.append(
                XgboostAlphaModel(
                    feature_columns=feature_columns,
                    train_months=train_months,
                    min_train_rows=min_train_rows,
                )
            )
        elif key == "gru":
            l1_models.append(
                GRUAlphaModel(
                    feature_columns=feature_columns,
                    train_months=train_months,
                    min_train_rows=min_train_rows,
                )
            )
        elif key == "ridge":
            l1_models.append(
                RidgeAlphaModel(
                    feature_columns=feature_columns,
                    train_months=train_months,
                    min_train_rows=min_train_rows,
                )
            )
    return StackingAlphaModel(
        l1_models=l1_models,
        feature_columns=feature_columns,
        train_months=train_months,
        min_train_rows=min_train_rows,
        meta_model=args.meta_model,
        n_jobs=args.n_jobs,
    )


def _run_variant(
    factor_panel: pd.DataFrame,
    model_factory: Callable[[], Any],
    top_n: int,
    use_optimizer: bool,
    optimization_config: OptimizationConfig,
    fee_rate: float,
    slippage_rate: float,
) -> dict[str, float]:
    """对单个 variant 跑训练 + 回测，提取关键指标。

    Args:
        factor_panel: 因子面板。
        model_factory: 模型工厂函数。
        top_n: 持仓股票数。
        use_optimizer: 是否启用带约束优化。
        optimization_config: 优化配置。
        fee_rate: 单边手续费率。
        slippage_rate: 单边滑点率。

    Returns:
        dict[str, float]: ``_KEY_METRICS`` 中各指标的数值。
    """
    model = model_factory()
    result = model.fit_predict(factor_panel)
    return _backtest_predictions(
        prediction_result=result,
        factor_panel=factor_panel,
        top_n=top_n,
        use_optimizer=use_optimizer,
        optimization_config=optimization_config,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )


def _backtest_predictions(
    prediction_result: Any,
    factor_panel: pd.DataFrame,
    top_n: int,
    use_optimizer: bool,
    optimization_config: OptimizationConfig,
    fee_rate: float,
    slippage_rate: float,
) -> dict[str, float]:
    """拿到 ``PredictionResult`` 后跑回测并抽取关键指标。"""
    nan_row = {key: float("nan") for key in _KEY_METRICS}
    if prediction_result.prediction_frame.empty:
        return nan_row
    merged = factor_panel.copy()
    merged["trade_date"] = merged["trade_date"].astype(str)
    pred = prediction_result.prediction_frame.copy()
    pred["trade_date"] = pred["trade_date"].astype(str)
    merged = merged.merge(
        pred[["trade_date", "ts_code", "ml_score"]],
        on=["trade_date", "ts_code"],
        how="left",
    )
    merged["score"] = merged["ml_score"]
    engine = BaselineBacktestEngine(
        top_n=top_n,
        use_optimizer=use_optimizer,
        optimization_config=optimization_config if use_optimizer else None,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )
    backtest_result: BaselineBacktestResult = engine.run(merged)
    if backtest_result.metrics.empty:
        return nan_row
    row = backtest_result.metrics.iloc[0].to_dict()
    return {key: float(row.get(key, float("nan"))) for key in _KEY_METRICS}


def _run_stacking_ablation(
    factor_panel: pd.DataFrame,
    feature_columns: list[str],
    args: argparse.Namespace,
    optimization_config: OptimizationConfig,
) -> pd.DataFrame:
    """逐个剔除一个 L1 重新跑 Stacking，并对比指标。

    Args:
        factor_panel: 因子面板。
        feature_columns: 特征列。
        args: 命令行参数。
        optimization_config: 优化配置。

    Returns:
        pd.DataFrame: 每行对应剔除一个 L1 后的指标。
    """
    rows: list[dict[str, Any]] = []
    for skip in _L1_KEYS:
        keep = [k for k in _L1_KEYS if k != skip]
        print(f"[Ablation] 剔除 {skip}，剩余 L1：{keep}")
        start_ts = time.perf_counter()
        stacking = _build_stacking(
            feature_columns=feature_columns,
            args=args,
            keep_l1=keep,
        )
        result = stacking.fit_predict(factor_panel)
        metrics = _backtest_predictions(
            prediction_result=result,
            factor_panel=factor_panel,
            top_n=args.top_n,
            use_optimizer=args.use_optimizer,
            optimization_config=optimization_config,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
        )
        elapsed = time.perf_counter() - start_ts
        rows.append(
            {
                "removed_l1": skip,
                "remaining_l1": ",".join(keep),
                "elapsed_seconds": elapsed,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
