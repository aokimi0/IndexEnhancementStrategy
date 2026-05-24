"""C3 创新点：Conformal Prediction 不确定性量化 + 置信加权仓位 实验流水线。

对照四种方案：

* ``baseline``：原 :class:`LightgbmAlphaModel` + :class:`ConstrainedPortfolioOptimizer`
* ``alpha_scale``：:class:`ConformalLightgbmModel` + ``UncertaintyAwareOptimizer(alpha_scale, β)``
* ``candidate_filter``：``UncertaintyAwareOptimizer(candidate_filter, top_pct)``
* ``objective_penalty``：``UncertaintyAwareOptimizer(objective_penalty, γ)``

输出文件：

1. 指标对比表（每种 scheme 一行）
2. 覆盖率验证表（按整体 / 按月份 / 按行业）
3. 按置信度分组的 IR 表（top 30% / mid 40% / bottom 30%）
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.backtest import BaselineBacktestEngine, BaselineBacktestResult
from src.backtest.metrics import TRADING_DAYS_PER_YEAR, compute_performance_metrics
from src.config import ProjectConfig
from src.models import LightgbmAlphaModel
from src.models.conformal_lightgbm import (
    ConformalLightgbmModel,
    ConformalPredictionResult,
)
from src.pipelines.run_lightgbm_experiment import resolve_feature_columns
from src.portfolio import ConstrainedPortfolioOptimizer, OptimizationConfig
from src.portfolio.uncertainty_aware_optimizer import (
    UncertaintyAwareConfig,
    UncertaintyAwarePortfolioOptimizer,
)
from src.utils.console import configure_console_output


SUPPORTED_SCHEMES = ("baseline", "alpha_scale", "candidate_filter", "objective_penalty")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        argparse.Namespace: 已解析的参数对象。
    """
    parser = argparse.ArgumentParser(description="Conformal Prediction 置信加权仓位实验")
    parser.add_argument(
        "--input",
        default="processed/hs300_factor_panel_extended_2023_2024.csv",
        help="位于 data/ 目录下的因子面板相对路径",
    )
    parser.add_argument(
        "--output-dir",
        default="processed/conformal",
        help="输出目录（位于 data/ 下）",
    )
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=list(SUPPORTED_SCHEMES),
        choices=list(SUPPORTED_SCHEMES),
        help="参与对比的方案集合",
    )
    parser.add_argument("--top-n", type=int, default=20, help="（仅 baseline 等权对照时）每次调仓持有股票数")
    parser.add_argument("--train-months", type=int, default=12, help="滚动训练窗口月数")
    parser.add_argument("--min-train-rows", type=int, default=1500, help="最小训练样本数")
    parser.add_argument("--alpha", type=float, default=0.1, help="Conformal 显著性水平")
    parser.add_argument("--calibration-ratio", type=float, default=0.3, help="校准集占比")
    parser.add_argument(
        "--group-column",
        default=None,
        help="启用 Mondrian Conformal 时的组别列名（一般为 industry_name）",
    )
    parser.add_argument(
        "--locally-adaptive",
        action="store_true",
        default=True,
        help="是否启用 Locally Adaptive Conformal（默认开启）",
    )
    parser.add_argument(
        "--no-locally-adaptive",
        action="store_false",
        dest="locally_adaptive",
        help="关闭 Locally Adaptive，回到经典 Split Conformal",
    )
    parser.add_argument("--beta", type=float, default=1.0, help="alpha_scale 方案的 β 指数")
    parser.add_argument("--top-pct", type=float, default=0.7, help="candidate_filter 保留比例")
    parser.add_argument("--gamma", type=float, default=0.1, help="objective_penalty 的强度")
    parser.add_argument("--max-tracking-error", type=float, default=0.08)
    parser.add_argument("--max-industry-deviation", type=float, default=0.02)
    parser.add_argument("--max-weight", type=float, default=0.05)
    parser.add_argument("--max-turnover", type=float, default=0.20)
    parser.add_argument("--risk-aversion", type=float, default=5.0)
    parser.add_argument("--turnover-penalty", type=float, default=0.002)
    parser.add_argument("--covariance-window", type=int, default=60)
    parser.add_argument("--covariance-ridge", type=float, default=1e-4)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--slippage-rate", type=float, default=0.001)
    parser.add_argument("--use-optimizer", action="store_true", default=True)
    parser.add_argument(
        "--feature-groups",
        help="按分组选择特征，使用逗号分隔",
    )
    parser.add_argument(
        "--feature-columns",
        help="显式指定特征列，使用逗号分隔，优先级高于 feature-groups",
    )
    parser.add_argument(
        "--use-external-features",
        action="store_true",
        help="将外部特征加入特征集",
    )
    parser.add_argument(
        "--test-start-date",
        help="测试区间起始日 YYYYMMDD（用于切片回测结果）",
    )
    parser.add_argument(
        "--test-end-date",
        help="测试区间结束日 YYYYMMDD",
    )
    return parser.parse_args()


def main() -> None:
    """执行 Conformal Prediction 对比实验。"""
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

    optimization_config = OptimizationConfig(
        max_tracking_error=args.max_tracking_error,
        max_industry_deviation=args.max_industry_deviation,
        max_weight=args.max_weight,
        max_turnover=args.max_turnover,
        risk_aversion=args.risk_aversion,
        turnover_penalty=args.turnover_penalty,
        covariance_window=args.covariance_window,
        covariance_ridge=args.covariance_ridge,
    )

    baseline_prediction_frame: pd.DataFrame | None = None
    conformal_prediction: ConformalPredictionResult | None = None

    if "baseline" in args.schemes:
        baseline_model = LightgbmAlphaModel(
            feature_columns=feature_columns,
            train_months=args.train_months,
            min_train_rows=args.min_train_rows,
        )
        baseline_prediction_frame = baseline_model.fit_predict(factor_panel).prediction_frame

    if any(scheme != "baseline" for scheme in args.schemes):
        conformal_model = ConformalLightgbmModel(
            feature_columns=feature_columns,
            train_months=args.train_months,
            min_train_rows=args.min_train_rows,
            alpha=args.alpha,
            calibration_ratio=args.calibration_ratio,
            group_column=args.group_column,
            locally_adaptive=args.locally_adaptive,
        )
        conformal_prediction = conformal_model.fit_predict(factor_panel)

    output_dir = config.data_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[pd.DataFrame] = []
    for scheme in args.schemes:
        backtest = _run_scheme_backtest(
            scheme=scheme,
            factor_panel=factor_panel,
            baseline_prediction_frame=baseline_prediction_frame,
            conformal_prediction=conformal_prediction,
            args=args,
            optimization_config=optimization_config,
        )
        if backtest is None:
            continue
        if args.test_start_date:
            backtest = _slice_backtest_result(
                result=backtest,
                test_start_date=args.test_start_date,
                test_end_date=args.test_end_date,
            )
        backtest.nav_frame.to_csv(
            output_dir / f"nav_{scheme}.csv", index=False, encoding="utf-8-sig"
        )
        backtest.positions.to_csv(
            output_dir / f"positions_{scheme}.csv", index=False, encoding="utf-8-sig"
        )
        scheme_metrics = backtest.metrics.copy()
        scheme_metrics.insert(0, "scheme", scheme)
        metrics_rows.append(scheme_metrics)

    if metrics_rows:
        metrics_table = pd.concat(metrics_rows, ignore_index=True)
        metrics_table.to_csv(
            output_dir / "metrics_compare.csv", index=False, encoding="utf-8-sig"
        )
        print(metrics_table.to_string(index=False))

    if conformal_prediction is not None and not conformal_prediction.prediction_frame.empty:
        coverage_table = compute_coverage_table(prediction_frame=conformal_prediction.prediction_frame)
        coverage_table.to_csv(
            output_dir / "coverage_report.csv", index=False, encoding="utf-8-sig"
        )
        confidence_ir_table = compute_ir_by_confidence_bucket(
            prediction_frame=conformal_prediction.prediction_frame,
            label_column="label_excess_return_20d",
        )
        confidence_ir_table.to_csv(
            output_dir / "confidence_ir.csv", index=False, encoding="utf-8-sig"
        )
        conformal_prediction.prediction_frame.to_csv(
            output_dir / "predictions_conformal.csv", index=False, encoding="utf-8-sig"
        )
        conformal_prediction.coverage_diagnostics.to_csv(
            output_dir / "calibration_diagnostics.csv", index=False, encoding="utf-8-sig"
        )
        print("\n[覆盖率验证]")
        print(coverage_table.to_string(index=False))
        print("\n[置信度分桶 IR]")
        print(confidence_ir_table.to_string(index=False))

    print(f"\n实验结果已写入：{output_dir}")


def _run_scheme_backtest(
    scheme: str,
    factor_panel: pd.DataFrame,
    baseline_prediction_frame: pd.DataFrame | None,
    conformal_prediction: ConformalPredictionResult | None,
    args: argparse.Namespace,
    optimization_config: OptimizationConfig,
) -> BaselineBacktestResult | None:
    """根据方案名跑单个变体的回测。

    Args:
        scheme: 方案名。
        factor_panel: 原始因子面板。
        baseline_prediction_frame: baseline 方案需要的预测帧。
        conformal_prediction: 其他方案需要的预测结果。
        args: 命令行参数。
        optimization_config: 组合优化参数。

    Returns:
        BaselineBacktestResult | None: 回测结果，缺数据时为 None。
    """
    if scheme == "baseline":
        if baseline_prediction_frame is None or baseline_prediction_frame.empty:
            return None
        merged = factor_panel.merge(
            baseline_prediction_frame[["trade_date", "ts_code", "ml_score"]],
            on=["trade_date", "ts_code"],
            how="left",
        )
    else:
        if conformal_prediction is None or conformal_prediction.prediction_frame.empty:
            return None
        prediction_subset = conformal_prediction.prediction_frame[
            [
                "trade_date",
                "ts_code",
                "ml_score",
                "ci_lower",
                "ci_upper",
                "ci_half_width",
                "confidence",
            ]
        ]
        merged = factor_panel.merge(
            prediction_subset,
            on=["trade_date", "ts_code"],
            how="left",
        )
    merged["score"] = merged["ml_score"]

    engine = BaselineBacktestEngine(
        top_n=args.top_n,
        use_optimizer=True,
        optimization_config=optimization_config,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    if scheme == "baseline":
        engine.optimizer = ConstrainedPortfolioOptimizer(config=optimization_config)
    else:
        engine.optimizer = UncertaintyAwarePortfolioOptimizer(
            config=optimization_config,
            uncertainty_config=UncertaintyAwareConfig(
                weighting_scheme=scheme,
                beta=args.beta,
                top_pct=args.top_pct,
                gamma=args.gamma,
            ),
        )
    return engine.run(merged)


def compute_coverage_table(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    """汇总整体、按月、按行业的实际覆盖率。

    Args:
        prediction_frame: ConformalLightgbmModel 输出的预测表。

    Returns:
        pd.DataFrame: 含 ``scope`` / ``bucket`` / ``coverage`` / ``n`` 列的长表。
    """
    frame = prediction_frame.copy()
    frame["label_excess_return_20d"] = pd.to_numeric(
        frame["label_excess_return_20d"], errors="coerce"
    )
    valid = frame.dropna(subset=["label_excess_return_20d", "ci_lower", "ci_upper"])
    if valid.empty:
        return pd.DataFrame(columns=["scope", "bucket", "coverage", "n"])

    rows: list[dict[str, object]] = []
    overall_cov = float(
        (
            (valid["label_excess_return_20d"] >= valid["ci_lower"])
            & (valid["label_excess_return_20d"] <= valid["ci_upper"])
        ).mean()
    )
    rows.append({"scope": "overall", "bucket": "all", "coverage": overall_cov, "n": len(valid)})

    valid_with_month = valid.copy()
    valid_with_month["month"] = (
        pd.to_datetime(valid_with_month["trade_date"], format="%Y%m%d").dt.to_period("M").astype(str)
    )
    for month, group in valid_with_month.groupby("month"):
        coverage = float(
            (
                (group["label_excess_return_20d"] >= group["ci_lower"])
                & (group["label_excess_return_20d"] <= group["ci_upper"])
            ).mean()
        )
        rows.append({"scope": "month", "bucket": month, "coverage": coverage, "n": len(group)})
    return pd.DataFrame(rows)


def compute_ir_by_confidence_bucket(
    prediction_frame: pd.DataFrame,
    label_column: str = "label_excess_return_20d",
    bucket_boundaries: tuple[float, float] = (0.3, 0.7),
) -> pd.DataFrame:
    """按置信度分桶计算未来超额收益的 IR 与命中率。

    Args:
        prediction_frame: Conformal 预测表。
        label_column: 标签列名。
        bucket_boundaries: 分位阈值，默认 (0.3, 0.7) 对应 bottom 30% / mid 40% / top 30%。

    Returns:
        pd.DataFrame: 含 ``bucket`` / ``n`` / ``mean_label`` / ``mean_score`` / ``ir`` / ``hit_rate`` 的表。
    """
    frame = prediction_frame.copy()
    frame[label_column] = pd.to_numeric(frame[label_column], errors="coerce")
    frame = frame.dropna(subset=[label_column, "confidence"])
    if frame.empty:
        return pd.DataFrame(columns=["bucket", "n", "mean_label", "mean_score", "ir", "hit_rate"])

    low, high = bucket_boundaries
    lower_q = frame["confidence"].quantile(low)
    upper_q = frame["confidence"].quantile(high)

    def _label_bucket(c: float) -> str:
        if c <= lower_q:
            return "bottom_30"
        if c >= upper_q:
            return "top_30"
        return "mid_40"

    frame["bucket"] = frame["confidence"].apply(_label_bucket)
    rows: list[dict[str, object]] = []
    for bucket, group in frame.groupby("bucket"):
        mean_label = float(group[label_column].mean())
        std_label = float(group[label_column].std(ddof=0))
        annualized_ir = (
            mean_label / std_label * np.sqrt(TRADING_DAYS_PER_YEAR / 20.0)
            if std_label and not np.isnan(std_label)
            else np.nan
        )
        hit_rate = float((group[label_column] > 0).mean())
        rows.append(
            {
                "bucket": bucket,
                "n": int(len(group)),
                "mean_label": mean_label,
                "mean_score": float(group["ml_score"].mean()) if "ml_score" in group.columns else np.nan,
                "ir": annualized_ir,
                "hit_rate": hit_rate,
            }
        )
    bucket_order = {"top_30": 0, "mid_40": 1, "bottom_30": 2}
    return (
        pd.DataFrame(rows)
        .assign(bucket_order=lambda df: df["bucket"].map(bucket_order))
        .sort_values("bucket_order")
        .drop(columns="bucket_order")
        .reset_index(drop=True)
    )


def _slice_backtest_result(
    result: BaselineBacktestResult,
    test_start_date: str,
    test_end_date: str | None,
) -> BaselineBacktestResult:
    """截取测试区间并重算指标。

    Args:
        result: 完整区间回测结果。
        test_start_date: 起始日 YYYYMMDD。
        test_end_date: 结束日 YYYYMMDD，None 则取最后日。

    Returns:
        BaselineBacktestResult: 截取后的结果。
    """
    nav_frame = result.nav_frame.copy()
    nav_frame["trade_date"] = pd.to_datetime(nav_frame["trade_date"])
    start_ts = pd.to_datetime(test_start_date, format="%Y%m%d")
    end_ts = (
        pd.to_datetime(test_end_date, format="%Y%m%d") if test_end_date else nav_frame["trade_date"].max()
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
