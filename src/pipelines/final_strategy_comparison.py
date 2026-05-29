"""端到端策略对照实验（论文综合对比表）。

在 ``data/processed/hs300_panel_2024_2025_v2.csv`` 面板上，对 4 种端到端策略做统一回测：

    1. ``baseline_equal``  多因子等权打分 + 等权 Top20（无 optimizer）
    2. ``lgbm_equal``      LightGBM 单模型 + 等权 Top20（无 optimizer）
    3. ``lgbm_opt``        LightGBM + ConstrainedPortfolioOptimizer
    4. ``conformal_opt``   Split Conformal LightGBM + UncertaintyAwareOptimizer(objective_penalty, γ=0.1)

特征统一为 ``value + quality + technical + liquidity + external`` 共 15 个因子（不含 sentiment）。
测试区间 ``2024-07-01 ~ 2025-05-30``，前 6 月做训练，月度调仓、Top20、fee/slippage=0.001、
``min_train_rows=500``、``train_months=6``。

输出：

    * ``data/processed/final_strategy_comparison.csv`` —— 4 行对比指标。
    * ``data/processed/final_nav_<strategy>.csv``     —— 每种策略测试区间 NAV 序列。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest import BaselineBacktestEngine, BaselineBacktestResult
from src.backtest.metrics import compute_performance_metrics
from src.factors import FactorEngine
from src.models import LightgbmAlphaModel
from src.models.conformal_lightgbm import ConformalLightgbmModel
from src.portfolio import ConstrainedPortfolioOptimizer, OptimizationConfig
from src.portfolio.uncertainty_aware_optimizer import (
    UncertaintyAwareConfig,
    UncertaintyAwarePortfolioOptimizer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = PROJECT_ROOT / "data/processed/hs300_panel_2024_2025_v2.csv"
OUTPUT_DIR = PROJECT_ROOT / "data/processed"
METRICS_PATH = OUTPUT_DIR / "final_strategy_comparison.csv"

FEATURE_GROUPS: tuple[str, ...] = ("value", "quality", "technical", "liquidity", "external")
TRAIN_MONTHS = 6
MIN_TRAIN_ROWS = 500
TOP_N = 20
FEE_RATE = 0.001
SLIPPAGE_RATE = 0.001
TEST_START_DATE = "20240701"
TEST_END_DATE = "20250530"

METRIC_KEYS: tuple[str, ...] = (
    "annual_return",
    "benchmark_annual_return",
    "annual_excess_return",
    "annual_volatility",
    "sharpe_ratio",
    "tracking_error",
    "information_ratio",
    "max_drawdown",
    "excess_max_drawdown",
    "monthly_win_rate",
    "annual_turnover",
)


def main() -> None:
    """运行五策略对照实验并落盘对比表与 NAV 序列。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Load] 读取面板：{PANEL_PATH}")
    factor_panel = pd.read_csv(PANEL_PATH)
    factor_panel["trade_date"] = factor_panel["trade_date"].astype(str)

    feature_columns = FactorEngine.resolve_feature_columns(
        feature_groups=list(FEATURE_GROUPS),
        available_columns=factor_panel.columns.tolist(),
    )
    print(f"[Features] 共 {len(feature_columns)} 个特征：{feature_columns}")

    optimization_config = OptimizationConfig(
        max_tracking_error=0.08,
        max_industry_deviation=0.02,
        max_weight=0.05,
        max_turnover=0.20,
    )

    metrics_rows: list[dict[str, Any]] = []

    # Strategy 1: baseline_equal —— 多因子等权打分 + 等权 Top20
    metrics_rows.append(
        _run_strategy(
            name="baseline_equal",
            merged_panel=_build_baseline_equal_panel(
                panel=factor_panel,
                feature_columns=feature_columns,
            ),
            feature_columns=feature_columns,
            use_optimizer=False,
            optimization_config=None,
            optimizer_override=None,
        )
    )

    # Strategy 2 & 3: LightGBM 单模型 —— 训练一次，给等权 / 优化器两次回测使用
    print("\n[Model] 训练 LightGBM 单模型 ...")
    lgbm_start = time.perf_counter()
    lgbm_pred = LightgbmAlphaModel(
        feature_columns=feature_columns,
        train_months=TRAIN_MONTHS,
        min_train_rows=MIN_TRAIN_ROWS,
    ).fit_predict(factor_panel)
    print(f"[Model] LightGBM 完成，用时 {time.perf_counter() - lgbm_start:.1f}s")
    lgbm_panel = _merge_score(panel=factor_panel, prediction_frame=lgbm_pred.prediction_frame)

    metrics_rows.append(
        _run_strategy(
            name="lgbm_equal",
            merged_panel=lgbm_panel,
            feature_columns=feature_columns,
            use_optimizer=False,
            optimization_config=None,
            optimizer_override=None,
        )
    )
    metrics_rows.append(
        _run_strategy(
            name="lgbm_opt",
            merged_panel=lgbm_panel,
            feature_columns=feature_columns,
            use_optimizer=True,
            optimization_config=optimization_config,
            optimizer_override=None,
        )
    )

    # Strategy 4: Conformal LightGBM + UncertaintyAwareOptimizer(objective_penalty, γ=0.1)
    print("\n[Model] 训练 Conformal LightGBM ...")
    conformal_start = time.perf_counter()
    conformal_pred = ConformalLightgbmModel(
        feature_columns=feature_columns,
        train_months=TRAIN_MONTHS,
        min_train_rows=MIN_TRAIN_ROWS,
        alpha=0.1,
        calibration_ratio=0.3,
        group_column=None,
        locally_adaptive=True,
    ).fit_predict(factor_panel)
    print(f"[Model] Conformal LightGBM 完成，用时 {time.perf_counter() - conformal_start:.1f}s")
    conformal_panel = _merge_conformal_score(
        panel=factor_panel,
        prediction_frame=conformal_pred.prediction_frame,
    )
    uncertainty_optimizer = UncertaintyAwarePortfolioOptimizer(
        config=optimization_config,
        uncertainty_config=UncertaintyAwareConfig(
            weighting_scheme="objective_penalty",
            gamma=0.1,
        ),
    )
    metrics_rows.append(
        _run_strategy(
            name="conformal_opt",
            merged_panel=conformal_panel,
            feature_columns=feature_columns,
            use_optimizer=True,
            optimization_config=optimization_config,
            optimizer_override=uncertainty_optimizer,
        )
    )

    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_frame.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")
    print("\n=== 对比指标表 ===")
    print(metrics_frame.to_string(index=False))
    print(f"\n[Done] 对比指标表已保存：{METRICS_PATH}")


def _run_strategy(
    name: str,
    merged_panel: pd.DataFrame,
    feature_columns: list[str],
    use_optimizer: bool,
    optimization_config: OptimizationConfig | None,
    optimizer_override: Any,
) -> dict[str, Any]:
    """对单个策略运行回测、截取测试区间并落盘 NAV。

    Args:
        name: 策略名，用于日志与 NAV 文件命名。
        merged_panel: 已包含 ``score`` 列的面板。
        feature_columns: 特征列名列表（仅在 ``score`` 缺失时供 ``BaselineBacktestEngine`` 兜底使用）。
        use_optimizer: 是否启用带约束的组合优化。
        optimization_config: 优化器配置。
        optimizer_override: 如不为 ``None``，则覆盖默认 :class:`ConstrainedPortfolioOptimizer`。

    Returns:
        dict[str, Any]: 含 ``strategy``、各项关键指标以及 ``elapsed_seconds`` 的字典。
    """
    print(f"\n[Strategy] {name} —— 启动回测 ...")
    started_at = time.perf_counter()
    engine = BaselineBacktestEngine(
        top_n=TOP_N,
        rebalance_frequency="M",
        factor_columns=feature_columns,
        use_optimizer=use_optimizer,
        optimization_config=optimization_config,
        fee_rate=FEE_RATE,
        slippage_rate=SLIPPAGE_RATE,
        use_numba=True,
    )
    if optimizer_override is not None:
        engine.optimizer = optimizer_override
    result = engine.run(merged_panel)
    sliced = _slice_to_test_window(result=result)
    nav_path = OUTPUT_DIR / f"final_nav_{name}.csv"
    sliced.nav_frame.to_csv(nav_path, index=False, encoding="utf-8-sig")
    elapsed = time.perf_counter() - started_at
    metrics = _extract_metrics(metrics_frame=sliced.metrics)
    print(
        f"[Strategy] {name} 完成，用时 {elapsed:.1f}s | "
        f"年化超额 {metrics['annual_excess_return']:+.2%} | "
        f"Sharpe {metrics['sharpe_ratio']:.3f} | "
        f"IR {metrics['information_ratio']:.3f} | "
        f"最大回撤 {metrics['max_drawdown']:.2%} | "
        f"年化换手 {metrics['annual_turnover']:.2f} | "
        f"NAV -> {nav_path.name}"
    )
    return {"strategy": name, "elapsed_seconds": elapsed, **metrics}


def _build_baseline_equal_panel(
    panel: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """构造 baseline_equal 策略所需的面板：每日截面 z-score 等权平均得到 ``score``。

    为了与 ML 策略对齐起跑线（首次月末调仓 = 2024-07-31），将 ``2024-07-01`` 之前的
    ``score`` 置为 NaN，避免训练区间内提前调仓。

    Args:
        panel: 原始因子面板。
        feature_columns: 用于打分的因子列名列表。

    Returns:
        pd.DataFrame: 在原面板上追加 ``score`` 列的副本。
    """
    frame = panel.copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    grouper = frame["trade_date"]
    components: list[pd.Series] = []
    for column in feature_columns:
        if column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        z = _cross_sectional_zscore(series=series, groups=grouper)
        if column == "volatility_20":
            z = -z
        components.append(z)
    frame["score"] = (
        pd.concat(components, axis=1).mean(axis=1, skipna=True)
        if components
        else np.nan
    )
    frame.loc[frame["trade_date"] < TEST_START_DATE, "score"] = np.nan
    return frame


def _cross_sectional_zscore(series: pd.Series, groups: pd.Series) -> pd.Series:
    """按 ``groups`` 分组对 ``series`` 做截面 z-score；零方差或全 NaN 组返回 NaN。

    Args:
        series: 待标准化的数值序列。
        groups: 与 ``series`` 等长的分组键（如 trade_date）。

    Returns:
        pd.Series: 同长度的 z-score 序列，输入为 NaN 处保持 NaN。
    """
    grouped = series.groupby(groups)
    means = grouped.transform("mean")
    stds = grouped.transform("std").replace(0.0, np.nan)
    return (series - means) / stds


def _merge_score(
    panel: pd.DataFrame,
    prediction_frame: pd.DataFrame,
) -> pd.DataFrame:
    """把模型预测的 ``ml_score`` 合并回面板并写入 ``score`` 列。

    Args:
        panel: 原始因子面板。
        prediction_frame: 模型预测帧，至少含 ``trade_date``、``ts_code``、``ml_score``。

    Returns:
        pd.DataFrame: 合并后的面板。
    """
    merged = panel.copy()
    merged["trade_date"] = merged["trade_date"].astype(str)
    pred = prediction_frame.copy()
    if pred.empty:
        merged["score"] = np.nan
        return merged
    pred["trade_date"] = pred["trade_date"].astype(str)
    merged = merged.merge(
        pred[["trade_date", "ts_code", "ml_score"]],
        on=["trade_date", "ts_code"],
        how="left",
    )
    merged["score"] = merged["ml_score"]
    return merged


def _merge_conformal_score(
    panel: pd.DataFrame,
    prediction_frame: pd.DataFrame,
) -> pd.DataFrame:
    """合并 Conformal 预测的 ``ml_score`` 与 ``confidence`` 等列到面板。

    Args:
        panel: 原始因子面板。
        prediction_frame: ConformalLightgbmModel 输出的预测帧。

    Returns:
        pd.DataFrame: 合并后的面板，含 ``score``、``confidence``、``ci_lower``、``ci_upper`` 等列。
    """
    merged = panel.copy()
    merged["trade_date"] = merged["trade_date"].astype(str)
    pred = prediction_frame.copy()
    if pred.empty:
        merged["score"] = np.nan
        return merged
    pred["trade_date"] = pred["trade_date"].astype(str)
    merge_columns = [
        col
        for col in [
            "trade_date",
            "ts_code",
            "ml_score",
            "ci_lower",
            "ci_upper",
            "ci_half_width",
            "confidence",
        ]
        if col in pred.columns
    ]
    merged = merged.merge(pred[merge_columns], on=["trade_date", "ts_code"], how="left")
    merged["score"] = merged["ml_score"]
    return merged


def _slice_to_test_window(result: BaselineBacktestResult) -> BaselineBacktestResult:
    """截取 ``[TEST_START_DATE, TEST_END_DATE]`` 区间并重算指标，NAV 在首日归一。

    Args:
        result: 全区间回测结果。

    Returns:
        BaselineBacktestResult: 截断后的结果。
    """
    nav_frame = result.nav_frame.copy()
    nav_frame["trade_date"] = pd.to_datetime(nav_frame["trade_date"])
    start_ts = pd.to_datetime(TEST_START_DATE, format="%Y%m%d")
    end_ts = pd.to_datetime(TEST_END_DATE, format="%Y%m%d")
    nav_frame = nav_frame[
        (nav_frame["trade_date"] >= start_ts) & (nav_frame["trade_date"] <= end_ts)
    ].copy()
    if not nav_frame.empty:
        nav_frame["portfolio_nav"] = (
            nav_frame["portfolio_nav"] / nav_frame["portfolio_nav"].iloc[0]
        )
        nav_frame["benchmark_nav"] = (
            nav_frame["benchmark_nav"] / nav_frame["benchmark_nav"].iloc[0]
        )
        nav_frame["excess_nav"] = nav_frame["portfolio_nav"] / nav_frame["benchmark_nav"]
        nav_frame["trade_date"] = nav_frame["trade_date"].dt.strftime("%Y-%m-%d")

    positions = result.positions.copy()
    if not positions.empty and "rebalance_date" in positions.columns:
        positions["rebalance_date"] = pd.to_datetime(positions["rebalance_date"])
        positions = positions[
            (positions["rebalance_date"] >= start_ts)
            & (positions["rebalance_date"] <= end_ts)
        ].copy()
        positions["rebalance_date"] = positions["rebalance_date"].dt.strftime("%Y-%m-%d")

    metrics = compute_performance_metrics(nav_frame=nav_frame, positions=positions)
    return BaselineBacktestResult(nav_frame=nav_frame, positions=positions, metrics=metrics)


def _extract_metrics(metrics_frame: pd.DataFrame) -> dict[str, float]:
    """从单行 metrics DataFrame 中抽出感兴趣的关键指标。

    Args:
        metrics_frame: :func:`compute_performance_metrics` 返回的单行表。

    Returns:
        dict[str, float]: 指标名到数值的映射；缺失项以 NaN 填充。
    """
    if metrics_frame.empty:
        return {key: float("nan") for key in METRIC_KEYS}
    row = metrics_frame.iloc[0].to_dict()
    return {key: float(row.get(key, float("nan"))) for key in METRIC_KEYS}


if __name__ == "__main__":
    main()
