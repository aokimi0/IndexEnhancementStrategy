"""生成论文与实验分析用图表。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import ProjectConfig
from src.utils.console import configure_console_output


def main() -> None:
    """生成核心实验图表。"""
    configure_console_output()
    config = ProjectConfig.from_root()
    output_dir = config.reports_dir / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    _clear_existing_pngs(output_dir)

    factor_only_nav = _load_nav(
        config.data_dir / "processed" / "lightgbm_nav_constrained_extended_2015_2024_v3.csv"
    )
    external_nav = _load_nav(
        config.data_dir / "processed" / "lightgbm_nav_external_2015_2024.csv"
    )
    baseline_nav = _load_nav(
        config.data_dir / "processed" / "baseline_nav_constrained_extended_2015_2024_v2.csv"
    )

    strict_oos_baseline_nav = _load_nav(
        config.data_dir / "processed" / "baseline_nav_strict_oos_2025.csv"
    )
    strict_oos_factor_only_2024_nav = _load_nav(
        config.data_dir / "processed" / "lightgbm_nav_strict_oos_factoronly_2024train.csv"
    )
    strict_oos_external_2024_nav = _load_nav(
        config.data_dir / "processed" / "lightgbm_nav_strict_oos_external_2024train.csv"
    )
    strict_oos_factor_only_2015_nav = _load_nav(
        config.data_dir / "processed" / "lightgbm_nav_strict_oos_factoronly_2015train.csv"
    )
    strict_oos_external_2015_nav = _load_nav(
        config.data_dir / "processed" / "lightgbm_nav_strict_oos_external_2015train.csv"
    )

    comparison = pd.read_csv(
        config.data_dir / "processed" / "external_feature_comparison_2015_2024.csv"
    )
    strict_oos_2024 = pd.read_csv(
        config.data_dir / "processed" / "strict_oos_comparison_2024train.csv"
    )
    strict_oos_2015 = pd.read_csv(
        config.data_dir / "processed" / "strict_oos_comparison_2015train.csv"
    )

    _plot_long_horizon_nav(
        baseline_nav=baseline_nav,
        factor_only_nav=factor_only_nav,
        external_nav=external_nav,
        output_path=output_dir / "chart_01_long_horizon_nav.png",
    )
    _plot_long_horizon_excess_nav(
        baseline_nav=baseline_nav,
        factor_only_nav=factor_only_nav,
        external_nav=external_nav,
        output_path=output_dir / "chart_02_long_horizon_excess_nav.png",
    )
    _plot_long_horizon_drawdown(
        baseline_nav=baseline_nav,
        factor_only_nav=factor_only_nav,
        external_nav=external_nav,
        output_path=output_dir / "chart_03_long_horizon_drawdown.png",
    )
    _plot_strict_oos_nav(
        baseline_nav=strict_oos_baseline_nav,
        factor_only_nav=strict_oos_factor_only_2024_nav,
        external_nav=strict_oos_external_2024_nav,
        output_path=output_dir / "chart_04_strict_oos_2025_nav_2024train.png",
    )
    _plot_strict_oos_nav(
        baseline_nav=strict_oos_baseline_nav,
        factor_only_nav=strict_oos_factor_only_2015_nav,
        external_nav=strict_oos_external_2015_nav,
        output_path=output_dir / "chart_05_strict_oos_2025_nav_2015train.png",
    )
    _plot_metric_bars(
        long_horizon=comparison,
        oos_2024=strict_oos_2024,
        oos_2015=strict_oos_2015,
        output_path=output_dir / "chart_06_metric_bar_comparison.png",
    )

    print(f"图表已生成：{output_dir}")


def _load_nav(path: Path) -> pd.DataFrame:
    """读取净值文件并统一日期格式。"""
    frame = pd.read_csv(path)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame.sort_values("trade_date").reset_index(drop=True)


def _clear_existing_pngs(output_dir: Path) -> None:
    """删除历史图表文件。"""
    for file in output_dir.glob("*.png"):
        file.unlink()


def _plot_long_horizon_nav(
    baseline_nav: pd.DataFrame,
    factor_only_nav: pd.DataFrame,
    external_nav: pd.DataFrame,
    output_path: Path,
) -> None:
    """绘制十年净值对比。"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(baseline_nav["trade_date"], baseline_nav["portfolio_nav"], label="Baseline")
    ax.plot(
        factor_only_nav["trade_date"],
        factor_only_nav["portfolio_nav"],
        label="LightGBM",
    )
    ax.plot(
        external_nav["trade_date"],
        external_nav["portfolio_nav"],
        label="LightGBM+External",
    )
    ax.plot(
        baseline_nav["trade_date"],
        baseline_nav["benchmark_nav"],
        label="Benchmark",
        linestyle="--",
    )
    ax.set_title("Long Horizon NAV Comparison")
    ax.set_xlabel("Trade Date")
    ax.set_ylabel("NAV")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_long_horizon_excess_nav(
    baseline_nav: pd.DataFrame,
    factor_only_nav: pd.DataFrame,
    external_nav: pd.DataFrame,
    output_path: Path,
) -> None:
    """绘制十年超额净值对比。"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(baseline_nav["trade_date"], baseline_nav["excess_nav"], label="Baseline")
    ax.plot(
        factor_only_nav["trade_date"],
        factor_only_nav["excess_nav"],
        label="LightGBM",
    )
    ax.plot(
        external_nav["trade_date"],
        external_nav["excess_nav"],
        label="LightGBM+External",
    )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Long Horizon Excess NAV Comparison")
    ax.set_xlabel("Trade Date")
    ax.set_ylabel("Excess NAV")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_long_horizon_drawdown(
    baseline_nav: pd.DataFrame,
    factor_only_nav: pd.DataFrame,
    external_nav: pd.DataFrame,
    output_path: Path,
) -> None:
    """绘制十年回撤对比。"""
    fig, ax = plt.subplots(figsize=(12, 6))
    for frame, label in (
        (baseline_nav, "Baseline"),
        (factor_only_nav, "LightGBM"),
        (external_nav, "LightGBM+External"),
    ):
        drawdown = frame["portfolio_nav"] / frame["portfolio_nav"].cummax() - 1.0
        ax.plot(frame["trade_date"], drawdown, label=label)
    benchmark_drawdown = (
        baseline_nav["benchmark_nav"] / baseline_nav["benchmark_nav"].cummax() - 1.0
    )
    ax.plot(
        baseline_nav["trade_date"],
        benchmark_drawdown,
        label="Benchmark",
        linestyle="--",
    )
    ax.set_title("Long Horizon Drawdown Comparison")
    ax.set_xlabel("Trade Date")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_strict_oos_nav(
    baseline_nav: pd.DataFrame,
    factor_only_nav: pd.DataFrame,
    external_nav: pd.DataFrame,
    output_path: Path,
) -> None:
    """绘制严格版 2025 OOS 净值对比。"""
    start_date = pd.Timestamp("2025-01-01")
    baseline_nav = baseline_nav[baseline_nav["trade_date"] >= start_date].copy()
    factor_only_nav = factor_only_nav[factor_only_nav["trade_date"] >= start_date].copy()
    external_nav = external_nav[external_nav["trade_date"] >= start_date].copy()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(baseline_nav["trade_date"], baseline_nav["portfolio_nav"], label="Baseline")
    ax.plot(
        factor_only_nav["trade_date"],
        factor_only_nav["portfolio_nav"],
        label="LightGBM",
    )
    ax.plot(
        external_nav["trade_date"],
        external_nav["portfolio_nav"],
        label="LightGBM+External",
    )
    ax.plot(
        baseline_nav["trade_date"],
        baseline_nav["benchmark_nav"],
        label="Benchmark",
        linestyle="--",
    )
    ax.set_title("Strict OOS 2025 NAV Comparison")
    ax.set_xlabel("Trade Date")
    ax.set_ylabel("NAV")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_metric_bars(
    long_horizon: pd.DataFrame,
    oos_2024: pd.DataFrame,
    oos_2015: pd.DataFrame,
    output_path: Path,
) -> None:
    """绘制关键指标条形图。"""
    long_horizon = long_horizon.copy()
    long_horizon["group"] = "LongHorizon"
    oos_2024 = oos_2024.copy()
    oos_2024["group"] = "StrictOOS2024Train"
    oos_2015 = oos_2015.copy()
    oos_2015["group"] = "StrictOOS2015Train"
    combined = pd.concat([long_horizon, oos_2024, oos_2015], ignore_index=True)
    metrics = ["annual_excess_return", "sharpe_ratio", "information_ratio", "max_drawdown"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, metric in zip(axes, metrics, strict=False):
        pivot = combined.pivot(index="strategy", columns="group", values=metric)
        pivot.plot(kind="bar", ax=ax, rot=0)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
