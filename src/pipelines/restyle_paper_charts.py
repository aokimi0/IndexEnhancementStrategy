"""以学术风格重绘论文保留的数据图表。

设计原则（对标研究生学位论文插图规范）：
    1. 不在图内嵌入大标题，标题交由 LaTeX ``\\caption`` 承载；
    2. 坐标轴标签、图例一律使用中文；
    3. 统一柔和配色与网格风格；
    4. 同时输出 png 与 pdf，便于 LaTeX 矢量插图。

当前覆盖第四章保留的 Conformal Prediction 相关图表（数据位于 ``data/processed``）。
长周期净值/回撤/特征重要性/状态分解等图依赖的中间数据已被清理，需重跑实验后再行重绘。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "paper" / "figures"

# 统一柔和配色
C_PRIMARY = "#3B6EA5"
C_ACCENT = "#C0504D"
C_GREEN = "#5B8C5A"
C_GREY = "#8C8C8C"
PALETTE4 = ["#6BAED6", "#3B6EA5", "#9C9EDE", "#31698A"]

# 策略统一中文图例与配色（净值/回撤/状态分解共用）
STRATEGY_LABELS = {
    "baseline": "基线",
    "factor_only": "LightGBM",
    "external": "LightGBM+宏观增强",
}
STRATEGY_COLORS = {
    "baseline": C_GREEN,
    "factor_only": C_PRIMARY,
    "external": C_ACCENT,
}
BENCHMARK_LABEL = "基准"
BENCHMARK_COLOR = C_GREY
LINE_WIDTH = 1.6

# 市场状态英文标签到中文的可读译名（保留区间年份信息）
REGIME_LABELS = {
    "Bull_2015": "2015 牛市",
    "Bear_2015": "2015 熊市",
    "Bull_2017": "2016-17 牛市",
    "Bear_2018": "2018 熊市",
    "Bull_2019_2021": "2019-21 牛市",
    "Bear_2021_2024": "2021-24 熊市",
}
# 状态分解图中的策略列名（来自 regime_analysis.csv 的 strategy 字段）
REGIME_STRATEGY_LABELS = {
    "Baseline": "基线",
    "LightGBM": "LightGBM",
    "LightGBM+External": "LightGBM+宏观增强",
}


def _apply_style() -> None:
    """应用学术风格的全局 matplotlib 配置。"""
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Songti SC",
                "STSong",
                "Heiti TC",
                "Arial Unicode MS",
                "PingFang SC",
                "Microsoft YaHei",
            ],
            "axes.unicode_minus": False,
            "axes.titlesize": 11,
            "axes.labelsize": 10.5,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "legend.frameon": True,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    """同时保存 png 与 pdf。"""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}")
    plt.close(fig)


def plot_metrics_compare() -> None:
    """三种置信加权方案与基线的四项指标对照（2x2）。"""
    df = pd.read_csv(DATA_DIR / "conformal_real_2024_2025" / "metrics_compare.csv")
    name_map = {
        "baseline": "基线",
        "alpha_scale": "Alpha\n缩放",
        "candidate_filter": "候选\n过滤",
        "objective_penalty": "目标\n惩罚",
    }
    df = df.set_index("scheme").loc[list(name_map.keys())]
    df.index = [name_map[k] for k in df.index]

    panels = [
        ("信息比率 IR", "information_ratio", "%.3f"),
        ("夏普比率", "sharpe_ratio", "%.3f"),
        ("年化超额收益", "annual_excess_return", "%.3f"),
        ("最大回撤", "max_drawdown", "%.3f"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4))
    for ax, (label, col, fmt) in zip(axes.ravel(), panels):
        values = df[col].to_numpy()
        bars = ax.bar(df.index, values, color=PALETTE4, edgecolor="white", width=0.62)
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v,
                fmt % v,
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=8.6,
            )
        ax.set_ylabel(label)
        ax.axhline(0, color="#444444", linewidth=0.7)
        ax.margins(y=0.18)
    fig.tight_layout()
    _save(fig, "chart_c3_metrics_compare")


def plot_coverage_timeline() -> None:
    """Conformal 区间月度覆盖率与名义参考线。"""
    df = pd.read_csv(DATA_DIR / "conformal_real_2024_2025" / "coverage_report.csv")
    month = df[df["scope"] == "month"].copy()
    month["month"] = pd.to_datetime(month["bucket"], format="%Y-%m")
    month = month.sort_values("month")
    overall = float(df[df["scope"] == "overall"].iloc[0]["coverage"])

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.plot(
        month["month"],
        month["coverage"],
        marker="o",
        markersize=5,
        linewidth=1.8,
        color=C_PRIMARY,
        label="月度实测覆盖率",
    )
    ax.axhline(0.90, color=C_ACCENT, linestyle="--", linewidth=1.4, label="名义覆盖率 0.90")
    ax.axhline(overall, color=C_GREEN, linestyle=":", linewidth=1.4, label=f"整体覆盖率 {overall:.3f}")
    for x, y in zip(month["month"], month["coverage"]):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.2)
    ax.set_ylim(0.35, 1.0)
    ax.set_xlabel("月份")
    ax.set_ylabel("区间覆盖率")
    ax.legend(loc="lower right")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    _save(fig, "chart_c3_coverage_timeline")


def plot_confidence_buckets() -> None:
    """置信度分桶的 IR 与命中率（揭示反向规律）。"""
    df = pd.read_csv(DATA_DIR / "conformal_real_2024_2025" / "confidence_ir.csv")
    label_map = {"top_30": "高置信\nTop 30%", "mid_40": "中置信\nMid 40%", "bottom_30": "低置信\nBottom 30%"}
    df = df.set_index("bucket").loc[list(label_map.keys())].reset_index()
    df["label"] = df["bucket"].map(label_map)
    colors = [C_ACCENT, "#D9A441", C_GREEN]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.4))
    bars1 = ax1.bar(df["label"], df["ir"], color=colors, edgecolor="white", width=0.6)
    for bar, v in zip(bars1, df["ir"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, v, f"{v:+.3f}", ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=9.5, fontweight="bold")
    ax1.axhline(0, color="#444444", linewidth=0.7)
    ax1.set_ylabel("信息比率 IR")
    ax1.margins(y=0.2)

    bars2 = ax2.bar(df["label"], df["hit_rate"], color=colors, edgecolor="white", width=0.6)
    for bar, v in zip(bars2, df["hit_rate"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.1%}", ha="center", va="bottom",
                 fontsize=9.5, fontweight="bold")
    ax2.axhline(0.5, color=C_GREY, linestyle="--", linewidth=1.0, label="随机基准 50%")
    ax2.set_ylabel("命中率")
    ax2.set_ylim(0.35, 0.55)
    ax2.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, "chart_c3_confidence_buckets")


def plot_alpha_sensitivity() -> None:
    """覆盖率对显著性水平的敏感性（实测 vs 名义、偏差恒定）。"""
    df = pd.read_csv(DATA_DIR / "conformal_alpha_sensitivity.csv").sort_values("alpha")
    labels = [f"α={a:g}" for a in df["alpha"]]
    x = np.arange(len(df))
    width = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.4))
    b1 = ax1.bar(x - width / 2, df["theoretical_coverage"], width, color=C_GREY,
                 edgecolor="white", label="名义覆盖率 $1-\\alpha$")
    b2 = ax1.bar(x + width / 2, df["empirical_coverage"], width, color=C_PRIMARY,
                 edgecolor="white", label="实测覆盖率")
    for bars in (b1, b2):
        for bar in bars:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{bar.get_height():.1%}", ha="center", va="bottom", fontsize=8.2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("覆盖率")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="lower left")

    dev = df["deviation_pp"].abs().to_numpy()
    bars = ax2.bar(x, dev, color=C_ACCENT, edgecolor="white", width=0.5)
    for bar, v in zip(bars, dev):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.12, f"-{v:.2f} pp", ha="center", va="bottom", fontsize=9)
    ax2.axhline(dev.mean(), color="#444444", linestyle="--", linewidth=1.0,
                label=f"平均偏差 ≈ -{dev.mean():.1f} pp")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("覆盖偏差幅度 (pp)")
    ax2.set_ylim(0, max(dev) * 1.35)
    ax2.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "chart_c3_alpha_sensitivity")


def _load_nav(name: str) -> pd.DataFrame:
    """读取净值 CSV 并统一日期格式（按 trade_date 升序）。"""
    frame = pd.read_csv(DATA_DIR / name)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame.sort_values("trade_date").reset_index(drop=True)


def plot_long_horizon_nav() -> None:
    """长周期组合净值对比（2015-2024，chart_01）。

    数据为点位(point-in-time)成分 + 20 日 purge/embargo 去前瞻的诚实口径。
    """
    baseline = _load_nav("baseline_nav_pit_2015_2024.csv")
    factor_only = _load_nav("lightgbm_nav_pit_leakfree_2015_2024.csv")
    external = _load_nav("lightgbm_external_nav_pit_leakfree_2015_2024.csv")

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    for key, frame in (("baseline", baseline), ("factor_only", factor_only), ("external", external)):
        ax.plot(
            frame["trade_date"],
            frame["portfolio_nav"],
            label=STRATEGY_LABELS[key],
            color=STRATEGY_COLORS[key],
            linewidth=LINE_WIDTH,
        )
    ax.plot(
        baseline["trade_date"],
        baseline["benchmark_nav"],
        label=BENCHMARK_LABEL,
        color=BENCHMARK_COLOR,
        linestyle="--",
        linewidth=1.3,
    )
    ax.set_xlabel("交易日")
    ax.set_ylabel("净值")
    ax.legend(loc="upper left")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    _save(fig, "chart_01_long_horizon_nav")


def plot_long_horizon_excess_nav() -> None:
    """长周期超额净值对比（chart_02）。"""
    baseline = _load_nav("baseline_nav_constrained_extended_2015_2024_v2.csv")
    factor_only = _load_nav("lightgbm_nav_constrained_extended_2015_2024_v3.csv")
    external = _load_nav("lightgbm_nav_external_2015_2024.csv")

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    for key, frame in (("baseline", baseline), ("factor_only", factor_only), ("external", external)):
        ax.plot(
            frame["trade_date"],
            frame["excess_nav"],
            label=STRATEGY_LABELS[key],
            color=STRATEGY_COLORS[key],
            linewidth=LINE_WIDTH,
        )
    ax.axhline(1.0, color=BENCHMARK_COLOR, linestyle="--", linewidth=1.2)
    ax.set_xlabel("交易日")
    ax.set_ylabel("超额净值")
    ax.legend(loc="upper left")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    _save(fig, "chart_02_long_horizon_excess_nav")


def plot_long_horizon_drawdown() -> None:
    """长周期回撤对比（chart_03，点位 + 去前瞻口径）。"""
    baseline = _load_nav("baseline_nav_pit_2015_2024.csv")
    factor_only = _load_nav("lightgbm_nav_pit_leakfree_2015_2024.csv")
    external = _load_nav("lightgbm_external_nav_pit_leakfree_2015_2024.csv")

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    for key, frame in (("baseline", baseline), ("factor_only", factor_only), ("external", external)):
        drawdown = frame["portfolio_nav"] / frame["portfolio_nav"].cummax() - 1.0
        ax.plot(
            frame["trade_date"],
            drawdown,
            label=STRATEGY_LABELS[key],
            color=STRATEGY_COLORS[key],
            linewidth=LINE_WIDTH,
        )
    benchmark_drawdown = baseline["benchmark_nav"] / baseline["benchmark_nav"].cummax() - 1.0
    ax.plot(
        baseline["trade_date"],
        benchmark_drawdown,
        label=BENCHMARK_LABEL,
        color=BENCHMARK_COLOR,
        linestyle="--",
        linewidth=1.3,
    )
    ax.set_xlabel("交易日")
    ax.set_ylabel("回撤")
    ax.legend(loc="lower left")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    _save(fig, "chart_03_long_horizon_drawdown")


def _plot_strict_oos_nav(
    baseline_name: str,
    factor_only_name: str,
    external_name: str,
    stem: str,
) -> None:
    """严格 OOS（2025）净值对比的通用绘制逻辑（chart_04 / chart_05）。"""
    start_date = pd.Timestamp("2025-01-01")
    baseline = _load_nav(baseline_name)
    factor_only = _load_nav(factor_only_name)
    external = _load_nav(external_name)
    baseline = baseline[baseline["trade_date"] >= start_date].copy()
    factor_only = factor_only[factor_only["trade_date"] >= start_date].copy()
    external = external[external["trade_date"] >= start_date].copy()

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for key, frame in (("baseline", baseline), ("factor_only", factor_only), ("external", external)):
        ax.plot(
            frame["trade_date"],
            frame["portfolio_nav"],
            label=STRATEGY_LABELS[key],
            color=STRATEGY_COLORS[key],
            linewidth=LINE_WIDTH,
        )
    ax.plot(
        baseline["trade_date"],
        baseline["benchmark_nav"],
        label=BENCHMARK_LABEL,
        color=BENCHMARK_COLOR,
        linestyle="--",
        linewidth=1.3,
    )
    ax.set_xlabel("交易日")
    ax.set_ylabel("净值")
    ax.legend(loc="upper left")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    _save(fig, stem)


def plot_strict_oos_2024train() -> None:
    """严格 OOS：近期 2024 训练 -> 2025（chart_04）。"""
    _plot_strict_oos_nav(
        baseline_name="baseline_nav_strict_oos_2025.csv",
        factor_only_name="lightgbm_nav_strict_oos_factoronly_2024train.csv",
        external_name="lightgbm_nav_strict_oos_external_2024train.csv",
        stem="chart_04_strict_oos_2025_nav_2024train",
    )


def plot_strict_oos_2015train() -> None:
    """严格 OOS：长历史 2015-2024 训练 -> 2025（chart_05）。"""
    _plot_strict_oos_nav(
        baseline_name="baseline_nav_strict_oos_2025.csv",
        factor_only_name="lightgbm_nav_strict_oos_factoronly_2015train.csv",
        external_name="lightgbm_nav_strict_oos_external_2015train.csv",
        stem="chart_05_strict_oos_2025_nav_2015train",
    )


def plot_feature_importance() -> None:
    """LightGBM 特征重要性（按分裂次数，带跨期误差棒，chart_07）。"""
    df = pd.read_csv(DATA_DIR / "lightgbm_feature_importance_external_2015_2024.csv")
    grouped = df.groupby("feature")["importance"]
    stats = pd.DataFrame({"mean": grouped.mean(), "std": grouped.std().fillna(0.0)})
    stats = stats.sort_values("mean", ascending=True)
    top = stats.tail(15)

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.barh(
        top.index,
        top["mean"],
        xerr=top["std"],
        color=C_PRIMARY,
        edgecolor="white",
        error_kw={"ecolor": C_GREY, "elinewidth": 1.0, "capsize": 3},
    )
    ax.set_xlabel("LightGBM 分裂次数（滚动训练均值）")
    ax.set_ylabel("特征")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "chart_07_feature_importance")


def plot_regime_analysis() -> None:
    """按市场状态分解的年化超额与信息比率（两子图，chart_08）。"""
    df = pd.read_csv(DATA_DIR / "regime_analysis.csv")
    regime_order = [r for r in REGIME_LABELS if r in set(df["regime"])]
    strat_order = [s for s in REGIME_STRATEGY_LABELS if s in set(df["strategy"])]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.6, 7.4))
    panels = (
        (ax1, "annual_excess_return", "年化超额收益"),
        (ax2, "information_ratio", "信息比率"),
    )
    x = np.arange(len(regime_order))
    width = 0.8 / max(len(strat_order), 1)
    for ax, col, ylabel in panels:
        pivot = df.pivot(index="regime", columns="strategy", values=col)
        for i, strat in enumerate(strat_order):
            values = [pivot.loc[r, strat] if r in pivot.index else np.nan for r in regime_order]
            ax.bar(
                x + (i - (len(strat_order) - 1) / 2) * width,
                values,
                width,
                label=REGIME_STRATEGY_LABELS[strat],
                color=PALETTE4[i % len(PALETTE4)],
                edgecolor="white",
            )
        ax.axhline(0, color="#444444", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([REGIME_LABELS[r] for r in regime_order], rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
    ax1.legend(loc="best", ncol=len(strat_order))
    fig.tight_layout()
    _save(fig, "chart_08_regime_analysis")


def _render_if_ready(plot_func, required: list[str]) -> bool:
    """当依赖 CSV 齐备时执行绘制，否则跳过并提示缺失文件。"""
    missing = [name for name in required if not (DATA_DIR / name).exists()]
    if missing:
        print(f"[跳过] {plot_func.__name__}：缺少数据 {missing}")
        return False
    plot_func()
    print(f"[完成] {plot_func.__name__}")
    return True


def main() -> None:
    """重绘全部已支持的论文图表。"""
    _apply_style()
    plot_metrics_compare()
    plot_coverage_timeline()
    plot_confidence_buckets()
    plot_alpha_sensitivity()
    print(f"已重绘 4 张 Conformal 图表 -> {FIG_DIR}")

    long_horizon_csvs = [
        "baseline_nav_constrained_extended_2015_2024_v2.csv",
        "lightgbm_nav_constrained_extended_2015_2024_v3.csv",
        "lightgbm_nav_external_2015_2024.csv",
    ]
    _render_if_ready(plot_long_horizon_nav, long_horizon_csvs)
    _render_if_ready(plot_long_horizon_excess_nav, long_horizon_csvs)
    _render_if_ready(plot_long_horizon_drawdown, long_horizon_csvs)
    _render_if_ready(
        plot_strict_oos_2024train,
        [
            "baseline_nav_strict_oos_2025.csv",
            "lightgbm_nav_strict_oos_factoronly_2024train.csv",
            "lightgbm_nav_strict_oos_external_2024train.csv",
        ],
    )
    _render_if_ready(
        plot_strict_oos_2015train,
        [
            "baseline_nav_strict_oos_2025.csv",
            "lightgbm_nav_strict_oos_factoronly_2015train.csv",
            "lightgbm_nav_strict_oos_external_2015train.csv",
        ],
    )
    _render_if_ready(
        plot_feature_importance,
        ["lightgbm_feature_importance_external_2015_2024.csv"],
    )
    _render_if_ready(plot_regime_analysis, ["regime_analysis.csv"])
    print(f"长周期/OOS/特征/状态图重绘流程结束 -> {FIG_DIR}")


if __name__ == "__main__":
    main()
