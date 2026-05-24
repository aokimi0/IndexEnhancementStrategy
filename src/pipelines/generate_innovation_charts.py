"""为 4 项创新点生成论文用 matplotlib 图表。

为论文 §4/§5 中 C1-C4 创新点以及综合策略对比生成高质量图表，
所有输出写入 ``paper/figures/``。
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "paper" / "figures"
STACK_LOG = Path("/tmp/stack3.log")


def _apply_style() -> None:
    """全局应用 seaborn + 中文字体配置。"""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["font.sans-serif"] = [
        "Heiti TC",
        "Arial Unicode MS",
        "Microsoft YaHei",
        "PingFang SC",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["savefig.bbox"] = "tight"


def _save(fig: plt.Figure, name: str) -> Path:
    """保存图像到 paper/figures。"""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# C1 Sentiment
# ---------------------------------------------------------------------------
def plot_c1_polarity_distribution() -> Path:
    """情感极性分布直方图与日频聚合时间序列。"""
    news = pd.read_csv(DATA_DIR / "news_scored_haiku.csv")
    daily = pd.read_csv(DATA_DIR / "sentiment_daily_haiku.csv")

    daily["trade_date"] = pd.to_datetime(daily["trade_date"], format="%Y%m%d")
    panel = (
        daily.groupby("trade_date")
        .agg(sentiment=("sentiment_daily", "mean"), count=("sentiment_count", "sum"))
        .reset_index()
        .sort_values("trade_date")
    )

    fig, (ax_hist, ax_ts) = plt.subplots(1, 2, figsize=(12, 4.5))

    bins = np.linspace(-1.0, 1.0, 21)
    ax_hist.hist(
        news["polarity"],
        bins=bins,
        color="#4C72B0",
        edgecolor="white",
        alpha=0.85,
    )
    ax_hist.axvline(news["polarity"].mean(), color="#C44E52", linestyle="--", linewidth=1.4,
                    label=f"均值 = {news['polarity'].mean():.2f}")
    ax_hist.axvline(0, color="grey", linewidth=0.8)
    ax_hist.set_title("Claude LLM 新闻极性分布")
    ax_hist.set_xlabel("polarity ∈ [-1, 1]")
    ax_hist.set_ylabel("新闻条数")
    ax_hist.legend(loc="upper left", frameon=True)

    color_line = "#4C72B0"
    color_bar = "#8C8C8C"
    ax_ts.bar(panel["trade_date"], panel["count"], color=color_bar, alpha=0.35,
              width=0.8, label="新闻条数（右轴）")
    ax_ts.set_ylabel("新闻条数", color=color_bar)
    ax_ts.tick_params(axis="y", labelcolor=color_bar)

    ax_sentiment = ax_ts.twinx()
    ax_sentiment.plot(panel["trade_date"], panel["sentiment"], color=color_line,
                      linewidth=1.6, marker="o", markersize=3, label="日均情感")
    ax_sentiment.axhline(0, color="grey", linestyle=":", linewidth=0.8)
    ax_sentiment.set_ylabel("日均情感（左轴方向）", color=color_line)
    ax_sentiment.tick_params(axis="y", labelcolor=color_line)
    ax_sentiment.grid(False)

    ax_ts.set_title("沪深 300 日频情感时间序列")
    ax_ts.set_xlabel("交易日")
    fig.autofmt_xdate(rotation=30)

    handles_bar, labels_bar = ax_ts.get_legend_handles_labels()
    handles_line, labels_line = ax_sentiment.get_legend_handles_labels()
    ax_sentiment.legend(handles_bar + handles_line, labels_bar + labels_line,
                        loc="upper left", frameon=True)

    fig.suptitle("C1 — LLM 中文财经舆情情感因子", fontsize=13, y=1.02)
    return _save(fig, "chart_c1_polarity_distribution.png")


def plot_c1_topic_distribution() -> Path:
    """新闻 topic 字段分布柱状图，区分正负面占比。"""
    news = pd.read_csv(DATA_DIR / "news_scored_haiku.csv")
    label_map = {
        "earnings": "财报 earnings",
        "policy": "政策 policy",
        "risk": "风险 risk",
        "macro": "宏观 macro",
        "industry": "行业 industry",
        "operation": "经营 operation",
        "other": "其它 other",
    }
    order = ["operation", "earnings", "industry", "risk", "macro", "other", "policy"]
    grouped = (
        news.assign(
            polarity_class=np.select(
                [news["polarity"] > 0.1, news["polarity"] < -0.1],
                ["正面", "负面"],
                default="中性",
            )
        )
        .groupby(["topic", "polarity_class"])
        .size()
        .unstack(fill_value=0)
        .reindex(order)
    )
    grouped = grouped[["正面", "中性", "负面"]]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    colors = {"正面": "#5BB75B", "中性": "#B0B0B0", "负面": "#D9534F"}
    bottom = np.zeros(len(grouped))
    for cls in grouped.columns:
        values = grouped[cls].to_numpy()
        ax.bar(
            [label_map[t] for t in grouped.index],
            values,
            bottom=bottom,
            color=colors[cls],
            edgecolor="white",
            label=cls,
        )
        for i, (v, b) in enumerate(zip(values, bottom)):
            if v > 0:
                ax.text(i, b + v / 2, f"{int(v)}", ha="center", va="center",
                        fontsize=8.5, color="white", fontweight="bold")
        bottom += values

    totals = grouped.sum(axis=1).to_numpy()
    for i, total in enumerate(totals):
        ax.text(i, total + max(totals) * 0.02, f"{int(total)}", ha="center",
                va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("C1 — 新闻话题分布（按 LLM 极性着色）")
    ax.set_ylabel("新闻条数")
    ax.set_xlabel("话题类别 topic")
    ax.legend(title="极性分类", frameon=True, loc="upper right")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    return _save(fig, "chart_c1_topic_distribution.png")


# ---------------------------------------------------------------------------
# C2 Stacking
# ---------------------------------------------------------------------------
def plot_c2_variant_comparison() -> Path:
    """单模型 vs Stacking 三指标分组柱状图。"""
    real = pd.read_csv(DATA_DIR / "stacking_real_partial.csv").set_index("variant")
    stack_metrics = pd.read_csv(DATA_DIR / "stacking_3model_metrics.csv").iloc[0]

    rows = []
    for var in ("lgbm", "xgb", "ridge"):
        rows.append({
            "variant": var,
            "IR": real.loc[var, "information_ratio"],
            "Sharpe": real.loc[var, "sharpe_ratio"],
            "年化超额": real.loc[var, "annual_excess_return"],
        })
    rows.append({
        "variant": "stacking",
        "IR": stack_metrics["information_ratio"],
        "Sharpe": stack_metrics["sharpe_ratio"],
        "年化超额": stack_metrics["annual_excess_return"],
    })
    df = pd.DataFrame(rows).set_index("variant")

    name_map = {
        "lgbm": "LightGBM",
        "xgb": "XGBoost",
        "ridge": "Ridge",
        "stacking": "Stacking (3-L1)",
    }
    df.index = [name_map[v] for v in df.index]

    metrics = ["IR", "Sharpe", "年化超额"]
    palette = sns.color_palette("Set2", n_colors=len(df))
    width = 0.22
    x = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i, model in enumerate(df.index):
        offset = (i - (len(df) - 1) / 2) * width
        values = df.loc[model, metrics].to_numpy()
        bars = ax.bar(x + offset, values, width, label=model, color=palette[i],
                      edgecolor="white")
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.08 * np.sign(v + 0.001),
                    f"{v:.2f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8.5)

    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("指标值")
    ax.set_title("C2 — 单模型基线与 Stacking 集成对比（2024/07 – 2025/05 真实回测）")
    ax.legend(loc="upper right", frameon=True)
    return _save(fig, "chart_c2_variant_comparison.png")


def _parse_meta_log() -> pd.DataFrame:
    """从 /tmp/stack3.log 解析元学习器权重表。"""
    if not STACK_LOG.exists():
        raise FileNotFoundError(f"未找到元学习器权重日志：{STACK_LOG}")
    text = STACK_LOG.read_text(encoding="utf-8")
    block = re.search(r"meta importance:\n(.*)", text, re.S)
    if block is None:
        raise ValueError("未找到 meta importance 段落")
    rows = []
    for line in block.group(1).splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            int(parts[1])  # trade_date
        except ValueError:
            continue
        rows.append({
            "trade_date": parts[1],
            "feature": parts[2],
            "importance": float(parts[3]),
        })
    return pd.DataFrame(rows)


def plot_c2_meta_weights() -> Path:
    """Stacking 元学习器权重的时间演化。"""
    df = _parse_meta_log()
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    pivot = df.pivot(index="trade_date", columns="feature", values="importance").sort_index()

    rename = {"l1_0_lightgbm": "LightGBM", "l1_1_xgboost": "XGBoost", "l1_2_ridge": "Ridge"}
    pivot = pivot.rename(columns=rename)
    color_map = {"LightGBM": "#4C72B0", "XGBoost": "#C44E52", "Ridge": "#55A868"}

    fig, ax = plt.subplots(figsize=(10, 5.2))
    for col in pivot.columns:
        ax.plot(pivot.index, pivot[col], marker="o", linewidth=1.6,
                markersize=4, label=col, color=color_map.get(col))
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_title("C2 — 元学习器 (Ridge) 对 L1 模型权重的滚动估计")
    ax.set_xlabel("调仓日期")
    ax.set_ylabel("元学习器权重 (importance)")
    ax.legend(title="L1 模型", frameon=True, loc="upper left")
    fig.autofmt_xdate(rotation=30)
    return _save(fig, "chart_c2_meta_weights.png")


def plot_c2_ablation() -> Path:
    """合成数据 stacking 消融：突出 GRU 的关键贡献。"""
    df = pd.read_csv(DATA_DIR / "stacking_ablation.csv")
    full = pd.read_csv(DATA_DIR / "stacking_metrics_compare.csv")
    full_row = full[full["variant"] == "stacking"].iloc[0]

    rename = {
        "lgbm": "去掉 LightGBM",
        "xgb": "去掉 XGBoost",
        "gru": "去掉 GRU",
        "ridge": "去掉 Ridge",
    }
    df["label"] = df["removed_l1"].map(rename)
    df = pd.concat([
        pd.DataFrame([{"label": "完整 4 模型", "information_ratio": full_row["information_ratio"],
                       "annual_excess_return": full_row["annual_excess_return"]}]),
        df[["label", "information_ratio", "annual_excess_return"]],
    ], ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    width = 0.36
    x = np.arange(len(df))
    bars_ir = ax.bar(x - width / 2, df["information_ratio"], width, color="#4C72B0",
                     edgecolor="white", label="IR")
    bars_er = ax.bar(x + width / 2, df["annual_excess_return"], width, color="#DD8452",
                     edgecolor="white", label="年化超额")
    for bars, values in [(bars_ir, df["information_ratio"]), (bars_er, df["annual_excess_return"])]:
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v,
                    f"{v:.2f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"])
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_title("C2 — Stacking 消融：去掉 GRU 后 IR 由 4.79 暴跌至 0.39（合成数据）")
    ax.set_ylabel("指标值")
    ax.legend(loc="upper right", frameon=True)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    return _save(fig, "chart_c2_ablation.png")


# ---------------------------------------------------------------------------
# C3 Conformal
# ---------------------------------------------------------------------------
def plot_c3_metrics_compare() -> Path:
    """4 种 conformal 方案的 IR/Sharpe/超额/回撤 对比。"""
    df = pd.read_csv(DATA_DIR / "conformal_real_2024_2025" / "metrics_compare.csv")
    name_map = {
        "baseline": "Baseline",
        "alpha_scale": "Alpha\nScale",
        "candidate_filter": "Candidate\nFilter",
        "objective_penalty": "Objective\nPenalty",
    }
    df = df.set_index("scheme").loc[list(name_map.keys())]
    df.index = [name_map[k] for k in df.index]

    metrics_show = {
        "信息比率 IR": "information_ratio",
        "夏普比率": "sharpe_ratio",
        "年化超额": "annual_excess_return",
        "最大回撤": "max_drawdown",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.ravel()
    palette = sns.color_palette("crest", n_colors=len(df))
    for ax, (label, col) in zip(axes, metrics_show.items()):
        values = df[col].to_numpy()
        bars = ax.bar(df.index, values, color=palette, edgecolor="white")
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v,
                    f"{v:.3f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9)
        ax.set_title(label)
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.tick_params(axis="x", labelrotation=0)

    fig.suptitle("C3 — Conformal Prediction 4 种集成方案对比（2024/07–2025/05 真实回测）",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    return _save(fig, "chart_c3_metrics_compare.png")


def plot_c3_coverage_timeline() -> Path:
    """按月份覆盖率折线 + 90% 理论参考线。"""
    df = pd.read_csv(DATA_DIR / "conformal_real_2024_2025" / "coverage_report.csv")
    month = df[df["scope"] == "month"].copy()
    month["month"] = pd.to_datetime(month["bucket"], format="%Y-%m")
    month = month.sort_values("month")
    overall = df[df["scope"] == "overall"].iloc[0]["coverage"]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(month["month"], month["coverage"], marker="o", linewidth=2,
            color="#4C72B0", label="月度覆盖率")
    ax.axhline(0.9, color="#C44E52", linestyle="--", linewidth=1.4,
               label="理论覆盖率 = 0.90")
    ax.axhline(overall, color="#55A868", linestyle=":", linewidth=1.4,
               label=f"整体覆盖率 = {overall:.3f}")
    for x, y in zip(month["month"], month["coverage"]):
        ax.text(x, y + 0.012, f"{y:.2f}", ha="center", fontsize=8.5,
                color="#333333")

    ax.set_ylim(0.35, 1.0)
    ax.set_title("C3 — Conformal 区间覆盖率随月份变化（α=0.10）")
    ax.set_xlabel("月份")
    ax.set_ylabel("Coverage")
    ax.legend(loc="lower right", frameon=True)
    fig.autofmt_xdate(rotation=30)
    return _save(fig, "chart_c3_coverage_timeline.png")


def plot_c3_confidence_buckets() -> Path:
    """置信度分桶 IR 柱状图，突出反向规律。"""
    df = pd.read_csv(DATA_DIR / "conformal_real_2024_2025" / "confidence_ir.csv")
    label_map = {"top_30": "高置信度\nTop 30%",
                 "mid_40": "中置信度\nMid 40%",
                 "bottom_30": "低置信度\nBottom 30%"}
    df["label"] = df["bucket"].map(label_map)
    palette = ["#C44E52", "#DDA15E", "#55A868"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    bars1 = ax1.bar(df["label"], df["ir"], color=palette, edgecolor="white")
    for bar, v in zip(bars1, df["ir"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, v,
                 f"{v:+.3f}", ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=10, fontweight="bold")
    ax1.axhline(0, color="grey", linewidth=0.8)
    ax1.set_title("置信度分桶 IR — 反向规律")
    ax1.set_ylabel("IR")

    bars2 = ax2.bar(df["label"], df["hit_rate"], color=palette, edgecolor="white")
    for bar, v in zip(bars2, df["hit_rate"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                 f"{v:.2%}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="随机基准 50%")
    ax2.set_title("置信度分桶命中率")
    ax2.set_ylabel("Hit Rate")
    ax2.set_ylim(0.35, 0.55)
    ax2.legend(loc="upper right", frameon=True)

    fig.suptitle("C3 — Mondrian Conformal 置信度分桶分析（低置信桶反而表现最好）",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    return _save(fig, "chart_c3_confidence_buckets.png")


# ---------------------------------------------------------------------------
# C4 Numba speedup
# ---------------------------------------------------------------------------
def plot_c4_speedup() -> Path:
    """Python vs Numba 不同规模运行时间 + 加速比双轴。"""
    files = [
        ("10y / 16 因子", DATA_DIR / "c4_benchmark_10y_16var.csv"),
        ("4y / 300 股票", DATA_DIR / "c4_benchmark_4y_300stocks.csv"),
        ("10y / 300 股票", DATA_DIR / "c4_benchmark_10y_300stocks.csv"),
    ]
    rows = []
    for label, path in files:
        df = pd.read_csv(path)
        df = df[df["comparison"] == "numba_vs_python"]
        py = df[df["mode"] == "python_loop"].iloc[0]
        nb = df[df["mode"] == "numba_jit"].iloc[0]
        rows.append({
            "scale": label,
            "python": py["wall_clock_seconds"],
            "numba": nb["wall_clock_seconds"],
            "speedup": nb["speedup_x"],
        })
    bench = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(bench))
    width = 0.35
    bars_py = ax.bar(x - width / 2, bench["python"], width, color="#C44E52",
                     edgecolor="white", label="Python Loop")
    bars_nb = ax.bar(x + width / 2, bench["numba"], width, color="#4C72B0",
                     edgecolor="white", label="Numba JIT")
    for bars in (bars_py, bars_nb):
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                    f"{v:.2f}s", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Wall-clock 秒数")
    ax.set_xticks(x)
    ax.set_xticklabels(bench["scale"])
    ax.set_title("C4 — Numba JIT 回测引擎在不同规模上的加速")
    ax.legend(loc="upper left", frameon=True)
    ax.set_ylim(0, bench["python"].max() * 1.25)

    ax2 = ax.twinx()
    ax2.plot(x, bench["speedup"], color="#2A9D8F", marker="D",
             markersize=9, linewidth=2.0, label="加速比 (右轴)",
             zorder=5)
    label_offset = max(bench["speedup"]) * 0.08
    for xi, s in zip(x, bench["speedup"]):
        ax2.text(xi, s + label_offset, f"{s:.1f}×",
                 ha="center", va="center",
                 color="#2A9D8F", fontweight="bold", fontsize=10.5,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                           edgecolor="#2A9D8F", linewidth=0.8, alpha=0.92),
                 zorder=6)
    ax2.set_ylabel("加速比 (×)", color="#2A9D8F")
    ax2.tick_params(axis="y", labelcolor="#2A9D8F")
    ax2.set_ylim(0, max(bench["speedup"]) * 1.55)
    ax2.grid(False)
    ax2.legend(loc="upper right", frameon=True)
    return _save(fig, "chart_c4_speedup.png")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def plot_summary_strategy_comparison() -> Path:
    """所有主要策略的 IR 横向柱状对比图。"""
    baseline = pd.read_csv(DATA_DIR / "baseline_metrics_2024_2025.csv").iloc[0]
    lgbm = pd.read_csv(DATA_DIR / "lgbm_metrics_2024_2025.csv").iloc[0]
    stack = pd.read_csv(DATA_DIR / "stacking_3model_metrics.csv").iloc[0]
    conformal = pd.read_csv(DATA_DIR / "conformal_real_2024_2025" / "metrics_compare.csv")
    conformal = conformal.set_index("scheme")

    cf_baseline = conformal.loc["baseline"]
    cf_non_baseline = conformal.drop(index="baseline")
    cf_best_name = cf_non_baseline["information_ratio"].idxmax()
    cf_best = cf_non_baseline.loc[cf_best_name]

    rows = [
        ("Baseline 等权 (基线)", baseline["information_ratio"],
         baseline["annual_excess_return"], "#8C8C8C"),
        ("LightGBM 等权", lgbm["information_ratio"],
         lgbm["annual_excess_return"], "#4C72B0"),
        ("Stacking 3-L1 (C2)", stack["information_ratio"],
         stack["annual_excess_return"], "#DD8452"),
        ("Conformal-Baseline (C3 默认)", cf_baseline["information_ratio"],
         cf_baseline["annual_excess_return"], "#937860"),
        (f"Conformal-{cf_best_name} (C3 候选)",
         cf_best["information_ratio"], cf_best["annual_excess_return"], "#55A868"),
    ]
    df = pd.DataFrame(rows, columns=["strategy", "IR", "excess", "color"])
    df = df.sort_values("IR").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.barh(df["strategy"], df["IR"], color=df["color"], edgecolor="white")
    for bar, ir, excess in zip(bars, df["IR"], df["excess"]):
        ax.text(bar.get_width() + 0.06, bar.get_y() + bar.get_height() / 2,
                f"IR = {ir:.3f}  |  年化超额 = {excess:.1%}",
                va="center", ha="left", fontsize=9.5)

    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlim(0, df["IR"].max() * 1.45)
    ax.set_xlabel("Information Ratio (IR)")
    ax.set_title("策略对比 — IR 横向柱状图（2024/07 – 2025/05）", fontsize=13)
    ax.set_ylabel("")
    return _save(fig, "chart_summary_strategy_comparison.png")


# ---------------------------------------------------------------------------
def main() -> None:
    """生成全部论文图表并打印输出路径。"""
    _apply_style()
    outputs: list[Path] = []
    generators = [
        ("C1 / polarity",  plot_c1_polarity_distribution),
        ("C1 / topic",     plot_c1_topic_distribution),
        ("C2 / variants",  plot_c2_variant_comparison),
        ("C2 / meta",      plot_c2_meta_weights),
        ("C2 / ablation",  plot_c2_ablation),
        ("C3 / metrics",   plot_c3_metrics_compare),
        ("C3 / coverage",  plot_c3_coverage_timeline),
        ("C3 / buckets",   plot_c3_confidence_buckets),
        ("C4 / speedup",   plot_c4_speedup),
        ("Summary",        plot_summary_strategy_comparison),
    ]
    for label, fn in generators:
        try:
            path = fn()
            outputs.append(path)
            print(f"[OK] {label:18s} -> {path.relative_to(ROOT)}")
        except Exception as exc:
            print(f"[FAIL] {label:18s} -> {exc!r}")
            raise

    print(f"\nGenerated {len(outputs)} charts in {FIG_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
