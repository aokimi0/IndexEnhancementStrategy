"""绘制挑战二改进方案的对比图（学术风格，输出至 paper/figures）。

左图：各方案在 leak-free 短样本上的信息比率（朴素三方案 + 两改进方案 vs 不加权基线）。
右图：信度门控方案与不加权基线在 5 个随机种子上的信息比率稳定性。
数据来源：logs/embargo_C2_schemes.csv、logs/improved_C4_schemes.csv、logs/improved_C4_seed_stability.csv。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOGDIR = ROOT / "logs"
FIG_DIR = ROOT / "paper" / "figures"

C_NEG = "#C0504D"
C_POS = "#3B6EA5"
C_GREY = "#8C8C8C"
C_BASE = "#C0504D"
C_GATE = "#3B6EA5"


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Songti SC", "STSong", "Heiti TC", "Arial Unicode MS",
                "PingFang SC", "Microsoft YaHei",
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


def main() -> None:
    _apply_style()
    c2 = pd.read_csv(LOGDIR / "embargo_C2_schemes.csv").set_index("scheme")["information_ratio"]
    c4 = pd.read_csv(LOGDIR / "improved_C4_schemes.csv").set_index("scheme")["information_ratio"]
    seeds = pd.read_csv(LOGDIR / "improved_C4_seed_stability.csv")

    labels = ["不加权\n基线", "alpha\n缩放", "候选\n过滤", "目标\n惩罚", "信度门控\n稳健混合", "不确定性\n风险项"]
    values = [
        float(c2.get("baseline")),
        float(c2.get("alpha_scale")),
        float(c2.get("candidate_filter")),
        float(c2.get("objective_penalty")),
        float(c4.get("trust_gated")),
        float(c4.get("uncertainty_risk")),
    ]
    colors = [C_GREY] + [C_NEG if v < 0 else C_POS for v in values[1:]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.2))

    bars = ax1.bar(range(len(values)), values, color=colors, width=0.62, edgecolor="#333333", linewidth=0.6)
    ax1.axhline(0.0, color="#444444", linewidth=0.9)
    ax1.set_xticks(range(len(values)))
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("信息比率 IR")
    ax1.set_title("(a) 各方案信息比率（leak-free 短样本）")
    for bar, v in zip(bars, values):
        ax1.annotate(
            f"{v:+.2f}",
            (bar.get_x() + bar.get_width() / 2, v),
            textcoords="offset points",
            xytext=(0, 6 if v >= 0 else -12),
            ha="center",
            fontsize=8.6,
            fontweight="bold",
        )
    ax1.margins(y=0.18)

    x = np.arange(len(seeds))
    ax2.plot(x, seeds["baseline_ir"], "o-", color=C_BASE, linewidth=1.6, label="不加权基线")
    ax2.plot(x, seeds["trust_gated_ir"], "s-", color=C_GATE, linewidth=1.6, label="信度门控稳健混合")
    ax2.axhline(0.0, color="#444444", linewidth=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(s) for s in seeds["seed"]], fontsize=9)
    ax2.set_xlabel("随机种子")
    ax2.set_ylabel("信息比率 IR")
    ax2.set_title("(b) 多种子稳定性")
    ax2.legend(loc="center left")
    ax2.margins(y=0.20)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"chart_04_improved_uncertainty.{ext}")
    plt.close(fig)
    print(f"已写入 {FIG_DIR / 'chart_04_improved_uncertainty.png'}")


if __name__ == "__main__":
    main()
