"""Phase 1.5 + 1.6：重跑三配置回测并汇总 IR/年化超额/最大回撤。

均启用 --use-optimizer, top-n 20：
  1) 静态多因子基线
  2) LightGBM(因子)
  3) LightGBM(外部增强)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY = "/Users/aokimi/miniconda3/envs/index-enhancement/bin/python"
BASE = "processed/hs300_factor_panel_pit_2015_2024.csv"
EXT = "processed/hs300_factor_panel_external_pit_2015_2024.csv"


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    res = subprocess.run([PY, "-m", *cmd], cwd=ROOT)
    if res.returncode != 0:
        raise RuntimeError(f"命令失败: {cmd}")


def main() -> None:
    # 1) 基线
    run([
        "src.pipelines.run_baseline_backtest",
        "--input", BASE, "--top-n", "20", "--use-optimizer",
        "--nav-output", "processed/baseline_nav_pit_2015_2024.csv",
        "--positions-output", "processed/baseline_positions_pit_2015_2024.csv",
        "--metrics-output", "processed/baseline_metrics_pit_2015_2024.csv",
    ])
    # 2) LightGBM 因子
    run([
        "src.pipelines.run_lightgbm_experiment",
        "--input", BASE, "--top-n", "20", "--use-optimizer",
        "--nav-output", "processed/lightgbm_nav_pit_2015_2024.csv",
        "--positions-output", "processed/lightgbm_positions_pit_2015_2024.csv",
        "--metrics-output", "processed/lightgbm_metrics_pit_2015_2024.csv",
        "--importance-output", "processed/lightgbm_importance_pit_2015_2024.csv",
        "--prediction-output", "processed/lightgbm_predictions_pit_2015_2024.csv",
    ])
    # 3) LightGBM 外部增强
    run([
        "src.pipelines.run_lightgbm_experiment",
        "--input", EXT, "--top-n", "20", "--use-optimizer", "--use-external-features",
        "--nav-output", "processed/lightgbm_nav_external_pit_2015_2024.csv",
        "--positions-output", "processed/lightgbm_positions_external_pit_2015_2024.csv",
        "--metrics-output", "processed/lightgbm_metrics_external_pit_2015_2024.csv",
        "--importance-output", "processed/lightgbm_importance_external_pit_2015_2024.csv",
        "--prediction-output", "processed/lightgbm_predictions_external_pit_2015_2024.csv",
    ])

    # 汇总
    cols = ["information_ratio", "annual_excess_return", "max_drawdown",
            "annual_return", "tracking_error", "annual_turnover"]
    rows = []
    for name, path in [
        ("baseline", "processed/baseline_metrics_pit_2015_2024.csv"),
        ("lightgbm", "processed/lightgbm_metrics_pit_2015_2024.csv"),
        ("lightgbm_external", "processed/lightgbm_metrics_external_pit_2015_2024.csv"),
    ]:
        m = pd.read_csv(ROOT / "data" / path)
        rows.append({"config": name, **{c: round(float(m[c].iloc[0]), 4) for c in cols if c in m.columns}})
    summary = pd.DataFrame(rows)
    summary.to_csv(ROOT / "logs" / "pit_backtest_summary.csv", index=False)
    print("\n===== 三配置点位回测汇总 =====")
    print(summary.to_string(index=False))
    print("\n对照: 幸存者偏差版 IR=0.362/4.073/4.613 ; 论文 Table4-2 IR=0.237/1.685/2.449")


if __name__ == "__main__":
    main()
