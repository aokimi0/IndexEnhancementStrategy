"""探索最佳训练窗口长度的消融实验。"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd

from src.config import ProjectConfig
from src.utils.console import configure_console_output

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="探索最佳训练窗口长度")
    parser.add_argument(
        "--input",
        required=True,
        help="增强后因子面板相对路径",
    )
    return parser.parse_args()

def main() -> None:
    configure_console_output()
    args = parse_args()
    config = ProjectConfig.from_root()
    
    windows = [12, 24, 36, 60]
    results = []
    
    for window in windows:
        print(f"正在运行 {window} 个月训练窗口的回测...")
        cmd = [
            "python", "-m", "src.pipelines.run_lightgbm_experiment",
            "--input", args.input,
            "--use-optimizer",
            "--train-months", str(window),
            "--prediction-output", f"processed/window_explore/pred_{window}m.csv",
            "--importance-output", f"processed/window_explore/imp_{window}m.csv",
            "--nav-output", f"processed/window_explore/nav_{window}m.csv",
            "--positions-output", f"processed/window_explore/pos_{window}m.csv",
            "--metrics-output", f"processed/window_explore/metrics_{window}m.csv",
        ]
        try:
            subprocess.run(cmd, check=True)
            metrics_path = config.data_dir / "processed" / "window_explore" / f"metrics_{window}m.csv"
            if metrics_path.exists():
                df = pd.read_csv(metrics_path)
                metrics = df.iloc[0].to_dict()
                metrics["train_window_months"] = window
                results.append(metrics)
        except subprocess.CalledProcessError as e:
            print(f"窗口 {window} 个月运行失败: {e}")
            
    if results:
        res_df = pd.DataFrame(results)
        out_path = config.data_dir / "processed" / "window_explore" / "window_comparison.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        res_df.to_csv(out_path, index=False)
        print(f"窗口长度探索完成，结果保存在 {out_path}")

if __name__ == "__main__":
    main()
