"""按市场状态分析策略表现。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import ProjectConfig
from src.utils.console import configure_console_output
from src.backtest.metrics import compute_performance_metrics

def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="分市场状态分析策略表现")
    parser.add_argument(
        "--output",
        default="processed/regime_analysis.csv",
        help="输出的 CSV 文件路径",
    )
    return parser.parse_args()

def analyze_regime(
    frame: pd.DataFrame,
    start_date: str,
    end_date: str,
    regime_name: str,
) -> dict:
    """计算特定区间的指标。"""
    mask = (frame["trade_date"] >= start_date) & (frame["trade_date"] <= end_date)
    sub_frame = frame[mask].copy()
    if sub_frame.empty:
        return {}
        
    # Re-normalize NAV to 1.0 at start
    sub_frame["portfolio_nav"] = sub_frame["portfolio_nav"] / sub_frame["portfolio_nav"].iloc[0]
    sub_frame["benchmark_nav"] = sub_frame["benchmark_nav"] / sub_frame["benchmark_nav"].iloc[0]
    sub_frame["excess_nav"] = sub_frame["portfolio_nav"] / sub_frame["benchmark_nav"]
    
    sub_frame["portfolio_daily_return"] = sub_frame["portfolio_nav"].pct_change().fillna(0)
    sub_frame["benchmark_daily_return"] = sub_frame["benchmark_nav"].pct_change().fillna(0)
    
    metrics_df = compute_performance_metrics(sub_frame)
    metrics = metrics_df.iloc[0].to_dict()
    metrics["regime"] = regime_name
    return metrics

def main() -> None:
    configure_console_output()
    args = parse_args()
    config = ProjectConfig.from_root()
    
    baseline_path = config.data_dir / "processed" / "baseline_nav_constrained_extended_2015_2024_v2.csv"
    lgbm_path = config.data_dir / "processed" / "lightgbm_nav_constrained_extended_2015_2024_v3.csv"
    external_path = config.data_dir / "processed" / "lightgbm_nav_external_2015_2024.csv"
    
    if not baseline_path.exists() or not lgbm_path.exists():
        print("缺少必要的净值文件，请先运行回测")
        return

    frames = {
        "Baseline": pd.read_csv(baseline_path),
        "LightGBM": pd.read_csv(lgbm_path),
        "LightGBM+External": pd.read_csv(external_path) if external_path.exists() else None
    }
    
    regimes = [
        ("Bull_2015", "2015-01-01", "2015-06-12"),
        ("Bear_2015", "2015-06-13", "2016-01-28"),
        ("Bull_2017", "2016-01-29", "2018-01-26"),
        ("Bear_2018", "2018-01-27", "2019-01-04"),
        ("Bull_2019_2021", "2019-01-05", "2021-02-18"),
        ("Bear_2021_2024", "2021-02-19", "2024-12-31"),
    ]
    
    results = []
    for strategy_name, frame in frames.items():
        if frame is None:
            continue
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        for regime_name, start_date, end_date in regimes:
            metrics = analyze_regime(
                frame,
                pd.to_datetime(start_date),
                pd.to_datetime(end_date),
                regime_name,
            )
            if metrics:
                metrics["strategy"] = strategy_name
                results.append(metrics)
                
    result_df = pd.DataFrame(results)
    columns = ["regime", "strategy", "annual_return", "annual_excess_return", "sharpe_ratio", "information_ratio", "max_drawdown"]
    result_df = result_df[columns]
    
    output_path = config.data_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"状态分析完成，结果已保存至 {output_path}")

if __name__ == "__main__":
    main()
