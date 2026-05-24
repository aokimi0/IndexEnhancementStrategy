"""Conformal Prediction 覆盖率验证工具。

读取 :class:`ConformalLightgbmModel` 输出的预测表，统计整体 / 按月份 / 按行业的实际覆盖率，
判断 ``label_excess_return_20d`` 是否落入 ``[ci_lower, ci_upper]``。
理论上整体覆盖率应接近 1 - α；本脚本用于检验 Conformal Prediction 在 A 股截面上的有效性。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import ProjectConfig
from src.utils.console import configure_console_output


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        argparse.Namespace: 已解析的参数。
    """
    parser = argparse.ArgumentParser(description="Conformal 覆盖率验证脚本")
    parser.add_argument(
        "--input",
        required=True,
        help="Conformal 预测表 CSV 路径，可为绝对路径或位于 data/ 目录下的相对路径",
    )
    parser.add_argument(
        "--panel",
        default=None,
        help="可选：因子面板路径，用于补齐 industry_name 列",
    )
    parser.add_argument(
        "--label-column",
        default="label_excess_return_20d",
        help="实际标签列名",
    )
    parser.add_argument(
        "--output",
        default="processed/conformal/coverage_report_validated.csv",
        help="覆盖率汇总输出相对路径",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="期望显著性水平，用于打印对比",
    )
    return parser.parse_args()


def _resolve_path(config: ProjectConfig, value: str) -> Path:
    """将绝对或相对路径解析为绝对路径。

    Args:
        config: 项目配置。
        value: 原始路径字符串。

    Returns:
        Path: 解析后的绝对路径。
    """
    candidate = Path(value)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    inside_data = config.data_dir / value
    if inside_data.exists():
        return inside_data
    return candidate


def compute_coverage_report(
    prediction_frame: pd.DataFrame,
    label_column: str = "label_excess_return_20d",
) -> pd.DataFrame:
    """汇总整体 / 月份 / 行业 维度的覆盖率统计。

    Args:
        prediction_frame: 含 ``ci_lower``、``ci_upper`` 与标签的预测表。
        label_column: 标签列名。

    Returns:
        pd.DataFrame: 含 ``scope`` / ``bucket`` / ``coverage`` / ``avg_half_width`` / ``n`` 列。

    Raises:
        ValueError: 当预测表缺少必要列时抛出。
    """
    required = {"trade_date", "ts_code", label_column, "ci_lower", "ci_upper"}
    missing = required.difference(prediction_frame.columns)
    if missing:
        raise ValueError(f"预测表缺少必要列：{sorted(missing)}")

    frame = prediction_frame.copy()
    frame[label_column] = pd.to_numeric(frame[label_column], errors="coerce")
    frame = frame.dropna(subset=[label_column, "ci_lower", "ci_upper"])
    if frame.empty:
        return pd.DataFrame(
            columns=["scope", "bucket", "coverage", "avg_half_width", "n"]
        )

    frame["inside_interval"] = (
        (frame[label_column] >= frame["ci_lower"])
        & (frame[label_column] <= frame["ci_upper"])
    )
    if "ci_half_width" not in frame.columns:
        frame["ci_half_width"] = (frame["ci_upper"] - frame["ci_lower"]) / 2.0

    rows: list[dict[str, object]] = []
    rows.append(
        {
            "scope": "overall",
            "bucket": "all",
            "coverage": float(frame["inside_interval"].mean()),
            "avg_half_width": float(frame["ci_half_width"].mean()),
            "n": int(len(frame)),
        }
    )

    month_frame = frame.copy()
    month_frame["month"] = (
        pd.to_datetime(month_frame["trade_date"], format="%Y%m%d").dt.to_period("M").astype(str)
    )
    for month, group in month_frame.groupby("month"):
        rows.append(
            {
                "scope": "month",
                "bucket": month,
                "coverage": float(group["inside_interval"].mean()),
                "avg_half_width": float(group["ci_half_width"].mean()),
                "n": int(len(group)),
            }
        )
    if "industry_name" in frame.columns:
        for industry, group in frame.groupby("industry_name"):
            rows.append(
                {
                    "scope": "industry",
                    "bucket": str(industry),
                    "coverage": float(group["inside_interval"].mean()),
                    "avg_half_width": float(group["ci_half_width"].mean()),
                    "n": int(len(group)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """运行覆盖率验证脚本。"""
    configure_console_output()
    args = parse_args()
    config = ProjectConfig.from_root()
    config.ensure_directories()

    prediction_path = _resolve_path(config=config, value=args.input)
    if not prediction_path.exists():
        raise FileNotFoundError(f"找不到预测表：{prediction_path}")
    prediction_frame = pd.read_csv(prediction_path)
    prediction_frame["trade_date"] = prediction_frame["trade_date"].astype(str)

    if args.panel and "industry_name" not in prediction_frame.columns:
        panel_path = _resolve_path(config=config, value=args.panel)
        panel = pd.read_csv(panel_path)
        panel["trade_date"] = panel["trade_date"].astype(str)
        if "industry_name" in panel.columns:
            prediction_frame = prediction_frame.merge(
                panel[["trade_date", "ts_code", "industry_name"]],
                on=["trade_date", "ts_code"],
                how="left",
            )

    report = compute_coverage_report(
        prediction_frame=prediction_frame,
        label_column=args.label_column,
    )

    output_path = config.data_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False, encoding="utf-8-sig")

    target = 1.0 - args.alpha
    overall = report.loc[report["scope"] == "overall"]
    overall_coverage = float(overall["coverage"].iloc[0]) if not overall.empty else float("nan")
    print(f"期望覆盖率 1 - α = {target:.3f}")
    print(f"实际整体覆盖率 = {overall_coverage:.3f}")
    print(f"覆盖率报告已写入：{output_path}")
    print(report.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
