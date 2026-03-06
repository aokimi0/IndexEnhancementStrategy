"""构建研究因子面板。"""

from __future__ import annotations

import argparse

from src.config import ProjectConfig
from src.data import DataService, TushareClient
from src.factors import FactorEngine


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        argparse.Namespace: 命令行参数对象。
    """
    parser = argparse.ArgumentParser(description="构建沪深300研究因子面板")
    parser.add_argument("--start-date", required=True, help="开始日期，格式为 YYYYMMDD")
    parser.add_argument("--end-date", required=True, help="结束日期，格式为 YYYYMMDD")
    parser.add_argument(
        "--index-code",
        default="000300.SH",
        help="基准指数代码，默认使用沪深300",
    )
    parser.add_argument(
        "--output",
        default="processed/hs300_factor_panel.csv",
        help="输出到 data/ 目录下的相对路径",
    )
    return parser.parse_args()


def main() -> None:
    """执行数据读取和因子构建流程。"""
    args = parse_args()
    config = ProjectConfig.from_root()
    config.ensure_directories()

    client = TushareClient(token=config.tushare_token or "")
    data_service = DataService(client=client, config=config)
    factor_engine = FactorEngine()

    bundle = data_service.build_research_panel(
        start_date=args.start_date,
        end_date=args.end_date,
        index_code=args.index_code,
    )
    factor_panel = factor_engine.compute_factors(bundle.research_panel)
    labeled_panel = factor_engine.build_excess_return_label(
        factor_panel=factor_panel,
        benchmark=bundle.benchmark,
    )
    model_panel = factor_engine.prepare_model_panel(labeled_panel)

    output_path = data_service.save_frame(model_panel, args.output)
    print(f"因子面板已生成：{output_path}")


if __name__ == "__main__":
    main()
