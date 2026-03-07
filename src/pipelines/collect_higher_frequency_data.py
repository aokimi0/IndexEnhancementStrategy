"""采集更高频数据。"""

from __future__ import annotations

import argparse

from src.config import ProjectConfig
from src.data import AkshareClient, DataService


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="采集沪深300更高频数据")
    parser.add_argument(
        "--start-datetime",
        required=True,
        help="开始时间，格式为 YYYY-MM-DD HH:MM:SS",
    )
    parser.add_argument(
        "--end-datetime",
        required=True,
        help="结束时间，格式为 YYYY-MM-DD HH:MM:SS",
    )
    parser.add_argument(
        "--period",
        default="5",
        help="分钟级周期，默认 5 分钟",
    )
    parser.add_argument(
        "--index-code",
        default="000300.SH",
        help="基准指数代码，默认沪深300",
    )
    parser.add_argument(
        "--universe-limit",
        type=int,
        default=0,
        help="限制股票池数量，默认 0 表示不限制",
    )
    parser.add_argument(
        "--stocks-output",
        default="processed/hs300_minute_5m.csv",
        help="个股分钟级数据输出到 data/ 下的相对路径",
    )
    parser.add_argument(
        "--benchmark-output",
        default="processed/hs300_benchmark_minute_5m.csv",
        help="指数分钟级数据输出到 data/ 下的相对路径",
    )
    return parser.parse_args()


def main() -> None:
    """执行高频数据采集。"""
    args = parse_args()
    config = ProjectConfig.from_root()
    config.ensure_directories()

    data_service = DataService(client=AkshareClient(), config=config)
    ts_codes = data_service.get_research_universe(
        index_code=args.index_code,
        start_date=args.start_datetime[:10].replace("-", ""),
        end_date=args.end_datetime[:10].replace("-", ""),
        universe_limit=args.universe_limit or None,
    )
    stock_minute = data_service.load_stock_minute(
        ts_codes=ts_codes,
        start_datetime=args.start_datetime,
        end_datetime=args.end_datetime,
        period=args.period,
    )
    benchmark_minute = data_service.load_benchmark_minute(
        start_datetime=args.start_datetime,
        end_datetime=args.end_datetime,
        index_code=args.index_code,
        period=args.period,
    )

    stock_path = data_service.save_frame(stock_minute, args.stocks_output)
    benchmark_path = data_service.save_frame(benchmark_minute, args.benchmark_output)
    print(f"个股分钟级数据已生成：{stock_path}")
    print(f"指数分钟级数据已生成：{benchmark_path}")


if __name__ == "__main__":
    main()
