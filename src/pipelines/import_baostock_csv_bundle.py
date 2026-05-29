"""将离线 BaoStock CSV 包导入 ``data/cache``，供后续流水线直接读取。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import ProjectConfig
from src.data.local_baostock_csv import HybridAkshareClient, LocalBaoStockCsvStore
from src.data.loaders import DataService
from src.utils.console import configure_console_output


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="导入离线 BaoStock CSV 包到 data/cache（估值/财务/可选行情）"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="解压后的 CSV 目录，需含 constituents/daily_valuation/quarterly_financials",
    )
    parser.add_argument(
        "--start-date",
        default="20150101",
        help="研究区间开始日期 YYYYMMDD",
    )
    parser.add_argument(
        "--end-date",
        default="20241231",
        help="研究区间结束日期 YYYYMMDD",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="仅校验 CSV 完整性，不写入缓存",
    )
    return parser.parse_args()


def main() -> None:
    """校验 CSV 并将 per-stock 切片写入 DataService 缓存。"""
    configure_console_output()
    args = parse_args()
    input_dir = Path(args.input_dir)

    store = LocalBaoStockCsvStore(input_dir)
    store.load_all()
    summary = store.validate()
    print("[Import] CSV 包摘要：")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if summary["pe_ttm_non_null_ratio"] < 0.05 or summary["roe_non_null_ratio"] < 0.05:
        raise RuntimeError(
            "估值或财务字段几乎全空，请检查 CSV 是否抓全（pe_ttm/roe 非空率过低）"
        )

    if args.validate_only:
        print("[Import] --validate-only：跳过缓存写入")
        return

    config = ProjectConfig.from_root()
    config.ensure_directories()
    client = HybridAkshareClient(input_dir)
    service = DataService(client=client, config=config)
    ts_codes = store.ts_codes

    print(f"[Import] 写入 daily 缓存，股票数 {len(ts_codes)}")
    service.load_stock_daily(ts_codes, args.start_date, args.end_date)

    print(f"[Import] 写入 daily_basic_v2 缓存，股票数 {len(ts_codes)}")
    service.load_stock_daily_basic(ts_codes, args.start_date, args.end_date)

    print(f"[Import] 写入 financial_indicators_v2 缓存，股票数 {len(ts_codes)}")
    service.load_financial_indicators(ts_codes, args.start_date, args.end_date)

    print(f"[Import] 完成。缓存目录：{config.cache_dir}")


if __name__ == "__main__":
    main()
