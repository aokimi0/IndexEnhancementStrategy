"""构建研究因子面板。"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import ProjectConfig
from src.data import AkshareClient, DataService, TushareClient
from src.data.local_baostock_csv import HybridAkshareClient
from src.factors import FactorEngine
from src.utils.console import configure_console_output


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
    parser.add_argument(
        "--data-source",
        choices=["akshare", "tushare", "auto"],
        default="akshare",
        help="数据源，默认使用 akshare",
    )
    parser.add_argument(
        "--universe-limit",
        type=int,
        default=0,
        help="限制股票池数量，默认 0 表示不限制，便于先做小样本验证",
    )
    parser.add_argument(
        "--local-csv-dir",
        default="",
        help="离线 BaoStock CSV 目录（含 constituents/daily_valuation/quarterly_financials），"
        "估值与财务走本地 CSV，其余仍用 akshare",
    )
    return parser.parse_args()


def build_client(
    data_source: str,
    config: ProjectConfig,
    local_csv_dir: str = "",
) -> AkshareClient | TushareClient | HybridAkshareClient:
    """根据参数创建数据客户端。

    Args:
        data_source: 数据源类型。
        config: 项目配置。
        local_csv_dir: 离线 BaoStock CSV 目录；非空时使用混合客户端。

    Returns:
        AkshareClient | TushareClient | HybridAkshareClient: 数据客户端实例。
    """
    if local_csv_dir:
        return HybridAkshareClient(Path(local_csv_dir))
    if data_source == "akshare":
        return AkshareClient()
    if data_source == "tushare":
        return TushareClient(token=config.tushare_token or "")
    if config.tushare_token:
        return TushareClient(token=config.tushare_token)
    return AkshareClient()


def main() -> None:
    """执行数据读取和因子构建流程。"""
    configure_console_output()
    args = parse_args()
    config = ProjectConfig.from_root()
    config.ensure_directories()

    client = build_client(args.data_source, config, local_csv_dir=args.local_csv_dir)
    data_service = DataService(client=client, config=config)
    factor_engine = FactorEngine()

    print("[Pipeline] 开始加载研究数据", flush=True)
    bundle = data_service.build_research_panel(
        start_date=args.start_date,
        end_date=args.end_date,
        index_code=args.index_code,
        universe_limit=args.universe_limit or None,
    )
    print("[Pipeline] 开始计算因子", flush=True)
    factor_panel = factor_engine.compute_factors(bundle.research_panel)
    print("[Pipeline] 开始生成超额收益标签", flush=True)
    labeled_panel = factor_engine.build_excess_return_label(
        factor_panel=factor_panel,
        benchmark=bundle.benchmark,
    )
    print("[Pipeline] 开始标准化模型面板", flush=True)
    model_panel = factor_engine.prepare_model_panel(labeled_panel)

    output_path = data_service.save_frame(model_panel, args.output)
    print(f"因子面板已生成：{output_path}")


if __name__ == "__main__":
    main()
