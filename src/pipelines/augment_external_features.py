"""为现有因子面板补充外部数据特征。"""

from __future__ import annotations

import argparse

import pandas as pd

from src.config import ProjectConfig
from src.data import AkshareClient, DataService
from src.utils.console import configure_console_output


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="为现有因子面板补充外部数据特征")
    parser.add_argument(
        "--input",
        required=True,
        help="位于 data/ 目录下的原始因子面板相对路径",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="位于 data/ 目录下的增强后因子面板相对路径",
    )
    return parser.parse_args()


def main() -> None:
    """执行外部数据增强。"""
    configure_console_output()
    args = parse_args()
    config = ProjectConfig.from_root()
    config.ensure_directories()
    data_service = DataService(client=AkshareClient(), config=config)

    panel_path = config.data_dir / args.input
    panel = pd.read_csv(panel_path)
    panel["trade_date"] = panel["trade_date"].astype(str)
    start_date = panel["trade_date"].min()
    end_date = panel["trade_date"].max()

    northbound = data_service.load_northbound_flow(start_date=start_date, end_date=end_date)
    if not northbound.empty and "north_money" not in panel.columns:
        northbound["trade_date"] = northbound["trade_date"].astype(str)
        panel = panel.merge(northbound, on="trade_date", how="left")
    if "north_money" in panel.columns and "northbound_net_inflow" not in panel.columns:
        panel["northbound_net_inflow"] = pd.to_numeric(panel["north_money"], errors="coerce")

    macro_m2 = data_service.load_macro_m2_yoy(start_date=start_date, end_date=end_date)
    if not macro_m2.empty and "m2_yoy" not in panel.columns:
        macro_m2["trade_date"] = macro_m2["trade_date"].astype(str)
        panel = data_service._merge_macro_series(panel=panel, macro_frame=macro_m2)
        
    macro_spread = data_service.load_macro_interest_rate_spread(start_date=start_date, end_date=end_date)
    if not macro_spread.empty and "cn_spread_10y_2y" not in panel.columns:
        macro_spread["trade_date"] = macro_spread["trade_date"].astype(str)
        panel = data_service._merge_macro_series(panel=panel, macro_frame=macro_spread)

    output_path = data_service.save_frame(panel, args.output)
    print(f"外部数据增强面板已生成：{output_path}")


if __name__ == "__main__":
    main()
