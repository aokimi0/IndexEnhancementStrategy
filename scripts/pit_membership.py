"""Phase 1.1：构建沪深300点位成分(point-in-time membership)。

对 2015-01 ~ 2024-12 每个月，调 baostock query_hs300_stocks(date=...)
取当时真实成分，转 ts_code 格式，存 data/processed/hs300_pit_membership.csv。

baostock query_hs300_stocks 半年度更新；对每月取一个交易日，若为空则在
该月内向后/向前尝试若干天。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pit_common import bs_query_with_retry, log, to_ts_code  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "processed" / "hs300_pit_membership.csv"


def month_iter(start: str, end: str):
    """生成 YYYY-MM 月份序列。"""
    rng = pd.date_range(start=start, end=end, freq="MS")
    for ts in rng:
        yield ts.strftime("%Y-%m")


def query_members_for_month(year_month: str) -> list[str]:
    """取某月成分；尝试多个候选日以避开非交易日/空返回。"""
    year, month = year_month.split("-")
    candidates = [f"{year}-{month}-{day:02d}" for day in (15, 16, 14, 20, 10, 25, 5, 28)]
    for date in candidates:
        try:
            frame = bs_query_with_retry(
                lambda bs, d=date: bs.query_hs300_stocks(date=d),
                label=f"hs300({date})",
                timeout=30.0,
                retries=2,
            )
        except Exception as exc:  # noqa: BLE001
            log(f"  {year_month} {date} 查询异常: {exc}")
            continue
        if not frame.empty and "code" in frame.columns:
            return [to_ts_code(c) for c in frame["code"].tolist()]
    return []


def main() -> None:
    log("===== Phase 1.1 点位成分构建开始 =====")
    rows: list[dict[str, str]] = []
    months = list(month_iter("2015-01-01", "2024-12-01"))
    for ym in months:
        members = query_members_for_month(ym)
        if not members:
            log(f"{ym}: 空（将沿用上月）")
            continue
        for code in members:
            rows.append({"month": ym, "ts_code": code})
        log(f"{ym}: {len(members)} 只")

    frame = pd.DataFrame(rows)
    # 对于个别空月，用前向填充补齐（保持每月都有成分）
    present_months = set(frame["month"].unique())
    missing = [m for m in months if m not in present_months]
    if missing:
        log(f"空月需前向填充: {missing}")
        filled: list[pd.DataFrame] = [frame]
        last_known: pd.DataFrame | None = None
        for ym in months:
            cur = frame[frame["month"] == ym]
            if not cur.empty:
                last_known = cur
            elif last_known is not None:
                tmp = last_known.copy()
                tmp["month"] = ym
                filled.append(tmp)
        frame = pd.concat(filled, ignore_index=True)

    frame = frame.drop_duplicates(subset=["month", "ts_code"]).sort_values(["month", "ts_code"]).reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    union = sorted(frame["ts_code"].unique().tolist())
    log(f"月份数: {frame['month'].nunique()}, 并集 universe: {len(union)} 只")
    log(f"membership 已写出: {OUT_PATH}")

    # 同时写出 union 列表，便于后续补数据
    pd.DataFrame({"ts_code": union}).to_csv(
        ROOT / "data" / "processed" / "hs300_pit_union.csv", index=False, encoding="utf-8-sig"
    )
    log(f"union 已写出: {ROOT / 'data' / 'processed' / 'hs300_pit_union.csv'}")


if __name__ == "__main__":
    main()
