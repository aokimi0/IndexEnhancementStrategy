"""Phase 0 可行性自检：退市/被调出历史成分的数据可得性。

取若干 "2015 在册、当前(已缓存 299)不在" 的历史成分，验证能否取得：
  1) 日线（收盘/复权/换手）
  2) 估值（pe/pb）
  3) 季度财务（roe/毛利率/净利率等）

akshare 取不到（多为退市股）时，用 baostock 兜底（带硬超时）。
若多数样本三类数据都拿不到 -> 报告点位重建在免费源下不可行。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pit_common import (  # noqa: E402
    bs_query_with_retry,
    log,
    to_baostock_code,
    to_ts_code,
)

ROOT = Path(__file__).resolve().parents[1]
EXISTING_PANEL = ROOT / "data" / "processed" / "hs300_factor_panel_constrained_fast_extended_2015_2024.csv"

PROBE_DATES = ["2015-01-30", "2016-06-30", "2018-06-29"]
N_SAMPLES = 6


def get_members(date: str) -> list[str]:
    """调 baostock query_hs300_stocks 取某日成分（ts_code 格式）。"""
    frame = bs_query_with_retry(
        lambda bs: bs.query_hs300_stocks(date=date),
        label=f"query_hs300_stocks({date})",
        timeout=30.0,
    )
    if frame.empty or "code" not in frame.columns:
        return []
    return [to_ts_code(c) for c in frame["code"].tolist()]


def test_akshare_daily(ts_code: str) -> tuple[bool, str]:
    """测试 akshare 腾讯日线。"""
    try:
        import akshare as ak

        symbol = _to_ak_symbol(ts_code)
        frame = ak.stock_zh_a_hist_tx(symbol=symbol, start_date="20150101", end_date="20151231", adjust="")
        if frame is not None and not frame.empty:
            return True, f"akshare {len(frame)} 行"
        return False, "akshare 空"
    except Exception as exc:  # noqa: BLE001
        return False, f"akshare 异常 {exc}"


def test_baostock_daily(ts_code: str) -> tuple[bool, str]:
    """测试 baostock 日线（含换手/估值/复权）。"""
    try:
        frame = bs_query_with_retry(
            lambda bs: bs.query_history_k_data_plus(
                to_baostock_code(ts_code),
                "date,code,close,volume,turn,peTTM,pbMRQ,psTTM",
                start_date="2015-01-01",
                end_date="2015-12-31",
                frequency="d",
                adjustflag="3",
            ),
            label=f"k_data({ts_code})",
            timeout=25.0,
            retries=2,
        )
        if not frame.empty:
            pe_ok = pd.to_numeric(frame.get("peTTM"), errors="coerce").notna().any()
            turn_ok = pd.to_numeric(frame.get("turn"), errors="coerce").notna().any()
            return True, f"baostock {len(frame)} 行 pe={pe_ok} turn={turn_ok}"
        return False, "baostock 空"
    except Exception as exc:  # noqa: BLE001
        return False, f"baostock 异常 {exc}"


def test_baostock_financial(ts_code: str) -> tuple[bool, str]:
    """测试 baostock 季度财务（盈利能力）。"""
    try:
        frame = bs_query_with_retry(
            lambda bs: bs.query_profit_data(code=to_baostock_code(ts_code), year=2015, quarter=2),
            label=f"profit({ts_code})",
            timeout=25.0,
            retries=2,
        )
        if not frame.empty and "roeAvg" in frame.columns:
            roe = pd.to_numeric(frame["roeAvg"], errors="coerce")
            return roe.notna().any(), f"baostock profit roe={roe.tolist()}"
        return False, "baostock profit 空"
    except Exception as exc:  # noqa: BLE001
        return False, f"baostock 异常 {exc}"


def _to_ak_symbol(ts_code: str) -> str:
    symbol, market = to_ts_code(ts_code).split(".")
    prefix = "sh" if market.upper() == "SH" else "sz"
    return f"{prefix}{symbol}"


def main() -> None:
    log("===== Phase 0 可行性自检开始 =====")
    existing = pd.read_csv(EXISTING_PANEL, usecols=["ts_code"])
    current_codes = set(existing["ts_code"].unique().tolist())
    log(f"已缓存当前成分数: {len(current_codes)}")

    union_hist: set[str] = set()
    for date in PROBE_DATES:
        members = get_members(date)
        log(f"{date} 成分数: {len(members)}")
        union_hist.update(members)
    log(f"探测日并集成分数: {len(union_hist)}")

    delisted_candidates = sorted(union_hist - current_codes)
    log(f"历史在册但当前未缓存的候选数: {len(delisted_candidates)}")
    if not delisted_candidates:
        log("没有候选历史成分（异常），中止。")
        return

    samples = delisted_candidates[:N_SAMPLES]
    log(f"抽样测试: {samples}")

    results = []
    for ts_code in samples:
        log(f"--- 测试 {ts_code} ---")
        ak_ok, ak_msg = test_akshare_daily(ts_code)
        log(f"  日线 akshare: {ak_ok} | {ak_msg}")
        bs_daily_ok, bs_daily_msg = (True, "skip(ak ok)") if ak_ok else test_baostock_daily(ts_code)
        if not ak_ok:
            log(f"  日线 baostock兜底: {bs_daily_ok} | {bs_daily_msg}")
        # 估值与换手始终用 baostock（项目 daily_basic 走 baostock）
        val_ok, val_msg = test_baostock_daily(ts_code)
        log(f"  估值/换手 baostock: {val_ok} | {val_msg}")
        fin_ok, fin_msg = test_baostock_financial(ts_code)
        log(f"  财务 baostock: {fin_ok} | {fin_msg}")

        daily_ok = ak_ok or bs_daily_ok
        results.append(
            {
                "ts_code": ts_code,
                "daily_ok": daily_ok,
                "valuation_ok": val_ok,
                "financial_ok": fin_ok,
                "all_three": daily_ok and val_ok and fin_ok,
            }
        )

    summary = pd.DataFrame(results)
    log("===== Phase 0 结果汇总 =====")
    log("\n" + summary.to_string(index=False))
    n_all = int(summary["all_three"].sum())
    n = len(summary)
    log(f"三类全可得样本: {n_all}/{n}")
    log(f"日线可得: {int(summary['daily_ok'].sum())}/{n}, "
        f"估值可得: {int(summary['valuation_ok'].sum())}/{n}, "
        f"财务可得: {int(summary['financial_ok'].sum())}/{n}")
    verdict = "PASS" if n_all >= (n + 1) // 2 else "FAIL"
    log(f"PHASE0_VERDICT={verdict}")
    summary.to_csv(ROOT / "logs" / "pit_phase0_summary.csv", index=False)


if __name__ == "__main__":
    main()
