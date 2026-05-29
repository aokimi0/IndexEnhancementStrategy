"""Phase 1.2：为点位并集中未缓存的成分补抓数据。

- daily：akshare 腾讯日线；空则 baostock 兜底（退市股）。
- daily_basic：baostock query_history_k_data_plus（换手/估值）。
- financial：baostock 季度财务，按个股成分起止年份限制查询区间以提速。

所有 baostock 调用包硬超时；单股失败记录并跳过。写入与项目一致的缓存路径，
后续 build_research_panel 可直接命中缓存。支持分片并行：--shard i --nshards N。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.akshare_client import AkshareClient  # noqa: E402
from scripts.pit_common import bs_result_to_frame, to_baostock_code  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
LOG_DIR = ROOT / "logs"
START = "20150101"
END = "20241231"


def shard_log(shard: int, message: str) -> None:
    """分片日志：写入 logs/pit_fetch_shard{n}.log 并打印。"""
    import datetime as dt

    line = f"[{dt.datetime.now():%H:%M:%S}][s{shard}] {message}"
    print(line, flush=True)
    with (LOG_DIR / f"pit_fetch_shard{shard}.log").open("a", encoding="utf-8") as h:
        h.write(line + "\n")


class TimeoutAkshareClient(AkshareClient):
    """在 baostock 低层调用上加硬超时；超时则重登录，避免单股卡死全局。"""

    def __init__(self, timeout: float = 25.0) -> None:
        super().__init__()
        self._timeout = timeout

    def _safe_bs(self, func, label: str):
        """单次新建线程执行 baostock 调用，超时抛错并标记重登录。"""
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            fut = ex.submit(func)
            return fut.result(timeout=self._timeout)
        except concurrent.futures.TimeoutError as exc:
            # 放弃挂死线程；强制下次重登录新建 socket
            self._bs_logged_in = False
            raise TimeoutError(f"{label} 超时{self._timeout}s") from exc
        finally:
            ex.shutdown(wait=False)

    def _query_baostock_history_metrics(self, ts_code, start_date, end_date):  # type: ignore[override]
        def _run():
            rs = self.bs.query_history_k_data_plus(
                self._to_baostock_code(ts_code),
                "date,code,turn,peTTM,pbMRQ,psTTM",
                start_date=self._to_baostock_date(start_date),
                end_date=self._to_baostock_date(end_date),
                frequency="d",
                adjustflag="3",
            )
            return bs_result_to_frame(rs)

        return self._safe_bs(_run, f"k_data({ts_code})")

    def _query_baostock_quarter_frame(self, ts_code, year, quarter):  # type: ignore[override]
        code = self._to_baostock_code(ts_code)
        base_keys = ["code", "pubDate", "statDate"]
        funcs = [
            self.bs.query_profit_data,
            self.bs.query_growth_data,
            self.bs.query_operation_data,
            self.bs.query_cash_flow_data,
            self.bs.query_dupont_data,
        ]
        merged: pd.DataFrame | None = None
        for fn in funcs:
            def _run(fn=fn):
                return bs_result_to_frame(fn(code=code, year=year, quarter=quarter))

            try:
                frame = self._safe_bs(_run, f"fin({ts_code},{year}Q{quarter})")
            except Exception:
                continue
            if frame.empty:
                continue
            merged = frame if merged is None else merged.merge(frame, on=base_keys, how="outer")
        return merged if merged is not None else pd.DataFrame()

    def daily_with_fallback(self, ts_code: str) -> pd.DataFrame:
        """akshare 日线；空则 baostock 兜底（保持同列结构）。"""
        frame = self.daily(ts_code=ts_code, start_date=START, end_date=END)
        if not frame.empty:
            return frame
        return self._baostock_daily(ts_code)

    def _baostock_daily(self, ts_code: str) -> pd.DataFrame:
        def _run():
            rs = self.bs.query_history_k_data_plus(
                self._to_baostock_code(ts_code),
                "date,open,high,low,close,preclose,volume,amount,turn",
                start_date=self._to_baostock_date(START),
                end_date=self._to_baostock_date(END),
                frequency="d",
                adjustflag="3",
            )
            return bs_result_to_frame(rs)

        frame = self._safe_bs(_run, f"bs_daily({ts_code})")
        if frame.empty:
            return pd.DataFrame(
                columns=["ts_code", "trade_date", "open", "high", "low", "close",
                         "pre_close", "vol", "amount", "turnover_rate"]
            )
        out = pd.DataFrame(
            {
                "ts_code": ts_code,
                "trade_date": pd.to_datetime(frame["date"]).dt.strftime("%Y%m%d"),
                "open": pd.to_numeric(frame["open"], errors="coerce"),
                "high": pd.to_numeric(frame["high"], errors="coerce"),
                "low": pd.to_numeric(frame["low"], errors="coerce"),
                "close": pd.to_numeric(frame["close"], errors="coerce"),
                "pre_close": pd.to_numeric(frame["preclose"], errors="coerce"),
                # baostock volume 单位为股，换算为手以与腾讯口径一致
                "vol": pd.to_numeric(frame["volume"], errors="coerce") / 100.0,
                "amount": pd.to_numeric(frame["amount"], errors="coerce"),
                "turnover_rate": pd.to_numeric(frame["turn"], errors="coerce"),
            }
        )
        return out.sort_values("trade_date").reset_index(drop=True)


def cache_path(dataset: str, ts_code: str) -> Path:
    sym = ts_code.replace(".", "_")
    return CACHE / dataset / f"{sym}_{START}_{END}.csv"


def fetch_one(client: TimeoutAkshareClient, ts_code: str, span_start_year: int, span_end_year: int, shard: int) -> dict:
    """补抓单只股票的三类数据，返回状态。"""
    status = {"ts_code": ts_code, "daily": "", "basic": "", "financial": ""}

    # daily
    p = cache_path("daily", ts_code)
    if p.exists():
        status["daily"] = "cached"
    else:
        try:
            df = client.daily_with_fallback(ts_code)
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(p, index=False, encoding="utf-8-sig")
            status["daily"] = f"ok({len(df)})"
        except Exception as exc:  # noqa: BLE001
            status["daily"] = f"FAIL:{exc}"

    # daily_basic
    p = cache_path("daily_basic_v2", ts_code)
    if p.exists():
        status["basic"] = "cached"
    else:
        try:
            df = client.daily_basic(ts_code=ts_code, start_date=START, end_date=END)
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(p, index=False, encoding="utf-8-sig")
            status["basic"] = f"ok({len(df)})"
        except Exception as exc:  # noqa: BLE001
            status["basic"] = f"FAIL:{exc}"

    # financial (span-limited)
    p = cache_path("financial_indicators_v2", ts_code)
    if p.exists():
        status["financial"] = "cached"
    else:
        try:
            fin_start = f"{max(2014, span_start_year - 1)}0101"
            # 财务区间末尾按个股最后在册年份 +1 收窄，覆盖标签前瞻窗口同时提速
            fin_end_year = min(2024, span_end_year + 1)
            fin_end = f"{fin_end_year}1231"
            df = client.fina_indicator(ts_code=ts_code, start_date=fin_start, end_date=fin_end)
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(p, index=False, encoding="utf-8-sig")
            status["financial"] = f"ok({len(df)})"
        except Exception as exc:  # noqa: BLE001
            status["financial"] = f"FAIL:{exc}"

    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=25.0)
    args = parser.parse_args()

    need = pd.read_csv(ROOT / "data" / "processed" / "hs300_pit_need_fetch.csv")["ts_code"].tolist()
    spans = pd.read_csv(ROOT / "data" / "processed" / "hs300_pit_spans.csv")
    span_start = {r["ts_code"]: int(str(r["min"])[:4]) for _, r in spans.iterrows()}
    span_end = {r["ts_code"]: int(str(r["max"])[:4]) for _, r in spans.iterrows()}

    my = [c for i, c in enumerate(need) if i % args.nshards == args.shard]
    shard_log(args.shard, f"分配 {len(my)} 只 (总 {len(need)}, nshards={args.nshards})")

    client = TimeoutAkshareClient(timeout=args.timeout)
    results = []
    t0 = time.time()
    for idx, ts_code in enumerate(my, 1):
        st = fetch_one(client, ts_code, span_start.get(ts_code, 2015), span_end.get(ts_code, 2024), args.shard)
        results.append(st)
        if "FAIL" in (st["daily"] + st["basic"] + st["financial"]):
            shard_log(args.shard, f"[{idx}/{len(my)}] {ts_code} {st}")
        elif idx % 10 == 0 or idx == len(my):
            rate = idx / max(time.time() - t0, 1e-6)
            shard_log(args.shard, f"[{idx}/{len(my)}] 进度 {ts_code} 速率{rate:.2f}股/s")

    pd.DataFrame(results).to_csv(LOG_DIR / f"pit_fetch_status_shard{args.shard}.csv", index=False)
    shard_log(args.shard, f"完成，用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
