"""构建沪深300日频情感因子面板。

流程：

1. 从沪深300成分股清单（DataService）取股票池；
2. 通过 `NewsClient` 抓取个股新闻并去重清洗；
3. 调 `ClaudeSentimentScorer` 打分（默认 haiku，可 `--use-opus` 升级）；
4. 调 `build_daily_sentiment_factor` 聚合到日频；
5. 落盘到 `data/<output>`。

默认走 haiku 控成本；`--dry-run` 只跑前 5 只 × 1 周，统计抓到的新闻数与预算占用后立即停止。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

from src.config import ProjectConfig
from src.data import (
    AkshareClient,
    BudgetExceededError,
    ClaudeSentimentScorer,
    DataService,
    NewsClient,
)
from src.factors.sentiment import build_daily_sentiment_factor
from src.utils.console import configure_console_output


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        argparse.Namespace: 命令行参数对象。
    """
    parser = argparse.ArgumentParser(description="构建沪深300日频情感因子面板")
    parser.add_argument("--start", required=True, help="开始日期，YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="结束日期，YYYY-MM-DD")
    parser.add_argument(
        "--output",
        default="processed/sentiment_panel.csv",
        help="输出到 data/ 下的相对路径",
    )
    parser.add_argument(
        "--index-code",
        default="000300.SH",
        help="股票池所属指数，默认沪深300",
    )
    parser.add_argument(
        "--max-codes",
        type=int,
        default=0,
        help="最多处理的股票数量，0 表示全部成分股",
    )
    parser.add_argument(
        "--codes",
        type=str,
        default=None,
        help="显式指定股票池（逗号分隔），优先级高于 --max-codes",
    )
    parser.add_argument(
        "--max-usd",
        type=float,
        default=None,
        help="覆盖 .env 中 ANTHROPIC_MAX_USD 的预算上限",
    )
    parser.add_argument(
        "--use-opus",
        action="store_true",
        help="使用 opus-4-7（默认走 haiku-4-5 控成本）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="LLM 单批新闻条数，<=10",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅取前 5 只股票最近 7 天数据评估成本后停止",
    )
    return parser.parse_args()


def _to_compact(date_string: str) -> str:
    """将 `YYYY-MM-DD` 转为 `YYYYMMDD`。"""
    return pd.to_datetime(date_string).strftime("%Y%m%d")


def _load_universe(
    data_service: DataService,
    index_code: str,
    start_date: str,
    end_date: str,
    max_codes: int,
) -> list[str]:
    """加载股票池并按 `max_codes` 截断。"""
    codes = data_service.get_research_universe(
        index_code=index_code,
        start_date=start_date,
        end_date=end_date,
    )
    if max_codes and max_codes > 0:
        return codes[:max_codes]
    return codes


def _load_calendar(data_service: DataService, start_date: str, end_date: str) -> list[str]:
    """从基准指数日线推断交易日历。"""
    benchmark = data_service.load_hs300_benchmark(
        start_date=start_date,
        end_date=end_date,
    )
    if benchmark.empty:
        return []
    return benchmark["trade_date"].astype(str).tolist()


def _summarize(metric: dict, header: str) -> None:
    """统一格式打印实验度量。"""
    print(f"\n[Pipeline] {header}", flush=True)
    for key, value in metric.items():
        print(f"  - {key}: {value}", flush=True)


def main() -> None:
    """执行情感因子构建主流程。"""
    configure_console_output()
    args = parse_args()
    config = ProjectConfig.from_root()
    config.ensure_directories()

    start_compact = _to_compact(args.start)
    end_compact = _to_compact(args.end)

    if args.dry_run:
        # dry-run：前 5 只股票 + 最近 7 天，强制覆盖。
        end_ts = pd.to_datetime(args.end)
        start_ts = end_ts - timedelta(days=7)
        start_compact = start_ts.strftime("%Y%m%d")
        end_compact = end_ts.strftime("%Y%m%d")
        args.max_codes = 5 if args.max_codes in (0, None) else min(args.max_codes, 5)

    data_service = DataService(client=AkshareClient(), config=config)

    if args.codes:
        universe = [code.strip() for code in args.codes.split(",") if code.strip()]
        print(f"[Pipeline] 使用显式股票池：{universe}", flush=True)
    else:
        print(
            f"[Pipeline] 加载沪深300成分（{args.start} -> {args.end}）",
            flush=True,
        )
        universe = _load_universe(
            data_service=data_service,
            index_code=args.index_code,
            start_date=start_compact,
            end_date=end_compact,
            max_codes=args.max_codes,
        )
    if not universe:
        print("[Pipeline] 未取到任何股票，退出。", flush=True)
        sys.exit(1)
    print(f"[Pipeline] 股票池规模：{len(universe)}", flush=True)

    print("[Pipeline] 加载交易日历", flush=True)
    calendar = _load_calendar(
        data_service=data_service,
        start_date=start_compact,
        end_date=end_compact,
    )
    if not calendar:
        print("[Pipeline] 交易日历为空，无法对齐，退出。", flush=True)
        sys.exit(1)

    print("[Pipeline] 抓取个股新闻", flush=True)
    news_client = NewsClient(
        cache_dir=config.cache_dir / "news",
        failure_log_path=config.logs_dir / "news_failures.log",
    )
    news_frame = news_client.batch_fetch(
        ts_codes=universe,
        start_date=args.start if not args.dry_run else (pd.to_datetime(args.end) - timedelta(days=7)).strftime("%Y-%m-%d"),
        end_date=args.end,
        use_cache=True,
        progress=True,
    )
    print(f"[Pipeline] 累计抓到新闻 {len(news_frame)} 条", flush=True)

    if news_frame.empty:
        print("[Pipeline] 无可打分新闻，落空表后退出。", flush=True)
        empty_path = data_service.save_frame(
            pd.DataFrame(columns=["trade_date", "ts_code", "sentiment_daily", "sentiment_count", "sentiment_ma5"]),
            args.output,
        )
        print(f"[Pipeline] 已写出空表：{empty_path}")
        return

    scorer = ClaudeSentimentScorer(
        cache_dir=config.cache_dir / "llm_sentiment",
        budget_usd=args.max_usd,
    )

    if args.dry_run:
        # dry-run 只对最多 10 条样本估算实际费用，估算其余样本的成本。
        sample_frame = news_frame.head(10)
        scored_sample = scorer.score_dataframe(
            sample_frame,
            batch_size=min(args.batch_size, 10),
            use_fast=not args.use_opus,
        )
        snapshot = scorer.tracker.snapshot()
        avg_cost = snapshot["spent_usd"] / max(1, len(sample_frame))
        projected = avg_cost * len(news_frame)
        _summarize(
            {
                "model": scorer.fast_model if not args.use_opus else scorer.model,
                "prompt_version": scorer.prompt_version,
                "sampled_news": len(sample_frame),
                "total_news": len(news_frame),
                "avg_cost_per_news_usd": round(avg_cost, 6),
                "projected_total_cost_usd": round(projected, 4),
                **snapshot,
            },
            header="dry-run 估算结果",
        )
        print("\n[Pipeline] dry-run 模式：打印前 5 行情感快照后退出。", flush=True)
        print(scored_sample.head().to_string(index=False), flush=True)
        return

    print("[Pipeline] 开始 LLM 打分", flush=True)
    try:
        scored_news = scorer.score_dataframe(
            news_frame,
            batch_size=min(args.batch_size, 10),
            use_fast=not args.use_opus,
        )
    except BudgetExceededError as exc:
        print(f"[Pipeline] 预算超限：{exc}", flush=True)
        scored_news = pd.DataFrame()

    if scored_news.empty:
        print("[Pipeline] 无打分结果，退出。", flush=True)
        return

    print("[Pipeline] 聚合日频情感因子", flush=True)
    factor_frame = build_daily_sentiment_factor(
        scored_news=scored_news,
        calendar=calendar,
    )

    snapshot = scorer.tracker.snapshot()
    _summarize(snapshot, header="LLM 用量与成本")

    output_path = data_service.save_frame(factor_frame, args.output)
    print(f"\n[Pipeline] 情感因子面板已生成：{output_path}", flush=True)

    snapshot_path: Path = config.logs_dir / "sentiment_run_snapshot.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with snapshot_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
                "rows": int(len(factor_frame)),
                "tracker": snapshot,
            },
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    print(f"[Pipeline] 运行快照已写入：{snapshot_path}", flush=True)


if __name__ == "__main__":
    main()
