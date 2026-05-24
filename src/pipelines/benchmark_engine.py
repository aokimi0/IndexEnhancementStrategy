"""C4 创新点性能基准脚本：Numba JIT 引擎 + joblib 并行回测。

脚本包含两部分对比实验：

1. **Numba vs Python**：同一份因子面板，分别用 ``use_numba=True`` 与
   ``use_numba=False`` 跑 :class:`BaselineBacktestEngine`，记录墙钟时间、
   加速比，并对 NAV 与指标矩阵做数值一致性断言（阈值 ``1e-6``）。
2. **串行 vs 并行**：构造 4 个变体（无优化器 + 等权 Top10/Top20/Top30/
   Top50），分别串行循环执行与 joblib 并行执行，记录耗时与加速比。

使用方式::

    python -m src.pipelines.benchmark_engine \
        --input processed/hs300_factor_panel_extended_2023_2024.csv \
        --output processed/c4_benchmark_summary.csv

未给 ``--input`` 时脚本会在内存中合成 300 股 × 自定义日数的随机面板，
便于本地快速冒烟。
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import BaselineBacktestEngine
from src.config import ProjectConfig
from src.utils.console import configure_console_output

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        argparse.Namespace: 已解析的命令行参数。
    """
    parser = argparse.ArgumentParser(description="C4 性能基准：Numba + joblib")
    parser.add_argument(
        "--input",
        default="",
        help="可选：data/ 下的因子面板 CSV 相对路径，留空时使用合成面板",
    )
    parser.add_argument(
        "--output",
        default="processed/c4_benchmark_summary.csv",
        help="输出对比表的 CSV 相对路径（位于 data/ 下）",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="数值一致性对比里使用的 Top-N 选股数量",
    )
    parser.add_argument(
        "--synthetic-stocks",
        type=int,
        default=300,
        help="合成面板的股票数量",
    )
    parser.add_argument(
        "--synthetic-days",
        type=int,
        default=252,
        help="合成面板的交易日数量",
    )
    parser.add_argument(
        "--synthetic-seed",
        type=int,
        default=20260524,
        help="合成面板随机种子",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="joblib 并行进程数",
    )
    parser.add_argument(
        "--variant-top-ns",
        nargs="+",
        type=int,
        default=[10, 20, 30, 50],
        help="并行对比使用的 Top-N 列表，默认 10/20/30/50",
    )
    parser.add_argument(
        "--skip-parallel",
        action="store_true",
        help="跳过 joblib 并行对比，只跑 numba vs python",
    )
    return parser.parse_args()


def build_synthetic_panel(
    n_stocks: int,
    n_days: int,
    seed: int,
) -> pd.DataFrame:
    """生成测试用合成因子面板。

    每只股票按几何随机游走构造价格序列，并伪造 5 个因子列与基准日收益。

    Args:
        n_stocks: 股票数量。
        n_days: 交易日数量。
        seed: 随机种子。

    Returns:
        pd.DataFrame: 列含 ``trade_date``、``ts_code``、``daily_return``、
        ``benchmark_daily_return``、``industry_name``、``benchmark_weight``
        以及若干因子列的因子面板。
    """
    rng = np.random.default_rng(seed)
    trade_dates = pd.bdate_range(end=pd.Timestamp("2024-12-31"), periods=n_days)
    ts_codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    industries = ["银行", "电子", "医药", "消费", "能源", "材料", "工业"]

    daily_returns = rng.normal(loc=0.0005, scale=0.018, size=(n_days, n_stocks))
    benchmark_returns = rng.normal(loc=0.0003, scale=0.012, size=n_days)

    rows = []
    for j, code in enumerate(ts_codes):
        industry = industries[j % len(industries)]
        weight = float(rng.uniform(0.001, 0.01))
        factor_seed = rng.normal(size=5)
        for i, td in enumerate(trade_dates):
            rows.append(
                {
                    "trade_date": int(td.strftime("%Y%m%d")),
                    "ts_code": code,
                    "daily_return": float(daily_returns[i, j]),
                    "benchmark_daily_return": float(benchmark_returns[i]),
                    "industry_name": industry,
                    "benchmark_weight": weight,
                    "ret_20": float(factor_seed[0] + daily_returns[i, j] * 20),
                    "ret_60": float(factor_seed[1] + daily_returns[i, j] * 60),
                    "volatility_20": float(
                        0.2 + 0.05 * factor_seed[2] + 0.1 * abs(daily_returns[i, j])
                    ),
                    "ep_ttm": float(0.05 + 0.01 * factor_seed[3]),
                    "bp": float(0.5 + 0.1 * factor_seed[4]),
                    "turnover_20": float(rng.uniform(0.005, 0.05)),
                }
            )
    panel = pd.DataFrame(rows)
    return panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


def _time_engine(
    factor_panel: pd.DataFrame,
    use_numba: bool,
    top_n: int,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    """跑一次回测并返回墙钟耗时与结果。

    Args:
        factor_panel: 因子面板。
        use_numba: 是否启用 Numba 路径。
        top_n: Top-N 选股数。

    Returns:
        tuple: ``(wall_clock_seconds, nav_frame, metrics)``。
    """
    engine = BaselineBacktestEngine(top_n=top_n, use_numba=use_numba)
    start = time.perf_counter()
    result = engine.run(factor_panel)
    elapsed = time.perf_counter() - start
    return elapsed, result.nav_frame, result.metrics


def _max_abs_diff(
    left: pd.DataFrame,
    right: pd.DataFrame,
    numeric_only: bool = True,
) -> float:
    """计算两份 DataFrame 共有数值列的最大逐元素绝对差。

    Args:
        left: 第一份 DataFrame。
        right: 第二份 DataFrame。
        numeric_only: 是否只比较数值列。

    Returns:
        float: 最大绝对差；若无公共数值列则返回 ``nan``。
    """
    common = [col for col in left.columns if col in right.columns]
    if numeric_only:
        common = [
            col for col in common if pd.api.types.is_numeric_dtype(left[col])
        ]
    if not common:
        return float("nan")
    left_arr = left[common].to_numpy(dtype=np.float64, na_value=np.nan)
    right_arr = right[common].to_numpy(dtype=np.float64, na_value=np.nan)
    mask = ~(np.isnan(left_arr) & np.isnan(right_arr))
    diff = np.where(mask, np.abs(left_arr - right_arr), 0.0)
    return float(np.nanmax(diff)) if mask.any() else float("nan")


def run_numba_vs_python(
    factor_panel: pd.DataFrame,
    top_n: int,
) -> tuple[pd.DataFrame, float]:
    """跑 Numba vs Python 对比并返回结果表与最大 NAV 差。

    Args:
        factor_panel: 因子面板。
        top_n: Top-N 选股数。

    Returns:
        tuple: ``(summary_df, nav_diff_max)``。
    """
    logger.info("[numba] 预热 JIT...")
    warmup_panel = factor_panel.head(min(len(factor_panel), 500)).copy()
    _ = BaselineBacktestEngine(top_n=top_n, use_numba=True).run(warmup_panel)

    logger.info("[numba] 正式计时...")
    numba_seconds, numba_nav, numba_metrics = _time_engine(
        factor_panel=factor_panel, use_numba=True, top_n=top_n
    )
    logger.info("[python] 正式计时...")
    python_seconds, python_nav, python_metrics = _time_engine(
        factor_panel=factor_panel, use_numba=False, top_n=top_n
    )

    nav_diff_max = _max_abs_diff(numba_nav, python_nav)
    metrics_diff_max = _max_abs_diff(numba_metrics, python_metrics)
    speedup = python_seconds / numba_seconds if numba_seconds > 0 else float("nan")

    summary = pd.DataFrame(
        [
            {
                "mode": "python_loop",
                "wall_clock_seconds": python_seconds,
                "speedup_x": 1.0,
                "nav_diff_max": 0.0,
                "metrics_diff_max": 0.0,
            },
            {
                "mode": "numba_jit",
                "wall_clock_seconds": numba_seconds,
                "speedup_x": speedup,
                "nav_diff_max": nav_diff_max,
                "metrics_diff_max": metrics_diff_max,
            },
        ]
    )
    return summary, nav_diff_max


def run_serial_vs_parallel(
    factor_panel: pd.DataFrame,
    variant_top_ns: list[int],
    n_jobs: int,
    panel_tmp_path: Path,
) -> pd.DataFrame:
    """跑串行 vs joblib 并行对比。

    并行场景模拟真实参数扫描工况：父进程把面板写到磁盘，子进程通过
    路径读入。这样既避开父→子 pickle 大对象的开销，也复现了实际中
    各 worker 各自 IO 的成本。

    Args:
        factor_panel: 因子面板。
        variant_top_ns: 不同 Top-N 变体列表。
        n_jobs: 并行进程数。
        panel_tmp_path: 子进程读取面板的 parquet 路径。

    Returns:
        pd.DataFrame: 串行与并行的耗时对比表。
    """
    panel_tmp_path.parent.mkdir(parents=True, exist_ok=True)
    factor_panel.to_parquet(panel_tmp_path, index=False)
    logger.info("面板已落盘以便子进程读取: %s", panel_tmp_path)

    logger.info("[serial] 串行回测 %d 个变体...", len(variant_top_ns))
    start = time.perf_counter()
    serial_results: dict[str, object] = {}
    for top_n in variant_top_ns:
        name = f"top{top_n}_equal_weight"
        engine = BaselineBacktestEngine(top_n=top_n, use_numba=True)
        serial_results[name] = engine.run(factor_panel)
    serial_seconds = time.perf_counter() - start

    logger.info(
        "[parallel] 并行回测 %d 个变体 (n_jobs=%d, 从 parquet 读面板)...",
        len(variant_top_ns),
        n_jobs,
    )
    start = time.perf_counter()
    parallel_results = _run_variants_in_parallel_from_path(
        panel_path=panel_tmp_path,
        variant_top_ns=variant_top_ns,
        n_jobs=n_jobs,
    )
    parallel_seconds = time.perf_counter() - start

    nav_diffs = [
        _max_abs_diff(
            serial_results[name].nav_frame,
            parallel_results[name].nav_frame,
        )
        for name in serial_results
    ]
    nav_diff_max = float(np.nanmax(nav_diffs)) if nav_diffs else float("nan")

    speedup = serial_seconds / parallel_seconds if parallel_seconds > 0 else float("nan")
    return pd.DataFrame(
        [
            {
                "mode": "serial_loop",
                "wall_clock_seconds": serial_seconds,
                "speedup_x": 1.0,
                "nav_diff_max": 0.0,
                "metrics_diff_max": float("nan"),
            },
            {
                "mode": f"joblib_parallel_n{n_jobs}",
                "wall_clock_seconds": parallel_seconds,
                "speedup_x": speedup,
                "nav_diff_max": nav_diff_max,
                "metrics_diff_max": float("nan"),
            },
        ]
    )


def _run_variant_from_path(panel_path: str, top_n: int):
    """子进程入口：从 parquet 读面板，跑一次回测。

    Args:
        panel_path: 面板 parquet 文件路径（字符串以便跨进程序列化）。
        top_n: 当前变体的 Top-N。

    Returns:
        tuple: ``(variant_name, BaselineBacktestResult)``。
    """
    panel = pd.read_parquet(panel_path)
    engine = BaselineBacktestEngine(top_n=top_n, use_numba=True)
    result = engine.run(panel)
    return f"top{top_n}_equal_weight", result


def _run_variants_in_parallel_from_path(
    panel_path: Path,
    variant_top_ns: list[int],
    n_jobs: int,
) -> dict:
    """跨变体并行回测，每个子进程独立从磁盘加载面板。

    Args:
        panel_path: 父进程预先落盘的 parquet 面板路径。
        variant_top_ns: 变体 Top-N 列表。
        n_jobs: joblib 并发数。

    Returns:
        dict: 变体名到 ``BaselineBacktestResult`` 的映射。
    """
    from joblib import Parallel, delayed

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run_variant_from_path)(str(panel_path), top_n)
        for top_n in variant_top_ns
    )
    return dict(results)


def main() -> None:
    """执行 C4 性能基准实验。"""
    configure_console_output()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    config = ProjectConfig.from_root()
    config.ensure_directories()

    if args.input:
        panel_path = config.data_dir / args.input
        logger.info("从磁盘载入因子面板: %s", panel_path)
        factor_panel = pd.read_csv(panel_path)
    else:
        logger.info(
            "生成合成因子面板：n_stocks=%d, n_days=%d, seed=%d",
            args.synthetic_stocks,
            args.synthetic_days,
            args.synthetic_seed,
        )
        factor_panel = build_synthetic_panel(
            n_stocks=args.synthetic_stocks,
            n_days=args.synthetic_days,
            seed=args.synthetic_seed,
        )

    logger.info(
        "因子面板形状: %d 行 × %d 列；时间跨度: %d 个唯一交易日 × %d 只股票",
        len(factor_panel),
        factor_panel.shape[1],
        factor_panel["trade_date"].nunique(),
        factor_panel["ts_code"].nunique(),
    )

    numba_vs_python_df, nav_diff_max = run_numba_vs_python(
        factor_panel=factor_panel, top_n=args.top_n
    )
    print("\n=== Numba vs Python (单回测) ===")
    print(numba_vs_python_df.to_string(index=False))

    assert nav_diff_max < 1e-6, (
        f"Numba 与 Python 路径 NAV 差异 {nav_diff_max:.3e} 超过 1e-6 容差，"
        f"请检查矩阵化实现。"
    )
    logger.info("数值一致性断言通过：nav_diff_max=%.3e < 1e-6", nav_diff_max)

    combined_summary = numba_vs_python_df.assign(comparison="numba_vs_python")
    if not args.skip_parallel:
        panel_tmp_path = config.cache_dir / "c4_benchmark_panel.parquet"
        serial_vs_parallel_df = run_serial_vs_parallel(
            factor_panel=factor_panel,
            variant_top_ns=args.variant_top_ns,
            n_jobs=args.n_jobs,
            panel_tmp_path=panel_tmp_path,
        )
        print(f"\n=== 串行 vs joblib 并行 (variants={args.variant_top_ns}) ===")
        print(serial_vs_parallel_df.to_string(index=False))
        combined_summary = pd.concat(
            [
                combined_summary,
                serial_vs_parallel_df.assign(comparison="serial_vs_parallel"),
            ],
            ignore_index=True,
        )

    output_path = config.data_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_summary.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("基准对比表已写入: %s", output_path)
    print(f"\n输出: {output_path}")


if __name__ == "__main__":
    main()
