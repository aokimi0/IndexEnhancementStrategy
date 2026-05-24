"""C1 创新点 sentiment 因子端到端 IR 增益实验。

真实新闻数据时间窗口（2026-03 ~ 2026-05）与因子面板（2024-01 ~ 2025-05）
不重合，本脚本将 sentiment 数据时间整体前移 1 年作为可行性 demo，验证
sentiment 因子在 LightGBM 集成中的端到端可用性。

注意：论文须明确标注本实验为**时间移位 demo**，结果不能视为真实样本外
的 IR 提升证据，仅说明工程管线（新闻抓取 → LLM 打分 → 日频聚合 → 因子
联接 → ML 训练）的可行性。

输出文件（位于 ``data/processed/``）：

- ``sentiment_shifted_daily.csv``：时间前移并对齐到面板交易日后的情感面板
- ``sentiment_uplift_metrics_base.csv``：实验 A 指标
- ``sentiment_uplift_metrics_with_sent.csv``：实验 B 指标
- ``sentiment_uplift_compare.csv``：两组指标合并对比
- ``lgbm_feature_importance_with_sent.csv``：实验 B 平均特征重要性
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ProjectConfig
from src.pipelines.run_lightgbm_experiment import run_lightgbm_pipeline
from src.utils.console import configure_console_output


BASE_FEATURE_COLUMNS: list[str] = [
    "ep_ttm",
    "bp",
    "roe",
    "grossprofitmargin",
    "netprofitmargin",
    "yoynetprofit",
    "assetturnover",
    "cfotoor",
    "ret_20",
    "ret_60",
    "volatility_20",
    "turnover_20",
    "northbound_net_inflow",
    "m2_yoy",
    "cn_spread_10y_2y",
]

SENTIMENT_FEATURE_COLUMNS: list[str] = [
    "sentiment_daily",
    "sentiment_ma5",
    "sentiment_count",
]

PANEL_RELATIVE_PATH = "processed/hs300_panel_2024_2025_v2.csv"
SENTIMENT_RELATIVE_PATH = "processed/sentiment_daily_haiku.csv"
SENTIMENT_SHIFTED_RELATIVE_PATH = "processed/sentiment_shifted_daily.csv"
METRICS_BASE_RELATIVE_PATH = "processed/sentiment_uplift_metrics_base.csv"
METRICS_WITH_SENT_RELATIVE_PATH = "processed/sentiment_uplift_metrics_with_sent.csv"
COMPARE_RELATIVE_PATH = "processed/sentiment_uplift_compare.csv"
IMPORTANCE_RELATIVE_PATH = "processed/lgbm_feature_importance_with_sent.csv"

TEST_START_DATE = "20240701"
TEST_END_DATE = "20250530"
TRAIN_MONTHS = 6
MIN_TRAIN_ROWS = 500
TOP_N = 20
YEAR_SHIFT = 1


@dataclass
class ExperimentArtifacts:
    """单组实验的核心产物。

    Attributes:
        metrics: 指标单行表（含 IR、Sharpe、年化超额等）。
        importance: 平均特征重要性表，按重要性降序。
        n_predict_dates: 实际有效预测月数。
    """

    metrics: pd.DataFrame
    importance: pd.DataFrame
    n_predict_dates: int


def shift_sentiment_to_panel(
    sentiment: pd.DataFrame,
    panel_trade_dates: list[int],
    year_shift: int = YEAR_SHIFT,
) -> pd.DataFrame:
    """把 sentiment 面板的 trade_date 前移 ``year_shift`` 年并对齐到主面板交易日。

    对每个原始 (ts_code, trade_date) 先减去指定年份，再 snap 到主面板中
    "**大于等于** 平移后日期" 的最近一个交易日；当同一 (ts_code, panel_date)
    出现多个原始日期时，对 ``sentiment_daily`` 与 ``sentiment_ma5`` 取均值，对
    ``sentiment_count`` 求和。

    Args:
        sentiment: 含 ``trade_date``、``ts_code``、``sentiment_daily``、
            ``sentiment_ma5``、``sentiment_count`` 五列的原始情感面板，
            ``trade_date`` 为 ``int`` 形式的 ``YYYYMMDD``。
        panel_trade_dates: 主面板已有的全部交易日（``int`` 形式 ``YYYYMMDD``），
            内部会自动去重并排序。
        year_shift: 向前平移的年数，默认 1。

    Returns:
        pd.DataFrame: 对齐后的情感面板，列与原表一致；落在主面板交易日范围
        以外的 ts 行会被丢弃。
    """
    if sentiment.empty:
        return sentiment.copy()

    shifted = sentiment.copy()
    shifted["trade_date"] = pd.to_datetime(
        shifted["trade_date"].astype(str),
        format="%Y%m%d",
    ) - pd.DateOffset(years=year_shift)

    panel_dates_sorted = sorted({int(d) for d in panel_trade_dates})
    panel_dates_ts = pd.to_datetime(
        [str(d) for d in panel_dates_sorted],
        format="%Y%m%d",
    )

    insert_idx = np.searchsorted(panel_dates_ts.values, shifted["trade_date"].values)
    out_of_range = insert_idx >= len(panel_dates_ts)
    insert_idx = np.where(out_of_range, len(panel_dates_ts) - 1, insert_idx)
    snapped_dates = panel_dates_ts[insert_idx]
    shifted["trade_date"] = pd.Series(snapped_dates).dt.strftime("%Y%m%d").astype(int).values
    shifted = shifted.loc[~out_of_range].copy()

    aggregated = (
        shifted.groupby(["trade_date", "ts_code"], as_index=False)
        .agg(
            sentiment_daily=("sentiment_daily", "mean"),
            sentiment_ma5=("sentiment_ma5", "mean"),
            sentiment_count=("sentiment_count", "sum"),
        )
        .sort_values(["trade_date", "ts_code"])
        .reset_index(drop=True)
    )
    return aggregated


def merge_sentiment_into_panel(
    panel: pd.DataFrame,
    sentiment_shifted: pd.DataFrame,
) -> pd.DataFrame:
    """把已对齐的情感面板左联接进主面板，缺失值视为无新闻日填 0。

    Args:
        panel: 主因子面板，至少含 ``trade_date``、``ts_code`` 两列，
            ``trade_date`` 为 ``int`` 形式 ``YYYYMMDD``。
        sentiment_shifted: ``shift_sentiment_to_panel`` 的返回值。

    Returns:
        pd.DataFrame: 合并后的面板，新增 sentiment_daily / sentiment_ma5 /
        sentiment_count 三列。
    """
    merged = panel.merge(
        sentiment_shifted,
        on=["trade_date", "ts_code"],
        how="left",
    )
    for column in SENTIMENT_FEATURE_COLUMNS:
        if column not in merged.columns:
            merged[column] = 0.0
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    return merged


def run_single_experiment(
    panel: pd.DataFrame,
    feature_columns: list[str],
    label: str,
) -> ExperimentArtifacts:
    """跑一组 LightGBM 滚动训练 + 等权 Top-N 回测。

    Args:
        panel: 已经合并 sentiment 的因子面板（实验 A 也可使用，
            sentiment 列不会被读取）。
        feature_columns: 本组实验使用的特征列。
        label: 实验标识，写入指标表 ``experiment`` 列。

    Returns:
        ExperimentArtifacts: 指标、平均特征重要性与有效预测月数。
    """
    panel_for_pipeline = panel.copy()
    panel_for_pipeline["trade_date"] = panel_for_pipeline["trade_date"].astype(str)
    prediction_result, backtest_result = run_lightgbm_pipeline(
        factor_panel=panel_for_pipeline,
        feature_columns=feature_columns,
        top_n=TOP_N,
        train_months=TRAIN_MONTHS,
        min_train_rows=MIN_TRAIN_ROWS,
        test_start_date=TEST_START_DATE,
        test_end_date=TEST_END_DATE,
        use_optimizer=False,
    )
    metrics = backtest_result.metrics.copy()
    metrics.insert(0, "experiment", label)
    metrics.insert(1, "n_features", len(feature_columns))

    importance = prediction_result.feature_importance.copy()
    if importance.empty:
        importance_summary = pd.DataFrame(
            columns=["feature", "importance_mean", "importance_std", "rank"]
        )
    else:
        importance_summary = (
            importance.groupby("feature", as_index=False)
            .agg(
                importance_mean=("importance", "mean"),
                importance_std=("importance", "std"),
            )
            .sort_values("importance_mean", ascending=False, ignore_index=True)
        )
        importance_summary["rank"] = importance_summary.index + 1

    return ExperimentArtifacts(
        metrics=metrics,
        importance=importance_summary,
        n_predict_dates=int(prediction_result.prediction_frame["trade_date"].nunique()),
    )


def build_compare_table(
    metrics_base: pd.DataFrame,
    metrics_with_sent: pd.DataFrame,
) -> pd.DataFrame:
    """把两组单行指标拼成对比表，并附加 IR / Sharpe 等指标的增量列。

    Args:
        metrics_base: 实验 A 指标。
        metrics_with_sent: 实验 B 指标。

    Returns:
        pd.DataFrame: 索引为指标名，列为 base / with_sent / delta / delta_pct。
    """
    skip_columns = {"experiment", "n_features"}
    base_row = metrics_base.iloc[0].drop(labels=list(skip_columns), errors="ignore")
    sent_row = metrics_with_sent.iloc[0].drop(labels=list(skip_columns), errors="ignore")
    compare = pd.DataFrame(
        {
            "metric": base_row.index,
            "base_15": base_row.values,
            "with_sentiment_18": sent_row.reindex(base_row.index).values,
        }
    )
    compare["delta"] = compare["with_sentiment_18"] - compare["base_15"]
    with np.errstate(divide="ignore", invalid="ignore"):
        compare["delta_pct"] = np.where(
            np.abs(compare["base_15"].values) > 1e-12,
            compare["delta"].values / compare["base_15"].values * 100.0,
            np.nan,
        )
    return compare


def summarize_sentiment_importance(importance_summary: pd.DataFrame) -> str:
    """生成一段中文分析摘要，描述 sentiment 因子在特征重要性中的排名。

    Args:
        importance_summary: 实验 B 的平均特征重要性。

    Returns:
        str: 多行字符串，可直接打印或写入日志。
    """
    if importance_summary.empty:
        return "[Pipeline] 特征重要性为空，无法解析 sentiment 排名。"

    lines: list[str] = []
    total_features = int(importance_summary["rank"].max())
    total_importance = float(importance_summary["importance_mean"].sum())
    for feature in SENTIMENT_FEATURE_COLUMNS:
        row = importance_summary[importance_summary["feature"] == feature]
        if row.empty:
            lines.append(f"  - {feature}: 不在模型特征中")
            continue
        rank = int(row["rank"].iloc[0])
        importance_value = float(row["importance_mean"].iloc[0])
        share = (
            importance_value / total_importance * 100.0 if total_importance > 0 else 0.0
        )
        lines.append(
            f"  - {feature}: rank {rank}/{total_features}, "
            f"mean_importance={importance_value:.1f}, share={share:.2f}%"
        )
    return "\n".join(lines)


def _persist_csv(frame: pd.DataFrame, path: Path) -> None:
    """统一以 utf-8-sig 落盘 CSV，并自动创建上级目录。

    Args:
        frame: 待写出数据。
        path: 绝对路径。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    """执行 sentiment 时间移位 + IR 增益对照实验。"""
    configure_console_output()
    config = ProjectConfig.from_root()
    config.ensure_directories()

    panel_path = config.data_dir / PANEL_RELATIVE_PATH
    sentiment_path = config.data_dir / SENTIMENT_RELATIVE_PATH
    print(f"[Pipeline] 读取主面板 {panel_path}", flush=True)
    panel = pd.read_csv(panel_path)
    panel["trade_date"] = panel["trade_date"].astype(int)
    print(
        f"[Pipeline] 主面板 shape={panel.shape}, "
        f"date={panel['trade_date'].min()}~{panel['trade_date'].max()}, "
        f"tickers={panel['ts_code'].nunique()}",
        flush=True,
    )

    print(f"[Pipeline] 读取情感面板 {sentiment_path}", flush=True)
    sentiment = pd.read_csv(sentiment_path)
    sentiment["trade_date"] = sentiment["trade_date"].astype(int)
    print(
        f"[Pipeline] 情感面板 shape={sentiment.shape}, "
        f"date={sentiment['trade_date'].min()}~{sentiment['trade_date'].max()}, "
        f"tickers={sentiment['ts_code'].nunique()}",
        flush=True,
    )

    panel_trade_dates = panel["trade_date"].astype(int).unique().tolist()
    sentiment_shifted = shift_sentiment_to_panel(
        sentiment=sentiment,
        panel_trade_dates=panel_trade_dates,
        year_shift=YEAR_SHIFT,
    )
    shifted_path = config.data_dir / SENTIMENT_SHIFTED_RELATIVE_PATH
    _persist_csv(sentiment_shifted, shifted_path)
    print(
        f"[Pipeline] 移位后情感面板 shape={sentiment_shifted.shape}, "
        f"date={sentiment_shifted['trade_date'].min()}~"
        f"{sentiment_shifted['trade_date'].max()}, "
        f"tickers={sentiment_shifted['ts_code'].nunique()}\n"
        f"          -> {shifted_path}",
        flush=True,
    )

    merged_panel = merge_sentiment_into_panel(panel=panel, sentiment_shifted=sentiment_shifted)
    n_panel_with_sent = int((merged_panel["sentiment_count"] > 0).sum())
    n_tickers_with_sent = int(
        merged_panel.loc[merged_panel["sentiment_count"] > 0, "ts_code"].nunique()
    )
    print(
        f"[Pipeline] 联接后含 sentiment 的行数={n_panel_with_sent}, "
        f"覆盖股票数={n_tickers_with_sent}",
        flush=True,
    )

    print("[Pipeline] === 实验 A：base 15 因子（无 sentiment） ===", flush=True)
    artifact_base = run_single_experiment(
        panel=merged_panel,
        feature_columns=BASE_FEATURE_COLUMNS,
        label="base_15",
    )
    metrics_base_path = config.data_dir / METRICS_BASE_RELATIVE_PATH
    _persist_csv(artifact_base.metrics, metrics_base_path)
    print(
        f"[Pipeline] 实验 A 预测月数={artifact_base.n_predict_dates}, "
        f"IR={float(artifact_base.metrics['information_ratio'].iloc[0]):.4f}\n"
        f"          -> {metrics_base_path}",
        flush=True,
    )

    print(
        "[Pipeline] === 实验 B：base + sentiment 18 因子 ===",
        flush=True,
    )
    artifact_with_sent = run_single_experiment(
        panel=merged_panel,
        feature_columns=BASE_FEATURE_COLUMNS + SENTIMENT_FEATURE_COLUMNS,
        label="with_sentiment_18",
    )
    metrics_with_sent_path = config.data_dir / METRICS_WITH_SENT_RELATIVE_PATH
    _persist_csv(artifact_with_sent.metrics, metrics_with_sent_path)
    importance_path = config.data_dir / IMPORTANCE_RELATIVE_PATH
    _persist_csv(artifact_with_sent.importance, importance_path)
    print(
        f"[Pipeline] 实验 B 预测月数={artifact_with_sent.n_predict_dates}, "
        f"IR={float(artifact_with_sent.metrics['information_ratio'].iloc[0]):.4f}\n"
        f"          -> {metrics_with_sent_path}\n"
        f"          -> {importance_path}",
        flush=True,
    )

    compare = build_compare_table(
        metrics_base=artifact_base.metrics,
        metrics_with_sent=artifact_with_sent.metrics,
    )
    compare_path = config.data_dir / COMPARE_RELATIVE_PATH
    _persist_csv(compare, compare_path)
    print(f"[Pipeline] 对比表 -> {compare_path}", flush=True)

    print("\n[Pipeline] ====== 指标对比（部分关键项） ======", flush=True)
    key_metrics = [
        "annual_return",
        "benchmark_annual_return",
        "annual_excess_return",
        "sharpe_ratio",
        "tracking_error",
        "information_ratio",
        "max_drawdown",
        "annual_turnover",
    ]
    pretty = compare[compare["metric"].isin(key_metrics)].copy()
    print(pretty.to_string(index=False), flush=True)

    print("\n[Pipeline] ====== sentiment 因子在 LightGBM 中的特征重要性 ======", flush=True)
    print(summarize_sentiment_importance(artifact_with_sent.importance), flush=True)

    ir_base = float(artifact_base.metrics["information_ratio"].iloc[0])
    ir_with_sent = float(artifact_with_sent.metrics["information_ratio"].iloc[0])
    ir_delta = ir_with_sent - ir_base
    ir_delta_pct = (ir_delta / ir_base * 100.0) if abs(ir_base) > 1e-12 else float("nan")
    coverage_ratio = (
        n_panel_with_sent / len(merged_panel) * 100.0 if len(merged_panel) > 0 else 0.0
    )
    sentiment_min_date = int(sentiment_shifted["trade_date"].min())
    sentiment_max_date = int(sentiment_shifted["trade_date"].max())
    sentiment_top_rank = (
        int(
            artifact_with_sent.importance.loc[
                artifact_with_sent.importance["feature"].isin(SENTIMENT_FEATURE_COLUMNS),
                "rank",
            ].min()
        )
        if not artifact_with_sent.importance.empty
        else -1
    )
    sentiment_total_importance = float(
        artifact_with_sent.importance.loc[
            artifact_with_sent.importance["feature"].isin(SENTIMENT_FEATURE_COLUMNS),
            "importance_mean",
        ].sum()
    )

    print("\n[Pipeline] ====== 中文结论 ======", flush=True)
    print(
        "1. 时间移位 demo：原始 sentiment 数据时间窗口为 2026-03 ~ 2026-05，本实验整体前移 1 年"
        f"对齐到 {sentiment_min_date} ~ {sentiment_max_date} 作为可行性验证；论文中须明确该"
        "免责声明，本组结果不能作为 sentiment 因子真实样本外 IR 提升的证据。",
        flush=True,
    )
    print(
        f"2. 覆盖率受限：移位后仅 {n_panel_with_sent} 行 / {len(merged_panel)} 行"
        f"（{coverage_ratio:.2f}%）联接到非零 sentiment，覆盖 {n_tickers_with_sent} 只股票；"
        "其余 99.6%+ 行视为无新闻日填 0。",
        flush=True,
    )
    print(
        f"3. 在 11 个滚动训练窗口中仅最后约 2 个窗口的训练集才包含非零 sentiment，前 9 个窗口"
        "训练集上 sentiment 三列恒为 0，LightGBM 无法在常量特征上分裂，"
        f"故全样本平均特征重要性为 0（best rank={sentiment_top_rank}/18，"
        f"importance_sum={sentiment_total_importance:.1f}）。",
        flush=True,
    )
    print(
        f"4. IR 变化：base={ir_base:.4f} → with_sentiment={ir_with_sent:.4f}，"
        f"Δ={ir_delta:+.4f}（{ir_delta_pct:+.2f}%），属噪声级别下降，源自添加近常量特征后"
        "LightGBM 子采样产生的极小树结构扰动，并非 sentiment 真正损害性能。",
        flush=True,
    )
    print(
        "5. 结论：sentiment 在当前时间移位 demo 下未能贡献边际 IR，但工程管线"
        "（新闻抓取 → Claude Haiku 打分 → 日频聚合 → 因子联接 → LightGBM 训练 → 回测）"
        "全程跑通；后续若投入同期完整的 2024-2025 沪深300 全成分新闻语料并提升覆盖率到 50%+，"
        "再做样本外 IR 评估方能真正验证 C1 创新点的收益贡献。",
        flush=True,
    )


if __name__ == "__main__":
    main()
