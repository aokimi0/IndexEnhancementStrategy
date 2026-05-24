"""日频舆情情感因子构建。

将 `ClaudeSentimentScorer` 的分词级打分聚合到 (trade_date, ts_code) 维度，
对齐到交易日历，并提供避免未来信息的 16:00 截断规则。

公式：
    sentiment_daily = Σ polarity_i * intensity_i * exp(-Δhours_i / 24) / N
    Δhours_i = (trade_date 16:00 锚点 - publish_time_i).hours
    若 publish_time 当天 16:00 之后，则归属到下一个交易日。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


_OUTPUT_COLUMNS = [
    "trade_date",
    "ts_code",
    "sentiment_daily",
    "sentiment_count",
]


def build_daily_sentiment_factor(
    scored_news: pd.DataFrame,
    calendar: pd.DataFrame | Iterable[str],
    cutoff_hour: int = 16,
    half_life_hours: float = 24.0,
    ma_window: int = 5,
) -> pd.DataFrame:
    """聚合个股新闻打分为日频情感因子。

    Args:
        scored_news: 至少包含 `ts_code, publish_time, polarity, intensity` 的 DataFrame。
        calendar: 交易日历，支持 DataFrame（含 `trade_date` 列）或字符串列表。
            日期可为 `YYYYMMDD` 或 `YYYY-MM-DD`。
        cutoff_hour: 当日截断时刻，缺省 16；超过此时刻的新闻顺延到下一交易日，
            避免未来信息泄露。
        half_life_hours: 时间衰减分母（注意公式 `exp(-Δhours/half_life_hours)`，
            实际半衰期为 `half_life_hours * ln 2` 小时）。
        ma_window: 低频移动平均窗口，默认 5 个交易日。

    Returns:
        pd.DataFrame: columns=`trade_date(YYYYMMDD), ts_code, sentiment_daily,
            sentiment_count, sentiment_ma{ma_window}`，按 (ts_code, trade_date) 排序。

    Raises:
        ValueError: 当传入的交易日历为空或新闻表缺少必要列时抛出。
    """
    required_columns = {"ts_code", "publish_time", "polarity", "intensity"}
    missing = required_columns - set(scored_news.columns)
    if missing:
        raise ValueError(f"scored_news 缺少列: {sorted(missing)}")

    trade_dates = _normalize_calendar(calendar)
    if trade_dates.empty:
        raise ValueError("交易日历为空，无法对齐。")

    aggregated = _aggregate_by_trade_date(
        scored_news=scored_news,
        trade_dates=trade_dates,
        cutoff_hour=cutoff_hour,
        half_life_hours=half_life_hours,
    )
    if aggregated.empty:
        empty_columns = _OUTPUT_COLUMNS + [f"sentiment_ma{ma_window}"]
        return pd.DataFrame(columns=empty_columns)

    aggregated = _attach_moving_average(aggregated, window=ma_window)
    aggregated["trade_date"] = aggregated["trade_date"].dt.strftime("%Y%m%d")
    return (
        aggregated[_OUTPUT_COLUMNS + [f"sentiment_ma{ma_window}"]]
        .sort_values(["ts_code", "trade_date"])
        .reset_index(drop=True)
    )


def _aggregate_by_trade_date(
    scored_news: pd.DataFrame,
    trade_dates: pd.DatetimeIndex,
    cutoff_hour: int,
    half_life_hours: float,
) -> pd.DataFrame:
    """按 (trade_date, ts_code) 加权聚合。"""
    frame = scored_news.copy()
    frame["publish_time"] = pd.to_datetime(frame["publish_time"], errors="coerce")
    frame = frame.dropna(subset=["publish_time"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    frame["polarity"] = pd.to_numeric(frame["polarity"], errors="coerce").fillna(0.0)
    frame["intensity"] = pd.to_numeric(frame["intensity"], errors="coerce").fillna(0.0)
    frame = frame[(frame["polarity"].abs() > 0) | (frame["intensity"] > 0) | True]

    frame["effective_trade_date"] = frame["publish_time"].apply(
        lambda ts: _effective_trade_date(
            publish_time=ts,
            trade_dates=trade_dates,
            cutoff_hour=cutoff_hour,
        )
    )
    frame = frame.dropna(subset=["effective_trade_date"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    anchor_offset = pd.Timedelta(hours=cutoff_hour)
    frame["delta_hours"] = (
        (frame["effective_trade_date"] + anchor_offset) - frame["publish_time"]
    ).dt.total_seconds() / 3600.0
    frame["delta_hours"] = frame["delta_hours"].clip(lower=0.0)
    frame["weight"] = np.exp(-frame["delta_hours"] / float(half_life_hours))
    frame["weighted_score"] = frame["polarity"] * frame["intensity"] * frame["weight"]

    grouped = frame.groupby(["effective_trade_date", "ts_code"], as_index=False).agg(
        sentiment_sum=("weighted_score", "sum"),
        sentiment_count=("weighted_score", "size"),
    )
    grouped["sentiment_daily"] = np.where(
        grouped["sentiment_count"] > 0,
        grouped["sentiment_sum"] / grouped["sentiment_count"],
        0.0,
    )
    grouped = grouped.rename(columns={"effective_trade_date": "trade_date"})
    return grouped[_OUTPUT_COLUMNS]


def _attach_moving_average(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """为每只股票在交易日序列上计算移动平均。"""
    if window <= 1:
        frame[f"sentiment_ma{window}"] = frame["sentiment_daily"]
        return frame
    frame = frame.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    ma_column = f"sentiment_ma{window}"
    frame[ma_column] = (
        frame.groupby("ts_code", group_keys=False)["sentiment_daily"]
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return frame


def _effective_trade_date(
    publish_time: pd.Timestamp,
    trade_dates: pd.DatetimeIndex,
    cutoff_hour: int,
) -> pd.Timestamp | None:
    """根据 16:00 截断规则定位新闻归属的交易日。

    Args:
        publish_time: 新闻发布时间。
        trade_dates: 升序交易日序列（已归一到 0 点）。
        cutoff_hour: 截断时刻。

    Returns:
        pd.Timestamp | None: 归属的交易日；若所有交易日都早于该新闻则返回 None。
    """
    if pd.isna(publish_time):
        return None
    candidate_day = publish_time.normalize()
    if publish_time.hour >= cutoff_hour:
        candidate_day = candidate_day + pd.Timedelta(days=1)

    position = trade_dates.searchsorted(candidate_day, side="left")
    if position >= len(trade_dates):
        return None
    return trade_dates[position]


def _normalize_calendar(calendar: pd.DataFrame | Iterable[str]) -> pd.DatetimeIndex:
    """将 DataFrame 或字符串迭代器统一为升序 DatetimeIndex。"""
    if isinstance(calendar, pd.DataFrame):
        if "trade_date" not in calendar.columns:
            raise ValueError("calendar DataFrame 必须包含 trade_date 列。")
        series = calendar["trade_date"]
    else:
        series = pd.Series(list(calendar))
    parsed = pd.to_datetime(series.astype(str), errors="coerce").dropna()
    if parsed.empty:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(sorted(parsed.dt.normalize().unique()))
