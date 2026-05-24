"""个股财经新闻抓取客户端。

封装 akshare 的财经新闻接口（如 `ak.stock_news_em`、`ak.news_cctv`），并叠加：

- 区间过滤、去重、脏字清洗与长度截断；
- 失败重试和失败 ts_code 日志；
- 单股 parquet 缓存（`data/cache/news/{ts_code}.parquet`）。

本模块只关注数据获取，情感打分由 `src.data.llm_sentiment` 负责。
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from tqdm.auto import tqdm


_NEWS_COLUMNS = ["ts_code", "publish_time", "title", "content", "source"]


class NewsClient:
    """基于 akshare 的财经新闻客户端。

    Attributes:
        cache_dir: 单股新闻缓存目录。
        failure_log_path: 失败 ts_code 日志路径。
        retries: 单次接口调用的最大重试次数。
        retry_sleep_seconds: 重试间隔的基础秒数。
        text_max_length: 单条新闻文本长度截断阈值（字符数）。
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/cache/news",
        failure_log_path: str | Path = "logs/news_failures.log",
        retries: int = 3,
        retry_sleep_seconds: float = 1.0,
        text_max_length: int = 500,
    ) -> None:
        """初始化新闻客户端。

        Args:
            cache_dir: 缓存目录。默认 `data/cache/news`。
            failure_log_path: 失败日志路径。默认 `logs/news_failures.log`。
            retries: 每只股票最多重试次数。
            retry_sleep_seconds: 重试基础间隔，指数退避起点。
            text_max_length: 单条文本截断长度（去脏字之后）。
        """
        self.cache_dir = Path(cache_dir)
        self.failure_log_path = Path(failure_log_path)
        self.retries = retries
        self.retry_sleep_seconds = retry_sleep_seconds
        self.text_max_length = text_max_length
        self._ak: Any = None
        self._logger = logging.getLogger("news_client")

    @property
    def ak(self) -> Any:
        """延迟加载 akshare 模块。

        Returns:
            Any: akshare 模块对象。

        Raises:
            ImportError: 未安装 akshare 时抛出。
        """
        if self._ak is None:
            try:
                import akshare as ak
            except ImportError as exc:
                raise ImportError("未安装 akshare，请先在 index-enhancement 环境安装。") from exc
            self._ak = ak
        return self._ak

    def fetch_news(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """抓取单只股票在指定区间的新闻。

        Args:
            ts_code: Tushare 风格代码，例如 `000001.SZ`。
            start_date: 起始日期，支持 `YYYY-MM-DD` 或 `YYYYMMDD`。
            end_date: 结束日期。
            use_cache: 是否读取/写入本地缓存。

        Returns:
            pd.DataFrame: columns=`ts_code, publish_time, title, content, source`，
                按 publish_time 升序。
        """
        start_ts = self._normalize_date(start_date, end_of_day=False)
        end_ts = self._normalize_date(end_date, end_of_day=True)

        cache_path = self.cache_dir / f"{ts_code.replace('.', '_')}.parquet"
        cached_frame = self._read_cache(cache_path) if use_cache else None
        if cached_frame is not None and not cached_frame.empty:
            filtered = self._filter_by_window(cached_frame, start_ts, end_ts)
            if not filtered.empty:
                return filtered

        symbol = self._normalize_symbol(ts_code)
        try:
            with pd.option_context("future.infer_string", False):
                raw_frame = self._with_retry(lambda: self.ak.stock_news_em(symbol=symbol))
        except Exception as exc:
            self._log_failure(ts_code=ts_code, error=str(exc))
            return self._empty_frame()

        normalized = self._normalize_news_frame(raw_frame=raw_frame, ts_code=ts_code)
        if normalized.empty:
            return self._empty_frame()

        if use_cache:
            merged = self._merge_with_cache(cache_path=cache_path, fresh_frame=normalized)
            self._write_cache(cache_path=cache_path, frame=merged)
            normalized = merged

        return self._filter_by_window(normalized, start_ts, end_ts)

    def batch_fetch(
        self,
        ts_codes: Iterable[str],
        start_date: str,
        end_date: str,
        use_cache: bool = True,
        progress: bool = True,
    ) -> pd.DataFrame:
        """批量抓取多只股票的新闻。

        Args:
            ts_codes: 股票代码集合。
            start_date: 起始日期。
            end_date: 结束日期。
            use_cache: 是否启用缓存。
            progress: 是否显示 tqdm 进度条。

        Returns:
            pd.DataFrame: 多股聚合后的新闻表，按 (ts_code, publish_time) 排序。
        """
        codes = list(dict.fromkeys(ts_codes))
        iterator: Iterable[str] = codes
        if progress:
            iterator = tqdm(
                codes,
                desc="news",
                unit="stock",
                dynamic_ncols=True,
                ascii=True,
            )
        frames: list[pd.DataFrame] = []
        for ts_code in iterator:
            try:
                frame = self.fetch_news(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=use_cache,
                )
                if not frame.empty:
                    frames.append(frame)
            except Exception as exc:
                self._log_failure(ts_code=ts_code, error=str(exc))
        if not frames:
            return self._empty_frame()
        return (
            pd.concat(frames, ignore_index=True)
            .sort_values(["ts_code", "publish_time"])
            .reset_index(drop=True)
        )

    def _normalize_news_frame(self, raw_frame: pd.DataFrame, ts_code: str) -> pd.DataFrame:
        """将 akshare 返回的中文列表标准化为统一 schema。

        Args:
            raw_frame: akshare `stock_news_em` 返回的原始 DataFrame。
            ts_code: 当前 ts_code，用于回填字段。

        Returns:
            pd.DataFrame: columns=`ts_code, publish_time, title, content, source`，
                已去重、清洗、截断。
        """
        if raw_frame is None or raw_frame.empty:
            return self._empty_frame()

        column_map = {
            "新闻标题": "title",
            "新闻内容": "content",
            "发布时间": "publish_time",
            "文章来源": "source",
        }
        frame = raw_frame.rename(columns=column_map).copy()
        for column in ("title", "content", "publish_time", "source"):
            if column not in frame.columns:
                frame[column] = pd.NA

        frame["publish_time"] = pd.to_datetime(frame["publish_time"], errors="coerce")
        frame = frame.dropna(subset=["publish_time"]).copy()
        frame["ts_code"] = ts_code
        frame["title"] = frame["title"].fillna("").map(self._clean_text)
        frame["content"] = frame["content"].fillna("").map(self._clean_text)
        frame["source"] = frame["source"].fillna("").astype(str).str.strip()

        # title 与 content 完全相同则只保留 title 视图，节约后续 LLM token。
        same_mask = frame["title"] == frame["content"]
        frame.loc[same_mask, "content"] = ""

        frame["content"] = frame["content"].map(lambda x: self._truncate(x, self.text_max_length))
        frame["title"] = frame["title"].map(lambda x: self._truncate(x, self.text_max_length))

        frame = frame[frame["title"].str.len() + frame["content"].str.len() > 0]
        frame = frame.drop_duplicates(subset=["ts_code", "publish_time", "title"])
        return (
            frame[_NEWS_COLUMNS]
            .sort_values(["ts_code", "publish_time"])
            .reset_index(drop=True)
        )

    def _merge_with_cache(
        self,
        cache_path: Path,
        fresh_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """将新拉取数据并入既有缓存。"""
        cached_frame = self._read_cache(cache_path)
        if cached_frame is None or cached_frame.empty:
            return fresh_frame
        merged = pd.concat([cached_frame, fresh_frame], ignore_index=True)
        merged = merged.drop_duplicates(subset=["ts_code", "publish_time", "title"])
        return merged.sort_values(["ts_code", "publish_time"]).reset_index(drop=True)

    def _read_cache(self, cache_path: Path) -> pd.DataFrame | None:
        """读取 parquet 缓存；缓存不存在或损坏时返回 None。"""
        if not cache_path.exists():
            return None
        try:
            frame = pd.read_parquet(cache_path)
        except Exception as exc:
            self._logger.warning("缓存读取失败 %s: %s", cache_path, exc)
            return None
        if frame.empty:
            return frame
        frame["publish_time"] = pd.to_datetime(frame["publish_time"], errors="coerce")
        return frame.dropna(subset=["publish_time"])

    def _write_cache(self, cache_path: Path, frame: pd.DataFrame) -> None:
        """将 DataFrame 写入 parquet 缓存。"""
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            frame.to_parquet(cache_path, index=False)
        except Exception as exc:
            self._logger.warning("缓存写入失败 %s: %s", cache_path, exc)

    def _filter_by_window(
        self,
        frame: pd.DataFrame,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
    ) -> pd.DataFrame:
        """按发布时间区间过滤。"""
        if frame.empty:
            return frame
        mask = (frame["publish_time"] >= start_ts) & (frame["publish_time"] <= end_ts)
        return frame.loc[mask].reset_index(drop=True)

    def _with_retry(self, func) -> pd.DataFrame:
        """对接口调用执行指数退避重试。

        Args:
            func: 无参的数据拉取函数。

        Returns:
            pd.DataFrame: 调用结果。

        Raises:
            Exception: 重试 `self.retries` 次后仍失败时抛出最后一次异常。
        """
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                return func()
            except Exception as exc:
                last_error = exc
                if attempt == self.retries - 1:
                    break
                time.sleep(self.retry_sleep_seconds * (2 ** attempt))
        assert last_error is not None
        raise last_error

    def _log_failure(self, ts_code: str, error: str) -> None:
        """追加单只股票的失败日志。"""
        self.failure_log_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{timestamp},{ts_code},{error}\n"
        with self.failure_log_path.open("a", encoding="utf-8") as file:
            file.write(line)

    @staticmethod
    def _empty_frame() -> pd.DataFrame:
        """构建空的标准化新闻表。"""
        return pd.DataFrame(columns=_NEWS_COLUMNS)

    @staticmethod
    def _normalize_symbol(ts_code: str) -> str:
        """将 `000001.SZ` 风格转为 akshare 所需 6 位代码。"""
        return ts_code.split(".")[0]

    @staticmethod
    def _normalize_date(date_string: str, end_of_day: bool) -> pd.Timestamp:
        """统一日期字符串为 Timestamp。

        Args:
            date_string: 日期字符串，允许 `YYYY-MM-DD`、`YYYYMMDD`、`YYYY/MM/DD`。
            end_of_day: 为 True 时返回 23:59:59，便于结束日包含全天。

        Returns:
            pd.Timestamp: 解析后的时间戳。
        """
        timestamp = pd.to_datetime(date_string, errors="coerce")
        if pd.isna(timestamp):
            raise ValueError(f"无法解析日期: {date_string}")
        if end_of_day:
            return timestamp.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
        return timestamp.normalize()

    @staticmethod
    def _clean_text(value: Any) -> str:
        """去除新闻文本中的脏字符与多余空白。"""
        if value is None:
            return ""
        text = str(value)
        text = text.replace("\u3000", " ").replace("\xa0", " ")
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    @staticmethod
    def _truncate(text: str, max_length: int) -> str:
        """按字符数截断长文本。"""
        if not text:
            return ""
        if len(text) <= max_length:
            return text
        return text[:max_length].rstrip()
