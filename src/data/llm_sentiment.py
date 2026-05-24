"""基于 Claude 的中文财经舆情情感打分器。

实现要点：

- `.env` 中读取 `ANTHROPIC_API_KEY`、`ANTHROPIC_MODEL`、`ANTHROPIC_FAST_MODEL`、
  `ANTHROPIC_MAX_USD`，并初始化 `anthropic.Anthropic` 客户端；
- 单次 batch 最多 10 条新闻，输入/输出严格 JSON；
- 解析失败时退化为 `polarity=0, intensity=0` 并写 log；
- 指数退避重试 3 次；
- 累计 USD 超过预算抛 `BudgetExceededError`；
- 文本级 sha256 缓存（key 包含 model 与 prompt_version），重复打分零成本。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator
from tqdm.auto import tqdm


PROMPT_VERSION = "v1.0"
"""Prompt 模板版本号。任何文案/字段变更必须递增以避免缓存命中错误结果。"""


SUPPORTED_TOPICS: tuple[str, ...] = (
    "earnings",
    "policy",
    "risk",
    "macro",
    "industry",
    "operation",
    "other",
)


PROMPT_TEMPLATE = """你是中国 A 股市场资深分析师。请对以下 N 条新闻标题/摘要，给出对相关上市公司的情感影响评分。

输入 JSON 数组每项 {{ "id": int, "ts_code": "xxx.SH", "text": "..." }}

严格只输出 JSON 数组，每项 {{ "id": int, "polarity": -1.0~1.0, "intensity": 0~1, "topic": "earnings/policy/risk/macro/industry/operation/other", "explain": "<=20字" }}

polarity 正负代表利空利好（正=利好），intensity 代表影响强度。语义中性时 polarity=0。请勿包含除 JSON 数组以外的任何字符。

输入：
{payload}
"""


PRICE_TABLE_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-5": (5.0, 25.0),
    "claude-opus-4": (15.0, 75.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-haiku-4": (0.8, 4.0),
}
"""不同模型的 (输入价, 输出价) 美元/百万 token。

用于本地预算估算；最终结果以 anthropic 返回的 token 数为准。
当模型名不在表中时，按 opus-4-7 估价以避免低估。
"""


class BudgetExceededError(RuntimeError):
    """预算超限异常。

    当本会话累计成本超过 `ANTHROPIC_MAX_USD` 时抛出。
    """


class _SentimentRecord(BaseModel):
    """LLM 单条情感打分结果，用于结构化校验。"""

    id: int
    polarity: float = Field(ge=-1.0, le=1.0)
    intensity: float = Field(ge=0.0, le=1.0)
    topic: str
    explain: str = ""

    @field_validator("topic")
    @classmethod
    def _normalize_topic(cls, value: str) -> str:
        """统一 topic 到既定枚举。"""
        normalized = (value or "other").strip().lower()
        return normalized if normalized in SUPPORTED_TOPICS else "other"

    @field_validator("explain")
    @classmethod
    def _truncate_explain(cls, value: str) -> str:
        """限制解释字段长度。"""
        text = (value or "").strip()
        return text[:40]


@dataclass
class ScoringRequest:
    """单条待打分文本。

    Attributes:
        id: 批次内连续编号，用于回填。
        ts_code: 关联股票代码。
        text: 文本（标题 + 摘要）。
        publish_time: 发布时间，用于聚合时回写。
    """

    id: int
    ts_code: str
    text: str
    publish_time: pd.Timestamp | None = None


@dataclass
class BudgetTracker:
    """本进程预算累计器。

    Attributes:
        budget_usd: 预算上限。
        spent_usd: 累计已花费。
        input_tokens: 累计输入 token。
        output_tokens: 累计输出 token。
        calls: 实际调用次数（不含缓存命中）。
        cache_hits: 缓存命中条数。
    """

    budget_usd: float
    spent_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    cache_hits: int = 0

    def add(self, input_tokens: int, output_tokens: int, model: str) -> None:
        """累计单次调用消费。

        Args:
            input_tokens: 输入 token 数。
            output_tokens: 输出 token 数。
            model: 模型名。
        """
        price_in, price_out = PRICE_TABLE_USD_PER_MTOK.get(
            model,
            PRICE_TABLE_USD_PER_MTOK["claude-opus-4-7"],
        )
        cost = input_tokens / 1_000_000 * price_in + output_tokens / 1_000_000 * price_out
        self.spent_usd += cost
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.calls += 1

    def assert_under_budget(self) -> None:
        """超额时抛 BudgetExceededError。"""
        if self.spent_usd >= self.budget_usd:
            raise BudgetExceededError(
                f"累计花费 {self.spent_usd:.4f} USD 已达预算 {self.budget_usd:.2f} USD"
            )

    def snapshot(self) -> dict[str, Any]:
        """返回当前用量快照，便于日志/CLI 展示。"""
        return {
            "spent_usd": round(self.spent_usd, 6),
            "budget_usd": self.budget_usd,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "calls": self.calls,
            "cache_hits": self.cache_hits,
        }


class ClaudeSentimentScorer:
    """Claude 中文舆情情感打分器。

    Attributes:
        model: 当前默认模型名。
        fast_model: 便宜路线模型名（haiku）。
        cache_dir: 缓存目录。
        tracker: 预算追踪器。
        prompt_version: prompt 版本号。
    """

    def __init__(
        self,
        env_path: str | Path | None = None,
        cache_dir: str | Path = "data/cache/llm_sentiment",
        retries: int = 3,
        retry_sleep_seconds: float = 2.0,
        budget_usd: float | None = None,
    ) -> None:
        """初始化打分器。

        Args:
            env_path: `.env` 文件路径，默认从项目根目录读取。
            cache_dir: 缓存目录。
            retries: 单次调用最大重试次数。
            retry_sleep_seconds: 重试基础间隔，指数退避起点。
            budget_usd: 显式预算覆盖，留空时使用 `ANTHROPIC_MAX_USD`。
        """
        self._load_env(env_path)
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError("未安装 anthropic，请先在 index-enhancement 环境安装。") from exc
        self._client = Anthropic()
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7")
        self.fast_model = os.getenv("ANTHROPIC_FAST_MODEL", "claude-haiku-4-5")
        env_budget = float(os.getenv("ANTHROPIC_MAX_USD", "5"))
        self.tracker = BudgetTracker(budget_usd=budget_usd if budget_usd is not None else env_budget)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.retries = retries
        self.retry_sleep_seconds = retry_sleep_seconds
        self.prompt_version = PROMPT_VERSION
        self._logger = logging.getLogger("llm_sentiment")

    def score(
        self,
        records: list[ScoringRequest],
        use_fast: bool = False,
        max_tokens: int = 1024,
    ) -> list[dict[str, Any]]:
        """对一个批次的文本打分。

        Args:
            records: 待打分文本批次。
            use_fast: 是否走便宜的 fast model。
            max_tokens: LLM 响应 max_tokens 上限。

        Returns:
            list[dict]: 与 records 等长，每项 `{id, ts_code, polarity, intensity, topic, explain, source}`，
                `source` 为 `cache`、`llm` 或 `fallback`。

        Raises:
            BudgetExceededError: 当本次调用前累计花费已经超过预算。
        """
        if not records:
            return []
        if len(records) > 10:
            raise ValueError("单批最多 10 条，请使用 score_dataframe 自动分批。")

        model = self.fast_model if use_fast else self.model
        results_by_id: dict[int, dict[str, Any]] = {}
        uncached: list[ScoringRequest] = []
        for record in records:
            cached = self._read_cache(record.text, model=model)
            if cached is not None:
                self.tracker.cache_hits += 1
                results_by_id[record.id] = {
                    "id": record.id,
                    "ts_code": record.ts_code,
                    "polarity": cached["polarity"],
                    "intensity": cached["intensity"],
                    "topic": cached["topic"],
                    "explain": cached.get("explain", ""),
                    "source": "cache",
                }
            else:
                uncached.append(record)

        if uncached:
            self.tracker.assert_under_budget()
            fresh = self._call_llm(records=uncached, model=model, max_tokens=max_tokens)
            for item in fresh:
                results_by_id[item["id"]] = item

        return [results_by_id[record.id] for record in records]

    def score_dataframe(
        self,
        df: pd.DataFrame,
        batch_size: int = 10,
        use_fast: bool = False,
        progress: bool = True,
    ) -> pd.DataFrame:
        """对整张新闻表打分。

        Args:
            df: 至少包含 `ts_code, publish_time, title, content` 的 DataFrame。
            batch_size: 单批条数，最大 10。
            use_fast: 是否使用 fast model。
            progress: 是否显示 tqdm。

        Returns:
            pd.DataFrame: 在原表上追加 `polarity, intensity, topic, explain, score_source` 列，
                按 (publish_time, ts_code) 排序。

        Raises:
            BudgetExceededError: 预算超限时立刻抛出，已完成部分仍保留。
        """
        if df.empty:
            return df.assign(polarity=[], intensity=[], topic=[], explain=[], score_source=[])

        batch_size = max(1, min(batch_size, 10))
        frame = df.copy().reset_index(drop=True)
        frame["_text"] = (
            frame.get("title", "").fillna("").astype(str).str.strip()
            + " "
            + frame.get("content", "").fillna("").astype(str).str.strip()
        ).str.strip()
        frame = frame[frame["_text"].str.len() > 0].reset_index(drop=True)
        if frame.empty:
            return df.assign(polarity=[], intensity=[], topic=[], explain=[], score_source=[])

        records: list[ScoringRequest] = [
            ScoringRequest(
                id=idx,
                ts_code=str(row["ts_code"]),
                text=str(row["_text"]),
                publish_time=row.get("publish_time"),
            )
            for idx, row in frame.iterrows()
        ]

        scored_rows: list[dict[str, Any]] = []
        batches = [records[i : i + batch_size] for i in range(0, len(records), batch_size)]
        iterator: Iterable[list[ScoringRequest]] = batches
        if progress:
            iterator = tqdm(batches, desc="llm_score", unit="batch", dynamic_ncols=True, ascii=True)

        for batch in iterator:
            try:
                scored = self.score(batch, use_fast=use_fast)
            except BudgetExceededError:
                self._logger.warning(
                    "预算超限，停止剩余批次。已完成 %d 条，快照=%s",
                    len(scored_rows),
                    self.tracker.snapshot(),
                )
                break
            scored_rows.extend(scored)

        if not scored_rows:
            return frame.drop(columns=["_text"]).assign(
                polarity=0.0,
                intensity=0.0,
                topic="other",
                explain="",
                score_source="empty",
            )

        score_frame = pd.DataFrame(scored_rows).rename(columns={"source": "score_source"})
        merged = frame.merge(
            score_frame[["id", "polarity", "intensity", "topic", "explain", "score_source"]],
            left_index=True,
            right_on="id",
            how="left",
        ).drop(columns=["id", "_text"])
        merged["polarity"] = merged["polarity"].fillna(0.0).astype(float)
        merged["intensity"] = merged["intensity"].fillna(0.0).astype(float)
        merged["topic"] = merged["topic"].fillna("other")
        merged["explain"] = merged["explain"].fillna("")
        merged["score_source"] = merged["score_source"].fillna("missing")
        return merged.sort_values(["publish_time", "ts_code"]).reset_index(drop=True)

    def _call_llm(
        self,
        records: list[ScoringRequest],
        model: str,
        max_tokens: int,
    ) -> list[dict[str, Any]]:
        """调用 Claude 完成单批打分，处理重试、缓存与失败回退。"""
        payload = [
            {"id": rec.id, "ts_code": rec.ts_code, "text": rec.text}
            for rec in records
        ]
        prompt = PROMPT_TEMPLATE.format(payload=json.dumps(payload, ensure_ascii=False))

        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                input_tokens = getattr(response.usage, "input_tokens", 0) or 0
                output_tokens = getattr(response.usage, "output_tokens", 0) or 0
                self.tracker.add(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=model,
                )

                raw_text = self._extract_text(response)
                parsed = self._parse_response(raw_text=raw_text, records=records)
                results: list[dict[str, Any]] = []
                for rec, item in zip(records, parsed):
                    result = {
                        "id": rec.id,
                        "ts_code": rec.ts_code,
                        "polarity": float(item["polarity"]),
                        "intensity": float(item["intensity"]),
                        "topic": item["topic"],
                        "explain": item.get("explain", ""),
                        "source": item.get("source", "llm"),
                    }
                    if result["source"] != "fallback":
                        self._write_cache(rec.text, model=model, payload=result)
                    results.append(result)
                return results
            except BudgetExceededError:
                raise
            except Exception as exc:
                last_error = exc
                self._logger.warning(
                    "llm 调用失败 attempt=%d/%d: %s",
                    attempt + 1,
                    self.retries,
                    exc,
                )
                if attempt < self.retries - 1:
                    time.sleep(self.retry_sleep_seconds * (2 ** attempt))

        self._logger.error("llm 调用最终失败：%s", last_error)
        return [
            {
                "id": rec.id,
                "ts_code": rec.ts_code,
                "polarity": 0.0,
                "intensity": 0.0,
                "topic": "other",
                "explain": "",
                "source": "fallback",
            }
            for rec in records
        ]

    def _parse_response(
        self,
        raw_text: str,
        records: list[ScoringRequest],
    ) -> list[dict[str, Any]]:
        """解析 LLM 返回 JSON；缺失/格式错误时使用零值兜底。"""
        json_text = self._extract_json_array(raw_text)
        parsed_items: list[Any]
        try:
            parsed_items = json.loads(json_text)
            if not isinstance(parsed_items, list):
                raise ValueError("响应不是 JSON 数组")
        except Exception as exc:
            self._logger.warning("JSON 解析失败: %s 原文=%r", exc, raw_text[:200])
            return [self._fallback_item(rec.id) for rec in records]

        normalized: dict[int, dict[str, Any]] = {}
        for item in parsed_items:
            try:
                record = _SentimentRecord.model_validate(item)
                normalized[record.id] = {
                    "polarity": record.polarity,
                    "intensity": record.intensity,
                    "topic": record.topic,
                    "explain": record.explain,
                    "source": "llm",
                }
            except ValidationError as exc:
                self._logger.warning("字段校验失败: %s, item=%s", exc, item)

        result: list[dict[str, Any]] = []
        for rec in records:
            if rec.id in normalized:
                result.append(normalized[rec.id])
            else:
                self._logger.warning("缺失 id=%s 的打分，回退零值", rec.id)
                result.append(self._fallback_item(rec.id))
        return result

    def _read_cache(self, text: str, model: str) -> dict[str, Any] | None:
        """读取单条文本的缓存。"""
        cache_path = self._cache_path(text, model=model)
        if not cache_path.exists():
            return None
        try:
            with cache_path.open("r", encoding="utf-8") as file:
                return json.load(file)
        except Exception as exc:
            self._logger.warning("缓存读取失败 %s: %s", cache_path, exc)
            return None

    def _write_cache(self, text: str, model: str, payload: dict[str, Any]) -> None:
        """写入单条文本的缓存。"""
        cache_path = self._cache_path(text, model=model)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            "polarity": payload["polarity"],
            "intensity": payload["intensity"],
            "topic": payload["topic"],
            "explain": payload.get("explain", ""),
        }
        with cache_path.open("w", encoding="utf-8") as file:
            json.dump(serializable, file, ensure_ascii=False)

    def _cache_path(self, text: str, model: str) -> Path:
        """生成单条文本的缓存路径。

        Args:
            text: 待打分文本。
            model: 模型名（写入 key 避免不同模型缓存串扰）。

        Returns:
            Path: 缓存文件路径。
        """
        digest_input = f"{model}|{self.prompt_version}|{text}".encode("utf-8")
        digest = hashlib.sha256(digest_input).hexdigest()
        return self.cache_dir / f"{digest}.json"

    @staticmethod
    def _fallback_item(record_id: int) -> dict[str, Any]:
        """兜底零值打分。"""
        return {
            "polarity": 0.0,
            "intensity": 0.0,
            "topic": "other",
            "explain": "",
            "source": "fallback",
            "id": record_id,
        }

    @staticmethod
    def _extract_text(response: Any) -> str:
        """从 anthropic response 中提取首个 text block。"""
        for block in response.content:
            if getattr(block, "type", None) == "text":
                return block.text or ""
            if hasattr(block, "text"):
                return block.text or ""
        return ""

    @staticmethod
    def _extract_json_array(raw_text: str) -> str:
        """从原文中抽出 JSON 数组段，兼容 ```json 包裹与前后说明文字。"""
        if not raw_text:
            return "[]"
        text = raw_text.strip()
        fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
        if fence_match:
            return fence_match.group(1).strip()
        bracket_match = re.search(r"\[[\s\S]*\]", text)
        if bracket_match:
            return bracket_match.group(0)
        return text

    @staticmethod
    def _load_env(env_path: str | Path | None) -> None:
        """加载 .env，缺失时不报错。"""
        try:
            from dotenv import load_dotenv
        except ImportError as exc:
            raise ImportError("未安装 python-dotenv") from exc
        candidate = Path(env_path) if env_path else Path(__file__).resolve().parents[2] / ".env"
        if candidate.exists():
            load_dotenv(candidate, override=False)


@dataclass
class _ScoringSummary:
    """便于外部脚本格式化展示的简单容器。"""

    tracker_snapshot: dict[str, Any] = field(default_factory=dict)
    rows: int = 0
