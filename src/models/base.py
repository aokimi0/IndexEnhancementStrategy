"""alpha 模型基础抽象与共用工具。

模块提供：
    * :class:`PredictionResult`：与 ``LightgbmPredictionResult`` 字段一致的通用结果容器。
    * :class:`AlphaModelBase`：抽象基类，统一封装滚动训练与冻结训练的循环逻辑。
    * 若干工具函数（:func:`prepare_panel`、:func:`iter_month_ends` 等），从既有的
      :mod:`src.models.lightgbm_model` 中提炼，方便各 L1 模型复用，并避免在新代码中
      重复实现日期对齐与月末切片。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class PredictionResult:
    """通用的 alpha 模型预测结果容器。

    Attributes:
        prediction_frame: 列为 ``[trade_date, ts_code, label_column, ml_score]`` 的预测表。
        feature_importance: 列为 ``[trade_date, feature, importance]`` 的特征重要性表。
    """

    prediction_frame: pd.DataFrame
    feature_importance: pd.DataFrame


def prepare_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """规范化因子面板：解析 ``trade_date`` 为 ``datetime`` 并按 (日期, 股票) 排序。

    Args:
        panel: 输入面板，``trade_date`` 列可为 ``YYYYMMDD`` 字符串、整数或 ``datetime``。

    Returns:
        pd.DataFrame: 重置过索引、按时间和股票排序的拷贝；索引为连续整数。
    """
    frame = panel.copy()
    frame["trade_date"] = pd.to_datetime(
        frame["trade_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    return frame.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def iter_month_ends(panel: pd.DataFrame) -> list[pd.Timestamp]:
    """提取面板中每个自然月最后一个交易日的列表。

    Args:
        panel: 已经过 :func:`prepare_panel` 规范化的面板。

    Returns:
        list[pd.Timestamp]: 按时间升序排列的月末日列表，可能为空。
    """
    return (
        panel[["trade_date"]]
        .drop_duplicates()
        .assign(month=lambda df: df["trade_date"].dt.to_period("M"))
        .groupby("month")["trade_date"]
        .max()
        .tolist()
    )


def build_train_test_slices(
    frame: pd.DataFrame,
    month_ends: list[pd.Timestamp],
    train_months: int,
    label_column: str,
    min_train_rows: int,
) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]]:
    """按滚动窗口枚举 (训练集, 预测集, 预测日) 三元组。

    Args:
        frame: 规范化后的面板。
        month_ends: 月末日列表。
        train_months: 滚动训练窗口月数。
        label_column: 标签列名，用于剔除 NaN 训练样本。
        min_train_rows: 最小训练样本数；不足时该窗口将被跳过。

    Returns:
        list[tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]]: 每个元素为
            ``(train_frame, predict_frame, predict_date)``。当训练样本不足或预测日无截面时跳过。
    """
    windows: list[tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]] = []
    for idx in range(train_months, len(month_ends)):
        predict_date = month_ends[idx]
        train_start = month_ends[idx - train_months]
        train_frame = frame[
            (frame["trade_date"] >= train_start) & (frame["trade_date"] < predict_date)
        ].dropna(subset=[label_column])
        if len(train_frame) < min_train_rows:
            continue
        predict_frame = frame[frame["trade_date"] == predict_date]
        if predict_frame.empty:
            continue
        windows.append((train_frame, predict_frame, predict_date))
    return windows


def empty_prediction_frame(label_column: str) -> pd.DataFrame:
    """构造列结构为预测帧标准格式的空表。

    Args:
        label_column: 标签列名。

    Returns:
        pd.DataFrame: 仅定义列名的空 DataFrame。
    """
    return pd.DataFrame(columns=["trade_date", "ts_code", label_column, "ml_score"])


def empty_importance_frame() -> pd.DataFrame:
    """构造列结构为重要性帧标准格式的空表。

    Returns:
        pd.DataFrame: 仅定义列名的空 DataFrame。
    """
    return pd.DataFrame(columns=["trade_date", "feature", "importance"])


def format_trade_date_string(frame: pd.DataFrame) -> pd.DataFrame:
    """将 ``trade_date`` 列由 ``datetime`` 转换为 ``YYYYMMDD`` 字符串。

    Args:
        frame: 含 ``trade_date`` 列的数据表。若为空或非时间类型则原样返回。

    Returns:
        pd.DataFrame: 同一对象（可能就地修改）的引用。
    """
    if frame.empty or "trade_date" not in frame.columns:
        return frame
    if pd.api.types.is_datetime64_any_dtype(frame["trade_date"]):
        frame["trade_date"] = frame["trade_date"].dt.strftime("%Y%m%d")
    return frame


class AlphaModelBase(ABC):
    """alpha 预测模型抽象基类，统一封装滚动与冻结两种训练接口。"""

    def __init__(
        self,
        feature_columns: list[str],
        label_column: str = "label_excess_return_20d",
        train_months: int = 12,
        min_train_rows: int = 2000,
    ) -> None:
        """初始化基础参数。

        Args:
            feature_columns: 模型使用的特征列名列表。
            label_column: 监督学习标签列名。
            train_months: 滚动训练窗口月数。
            min_train_rows: 单次训练所需最小样本数，不足则跳过当前窗口。
        """
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.train_months = train_months
        self.min_train_rows = min_train_rows

    def prepare(self, panel: pd.DataFrame) -> None:
        """对全量面板执行一次性预处理，默认无操作。

        子类（如 :class:`GRUAlphaModel`）可在此构建特征序列等代价较高的缓存。

        Args:
            panel: 已经过 :func:`prepare_panel` 规范化的面板。
        """
        return None

    @abstractmethod
    def fit_predict_batch(
        self,
        train_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        compute_importance: bool = True,
    ) -> dict[str, Any]:
        """单次训练并对测试集打分，子类必须实现。

        Args:
            train_frame: 训练子集，需含 ``trade_date``、``ts_code``、特征列与标签列。
            test_frame: 测试子集，需含 ``trade_date``、``ts_code`` 与特征列。
            compute_importance: 是否需要计算特征重要性。对于树模型一般免费返回；
                对于代价较高的实现（如 GRU 的 permutation importance）可在为 ``False`` 时跳过。

        Returns:
            dict[str, Any]: 至少含
                * ``predictions``: 与 ``test_frame`` 行序一致的 ``numpy.ndarray``。
                * ``importance``: ``dict[str, float]``，从特征名到重要性的映射。
        """

    def fit_predict(self, panel: pd.DataFrame) -> PredictionResult:
        """按月末滚动训练并预测。

        Args:
            panel: 因子面板。

        Returns:
            PredictionResult: 预测结果与按月特征重要性。
        """
        frame = prepare_panel(panel)
        self.prepare(frame)
        month_ends = iter_month_ends(frame)
        windows = build_train_test_slices(
            frame=frame,
            month_ends=month_ends,
            train_months=self.train_months,
            label_column=self.label_column,
            min_train_rows=self.min_train_rows,
        )
        prediction_frames: list[pd.DataFrame] = []
        importance_frames: list[pd.DataFrame] = []
        for train_frame, predict_frame, predict_date in windows:
            batch_result = self.fit_predict_batch(
                train_frame=train_frame,
                test_frame=predict_frame,
                compute_importance=True,
            )
            month_predict = predict_frame[
                ["trade_date", "ts_code", self.label_column]
            ].copy()
            month_predict["ml_score"] = batch_result["predictions"]
            prediction_frames.append(month_predict)
            importance_frames.append(
                self._build_importance_frame(predict_date, batch_result["importance"])
            )
        return self._build_prediction_result(prediction_frames, importance_frames)

    def fit_predict_frozen(
        self,
        panel: pd.DataFrame,
        train_end_date: str,
        test_start_date: str,
        test_end_date: str | None = None,
    ) -> PredictionResult:
        """冻结训练：训练一次，对测试区间内的每个月末做推断。

        Args:
            panel: 因子面板。
            train_end_date: 训练集截止日，格式 ``YYYYMMDD``。
            test_start_date: 测试集开始日，格式 ``YYYYMMDD``。
            test_end_date: 测试集结束日，格式 ``YYYYMMDD``；为 ``None`` 时取面板末日。

        Returns:
            PredictionResult: 预测结果与特征重要性（每个月末重复同一组重要性）。
        """
        frame = prepare_panel(panel)
        self.prepare(frame)
        train_end_ts = pd.to_datetime(train_end_date, format="%Y%m%d")
        test_start_ts = pd.to_datetime(test_start_date, format="%Y%m%d")
        test_end_ts = (
            pd.to_datetime(test_end_date, format="%Y%m%d")
            if test_end_date
            else frame["trade_date"].max()
        )
        train_frame = frame[frame["trade_date"] <= train_end_ts].dropna(
            subset=[self.label_column]
        )
        if len(train_frame) < self.min_train_rows:
            return PredictionResult(
                prediction_frame=empty_prediction_frame(self.label_column),
                feature_importance=empty_importance_frame(),
            )
        test_window = frame[
            (frame["trade_date"] >= test_start_ts) & (frame["trade_date"] <= test_end_ts)
        ]
        month_ends = iter_month_ends(test_window)
        if not month_ends:
            return PredictionResult(
                prediction_frame=empty_prediction_frame(self.label_column),
                feature_importance=empty_importance_frame(),
            )
        combined_test = frame[frame["trade_date"].isin(month_ends)]
        if combined_test.empty:
            return PredictionResult(
                prediction_frame=empty_prediction_frame(self.label_column),
                feature_importance=empty_importance_frame(),
            )
        batch_result = self.fit_predict_batch(
            train_frame=train_frame,
            test_frame=combined_test,
            compute_importance=True,
        )
        combined_test = combined_test.copy()
        combined_test["ml_score"] = batch_result["predictions"]
        prediction_frames: list[pd.DataFrame] = []
        importance_frames: list[pd.DataFrame] = []
        for predict_date in month_ends:
            month_predict = combined_test[
                combined_test["trade_date"] == predict_date
            ][["trade_date", "ts_code", self.label_column, "ml_score"]]
            if month_predict.empty:
                continue
            prediction_frames.append(month_predict)
            importance_frames.append(
                self._build_importance_frame(predict_date, batch_result["importance"])
            )
        return self._build_prediction_result(prediction_frames, importance_frames)

    def _build_importance_frame(
        self,
        predict_date: pd.Timestamp,
        importance: dict[str, float],
    ) -> pd.DataFrame:
        """根据 (特征 -> 重要性) 字典构造单期重要性表。

        Args:
            predict_date: 当前预测日。
            importance: 特征名到重要性值的映射。

        Returns:
            pd.DataFrame: 含 ``trade_date``、``feature``、``importance`` 三列的表。
        """
        return pd.DataFrame(
            {
                "trade_date": predict_date,
                "feature": list(importance.keys()),
                "importance": list(importance.values()),
            }
        )

    def _build_prediction_result(
        self,
        prediction_frames: list[pd.DataFrame],
        importance_frames: list[pd.DataFrame],
    ) -> PredictionResult:
        """合并多窗口结果并把 ``trade_date`` 转为字符串。

        Args:
            prediction_frames: 每个窗口的预测表列表。
            importance_frames: 每个窗口的重要性表列表。

        Returns:
            PredictionResult: 合并后的预测结果。
        """
        prediction_result = (
            pd.concat(prediction_frames, ignore_index=True)
            if prediction_frames
            else empty_prediction_frame(self.label_column)
        )
        importance_result = (
            pd.concat(importance_frames, ignore_index=True)
            if importance_frames
            else empty_importance_frame()
        )
        return PredictionResult(
            prediction_frame=format_trade_date_string(prediction_result),
            feature_importance=format_trade_date_string(importance_result),
        )
