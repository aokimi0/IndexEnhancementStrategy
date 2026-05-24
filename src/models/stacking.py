"""异构轻量 L1 模型 + 元学习器的 Stacking 集成框架。

总体流程（滚动模式 :meth:`StackingAlphaModel.fit_predict` / 冻结模式
:meth:`StackingAlphaModel.fit_predict_frozen`）：

    1. 对每个调仓窗口（rolling）或一次冻结训练（frozen），先在训练子集上做 ``n_oof_splits`` 折
       KFold OOF 预测，得到 ``[n_train, n_l1]`` 的元特征矩阵。
    2. 用各 L1 模型在全量训练子集上各训练一次，得到 ``[n_test, n_l1]`` 的测试集预测矩阵。
    3. 训练元模型（``logistic_avg`` / ``linear`` / ``mlp``）拟合元特征到标签，再对测试集
       预测矩阵做最终融合。
    4. 元模型的系数（或一阶层权重和）被作为 ``feature_importance`` 输出。

由于不能修改 :mod:`src.models.lightgbm_model`，对 :class:`LightgbmAlphaModel` 采用与其默认
超参一致的内部适配函数 :func:`_lgbm_fit_predict`，避免重复绑定行为。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor

from src.models.base import (
    AlphaModelBase,
    PredictionResult,
    empty_importance_frame,
    empty_prediction_frame,
    format_trade_date_string,
    iter_month_ends,
    prepare_panel,
)
from src.models.lightgbm_model import LightgbmAlphaModel


_LGBM_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}


def _lgbm_fit_predict(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
) -> np.ndarray:
    """以 :class:`LightgbmAlphaModel` 默认超参做一次训练 + 预测。

    Args:
        train_frame: 训练子集。
        test_frame: 测试子集。
        feature_columns: 特征列。
        label_column: 标签列。

    Returns:
        np.ndarray: 长度为 ``len(test_frame)`` 的预测向量。
    """
    import lightgbm as lgb

    model = lgb.LGBMRegressor(**_LGBM_PARAMS)
    model.fit(train_frame[feature_columns], train_frame[label_column])
    return np.asarray(model.predict(test_frame[feature_columns]))


def _l1_fit_predict(
    l1_model: Any,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> np.ndarray:
    """对任意 L1 模型执行一次训练 + 预测，返回测试集预测向量。

    Args:
        l1_model: L1 模型实例，可以是 :class:`LightgbmAlphaModel` 或
            :class:`src.models.base.AlphaModelBase` 子类。
        train_frame: 训练子集。
        test_frame: 测试子集。

    Returns:
        np.ndarray: 长度为 ``len(test_frame)`` 的预测向量。

    Raises:
        TypeError: 当传入未知类型时抛出。
    """
    if isinstance(l1_model, LightgbmAlphaModel):
        return _lgbm_fit_predict(
            train_frame=train_frame,
            test_frame=test_frame,
            feature_columns=l1_model.feature_columns,
            label_column=l1_model.label_column,
        )
    if isinstance(l1_model, AlphaModelBase):
        batch_result = l1_model.fit_predict_batch(
            train_frame=train_frame,
            test_frame=test_frame,
            compute_importance=False,
        )
        return np.asarray(batch_result["predictions"])
    raise TypeError(f"不支持的 L1 模型类型: {type(l1_model)!r}")


def _model_short_name(model: Any) -> str:
    """从模型类名提取简短标识，用于元特征列命名。"""
    cls_name = type(model).__name__
    return cls_name.replace("AlphaModel", "").lower() or cls_name.lower()


class StackingAlphaModel:
    """异构 L1 + 元学习器的 Stacking 集成模型。

    与 :class:`src.models.base.AlphaModelBase` 暴露同名接口（``fit_predict`` /
    ``fit_predict_frozen``）以便上游 pipeline 复用，但本身并非 ``AlphaModelBase`` 子类：
    它在内部协调多个 L1 实例完成 OOF 与最终融合预测。
    """

    def __init__(
        self,
        l1_models: list[Any],
        feature_columns: list[str],
        label_column: str = "label_excess_return_20d",
        train_months: int = 12,
        min_train_rows: int = 2000,
        meta_model: str = "linear",
        n_oof_splits: int = 3,
        random_state: int = 42,
        n_jobs: int = 4,
    ) -> None:
        """初始化 Stacking 框架。

        Args:
            l1_models: L1 模型列表，元素可为 :class:`LightgbmAlphaModel` 或
                :class:`AlphaModelBase` 子类实例。
            feature_columns: 数据面板的特征列，用于日志与元特征命名。
            label_column: 标签列名。
            train_months: 滚动训练窗口月数。
            min_train_rows: 最小训练样本数。
            meta_model: 元学习器类型，可选 ``"linear"`` / ``"logistic_avg"`` / ``"mlp"``。
            n_oof_splits: OOF KFold 折数。
            random_state: 随机种子。
            n_jobs: joblib 并行任务数；OOF 与多 L1 训练均使用线程后端。

        Raises:
            ValueError: 当 ``meta_model`` 取值不被支持时抛出。
        """
        if meta_model not in ("linear", "logistic_avg", "mlp"):
            raise ValueError(f"未知 meta_model: {meta_model}")
        self.l1_models = list(l1_models)
        self.feature_columns = list(feature_columns)
        self.label_column = label_column
        self.train_months = train_months
        self.min_train_rows = min_train_rows
        self.meta_model = meta_model
        self.n_oof_splits = n_oof_splits
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit_predict(self, panel: pd.DataFrame) -> PredictionResult:
        """滚动 Stacking 训练与预测。

        Args:
            panel: 因子面板。

        Returns:
            PredictionResult: 含每月预测得分与元模型权重。
        """
        frame = prepare_panel(panel)
        self._prepare_l1_models(frame)
        month_ends = iter_month_ends(frame)
        prediction_frames: list[pd.DataFrame] = []
        importance_frames: list[pd.DataFrame] = []
        for idx in range(self.train_months, len(month_ends)):
            predict_date = month_ends[idx]
            train_start = month_ends[idx - self.train_months]
            train_frame = frame[
                (frame["trade_date"] >= train_start)
                & (frame["trade_date"] < predict_date)
            ].dropna(subset=[self.label_column])
            if len(train_frame) < self.min_train_rows:
                continue
            predict_frame = frame[frame["trade_date"] == predict_date]
            if predict_frame.empty:
                continue
            month_pred, importance = self._stack_once(
                train_frame=train_frame,
                test_frame=predict_frame,
                stamp_date=predict_date,
            )
            if month_pred is None or importance is None:
                continue
            prediction_frames.append(month_pred)
            importance_frames.append(importance)
        return self._build_result(prediction_frames, importance_frames)

    def fit_predict_frozen(
        self,
        panel: pd.DataFrame,
        train_end_date: str,
        test_start_date: str,
        test_end_date: str | None = None,
    ) -> PredictionResult:
        """冻结模式：一次 Stacking 训练，对测试区间内的每个月末做推断。

        Args:
            panel: 因子面板。
            train_end_date: 训练集截止日，格式 ``YYYYMMDD``。
            test_start_date: 测试集开始日，格式 ``YYYYMMDD``。
            test_end_date: 测试集结束日，格式 ``YYYYMMDD``；为 ``None`` 时取面板末日。

        Returns:
            PredictionResult: 含每月预测得分与元模型权重。
        """
        frame = prepare_panel(panel)
        self._prepare_l1_models(frame)
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
        month_pred, importance = self._stack_once(
            train_frame=train_frame,
            test_frame=combined_test,
            stamp_date=train_end_ts,
        )
        if month_pred is None or importance is None:
            return PredictionResult(
                prediction_frame=empty_prediction_frame(self.label_column),
                feature_importance=empty_importance_frame(),
            )
        prediction_frames: list[pd.DataFrame] = []
        importance_frames: list[pd.DataFrame] = []
        for predict_date in month_ends:
            month_subset = month_pred[month_pred["trade_date"] == predict_date]
            if month_subset.empty:
                continue
            prediction_frames.append(month_subset)
            month_imp = importance.copy()
            month_imp["trade_date"] = predict_date
            importance_frames.append(month_imp)
        return self._build_result(prediction_frames, importance_frames)

    def _prepare_l1_models(self, frame: pd.DataFrame) -> None:
        """为所有 :class:`AlphaModelBase` 子类调用一次 ``prepare(panel)``。"""
        for l1 in self.l1_models:
            if isinstance(l1, AlphaModelBase):
                l1.prepare(frame)

    def _stack_once(
        self,
        train_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        stamp_date: pd.Timestamp,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """完成一次 Stacking：OOF -> 元模型训练 -> 测试集融合预测。

        Args:
            train_frame: 训练子集。
            test_frame: 一个或多个月末的测试子集。
            stamp_date: 用于打在 ``importance.trade_date`` 上的日期戳。

        Returns:
            tuple[pd.DataFrame | None, pd.DataFrame | None]:
                ``(prediction_frame, importance_frame)``，失败时返回 ``(None, None)``。
        """
        oof_matrix = self._compute_oof(train_frame)
        if oof_matrix is None or len(oof_matrix) == 0:
            return None, None
        y_train = train_frame[self.label_column].to_numpy(dtype=np.float64)
        test_matrix = self._predict_test(train_frame, test_frame)
        meta_weights, meta_predict = self._train_meta(
            oof_matrix=oof_matrix,
            y_train=y_train,
            test_matrix=test_matrix,
        )
        prediction_frame = test_frame[
            ["trade_date", "ts_code", self.label_column]
        ].copy()
        prediction_frame["ml_score"] = meta_predict
        importance_frame = pd.DataFrame(
            {
                "trade_date": stamp_date,
                "feature": [
                    f"l1_{i}_{_model_short_name(model)}"
                    for i, model in enumerate(self.l1_models)
                ],
                "importance": meta_weights,
            }
        )
        return prediction_frame, importance_frame

    def _compute_oof(self, train_frame: pd.DataFrame) -> np.ndarray | None:
        """对每个 L1 模型在训练集上做 KFold OOF 预测。

        Args:
            train_frame: 训练子集。

        Returns:
            np.ndarray | None: ``[n_train, n_l1]`` 元特征矩阵，无 L1 模型时返回 ``None``。
        """
        n_models = len(self.l1_models)
        n_train = len(train_frame)
        if n_models == 0 or n_train == 0:
            return None
        n_splits = max(2, min(self.n_oof_splits, n_train))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        splits = list(kf.split(np.arange(n_train)))
        tasks: list[tuple[int, np.ndarray, pd.DataFrame, pd.DataFrame]] = []
        for l1_idx, l1 in enumerate(self.l1_models):
            for tr_idx, val_idx in splits:
                fold_train = train_frame.iloc[tr_idx]
                fold_val = train_frame.iloc[val_idx]
                tasks.append((l1_idx, val_idx, fold_train, fold_val))
        predictions = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_l1_fit_predict)(self.l1_models[l1_idx], fold_train, fold_val)
            for l1_idx, _, fold_train, fold_val in tasks
        )
        oof_matrix = np.zeros((n_train, n_models), dtype=np.float64)
        for (l1_idx, val_idx, _, _), preds in zip(tasks, predictions, strict=False):
            oof_matrix[val_idx, l1_idx] = preds
        return oof_matrix

    def _predict_test(
        self,
        train_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
    ) -> np.ndarray:
        """对每个 L1 在全量训练子集上各训练一次并对测试集预测。

        Args:
            train_frame: 训练子集。
            test_frame: 测试子集。

        Returns:
            np.ndarray: ``[n_test, n_l1]`` 预测矩阵。
        """
        n_models = len(self.l1_models)
        n_test = len(test_frame)
        if n_models == 0:
            return np.zeros((n_test, 0), dtype=np.float64)
        predictions = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_l1_fit_predict)(l1, train_frame, test_frame)
            for l1 in self.l1_models
        )
        test_matrix = np.zeros((n_test, n_models), dtype=np.float64)
        for l1_idx, preds in enumerate(predictions):
            test_matrix[:, l1_idx] = preds
        return test_matrix

    def _train_meta(
        self,
        oof_matrix: np.ndarray,
        y_train: np.ndarray,
        test_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """训练元模型并对测试集预测。

        Args:
            oof_matrix: 形如 ``[n_train, n_l1]`` 的元特征矩阵。
            y_train: 长度为 ``n_train`` 的标签。
            test_matrix: 形如 ``[n_test, n_l1]`` 的测试集 L1 预测矩阵。

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(meta_weights, meta_predictions)``。
                * ``meta_weights``: 长度为 ``n_l1`` 的权重向量（不同元模型口径略有差异）。
                * ``meta_predictions``: 长度为 ``n_test`` 的最终融合预测。
        """
        n_models = oof_matrix.shape[1]
        if self.meta_model == "logistic_avg":
            weights = np.ones(n_models, dtype=np.float64) / max(n_models, 1)
            predictions = test_matrix.mean(axis=1)
            return weights, predictions
        if self.meta_model == "linear":
            linear = LinearRegression(fit_intercept=True)
            linear.fit(oof_matrix, y_train)
            weights = np.asarray(linear.coef_, dtype=np.float64)
            predictions = np.asarray(linear.predict(test_matrix), dtype=np.float64)
            return weights, predictions
        # mlp 元学习器
        mlp = MLPRegressor(
            hidden_layer_sizes=(64, 64),
            activation="relu",
            solver="adam",
            max_iter=200,
            random_state=self.random_state,
            early_stopping=False,
        )
        mlp.fit(oof_matrix, y_train)
        predictions = np.asarray(mlp.predict(test_matrix), dtype=np.float64)
        first_layer = np.asarray(mlp.coefs_[0], dtype=np.float64)
        weights = np.abs(first_layer).sum(axis=1)
        return weights, predictions

    def _build_result(
        self,
        prediction_frames: list[pd.DataFrame],
        importance_frames: list[pd.DataFrame],
    ) -> PredictionResult:
        """合并多窗口结果并把 ``trade_date`` 转为字符串。"""
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
