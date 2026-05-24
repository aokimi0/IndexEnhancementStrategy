"""Conformal Prediction 不确定性量化模块。

实现 Split Conformal Prediction（含 Locally Adaptive 可选扩展）以及
按离散组别（行业等）独立校准的 Mondrian 变体，用于给点预测附带置信区间和置信度。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


_EPSILON = 1e-6


@dataclass
class ConformalCalibration:
    """Split Conformal 单次拟合的校准产物。

    Attributes:
        base_model: 已在 train_proper 上拟合的回归模型。
        residual_model: 可选的残差幅度模型（启用 Locally Adaptive 时存在）。
        quantile: 校准集合上保形得分的 (n+1)(1-α)/n 分位数（Vovk 修正）。
        n_calib: 校准集合样本数。
    """

    base_model: Any
    residual_model: Optional[Any]
    quantile: float
    n_calib: int


@dataclass
class MondrianCalibration:
    """Mondrian Conformal 单次拟合的校准产物。

    Attributes:
        base_model: 已在 train_proper 上拟合的回归模型。
        residual_model: 可选的残差幅度模型。
        group_quantiles: 各组的分位数映射。
        global_quantile: 全局分位数，当测试样本所在组未在校准集中出现时回退使用。
        group_sample_counts: 各组校准样本数，用于诊断。
    """

    base_model: Any
    residual_model: Optional[Any]
    group_quantiles: dict[Any, float]
    global_quantile: float
    group_sample_counts: dict[Any, int]


def _to_numpy_features(features: pd.DataFrame | np.ndarray) -> np.ndarray:
    """将特征转为 numpy 数组以便兼容多种 sklearn-like 模型。

    Args:
        features: pandas DataFrame 或 numpy 数组形式的特征矩阵。

    Returns:
        np.ndarray: 二维浮点数组。
    """
    if isinstance(features, pd.DataFrame):
        return features.to_numpy()
    return np.asarray(features)


def _vovk_quantile_level(n: int, alpha: float) -> float:
    """计算 Vovk 修正后的覆盖率分位数水平。

    Args:
        n: 校准集合样本数。
        alpha: 显著性水平，覆盖率目标为 1 - alpha。

    Returns:
        float: 截断到 [0, 1] 的 (n+1)(1-α)/n 水平。
    """
    if n <= 0:
        return 1.0 - alpha
    return float(min(np.ceil((n + 1) * (1.0 - alpha)) / n, 1.0))


def _empirical_quantile(scores: np.ndarray, level: float) -> float:
    """按上界（higher 插值）取经验分位数，符合 Conformal 保形要求。

    Args:
        scores: 保形得分数组。
        level: 分位数水平。

    Returns:
        float: 对应分位数。
    """
    if scores.size == 0:
        return 0.0
    return float(np.quantile(scores, level, method="higher"))


def _normalize_to_unit(values: np.ndarray) -> np.ndarray:
    """min-max 归一化到 [0, 1]，若所有值相同则统一返回 1.0。

    Args:
        values: 待归一化的一维数组。

    Returns:
        np.ndarray: 归一化结果。
    """
    if values.size == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo <= _EPSILON:
        return np.ones_like(values, dtype=float)
    return (values - lo) / (hi - lo + _EPSILON)


class SplitConformalPredictor:
    """Split Conformal Prediction 回归预测器。

    标准模式下 half_width 为标量（全样本一致）；当提供 ``residual_model_factory``
    时切换为 Locally Adaptive Conformal Prediction，half_width 与样本特征相关，
    从而下游的置信度加权才会出现样本间差异。
    """

    def __init__(
        self,
        base_model_factory: Callable[[], Any],
        residual_model_factory: Optional[Callable[[], Any]] = None,
        calibration_ratio: float = 0.3,
        alpha: float = 0.1,
        random_state: int = 42,
    ) -> None:
        """初始化 Split Conformal 预测器。

        Args:
            base_model_factory: 返回新 sklearn-like 回归模型实例的工厂函数。
            residual_model_factory: 可选的残差模型工厂；提供后启用 Locally Adaptive 模式。
            calibration_ratio: 校准集占整个训练样本的比例，取值 (0, 1)。
            alpha: 显著性水平，覆盖率目标为 1 - alpha。
            random_state: train_proper / calib 切分的随机种子。
        """
        self.base_model_factory = base_model_factory
        self.residual_model_factory = residual_model_factory
        self.calibration_ratio = calibration_ratio
        self.alpha = alpha
        self.random_state = random_state
        self.calibration_: Optional[ConformalCalibration] = None

    @property
    def locally_adaptive(self) -> bool:
        """是否启用 Locally Adaptive 模式。"""
        return self.residual_model_factory is not None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "SplitConformalPredictor":
        """切分校准集合，拟合主模型并完成保形校准。

        Args:
            X: 训练集特征。
            y: 训练集标签。

        Returns:
            SplitConformalPredictor: 自身，便于链式调用。

        Raises:
            ValueError: 当样本量不足以切分 train_proper / calib 时抛出。
        """
        features = _to_numpy_features(X)
        labels = np.asarray(y, dtype=float)
        if len(features) < 4:
            raise ValueError("训练样本不足以切分 train_proper/calib，至少需要 4 条记录")

        X_train, X_calib, y_train, y_calib = train_test_split(
            features,
            labels,
            test_size=self.calibration_ratio,
            random_state=self.random_state,
            shuffle=True,
        )

        base_model = self.base_model_factory()
        base_model.fit(X_train, y_train)

        residual_model: Optional[Any] = None
        if self.locally_adaptive:
            train_predictions = base_model.predict(X_train)
            train_residuals = np.abs(y_train - train_predictions)
            log_residuals = np.log(train_residuals + _EPSILON)
            residual_model = self.residual_model_factory()
            residual_model.fit(X_train, log_residuals)

            calib_predictions = base_model.predict(X_calib)
            calib_sigma = np.exp(residual_model.predict(X_calib)) + _EPSILON
            calib_scores = np.abs(y_calib - calib_predictions) / calib_sigma
        else:
            calib_predictions = base_model.predict(X_calib)
            calib_scores = np.abs(y_calib - calib_predictions)

        level = _vovk_quantile_level(n=len(calib_scores), alpha=self.alpha)
        quantile = _empirical_quantile(scores=calib_scores, level=level)

        self.calibration_ = ConformalCalibration(
            base_model=base_model,
            residual_model=residual_model,
            quantile=quantile,
            n_calib=len(calib_scores),
        )
        return self

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """对测试集生成中心预测和保形区间。

        Args:
            X: 测试集特征。

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                依次为 ``(y_hat, lower, upper, half_width)``。

        Raises:
            RuntimeError: 当未先调用 fit 时抛出。
        """
        if self.calibration_ is None:
            raise RuntimeError("predict 前必须先调用 fit")
        features = _to_numpy_features(X)
        y_hat = np.asarray(self.calibration_.base_model.predict(features), dtype=float)

        if self.locally_adaptive and self.calibration_.residual_model is not None:
            sigma = np.exp(self.calibration_.residual_model.predict(features)) + _EPSILON
            half_width = self.calibration_.quantile * sigma
        else:
            half_width = np.full(len(y_hat), self.calibration_.quantile, dtype=float)

        lower = y_hat - half_width
        upper = y_hat + half_width
        return y_hat, lower, upper, half_width

    def confidence(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """计算样本置信度并归一化到 [0, 1]。

        定义 c_i = 1 / (2 * half_width_i + ε)，半宽越小越置信，再做 min-max 归一化。

        Args:
            X: 测试集特征。

        Returns:
            np.ndarray: 长度等于样本数的置信度向量。
        """
        _, _, _, half_width = self.predict(X)
        raw_confidence = 1.0 / (2.0 * half_width + _EPSILON)
        return _normalize_to_unit(raw_confidence)


class MondrianConformalPredictor:
    """按离散组别独立校准的 Mondrian Conformal 预测器。

    每个组别独立计算保形分位数，能够缓解组间预测难度不均带来的覆盖率偏差，
    并使下游置信度加权方案在 Split 模式下也有组间差异。
    """

    def __init__(
        self,
        base_model_factory: Callable[[], Any],
        group_column: str,
        residual_model_factory: Optional[Callable[[], Any]] = None,
        calibration_ratio: float = 0.3,
        alpha: float = 0.1,
        random_state: int = 42,
        min_group_size: int = 5,
    ) -> None:
        """初始化 Mondrian Conformal 预测器。

        Args:
            base_model_factory: 返回新 sklearn-like 回归模型实例的工厂函数。
            group_column: 组别字段名，调用 fit/predict 时 X 必须包含该列。
            residual_model_factory: 可选的残差模型工厂，启用后切换 Locally Adaptive。
            calibration_ratio: 校准集占比。
            alpha: 显著性水平。
            random_state: 切分随机种子。
            min_group_size: 组校准样本数低于该阈值时回退到全局分位数。
        """
        self.base_model_factory = base_model_factory
        self.group_column = group_column
        self.residual_model_factory = residual_model_factory
        self.calibration_ratio = calibration_ratio
        self.alpha = alpha
        self.random_state = random_state
        self.min_group_size = min_group_size
        self.calibration_: Optional[MondrianCalibration] = None

    @property
    def locally_adaptive(self) -> bool:
        """是否启用 Locally Adaptive 模式。"""
        return self.residual_model_factory is not None

    def _split_feature_group(
        self,
        frame: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """将含 group_column 的 DataFrame 拆为特征矩阵和组别数组。

        Args:
            frame: 输入 DataFrame，需包含 group_column。

        Returns:
            tuple[np.ndarray, np.ndarray]: 特征矩阵和组别向量。

        Raises:
            ValueError: 当输入不是 DataFrame 或缺少 group_column 时抛出。
        """
        if not isinstance(frame, pd.DataFrame):
            raise ValueError("MondrianConformalPredictor 要求输入为 DataFrame")
        if self.group_column not in frame.columns:
            raise ValueError(f"输入缺少 group_column={self.group_column}")
        groups = frame[self.group_column].astype(str).to_numpy()
        features = frame.drop(columns=[self.group_column]).to_numpy(dtype=float)
        return features, groups

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> "MondrianConformalPredictor":
        """完成 train_proper 模型训练与组别保形校准。

        Args:
            X: 训练集 DataFrame，必须包含 group_column 列。
            y: 训练集标签。

        Returns:
            MondrianConformalPredictor: 自身。

        Raises:
            ValueError: 当样本量过少时抛出。
        """
        features, groups = self._split_feature_group(frame=X)
        labels = np.asarray(y, dtype=float)
        if len(features) < 4:
            raise ValueError("训练样本不足以切分 train_proper/calib")

        indices = np.arange(len(features))
        train_idx, calib_idx = train_test_split(
            indices,
            test_size=self.calibration_ratio,
            random_state=self.random_state,
            shuffle=True,
        )

        X_train = features[train_idx]
        X_calib = features[calib_idx]
        y_train = labels[train_idx]
        y_calib = labels[calib_idx]
        g_calib = groups[calib_idx]

        base_model = self.base_model_factory()
        base_model.fit(X_train, y_train)
        residual_model: Optional[Any] = None
        if self.locally_adaptive:
            train_residuals = np.abs(y_train - base_model.predict(X_train))
            log_residuals = np.log(train_residuals + _EPSILON)
            residual_model = self.residual_model_factory()
            residual_model.fit(X_train, log_residuals)

            calib_predictions = base_model.predict(X_calib)
            calib_sigma = np.exp(residual_model.predict(X_calib)) + _EPSILON
            calib_scores = np.abs(y_calib - calib_predictions) / calib_sigma
        else:
            calib_predictions = base_model.predict(X_calib)
            calib_scores = np.abs(y_calib - calib_predictions)

        global_level = _vovk_quantile_level(n=len(calib_scores), alpha=self.alpha)
        global_quantile = _empirical_quantile(scores=calib_scores, level=global_level)

        group_quantiles: dict[Any, float] = {}
        group_counts: dict[Any, int] = {}
        for group in np.unique(g_calib):
            mask = g_calib == group
            group_scores = calib_scores[mask]
            count = int(mask.sum())
            group_counts[group] = count
            if count < self.min_group_size:
                group_quantiles[group] = global_quantile
                continue
            level = _vovk_quantile_level(n=count, alpha=self.alpha)
            group_quantiles[group] = _empirical_quantile(scores=group_scores, level=level)

        self.calibration_ = MondrianCalibration(
            base_model=base_model,
            residual_model=residual_model,
            group_quantiles=group_quantiles,
            global_quantile=global_quantile,
            group_sample_counts=group_counts,
        )
        return self

    def predict(
        self,
        X: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """生成中心预测与组别相关的保形区间。

        Args:
            X: 测试集 DataFrame，必须包含 group_column。

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                ``(y_hat, lower, upper, half_width)``。

        Raises:
            RuntimeError: 当未先调用 fit 时抛出。
        """
        if self.calibration_ is None:
            raise RuntimeError("predict 前必须先调用 fit")
        features, groups = self._split_feature_group(frame=X)

        y_hat = np.asarray(self.calibration_.base_model.predict(features), dtype=float)
        q_arr = np.array(
            [
                self.calibration_.group_quantiles.get(group, self.calibration_.global_quantile)
                for group in groups
            ],
            dtype=float,
        )

        if self.locally_adaptive and self.calibration_.residual_model is not None:
            sigma = np.exp(self.calibration_.residual_model.predict(features)) + _EPSILON
            half_width = q_arr * sigma
        else:
            half_width = q_arr

        lower = y_hat - half_width
        upper = y_hat + half_width
        return y_hat, lower, upper, half_width

    def confidence(self, X: pd.DataFrame) -> np.ndarray:
        """计算样本置信度并归一化到 [0, 1]。

        Args:
            X: 测试集 DataFrame。

        Returns:
            np.ndarray: 置信度向量。
        """
        _, _, _, half_width = self.predict(X)
        raw_confidence = 1.0 / (2.0 * half_width + _EPSILON)
        return _normalize_to_unit(raw_confidence)
