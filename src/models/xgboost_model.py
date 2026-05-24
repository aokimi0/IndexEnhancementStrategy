"""XGBoost alpha 模型实现。

直接继承 :class:`src.models.base.AlphaModelBase`，复用滚动 / 冻结训练循环；只需实现
:meth:`fit_predict_batch`。模型主要通过 ``tree_method="hist"`` 控制内存占用，
并通过 ``n_jobs=-1`` 利用全部 CPU 核心。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.base import AlphaModelBase


class XgboostAlphaModel(AlphaModelBase):
    """基于 XGBoost 的滚动 / 冻结训练 alpha 模型。"""

    def __init__(
        self,
        feature_columns: list[str],
        label_column: str = "label_excess_return_20d",
        train_months: int = 12,
        min_train_rows: int = 2000,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        tree_method: str = "hist",
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        """初始化超参数。

        Args:
            feature_columns: 模型使用的特征列名。
            label_column: 监督学习标签列。
            train_months: 滚动训练窗口月数。
            min_train_rows: 最小训练样本数。
            n_estimators: 提升树轮数。
            max_depth: 单棵树最大深度。
            learning_rate: 学习率。
            subsample: 行采样比例。
            colsample_bytree: 列采样比例。
            tree_method: XGBoost 树算法，默认 ``hist`` 以兼顾速度。
            random_state: 随机种子。
            n_jobs: 并行线程数，``-1`` 表示使用全部 CPU。
        """
        super().__init__(
            feature_columns=feature_columns,
            label_column=label_column,
            train_months=train_months,
            min_train_rows=min_train_rows,
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.tree_method = tree_method
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit_predict_batch(
        self,
        train_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        compute_importance: bool = True,
    ) -> dict[str, Any]:
        """单次 XGBoost 训练与预测。

        Args:
            train_frame: 训练子集。
            test_frame: 测试子集。
            compute_importance: 是否在结果中返回特征重要性。XGBoost 本身已带，开销可忽略。

        Returns:
            dict[str, Any]: 含 ``predictions``（``np.ndarray``）与 ``importance``
                （``dict[str, float]``）；当 ``compute_importance`` 为 ``False`` 时
                重要性字段为全零字典。
        """
        import xgboost as xgb

        model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            tree_method=self.tree_method,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            objective="reg:squarederror",
            verbosity=0,
        )
        x_train = train_frame[self.feature_columns].to_numpy(dtype=np.float32)
        y_train = train_frame[self.label_column].to_numpy(dtype=np.float32)
        x_test = test_frame[self.feature_columns].to_numpy(dtype=np.float32)
        model.fit(x_train, y_train)
        predictions = np.asarray(model.predict(x_test), dtype=np.float64)
        if compute_importance:
            importance = dict(
                zip(
                    self.feature_columns,
                    model.feature_importances_.astype(float).tolist(),
                    strict=False,
                )
            )
        else:
            importance = {name: 0.0 for name in self.feature_columns}
        return {"predictions": predictions, "importance": importance}
