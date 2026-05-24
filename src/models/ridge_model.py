"""Ridge 线性 alpha 模型实现。

继承 :class:`src.models.base.AlphaModelBase`，内部使用 :class:`sklearn.linear_model.RidgeCV`
进行 alpha 网格搜索，特征端用 :class:`sklearn.preprocessing.StandardScaler` 标准化
（仅在训练集上 fit，在测试集上 transform）。特征重要性使用系数绝对值。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from src.models.base import AlphaModelBase


class RidgeAlphaModel(AlphaModelBase):
    """基于 RidgeCV + StandardScaler 的线性 alpha 模型。"""

    def __init__(
        self,
        feature_columns: list[str],
        label_column: str = "label_excess_return_20d",
        train_months: int = 12,
        min_train_rows: int = 2000,
        alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
        cv: int = 3,
    ) -> None:
        """初始化超参数。

        Args:
            feature_columns: 特征列。
            label_column: 标签列。
            train_months: 滚动训练窗口月数。
            min_train_rows: 最小训练样本数。
            alphas: ``RidgeCV`` 搜索的正则系数网格。
            cv: ``RidgeCV`` 选择 alpha 时使用的交叉验证折数。
        """
        super().__init__(
            feature_columns=feature_columns,
            label_column=label_column,
            train_months=train_months,
            min_train_rows=min_train_rows,
        )
        self.alphas = tuple(alphas)
        self.cv = cv

    def fit_predict_batch(
        self,
        train_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        compute_importance: bool = True,
    ) -> dict[str, Any]:
        """单次 Ridge 训练与预测。

        Args:
            train_frame: 训练子集。
            test_frame: 测试子集。
            compute_importance: 是否在结果中返回特征重要性；Ridge 系数绝对值本身即重要性，开销忽略。

        Returns:
            dict[str, Any]: 含 ``predictions``（``np.ndarray``）与 ``importance``
                （``dict[str, float]``）；当 ``compute_importance`` 为 ``False`` 时
                重要性字段为全零字典。
        """
        scaler = StandardScaler()
        x_train = train_frame[self.feature_columns].fillna(0.0).to_numpy(
            dtype=np.float64
        )
        y_train = train_frame[self.label_column].to_numpy(dtype=np.float64)
        x_test = test_frame[self.feature_columns].fillna(0.0).to_numpy(
            dtype=np.float64
        )

        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        cv = min(self.cv, max(2, len(train_frame) // 2))
        model = RidgeCV(alphas=self.alphas, cv=cv)
        model.fit(x_train_scaled, y_train)
        predictions = np.asarray(model.predict(x_test_scaled), dtype=np.float64)
        if compute_importance:
            importance = dict(
                zip(
                    self.feature_columns,
                    np.abs(np.asarray(model.coef_, dtype=np.float64)).tolist(),
                    strict=False,
                )
            )
        else:
            importance = {name: 0.0 for name in self.feature_columns}
        return {"predictions": predictions, "importance": importance}
