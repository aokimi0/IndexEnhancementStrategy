"""LightGBM Alpha 模型。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class LightgbmPredictionResult:
    """LightGBM 预测结果。"""

    prediction_frame: pd.DataFrame
    feature_importance: pd.DataFrame


class LightgbmAlphaModel:
    """基于滚动窗口的 LightGBM 个股超额收益预测模型。"""

    def __init__(
        self,
        feature_columns: list[str],
        label_column: str = "label_excess_return_20d",
        train_months: int = 12,
        min_train_rows: int = 2000,
    ) -> None:
        """初始化模型参数。

        Args:
            feature_columns: 特征列列表。
            label_column: 标签列名。
            train_months: 滚动训练窗口月数。
            min_train_rows: 最小训练样本数。
        """
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.train_months = train_months
        self.min_train_rows = min_train_rows

    def fit_predict(self, panel: pd.DataFrame) -> LightgbmPredictionResult:
        """执行滚动训练和预测。

        Args:
            panel: 因子面板。

        Returns:
            LightgbmPredictionResult: 预测结果与特征重要性。
        """
        import lightgbm as lgb

        frame = panel.copy()
        frame["trade_date"] = pd.to_datetime(
            frame["trade_date"].astype(str),
            format="%Y%m%d",
            errors="coerce",
        )
        frame = frame.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
        month_ends = (
            frame[["trade_date"]]
            .drop_duplicates()
            .assign(month=lambda df: df["trade_date"].dt.to_period("M"))
            .groupby("month")["trade_date"]
            .max()
            .tolist()
        )

        prediction_frames: list[pd.DataFrame] = []
        importance_frames: list[pd.DataFrame] = []

        for idx in range(self.train_months, len(month_ends)):
            predict_date = month_ends[idx]
            train_start = month_ends[idx - self.train_months]
            train_frame = frame[
                (frame["trade_date"] >= train_start) & (frame["trade_date"] < predict_date)
            ].copy()
            train_frame = train_frame.dropna(subset=[self.label_column])
            if len(train_frame) < self.min_train_rows:
                continue

            predict_frame = frame[frame["trade_date"] == predict_date].copy()
            if predict_frame.empty:
                continue

            model = lgb.LGBMRegressor(
                objective="regression",
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(train_frame[self.feature_columns], train_frame[self.label_column])

            predict_frame["ml_score"] = model.predict(predict_frame[self.feature_columns])
            prediction_frames.append(
                predict_frame[
                    ["trade_date", "ts_code", self.label_column, "ml_score"]
                ]
            )

            importance_frames.append(
                pd.DataFrame(
                    {
                        "trade_date": predict_date,
                        "feature": self.feature_columns,
                        "importance": model.feature_importances_,
                    }
                )
            )

        prediction_result = (
            pd.concat(prediction_frames, ignore_index=True)
            if prediction_frames
            else pd.DataFrame(columns=["trade_date", "ts_code", self.label_column, "ml_score"])
        )
        importance_result = (
            pd.concat(importance_frames, ignore_index=True)
            if importance_frames
            else pd.DataFrame(columns=["trade_date", "feature", "importance"])
        )
        prediction_result["trade_date"] = prediction_result["trade_date"].dt.strftime("%Y%m%d")
        if not importance_result.empty:
            importance_result["trade_date"] = importance_result["trade_date"].dt.strftime("%Y%m%d")
        return LightgbmPredictionResult(
            prediction_frame=prediction_result,
            feature_importance=importance_result,
        )

    def fit_predict_frozen(
        self,
        panel: pd.DataFrame,
        train_end_date: str,
        test_start_date: str,
        test_end_date: str | None = None,
    ) -> LightgbmPredictionResult:
        """执行冻结训练模式下的预测。

        Args:
            panel: 因子面板。
            train_end_date: 训练集截止日，格式为 `YYYYMMDD`。
            test_start_date: 测试集开始日，格式为 `YYYYMMDD`。
            test_end_date: 测试集结束日，格式为 `YYYYMMDD`。

        Returns:
            LightgbmPredictionResult: 预测结果与特征重要性。
        """
        import lightgbm as lgb

        frame = panel.copy()
        frame["trade_date"] = pd.to_datetime(
            frame["trade_date"].astype(str),
            format="%Y%m%d",
            errors="coerce",
        )
        frame = frame.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
        train_end_ts = pd.to_datetime(train_end_date, format="%Y%m%d")
        test_start_ts = pd.to_datetime(test_start_date, format="%Y%m%d")
        test_end_ts = (
            pd.to_datetime(test_end_date, format="%Y%m%d")
            if test_end_date
            else frame["trade_date"].max()
        )

        train_frame = frame[frame["trade_date"] <= train_end_ts].copy()
        train_frame = train_frame.dropna(subset=[self.label_column])
        if len(train_frame) < self.min_train_rows:
            return LightgbmPredictionResult(
                prediction_frame=pd.DataFrame(
                    columns=["trade_date", "ts_code", self.label_column, "ml_score"]
                ),
                feature_importance=pd.DataFrame(
                    columns=["trade_date", "feature", "importance"]
                ),
            )

        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(train_frame[self.feature_columns], train_frame[self.label_column])

        test_frame = frame[
            (frame["trade_date"] >= test_start_ts) & (frame["trade_date"] <= test_end_ts)
        ].copy()
        month_ends = (
            test_frame[["trade_date"]]
            .drop_duplicates()
            .assign(month=lambda df: df["trade_date"].dt.to_period("M"))
            .groupby("month")["trade_date"]
            .max()
            .tolist()
        )
        prediction_frames: list[pd.DataFrame] = []
        importance_frames: list[pd.DataFrame] = []
        for predict_date in month_ends:
            predict_frame = frame[frame["trade_date"] == predict_date].copy()
            if predict_frame.empty:
                continue
            predict_frame["ml_score"] = model.predict(predict_frame[self.feature_columns])
            prediction_frames.append(
                predict_frame[
                    ["trade_date", "ts_code", self.label_column, "ml_score"]
                ]
            )
            importance_frames.append(
                pd.DataFrame(
                    {
                        "trade_date": predict_date,
                        "feature": self.feature_columns,
                        "importance": model.feature_importances_,
                    }
                )
            )

        prediction_result = (
            pd.concat(prediction_frames, ignore_index=True)
            if prediction_frames
            else pd.DataFrame(columns=["trade_date", "ts_code", self.label_column, "ml_score"])
        )
        importance_result = (
            pd.concat(importance_frames, ignore_index=True)
            if importance_frames
            else pd.DataFrame(columns=["trade_date", "feature", "importance"])
        )
        if not prediction_result.empty:
            prediction_result["trade_date"] = prediction_result["trade_date"].dt.strftime("%Y%m%d")
        if not importance_result.empty:
            importance_result["trade_date"] = importance_result["trade_date"].dt.strftime("%Y%m%d")
        return LightgbmPredictionResult(
            prediction_frame=prediction_result,
            feature_importance=importance_result,
        )
