"""ConformalLightgbmModel：LightGBM Alpha 模型的不确定性量化包装。

通过组合 :class:`SplitConformalPredictor` 与滚动训练流程，输出 ``ml_score`` 同时附带
保形预测区间 ``ci_lower`` / ``ci_upper`` 与归一化置信度 ``confidence``。
该类不修改原始 :class:`LightgbmAlphaModel`，而是直接复用其滚动调仓窗口约定并替换内部
模型为带保形校准的版本。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from src.models.conformal import (
    MondrianConformalPredictor,
    SplitConformalPredictor,
)


@dataclass
class ConformalPredictionResult:
    """Conformal LightGBM 预测结果。

    Attributes:
        prediction_frame: 含 ``trade_date``、``ts_code``、label、``ml_score``、
            ``ci_lower``、``ci_upper``、``ci_half_width``、``confidence`` 的预测表。
        feature_importance: 每月主模型的特征重要性长表。
        coverage_diagnostics: 每月校准集合诊断信息（分位数、样本量等）。
    """

    prediction_frame: pd.DataFrame
    feature_importance: pd.DataFrame
    coverage_diagnostics: pd.DataFrame


def _default_base_model_factory() -> Any:
    """构造默认 LightGBM 回归器。

    Returns:
        Any: 一个未训练的 ``LGBMRegressor`` 实例。
    """
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def _default_residual_model_factory() -> Any:
    """构造默认的残差幅度模型（更浅的 LGBM）。

    Returns:
        Any: 一个未训练的 ``LGBMRegressor`` 实例。
    """
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=43,
        n_jobs=-1,
        verbose=-1,
    )


class ConformalLightgbmModel:
    """带 Split Conformal 校准的滚动 LightGBM Alpha 模型。

    使用方式与 :class:`LightgbmAlphaModel` 一致，但每次训练窗口内会切出校准集合，
    在校准集合上估计保形分位数；预测时同时输出置信区间。当 ``group_column`` 给定时
    切换为 Mondrian Conformal，按行业等组别独立校准。
    """

    def __init__(
        self,
        feature_columns: list[str],
        label_column: str = "label_excess_return_20d",
        train_months: int = 12,
        min_train_rows: int = 2000,
        alpha: float = 0.1,
        calibration_ratio: float = 0.3,
        group_column: Optional[str] = None,
        locally_adaptive: bool = True,
        base_model_factory: Optional[Callable[[], Any]] = None,
        residual_model_factory: Optional[Callable[[], Any]] = None,
        random_state: int = 42,
        min_group_size: int = 5,
    ) -> None:
        """初始化模型。

        Args:
            feature_columns: 特征列名列表。
            label_column: 标签列名，默认 ``label_excess_return_20d``。
            train_months: 滚动训练窗口月数。
            min_train_rows: 当训练窗口样本数低于该阈值时跳过该期。
            alpha: 显著性水平，覆盖率目标 = 1 - alpha。
            calibration_ratio: 校准集合占训练样本的比例。
            group_column: 组别字段名，提供则启用 Mondrian Conformal。
            locally_adaptive: 是否启用 Locally Adaptive Conformal（per-sample 半宽）。
            base_model_factory: 主模型工厂，缺省使用内置 LGBMRegressor。
            residual_model_factory: 残差模型工厂，缺省使用更浅的 LGBMRegressor。
            random_state: 随机种子。
            min_group_size: Mondrian 模式下分组最小校准样本数。
        """
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.train_months = train_months
        self.min_train_rows = min_train_rows
        self.alpha = alpha
        self.calibration_ratio = calibration_ratio
        self.group_column = group_column
        self.locally_adaptive = locally_adaptive
        self._base_factory = base_model_factory or _default_base_model_factory
        self._residual_factory = (
            residual_model_factory if residual_model_factory is not None else _default_residual_model_factory
        )
        self.random_state = random_state
        self.min_group_size = min_group_size

    def _build_predictor(self):
        """根据配置构造一个新的 Conformal 预测器实例。

        Returns:
            SplitConformalPredictor | MondrianConformalPredictor: 新的预测器。
        """
        residual_factory = self._residual_factory if self.locally_adaptive else None
        if self.group_column is None:
            return SplitConformalPredictor(
                base_model_factory=self._base_factory,
                residual_model_factory=residual_factory,
                calibration_ratio=self.calibration_ratio,
                alpha=self.alpha,
                random_state=self.random_state,
            )
        return MondrianConformalPredictor(
            base_model_factory=self._base_factory,
            group_column=self.group_column,
            residual_model_factory=residual_factory,
            calibration_ratio=self.calibration_ratio,
            alpha=self.alpha,
            random_state=self.random_state,
            min_group_size=self.min_group_size,
        )

    def _normalize_panel(self, panel: pd.DataFrame) -> pd.DataFrame:
        """统一面板的日期格式和排序。

        Args:
            panel: 原始因子面板。

        Returns:
            pd.DataFrame: 已统一格式的副本。
        """
        frame = panel.copy()
        frame["trade_date"] = pd.to_datetime(
            frame["trade_date"].astype(str),
            format="%Y%m%d",
            errors="coerce",
        )
        return frame.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    def _month_ends(self, frame: pd.DataFrame) -> list[pd.Timestamp]:
        """提取每月最后一个交易日。

        Args:
            frame: 已规范化日期的面板。

        Returns:
            list[pd.Timestamp]: 月末日期列表。
        """
        return (
            frame[["trade_date"]]
            .drop_duplicates()
            .assign(month=lambda df: df["trade_date"].dt.to_period("M"))
            .groupby("month")["trade_date"]
            .max()
            .tolist()
        )

    def _select_training_columns(self, frame: pd.DataFrame) -> list[str]:
        """根据是否启用 Mondrian 选择 fit 用的列。

        Args:
            frame: 训练 / 测试帧。

        Returns:
            list[str]: 传给 conformal predictor 的列名列表。
        """
        if self.group_column is None:
            return list(self.feature_columns)
        if self.group_column in self.feature_columns:
            return list(self.feature_columns)
        return list(self.feature_columns) + [self.group_column]

    def _extract_feature_importance(self, predictor) -> Optional[np.ndarray]:
        """从已拟合的 conformal predictor 中提取主模型的特征重要性。

        Args:
            predictor: Split 或 Mondrian Conformal 预测器实例。

        Returns:
            np.ndarray | None: 特征重要性数组，若不可用则返回 None。
        """
        calibration = getattr(predictor, "calibration_", None)
        if calibration is None:
            return None
        importances = getattr(calibration.base_model, "feature_importances_", None)
        if importances is None:
            return None
        return np.asarray(importances)

    def _predict_block(
        self,
        predictor,
        predict_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """对单一调仓日截面执行预测并落入标准列。

        Args:
            predictor: 已拟合的 conformal predictor。
            predict_frame: 调仓日截面 DataFrame。

        Returns:
            pd.DataFrame: 含 ml_score、ci_lower、ci_upper、ci_half_width、confidence 等的预测帧。
        """
        if self.group_column is None:
            features = predict_frame[self.feature_columns]
        else:
            columns = self._select_training_columns(frame=predict_frame)
            features = predict_frame[columns]
        y_hat, lower, upper, half_width = predictor.predict(features)
        confidence = predictor.confidence(features)
        block = predict_frame[["trade_date", "ts_code", self.label_column]].copy()
        block["ml_score"] = y_hat
        block["ci_lower"] = lower
        block["ci_upper"] = upper
        block["ci_half_width"] = half_width
        block["confidence"] = confidence
        return block

    def _build_diagnostics(
        self,
        predictor,
        predict_date: pd.Timestamp,
    ) -> dict[str, Any]:
        """收集单期校准诊断信息。

        Args:
            predictor: 已拟合的 conformal predictor。
            predict_date: 调仓日。

        Returns:
            dict[str, Any]: 含 quantile、n_calib 等字段的字典。
        """
        calibration = predictor.calibration_
        diag: dict[str, Any] = {
            "trade_date": predict_date,
            "alpha": self.alpha,
            "locally_adaptive": self.locally_adaptive,
            "mondrian": self.group_column is not None,
        }
        if isinstance(predictor, SplitConformalPredictor):
            diag.update(
                {
                    "quantile": calibration.quantile,
                    "n_calib": calibration.n_calib,
                }
            )
        else:
            diag.update(
                {
                    "global_quantile": calibration.global_quantile,
                    "n_groups": len(calibration.group_quantiles),
                    "n_calib": int(sum(calibration.group_sample_counts.values())),
                }
            )
        return diag

    def fit_predict(self, panel: pd.DataFrame) -> ConformalPredictionResult:
        """按月滚动训练并产生保形预测。

        Args:
            panel: 因子面板。

        Returns:
            ConformalPredictionResult: 预测、重要性、校准诊断三部分。
        """
        frame = self._normalize_panel(panel=panel)
        month_ends = self._month_ends(frame=frame)

        prediction_blocks: list[pd.DataFrame] = []
        importance_blocks: list[pd.DataFrame] = []
        diagnostic_rows: list[dict[str, Any]] = []

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

            predictor = self._build_predictor()
            training_columns = self._select_training_columns(frame=train_frame)
            predictor.fit(train_frame[training_columns], train_frame[self.label_column])

            block = self._predict_block(predictor=predictor, predict_frame=predict_frame)
            prediction_blocks.append(block)

            importances = self._extract_feature_importance(predictor=predictor)
            if importances is not None:
                importance_blocks.append(
                    pd.DataFrame(
                        {
                            "trade_date": predict_date,
                            "feature": self.feature_columns,
                            "importance": importances,
                        }
                    )
                )
            diagnostic_rows.append(
                self._build_diagnostics(predictor=predictor, predict_date=predict_date)
            )

        return self._finalize_result(
            prediction_blocks=prediction_blocks,
            importance_blocks=importance_blocks,
            diagnostic_rows=diagnostic_rows,
        )

    def fit_predict_frozen(
        self,
        panel: pd.DataFrame,
        train_end_date: str,
        test_start_date: str,
        test_end_date: Optional[str] = None,
    ) -> ConformalPredictionResult:
        """冻结训练集，跨多个测试月共用一套保形校准。

        Args:
            panel: 因子面板。
            train_end_date: 训练集截止日（YYYYMMDD）。
            test_start_date: 测试区间开始日。
            test_end_date: 测试区间结束日，None 时取面板末日。

        Returns:
            ConformalPredictionResult: 与 fit_predict 一致的三段产物。
        """
        frame = self._normalize_panel(panel=panel)
        train_end_ts = pd.to_datetime(train_end_date, format="%Y%m%d")
        test_start_ts = pd.to_datetime(test_start_date, format="%Y%m%d")
        test_end_ts = (
            pd.to_datetime(test_end_date, format="%Y%m%d") if test_end_date else frame["trade_date"].max()
        )

        train_frame = frame[frame["trade_date"] <= train_end_ts].copy()
        train_frame = train_frame.dropna(subset=[self.label_column])
        if len(train_frame) < self.min_train_rows:
            empty_columns = [
                "trade_date",
                "ts_code",
                self.label_column,
                "ml_score",
                "ci_lower",
                "ci_upper",
                "ci_half_width",
                "confidence",
            ]
            return ConformalPredictionResult(
                prediction_frame=pd.DataFrame(columns=empty_columns),
                feature_importance=pd.DataFrame(
                    columns=["trade_date", "feature", "importance"]
                ),
                coverage_diagnostics=pd.DataFrame(),
            )

        predictor = self._build_predictor()
        training_columns = self._select_training_columns(frame=train_frame)
        predictor.fit(train_frame[training_columns], train_frame[self.label_column])

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

        prediction_blocks: list[pd.DataFrame] = []
        importance_blocks: list[pd.DataFrame] = []
        diagnostic_rows: list[dict[str, Any]] = []
        importances = self._extract_feature_importance(predictor=predictor)
        for predict_date in month_ends:
            predict_frame = frame[frame["trade_date"] == predict_date].copy()
            if predict_frame.empty:
                continue
            block = self._predict_block(predictor=predictor, predict_frame=predict_frame)
            prediction_blocks.append(block)
            if importances is not None:
                importance_blocks.append(
                    pd.DataFrame(
                        {
                            "trade_date": predict_date,
                            "feature": self.feature_columns,
                            "importance": importances,
                        }
                    )
                )
            diagnostic_rows.append(
                self._build_diagnostics(predictor=predictor, predict_date=predict_date)
            )
        return self._finalize_result(
            prediction_blocks=prediction_blocks,
            importance_blocks=importance_blocks,
            diagnostic_rows=diagnostic_rows,
        )

    def _finalize_result(
        self,
        prediction_blocks: list[pd.DataFrame],
        importance_blocks: list[pd.DataFrame],
        diagnostic_rows: list[dict[str, Any]],
    ) -> ConformalPredictionResult:
        """合并各期结果并把日期统一为字符串。

        Args:
            prediction_blocks: 各期预测帧。
            importance_blocks: 各期特征重要性帧。
            diagnostic_rows: 各期校准诊断行。

        Returns:
            ConformalPredictionResult: 已合并的结果对象。
        """
        prediction_columns = [
            "trade_date",
            "ts_code",
            self.label_column,
            "ml_score",
            "ci_lower",
            "ci_upper",
            "ci_half_width",
            "confidence",
        ]
        prediction_frame = (
            pd.concat(prediction_blocks, ignore_index=True)
            if prediction_blocks
            else pd.DataFrame(columns=prediction_columns)
        )
        if not prediction_frame.empty:
            prediction_frame["trade_date"] = prediction_frame["trade_date"].dt.strftime("%Y%m%d")
        importance_frame = (
            pd.concat(importance_blocks, ignore_index=True)
            if importance_blocks
            else pd.DataFrame(columns=["trade_date", "feature", "importance"])
        )
        if not importance_frame.empty:
            importance_frame["trade_date"] = importance_frame["trade_date"].dt.strftime("%Y%m%d")
        diagnostics_frame = (
            pd.DataFrame(diagnostic_rows) if diagnostic_rows else pd.DataFrame()
        )
        if not diagnostics_frame.empty and "trade_date" in diagnostics_frame.columns:
            diagnostics_frame["trade_date"] = pd.to_datetime(
                diagnostics_frame["trade_date"]
            ).dt.strftime("%Y%m%d")
        return ConformalPredictionResult(
            prediction_frame=prediction_frame,
            feature_importance=importance_frame,
            coverage_diagnostics=diagnostics_frame,
        )
