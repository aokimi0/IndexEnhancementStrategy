"""带置信度加权的组合优化器。

继承自 :class:`ConstrainedPortfolioOptimizer`，在不修改父类的前提下，提供三种基于
Conformal Prediction 置信度的加权方案：

* ``alpha_scale``：在标准化 alpha 上按 c^β 缩放
* ``candidate_filter``：仅保留置信度排名前 X% 的候选股
* ``objective_penalty``：在二次规划目标中加入 -γ Σ (1 - c_i) w_i^2 惩罚

当输入 snapshot 缺少置信度列时，行为自动退化为父类。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import warnings

import cvxpy as cp
import numpy as np
import pandas as pd

from src.portfolio.optimizer import (
    ConstrainedPortfolioOptimizer,
    OptimizationConfig,
    OptimizationResult,
)


WeightingScheme = Literal[
    "alpha_scale",
    "candidate_filter",
    "objective_penalty",
    "uncertainty_risk",
]


@dataclass(frozen=True)
class UncertaintyAwareConfig:
    """置信度加权方案的超参数。

    Attributes:
        weighting_scheme: 加权方案名称。
        beta: ``alpha_scale`` 方案下 c^β 的指数。
        top_pct: ``candidate_filter`` 方案下保留的置信度百分位（0-1）。
        gamma: ``objective_penalty`` 方案下的惩罚强度。
        confidence_column: 置信度列名。
        min_candidates: 候选过滤后保留的最少股票数，避免过度收缩。
        risk_uncertainty_coef: ``uncertainty_risk`` 方案下预测方差并入风险项的系数 τ。
        half_width_column: ``uncertainty_risk`` 方案下保形区间半宽列名。
    """

    weighting_scheme: WeightingScheme = "alpha_scale"
    beta: float = 1.0
    top_pct: float = 0.7
    gamma: float = 0.1
    confidence_column: str = "confidence"
    min_candidates: int = 30
    risk_uncertainty_coef: float = 1.0
    half_width_column: str = "ci_half_width"


class UncertaintyAwarePortfolioOptimizer(ConstrainedPortfolioOptimizer):
    """带 Conformal 置信度加权的指数增强组合优化器。

    本类只在 :class:`ConstrainedPortfolioOptimizer` 的基础上扩展 ``optimize`` 流程，
    不修改父类源码，所有约束、协方差估计、行业偏离逻辑均通过受保护方法复用。
    """

    def __init__(
        self,
        config: OptimizationConfig | None = None,
        uncertainty_config: UncertaintyAwareConfig | None = None,
    ) -> None:
        """初始化带置信度加权的优化器。

        Args:
            config: 与父类一致的组合优化参数。
            uncertainty_config: 置信度加权超参数。
        """
        super().__init__(config=config)
        self.uncertainty_config = uncertainty_config or UncertaintyAwareConfig()
        if self.uncertainty_config.weighting_scheme not in (
            "alpha_scale",
            "candidate_filter",
            "objective_penalty",
            "uncertainty_risk",
        ):
            raise ValueError(
                "weighting_scheme 必须是 alpha_scale / candidate_filter / "
                "objective_penalty / uncertainty_risk 之一"
            )

    def optimize(
        self,
        snapshot: pd.DataFrame,
        return_history: pd.DataFrame,
        previous_weights: dict[str, float] | None = None,
        score_column: str = "score",
    ) -> OptimizationResult:
        """根据 alpha 信号与置信度生成调仓权重。

        Args:
            snapshot: 调仓日截面，必须包含 ``confidence`` 列才会启用加权方案，
                否则退化为父类行为。
            return_history: 截面股票的历史日收益序列。
            previous_weights: 上一期组合权重。
            score_column: alpha 信号列名。

        Returns:
            OptimizationResult: 优化后的权重与诊断信息。
        """
        confidence_column = self.uncertainty_config.confidence_column
        if confidence_column not in snapshot.columns:
            return super().optimize(
                snapshot=snapshot,
                return_history=return_history,
                previous_weights=previous_weights,
                score_column=score_column,
            )

        working_snapshot = snapshot.copy()
        working_snapshot[confidence_column] = pd.to_numeric(
            working_snapshot[confidence_column], errors="coerce"
        ).fillna(working_snapshot[confidence_column].astype(float).median())
        working_snapshot[confidence_column] = working_snapshot[confidence_column].clip(0.0, 1.0)

        if self.uncertainty_config.weighting_scheme == "candidate_filter":
            working_snapshot = self._apply_candidate_filter(snapshot=working_snapshot)
            if working_snapshot.empty:
                return OptimizationResult(
                    weights=pd.DataFrame(columns=["ts_code", "weight"]),
                    diagnostics=self._empty_diagnostics(),
                    status="empty_after_filter",
                )

        prepared = self._prepare_snapshot(snapshot=working_snapshot, score_column=score_column)
        if prepared.empty:
            return OptimizationResult(
                weights=pd.DataFrame(columns=["ts_code", "weight"]),
                diagnostics=self._empty_diagnostics(),
                status="empty_snapshot",
            )

        confidence_map = dict(
            zip(
                working_snapshot["ts_code"].astype(str),
                working_snapshot[confidence_column].astype(float),
                strict=False,
            )
        )
        confidence_vector = (
            prepared["ts_code"].astype(str).map(confidence_map).fillna(0.5).to_numpy(dtype=float)
        )

        codes = prepared["ts_code"].tolist()
        benchmark_weights = prepared["benchmark_weight"].to_numpy(dtype=float)
        previous_weight_vector = self._build_previous_weight_vector(
            codes=codes,
            previous_weights=previous_weights or {},
        )
        covariance = self._estimate_covariance_matrix(
            return_history=return_history,
            codes=codes,
        )
        if self.uncertainty_config.weighting_scheme == "uncertainty_risk":
            covariance = self._augment_covariance_with_uncertainty(
                covariance=covariance,
                prepared=prepared,
                working_snapshot=working_snapshot,
            )
        alpha = prepared["alpha_score"].to_numpy(dtype=float)
        if self.uncertainty_config.weighting_scheme == "alpha_scale":
            alpha = alpha * np.power(confidence_vector, self.uncertainty_config.beta)

        industry_matrix, benchmark_industry_weight = self._build_industry_constraints(
            prepared=prepared,
            benchmark_weights=benchmark_weights,
        )

        if self.uncertainty_config.weighting_scheme == "objective_penalty":
            optimized_weights, status = self._solve_with_uncertainty_penalty(
                alpha=alpha,
                confidence=confidence_vector,
                benchmark_weights=benchmark_weights,
                previous_weights=previous_weight_vector,
                covariance=covariance,
                industry_matrix=industry_matrix,
                benchmark_industry_weight=benchmark_industry_weight,
            )
        else:
            optimized_weights, status = self._solve_problem(
                alpha=alpha,
                benchmark_weights=benchmark_weights,
                previous_weights=previous_weight_vector,
                covariance=covariance,
                industry_matrix=industry_matrix,
                benchmark_industry_weight=benchmark_industry_weight,
            )

        diagnostics = self._compute_diagnostics(
            weights=optimized_weights,
            benchmark_weights=benchmark_weights,
            previous_weights=previous_weight_vector,
            covariance=covariance,
            industry_matrix=industry_matrix,
            benchmark_industry_weight=benchmark_industry_weight,
        )
        diagnostics["avg_confidence_weighted"] = float(
            np.average(confidence_vector, weights=optimized_weights + 1e-12)
        )
        diagnostics["weighting_scheme"] = self.uncertainty_config.weighting_scheme

        result = prepared[["ts_code"]].copy()
        result["weight"] = optimized_weights
        result["benchmark_weight"] = benchmark_weights
        result["industry_name"] = prepared["industry_name"].to_numpy()
        result["confidence"] = confidence_vector
        return OptimizationResult(weights=result, diagnostics=diagnostics, status=status)

    def _apply_candidate_filter(self, snapshot: pd.DataFrame) -> pd.DataFrame:
        """按置信度排名筛选候选池。

        Args:
            snapshot: 含置信度列的截面。

        Returns:
            pd.DataFrame: 保留前 ``top_pct`` 置信度的子集，至少保留 ``min_candidates`` 条。
        """
        confidence_column = self.uncertainty_config.confidence_column
        top_pct = float(self.uncertainty_config.top_pct)
        if top_pct >= 1.0 or snapshot.empty:
            return snapshot
        sorted_frame = snapshot.sort_values(
            confidence_column, ascending=False
        ).reset_index(drop=True)
        keep_count = max(
            int(np.ceil(len(sorted_frame) * top_pct)),
            min(self.uncertainty_config.min_candidates, len(sorted_frame)),
        )
        return sorted_frame.iloc[:keep_count].reset_index(drop=True)

    def _augment_covariance_with_uncertainty(
        self,
        covariance: np.ndarray,
        prepared: pd.DataFrame,
        working_snapshot: pd.DataFrame,
    ) -> np.ndarray:
        """将 Conformal 预测半宽的平方作为预测方差并入协方差对角。

        贝叶斯/Black-Litterman 视角下，alpha 估计自带估计方差，半宽越大越不可信。
        将其归一化到与协方差对角同量级后按系数 τ 加到对角，使优化器与跟踪误差约束
        天然回避高不确定个股，且不会因量纲失配而压垮 TE 约束。

        Args:
            covariance: 原始（已修正半正定）协方差矩阵。
            prepared: 已清洗并标准化的截面（决定股票顺序）。
            working_snapshot: 含半宽列的工作截面。

        Returns:
            np.ndarray: 对角增广后的协方差矩阵。
        """
        half_width_column = self.uncertainty_config.half_width_column
        if half_width_column not in working_snapshot.columns:
            return covariance
        half_width_map = dict(
            zip(
                working_snapshot["ts_code"].astype(str),
                pd.to_numeric(working_snapshot[half_width_column], errors="coerce").astype(float),
                strict=False,
            )
        )
        half_width = (
            prepared["ts_code"].astype(str).map(half_width_map).to_numpy(dtype=float)
        )
        median_hw = np.nanmedian(half_width)
        if not np.isfinite(median_hw) or median_hw <= 0:
            median_hw = 1.0
        half_width = np.where(np.isfinite(half_width), half_width, median_hw)
        predictive_variance = half_width**2
        variance_mean = float(np.mean(predictive_variance))
        if variance_mean <= 0:
            return covariance
        diag_mean = float(np.mean(np.diag(covariance)))
        scaled_variance = predictive_variance / variance_mean * diag_mean
        return covariance + np.diag(
            self.uncertainty_config.risk_uncertainty_coef * scaled_variance
        )

    def _solve_with_uncertainty_penalty(
        self,
        alpha: np.ndarray,
        confidence: np.ndarray,
        benchmark_weights: np.ndarray,
        previous_weights: np.ndarray,
        covariance: np.ndarray,
        industry_matrix: np.ndarray,
        benchmark_industry_weight: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        """带置信度惩罚的二次规划求解。

        目标函数：``max α^T w - λ (w-w_b)^T Σ (w-w_b) - κ |w-w_prev|_1 - γ Σ (1-c_i) w_i^2``。

        Args:
            alpha: 标准化 alpha 信号。
            confidence: 个股置信度向量，范围 [0, 1]。
            benchmark_weights: 基准权重。
            previous_weights: 上一期权重。
            covariance: 协方差矩阵。
            industry_matrix: 行业指示矩阵。
            benchmark_industry_weight: 基准行业权重。

        Returns:
            tuple[np.ndarray, str]: 优化权重与求解状态。
        """
        stock_count = len(alpha)
        weights = cp.Variable(stock_count)
        active_weights = weights - benchmark_weights
        penalty_coef = np.clip(1.0 - confidence, 0.0, 1.0)
        gamma = float(self.uncertainty_config.gamma)
        risk_term = self.config.risk_aversion * cp.quad_form(active_weights, covariance)
        turnover_term = self.config.turnover_penalty * cp.norm1(weights - previous_weights)
        uncertainty_term = gamma * cp.sum(cp.multiply(penalty_coef, cp.square(weights)))

        objective = cp.Maximize(alpha @ weights - risk_term - turnover_term - uncertainty_term)
        constraints: list[cp.Constraint] = [
            cp.sum(weights) == 1.0,
            weights >= 0.0,
            weights <= self.config.max_weight,
            cp.quad_form(active_weights, covariance)
            <= (self.config.max_tracking_error / np.sqrt(252.0)) ** 2,
        ]
        if np.sum(previous_weights) > 0:
            constraints.append(
                cp.norm1(weights - previous_weights) <= 2.0 * self.config.max_turnover
            )
        for idx in range(industry_matrix.shape[0]):
            exposure = industry_matrix[idx] @ weights
            constraints.append(
                exposure <= benchmark_industry_weight[idx] + self.config.max_industry_deviation
            )
            constraints.append(
                exposure >= benchmark_industry_weight[idx] - self.config.max_industry_deviation
            )

        problem = cp.Problem(objective, constraints)
        for solver in (cp.OSQP, cp.CLARABEL, cp.SCS):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    if solver == cp.OSQP:
                        problem.solve(
                            solver=solver,
                            verbose=False,
                            eps_abs=1e-5,
                            eps_rel=1e-5,
                            max_iter=10000,
                        )
                    else:
                        problem.solve(solver=solver, verbose=False)
            except Exception:
                continue
            if weights.value is not None and problem.status in {
                cp.OPTIMAL,
                cp.OPTIMAL_INACCURATE,
            }:
                solved = np.clip(
                    np.asarray(weights.value).reshape(-1), 0.0, self.config.max_weight
                )
                solved = solved / solved.sum() if solved.sum() > 0 else benchmark_weights.copy()
                return solved, str(problem.status)

        fallback = previous_weights.copy()
        if fallback.sum() <= 0:
            fallback = benchmark_weights.copy()
        fallback = (
            fallback / fallback.sum()
            if fallback.sum() > 0
            else np.full(stock_count, 1.0 / stock_count)
        )
        return fallback, "fallback"
