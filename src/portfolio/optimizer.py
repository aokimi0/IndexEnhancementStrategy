"""带约束的组合优化器。"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import cvxpy as cp
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OptimizationConfig:
    """组合优化参数配置。"""

    max_tracking_error: float = 0.08
    max_industry_deviation: float = 0.02
    max_weight: float = 0.05
    max_turnover: float = 0.20
    risk_aversion: float = 5.0
    turnover_penalty: float = 0.002
    covariance_window: int = 60
    covariance_ridge: float = 1e-4


@dataclass
class OptimizationResult:
    """单次调仓优化结果。"""

    weights: pd.DataFrame
    diagnostics: dict[str, float]
    status: str


class ConstrainedPortfolioOptimizer:
    """基于二次规划的指数增强组合优化器。"""

    def __init__(self, config: OptimizationConfig | None = None) -> None:
        """初始化优化器。

        Args:
            config: 组合优化参数。
        """
        self.config = config or OptimizationConfig()

    def optimize(
        self,
        snapshot: pd.DataFrame,
        return_history: pd.DataFrame,
        previous_weights: dict[str, float] | None = None,
        score_column: str = "score",
    ) -> OptimizationResult:
        """根据给定信号和约束生成调仓权重。

        Args:
            snapshot: 调仓日截面数据。
            return_history: 截面股票的历史日收益序列。
            previous_weights: 上一期组合权重。
            score_column: alpha 信号列名。

        Returns:
            OptimizationResult: 优化后的权重与诊断信息。
        """
        prepared = self._prepare_snapshot(snapshot=snapshot, score_column=score_column)
        if prepared.empty:
            return OptimizationResult(
                weights=pd.DataFrame(columns=["ts_code", "weight"]),
                diagnostics=self._empty_diagnostics(),
                status="empty_snapshot",
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
        alpha = prepared["alpha_score"].to_numpy(dtype=float)
        industry_matrix, benchmark_industry_weight = self._build_industry_constraints(
            prepared=prepared,
            benchmark_weights=benchmark_weights,
        )

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
        result = prepared[["ts_code"]].copy()
        result["weight"] = optimized_weights
        result["benchmark_weight"] = benchmark_weights
        result["industry_name"] = prepared["industry_name"].to_numpy()
        return OptimizationResult(weights=result, diagnostics=diagnostics, status=status)

    def _prepare_snapshot(
        self,
        snapshot: pd.DataFrame,
        score_column: str,
    ) -> pd.DataFrame:
        """清洗单期截面并标准化 alpha 信号。"""
        required_columns = {"ts_code", "industry_name", "benchmark_weight", score_column}
        missing_columns = required_columns.difference(snapshot.columns)
        if missing_columns:
            raise ValueError(f"优化快照缺少必要字段: {sorted(missing_columns)}")

        prepared = snapshot.copy()
        prepared["benchmark_weight"] = pd.to_numeric(
            prepared["benchmark_weight"],
            errors="coerce",
        ).fillna(0.0)
        prepared[score_column] = pd.to_numeric(prepared[score_column], errors="coerce")
        prepared["industry_name"] = prepared["industry_name"].fillna("未知行业").astype(str)
        prepared = prepared.dropna(subset=[score_column]).drop_duplicates(subset=["ts_code"])
        if prepared.empty:
            return prepared

        benchmark_sum = prepared["benchmark_weight"].sum()
        if benchmark_sum <= 0:
            prepared["benchmark_weight"] = 1.0 / len(prepared)
        else:
            prepared["benchmark_weight"] = prepared["benchmark_weight"] / benchmark_sum

        score_std = prepared[score_column].std(ddof=0)
        if score_std and not np.isnan(score_std):
            prepared["alpha_score"] = (
                prepared[score_column] - prepared[score_column].mean()
            ) / score_std
        else:
            prepared["alpha_score"] = 0.0
        return prepared.reset_index(drop=True)

    @staticmethod
    def _build_previous_weight_vector(
        codes: list[str],
        previous_weights: dict[str, float],
    ) -> np.ndarray:
        """将上一期权重映射到当前股票顺序。"""
        return np.array([float(previous_weights.get(code, 0.0)) for code in codes])

    def _estimate_covariance_matrix(
        self,
        return_history: pd.DataFrame,
        codes: list[str],
    ) -> np.ndarray:
        """估计调仓截面的日收益协方差矩阵。"""
        if return_history.empty:
            return np.eye(len(codes)) * self.config.covariance_ridge

        pivot = (
            return_history[return_history["ts_code"].isin(codes)]
            .pivot_table(index="trade_date", columns="ts_code", values="daily_return")
            .reindex(columns=codes)
            .sort_index()
            .tail(self.config.covariance_window)
            .fillna(0.0)
        )
        if len(pivot) < 2:
            return np.eye(len(codes)) * self.config.covariance_ridge

        covariance = pivot.cov().to_numpy(dtype=float)
        covariance = np.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
        covariance = (covariance + covariance.T) / 2.0
        min_eigenvalue = float(np.min(np.linalg.eigvalsh(covariance)))
        if min_eigenvalue < 0:
            covariance += np.eye(len(codes)) * (-min_eigenvalue + self.config.covariance_ridge)
        else:
            covariance += np.eye(len(codes)) * self.config.covariance_ridge
        return covariance

    @staticmethod
    def _build_industry_constraints(
        prepared: pd.DataFrame,
        benchmark_weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """构建行业约束矩阵。"""
        industries = prepared["industry_name"].astype(str).unique().tolist()
        matrix = np.zeros((len(industries), len(prepared)))
        benchmark_industry_weight = np.zeros(len(industries))
        for idx, industry in enumerate(industries):
            mask = (prepared["industry_name"].astype(str) == industry).to_numpy(dtype=float)
            matrix[idx, :] = mask
            benchmark_industry_weight[idx] = float(mask @ benchmark_weights)
        return matrix, benchmark_industry_weight

    def _solve_problem(
        self,
        alpha: np.ndarray,
        benchmark_weights: np.ndarray,
        previous_weights: np.ndarray,
        covariance: np.ndarray,
        industry_matrix: np.ndarray,
        benchmark_industry_weight: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        """求解带约束的组合权重。"""
        stock_count = len(alpha)
        weights = cp.Variable(stock_count)
        active_weights = weights - benchmark_weights

        objective = cp.Maximize(
            alpha @ weights
            - self.config.risk_aversion * cp.quad_form(active_weights, covariance)
            - self.config.turnover_penalty * cp.norm1(weights - previous_weights)
        )
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
                solved = np.clip(np.asarray(weights.value).reshape(-1), 0.0, self.config.max_weight)
                solved = solved / solved.sum() if solved.sum() > 0 else benchmark_weights.copy()
                return solved, str(problem.status)

        fallback = previous_weights.copy()
        if fallback.sum() <= 0:
            fallback = benchmark_weights.copy()
        fallback = fallback / fallback.sum() if fallback.sum() > 0 else np.full(stock_count, 1.0 / stock_count)
        return fallback, "fallback"

    def _compute_diagnostics(
        self,
        weights: np.ndarray,
        benchmark_weights: np.ndarray,
        previous_weights: np.ndarray,
        covariance: np.ndarray,
        industry_matrix: np.ndarray,
        benchmark_industry_weight: np.ndarray,
    ) -> dict[str, float]:
        """计算单次调仓的约束诊断指标。"""
        active_weights = weights - benchmark_weights
        industry_exposure = industry_matrix @ weights if industry_matrix.size else np.array([])
        industry_deviation = (
            np.max(np.abs(industry_exposure - benchmark_industry_weight))
            if industry_exposure.size
            else np.nan
        )
        return {
            "turnover": float(0.5 * np.abs(weights - previous_weights).sum()),
            "ex_ante_tracking_error": float(
                np.sqrt(max(active_weights @ covariance @ active_weights, 0.0) * 252.0)
            ),
            "max_industry_deviation": float(industry_deviation) if not np.isnan(industry_deviation) else np.nan,
        }

    @staticmethod
    def _empty_diagnostics() -> dict[str, float]:
        """构建空诊断指标。"""
        return {
            "turnover": np.nan,
            "ex_ante_tracking_error": np.nan,
            "max_industry_deviation": np.nan,
        }
