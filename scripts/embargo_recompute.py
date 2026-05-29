"""第四章"诚实口径"(embargo=20)全套数字重算。

严禁修改 paper/，只读/写 data/processed 与 logs。核心去前瞻函数复用
``scripts/pit_run_leakfree.rolling_predict_embargo``：训练截止 = 预测日 − 20 交易日。

模块结构：
  A  长周期约束消融（PIT factor 面板，LightGBM+优化器，embargo 预测一次复用四配置）
  B1 短周期三策略（等权基线 / LightGBM+优化器 / Conformal objective_penalty+优化器）
  B2 短周期持仓数敏感性（LightGBM 等权 Top 10/20/30/50）
  B3 短周期随机种子稳定性（LightGBM 等权 Top20，5 个种子）
  C1 Split CP 覆盖率（α∈{0.05,0.1,0.2}）
  C2 短周期 Conformal 四方案 IR
  C3 置信度分桶 IR 与方向命中率

所有结果来自真实 embargo 重跑，不沿用旧（泄漏）CSV。
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pit_run_leakfree import rolling_predict_embargo  # noqa: E402  复用去前瞻核心

from src.backtest import BaselineBacktestEngine  # noqa: E402
from src.backtest.metrics import (  # noqa: E402
    TRADING_DAYS_PER_YEAR,
    compute_performance_metrics,
)
from src.factors import FactorEngine  # noqa: E402
from src.models.conformal import SplitConformalPredictor  # noqa: E402
from src.portfolio import ConstrainedPortfolioOptimizer, OptimizationConfig  # noqa: E402
from src.portfolio.uncertainty_aware_optimizer import (  # noqa: E402
    UncertaintyAwareConfig,
    UncertaintyAwarePortfolioOptimizer,
)

PROC = ROOT / "data" / "processed"
LOGDIR = ROOT / "logs"
HORIZON = 20
LABEL = "label_excess_return_20d"
EMBARGO = 20

# 短周期口径（与 final_strategy_comparison 对齐）
SHORT_PANEL = "hs300_panel_2024_2025_v2.csv"
SHORT_TRAIN_MONTHS = 6
SHORT_MIN_ROWS = 500
SHORT_TEST_START = "20240701"
SHORT_TEST_END = "20250530"
SHORT_GROUPS = ("value", "quality", "technical", "liquidity", "external")

# 长周期口径
LONG_PANEL = "hs300_factor_panel_pit_2015_2024.csv"
LONG_TRAIN_MONTHS = 12
LONG_MIN_ROWS = 1500


logger = logging.getLogger("embargo")


def setup_logging() -> None:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOGDIR / "embargo_recompute.log", mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.handlers = [fh, sh]


def log(msg: str) -> None:
    logger.info(msg)


# --------------------------------------------------------------------------- #
# 通用工具
# --------------------------------------------------------------------------- #
def _base_factory(seed: int = 42):
    def factory():
        return lgb.LGBMRegressor(
            objective="regression", n_estimators=300, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=-1, verbose=-1,
        )
    return factory


def _residual_factory(seed: int = 43):
    def factory():
        return lgb.LGBMRegressor(
            objective="regression", n_estimators=150, learning_rate=0.05,
            num_leaves=15, subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=-1, verbose=-1,
        )
    return factory


def rolling_predict_embargo_seeded(frame, features, train_months, min_train_rows,
                                   embargo=EMBARGO, seed=42):
    """rolling_predict_embargo 的种子可调版本（仅 B3 用；逻辑与原函数一致）。"""
    frame = frame.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"].astype(str), format="%Y%m%d")
    frame = frame.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    dates = sorted(frame["trade_date"].unique())
    di = {d: i for i, d in enumerate(dates)}
    month_ends = (frame[["trade_date"]].drop_duplicates()
                  .assign(m=lambda d: d["trade_date"].dt.to_period("M"))
                  .groupby("m")["trade_date"].max().tolist())
    preds = []
    for idx in range(train_months, len(month_ends)):
        pdate = month_ends[idx]; tstart = month_ends[idx - train_months]
        cut_i = di[pdate] - embargo
        if cut_i <= 0:
            continue
        tr = frame[(frame["trade_date"] >= tstart) & (frame["trade_date"] <= dates[cut_i])].dropna(subset=[LABEL])
        if len(tr) < min_train_rows:
            continue
        pf = frame[frame["trade_date"] == pdate].copy()
        if pf.empty:
            continue
        model = lgb.LGBMRegressor(objective="regression", n_estimators=300, learning_rate=0.05,
                                  num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                                  random_state=seed, n_jobs=-1, verbose=-1)
        model.fit(tr[features], tr[LABEL])
        pf["ml_score"] = model.predict(pf[features])
        preds.append(pf[["trade_date", "ts_code", "ml_score"]])
    out = pd.concat(preds, ignore_index=True)
    out["trade_date"] = out["trade_date"].dt.strftime("%Y%m%d")
    return out


def rolling_conformal_embargo(frame, features, train_months, min_train_rows,
                              alpha=0.1, calibration_ratio=0.3, locally_adaptive=True,
                              embargo=EMBARGO, seed=42):
    """embargo 口径的滚动 Split Conformal 预测（复用 SplitConformalPredictor）。

    训练截止 = 预测日 − embargo 交易日，避免 20 日前瞻标签泄漏。
    返回 [trade_date, ts_code, label, ml_score, ci_lower, ci_upper, ci_half_width, confidence]。
    """
    frame = frame.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"].astype(str), format="%Y%m%d")
    frame = frame.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    dates = sorted(frame["trade_date"].unique())
    di = {d: i for i, d in enumerate(dates)}
    month_ends = (frame[["trade_date"]].drop_duplicates()
                  .assign(m=lambda d: d["trade_date"].dt.to_period("M"))
                  .groupby("m")["trade_date"].max().tolist())
    blocks = []
    diag_rows = []
    for idx in range(train_months, len(month_ends)):
        pdate = month_ends[idx]; tstart = month_ends[idx - train_months]
        cut_i = di[pdate] - embargo
        if cut_i <= 0:
            continue
        tr = frame[(frame["trade_date"] >= tstart) & (frame["trade_date"] <= dates[cut_i])].dropna(subset=[LABEL])
        if len(tr) < min_train_rows:
            continue
        pf = frame[frame["trade_date"] == pdate].copy()
        if pf.empty:
            continue
        predictor = SplitConformalPredictor(
            base_model_factory=_base_factory(seed),
            residual_model_factory=_residual_factory(seed + 1) if locally_adaptive else None,
            calibration_ratio=calibration_ratio,
            alpha=alpha,
            random_state=seed,
        )
        predictor.fit(tr[features], tr[LABEL])
        y_hat, lower, upper, half_width = predictor.predict(pf[features])
        confidence = predictor.confidence(pf[features])
        block = pf[["trade_date", "ts_code", LABEL]].copy()
        block["ml_score"] = y_hat
        block["ci_lower"] = lower
        block["ci_upper"] = upper
        block["ci_half_width"] = half_width
        block["confidence"] = confidence
        blocks.append(block)
        diag_rows.append({"trade_date": pdate, "alpha": alpha,
                          "quantile": predictor.calibration_.quantile,
                          "n_calib": predictor.calibration_.n_calib,
                          "n_predict": len(pf)})
    out = pd.concat(blocks, ignore_index=True)
    out["trade_date"] = out["trade_date"].dt.strftime("%Y%m%d")
    diag = pd.DataFrame(diag_rows)
    return out, diag


def slice_to_window(result, start, end):
    """截取 [start,end] 区间并在首日归一后重算指标。"""
    nav = result.nav_frame.copy()
    nav["trade_date"] = pd.to_datetime(nav["trade_date"])
    s = pd.to_datetime(start, format="%Y%m%d")
    e = pd.to_datetime(end, format="%Y%m%d")
    nav = nav[(nav["trade_date"] >= s) & (nav["trade_date"] <= e)].copy()
    if not nav.empty:
        nav["portfolio_nav"] = nav["portfolio_nav"] / nav["portfolio_nav"].iloc[0]
        nav["benchmark_nav"] = nav["benchmark_nav"] / nav["benchmark_nav"].iloc[0]
        nav["excess_nav"] = nav["portfolio_nav"] / nav["benchmark_nav"]
        nav["trade_date"] = nav["trade_date"].dt.strftime("%Y-%m-%d")
    pos = result.positions.copy()
    if not pos.empty and "rebalance_date" in pos.columns:
        pos["rebalance_date"] = pd.to_datetime(pos["rebalance_date"])
        pos = pos[(pos["rebalance_date"] >= s) & (pos["rebalance_date"] <= e)].copy()
        pos["rebalance_date"] = pos["rebalance_date"].dt.strftime("%Y-%m-%d")
    metrics = compute_performance_metrics(nav, positions=pos)
    return nav, pos, metrics


def metric_row(metrics: pd.DataFrame) -> dict:
    m = metrics.iloc[0]
    def g(k):
        v = m.get(k, np.nan)
        return float(v) if pd.notna(v) else np.nan
    return {
        "information_ratio": g("information_ratio"),
        "sharpe_ratio": g("sharpe_ratio"),
        "annual_excess_return": g("annual_excess_return"),
        "max_drawdown": g("max_drawdown"),
        "annual_turnover": g("annual_turnover"),
        "tracking_error": g("tracking_error"),
    }


def cross_sectional_zscore(series, groups):
    grouped = series.groupby(groups)
    means = grouped.transform("mean")
    stds = grouped.transform("std").replace(0.0, np.nan)
    return (series - means) / stds


def build_equal_weight_score_panel(panel, feature_columns, test_start):
    """多因子截面 z-score 等权合成 score（baseline 等权口径），训练区间内置 NaN。"""
    frame = panel.copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    grouper = frame["trade_date"]
    comps = []
    for col in feature_columns:
        if col not in frame.columns:
            continue
        z = cross_sectional_zscore(pd.to_numeric(frame[col], errors="coerce"), grouper)
        if col == "volatility_20":
            z = -z
        comps.append(z)
    frame["score"] = pd.concat(comps, axis=1).mean(axis=1, skipna=True) if comps else np.nan
    frame.loc[frame["trade_date"] < test_start, "score"] = np.nan
    return frame


# --------------------------------------------------------------------------- #
# A. 长周期约束消融
# --------------------------------------------------------------------------- #
def run_A():
    log("\n========== A. 长周期约束消融 (embargo=20, PIT factor 面板) ==========")
    panel = pd.read_csv(PROC / LONG_PANEL)
    panel["trade_date"] = panel["trade_date"].astype(str)
    features = FactorEngine.resolve_feature_columns(available_columns=panel.columns.tolist())
    log(f"[A] 面板 rows={len(panel)} 特征={features}")
    t0 = time.perf_counter()
    pred = rolling_predict_embargo(panel, features,
                                   train_months=LONG_TRAIN_MONTHS, min_train_rows=LONG_MIN_ROWS)
    log(f"[A] embargo 预测完成 用时 {time.perf_counter()-t0:.1f}s, 预测样本 {len(pred)}, "
        f"预测期 {pred['trade_date'].min()}~{pred['trade_date'].max()}")
    pred.to_csv(PROC / "lightgbm_predictions_embargo_2015_2024.csv", index=False, encoding="utf-8-sig")
    merged = panel.merge(pred, on=["trade_date", "ts_code"], how="left")
    merged["score"] = merged["ml_score"]

    BIG = 10.0
    configs = {
        "full_constraints": OptimizationConfig(),
        "no_tracking_error": OptimizationConfig(max_tracking_error=BIG),
        "no_industry_deviation": OptimizationConfig(max_industry_deviation=BIG),
        "no_turnover": OptimizationConfig(max_turnover=BIG),
    }
    rows = []
    for name, cfg in configs.items():
        engine = BaselineBacktestEngine(top_n=20, use_optimizer=True, optimization_config=cfg)
        res = engine.run(merged)
        mr = metric_row(res.metrics)
        mr["config"] = name
        rows.append(mr)
        log(f"[A] {name}: IR={mr['information_ratio']:.4f} "
            f"年化超额={mr['annual_excess_return']:.4f} MDD={mr['max_drawdown']:.4f} "
            f"年化换手={mr['annual_turnover']:.4f}")
    df = pd.DataFrame(rows)[["config", "information_ratio", "annual_excess_return",
                             "max_drawdown", "annual_turnover", "tracking_error", "sharpe_ratio"]]
    df.to_csv(LOGDIR / "embargo_A_constraint_ablation.csv", index=False, encoding="utf-8-sig")
    return df


# --------------------------------------------------------------------------- #
# B. 短周期
# --------------------------------------------------------------------------- #
def run_B(panel, features):
    log("\n========== B. 短周期 (embargo=20, train_months=6) ==========")
    opt_cfg = OptimizationConfig(max_tracking_error=0.08, max_industry_deviation=0.02,
                                 max_weight=0.05, max_turnover=0.20)

    # ---- 预测一次：LightGBM embargo (seed 42)，B1②/B2 复用 ----
    t0 = time.perf_counter()
    lgbm_pred = rolling_predict_embargo(panel, features,
                                        train_months=SHORT_TRAIN_MONTHS, min_train_rows=SHORT_MIN_ROWS)
    log(f"[B] LightGBM embargo 预测完成 用时 {time.perf_counter()-t0:.1f}s 样本 {len(lgbm_pred)} "
        f"期 {lgbm_pred['trade_date'].min()}~{lgbm_pred['trade_date'].max()}")
    lgbm_pred.to_csv(PROC / "lightgbm_predictions_embargo_2024_2025.csv", index=False, encoding="utf-8-sig")
    lgbm_panel = panel.merge(lgbm_pred, on=["trade_date", "ts_code"], how="left")
    lgbm_panel["score"] = lgbm_panel["ml_score"]

    # ===== B1 三策略 =====
    log("\n----- B1 三策略 -----")
    b1 = []

    # ① 等权基线（无优化器，Top20）
    eq_panel = build_equal_weight_score_panel(panel, features, SHORT_TEST_START)
    eng = BaselineBacktestEngine(top_n=20, rebalance_frequency="M", factor_columns=features,
                                 use_optimizer=False, fee_rate=0.001, slippage_rate=0.001)
    res = eng.run(eq_panel)
    _, _, met = slice_to_window(res, SHORT_TEST_START, SHORT_TEST_END)
    mr = metric_row(met); mr["strategy"] = "equal_weight_baseline"; b1.append(mr)
    log(f"[B1①] 等权基线: IR={mr['information_ratio']:.4f} Sharpe={mr['sharpe_ratio']:.4f} "
        f"年化超额={mr['annual_excess_return']:.4f} MDD={mr['max_drawdown']:.4f} 年化换手={mr['annual_turnover']:.4f}")

    # ② LightGBM + 优化器
    eng = BaselineBacktestEngine(top_n=20, rebalance_frequency="M", factor_columns=features,
                                 use_optimizer=True, optimization_config=opt_cfg,
                                 fee_rate=0.001, slippage_rate=0.001)
    res = eng.run(lgbm_panel)
    _, _, met = slice_to_window(res, SHORT_TEST_START, SHORT_TEST_END)
    mr = metric_row(met); mr["strategy"] = "lgbm_optimizer"; b1.append(mr)
    log(f"[B1②] LightGBM+优化器: IR={mr['information_ratio']:.4f} Sharpe={mr['sharpe_ratio']:.4f} "
        f"年化超额={mr['annual_excess_return']:.4f} MDD={mr['max_drawdown']:.4f} 年化换手={mr['annual_turnover']:.4f}")

    # ③ Conformal objective_penalty + 优化器
    t0 = time.perf_counter()
    conf_pred, conf_diag = rolling_conformal_embargo(
        panel, features, train_months=SHORT_TRAIN_MONTHS, min_train_rows=SHORT_MIN_ROWS,
        alpha=0.1, calibration_ratio=0.3, locally_adaptive=True)
    log(f"[B1③] Conformal embargo 预测完成 用时 {time.perf_counter()-t0:.1f}s 样本 {len(conf_pred)}")
    conf_pred.to_csv(PROC / "conformal_predictions_embargo_2024_2025_alpha010.csv",
                     index=False, encoding="utf-8-sig")
    conf_diag.to_csv(PROC / "conformal_diag_embargo_2024_2025_alpha010.csv",
                     index=False, encoding="utf-8-sig")
    conf_panel = panel.merge(
        conf_pred[["trade_date", "ts_code", "ml_score", "ci_lower", "ci_upper",
                   "ci_half_width", "confidence"]],
        on=["trade_date", "ts_code"], how="left")
    conf_panel["score"] = conf_panel["ml_score"]
    eng = BaselineBacktestEngine(top_n=20, rebalance_frequency="M", factor_columns=features,
                                 use_optimizer=True, optimization_config=opt_cfg,
                                 fee_rate=0.001, slippage_rate=0.001)
    eng.optimizer = UncertaintyAwarePortfolioOptimizer(
        config=opt_cfg, uncertainty_config=UncertaintyAwareConfig(
            weighting_scheme="objective_penalty", gamma=0.1))
    res = eng.run(conf_panel)
    _, _, met = slice_to_window(res, SHORT_TEST_START, SHORT_TEST_END)
    mr = metric_row(met); mr["strategy"] = "conformal_objpenalty_optimizer"; b1.append(mr)
    log(f"[B1③] Conformal惩罚+优化器: IR={mr['information_ratio']:.4f} Sharpe={mr['sharpe_ratio']:.4f} "
        f"年化超额={mr['annual_excess_return']:.4f} MDD={mr['max_drawdown']:.4f} 年化换手={mr['annual_turnover']:.4f}")

    b1_df = pd.DataFrame(b1)[["strategy", "information_ratio", "sharpe_ratio",
                              "annual_excess_return", "max_drawdown", "annual_turnover"]]
    b1_df.to_csv(LOGDIR / "embargo_B1_three_strategies.csv", index=False, encoding="utf-8-sig")

    # ===== B2 持仓数敏感性（LightGBM 等权，无优化器） =====
    log("\n----- B2 持仓数敏感性 (LightGBM 等权, 无优化器) -----")
    b2 = []
    for tn in (10, 20, 30, 50):
        eng = BaselineBacktestEngine(top_n=tn, rebalance_frequency="M", factor_columns=features,
                                     use_optimizer=False, fee_rate=0.001, slippage_rate=0.001)
        res = eng.run(lgbm_panel)
        _, _, met = slice_to_window(res, SHORT_TEST_START, SHORT_TEST_END)
        ir = metric_row(met)["information_ratio"]
        b2.append({"top_n": tn, "information_ratio": ir})
        log(f"[B2] Top{tn}: IR={ir:.4f}")
    b2_df = pd.DataFrame(b2)
    b2_df.to_csv(LOGDIR / "embargo_B2_topn_sensitivity.csv", index=False, encoding="utf-8-sig")

    # ===== B3 随机种子稳定性（LightGBM 等权 Top20） =====
    log("\n----- B3 随机种子稳定性 (LightGBM 等权 Top20) -----")
    seeds = [7, 42, 123, 2024, 314159]
    b3 = []
    for sd in seeds:
        pred_sd = rolling_predict_embargo_seeded(panel, features,
                                                 train_months=SHORT_TRAIN_MONTHS,
                                                 min_train_rows=SHORT_MIN_ROWS, seed=sd)
        msd = panel.merge(pred_sd, on=["trade_date", "ts_code"], how="left")
        msd["score"] = msd["ml_score"]
        eng = BaselineBacktestEngine(top_n=20, rebalance_frequency="M", factor_columns=features,
                                     use_optimizer=False, fee_rate=0.001, slippage_rate=0.001)
        res = eng.run(msd)
        _, _, met = slice_to_window(res, SHORT_TEST_START, SHORT_TEST_END)
        ir = metric_row(met)["information_ratio"]
        b3.append({"seed": sd, "information_ratio": ir})
        log(f"[B3] seed={sd}: IR={ir:.4f}")
    b3_df = pd.DataFrame(b3)
    irs = b3_df["information_ratio"].to_numpy()
    b3_stats = {"mean": float(np.mean(irs)), "std": float(np.std(irs, ddof=1)),
                "cv": float(np.std(irs, ddof=1) / np.mean(irs)) if np.mean(irs) != 0 else np.nan}
    log(f"[B3] IR 均值={b3_stats['mean']:.4f} 标准差={b3_stats['std']:.4f} 变异系数={b3_stats['cv']:.4f}")
    b3_df.to_csv(LOGDIR / "embargo_B3_seed_stability.csv", index=False, encoding="utf-8-sig")

    return b1_df, b2_df, b3_df, b3_stats, conf_pred, conf_panel, opt_cfg


# --------------------------------------------------------------------------- #
# C. 短周期 Conformal
# --------------------------------------------------------------------------- #
def compute_coverage(pred):
    f = pred.copy()
    f[LABEL] = pd.to_numeric(f[LABEL], errors="coerce")
    v = f.dropna(subset=[LABEL, "ci_lower", "ci_upper"])
    if v.empty:
        return np.nan, 0
    cov = float(((v[LABEL] >= v["ci_lower"]) & (v[LABEL] <= v["ci_upper"])).mean())
    return cov, len(v)


def run_C(panel, features, conf_pred_a010, conf_panel_a010, opt_cfg):
    log("\n========== C. 短周期 Conformal (embargo=20) ==========")

    # ===== C1 覆盖率 α∈{0.05,0.1,0.2} =====
    log("\n----- C1 Split CP 覆盖率 -----")
    c1 = []
    for a in (0.05, 0.1, 0.2):
        if abs(a - 0.1) < 1e-9:
            pred = conf_pred_a010
        else:
            pred, _ = rolling_conformal_embargo(
                panel, features, train_months=SHORT_TRAIN_MONTHS, min_train_rows=SHORT_MIN_ROWS,
                alpha=a, calibration_ratio=0.3, locally_adaptive=True)
            pred.to_csv(PROC / f"conformal_predictions_embargo_2024_2025_alpha{int(a*100):03d}.csv",
                        index=False, encoding="utf-8-sig")
        cov, n = compute_coverage(pred)
        target = 1.0 - a
        c1.append({"alpha": a, "target_coverage": target, "empirical_coverage": cov,
                   "deviation_pp": (cov - target) * 100.0, "n": n})
        log(f"[C1] α={a}: 目标={target:.2%} 实测={cov:.4f} 偏差={ (cov-target)*100:.2f}pp n={n}")
    c1_df = pd.DataFrame(c1)
    c1_df.to_csv(LOGDIR / "embargo_C1_coverage.csv", index=False, encoding="utf-8-sig")

    # ===== C2 四方案 IR (α=0.1) =====
    log("\n----- C2 四方案 IR -----")
    c2 = []
    schemes = {
        "baseline": None,
        "alpha_scale": UncertaintyAwareConfig(weighting_scheme="alpha_scale", beta=1.0),
        "candidate_filter": UncertaintyAwareConfig(weighting_scheme="candidate_filter", top_pct=0.7),
        "objective_penalty": UncertaintyAwareConfig(weighting_scheme="objective_penalty", gamma=0.1),
    }
    for name, ucfg in schemes.items():
        eng = BaselineBacktestEngine(top_n=20, rebalance_frequency="M", factor_columns=features,
                                     use_optimizer=True, optimization_config=opt_cfg,
                                     fee_rate=0.001, slippage_rate=0.001)
        if ucfg is None:
            eng.optimizer = ConstrainedPortfolioOptimizer(config=opt_cfg)
        else:
            eng.optimizer = UncertaintyAwarePortfolioOptimizer(config=opt_cfg, uncertainty_config=ucfg)
        res = eng.run(conf_panel_a010)
        _, _, met = slice_to_window(res, SHORT_TEST_START, SHORT_TEST_END)
        mr = metric_row(met)
        c2.append({"scheme": name, "information_ratio": mr["information_ratio"],
                   "annual_excess_return": mr["annual_excess_return"],
                   "max_drawdown": mr["max_drawdown"], "annual_turnover": mr["annual_turnover"]})
        log(f"[C2] {name}: IR={mr['information_ratio']:.4f} 年化超额={mr['annual_excess_return']:.4f}")
    c2_df = pd.DataFrame(c2)
    c2_df.to_csv(LOGDIR / "embargo_C2_schemes.csv", index=False, encoding="utf-8-sig")

    # ===== C3 置信度分桶 IR 与方向命中率 (α=0.1) =====
    log("\n----- C3 置信度分桶 -----")
    f = conf_pred_a010.copy()
    f[LABEL] = pd.to_numeric(f[LABEL], errors="coerce")
    f = f.dropna(subset=[LABEL, "confidence"])
    lo_q = f["confidence"].quantile(0.3)
    hi_q = f["confidence"].quantile(0.7)

    def bucket(c):
        if c <= lo_q:
            return "bottom_30"
        if c >= hi_q:
            return "top_30"
        return "mid_40"

    f["bucket"] = f["confidence"].apply(bucket)
    c3 = []
    for bk in ("top_30", "mid_40", "bottom_30"):
        g = f[f["bucket"] == bk]
        if g.empty:
            continue
        mean_l = float(g[LABEL].mean())
        std_l = float(g[LABEL].std(ddof=0))
        ir = mean_l / std_l * np.sqrt(TRADING_DAYS_PER_YEAR / HORIZON) if std_l else np.nan
        hit = float((g[LABEL] > 0).mean())
        c3.append({"bucket": bk, "n": int(len(g)), "ir": ir, "hit_rate": hit,
                   "mean_label": mean_l})
        log(f"[C3] {bk}: n={len(g)} IR={ir:.4f} 方向命中率={hit:.4f}")
    c3_df = pd.DataFrame(c3)
    c3_df.to_csv(LOGDIR / "embargo_C3_confidence_buckets.csv", index=False, encoding="utf-8-sig")

    return c1_df, c2_df, c3_df


# --------------------------------------------------------------------------- #
def main():
    setup_logging()
    log("\n" + "=" * 70)
    log(f"embargo 重算开始 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)

    short_panel = pd.read_csv(PROC / SHORT_PANEL)
    short_panel["trade_date"] = short_panel["trade_date"].astype(str)
    short_features = FactorEngine.resolve_feature_columns(
        feature_groups=list(SHORT_GROUPS), available_columns=short_panel.columns.tolist())
    log(f"[short] 面板 rows={len(short_panel)} 特征({len(short_features)})={short_features}")

    A_df = run_A()
    b1_df, b2_df, b3_df, b3_stats, conf_pred, conf_panel, opt_cfg = run_B(short_panel, short_features)
    c1_df, c2_df, c3_df = run_C(short_panel, short_features, conf_pred, conf_panel, opt_cfg)

    summary = {
        "A_constraint_ablation": A_df.to_dict(orient="records"),
        "B1_three_strategies": b1_df.to_dict(orient="records"),
        "B2_topn_sensitivity": b2_df.to_dict(orient="records"),
        "B3_seed_stability": {"rows": b3_df.to_dict(orient="records"), "stats": b3_stats},
        "C1_coverage": c1_df.to_dict(orient="records"),
        "C2_schemes": c2_df.to_dict(orient="records"),
        "C3_confidence_buckets": c3_df.to_dict(orient="records"),
    }
    (LOGDIR / "embargo_recompute_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log("\n[完成] 汇总写入 logs/embargo_recompute_summary.json")


if __name__ == "__main__":
    main()
