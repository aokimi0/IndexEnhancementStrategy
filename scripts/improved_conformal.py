"""挑战二改进方案：稳定超越不加权约束优化器的不确定性感知设计。

本脚本在与第四章完全一致的 leak-free 短样本口径（训练截止 = 预测日 − 20 交易日，
SHORT_TEST = 20240701~20250530）下，实现并对比四类"真正的不确定性感知"方案，
对标基线为**同口径、不加权的约束优化器**（使用同一套 Conformal ml_score）：

* ``conformal_ml_baseline``：ConstrainedPortfolioOptimizer + 同口径 conformal ml_score（控制组）
* ``factor_robust_optimizer``：用简单因子合成 z-score 跑同一约束优化器（稳健腿，仅供参考）
* ``trust_gated``（方案①，主推）：用 CP 覆盖率本身作模型可信度温控，逐期 leak-free
  实测覆盖率 → 信任度 g_t，调仓日按 ``w = g_t·w_ML + (1−g_t)·w_稳健`` 混合持仓
* ``uncertainty_risk``（方案②）：把保形半宽平方并入协方差对角，让优化器与 TE 约束
  天然回避高不确定个股
* ``sign_adaptive``（方案③）：把归一化置信度当正交因子，用历史已实现数据滚动估计其
  与收益的符号与强度再注入 alpha（直接利用"倒挂"，作对照与稳健性讨论）

所有"自适应"量（覆盖率、信任度、置信-收益符号）只用 t − embargo 之前**已实现**的标签，
严格 leak-free，无全样本窥探。结果写入 logs/improved_C4_*。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from embargo_recompute import (  # noqa: E402  复用 leak-free 口径与工具
    EMBARGO,
    LABEL,
    LOGDIR,
    PROC,
    SHORT_GROUPS,
    SHORT_MIN_ROWS,
    SHORT_PANEL,
    SHORT_TEST_END,
    SHORT_TEST_START,
    SHORT_TRAIN_MONTHS,
    build_equal_weight_score_panel,
    metric_row,
    rolling_conformal_embargo,
    slice_to_window,
)

from src.backtest import BaselineBacktestEngine, BaselineBacktestResult  # noqa: E402
from src.backtest.metrics import compute_performance_metrics  # noqa: E402
from src.factors import FactorEngine  # noqa: E402
from src.portfolio import ConstrainedPortfolioOptimizer, OptimizationConfig  # noqa: E402
from src.portfolio.uncertainty_aware_optimizer import (  # noqa: E402
    UncertaintyAwareConfig,
    UncertaintyAwarePortfolioOptimizer,
)

OPT_CFG = OptimizationConfig(
    max_tracking_error=0.08,
    max_industry_deviation=0.02,
    max_weight=0.05,
    max_turnover=0.20,
)

# 信度门控参数
TARGET_COVERAGE = 0.90  # 名义覆盖率 1 − α，α = 0.1
COVERAGE_FLOOR = 0.50   # 覆盖率低于该地板时完全退向稳健腿
WARMUP_TRUST = 0.50     # 暖机期（尚无已实现覆盖样本）默认信任度
MIN_REALIZED = 50       # 计算覆盖率所需的最小已实现样本数

CONF_CACHE = PROC / "conformal_predictions_embargo_2024_2025_alpha010.csv"


def log(msg: str) -> None:
    print(msg, flush=True)


def load_short_panel() -> tuple[pd.DataFrame, list[str]]:
    """加载短周期面板与 15 因子列。"""
    panel = pd.read_csv(PROC / SHORT_PANEL)
    panel["trade_date"] = panel["trade_date"].astype(str)
    features = FactorEngine.resolve_feature_columns(
        feature_groups=list(SHORT_GROUPS), available_columns=panel.columns.tolist()
    )
    return panel, features


def get_conformal_predictions(
    panel: pd.DataFrame, features: list[str], seed: int = 42, reuse: bool = True
) -> pd.DataFrame:
    """获取 embargo 口径滚动 Split Conformal 预测；seed=42 时复用缓存。"""
    if reuse and seed == 42 and CONF_CACHE.exists():
        df = pd.read_csv(CONF_CACHE)
        df["trade_date"] = df["trade_date"].astype(str)
        return df
    pred, _ = rolling_conformal_embargo(
        panel,
        features,
        train_months=SHORT_TRAIN_MONTHS,
        min_train_rows=SHORT_MIN_ROWS,
        alpha=0.1,
        calibration_ratio=0.3,
        locally_adaptive=True,
        seed=seed,
    )
    pred["trade_date"] = pred["trade_date"].astype(str)
    return pred


def build_trading_calendar(panel: pd.DataFrame):
    """返回排序后的交易日列表与日期→序号映射。"""
    dates = sorted(
        pd.to_datetime(panel["trade_date"].astype(str), format="%Y%m%d").unique()
    )
    return dates, {pd.Timestamp(d): i for i, d in enumerate(dates)}


def make_ml_panel(panel: pd.DataFrame, conf_pred: pd.DataFrame) -> pd.DataFrame:
    """把 conformal 预测并入面板，score = ml_score。"""
    cols = ["trade_date", "ts_code", "ml_score", "ci_lower", "ci_upper", "ci_half_width", "confidence"]
    merged = panel.merge(conf_pred[cols], on=["trade_date", "ts_code"], how="left")
    merged["score"] = merged["ml_score"]
    return merged


def run_optimizer_leg(
    scored_panel: pd.DataFrame, features: list[str], optimizer
) -> BaselineBacktestResult:
    """以给定 optimizer 跑一条约束优化腿，返回完整区间回测结果。"""
    eng = BaselineBacktestEngine(
        top_n=20,
        rebalance_frequency="M",
        factor_columns=features,
        use_optimizer=True,
        optimization_config=OPT_CFG,
        fee_rate=0.001,
        slippage_rate=0.001,
    )
    eng.optimizer = optimizer
    return eng.run(scored_panel)


def compute_trust_by_date(
    conf_pred: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    calendar: list,
    date_index: dict,
) -> tuple[dict, dict]:
    """leak-free 逐调仓日信任度 g_t 与实测覆盖率。

    一只股票在预测日 p 的 20 日标签需 p+embargo 交易日后才完全实现；调仓日 t 只用
    realized_date ≤ t 的已实现行计算扩展窗口覆盖率，再线性映射为信任度
    g_t = clip((cov_t − floor)/(target − floor), 0, 1)。
    """
    frame = conf_pred.copy()
    frame[LABEL] = pd.to_numeric(frame[LABEL], errors="coerce")
    frame["pdate"] = pd.to_datetime(frame["trade_date"].astype(str), format="%Y%m%d")
    frame = frame.dropna(subset=[LABEL, "ci_lower", "ci_upper"])

    n_days = len(calendar)

    def realized_date(p: pd.Timestamp):
        i = date_index.get(pd.Timestamp(p))
        if i is None or i + EMBARGO >= n_days:
            return pd.NaT
        return calendar[i + EMBARGO]

    frame["realized_date"] = frame["pdate"].map(realized_date)
    frame["covered"] = (frame[LABEL] >= frame["ci_lower"]) & (frame[LABEL] <= frame["ci_upper"])

    trust: dict = {}
    coverage: dict = {}
    for t in rebalance_dates:
        realized = frame[frame["realized_date"].notna() & (frame["realized_date"] <= t)]
        if len(realized) < MIN_REALIZED:
            trust[t] = WARMUP_TRUST
            coverage[t] = np.nan
            continue
        cov = float(realized["covered"].mean())
        g = (cov - COVERAGE_FLOOR) / (TARGET_COVERAGE - COVERAGE_FLOOR)
        trust[t] = float(np.clip(g, 0.0, 1.0))
        coverage[t] = cov
    return trust, coverage


def blend_positions(
    pos_ml: pd.DataFrame, pos_robust: pd.DataFrame, trust_by_date: dict
) -> pd.DataFrame:
    """按逐期信任度在调仓日混合两条腿的持仓，重算换手。"""
    ml_groups = {d: dict(zip(g["ts_code"], g["weight"])) for d, g in pos_ml.groupby("rebalance_date")}
    rob_groups = {d: dict(zip(g["ts_code"], g["weight"])) for d, g in pos_robust.groupby("rebalance_date")}
    dates = sorted(set(ml_groups) | set(rob_groups))

    rows: list[dict] = []
    prev: dict = {}
    for d in dates:
        g = float(trust_by_date.get(d, WARMUP_TRUST))
        w_ml = ml_groups.get(d, {})
        w_rob = rob_groups.get(d, {})
        codes = set(w_ml) | set(w_rob)
        blended = {c: g * w_ml.get(c, 0.0) + (1.0 - g) * w_rob.get(c, 0.0) for c in codes}
        total = sum(blended.values())
        if total <= 0:
            continue
        blended = {c: v / total for c, v in blended.items()}
        turnover = 0.5 * sum(
            abs(blended.get(c, 0.0) - prev.get(c, 0.0)) for c in set(blended) | set(prev)
        )
        for c, v in blended.items():
            rows.append(
                {
                    "rebalance_date": d,
                    "ts_code": c,
                    "weight": v,
                    "turnover": turnover,
                    "ex_ante_tracking_error": np.nan,
                    "max_industry_deviation": np.nan,
                }
            )
        prev = blended
    return pd.DataFrame(rows)


def nav_result_from_positions(
    panel: pd.DataFrame, positions: pd.DataFrame
) -> BaselineBacktestResult:
    """由持仓表重算 NAV 与指标（计入双边费用与滑点）。"""
    frame = panel.copy()
    frame["trade_date"] = pd.to_datetime(
        frame["trade_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    frame = frame.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    frame["daily_return"] = pd.to_numeric(frame["daily_return"], errors="coerce")
    eng = BaselineBacktestEngine(
        top_n=20,
        use_optimizer=True,
        optimization_config=OPT_CFG,
        fee_rate=0.001,
        slippage_rate=0.001,
    )
    nav = eng._build_nav(frame=frame, positions=positions)
    metrics = compute_performance_metrics(nav, positions=positions)
    return BaselineBacktestResult(nav_frame=nav, positions=positions, metrics=metrics)


def build_sign_adaptive_panel(
    panel: pd.DataFrame,
    conf_pred: pd.DataFrame,
    calendar: list,
    date_index: dict,
    lam: float = 1.0,
    min_realized: int = 200,
) -> pd.DataFrame:
    """方案③：把置信度当正交因子，按 leak-free 滚动估计的符号/强度注入 score。

    每个调仓日 t 用 realized_date ≤ t 的已实现行估计 confidence 与 label 的相关 r_t，
    令 score_i = z_cs(ml_score_i) + lam·r_t·z_cs(confidence_i)（截面内 z-score）。
    r_t 仅依赖历史已实现数据，leak-free；样本不足时 r_t = 0（不 tilt）。
    """
    frame = conf_pred.copy()
    frame[LABEL] = pd.to_numeric(frame[LABEL], errors="coerce")
    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce")
    frame["ml_score"] = pd.to_numeric(frame["ml_score"], errors="coerce")
    frame["pdate"] = pd.to_datetime(frame["trade_date"].astype(str), format="%Y%m%d")

    n_days = len(calendar)

    def realized_date(p: pd.Timestamp):
        i = date_index.get(pd.Timestamp(p))
        if i is None or i + EMBARGO >= n_days:
            return pd.NaT
        return calendar[i + EMBARGO]

    frame["realized_date"] = frame["pdate"].map(realized_date)
    realized_pool = frame.dropna(subset=[LABEL, "confidence", "realized_date"])

    rebalance_dates = sorted(frame["pdate"].unique())
    sign_by_date: dict = {}
    for t in rebalance_dates:
        hist = realized_pool[realized_pool["realized_date"] <= t]
        if len(hist) < min_realized or hist["confidence"].std(ddof=0) == 0:
            sign_by_date[t] = 0.0
            continue
        r = float(np.corrcoef(hist["confidence"], hist[LABEL])[0, 1])
        sign_by_date[t] = 0.0 if np.isnan(r) else r

    def _zscore(s: pd.Series) -> pd.Series:
        std = s.std(ddof=0)
        if not std or np.isnan(std):
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / std

    blocks: list[pd.DataFrame] = []
    for pdate, grp in frame.groupby("pdate"):
        g = grp.copy()
        z_ml = _zscore(g["ml_score"])
        z_conf = _zscore(g["confidence"])
        r_t = sign_by_date.get(pdate, 0.0)
        g["score"] = z_ml + lam * r_t * z_conf
        blocks.append(g[["trade_date", "ts_code", "score"]])
    score_frame = pd.concat(blocks, ignore_index=True)
    score_frame["trade_date"] = score_frame["trade_date"].astype(str)

    out = panel.merge(score_frame, on=["trade_date", "ts_code"], how="left")
    return out, sign_by_date


def evaluate(result: BaselineBacktestResult) -> dict:
    """切到 SHORT_TEST 区间并取指标。"""
    _, _, met = slice_to_window(result, SHORT_TEST_START, SHORT_TEST_END)
    return metric_row(met)


def run_all_schemes(
    panel: pd.DataFrame,
    features: list[str],
    conf_pred: pd.DataFrame,
    seed: int = 42,
) -> tuple[dict, dict, dict]:
    """跑五条方案，返回 {scheme: metrics}、信任度序列、③ 符号序列。"""
    calendar, date_index = build_trading_calendar(panel)
    ml_panel = make_ml_panel(panel, conf_pred)
    eq_panel = build_equal_weight_score_panel(panel, features, SHORT_TEST_START)

    results: dict = {}

    # 控制组：不加权 conformal-ml 约束优化器
    res_baseline = run_optimizer_leg(ml_panel, features, ConstrainedPortfolioOptimizer(config=OPT_CFG))
    results["conformal_ml_baseline"] = evaluate(res_baseline)

    # 稳健腿：因子合成 z-score 约束优化器
    res_robust = run_optimizer_leg(eq_panel, features, ConstrainedPortfolioOptimizer(config=OPT_CFG))
    results["factor_robust_optimizer"] = evaluate(res_robust)

    # 方案① 信度门控稳健混合
    rebalance_dates = sorted(res_baseline.positions["rebalance_date"].unique())
    rebalance_dates = [pd.Timestamp(d) for d in rebalance_dates]
    trust_by_date, coverage_by_date = compute_trust_by_date(
        conf_pred, rebalance_dates, calendar, date_index
    )
    blended = blend_positions(res_baseline.positions, res_robust.positions, trust_by_date)
    res_gated = nav_result_from_positions(panel, blended)
    results["trust_gated"] = evaluate(res_gated)

    # 方案② 不确定性进风险项
    res_urisk = run_optimizer_leg(
        ml_panel,
        features,
        UncertaintyAwarePortfolioOptimizer(
            config=OPT_CFG,
            uncertainty_config=UncertaintyAwareConfig(
                weighting_scheme="uncertainty_risk", risk_uncertainty_coef=1.0
            ),
        ),
    )
    results["uncertainty_risk"] = evaluate(res_urisk)

    # 方案③ 方向自适应置信 tilt
    sa_panel, sign_by_date = build_sign_adaptive_panel(panel, conf_pred, calendar, date_index, lam=1.0)
    res_sign = run_optimizer_leg(sa_panel, features, ConstrainedPortfolioOptimizer(config=OPT_CFG))
    results["sign_adaptive"] = evaluate(res_sign)

    trust_series = {pd.Timestamp(k).strftime("%Y%m%d"): {"trust": v, "coverage": coverage_by_date.get(k)} for k, v in trust_by_date.items()}
    sign_series = {pd.Timestamp(k).strftime("%Y%m%d"): v for k, v in sign_by_date.items()}
    return results, trust_series, sign_series


def run_seed_stability(
    panel: pd.DataFrame, features: list[str], seeds: list[int]
) -> pd.DataFrame:
    """对 baseline / trust_gated / uncertainty_risk 跑多种子稳定性（复用 B3 口径）。

    稳健因子腿与种子无关，仅算一次；ML 腿、conformal 覆盖率随种子变化。
    """
    calendar, date_index = build_trading_calendar(panel)
    eq_panel = build_equal_weight_score_panel(panel, features, SHORT_TEST_START)
    res_robust = run_optimizer_leg(eq_panel, features, ConstrainedPortfolioOptimizer(config=OPT_CFG))

    rows = []
    for sd in seeds:
        t0 = time.perf_counter()
        conf_pred = get_conformal_predictions(panel, features, seed=sd, reuse=(sd == 42))
        ml_panel = make_ml_panel(panel, conf_pred)

        res_baseline = run_optimizer_leg(
            ml_panel, features, ConstrainedPortfolioOptimizer(config=OPT_CFG)
        )
        ir_baseline = evaluate(res_baseline)["information_ratio"]

        rebalance_dates = [pd.Timestamp(d) for d in sorted(res_baseline.positions["rebalance_date"].unique())]
        trust_by_date, _ = compute_trust_by_date(conf_pred, rebalance_dates, calendar, date_index)
        blended = blend_positions(res_baseline.positions, res_robust.positions, trust_by_date)
        ir_gated = evaluate(nav_result_from_positions(panel, blended))["information_ratio"]

        res_urisk = run_optimizer_leg(
            ml_panel,
            features,
            UncertaintyAwarePortfolioOptimizer(
                config=OPT_CFG,
                uncertainty_config=UncertaintyAwareConfig(
                    weighting_scheme="uncertainty_risk", risk_uncertainty_coef=1.0
                ),
            ),
        )
        ir_urisk = evaluate(res_urisk)["information_ratio"]

        rows.append(
            {
                "seed": sd,
                "baseline_ir": ir_baseline,
                "trust_gated_ir": ir_gated,
                "uncertainty_risk_ir": ir_urisk,
            }
        )
        log(
            f"[seed {sd}] baseline={ir_baseline:+.4f} trust_gated={ir_gated:+.4f} "
            f"uncertainty_risk={ir_urisk:+.4f} ({time.perf_counter() - t0:.1f}s)"
        )
    return pd.DataFrame(rows)


def main_seeds() -> None:
    t_start = time.perf_counter()
    log("=" * 70)
    log(f"[improved] 多种子稳定性实验开始 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)
    panel, features = load_short_panel()
    seeds = [7, 42, 123, 2024, 314159]
    df = run_seed_stability(panel, features, seeds)

    stats = {}
    for col in ("baseline_ir", "trust_gated_ir", "uncertainty_risk_ir"):
        vals = df[col].to_numpy()
        stats[col] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)),
                      "min": float(np.min(vals)), "max": float(np.max(vals))}
    win_gated = int((df["trust_gated_ir"] > df["baseline_ir"]).sum())
    win_urisk = int((df["uncertainty_risk_ir"] > df["baseline_ir"]).sum())

    df.to_csv(LOGDIR / "improved_C4_seed_stability.csv", index=False, encoding="utf-8-sig")
    log("\n[improved] ====== 多种子 IR ======")
    log(df.to_string(index=False))
    log("\n[improved] ====== 统计 ======")
    for col, s in stats.items():
        log(f"  {col}: 均值={s['mean']:+.4f} 标准差={s['std']:.4f} 区间=[{s['min']:+.4f},{s['max']:+.4f}]")
    log(f"  trust_gated 胜出种子数={win_gated}/{len(seeds)}；uncertainty_risk 胜出={win_urisk}/{len(seeds)}")

    (LOGDIR / "improved_C4_seed_stability_summary.json").write_text(
        json.dumps({"rows": df.to_dict(orient="records"), "stats": stats,
                    "win_gated": win_gated, "win_urisk": win_urisk}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(f"\n[improved] 多种子完成，用时 {time.perf_counter() - t_start:.1f}s")


def main() -> None:
    t_start = time.perf_counter()
    log("=" * 70)
    log(f"[improved] 挑战二改进方案实验开始 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)

    panel, features = load_short_panel()
    log(f"[improved] 面板 rows={len(panel)} 特征({len(features)})={features}")

    conf_pred = get_conformal_predictions(panel, features, seed=42, reuse=True)
    log(f"[improved] conformal 预测 样本={len(conf_pred)} 期数={conf_pred['trade_date'].nunique()}")

    results, trust_series, sign_series = run_all_schemes(panel, features, conf_pred, seed=42)

    order = [
        "conformal_ml_baseline",
        "factor_robust_optimizer",
        "trust_gated",
        "uncertainty_risk",
        "sign_adaptive",
    ]
    rows = []
    for name in order:
        m = results[name]
        rows.append(
            {
                "scheme": name,
                "information_ratio": m["information_ratio"],
                "sharpe_ratio": m["sharpe_ratio"],
                "annual_excess_return": m["annual_excess_return"],
                "max_drawdown": m["max_drawdown"],
                "annual_turnover": m["annual_turnover"],
                "tracking_error": m["tracking_error"],
            }
        )
    schemes_df = pd.DataFrame(rows)
    schemes_df.to_csv(LOGDIR / "improved_C4_schemes.csv", index=False, encoding="utf-8-sig")

    log("\n[improved] ====== 主结果（SHORT_TEST，对标 conformal_ml_baseline）======")
    log(schemes_df.to_string(index=False))

    baseline_ir = results["conformal_ml_baseline"]["information_ratio"]
    log("\n[improved] ====== 相对不加权基线的 IR 增量 ======")
    for name in order[2:]:
        delta = results[name]["information_ratio"] - baseline_ir
        flag = "胜" if delta > 0 else "负"
        log(f"  {name}: ΔIR={delta:+.4f} [{flag}]")

    trust_df = pd.DataFrame(
        [{"trade_date": k, "coverage": v["coverage"], "trust_g": v["trust"]} for k, v in trust_series.items()]
    )
    trust_df.to_csv(LOGDIR / "improved_C4_trust_series.csv", index=False, encoding="utf-8-sig")
    log("\n[improved] ====== 信任度/覆盖率时间序列 ======")
    log(trust_df.to_string(index=False))

    summary = {
        "schemes": schemes_df.to_dict(orient="records"),
        "baseline_ir": baseline_ir,
        "trust_series": trust_series,
        "sign_series": sign_series,
    }
    (LOGDIR / "improved_C4_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"\n[improved] 完成，用时 {time.perf_counter() - t_start:.1f}s，结果写入 logs/improved_C4_*")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="挑战二改进方案实验")
    parser.add_argument("--mode", choices=["main", "seeds"], default="main")
    cli_args = parser.parse_args()
    if cli_args.mode == "seeds":
        main_seeds()
    else:
        main()
