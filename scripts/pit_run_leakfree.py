"""产出“点位 + 去前瞻(embargo)”的 LightGBM 两配置净值/指标，作为完全去偏版本。

仅离线脚本，不修改任何共享代码。embargo=20 交易日，保证训练标签前瞻窗口
不越过 predict_date。基线为无模型配置，去前瞻不影响，直接沿用 PIT 基线。
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import BaselineBacktestEngine  # noqa: E402
from src.backtest.metrics import compute_performance_metrics  # noqa: E402
from src.factors import FactorEngine  # noqa: E402
from src.portfolio import OptimizationConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
HORIZON = 20
LABEL = "label_excess_return_20d"


def rolling_predict_embargo(frame, features, train_months=12, min_train_rows=1500, embargo=HORIZON):
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
                                  random_state=42, n_jobs=-1, verbose=-1)
        model.fit(tr[features], tr[LABEL])
        pf["ml_score"] = model.predict(pf[features])
        preds.append(pf[["trade_date", "ts_code", "ml_score"]])
    out = pd.concat(preds, ignore_index=True)
    out["trade_date"] = out["trade_date"].dt.strftime("%Y%m%d")
    return out


def run_config(panel_file: str, use_external: bool, tag: str) -> dict:
    panel = pd.read_csv(PROC / panel_file)
    panel["trade_date"] = panel["trade_date"].astype(str)
    extra = FactorEngine.feature_groups()["external"] if use_external else None
    features = FactorEngine.resolve_feature_columns(extra_columns=extra,
                                                    available_columns=panel.columns.tolist())
    pred = rolling_predict_embargo(panel, features)
    merged = panel.merge(pred, on=["trade_date", "ts_code"], how="left")
    merged["score"] = merged["ml_score"]
    engine = BaselineBacktestEngine(top_n=20, use_optimizer=True, optimization_config=OptimizationConfig())
    res = engine.run(merged)
    res.nav_frame.to_csv(PROC / f"{tag}_nav_pit_leakfree_2015_2024.csv", index=False, encoding="utf-8-sig")
    res.metrics.to_csv(PROC / f"{tag}_metrics_pit_leakfree_2015_2024.csv", index=False, encoding="utf-8-sig")
    m = res.metrics.iloc[0]
    return {"config": tag, "information_ratio": round(float(m["information_ratio"]), 4),
            "annual_excess_return": round(float(m["annual_excess_return"]), 4),
            "max_drawdown": round(float(m["max_drawdown"]), 4),
            "tracking_error": round(float(m["tracking_error"]), 4),
            "final_nav": round(float(res.nav_frame["portfolio_nav"].iloc[-1]), 3)}


def main() -> None:
    rows = []
    # 基线(去前瞻不影响)：直接读取已生成 PIT 基线指标
    bm = pd.read_csv(PROC / "baseline_metrics_pit_2015_2024.csv").iloc[0]
    bnav = pd.read_csv(PROC / "baseline_nav_pit_2015_2024.csv")["portfolio_nav"].iloc[-1]
    rows.append({"config": "baseline", "information_ratio": round(float(bm["information_ratio"]), 4),
                 "annual_excess_return": round(float(bm["annual_excess_return"]), 4),
                 "max_drawdown": round(float(bm["max_drawdown"]), 4),
                 "tracking_error": round(float(bm["tracking_error"]), 4),
                 "final_nav": round(float(bnav), 3)})
    rows.append(run_config("hs300_factor_panel_pit_2015_2024.csv", False, "lightgbm"))
    rows.append(run_config("hs300_factor_panel_external_pit_2015_2024.csv", True, "lightgbm_external"))
    summary = pd.DataFrame(rows)
    summary.to_csv(ROOT / "logs" / "pit_backtest_summary_leakfree.csv", index=False)
    print("\n===== 点位 + 去前瞻(embargo) 三配置汇总 =====")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
