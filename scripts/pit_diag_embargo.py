"""诊断：在 LightGBM 滚动训练中加入 purge/embargo，检验“标签前瞻重叠”是否
为残余 IR 虚高的主因。不修改任何共享代码，仅作离线诊断。

embargo：训练样本的标签前瞻窗口(20 交易日)不得越过 predict_date，
即 训练 trade_date <= 第 (idx(predict_date)-20) 个交易日。
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import BaselineBacktestEngine  # noqa: E402
from src.factors import FactorEngine  # noqa: E402
from src.portfolio import OptimizationConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HORIZON = 20
LABEL = "label_excess_return_20d"


def rolling_predict_embargo(frame: pd.DataFrame, features: list[str], train_months: int = 12,
                            min_train_rows: int = 1500, embargo: int = HORIZON) -> pd.DataFrame:
    frame = frame.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"].astype(str), format="%Y%m%d")
    frame = frame.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    dates = sorted(frame["trade_date"].unique())
    date_idx = {d: i for i, d in enumerate(dates)}
    month_ends = (
        frame[["trade_date"]].drop_duplicates()
        .assign(m=lambda d: d["trade_date"].dt.to_period("M"))
        .groupby("m")["trade_date"].max().tolist()
    )
    preds = []
    for idx in range(train_months, len(month_ends)):
        predict_date = month_ends[idx]
        train_start = month_ends[idx - train_months]
        cut_i = date_idx[predict_date] - embargo
        if cut_i <= 0:
            continue
        embargo_cut = dates[cut_i]
        tr = frame[(frame["trade_date"] >= train_start) & (frame["trade_date"] <= embargo_cut)]
        tr = tr.dropna(subset=[LABEL])
        if len(tr) < min_train_rows:
            continue
        pf = frame[frame["trade_date"] == predict_date].copy()
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


def main() -> None:
    panel = pd.read_csv(ROOT / "data/processed/hs300_factor_panel_pit_2015_2024.csv")
    panel["trade_date"] = panel["trade_date"].astype(str)
    features = FactorEngine.resolve_feature_columns(available_columns=panel.columns.tolist())
    print("features:", features)

    pred = rolling_predict_embargo(panel, features)
    merged = panel.merge(pred, on=["trade_date", "ts_code"], how="left")
    merged["score"] = merged["ml_score"]

    engine = BaselineBacktestEngine(top_n=20, use_optimizer=True,
                                    optimization_config=OptimizationConfig())
    res = engine.run(merged)
    m = res.metrics.iloc[0]
    print("\n===== embargo 诊断结果 (LightGBM 因子) =====")
    for c in ["information_ratio", "annual_excess_return", "annual_return", "max_drawdown", "tracking_error"]:
        print(f"{c}: {m[c]:.4f}")
    print("final nav:", round(res.nav_frame['portfolio_nav'].iloc[-1], 3))
    res.nav_frame.to_csv(ROOT / "data/processed/lightgbm_nav_pit_embargo_diag.csv", index=False)


if __name__ == "__main__":
    main()
