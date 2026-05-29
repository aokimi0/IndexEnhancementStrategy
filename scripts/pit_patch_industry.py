"""给点位面板补真实行业（baostock，免费），替换 56% 的"未知行业"占位。

baostock `query_stock_industry()` 一次性返回全市场行业分类（申万），
据此把 PIT 面板的 industry_name 由占位替换为真实行业，缺失者保留"未知行业"。
仅离线脚本，只写 data/processed 下的 PIT 面板。
"""

from __future__ import annotations

from pathlib import Path

import baostock as bs
import pandas as pd

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
PANELS = [
    "hs300_factor_panel_pit_2015_2024.csv",
    "hs300_factor_panel_external_pit_2015_2024.csv",
]


def _to_ts(code: str) -> str:
    """sh.600000 -> 600000.SH。"""
    if "." not in code:
        return code
    mkt, num = code.split(".")
    return f"{num}.{mkt.upper()}"


def fetch_industry_map() -> dict[str, str]:
    """一次性取全市场申万行业，返回 ts_code -> industry。"""
    bs.login()
    rs = bs.query_stock_industry()
    rows = []
    while (rs.error_code == "0") & rs.next():
        rows.append(rs.get_row_data())
    bs.logout()
    df = pd.DataFrame(rows, columns=rs.fields)
    df = df[df["industry"].astype(str).str.len() > 0]
    df["ts_code"] = df["code"].map(_to_ts)
    return dict(zip(df["ts_code"], df["industry"]))


def main() -> None:
    """补行业并覆盖保存 PIT 面板。"""
    imap = fetch_industry_map()
    print(f"baostock 行业映射条数: {len(imap)}")
    for f in PANELS:
        path = PROC / f
        if not path.exists():
            print(f"[跳过] 不存在: {f}")
            continue
        df = pd.read_csv(path)
        mapped = df["ts_code"].map(imap)
        before = (df["industry_name"] == "未知行业").mean() if "industry_name" in df.columns else 1.0
        df["industry_name"] = mapped.fillna(
            df["industry_name"] if "industry_name" in df.columns else "未知行业"
        ).fillna("未知行业")
        after = (df["industry_name"] == "未知行业").mean()
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"{f}: 未知行业占比 {before:.2%} -> {after:.2%}，已保存")


if __name__ == "__main__":
    main()
