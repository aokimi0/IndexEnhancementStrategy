"""
CSI 300 历史数据采集脚本
数据源: BaoStock (免费, 无需 token) + AkShare (成分股名单)
时间范围: 2015-01-01 ~ 2024-12-31
产出: 4 个 CSV + 打包 zip
"""

import os
import sys
import time
import zipfile
import warnings
import logging
from pathlib import Path

import baostock as bs
import akshare as ak
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ── 路径配置 ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
OUT_DIR  = BASE_DIR / "data" / "csi300_raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2015-01-01"
END_DATE   = "2024-12-31"

# ── 工具函数 ────────────────────────────────────────────────────────────────
def login():
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {lg.error_msg}")

def logout():
    bs.logout()

def _to_df(rs):
    """BaoStock ResultData → DataFrame"""
    rows, fields = [], rs.fields
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=fields) if rows else pd.DataFrame(columns=fields)

def bs_code_to_6(code: str) -> str:
    """sh.600519 → 600519"""
    return code.split(".")[-1] if "." in code else code

def print_summary(label: str, df: pd.DataFrame, date_col: str = "date"):
    print(f"\n{'='*60}")
    print(f"[{label}]")
    print(f"  行数        : {len(df):,}")
    if "bs_code" in df.columns:
        print(f"  股票数      : {df['bs_code'].nunique():,}")
    if date_col in df.columns:
        print(f"  日期范围    : {df[date_col].min()} ~ {df[date_col].max()}")
    print(f"{'='*60}")

# ════════════════════════════════════════════════════════════════════════════
# 交付物 1: 成分股名单
# ════════════════════════════════════════════════════════════════════════════
def fetch_constituents() -> pd.DataFrame:
    print("\n>>> 正在获取沪深300成分股名单...")
    df_ak = ak.index_stock_cons_weight_csindex(symbol="000300")
    # akshare 字段: 成分券代码, 成分券名称, 交易所, 权重
    df_ak = df_ak.rename(columns={
        "成分券代码": "code_6",
        "成分券名称": "stock_name",
        "交易所":    "exchange",
        "权重":      "weight_pct",
    })
    # 生成 BaoStock 风格代码
    def to_bs_code(row):
        mkt = str(row.get("exchange", "")).strip()
        c   = str(row["code_6"]).strip().zfill(6)
        if mkt in ("上交所", "SSE") or c.startswith(("6", "9")):
            return f"sh.{c}"
        else:
            return f"sz.{c}"

    df_ak["code_6"] = df_ak["code_6"].astype(str).str.zfill(6)
    df_ak["bs_code"] = df_ak.apply(to_bs_code, axis=1)

    keep = ["bs_code", "code_6", "stock_name", "exchange", "weight_pct"]
    keep = [c for c in keep if c in df_ak.columns]
    result = df_ak[keep].reset_index(drop=True)

    out_path = OUT_DIR / "01_constituents.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print_summary("交付物1: 成分股名单", result, date_col="__none__")
    print(f"  已保存 → {out_path}")
    return result


# ════════════════════════════════════════════════════════════════════════════
# 交付物 2: 日频估值与换手
# ════════════════════════════════════════════════════════════════════════════
VALUATION_FIELDS = "date,code,turn,peTTM,pbMRQ,psTTM"

def fetch_one_valuation(bs_code: str) -> pd.DataFrame:
    rs = bs.query_history_k_data_plus(
        bs_code,
        VALUATION_FIELDS,
        start_date=START_DATE,
        end_date=END_DATE,
        frequency="d",
        adjustflag="3",
    )
    df = _to_df(rs)
    if df.empty:
        return df
    df.rename(columns={
        "date": "date",
        "code": "bs_code",
        "turn": "turn_rate",
        "peTTM": "pe_ttm",
        "pbMRQ": "pb_mrq",
        "psTTM": "ps_ttm",
    }, inplace=True)
    df["code_6"] = df["bs_code"].apply(bs_code_to_6)
    for col in ["turn_rate", "pe_ttm", "pb_mrq", "ps_ttm"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].where(df[col].notna(), other="")
    return df[["date", "bs_code", "code_6", "turn_rate", "pe_ttm", "pb_mrq", "ps_ttm"]]

def fetch_valuation(stocks: pd.DataFrame) -> pd.DataFrame:
    print("\n>>> 正在抓取日频估值与换手数据（约300只 × 10年，预计需要15~30分钟）...")
    codes = stocks["bs_code"].tolist()
    total = len(codes)
    all_dfs = []
    for i, code in enumerate(codes, 1):
        df = fetch_one_valuation(code)
        if not df.empty:
            all_dfs.append(df)
        if i % 30 == 0 or i == total:
            print(f"  进度: {i}/{total}", flush=True)
        time.sleep(0.05)

    result = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    out_path = OUT_DIR / "02_valuation_daily.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print_summary("交付物2: 日频估值与换手", result)

    # 非空比例
    if not result.empty:
        for col in ["pe_ttm", "pb_mrq"]:
            if col in result.columns:
                non_empty = result[col].replace("", pd.NA).notna().sum()
                pct = non_empty / len(result) * 100
                print(f"  {col} 非空比例: {pct:.1f}%")
    print(f"  已保存 → {out_path}")
    return result


# ════════════════════════════════════════════════════════════════════════════
# 交付物 3: 季度财务指标
# ════════════════════════════════════════════════════════════════════════════
def _quarter_dates():
    """生成 2015Q1~2024Q4 的 (start, end) 列表"""
    quarters = []
    for year in range(2015, 2025):
        for q, (m_start, m_end, d_end) in enumerate(
            [(1,3,31),(4,6,30),(7,9,30),(10,12,31)], 1
        ):
            quarters.append((
                f"{year}-{m_start:02d}-01",
                f"{year}-{m_end:02d}-{d_end:02d}",
            ))
    return quarters

def _fetch_profit(code, start, end):
    rs = bs.query_profit_data(code=code, year=start[:4], quarter=_quarter_num(start))
    df = _to_df(rs)
    return df

def _quarter_num(start_str):
    m = int(start_str[5:7])
    return (m - 1) // 3 + 1

def fetch_financials_one(bs_code: str) -> pd.DataFrame:
    """对一只股票抓全部季度财务指标并合并
    BaoStock 实际字段:
      profit_data  : roeAvg, gpMargin(毛利率), npMargin(净利率)
      operation_data: AssetTurnRatio(总资产周转率)
      growth_data  : YOYPNI(归母净利润同比)
      cashflow_data: CFOToOR(经营现金流/营业收入)
      balance_data : assetToEquity(权益乘数=总资产/净资产)
    """
    rows = []
    for year in range(2015, 2025):
        for quarter in range(1, 5):
            row = {"bs_code": bs_code, "code_6": bs_code_to_6(bs_code),
                   "year": year, "quarter": quarter,
                   "pubDate": "", "statDate": "",
                   "roe_avg": "", "gross_profit_margin": "",
                   "net_profit_margin": "", "yoy_net_profit": "",
                   "asset_turnover": "", "cfo_to_sales": "",
                   "equity_multiplier": ""}

            # profit_data: roeAvg, gpMargin, npMargin
            rs = bs.query_profit_data(code=bs_code, year=year, quarter=quarter)
            df = _to_df(rs)
            if not df.empty:
                r = df.iloc[0]
                row["pubDate"]             = r.get("pubDate", "")
                row["statDate"]            = r.get("statDate", "")
                row["roe_avg"]             = r.get("roeAvg", "")
                row["gross_profit_margin"] = r.get("gpMargin", "")
                row["net_profit_margin"]   = r.get("npMargin", "")

            # operation_data: AssetTurnRatio
            rs2 = bs.query_operation_data(code=bs_code, year=year, quarter=quarter)
            df2 = _to_df(rs2)
            if not df2.empty:
                r2 = df2.iloc[0]
                row["asset_turnover"] = r2.get("AssetTurnRatio", "")
                if not row["pubDate"]:
                    row["pubDate"]  = r2.get("pubDate", "")
                    row["statDate"] = r2.get("statDate", "")

            # growth_data: YOYPNI (归母净利润同比增长率)
            rs3 = bs.query_growth_data(code=bs_code, year=year, quarter=quarter)
            df3 = _to_df(rs3)
            if not df3.empty:
                r3 = df3.iloc[0]
                row["yoy_net_profit"] = r3.get("YOYPNI", "")

            # cashflow_data: CFOToOR (经营现金流/营业收入)
            rs4 = bs.query_cash_flow_data(code=bs_code, year=year, quarter=quarter)
            df4 = _to_df(rs4)
            if not df4.empty:
                r4 = df4.iloc[0]
                row["cfo_to_sales"] = r4.get("CFOToOR", "")

            # balance_data: assetToEquity (权益乘数)
            rs5 = bs.query_balance_data(code=bs_code, year=year, quarter=quarter)
            df5 = _to_df(rs5)
            if not df5.empty:
                r5 = df5.iloc[0]
                row["equity_multiplier"] = r5.get("assetToEquity", "")

            rows.append(row)
    return pd.DataFrame(rows)

FINANCIAL_COLS = [
    "bs_code", "code_6", "year", "quarter",
    "pubDate", "statDate",
    "roe_avg", "gross_profit_margin", "net_profit_margin",
    "yoy_net_profit", "asset_turnover", "cfo_to_sales", "equity_multiplier",
]

def fetch_financials(stocks: pd.DataFrame) -> pd.DataFrame:
    print("\n>>> 正在抓取季度财务指标（约300只 × 40季度，预计需要60~120分钟）...")
    codes = stocks["bs_code"].tolist()
    total = len(codes)
    all_dfs = []
    for i, code in enumerate(codes, 1):
        df = fetch_financials_one(code)
        if not df.empty:
            all_dfs.append(df)
        if i % 10 == 0 or i == total:
            print(f"  进度: {i}/{total}", flush=True)
        time.sleep(0.1)

    result = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    for col in FINANCIAL_COLS:
        if col not in result.columns:
            result[col] = ""
    result = result[FINANCIAL_COLS]

    out_path = OUT_DIR / "03_financials_quarterly.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print_summary("交付物3: 季度财务指标", result, date_col="statDate")
    print(f"  已保存 → {out_path}")
    return result


# ════════════════════════════════════════════════════════════════════════════
# 交付物 4: 日频行情（前复权）
# ════════════════════════════════════════════════════════════════════════════
PRICE_FIELDS = "date,code,open,high,low,close,volume,amount,adjustflag,turn,tradestatus,pctChg"

def fetch_one_price(bs_code: str) -> pd.DataFrame:
    rs = bs.query_history_k_data_plus(
        bs_code,
        PRICE_FIELDS,
        start_date=START_DATE,
        end_date=END_DATE,
        frequency="d",
        adjustflag="1",    # 1=前复权
    )
    df = _to_df(rs)
    if df.empty:
        return df
    df.rename(columns={
        "code":        "bs_code",
        "open":        "open",
        "high":        "high",
        "low":         "low",
        "close":       "close",
        "volume":      "volume",
        "amount":      "amount",
        "adjustflag":  "adjustflag",
        "turn":        "turn_rate",
        "tradestatus": "trade_status",
        "pctChg":      "pct_chg",
    }, inplace=True)
    df["code_6"] = df["bs_code"].apply(bs_code_to_6)
    for col in ["open","high","low","close","volume","amount","turn_rate","pct_chg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].where(df[col].notna(), other="")
    return df[["date","bs_code","code_6","open","high","low","close",
               "volume","amount","turn_rate","trade_status","pct_chg","adjustflag"]]

def fetch_prices(stocks: pd.DataFrame) -> pd.DataFrame:
    print("\n>>> 正在抓取日频行情数据（前复权，约300只 × 10年）...")
    codes = stocks["bs_code"].tolist()
    total = len(codes)
    all_dfs = []
    for i, code in enumerate(codes, 1):
        df = fetch_one_price(code)
        if not df.empty:
            all_dfs.append(df)
        if i % 30 == 0 or i == total:
            print(f"  进度: {i}/{total}", flush=True)
        time.sleep(0.05)

    result = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    out_path = OUT_DIR / "04_price_daily.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print_summary("交付物4: 日频行情", result)
    print(f"  已保存 → {out_path}")
    return result


# ════════════════════════════════════════════════════════════════════════════
# 打包 zip
# ════════════════════════════════════════════════════════════════════════════
def make_zip():
    zip_path = BASE_DIR / "data" / "csi300_dataset.zip"
    csv_files = list(OUT_DIR.glob("*.csv"))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(csv_files):
            zf.write(f, arcname=f"csi300_raw/{f.name}")
    print(f"\n>>> ZIP 已打包 → {zip_path}  ({zip_path.stat().st_size/1024/1024:.1f} MB)")
    return zip_path


# ════════════════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    login()
    try:
        # 1. 成分股
        stocks = fetch_constituents()
        print(f"  股票池大小: {len(stocks)}")

        # 2. 估值换手
        val_df = fetch_valuation(stocks)

        # 3. 季度财务
        fin_df = fetch_financials(stocks)

        # 4. 行情
        price_df = fetch_prices(stocks)

        # 5. 打包
        make_zip()

        elapsed = (time.time() - t0) / 60
        print(f"\n全部完成，总耗时: {elapsed:.1f} 分钟")

        # 汇总清单
        print("\n" + "="*60)
        print("汇总清单")
        print("="*60)
        for label, df, dcol in [
            ("01_constituents",      stocks,   None),
            ("02_valuation_daily",   val_df,   "date"),
            ("03_financials_quarterly", fin_df, "statDate"),
            ("04_price_daily",       price_df, "date"),
        ]:
            n_rows  = len(df)
            n_codes = df["bs_code"].nunique() if "bs_code" in df.columns else "-"
            dmin = df[dcol].min() if dcol and dcol in df.columns and not df.empty else "-"
            dmax = df[dcol].max() if dcol and dcol in df.columns and not df.empty else "-"
            print(f"{label}: 行数={n_rows:,}  股票数={n_codes}  日期={dmin}~{dmax}")

        if not val_df.empty:
            print("\n估值数据非空比例:")
            for col in ["pe_ttm", "pb_mrq"]:
                if col in val_df.columns:
                    non_empty = val_df[col].replace("", pd.NA).notna().sum()
                    pct = non_empty / len(val_df) * 100
                    print(f"  {col}: {pct:.1f}%")

    finally:
        logout()


if __name__ == "__main__":
    main()
